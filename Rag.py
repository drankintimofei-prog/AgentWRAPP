"""
rag.py — RAG-augmented mystery shopper answer evaluator.

Train/test split : 111 / 111 random rows from llm_reviews (seed=42).
Ground truth     : Nigel's true verdict — llm_verdict flipped where
                   nigel_rating == 'disagree'.

Usage
-----
    python3 rag.py            # build index, then compare Agent1 vs RAG
    python3 rag.py --k 3      # use top-3 retrieved examples instead of 5
"""

from __future__ import annotations

import argparse, json, os, time, uuid
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]

TRAIN_SEED    = 42
N_TRAIN       = 111
# TF-IDF retrieval -- no torch/GPU needed
K_EXAMPLES    = 5
CACHE_FILE    = "rag_verdict_cache.json"

from Agent1 import AnswerEvaluator, _parse_verdict, SYSTEM_PROMPT


# ── 1. Data loading & splitting ───────────────────────────────────────────────

def load_reviews() -> pd.DataFrame:
    """Fetch all 222 rows from llm_reviews; compute Nigel-corrected verdicts."""
    from supabase import create_client
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    rows   = client.table("llm_reviews").select("*").order("id").execute().data
    df     = pd.DataFrame(rows)
    # True verdict: if Nigel agrees keep llm_verdict; if disagrees flip it
    df["true_verdict"] = df.apply(
        lambda r: r["llm_verdict"] if r["nigel_rating"] == "agree"
                  else ("good" if r["llm_verdict"] == "bad" else "bad"),
        axis=1,
    )
    return df


def split_train_test(df: pd.DataFrame, n_train: int = N_TRAIN, seed: int = TRAIN_SEED):
    train = df.sample(n=n_train, random_state=seed)
    test  = df.drop(train.index).reset_index(drop=True)
    return train.reset_index(drop=True), test


# ── 2. Semantic index (sentence-transformers + cosine similarity) ─────────────

def _row_to_embed_text(row: pd.Series) -> str:
    parts = [str(row["question_text"])]
    pq = str(row.get("parent_question") or "")
    if pq and pq not in ("None", "nan", ""):
        parts.append(f"context: {pq}")
    parts.append(str(row["shopper_answer"]))
    return " | ".join(parts)


class RAGIndex:
    """TF-IDF semantic index over the 111 training examples (no torch needed)."""

    def __init__(self, train_df: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        self._cos_sim  = cos_sim
        self.train_df  = train_df.reset_index(drop=True)
        texts           = [_row_to_embed_text(r) for _, r in train_df.iterrows()]
        print(f"Building TF-IDF index over {len(texts)} training examples ...")
        self.vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            min_df=1, sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)  # (n_train, vocab)

    def retrieve(self, question: str, answer: str,
                 k: int = K_EXAMPLES) -> list[tuple[pd.Series, float]]:
        query   = f"{question} | {answer}"
        q_vec   = self.vectorizer.transform([query])
        scores  = self._cos_sim(q_vec, self.matrix).squeeze()
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.train_df.iloc[i], float(scores[i])) for i in top_idx]


# ── 3. RAG Evaluator ──────────────────────────────────────────────────────────

_FEW_SHOT_HEADER = "\n\nVOORBEELDEN UIT EERDERE BEOORDELINGEN (vergelijkbare vragen):\n"


def _build_augmented_system(examples: list[tuple[pd.Series, float]]) -> str:
    lines = [SYSTEM_PROMPT, _FEW_SHOT_HEADER]
    for i, (row, score) in enumerate(examples, 1):
        verdict = row["true_verdict"]
        reason  = (str(row.get("nigel_comment") or "") or
                   str(row.get("llm_reason") or "")).strip()
        q  = str(row["question_text"])[:120]
        a  = str(row["shopper_answer"])[:150]
        pq = str(row.get("parent_question") or "")
        pa = str(row.get("parent_answer")   or "")
        ctx = (f'[context: "{pq}" -> "{pa}"] '
               if pq and pq not in ("None", "nan") else "")
        lines.append(
            f"Voorbeeld {i} (overeenkomst {score:.2f}):\n"
            f"  Vraag: {ctx}{q}\n"
            f"  Antwoord: {a}\n"
            f"  Beoordeling: {verdict}: {reason}\n"
        )
    return "\n".join(lines)


class RAGEvaluator(AnswerEvaluator):
    """Groq evaluator augmented with retrieved few-shot examples at call time."""

    def __init__(self, groq_api_key: str, index: RAGIndex,
                 model: str = "llama-3.1-8b-instant", k: int = K_EXAMPLES):
        from groq import Groq
        self.client = Groq(api_key=groq_api_key)
        self.model  = model
        self.index  = index
        self.k      = k

    def evaluate_text(self, question: str, answer: str) -> tuple[bool, str]:
        examples         = self.index.retrieve(question, answer, k=self.k)
        augmented_system = _build_augmented_system(examples)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user",   "content": f"Vraag: {question}\nAntwoord: {answer}"},
            ],
            max_tokens=60,
            temperature=0,
        )
        return _parse_verdict(response.choices[0].message.content)


# ── 4. Helpers ────────────────────────────────────────────────────────────────

def _build_question_text(row: pd.Series) -> str:
    """Reconstruct the question string exactly as Agent1 passes it to evaluate_text."""
    q  = str(row["question_text"])
    pq = str(row.get("parent_question") or "")
    pa = str(row.get("parent_answer")   or "")
    if pq and pq not in ("None", "nan", ""):
        ctx = f'[toelichting op: "{pq}"'
        if pa and pa not in ("None", "nan", ""):
            ctx += f' -> antwoord: "{pa}"'
        return f"{ctx}]\n{q}"
    return q


def _eval_row(row: pd.Series, evaluator: AnswerEvaluator) -> str:
    question   = _build_question_text(row)
    answer     = str(row["shopper_answer"])
    is_good, _ = evaluator.evaluate_text(question, answer)
    return "good" if is_good else "bad"


# ── 5. Comparison runner ──────────────────────────────────────────────────────

def _run_evaluator_on_test(
    test_df: pd.DataFrame,
    evaluator: AnswerEvaluator,
    label: str,
    cache: dict,
    cache_key_prefix: str,
) -> list[dict]:
    results = []
    n = len(test_df)
    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        row_id       = str(row["id"])
        cache_key    = f"{cache_key_prefix}_{row_id}"
        true_verdict = row["true_verdict"]

        if cache_key in cache:
            predicted = cache[cache_key]
        else:
            try:
                predicted = _eval_row(row, evaluator)
            except Exception as e:
                print(f"    [!] Error on row {row_id}: {e}; defaulting to 'good'")
                predicted = "good"
            cache[cache_key] = predicted
            json.dump(cache, open(CACHE_FILE, "w"))
            time.sleep(0.25)

        correct = predicted == true_verdict
        results.append({
            "id": row["id"], "true": true_verdict,
            "predicted": predicted, "correct": correct,
        })
        icon = "+" if correct else "x"
        print(f"  [{label}] {i:>3}/{n}  row {str(row['id']):>3}  "
              f"true={true_verdict:<4}  pred={predicted:<4}  {icon}")

    return results


def _print_metrics(results: list[dict], label: str) -> dict:
    n       = len(results)
    correct = sum(r["correct"] for r in results)
    tp = sum(1 for r in results if r["true"] == "good" and r["predicted"] == "good")
    tn = sum(1 for r in results if r["true"] == "bad"  and r["predicted"] == "bad")
    fp = sum(1 for r in results if r["true"] == "bad"  and r["predicted"] == "good")
    fn = sum(1 for r in results if r["true"] == "good" and r["predicted"] == "bad")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Accuracy : {correct}/{n} = {correct/n*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%  "
          f"Recall: {recall*100:.1f}%  F1: {f1*100:.1f}%")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'─'*60}")
    return {"accuracy": correct/n, "precision": precision,
            "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compare_agents(k: int = K_EXAMPLES) -> None:
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set in environment.")

    # Unique ID per run so cache never carries over between runs
    run_id = uuid.uuid4().hex[:8]

    print("=" * 60)
    print(f"Run ID: {run_id}")
    print("Loading llm_reviews from Supabase ...")
    df = load_reviews()
    train_df, full_test = split_train_test(df)
    # Sample 20 fresh rows every run (random_state=None → different each time)
    test_df = full_test.sample(n=20, random_state=None).reset_index(drop=True)

    n_agree    = (train_df["nigel_rating"] == "agree").sum()
    n_disagree = (train_df["nigel_rating"] == "disagree").sum()
    n_good     = (train_df["true_verdict"] == "good").sum()
    n_bad      = (train_df["true_verdict"] == "bad").sum()
    print(f"Train: {len(train_df)} rows  "
          f"(Nigel agree={n_agree}, disagree={n_disagree} | good={n_good}, bad={n_bad})")
    print(f"Test : {len(test_df)}/111 sampled rows  "
          f"(good={(test_df['true_verdict']=='good').sum()}, "
          f"bad={(test_df['true_verdict']=='bad').sum()})")
    print("=" * 60)

    index = RAGIndex(train_df)

    from Agent1 import GroqEvaluator
    agent1_evaluator = GroqEvaluator(api_key=groq_key)
    rag_evaluator    = RAGEvaluator(groq_api_key=groq_key, index=index, k=k)

    cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}

    print(f"\nRunning Agent1 (baseline) on {len(test_df)} test rows ...\n")
    res_agent1 = _run_evaluator_on_test(test_df, agent1_evaluator, "Agent1", cache, f"a1_{run_id}")

    print(f"\nRunning RAG evaluator (k={k}) on {len(test_df)} test rows ...\n")
    res_rag = _run_evaluator_on_test(test_df, rag_evaluator, "RAG", cache, f"rag_k{k}_{run_id}")

    print("\n" + "=" * 60)
    print("RESULTS  (ground truth = Nigel-corrected verdict)")
    m1 = _print_metrics(res_agent1, "Agent1  -- system prompt only")
    m2 = _print_metrics(res_rag,    f"RAG     -- k={k} retrieved examples")

    delta_acc = (m2["accuracy"] - m1["accuracy"]) * 100
    delta_f1  = (m2["f1"]       - m1["f1"])       * 100
    print(f"\n  Delta Accuracy : {delta_acc:+.1f}%")
    print(f"  Delta F1       : {delta_f1:+.1f}%")
    print("=" * 60)

    fixed = [(r1, r2) for r1, r2 in zip(res_agent1, res_rag)
             if not r1["correct"] and r2["correct"]]
    broke = [(r1, r2) for r1, r2 in zip(res_agent1, res_rag)
             if r1["correct"] and not r2["correct"]]
    print(f"\n  RAG fixed {len(fixed)} Agent1 mistakes")
    print(f"  RAG broke {len(broke)} previously-correct answers")
    if fixed:
        print(f"    Fixed IDs : {', '.join(str(r1['id']) for r1, _ in fixed)}")
    if broke:
        print(f"    Broke IDs : {', '.join(str(r1['id']) for r1, _ in broke)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Agent1 vs RAG on llm_reviews test set"
    )
    parser.add_argument("--k", type=int, default=K_EXAMPLES,
                        help=f"Retrieved examples per query (default {K_EXAMPLES})")
    args = parser.parse_args()
    compare_agents(k=args.k)
