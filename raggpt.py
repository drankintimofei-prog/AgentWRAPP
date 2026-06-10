"""
raggpt.py — RAG-augmented mystery shopper answer evaluator (GPT-4o variant).

Train/test split : 111 / 111 random rows from llm_reviews (seed=42).
Ground truth     : Nigel's true verdict — llm_verdict flipped where
                   nigel_rating == 'disagree'.

Usage
-----
    python3 raggpt.py               # compare Agent1 vs RAG (both on GPT-4o)
    python3 raggpt.py --k 3         # use top-3 retrieved examples instead of 10
    python3 raggpt.py --seed 12345  # reproduce a specific random test set
"""

from __future__ import annotations

import argparse, json, os, random, time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_KEY"]

TRAIN_SEED    = 42
N_TRAIN       = 111
# TF-IDF retrieval -- no torch/GPU needed
K_EXAMPLES    = 10
CACHE_FILE    = "rag_verdict_cache.json"

from agent1gpt import AnswerEvaluator, _parse_verdict, SYSTEM_PROMPT


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

_FEW_SHOT_HEADER = """

EERDERE BEOORDELINGEN VAN VERGELIJKBARE VRAGEN (alleen ter referentie):

De voorbeelden hieronder zijn ter ondersteuning. De REGELS BOVENAAN gelden
ALTIJD en mogen NOOIT door een voorbeeld worden overruled. In het bijzonder
blijft gelden: voor feitelijke vragen (adres, tijd, bedrag, product, naam)
is een kort feitelijk antwoord ALTIJD goed.

Antwoord in het Nederlands met "goed:" of "slecht:" — niet in het Engels.
"""

# The corpus stores verdicts as English "good"/"bad". The system prompt and
# the response parser (Agent1._parse_verdict) both work in Dutch only. Show
# the model Dutch labels in the few-shots so it doesn't pattern-match an
# English label into its own reply (which then parses incorrectly).
_VERDICT_NL = {"good": "goed", "bad": "slecht"}


def _build_augmented_system(examples: list[tuple[pd.Series, float]]) -> str:
    lines = [SYSTEM_PROMPT, _FEW_SHOT_HEADER]
    for i, (row, score) in enumerate(examples, 1):
        verdict_en = str(row["true_verdict"]).strip().lower()
        verdict    = _VERDICT_NL.get(verdict_en, verdict_en)
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
    """GPT-4o evaluator augmented with retrieved few-shot examples at call
    time. Uses the OpenAI-compatible API so the same code works against
    api.openai.com or the UvA LiteLLM proxy (set UVA_API_BASE in .env)."""

    def __init__(self, index: RAGIndex, k: int = K_EXAMPLES,
                 api_key: str | None = None, base_url: str | None = None,
                 model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
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
            # In-memory only: the cache is NOT persisted to disk between runs,
            # so every run is a genuinely fresh evaluation of freshly-sampled
            # test rows rather than a replay of a previous run's verdicts.
            cache[cache_key] = predicted
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


def compare_agents(k: int = K_EXAMPLES, seed: int | None = None,
                   n_test: int = 20) -> None:
    api_key  = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("UVA_API_BASE") or None
    model    = "gpt-4o"
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    # A fresh random test set on every run. The train/test SPLIT below stays
    # fixed (TRAIN_SEED) so the TF-IDF retrieval corpus — and therefore the
    # examples RAG retrieves — never changes; the only thing that varies run
    # to run is WHICH test questions we sample. Pass --seed N to reproduce a
    # specific run.
    if seed is None:
        seed = random.randrange(1_000_000)

    print("=" * 60)
    print(f"Test seed: {seed}   (re-run with  --seed {seed}  to reproduce)")
    print("Loading llm_reviews from Supabase ...")
    df = load_reviews()
    train_df, full_test = split_train_test(df)

    # Keep only rows Nigel marked as truly "bad", then take a NEW random
    # sample of them each run (random_state=seed, not the fixed TRAIN_SEED).
    bad_rows = full_test[full_test["true_verdict"] == "bad"]
    test_df  = bad_rows.sample(
        n=min(n_test, len(bad_rows)), random_state=seed
    ).reset_index(drop=True)

    print(f"Train: {len(train_df)} rows")
    print(f"Test : {len(test_df)} rows  (all true_verdict='bad', Nigel-corrected)")
    print("=" * 60)

    index = RAGIndex(train_df)

    # Both agents share the same GPT-4o model, so the only variable between
    # Agent1 and RAG is the RAG retrieval step.
    from agent1gpt import OpenAIEvaluator
    agent1_evaluator = OpenAIEvaluator(api_key=api_key, base_url=base_url, model=model)
    rag_evaluator = RAGEvaluator(index=index, k=k,
                                 api_key=api_key, base_url=base_url, model=model)

    # Fresh in-memory cache — nothing is read back from a previous run, so
    # the verdicts you see are always computed live for this run's rows.
    cache: dict = {}

    print(f"\nRunning Agent1 (baseline) on {len(test_df)} test rows ...\n")
    res_agent1 = _run_evaluator_on_test(test_df, agent1_evaluator, "Agent1", cache, "a1")

    print(f"\nRunning RAG evaluator (k={k}) on {len(test_df)} test rows ...\n")
    res_rag = _run_evaluator_on_test(test_df, rag_evaluator, "RAG", cache, f"rag_k{k}")

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
    parser.add_argument("--seed", type=int, default=None,
                        help="Test-sample seed. Omit for a new random test "
                             "each run; pass a value to reproduce a run.")
    parser.add_argument("--n-test", type=int, default=20, dest="n_test",
                        help="Number of 'bad' test rows to sample (default 20)")
    args = parser.parse_args()
    compare_agents(k=args.k, seed=args.seed, n_test=args.n_test)
