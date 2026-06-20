"""
final_test_Timofei.py — Timofei's final benchmark (Agent1 vs RAG).

Uses the exact same evaluation flows as Agent1.py and Rag.py.
GPT-4o comparison uses the same system prompt as the Streamlit app (agent1gpt.py).

Run:
    python3 final_test_Timofei.py              # both GPT and Groq
    python3 final_test_Timofei.py --mode gpt   # GPT only
    python3 final_test_Timofei.py --mode groq  # Groq only

Train/test split is loaded from final_split.json (committed to git).
"""

from __future__ import annotations

import argparse, json, os, time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"
GROQ_MODEL   = "llama-3.1-8b-instant"
K_EXAMPLES   = 5          # same default as Rag.py
GROQ_DELAY   = 3.0        # seconds between Groq calls (rate-limit guard)
SPLIT_FILE   = "final_split.json"

# ── Exact flows from Agent1.py and Rag.py ────────────────────────────────────
from Agent1 import AnswerEvaluator, _parse_verdict, SYSTEM_PROMPT, GroqEvaluator
from Rag    import (RAGIndex, RAGEvaluator, load_reviews, split_train_test,
                    _FEW_SHOT_HEADER, _build_question_text)
from agent1gpt import OpenAIEvaluator, SYSTEM_PROMPT as GPT_SYSTEM_PROMPT


# ── GPT RAG evaluator: same retrieval + few-shot format as Rag.py ─────────────
class GPTRAGEvaluator(AnswerEvaluator):
    def __init__(self, api_key: str, index: RAGIndex,
                 model: str = OPENAI_MODEL, base_url: str | None = None,
                 k: int = K_EXAMPLES):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model  = model
        self.index  = index
        self.k      = k

    def evaluate_text(self, question: str, answer: str) -> tuple[bool, str]:
        examples = self.index.retrieve(question, answer, k=self.k)
        lines    = [GPT_SYSTEM_PROMPT, _FEW_SHOT_HEADER]
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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "\n".join(lines)},
                {"role": "user",   "content": f"Vraag: {question}\nAntwoord: {answer}"},
            ],
            max_tokens=60, temperature=0,
        )
        return _parse_verdict(response.choices[0].message.content)


# ── Split ─────────────────────────────────────────────────────────────────────
def get_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading llm_reviews from Supabase ...")
    df = load_reviews()

    if os.path.exists(SPLIT_FILE):
        split    = json.load(open(SPLIT_FILE))
        train_df = df[df["id"].isin(split["train_ids"])].reset_index(drop=True)
        test_df  = df[df["id"].isin(split["test_ids"])].reset_index(drop=True)
        print(f"Loaded split from {SPLIT_FILE}  (train={len(train_df)}  test={len(test_df)})")
    else:
        train_df, test_df = split_train_test(df)
        json.dump(
            {"train_ids": train_df["id"].tolist(), "test_ids": test_df["id"].tolist()},
            open(SPLIT_FILE, "w"), indent=2,
        )
        print(f"Generated and saved split → {SPLIT_FILE}")

    return train_df, test_df


# ── Helpers ───────────────────────────────────────────────────────────────────
def _eval_row(row: pd.Series, evaluator: AnswerEvaluator) -> str:
    question   = _build_question_text(row)
    answer     = str(row["shopper_answer"])
    is_good, _ = evaluator.evaluate_text(question, answer)
    return "good" if is_good else "bad"


def _run(test_df: pd.DataFrame, evaluator: AnswerEvaluator,
         label: str, delay: float = 0.25) -> list[dict]:
    results, n = [], len(test_df)
    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        rid = str(row["id"])
        tv  = row["true_verdict"]
        try:
            pred = _eval_row(row, evaluator)
        except Exception as e:
            print(f"    [!] row {rid}: {e} — defaulting to 'good'")
            pred = "good"
        time.sleep(delay)
        ok = pred == tv
        print(f"  [{label[:28]:<28}] {i:>3}/{n}  id={rid:>4}  "
              f"true={tv:<4}  pred={pred:<4}  {'+'if ok else 'x'}")
        results.append({"id": int(row["id"]), "true": tv, "predicted": pred, "correct": ok})
    return results


def _metrics(results: list[dict], label: str) -> dict:
    n  = len(results)
    ok = sum(r["correct"] for r in results)
    tp = sum(1 for r in results if r["true"] == "good" and r["predicted"] == "good")
    tn = sum(1 for r in results if r["true"] == "bad"  and r["predicted"] == "bad")
    fp = sum(1 for r in results if r["true"] == "bad"  and r["predicted"] == "good")
    fn = sum(1 for r in results if r["true"] == "good" and r["predicted"] == "bad")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"{'─'*62}")
    print(f"  Accuracy : {ok}/{n} = {ok/n*100:.1f}%")
    print(f"  Precision: {prec*100:.1f}%  Recall: {rec*100:.1f}%  F1: {f1*100:.1f}%")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"{'─'*62}")
    return {"label": label, "accuracy": ok/n, "precision": prec,
            "recall": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n}


# ── Main ──────────────────────────────────────────────────────────────────────
def main(mode: str = "both") -> None:
    base_url   = os.environ.get("UVA_API_BASE") or None
    groq_key   = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    print("=" * 62)
    print(f"  FINAL TEST — Timofei  |  mode={mode}  K={K_EXAMPLES}")
    print("=" * 62)

    train_df, test_df = get_split()
    n_good = (test_df["true_verdict"] == "good").sum()
    n_bad  = (test_df["true_verdict"] == "bad").sum()
    print(f"Test: {len(test_df)} rows  (good={n_good}  bad={n_bad})")
    print("=" * 62)

    print("\nBuilding TF-IDF index over train set ...")
    index = RAGIndex(train_df)

    pairs: list[tuple[str, str, list, list]] = []

    if mode in ("gpt", "both"):
        if not openai_key:
            print("  [!] OPENAI_API_KEY not set — skipping GPT")
        else:
            lbl_a1  = f"Agent1+{OPENAI_MODEL}"
            lbl_rag = f"RAG+{OPENAI_MODEL}(k={K_EXAMPLES})"
            a1  = OpenAIEvaluator(api_key=openai_key, base_url=base_url, model=OPENAI_MODEL)
            rag = GPTRAGEvaluator(api_key=openai_key, index=index,
                                  model=OPENAI_MODEL, base_url=base_url, k=K_EXAMPLES)
            print(f"\n▶ {lbl_a1}  ({len(test_df)} rows)\n")
            r_a1  = _run(test_df, a1,  lbl_a1,  delay=0.25)
            print(f"\n▶ {lbl_rag}  ({len(test_df)} rows)\n")
            r_rag = _run(test_df, rag, lbl_rag, delay=0.25)
            pairs.append((lbl_a1, lbl_rag, r_a1, r_rag))

    if mode in ("groq", "both"):
        if not groq_key:
            print("  [!] GROQ_API_KEY not set — skipping Groq")
        else:
            lbl_a1  = f"Agent1+{GROQ_MODEL}"
            lbl_rag = f"RAG+{GROQ_MODEL}(k={K_EXAMPLES})"
            a1  = GroqEvaluator(api_key=groq_key)
            rag = RAGEvaluator(groq_api_key=groq_key, index=index, k=K_EXAMPLES)
            print(f"\n▶ {lbl_a1}  ({len(test_df)} rows, delay={GROQ_DELAY}s)\n")
            r_a1  = _run(test_df, a1,  lbl_a1,  delay=GROQ_DELAY)
            print(f"\n▶ {lbl_rag}  ({len(test_df)} rows, delay={GROQ_DELAY}s)\n")
            r_rag = _run(test_df, rag, lbl_rag, delay=GROQ_DELAY)
            pairs.append((lbl_a1, lbl_rag, r_a1, r_rag))

    # ── Summary ───────────────────────────────────────────────────────────────
    all_metrics = {}
    for lbl_a1, lbl_rag, r_a1, r_rag in pairs:
        all_metrics[lbl_a1]  = _metrics(r_a1,  lbl_a1)
        all_metrics[lbl_rag] = _metrics(r_rag, lbl_rag)

    W = 48
    print(f"\n{'='*62}")
    print(f"  SUMMARY — Timofei")
    print(f"{'='*62}")
    print(f"  {'Evaluator':<{W}} {'Acc':>5}  {'F1':>5}  {'Prec':>5}  {'Rec':>5}")
    print(f"  {'─'*W}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
    for m in all_metrics.values():
        print(f"  {m['label']:<{W}} {m['accuracy']*100:>4.1f}%  "
              f"{m['f1']*100:>4.1f}%  {m['precision']*100:>4.1f}%  {m['recall']*100:>4.1f}%")

    for lbl_a1, lbl_rag, r_a1, r_rag in pairs:
        m1, m2 = all_metrics[lbl_a1], all_metrics[lbl_rag]
        d_acc  = (m2["accuracy"] - m1["accuracy"]) * 100
        d_f1   = (m2["f1"] - m1["f1"]) * 100
        fixed  = sum(1 for a, b in zip(r_a1, r_rag) if not a["correct"] and b["correct"])
        broke  = sum(1 for a, b in zip(r_a1, r_rag) if a["correct"] and not b["correct"])
        print(f"\n  {lbl_a1} vs {lbl_rag}:")
        print(f"  Δacc={d_acc:+.1f}%  Δf1={d_f1:+.1f}%  fixed={fixed}  broke={broke}")
    print(f"{'='*62}")

    output = {
        "name": "Timofei",
        "date": datetime.now().isoformat(),
        "k_examples": K_EXAMPLES,
        "evaluators": {
            lbl: {"metrics": m}
            for lbl, m in all_metrics.items()
        },
    }
    results_file = f"results_Timofei_{mode}.json"
    json.dump(output, open(results_file, "w"), indent=2)
    print(f"\n  Saved → {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gpt", "groq", "both"], default="groq",
                        help="Which model pair to run (default: both)")
    args = parser.parse_args()
    main(mode=args.mode)
