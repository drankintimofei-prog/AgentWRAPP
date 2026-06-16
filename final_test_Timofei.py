"""
final_test_Timofei.py — Final benchmark on the fixed 111-row test set.

Runs four evaluator configurations on the SAME deterministic split (seed=42):
  1. Agent1 + GPT   — system prompt only, OpenAI model
  2. RAG   + GPT   — 10 retrieved examples, OpenAI model
  3. Agent1 + Groq  — system prompt only, Groq model
  4. RAG   + Groq  — 10 retrieved examples, Groq model

Train/test split is FIXED at 111/111 (seed=42). Everyone who runs this file
evaluates on the same 111 test rows so results are directly comparable.

Results are saved to results_<YOUR_NAME>.json.
LLM calls are cached to cache_final_<YOUR_NAME>.json so re-runs are free.

HOW OTHERS CAN USE THIS
───────────────────────
Change YOUR_NAME, OPENAI_MODEL, and/or GROQ_MODEL below, then run:
    python3 final_test_Timofei.py

If you only have one API key set, the other evaluator pair is skipped
automatically. Share your results_<name>.json for cross-person comparison.
"""

from __future__ import annotations

import json, os, time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURE YOUR RUN — change these variables, then run the script
# ══════════════════════════════════════════════════════════════════════════════
YOUR_NAME    = "Timofei"               # results saved to results_<YOUR_NAME>.json
OPENAI_MODEL = "gpt-4o-mini"           # e.g. "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"
GROQ_MODEL   = "llama-3.1-8b-instant"  # e.g. "llama-3.3-70b-versatile", "mixtral-8x7b-32768"
K_EXAMPLES   = 10                      # RAG: retrieved few-shot examples per query
GROQ_DELAY   = 3.0                     # seconds between Groq calls (rate-limit guard)
#              ↑ increase to 5–6 if you hit Groq 429 errors
# ══════════════════════════════════════════════════════════════════════════════

# Fixed split — do NOT change; all team members must use these values
TRAIN_SEED = 42
N_TRAIN    = 111

from agent1gpt import (
    AnswerEvaluator, OpenAIEvaluator, GroqEvaluator, _parse_verdict,
)
from raggpt import (
    RAGIndex, RAGEvaluator,
    _build_augmented_system, _build_question_text,
    load_reviews, split_train_test,
)


# ── Groq RAG Evaluator ───────────────────────────────────────────────────────
# raggpt.RAGEvaluator is GPT-only; this wraps the same RAG logic for Groq.

class GroqRAGEvaluator(AnswerEvaluator):
    """RAG-augmented evaluator using Groq instead of OpenAI.
    Retrieves few-shots from the same TF-IDF index and uses the same
    Dutch-labelled few-shot format as RAGEvaluator for consistency.
    """

    def __init__(self, groq_api_key: str, index: RAGIndex,
                 model: str = GROQ_MODEL, k: int = K_EXAMPLES):
        from groq import Groq
        self.client = Groq(api_key=groq_api_key)
        self.model  = model
        self.index  = index
        self.k      = k

    def evaluate_text(self, question: str, answer: str) -> tuple[bool, str]:
        examples   = self.index.retrieve(question, answer, k=self.k)
        system_msg = _build_augmented_system(examples)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": f"Vraag: {question}\nAntwoord: {answer}"},
            ],
            max_tokens=60,
            temperature=0,
        )
        return _parse_verdict(resp.choices[0].message.content)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _eval_row(row: pd.Series, evaluator: AnswerEvaluator) -> str:
    question   = _build_question_text(row)
    answer     = str(row["shopper_answer"])
    is_good, _ = evaluator.evaluate_text(question, answer)
    return "good" if is_good else "bad"


def _run(
    test_df: pd.DataFrame,
    evaluator: AnswerEvaluator,
    label: str,
    cache: dict,
    delay: float = 0.25,
) -> list[dict]:
    results, n = [], len(test_df)
    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        rid, tv = str(row["id"]), row["true_verdict"]
        ck = f"{label}_{rid}"
        if ck in cache:
            pred = cache[ck]
            src  = "cache"
        else:
            try:
                pred = _eval_row(row, evaluator)
            except Exception as e:
                print(f"    [!] row {rid}: {e} — defaulting to 'good'")
                pred = "good"
            cache[ck] = pred
            src = "live"
            time.sleep(delay)
        ok = pred == tv
        print(
            f"  [{label[:28]:<28}] {i:>3}/{n}  id={rid:>4}  "
            f"true={tv:<4}  pred={pred:<4}  {'+'if ok else 'x'}  ({src})"
        )
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
    return {
        "label": label, "accuracy": ok / n, "precision": prec,
        "recall": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    openai_key = os.environ.get("OPENAI_API_KEY")
    groq_key   = os.environ.get("GROQ_API_KEY")
    base_url   = os.environ.get("UVA_API_BASE") or None

    cache_file = f"cache_final_{YOUR_NAME}.json"
    cache: dict = json.load(open(cache_file)) if os.path.exists(cache_file) else {}

    print("=" * 62)
    print(f"  FINAL TEST — {YOUR_NAME}")
    print(f"  OpenAI : {OPENAI_MODEL if openai_key else 'SKIPPED (no OPENAI_API_KEY)'}")
    print(f"  Groq   : {GROQ_MODEL   if groq_key   else 'SKIPPED (no GROQ_API_KEY)'}")
    print(f"  K      : {K_EXAMPLES}  |  train_seed={TRAIN_SEED}  n_train={N_TRAIN}  (fixed)")
    print("=" * 62)

    print("\nLoading llm_reviews from Supabase ...")
    df = load_reviews()
    train_df, test_df = split_train_test(df, n_train=N_TRAIN, seed=TRAIN_SEED)

    n_good = (test_df["true_verdict"] == "good").sum()
    n_bad  = (test_df["true_verdict"] == "bad").sum()
    print(f"Train : {len(train_df)} rows")
    print(f"Test  : {len(test_df)} rows  (good={n_good}  bad={n_bad})")
    print("=" * 62)

    print("\nBuilding TF-IDF index ...")
    index = RAGIndex(train_df)

    collected: list[tuple[dict, list[dict]]] = []

    # ── GPT runs ──────────────────────────────────────────────────────────────
    if openai_key:
        lbl_a1  = f"Agent1+{OPENAI_MODEL}"
        lbl_rag = f"RAG+{OPENAI_MODEL}(k={K_EXAMPLES})"

        a1_gpt  = OpenAIEvaluator(api_key=openai_key, base_url=base_url, model=OPENAI_MODEL)
        rag_gpt = RAGEvaluator(index=index, k=K_EXAMPLES,
                               api_key=openai_key, base_url=base_url, model=OPENAI_MODEL)

        print(f"\n▶ {lbl_a1}  ({len(test_df)} rows)\n")
        r_a1_gpt = _run(test_df, a1_gpt, lbl_a1, cache, delay=0.25)

        print(f"\n▶ {lbl_rag}  ({len(test_df)} rows)\n")
        r_rag_gpt = _run(test_df, rag_gpt, lbl_rag, cache, delay=0.25)

        collected.append((_metrics(r_a1_gpt,  lbl_a1),  r_a1_gpt))
        collected.append((_metrics(r_rag_gpt, lbl_rag), r_rag_gpt))
    else:
        print("\n  [SKIP] GPT — OPENAI_API_KEY not set")

    # ── Groq runs ─────────────────────────────────────────────────────────────
    if groq_key:
        lbl_ag = f"Agent1+{GROQ_MODEL}"
        lbl_rg = f"RAG+{GROQ_MODEL}(k={K_EXAMPLES})"

        a1_groq  = GroqEvaluator(api_key=groq_key, model=GROQ_MODEL)
        rag_groq = GroqRAGEvaluator(groq_api_key=groq_key, index=index,
                                     model=GROQ_MODEL, k=K_EXAMPLES)

        print(f"\n▶ {lbl_ag}  ({len(test_df)} rows, delay={GROQ_DELAY}s)\n")
        r_a1_groq = _run(test_df, a1_groq, lbl_ag, cache, delay=GROQ_DELAY)

        print(f"\n▶ {lbl_rg}  ({len(test_df)} rows, delay={GROQ_DELAY}s)\n")
        r_rag_groq = _run(test_df, rag_groq, lbl_rg, cache, delay=GROQ_DELAY)

        collected.append((_metrics(r_a1_groq,  lbl_ag), r_a1_groq))
        collected.append((_metrics(r_rag_groq, lbl_rg), r_rag_groq))
    else:
        print("\n  [SKIP] Groq — GROQ_API_KEY not set")

    if not collected:
        print("\n  No evaluators ran — set OPENAI_API_KEY or GROQ_API_KEY.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 48
    print(f"\n{'='*62}")
    print(f"  FINAL SUMMARY — {YOUR_NAME}")
    print(f"{'='*62}")
    print(f"  {'Evaluator':<{W}} {'Acc':>5}  {'F1':>5}  {'Prec':>5}  {'Rec':>5}")
    print(f"  {'─'*W}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
    for m, _ in collected:
        print(
            f"  {m['label']:<{W}} {m['accuracy']*100:>4.1f}%  "
            f"{m['f1']*100:>4.1f}%  {m['precision']*100:>4.1f}%  {m['recall']*100:>4.1f}%"
        )
    print(f"{'='*62}")

    # Delta rows: Agent1 vs RAG for each model family
    pairs: list[tuple[tuple, tuple, str]] = []
    if openai_key and len(collected) >= 2:
        pairs.append((collected[0], collected[1], OPENAI_MODEL))
    if groq_key:
        off = 2 if openai_key else 0
        if len(collected) >= off + 2:
            pairs.append((collected[off], collected[off + 1], GROQ_MODEL))

    if pairs:
        print()
    for (m1, r1), (m2, r2), family in pairs:
        d_acc = (m2["accuracy"] - m1["accuracy"]) * 100
        d_f1  = (m2["f1"]       - m1["f1"])       * 100
        fixed = sum(1 for a, b in zip(r1, r2) if not a["correct"] and b["correct"])
        broke = sum(1 for a, b in zip(r1, r2) if     a["correct"] and not b["correct"])
        print(
            f"  {family}:  RAG vs Agent1  "
            f"Δacc={d_acc:+.1f}%  Δf1={d_f1:+.1f}%  "
            f"fixed={fixed}  broke={broke}"
        )

    # ── Persist cache and results ──────────────────────────────────────────────
    json.dump(cache, open(cache_file, "w"), indent=2)

    output = {
        "name":         YOUR_NAME,
        "date":         datetime.now().isoformat(),
        "openai_model": OPENAI_MODEL,
        "groq_model":   GROQ_MODEL,
        "k_examples":   K_EXAMPLES,
        "train_seed":   TRAIN_SEED,
        "n_test":       len(test_df),
        "evaluators": {
            m["label"]: {"metrics": m, "rows": r}
            for m, r in collected
        },
    }
    results_file = f"results_{YOUR_NAME}.json"
    json.dump(output, open(results_file, "w"), indent=2)

    print(f"\n  Saved → {results_file}  (share this for cross-person comparison)")
    print(f"  Cache → {cache_file}  (re-running is free)")


if __name__ == "__main__":
    main()
