"""
final_test_Timofei.py — Timofei's final benchmark (Agent1 vs RAG).

Set MODE = "gpt" or "groq" below and run:
    python3 final_test_Timofei.py

Train/test split is loaded from final_split.json (committed to git).
Results → results_Timofei_<mode>.json
"""

from __future__ import annotations

import json, os, time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURE YOUR RUN
# ══════════════════════════════════════════════════════════════════════════════
MODE         = "gpt"                   # "gpt"  → tests Agent1+GPT and RAG+GPT
                                       # "groq" → tests Agent1+Groq and RAG+Groq

OPENAI_MODEL = "gpt-4o-mini"           # only used when MODE = "gpt"
GROQ_MODEL   = "llama-3.1-8b-instant"  # only used when MODE = "groq"

K_EXAMPLES   = 10                      # RAG: retrieved few-shot examples per query
GROQ_DELAY   = 3.0                     # seconds between Groq calls (rate-limit guard)
#              ↑ increase to 5–6 if you hit Groq 429 errors
# ══════════════════════════════════════════════════════════════════════════════

# Split config — do NOT change; must be identical for all team members
TRAIN_SEED  = 42
SPLIT_FILE  = "final_split.json"   # generated once, committed to git

from agent1gpt import (
    AnswerEvaluator, OpenAIEvaluator, GroqEvaluator, _parse_verdict,
)
from raggpt import (
    RAGIndex, RAGEvaluator,
    _build_augmented_system, _build_question_text,
    load_reviews,
)


# ── Groq RAG Evaluator ───────────────────────────────────────────────────────

class GroqRAGEvaluator(AnswerEvaluator):
    """RAG-augmented evaluator using Groq instead of OpenAI."""

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


# ── Split: generate once, always load from disk ───────────────────────────────

def get_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df). Generates final_split.json on first call."""
    print("Loading llm_reviews from Supabase ...")
    df = load_reviews()

    if os.path.exists(SPLIT_FILE):
        split = json.load(open(SPLIT_FILE))
        train_ids = set(split["train_ids"])
        test_ids  = set(split["test_ids"])
        train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
        test_df  = df[df["id"].isin(test_ids)].reset_index(drop=True)
        print(f"Loaded split from {SPLIT_FILE}  "
              f"(train={len(train_df)}  test={len(test_df)})")
    else:
        from sklearn.model_selection import train_test_split as skl_split
        train_df, test_df = skl_split(
            df, test_size=0.5, random_state=TRAIN_SEED,
            stratify=df["nigel_rating"],
        )
        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)

        # Verify stratification
        for label, part in [("train", train_df), ("test", test_df)]:
            counts = part["nigel_rating"].value_counts().to_dict()
            print(f"  {label}: {len(part)} rows  {counts}")

        json.dump(
            {"train_ids": train_df["id"].tolist(), "test_ids": test_df["id"].tolist()},
            open(SPLIT_FILE, "w"), indent=2,
        )
        print(f"Generated and saved split → {SPLIT_FILE}")
        print(f"  → Commit {SPLIT_FILE} to git so teammates use the same rows.")

    return train_df, test_df


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
            pred, src = cache[ck], "cache"
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
    if MODE not in ("gpt", "groq"):
        raise ValueError(f"MODE must be 'gpt' or 'groq', got {MODE!r}")

    base_url = os.environ.get("UVA_API_BASE") or None
    cache_file = f"cache_final_Timofei_{MODE}.json"
    cache: dict = json.load(open(cache_file)) if os.path.exists(cache_file) else {}

    print("=" * 62)
    print(f"  FINAL TEST — Timofei  |  MODE = {MODE}")
    if MODE == "gpt":
        print(f"  Model : {OPENAI_MODEL}")
    else:
        print(f"  Model : {GROQ_MODEL}  (delay={GROQ_DELAY}s)")
    print(f"  K     : {K_EXAMPLES}  |  split file : {SPLIT_FILE}")
    print("=" * 62)

    train_df, test_df = get_split()

    n_good = (test_df["true_verdict"] == "good").sum()
    n_bad  = (test_df["true_verdict"] == "bad").sum()
    print(f"Test : {len(test_df)} rows  (good={n_good}  bad={n_bad})")
    print("=" * 62)

    print("\nBuilding TF-IDF index over train set ...")
    index = RAGIndex(train_df)

    collected: list[tuple[dict, list[dict]]] = []

    if MODE == "gpt":
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")

        lbl_a1  = f"Agent1+{OPENAI_MODEL}"
        lbl_rag = f"RAG+{OPENAI_MODEL}(k={K_EXAMPLES})"

        a1  = OpenAIEvaluator(api_key=openai_key, base_url=base_url, model=OPENAI_MODEL)
        rag = RAGEvaluator(index=index, k=K_EXAMPLES,
                           api_key=openai_key, base_url=base_url, model=OPENAI_MODEL)

        print(f"\n▶ {lbl_a1}  ({len(test_df)} rows)\n")
        r_a1 = _run(test_df, a1, lbl_a1, cache, delay=0.25)

        print(f"\n▶ {lbl_rag}  ({len(test_df)} rows)\n")
        r_rag = _run(test_df, rag, lbl_rag, cache, delay=0.25)

        collected.append((_metrics(r_a1,  lbl_a1),  r_a1))
        collected.append((_metrics(r_rag, lbl_rag), r_rag))

    else:  # groq
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY not set in environment.")

        lbl_a1  = f"Agent1+{GROQ_MODEL}"
        lbl_rag = f"RAG+{GROQ_MODEL}(k={K_EXAMPLES})"

        a1  = GroqEvaluator(api_key=groq_key, model=GROQ_MODEL)
        rag = GroqRAGEvaluator(groq_api_key=groq_key, index=index,
                               model=GROQ_MODEL, k=K_EXAMPLES)

        print(f"\n▶ {lbl_a1}  ({len(test_df)} rows, delay={GROQ_DELAY}s)\n")
        r_a1 = _run(test_df, a1, lbl_a1, cache, delay=GROQ_DELAY)

        print(f"\n▶ {lbl_rag}  ({len(test_df)} rows, delay={GROQ_DELAY}s)\n")
        r_rag = _run(test_df, rag, lbl_rag, cache, delay=GROQ_DELAY)

        collected.append((_metrics(r_a1,  lbl_a1),  r_a1))
        collected.append((_metrics(r_rag, lbl_rag), r_rag))

    # ── Summary ───────────────────────────────────────────────────────────────
    (m1, r1), (m2, r2) = collected
    d_acc = (m2["accuracy"] - m1["accuracy"]) * 100
    d_f1  = (m2["f1"]       - m1["f1"])       * 100
    fixed = sum(1 for a, b in zip(r1, r2) if not a["correct"] and b["correct"])
    broke = sum(1 for a, b in zip(r1, r2) if     a["correct"] and not b["correct"])

    W = 48
    print(f"\n{'='*62}")
    print(f"  SUMMARY — Timofei  ({MODE.upper()})")
    print(f"{'='*62}")
    print(f"  {'Evaluator':<{W}} {'Acc':>5}  {'F1':>5}  {'Prec':>5}  {'Rec':>5}")
    print(f"  {'─'*W}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")
    for m, _ in collected:
        print(
            f"  {m['label']:<{W}} {m['accuracy']*100:>4.1f}%  "
            f"{m['f1']*100:>4.1f}%  {m['precision']*100:>4.1f}%  {m['recall']*100:>4.1f}%"
        )
    print(f"{'─'*62}")
    print(f"  RAG vs Agent1:  Δacc={d_acc:+.1f}%  Δf1={d_f1:+.1f}%  "
          f"fixed={fixed}  broke={broke}")
    print(f"{'='*62}")

    # ── Persist ───────────────────────────────────────────────────────────────
    json.dump(cache, open(cache_file, "w"), indent=2)

    output = {
        "name":  "Timofei",
        "mode":  MODE,
        "date":  datetime.now().isoformat(),
        "model": OPENAI_MODEL if MODE == "gpt" else GROQ_MODEL,
        "k_examples":  K_EXAMPLES,
        "split_file":  SPLIT_FILE,
        "n_test":      len(test_df),
        "evaluators": {
            m["label"]: {"metrics": m, "rows": r}
            for m, r in collected
        },
    }
    results_file = f"results_Timofei_{MODE}.json"
    json.dump(output, open(results_file, "w"), indent=2)

    print(f"\n  Saved → {results_file}")
    print(f"  Cache → {cache_file}")


if __name__ == "__main__":
    main()
