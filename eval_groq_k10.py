"""
eval_groq_k10.py — Agent1+Groq vs RAG+Groq(k=10) on the fixed final_split.json test set.

Saves to results_Timofei_groq_k10.json — existing result files are never touched.
Resume-safe: if interrupted, re-running picks up from the cache.

Usage:
    python3 eval_groq_k10.py
"""
from __future__ import annotations

import json, os, time
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Users/timofeidrankin/Desktop/project_Agent/.env")

K            = 10
MODEL        = "llama-3.1-8b-instant"
RESULTS_FILE = "results_Timofei_groq_k10.json"
CACHE_FILE   = "cache_groq_k10.json"
SPLIT_FILE   = "final_split.json"

# ── Data ──────────────────────────────────────────────────────────────────────

def load_reviews() -> pd.DataFrame:
    from supabase import create_client
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    rows   = client.table("llm_reviews").select("*").order("id").execute().data
    df     = pd.DataFrame(rows)
    df["true_verdict"] = df.apply(
        lambda r: r["llm_verdict"] if r["nigel_rating"] == "agree"
                  else ("good" if r["llm_verdict"] == "bad" else "bad"),
        axis=1,
    )
    return df

def apply_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split    = json.load(open(SPLIT_FILE))
    train_df = df[df["id"].isin(split["train_ids"])].reset_index(drop=True)
    test_df  = df[df["id"].isin(split["test_ids"])].reset_index(drop=True)
    return train_df, test_df

# ── TF-IDF index ──────────────────────────────────────────────────────────────

def _row_to_text(row: pd.Series) -> str:
    parts = [str(row["question_text"])]
    pq = str(row.get("parent_question") or "")
    if pq and pq not in ("None", "nan", ""):
        parts.append(f"context: {pq}")
    parts.append(str(row["shopper_answer"]))
    return " | ".join(parts)

class RAGIndex:
    def __init__(self, train_df: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        self._cos   = cos_sim
        self.df     = train_df.reset_index(drop=True)
        texts       = [_row_to_text(r) for _, r in train_df.iterrows()]
        print(f"Building TF-IDF index over {len(texts)} training rows ...")
        self.vec    = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                      min_df=1, sublinear_tf=True)
        self.mat    = self.vec.fit_transform(texts)

    def retrieve(self, question: str, answer: str, k: int):
        q   = self.vec.transform([f"{question} | {answer}"])
        sim = self._cos(q, self.mat).squeeze()
        top = np.argsort(sim)[::-1][:k]
        return [(self.df.iloc[i], float(sim[i])) for i in top]

# ── Prompts ───────────────────────────────────────────────────────────────────

from Agent1 import SYSTEM_PROMPT

_VERDICT_NL = {"good": "goed", "bad": "slecht"}

def build_rag_prompt(examples) -> str:
    header = "\n\nVOORBEELDEN UIT EERDERE BEOORDELINGEN (vergelijkbare vragen):\n"
    lines  = [SYSTEM_PROMPT, header]
    for i, (row, score) in enumerate(examples, 1):
        verdict = _VERDICT_NL.get(str(row["true_verdict"]).strip().lower(), "slecht")
        reason  = (str(row.get("nigel_comment") or "") or
                   str(row.get("llm_reason") or "")).strip()
        q  = str(row["question_text"])[:120]
        a  = str(row["shopper_answer"])[:150]
        pq = str(row.get("parent_question") or "")
        pa = str(row.get("parent_answer")   or "")
        ctx = f'[context: "{pq}" -> "{pa}"] ' if pq and pq not in ("None", "nan") else ""
        lines.append(
            f"Voorbeeld {i} (overeenkomst {score:.2f}):\n"
            f"  Vraag: {ctx}{q}\n"
            f"  Antwoord: {a}\n"
            f"  Beoordeling: {verdict}: {reason}\n"
        )
    return "\n".join(lines)

def _build_question(row: pd.Series) -> str:
    q  = str(row["question_text"])
    pq = str(row.get("parent_question") or "")
    pa = str(row.get("parent_answer")   or "")
    if pq and pq not in ("None", "nan", ""):
        ctx = f'[toelichting op: "{pq}"'
        if pa and pa not in ("None", "nan", ""):
            ctx += f' -> antwoord: "{pa}"'
        return f"{ctx}]\n{q}"
    return q

# ── LLM call ──────────────────────────────────────────────────────────────────

def _parse(raw: str) -> str:
    return "good" if raw.strip().lower().startswith("goed") else "bad"

def call_groq(client, system: str, question: str, answer: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Vraag: {question}\nAntwoord: {answer}"},
        ],
        max_tokens=60,
        temperature=0,
    )
    return _parse(resp.choices[0].message.content)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict], label: str) -> dict:
    n  = len(rows)
    tp = sum(1 for r in rows if r["true"] == "good" and r["predicted"] == "good")
    tn = sum(1 for r in rows if r["true"] == "bad"  and r["predicted"] == "bad")
    fp = sum(1 for r in rows if r["true"] == "bad"  and r["predicted"] == "good")
    fn = sum(1 for r in rows if r["true"] == "good" and r["predicted"] == "bad")
    prec      = tp / (tp + fp) if (tp + fp) else 0.0
    rec       = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc       = (tp + tn) / n
    bad_rec   = tn / (tn + fp) if (tn + fp) else 0.0
    bad_prec  = tn / (tn + fn) if (tn + fn) else 0.0
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Accuracy : {acc*100:.1f}%")
    print(f"  [good]  Prec={prec*100:.1f}%  Rec={rec*100:.1f}%  F1={f1*100:.1f}%")
    print(f"  [bad ]  Prec={bad_prec*100:.1f}%  Rec={bad_rec*100:.1f}%")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    return dict(label=label, accuracy=acc, precision=prec, recall=rec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn, n=n,
                bad_recall=bad_rec, bad_precision=bad_prec)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    print("Downloading reviews from Supabase ...")
    df = load_reviews()
    train_df, test_df = apply_split(df)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    index = RAGIndex(train_df)

    # Load resume cache
    cache: dict = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}

    agent1_rows, rag_rows = [], []
    n = len(test_df)

    print(f"\nRunning Agent1 + RAG(k={K}) on {n} test rows ...\n")

    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        row_id   = int(row["id"])
        question = _build_question(row)
        answer   = str(row["shopper_answer"])
        true_v   = row["true_verdict"]

        # Agent1
        key_a1 = f"a1_{row_id}"
        if key_a1 in cache:
            pred_a1 = cache[key_a1]
        else:
            try:
                pred_a1 = call_groq(client, SYSTEM_PROMPT, question, answer)
            except Exception as e:
                print(f"  [!] Agent1 error row {row_id}: {e}")
                pred_a1 = "good"
            cache[key_a1] = pred_a1
            json.dump(cache, open(CACHE_FILE, "w"))
            time.sleep(0.3)

        # RAG k=10
        key_rag = f"rag_k{K}_{row_id}"
        if key_rag in cache:
            pred_rag = cache[key_rag]
        else:
            examples   = index.retrieve(question, answer, k=K)
            aug_prompt = build_rag_prompt(examples)
            try:
                pred_rag = call_groq(client, aug_prompt, question, answer)
            except Exception as e:
                print(f"  [!] RAG error row {row_id}: {e}")
                pred_rag = "good"
            cache[key_rag] = pred_rag
            json.dump(cache, open(CACHE_FILE, "w"))
            time.sleep(0.3)

        ia = "+" if pred_a1  == true_v else "x"
        ir = "+" if pred_rag == true_v else "x"
        print(f"  {i:>3}/{n}  id={row_id:>3}  true={true_v:<4}  "
              f"a1={pred_a1:<4}{ia}  rag={pred_rag:<4}{ir}")

        agent1_rows.append({"id": row_id, "true": true_v,
                             "predicted": pred_a1,  "correct": pred_a1  == true_v})
        rag_rows.append(   {"id": row_id, "true": true_v,
                             "predicted": pred_rag, "correct": pred_rag == true_v})

    print("\n" + "=" * 55)
    m_a1  = compute_metrics(agent1_rows, f"Agent1+{MODEL}")
    m_rag = compute_metrics(rag_rows,    f"RAG+{MODEL}(k={K})")

    output = {
        "name":       "Timofei",
        "mode":       "groq",
        "date":       datetime.now().isoformat(),
        "model":      MODEL,
        "k_examples": K,
        "split_file": SPLIT_FILE,
        "n_test":     n,
        "evaluators": {
            m_a1["label"]:  {"metrics": m_a1,  "rows": agent1_rows},
            m_rag["label"]: {"metrics": m_rag, "rows": rag_rows},
        },
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")

if __name__ == "__main__":
    main()
