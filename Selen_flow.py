"""
Selen_flow.py — Prosecutor multi-agent flow on the fixed final_split.json test set.

Agent 1 results are loaded from cache_selen.json (already computed — no re-runs).
Agent 2 (Prosecutor) receives Agent 1's full reasoning and acts as a strict critic:
  - Skeptical of "goed" verdicts, looks for missing detail or vagueness
  - Confirms "slecht" verdicts unless the answer clearly meets all requirements
  - Final verdict is Agent 2's decision

Saves to selen_prosecutor.json.
Resume-safe: prosecutor results cached under "pros_*" keys in cache_selen.json.

Usage:
    python3 Selen_flow.py
"""
from __future__ import annotations

import json, os, time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Users/timofeidrankin/Desktop/project_Agent/.env")

MODEL        = "gpt-4o-mini"
CACHE_FILE   = "cache_selen.json"
SPLIT_FILE   = "final_split.json"
RESULTS_PROS = "selen_prosecutor.json"

# ── System prompts ────────────────────────────────────────────────────────────

from Agent1 import SYSTEM_PROMPT

PROSECUTOR_PROMPT = """Je bent een kritische kwaliteitscontroleur voor WRAPP. Je controleert beoordelingen van mystery shopper antwoorden.

Je ontvangt:
- De beoordelingsrichtlijnen van Agent 1
- De vraag en het shopper-antwoord
- De beslissing en redenering van Agent 1

Jouw taak: bevestig de beslissing van Agent 1 in de meeste gevallen, maar corrigeer naar "slecht" als er een DUIDELIJKE reden is:

Corrigeer "goed" naar "slecht" alleen als het antwoord:
- Puur een conclusie is zonder enige situatiebeschrijving ("goed", "prima", "ja het klopte")
- Tegenstrijdig is met de vraag of het eerdere ja/nee antwoord
- Volledig vaag is en de eigenaar van WRAPP er niets mee kan ("het was oké")
- Onzin of niet-relevant is

Bevestig "goed" als het antwoord ook maar enige concrete observatie bevat, zelfs als het kort is.
Bevestig "slecht" van Agent 1 altijd, tenzij het antwoord duidelijk wél aan de eisen voldoet.

Antwoord ALTIJD in dit formaat (max 1 zin reden):
goed: <korte reden>
of
slecht: <korte reden>"""

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

def apply_split(df: pd.DataFrame) -> pd.DataFrame:
    split   = json.load(open(SPLIT_FILE))
    return df[df["id"].isin(split["test_ids"])].reset_index(drop=True)

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

def call_prosecutor(client, question: str, answer: str, a1_raw: str) -> tuple[str, str]:
    """Returns (verdict, raw_response)."""
    user_msg = (
        f"=== BEOORDELINGSRICHTLIJNEN (gebruikt door Agent 1) ===\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"=== TE BEOORDELEN ANTWOORD ===\n"
        f"Vraag: {question}\n"
        f"Antwoord: {answer}\n\n"
        f"=== BESLISSING VAN AGENT 1 ===\n"
        f"{a1_raw}"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROSECUTOR_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=100,
        temperature=0,
    )
    raw = resp.choices[0].message.content
    return _parse(raw), raw

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict], label: str) -> dict:
    n        = len(rows)
    tp       = sum(1 for r in rows if r["true"] == "good" and r["predicted"] == "good")
    tn       = sum(1 for r in rows if r["true"] == "bad"  and r["predicted"] == "bad")
    fp       = sum(1 for r in rows if r["true"] == "bad"  and r["predicted"] == "good")
    fn       = sum(1 for r in rows if r["true"] == "good" and r["predicted"] == "bad")
    prec     = tp / (tp + fp) if (tp + fp) else 0.0
    rec      = tp / (tp + fn) if (tp + fn) else 0.0
    f1       = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc      = (tp + tn) / n
    bad_rec  = tn / (tn + fp) if (tn + fp) else 0.0
    bad_prec = tn / (tn + fn) if (tn + fn) else 0.0
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
    from openai import OpenAI
    client = OpenAI(
        api_key  = os.environ["OPENAI_API_KEY"],
        base_url = os.environ.get("UVA_API_BASE") or None,
    )

    print("Downloading reviews from Supabase ...")
    df      = load_reviews()
    test_df = apply_split(df)
    print(f"Test: {len(test_df)}")

    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found — run the original Selen_flow.py first to generate Agent 1 results.")
        return

    cache: dict = json.load(open(CACHE_FILE))

    pros_rows = []
    skipped   = 0
    errors    = 0
    n         = len(test_df)

    print(f"\nRunning Prosecutor on {n} test rows (Agent 1 loaded from cache) ...\n")

    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        row_id   = int(row["id"])
        question = _build_question(row)
        answer   = str(row["shopper_answer"])
        true_v   = row["true_verdict"]

        # Load Agent 1 from cache — skip row if missing
        key_a1 = f"a1_{row_id}"
        if key_a1 not in cache:
            print(f"  [!] Agent1 cache missing for row {row_id} — skipping")
            skipped += 1
            continue
        pred_a1 = cache[key_a1]["verdict"]
        a1_raw  = cache[key_a1]["raw"]

        # ── Prosecutor ───────────────────────────────────────────────────────
        key_pros = f"pros_{row_id}"
        if key_pros in cache:
            pred_pros = cache[key_pros]["verdict"]
        else:
            try:
                pred_pros, pros_raw = call_prosecutor(client, question, answer, a1_raw)
                cache[key_pros] = {"verdict": pred_pros, "raw": pros_raw}
                json.dump(cache, open(CACHE_FILE, "w"))
                time.sleep(0.5)
            except Exception as e:
                print(f"  [!] Prosecutor ERROR row {row_id}: {e}")
                errors += 1
                continue

        ia = "+" if pred_a1   == true_v else "x"
        ip = "+" if pred_pros == true_v else "x"
        print(f"  {i:>3}/{n}  id={row_id:>3}  true={true_v:<4}  "
              f"a1={pred_a1:<4}{ia}  pros={pred_pros:<4}{ip}")

        pros_rows.append({"id": row_id, "true": true_v,
                          "predicted": pred_pros, "correct": pred_pros == true_v})

    if skipped:
        print(f"\n  ⚠ {skipped} rows skipped — Agent 1 cache missing")
    if errors:
        print(f"\n  ⚠ {errors} rows skipped due to API errors — re-run to retry them")

    if not pros_rows:
        print("No results to save.")
        return

    print("\n" + "=" * 55)
    m_pros = compute_metrics(pros_rows, f"Prosecutor+{MODEL}")

    output = {
        "name":       "Timofei",
        "mode":       "prosecutor-gpt",
        "date":       datetime.now().isoformat(),
        "model":      MODEL,
        "split_file": SPLIT_FILE,
        "n_test":     len(pros_rows),
        "skipped":    skipped,
        "errors":     errors,
        "evaluators": {
            m_pros["label"]: {"metrics": m_pros, "rows": pros_rows},
        },
    }
    with open(RESULTS_PROS, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {RESULTS_PROS}")


if __name__ == "__main__":
    main()
