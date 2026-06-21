"""
Selen_flow.py — Multi-agent flow (Agent 1 + Agent 2) on the fixed final_split.json test set.

Set MODE at the top to choose which Agent 2 personality runs:
  "prosecutor" — critical, skeptical; only overrides to "goed" when clearly justified
  "reviewer"   — lenient, checks reasoning; tends to confirm or soften "slecht" verdicts

Agent 1 results are always loaded from cache_selen.json (no re-runs).
Agent 2 results are cached under "{mode}_*" keys in cache_selen.json.

Output files:
  prosecutor → selen_prosecutor.json
  reviewer   → multi_agent_Selen.json

Usage:
    python3 Selen_flow.py
"""
from __future__ import annotations

import json, os, time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv("/Users/timofeidrankin/Desktop/project_Agent/.env")

# ── Mode switch ───────────────────────────────────────────────────────────────
# Change this to "reviewer" or "prosecutor"
MODE = "prosecutor"

MODEL      = "gpt-4o-mini"
CACHE_FILE = "cache_selen.json"
SPLIT_FILE = "final_split.json"

_RESULTS_FILE = {
    "prosecutor": "selen_prosecutor.json",
    "reviewer":   "multi_agent_Selen.json",
}
_CACHE_PREFIX = {
    "prosecutor": "pros",
    "reviewer":   "ma",
}

assert MODE in _RESULTS_FILE, f"MODE must be 'prosecutor' or 'reviewer', got {MODE!r}"

RESULTS_FILE = _RESULTS_FILE[MODE]
CACHE_PREFIX = _CACHE_PREFIX[MODE]

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

REVIEWER_PROMPT = """Je bent een kwaliteitscontroleur voor WRAPP. Je controleert beoordelingen van mystery shopper antwoorden.

Je taak: Beoordeel of de redenering van Agent 1 klopt.

Je ontvangt de volledige context:
- De richtlijnen die Agent 1 heeft gebruikt
- De vraag en het shopper-antwoord
- De beslissing en redenering van Agent 1

Ga na of de redenering logisch en consistent is met de richtlijnen.
Bevestig de beslissing als de redenering klopt.
Corrigeer de beslissing als de redenering onjuist of onvolledig is.

Antwoord ALTIJD in dit formaat (max 1 zin reden):
goed: <korte reden>
of
slecht: <korte reden>"""

AGENT2_PROMPT = PROSECUTOR_PROMPT if MODE == "prosecutor" else REVIEWER_PROMPT

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
    split = json.load(open(SPLIT_FILE))
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

def call_agent2(client, question: str, answer: str, a1_raw: str) -> tuple[str, str]:
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
            {"role": "system", "content": AGENT2_PROMPT},
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

    print(f"Mode: {MODE.upper()}")
    print("Downloading reviews from Supabase ...")
    df      = load_reviews()
    test_df = apply_split(df)
    print(f"Test: {len(test_df)}")

    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found — run the original Agent1 flow first.")
        return

    cache: dict = json.load(open(CACHE_FILE))

    a2_rows = []
    skipped = 0
    errors  = 0
    n       = len(test_df)

    print(f"\nRunning {MODE.capitalize()} on {n} test rows (Agent 1 from cache) ...\n")

    for i, (_, row) in enumerate(test_df.iterrows(), 1):
        row_id   = int(row["id"])
        question = _build_question(row)
        answer   = str(row["shopper_answer"])
        true_v   = row["true_verdict"]

        # Load Agent 1 from cache — skip if missing
        key_a1 = f"a1_{row_id}"
        if key_a1 not in cache:
            print(f"  [!] Agent1 cache missing for row {row_id} — skipping")
            skipped += 1
            continue
        pred_a1 = cache[key_a1]["verdict"]
        a1_raw  = cache[key_a1]["raw"]

        # ── Agent 2 ───────────────────────────────────────────────────────────
        key_a2 = f"{CACHE_PREFIX}_{row_id}"
        if key_a2 in cache:
            pred_a2 = cache[key_a2]["verdict"]
        else:
            try:
                pred_a2, a2_raw = call_agent2(client, question, answer, a1_raw)
                cache[key_a2] = {"verdict": pred_a2, "raw": a2_raw}
                json.dump(cache, open(CACHE_FILE, "w"))
                time.sleep(0.5)
            except Exception as e:
                print(f"  [!] {MODE.capitalize()} ERROR row {row_id}: {e}")
                errors += 1
                continue

        ia = "+" if pred_a1 == true_v else "x"
        i2 = "+" if pred_a2 == true_v else "x"
        tag = "pros" if MODE == "prosecutor" else "rev"
        print(f"  {i:>3}/{n}  id={row_id:>3}  true={true_v:<4}  "
              f"a1={pred_a1:<4}{ia}  {tag}={pred_a2:<4}{i2}")

        a2_rows.append({"id": row_id, "true": true_v,
                        "predicted": pred_a2, "correct": pred_a2 == true_v})

    if skipped:
        print(f"\n  ⚠ {skipped} rows skipped — Agent 1 cache missing")
    if errors:
        print(f"\n  ⚠ {errors} rows skipped due to API errors — re-run to retry")

    if not a2_rows:
        print("No results to save.")
        return

    print("\n" + "=" * 55)
    label  = f"{'Prosecutor' if MODE == 'prosecutor' else 'MultiAgent'}+{MODEL}"
    m_a2   = compute_metrics(a2_rows, label)

    output = {
        "name":       "Timofei",
        "mode":       f"{MODE}-gpt",
        "date":       datetime.now().isoformat(),
        "model":      MODEL,
        "split_file": SPLIT_FILE,
        "n_test":     len(a2_rows),
        "skipped":    skipped,
        "errors":     errors,
        "evaluators": {
            m_a2["label"]: {"metrics": m_a2, "rows": a2_rows},
        },
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
