"""
sql_scores.py — Retrieve LLM verdicts and Nigel ratings from the Supabase table.

Fetches all rows from the llm_reviews table, computes the ground-truth verdict
(flipping llm_verdict where Nigel disagreed), and saves the result to
llm_nigel_scores.csv for inspection.

Usage:
    python3 sql_scores.py
"""
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  fill in your Supabase credentials, or set in .env
# ═════════════════════════════════════════════════════════════════════════════

SUPABASE_URL = ""   # leave empty to read from SUPABASE_URL in .env
SUPABASE_KEY = ""   # leave empty to read from SUPABASE_KEY in .env

# ═════════════════════════════════════════════════════════════════════════════

def _key(explicit: str, env_var: str) -> str:
    return explicit.strip() or os.environ.get(env_var, "")

_url = _key(SUPABASE_URL, "SUPABASE_URL")
_key_ = _key(SUPABASE_KEY, "SUPABASE_KEY")

OUTPUT_CSV = "llm_nigel_scores.csv"

def main():
    from supabase import create_client

    print("Connecting to Supabase ...")
    client = create_client(_url, _key_)
    rows   = client.table("llm_reviews").select("*").order("id").execute().data
    df     = pd.DataFrame(rows)
    print(f"Fetched {len(df)} rows from llm_reviews table.")

    # Ground-truth verdict: flip llm_verdict where Nigel disagreed
    df["true_verdict"] = df.apply(
        lambda r: r["llm_verdict"] if r["nigel_rating"] == "agree"
                  else ("good" if r["llm_verdict"] == "bad" else "bad"),
        axis=1,
    )

    # ── Summary stats ─────────────────────────────────────────────────────────
    total    = len(df)
    agree    = (df["nigel_rating"] == "agree").sum()
    disagree = (df["nigel_rating"] == "disagree").sum()
    good_llm = (df["llm_verdict"] == "good").sum()
    bad_llm  = (df["llm_verdict"] == "bad").sum()
    good_gt  = (df["true_verdict"] == "good").sum()
    bad_gt   = (df["true_verdict"] == "bad").sum()

    print(f"\n{'═'*50}")
    print(f"  Total reviewed answers : {total}")
    print(f"{'─'*50}")
    print(f"  Nigel agreement        : {agree}  ({agree/total*100:.1f}%)")
    print(f"  Nigel disagreement     : {disagree}  ({disagree/total*100:.1f}%)")
    print(f"{'─'*50}")
    print(f"  LLM verdict good       : {good_llm}  ({good_llm/total*100:.1f}%)")
    print(f"  LLM verdict bad        : {bad_llm}  ({bad_llm/total*100:.1f}%)")
    print(f"{'─'*50}")
    print(f"  Ground-truth good      : {good_gt}  ({good_gt/total*100:.1f}%)")
    print(f"  Ground-truth bad       : {bad_gt}  ({bad_gt/total*100:.1f}%)")
    print(f"{'═'*50}")

    # ── Export ────────────────────────────────────────────────────────────────
    export_cols = [
        "id", "visit_id", "qaas_question_id",
        "question_text", "parent_question", "parent_answer",
        "shopper_answer",
        "llm_verdict", "llm_reason",
        "nigel_rating", "nigel_comment",
        "true_verdict",
        "reviewed_at",
    ]
    # Keep only columns that actually exist in the table
    export_cols = [c for c in export_cols if c in df.columns]
    df[export_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}  ({len(df)} rows, {len(export_cols)} columns)")

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\nFirst 5 rows:")
    preview = df[export_cols].head(5).to_string(index=False, max_colwidth=40)
    print(preview)


if __name__ == "__main__":
    main()
