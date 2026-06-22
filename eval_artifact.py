"""
eval_artifact.py — RAG vs Agent1 evaluation for thesis artifact.

Evaluates shopper answers from the llm_reviews Supabase table using:
  - Agent1: single LLM call with the WRAPP evaluation system prompt
  - RAG:    same prompt augmented with k similar labelled examples (TF-IDF retrieval)

Configuration is at the top of this file. Supported backends:
  "groq"      — llama-3.1-8b-instant (free, via Groq API)
  "gpt-mini"  — gpt-4o-mini (via OpenAI or UvA LiteLLM proxy)

Results are saved to a JSON file and never overwrite existing files.
The script is resume-safe: interrupting and restarting picks up from the cache.

Usage:
    python3 eval_artifact.py
"""
from __future__ import annotations

import json, os, time
from datetime import datetime

import numpy as np
import pandas as pd

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  edit these before running
# ═════════════════════════════════════════════════════════════════════════════

BACKEND = "gpt-mini"   # "groq"  or  "gpt-mini"
K       = 10           # number of RAG examples (5 or 10 recommended)

# API keys — fill in directly, or leave empty to read from environment variables
OPENAI_API_KEY = ""    # needed when BACKEND = "gpt-mini"
GROQ_API_KEY   = ""    # needed when BACKEND = "groq"
SUPABASE_URL   = ""    # always required (to fetch review data)
SUPABASE_KEY   = ""    # always required

# Optional: UvA LiteLLM proxy base URL (only for gpt-mini via UvA workspace key)
# Leave empty to call api.openai.com directly.
UVA_API_BASE   = ""

# ═════════════════════════════════════════════════════════════════════════════

assert BACKEND in ("groq", "gpt-mini"), f"BACKEND must be 'groq' or 'gpt-mini', got {BACKEND!r}"

# Resolve keys: explicit value wins, else fall back to environment variable
def _key(explicit: str, env_var: str) -> str:
    return explicit.strip() or os.environ.get(env_var, "")

_openai_key  = _key(OPENAI_API_KEY, "OPENAI_API_KEY")
_groq_key    = _key(GROQ_API_KEY,   "GROQ_API_KEY")
_supa_url    = _key(SUPABASE_URL,   "SUPABASE_URL")
_supa_key    = _key(SUPABASE_KEY,   "SUPABASE_KEY")
_uva_base    = _key(UVA_API_BASE,   "UVA_API_BASE") or None

_MODELS = {"groq": "llama-3.1-8b-instant", "gpt-mini": "gpt-4o-mini"}
MODEL = _MODELS[BACKEND]

SPLIT_FILE   = "final_split.json"
RESULTS_FILE = f"results_artifact_{BACKEND}_k{K}.json"
CACHE_FILE   = f"cache_artifact_{BACKEND}_k{K}.json"

# ── System prompt (WRAPP evaluation guidelines) ───────────────────────────────

SYSTEM_PROMPT = """Je beoordeelt open antwoorden van mystery shoppers voor WRAPP in het Nederlands.

ALGEMENE REGEL (van de eigenaar van WRAPP):
Leg je ja/nee of 1-5 antwoord kort uit. Beschrijf de situatie die je antwoord bevestigt en verklaart.
Bij subjectieve vragen (eten, smaak, sfeer): leg ook uit WAAROM je het goed of niet goed vond.
Het moet beschrijvend genoeg zijn om nuttig te zijn voor de eigenaar.
"Het eten was niet lekker" is NIET genoeg. Goed is: "De friet was niet knapperig en de saus was erg zuur."

EXTRA REGELS:
- Korte antwoorden zijn ok als de vraag om een korte uitleg vraagt
- Voor feitelijke vragen (adres, tijd, bedrag, product, naam): een kort feitelijk antwoord is ALTIJD goed
- Bij een vraag over een ingrediënt: vermeld zowel de vraag die je stelde als het antwoord van de medewerker

GOEDE ANTWOORDEN (echte voorbeelden):
- Toelichting begroeting: "De medewerker achter de toonbank keek mij vriendelijk aan en heette mij welkom." → goed
- Toelichting eten (positief): "Beide pita's waren goed op smaak en een lekkere mix van ingrediënten." → goed
- Toelichting eten (negatief): "De zoete aardappelfriet had teveel zout. De kipshoarma was alleen onderin te proeven." → goed
- Toelichting ingrediëntvraag: "Ik vroeg wat za'atar was. De medewerker gaf aan dat dit een kruid was." → goed
- Toelichting wachttijd: "We werden binnen een minuut na binnenkomst geholpen. Er waren geen andere klanten." → goed

SLECHTE ANTWOORDEN:
- Vloeken in welke taal dan ook → slecht
- Onzinnige woorden: "asdfsda", "weasdfsd" → slecht
- Antwoord van 1 woord als uitleg gevraagd wordt: "ja", "goed", "prima" → slecht
- Antwoord tegenstrijdig met het vorige ja/nee antwoord → slecht

Antwoord ALTIJD in dit formaat (max 1 zin reden):
goed: <korte reden>
of
slecht: <korte reden>"""

# ── Data loading ──────────────────────────────────────────────────────────────

def load_reviews() -> pd.DataFrame:
    from supabase import create_client
    client = create_client(_supa_url, _supa_key)
    rows   = client.table("llm_reviews").select("*").order("id").execute().data
    df     = pd.DataFrame(rows)
    # Ground truth: flip llm_verdict where Nigel disagreed
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

def _build_question(row: pd.Series) -> str:
    """Construct the question string, prepending parent context if present."""
    q  = str(row["question_text"])
    pq = str(row.get("parent_question") or "")
    pa = str(row.get("parent_answer")   or "")
    if pq and pq not in ("None", "nan", ""):
        ctx = f'[toelichting op: "{pq}"'
        if pa and pa not in ("None", "nan", ""):
            ctx += f' -> antwoord: "{pa}"'
        return f"{ctx}]\n{q}"
    return q

# ── TF-IDF RAG index ──────────────────────────────────────────────────────────

def _row_to_text(row: pd.Series) -> str:
    parts = [str(row["question_text"])]
    pq = str(row.get("parent_question") or "")
    if pq and pq not in ("None", "nan", ""):
        parts.append(f"context: {pq}")
    parts.append(str(row["shopper_answer"]))
    return " | ".join(parts)

class RAGIndex:
    """TF-IDF cosine similarity retrieval over the training set."""

    def __init__(self, train_df: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self._cos = cosine_similarity
        self.df   = train_df.reset_index(drop=True)
        texts     = [_row_to_text(r) for _, r in train_df.iterrows()]
        print(f"Building TF-IDF index over {len(texts)} training rows ...")
        self.vec  = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                    min_df=1, sublinear_tf=True)
        self.mat  = self.vec.fit_transform(texts)

    def retrieve(self, question: str, answer: str, k: int) -> list:
        q   = self.vec.transform([f"{question} | {answer}"])
        sim = self._cos(q, self.mat).squeeze()
        top = np.argsort(sim)[::-1][:k]
        return [(self.df.iloc[i], float(sim[i])) for i in top]

# ── Prompt builders ───────────────────────────────────────────────────────────

_VERDICT_NL = {"good": "goed", "bad": "slecht"}

def build_rag_prompt(examples: list) -> str:
    """Augment SYSTEM_PROMPT with k labelled examples."""
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

# ── LLM call ──────────────────────────────────────────────────────────────────

def _parse(raw: str) -> str:
    return "good" if raw.strip().lower().startswith("goed") else "bad"

def call_llm(client, system: str, question: str, answer: str) -> str:
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
    # Build LLM client
    if BACKEND == "groq":
        from groq import Groq
        client = Groq(api_key=_groq_key)
    else:
        from openai import OpenAI
        client = OpenAI(api_key=_openai_key, base_url=_uva_base)

    print(f"Backend : {BACKEND}  |  Model : {MODEL}  |  k = {K}")
    print("Downloading reviews from Supabase ...")
    df = load_reviews()
    train_df, test_df = apply_split(df)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    index = RAGIndex(train_df)
    cache: dict = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}

    a1_rows, rag_rows = [], []
    errors = 0
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
                pred_a1 = call_llm(client, SYSTEM_PROMPT, question, answer)
                cache[key_a1] = pred_a1
                json.dump(cache, open(CACHE_FILE, "w"))
                time.sleep(0.5)
            except Exception as e:
                print(f"  [!] Agent1 ERROR row {row_id}: {e}")
                errors += 1
                continue

        # RAG
        key_rag = f"rag_k{K}_{row_id}"
        if key_rag in cache:
            pred_rag = cache[key_rag]
        else:
            examples   = index.retrieve(question, answer, k=K)
            aug_prompt = build_rag_prompt(examples)
            try:
                pred_rag = call_llm(client, aug_prompt, question, answer)
                cache[key_rag] = pred_rag
                json.dump(cache, open(CACHE_FILE, "w"))
                time.sleep(0.5)
            except Exception as e:
                print(f"  [!] RAG ERROR row {row_id}: {e}")
                errors += 1
                continue

        ia = "+" if pred_a1  == true_v else "x"
        ir = "+" if pred_rag == true_v else "x"
        print(f"  {i:>3}/{n}  id={row_id:>3}  true={true_v:<4}  "
              f"a1={pred_a1:<4}{ia}  rag={pred_rag:<4}{ir}")

        a1_rows.append({"id": row_id, "true": true_v,
                        "predicted": pred_a1,  "correct": pred_a1  == true_v})
        rag_rows.append({"id": row_id, "true": true_v,
                         "predicted": pred_rag, "correct": pred_rag == true_v})

    if errors:
        print(f"\n  ⚠ {errors} rows skipped due to API errors — re-run to retry")

    if not a1_rows:
        print("No results to save.")
        return

    print("\n" + "=" * 55)
    m_a1  = compute_metrics(a1_rows,  f"Agent1+{MODEL}")
    m_rag = compute_metrics(rag_rows, f"RAG+{MODEL}(k={K})")

    output = {
        "backend":    BACKEND,
        "date":       datetime.now().isoformat(),
        "model":      MODEL,
        "k_examples": K,
        "split_file": SPLIT_FILE,
        "n_test":     len(a1_rows),
        "errors":     errors,
        "evaluators": {
            m_a1["label"]:  {"metrics": m_a1,  "rows": a1_rows},
            m_rag["label"]: {"metrics": m_rag, "rows": rag_rows},
        },
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
