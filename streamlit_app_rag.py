"""streamlit_app_rag.py — Streamlit UI for the RAG-augmented version of the
WRAPP visit evaluator.

Runnable independently from streamlit_app.py (Agent1) and
streamlit_app_react.py (ReAct). Lives inside AgentWRAPP-mainV2/ so that
its imports (raggpt.RAGIndex, raggpt._build_augmented_system) resolve.

What this app does, in one sentence: for every text question in a visit
it (1) retrieves the top-k most similar Nigel-rated past Q/A examples
from Supabase via TF-IDF, (2) injects them as few-shot examples on top
of Nigel's original system prompt, and (3) asks GPT-4o to rate the
shopper's answer with that augmented context. The verdict-aggregation
rule is identical to Agent1 (text ≥ 70% ∧ struct ≥ 90%) so any
verdict-level difference between this app and Agent1 isolates the
contribution of RAG retrieval.

Run with:
    cd AgentWRAPP-mainV2
    streamlit run streamlit_app_rag.py --server.port 8503
"""

import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Pull the same constants + helpers Agent1 uses so the verdict math is
# byte-identical to the Agent1 UI. SYSTEM_PROMPT is Nigel's Dutch rubric.
from agent1gpt import (
    load_data,
    TEXT_THRESHOLD,
    STRUCTURED_THRESHOLD,
    RULE_BASED_TYPES,
    FOLLOW_UP_LABELS,
    _parse_verdict,
)
from raggpt import (
    load_reviews,
    split_train_test,
    RAGIndex,
    _build_augmented_system,
    TRAIN_SEED,
    N_TRAIN,
    K_EXAMPLES,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="QA Visit Evaluator — RAG", layout="wide")
st.title("QA Visit Evaluator — RAG")
st.caption("Retrieval-Augmented Generation prototype. For each text question "
           "the agent retrieves the most similar Nigel-rated past examples "
           "and uses them as few-shot context for GPT-4o.")

# ── Name gate (same UX as the other two apps) ────────────────────────────────

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.user_name:
    st.text_input("Enter your name to continue:", key="name_input")
    if st.button("Continue") and st.session_state.name_input.strip():
        st.session_state.user_name = st.session_state.name_input.strip()
        st.rerun()
    st.stop()

is_nigel = st.session_state.user_name.strip().lower() == "nigel"

# ── Defensive secrets accessor ────────────────────────────────────────────────

def _safe_secret(key: str):
    """Look in os.environ first (works locally via .env), then fall back
    to st.secrets (works on Streamlit Cloud). Doing it in this order
    means we never trigger Streamlit's 'No secrets.toml found' warning
    when a local .env is providing the value."""
    val = os.environ.get(key)
    if val:
        return val
    try:
        return st.secrets.get(key)
    except (FileNotFoundError, KeyError):
        return None


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_data()


@st.cache_data(show_spinner="Loading Nigel ratings from Supabase…")
def get_train_df():
    """Fetch llm_reviews and take the 111-row training split (same seed
    Rag.py uses, so the index built here matches what Rag.py builds)."""
    df = load_reviews()
    train, _ = split_train_test(df)
    return train


@st.cache_resource(show_spinner="Building TF-IDF retrieval index…")
def get_rag_index():
    return RAGIndex(get_train_df())


@st.cache_resource
def get_openai_client():
    from openai import OpenAI
    api_key = _safe_secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = _safe_secret("UVA_API_BASE") or os.environ.get("UVA_API_BASE")
    return OpenAI(api_key=api_key, base_url=base_url or None)


# ── Visit row prep (same parent-question linking as Agent1 / Agent_ReAct) ─────

def _prepare_visit_rows(visit_id: int, df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["visit_id"] == visit_id].copy()
    rows = rows.sort_values(["section_id", "question_order"]).reset_index(drop=True)

    descriptions = rows["description"].tolist()
    raw_answers = rows.apply(
        lambda r: str(r["answer_text"]) if pd.notna(r["answer_text"])
                  else (str(r["answer_value"]) if pd.notna(r["answer_value"]) else ""),
        axis=1,
    ).tolist()

    parent_qs, parent_as = [], []
    for i, desc in enumerate(descriptions):
        if str(desc).strip().lower().rstrip(".") in FOLLOW_UP_LABELS and i > 0:
            pq, pa = None, None
            for j in range(i - 1, -1, -1):
                if str(descriptions[j]).strip().lower().rstrip(".") not in FOLLOW_UP_LABELS:
                    pq, pa = descriptions[j], raw_answers[j]
                    break
            parent_qs.append(pq)
            parent_as.append(pa)
        else:
            parent_qs.append(None)
            parent_as.append(None)

    rows["parent_question"] = parent_qs
    rows["parent_answer"] = parent_as
    return rows


# ── Per-visit RAG evaluation ──────────────────────────────────────────────────

def rag_evaluate_visit(visit_id: int,
                       df: pd.DataFrame,
                       index: RAGIndex,
                       client,
                       k: int = K_EXAMPLES,
                       model: str = "gpt-4o") -> dict:
    """For each text question in the visit:
       1. Build the question-with-parent-context string (same as Agent1).
       2. Retrieve top-k similar Nigel-rated examples from the train index.
       3. Inject them as few-shots in the system prompt.
       4. Ask GPT-4o for goed/slecht.
       Then apply Agent1's PASS/FAIL threshold rule.
       Returns a dict the UI can render directly."""
    rows = _prepare_visit_rows(visit_id, df)

    text_judgements = []
    for _, row in rows.iterrows():
        if row["question_type_label"] != "Tekst":
            continue

        qid = int(row["qaas_question_id"])
        ans = row["answer_text"]

        if pd.isna(ans) or str(ans).strip() == "":
            text_judgements.append({
                "qid": qid,
                "question": str(row["description"]),
                "parent_question": str(row["parent_question"]) if row["parent_question"] else None,
                "parent_answer": str(row["parent_answer"]) if row["parent_answer"] else None,
                "answer": "",
                "verdict": "bad",
                "reason": "Geen antwoord gegeven",
                "retrieved": [],
            })
            continue

        # Wrap follow-ups with their parent context (identical to Agent1's
        # evaluate_answer for the Tekst branch).
        label = str(row["description"]).strip().lower().rstrip(".")
        if label in FOLLOW_UP_LABELS and row["parent_question"]:
            ctx = f'[toelichting op: "{row["parent_question"]}"'
            if row["parent_answer"]:
                ctx += f' → antwoord: "{row["parent_answer"]}"'
            ctx += f']\n{row["description"]}'
            question = ctx
        else:
            question = str(row["description"])

        # RAG step: retrieve similar Nigel-rated examples and build a
        # system prompt that is SYSTEM_PROMPT + few-shot examples block.
        examples = index.retrieve(question, str(ans), k=k)
        augmented_system = _build_augmented_system(examples)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": f"Vraag: {question}\nAntwoord: {ans}"},
            ],
            max_tokens=60,
            temperature=0,
        )
        is_good, reason = _parse_verdict(response.choices[0].message.content)

        # Repackage retrieved examples for the UI.
        retrieved = []
        for ex_row, score in examples:
            retrieved.append({
                "question": str(ex_row["question_text"]),
                "parent_question": (str(ex_row.get("parent_question") or "") or None)
                                   if str(ex_row.get("parent_question") or "") not in ("None", "nan") else None,
                "parent_answer": (str(ex_row.get("parent_answer") or "") or None)
                                 if str(ex_row.get("parent_answer") or "") not in ("None", "nan") else None,
                "answer": str(ex_row["shopper_answer"]),
                "true_verdict": str(ex_row["true_verdict"]),
                "nigel_comment": (str(ex_row.get("nigel_comment") or "") or None)
                                 if str(ex_row.get("nigel_comment") or "") not in ("None", "nan") else None,
                "llm_reason": (str(ex_row.get("llm_reason") or "") or None)
                              if str(ex_row.get("llm_reason") or "") not in ("None", "nan") else None,
                "score": float(score),
            })

        text_judgements.append({
            "qid": qid,
            "question": str(row["description"]),
            "parent_question": str(row["parent_question"]) if row["parent_question"] else None,
            "parent_answer": str(row["parent_answer"]) if row["parent_answer"] else None,
            "answer": str(ans),
            "verdict": "good" if is_good else "bad",
            "reason": reason,
            "retrieved": retrieved,
        })

    # Structured completeness (same rule Agent1 + Agent_ReAct apply):
    # RULE_BASED_TYPES must have an answer; other non-Tekst types are auto-good.
    structured = rows[rows["question_type_label"] != "Tekst"]
    rule_based_mask = structured["question_type_label"].isin(RULE_BASED_TYPES)
    has_text = structured["answer_text"].notna() & (
        structured["answer_text"].astype(str).str.strip() != ""
    )
    has_value = structured["answer_value"].notna()
    struct_is_good = (~rule_based_mask) | has_text | has_value
    struct_good = int(struct_is_good.sum())
    struct_total = int(len(structured))
    struct_pct = struct_good / struct_total if struct_total > 0 else 1.0

    text_total = len(text_judgements)
    text_good = sum(1 for j in text_judgements if j["verdict"] == "good")
    text_pct = text_good / text_total if text_total > 0 else 1.0

    passed = text_pct >= TEXT_THRESHOLD and struct_pct >= STRUCTURED_THRESHOLD

    return {
        "visit_id": int(visit_id),
        "verdict": "PASS" if passed else "FAIL",
        "text_good": text_good,
        "text_total": text_total,
        "text_pct": round(text_pct, 4),
        "struct_good": struct_good,
        "struct_total": struct_total,
        "struct_pct": round(struct_pct, 4),
        "text_judgements": text_judgements,
        "k_used": k,
        "structured_rows": structured,
    }


# ── Load everything ───────────────────────────────────────────────────────────

df = get_data()
client = get_openai_client()
available_ids = sorted(df["visit_id"].unique())

# Heavy resources — built once, cached.
try:
    index = get_rag_index()
    train_size = len(get_train_df())
except Exception as e:
    st.error(f"Could not load RAG index from Supabase: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.caption(f"Logged in as: **{st.session_state.user_name}**")
    if st.button("Change name", use_container_width=True):
        st.session_state.user_name = ""
        st.rerun()
    st.divider()
    st.header("Select visit")
    visit_id = st.selectbox("Visit ID", available_ids)
    k = st.slider("Retrieved examples per question (k)",
                  min_value=1, max_value=10, value=K_EXAMPLES,
                  help="How many Nigel-rated past examples the retriever "
                       "injects as few-shots. Higher k = more context, more "
                       "tokens per call.")
    run = st.button("Evaluate (RAG)", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"TF-IDF index size: **{train_size}** training rows "
               f"(seed={TRAIN_SEED})")

# ── Run evaluation ────────────────────────────────────────────────────────────

if run:
    with st.spinner(f"Running RAG evaluation on visit {visit_id}…"):
        report = rag_evaluate_visit(int(visit_id), df, index, client, k=k)
    st.session_state.rag_report = report

if "rag_report" not in st.session_state:
    st.info("Select a visit ID in the sidebar and click **Evaluate (RAG)**.")
    st.stop()

report = st.session_state.rag_report

# ── Verdict banner + metrics ──────────────────────────────────────────────────

verdict = report["verdict"]
color = "green" if verdict == "PASS" else "red"
st.markdown(
    f"<h2 style='color:{color}'>{'✅ PASS' if verdict == 'PASS' else '❌ FAIL'}</h2>",
    unsafe_allow_html=True,
)
st.caption(f"k = {report['k_used']} retrieved examples per question. "
           f"Same PASS rule as Agent1: text ≥ {int(TEXT_THRESHOLD*100)}% "
           f"AND struct ≥ {int(STRUCTURED_THRESHOLD*100)}%.")

m1, m2 = st.columns(2)
m1.metric("Structured", f"{report['struct_good']}/{report['struct_total']}",
          f"{report['struct_pct']*100:.1f}% (threshold {STRUCTURED_THRESHOLD*100:.0f}%)")
m2.metric("Text",      f"{report['text_good']}/{report['text_total']}",
          f"{report['text_pct']*100:.1f}% (threshold {TEXT_THRESHOLD*100:.0f}%)")

st.divider()

# ── Structured questions (read-only, same display as v1) ──────────────────────

structured = report["structured_rows"]
col1, col2 = st.columns(2)

with col1:
    st.subheader("Structured questions")
    rule_based_mask = structured["question_type_label"].isin(RULE_BASED_TYPES)
    has_text = structured["answer_text"].notna() & (
        structured["answer_text"].astype(str).str.strip() != ""
    )
    has_value = structured["answer_value"].notna()
    struct_is_good = (~rule_based_mask) | has_text | has_value
    good_s = int(struct_is_good.sum())
    st.caption(f"{good_s}/{len(structured)} answered")

    for (_, row), is_good in zip(structured.iterrows(), struct_is_good):
        ans = row["answer_text"] if pd.notna(row["answer_text"]) else row["answer_value"]
        icon = "✅" if is_good else "❌"
        st.markdown(f"{icon} **[{row['question_type_label']}]** {row['description']}")
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;→ `{ans}`")

# ── Text questions with retrieved examples ────────────────────────────────────

with col2:
    st.subheader("Text questions (RAG-evaluated)")
    st.caption(f"{report['text_good']}/{report['text_total']} good")

    for j in report["text_judgements"]:
        qid = j["qid"]
        icon = "✅" if j["verdict"] == "good" else "❌"
        with st.expander(f"{icon} Q{qid} — {j['question']}",
                         expanded=(j["verdict"] == "bad")):
            if j.get("parent_question"):
                pa_str = f' → **"{j["parent_answer"]}"**' if j.get("parent_answer") else ""
                st.caption(f'Follow-up on: *{j["parent_question"]}*{pa_str}')

            st.markdown(f"**Answer:** {j['answer']}" if j["answer"]
                        else "**Answer:** _(empty)_")
            badge = "🟢 Good" if j["verdict"] == "good" else "🔴 Bad"
            st.markdown(f"**RAG verdict:** {badge}")
            if j["reason"]:
                st.caption(f"ℹ️ {j['reason']}")

            # The interesting part — what the retriever pulled in.
            if j["retrieved"]:
                st.markdown("---")
                st.markdown(f"**🔍 Retrieved examples (top {len(j['retrieved'])} "
                            f"by TF-IDF cosine similarity):**")
                for i, ex in enumerate(j["retrieved"], 1):
                    ex_icon = "🟢" if ex["true_verdict"] == "good" else "🔴"
                    score_pct = ex["score"] * 100
                    ex_q = ex["question"]
                    if ex.get("parent_question"):
                        ex_q = f'[context: "{ex["parent_question"]}"] {ex_q}'
                    reason_text = ex.get("nigel_comment") or ex.get("llm_reason") or ""
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;**{i}.** _similarity {score_pct:.1f}%_ "
                        f"&nbsp; {ex_icon} **{ex['true_verdict']}**"
                    )
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                        f"**Q:** {ex_q[:160]}"
                    )
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                        f"**A:** {ex['answer'][:200]}"
                    )
                    if reason_text:
                        st.caption(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                            f"_Nigel/LLM note: {reason_text[:200]}_"
                        )
            else:
                if j["answer"]:
                    st.caption("No examples retrieved (this shouldn't happen — "
                               "check the index).")
