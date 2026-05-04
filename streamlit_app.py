import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from Agent1 import load_data, evaluate_visit, GroqEvaluator, RULE_BASED_TYPES

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="QA Visit Evaluator", layout="wide")
st.title("QA Visit Evaluator")

# ── Load data once ────────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_evaluator():
    return GroqEvaluator(api_key=os.environ["GROQ_API_KEY"])

df = get_data()
evaluator = get_evaluator()

available_ids = sorted(df["visit_id"].unique())

# ── Sidebar input ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Select visit")
    visit_id = st.selectbox("Visit ID", available_ids)
    run = st.button("Evaluate", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────

if run:
    with st.spinner(f"Evaluating visit {visit_id}…"):
        report = evaluate_visit(visit_id, df, evaluator)

    details = report["details"]

    # ── Verdict banner ────────────────────────────────────────────────────────
    verdict = report["verdict"]
    color = "green" if verdict == "PASS" else "red"
    st.markdown(
        f"<h2 style='color:{color}'>{'✅ PASS' if verdict == 'PASS' else '❌ FAIL'}</h2>",
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    m1.metric("Structured answers", f"{report['struct_good']}/{report['struct_total']}",
              f"{report['struct_pct']}% (threshold 90%)")
    m2.metric("Text answers", f"{report['text_good']}/{report['text_total']}",
              f"{report['text_pct']}% (threshold 70%)")
    st.divider()

    text_rows = details[details["question_type_label"] == "Tekst"]
    structured_rows = details[details["question_type_label"] != "Tekst"]

    col1, col2 = st.columns(2)

    # ── Structured questions summary ──────────────────────────────────────────
    with col1:
        st.subheader("Structured questions")
        total_s = len(structured_rows)
        good_s = (structured_rows["verdict"] == "good").sum()
        st.caption(f"{good_s}/{total_s} answered")

        for _, row in structured_rows.iterrows():
            ans = row["answer_text"] if pd.notna(row["answer_text"]) else row["answer_value"]
            icon = "✅" if row["verdict"] == "good" else "❌"
            q_label = str(row["description"])[:60]
            st.markdown(f"{icon} **[{row['question_type_label']}]** {q_label}")
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;→ `{ans}`")

    # ── Text questions with LLM decision ──────────────────────────────────────
    with col2:
        st.subheader("Text questions (LLM evaluated)")
        total_t = len(text_rows)
        good_t = (text_rows["verdict"] == "good").sum()
        st.caption(f"{good_t}/{total_t} good")

        for _, row in text_rows.iterrows():
            verdict_icon = "✅" if row["verdict"] == "good" else "❌"
            has_parent = pd.notna(row["parent_question"])

            # Build expander title: include parent context if present
            if has_parent:
                parent_a_str = f' → "{str(row["parent_answer"])}"' if pd.notna(row["parent_answer"]) and str(row["parent_answer"]).strip() else ""
                parent_ctx = f' [re: *{str(row["parent_question"])[:40]}{parent_a_str}*]'
            else:
                parent_ctx = ""

            with st.expander(
                f"{verdict_icon} Q{row['qaas_question_id']} — {str(row['description'])[:60]}{parent_ctx}",
                expanded=(row["verdict"] == "bad"),
            ):
                ans = row["answer_text"] if pd.notna(row["answer_text"]) else row["answer_value"]
                st.markdown(f"**Answer:** {ans}")
                badge = "🟢 Good" if row["verdict"] == "good" else "🔴 Bad"
                st.markdown(f"**LLM verdict:** {badge}")
                if row["reason"]:
                    st.caption(f"ℹ️ {row['reason']}")

else:
    st.info("Select a visit ID in the sidebar and click **Evaluate**.")
