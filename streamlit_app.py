import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from agent1gpt import load_data, evaluate_visit, OpenAIEvaluator
from image_analysis import load_all_data, analyse_visit_photos

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="QA Visit Evaluator", layout="wide")
st.title("QA Visit Evaluator")

# ── Name gate ─────────────────────────────────────────────────────────────────

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.user_name:
    st.text_input("Enter your name to continue:", key="name_input")
    if st.button("Continue") and st.session_state.name_input.strip():
        st.session_state.user_name = st.session_state.name_input.strip()
        st.rerun()
    st.stop()

is_nigel = st.session_state.user_name.strip().lower() == "nigel"

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def get_photo_data():
    return load_all_data()

@st.cache_resource
def get_evaluator():
    api_key  = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = st.secrets.get("UVA_API_BASE")   or os.environ.get("UVA_API_BASE") or None
    return OpenAIEvaluator(api_key=api_key, base_url=base_url, model="gpt-4o")

@st.cache_resource
def get_supabase():
    from supabase import create_client
    url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    return create_client(url, key)

df = get_data()
evaluator = get_evaluator()
available_ids = sorted(df["visit_id"].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.caption(f"Logged in as: **{st.session_state.user_name}**")
    if st.button("Change name", use_container_width=True):
        st.session_state.user_name = ""
        st.rerun()
    st.divider()
    st.header("Select visit")
    visit_id = st.selectbox("Visit ID", available_ids)
    run = st.button("Evaluate", type="primary", use_container_width=True)
    analyse_photos = st.button("Analyse Photos 📸", use_container_width=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def verdict_icon(v: str) -> str:
    return {"good": "✅", "bad": "❌", "unsure": "🟡"}.get(v, "❌")

def verdict_badge(v: str) -> str:
    return {"good": "🟢 Good", "bad": "🔴 Bad", "unsure": "🟡 Unsure"}.get(v, "🔴 Bad")

# ── Run evaluation ────────────────────────────────────────────────────────────

if run:
    with st.spinner(f"Evaluating visit {visit_id}…"):
        report = evaluate_visit(visit_id, df, evaluator)
    st.session_state.report = report
    st.session_state.photo_result = None
    st.session_state.saved_ratings = {}

# ── Main panel ────────────────────────────────────────────────────────────────

if "report" not in st.session_state:
    st.info("Select a visit ID in the sidebar and click **Evaluate**.")
    st.stop()

report = st.session_state.report
details = report["details"]

# Verdict banner
visit_verdict = report["verdict"]
color = "green" if visit_verdict == "PASS" else "red"
st.markdown(
    f"<h2 style='color:{color}'>{'✅ PASS' if visit_verdict == 'PASS' else '❌ FAIL'}</h2>",
    unsafe_allow_html=True,
)
m1, m2, m3 = st.columns(3)
m1.metric("Structured answers", f"{report['struct_good']}/{report['struct_total']}",
          f"{report['struct_pct']}% (threshold 90%)")
m2.metric("Text answers (good)", f"{report['text_good']}/{report['text_total']}",
          f"{report['text_pct']}% (threshold 70%)")
m3.metric("Unsure answers", report["unsure"])
st.divider()

text_rows = details[details["question_type_label"] == "Tekst"]
structured_rows = details[details["question_type_label"] != "Tekst"]

col1, col2 = st.columns(2)

# ── Structured questions ──────────────────────────────────────────────────────

with col1:
    st.subheader("Structured questions")
    good_s = (structured_rows["verdict"] == "good").sum()
    st.caption(f"{good_s}/{len(structured_rows)} answered")

    for _, row in structured_rows.iterrows():
        ans = row["answer_text"] if pd.notna(row["answer_text"]) else row["answer_value"]
        icon = verdict_icon(row["verdict"])
        st.markdown(f"{icon} **[{row['question_type_label']}]** {row['description']}")
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;→ `{ans}`")

# ── Text questions ────────────────────────────────────────────────────────────

with col2:
    if is_nigel:
        st.subheader("Text questions (LLM evaluated) — Nigel mode 🔑")
    else:
        st.subheader("Text questions (LLM evaluated)")

    good_t   = (text_rows["verdict"] == "good").sum()
    unsure_t = (text_rows["verdict"] == "unsure").sum()
    bad_t    = (text_rows["verdict"] == "bad").sum()
    st.caption(f"{good_t} good · {unsure_t} unsure · {bad_t} bad")

    for _, row in text_rows.iterrows():
        qid = int(row["qaas_question_id"])
        icon = verdict_icon(row["verdict"])
        ans = row["answer_text"] if pd.notna(row["answer_text"]) else row["answer_value"]

        with st.expander(
            f"{icon} Q{qid} — {row['description']}",
            expanded=(row["verdict"] != "good"),
        ):
            if pd.notna(row["parent_question"]):
                parent_a_str = (
                    f' → **"{str(row["parent_answer"])}"**'
                    if pd.notna(row["parent_answer"]) and str(row["parent_answer"]).strip()
                    else ""
                )
                st.caption(f"Follow-up on: *{row['parent_question']}*{parent_a_str}")

            st.markdown(f"**Answer:** {ans}")
            st.markdown(f"**LLM verdict:** {verdict_badge(row['verdict'])}")
            if row["reason"]:
                st.caption(f"ℹ️ {row['reason']}")

            # ── Nigel rating controls ──────────────────────────────────────────
            if is_nigel:
                st.divider()
                saved = st.session_state.get("saved_ratings", {}).get(qid)
                if saved:
                    label_map = {"agree": "✅ Agree", "disagree": "❌ Disagree", "unsure": "🟡 Unsure"}
                    saved_label = label_map.get(saved["rating"], saved["rating"])
                    msg = f"Saved: **{saved_label}**"
                    if saved["comment"]:
                        msg += f' — *"{saved["comment"]}"*'
                    st.success(msg)
                else:
                    rating = st.radio(
                        "Rate LLM decision:",
                        ["agree", "disagree", "unsure"],
                        key=f"rating_{qid}",
                        horizontal=True,
                    )
                    comment = st.text_area(
                        "Comment (optional):",
                        key=f"comment_{qid}",
                        height=68,
                    )
                    if st.button("Save rating", key=f"save_{qid}"):
                        supabase = get_supabase()
                        supabase.table("llm_reviews").insert({
                            "visit_id":         int(report["visit_id"]),
                            "qaas_question_id": qid,
                            "question_text":    str(row["description"]),
                            "parent_question":  str(row["parent_question"]) if pd.notna(row["parent_question"]) else None,
                            "parent_answer":    str(row["parent_answer"])   if pd.notna(row["parent_answer"])   else None,
                            "shopper_answer":   str(ans),
                            "llm_verdict":      str(row["verdict"]),
                            "llm_reason":       str(row["reason"]) if row["reason"] else None,
                            "nigel_rating":     rating,
                            "nigel_comment":    comment.strip() if comment.strip() else None,
                        }).execute()
                        if "saved_ratings" not in st.session_state:
                            st.session_state.saved_ratings = {}
                        st.session_state.saved_ratings[qid] = {
                            "rating": rating,
                            "comment": comment.strip(),
                        }
                        st.rerun()

# ── Photo analysis ────────────────────────────────────────────────────────────

def photo_verdict_icon(analysis: str) -> str:
    text = analysis.upper()
    if "CORRECT AFGEROND" in text or "PASSEND" in text:
        return "✅"
    elif "NIET CORRECT" in text or "NIET PASSEND" in text:
        return "❌"
    else:
        return "⚠️"

if "report" in st.session_state and analyse_photos:
    with st.spinner("Analysing photos… (this may take a minute)"):
        try:
            questions_d, assignments_d, answers_d, visits_d = get_photo_data()
            photo_result = analyse_visit_photos(
                st.session_state.report["visit_id"],
                questions_d, assignments_d, answers_d, visits_d
            )
            st.session_state.photo_result = photo_result
        except Exception as e:
            st.error(f"Photo analysis failed: {e}")

st.divider()
st.subheader("📸 Photo analysis")

pr = st.session_state.get("photo_result")

if pr is None:
    st.info("Click **Analyse Photos** in the sidebar to run photo analysis.")
elif "error" in pr:
    st.warning("No matching assignment found for this visit.")
else:
    st.caption(f"Assignment: **{pr['assignment']}** — {pr['products']}")

    if pr["receipts"]:
        st.markdown("**Receipt**")
        for r in pr["receipts"]:
            icon = photo_verdict_icon(r["analysis"])
            with st.expander(f"{icon} {r['file']}", expanded=True):
                col_img, col_text = st.columns([1, 2])
                with col_img:
                    if r.get("path"):
                        st.image(r["path"])
                with col_text:
                    st.text(r["analysis"])
    else:
        st.info("No receipt photo found for this visit.")

    if pr["questionnaire_photos"]:
        st.markdown("**Meal / questionnaire photos**")
        for r in pr["questionnaire_photos"]:
            icon = photo_verdict_icon(r["analysis"])
            with st.expander(f"{icon} {r['file']}", expanded=False):
                col_img, col_text = st.columns([1, 2])
                with col_img:
                    if r.get("path"):
                        st.image(r["path"])
                with col_text:
                    st.text(r["analysis"])
