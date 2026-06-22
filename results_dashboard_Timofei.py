"""
results_dashboard_Timofei.py — RAG vs Agent1 evaluation dashboard.

No API keys needed — reads only from local result JSON files.

Usage:
    streamlit run results_dashboard_Timofei.py
"""
import json, os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import binomtest

st.set_page_config(page_title="RAG Evaluation", layout="wide")

# ── Load data ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_all():
    base = os.path.dirname(__file__)
    files = {
        "groq_k5":   "results_Timofei_groq.json",
        "groq_k10":  "results_Timofei_groq_k10.json",
        "gpt_k5":    "results_Timofei_gpt_mini_k5.json",
        "gpt_k10":   "results_Timofei_gpt_mini_k10.json",
    }
    return {k: json.load(open(os.path.join(base, v)))
            for k, v in files.items() if os.path.exists(os.path.join(base, v))}

def get_metrics(data, label_fragment):
    for label, ev in data["evaluators"].items():
        if label_fragment in label:
            m  = ev["metrics"]
            tn, fp, fn = m["tn"], m["fp"], m["fn"]
            return {
                "Accuracy":      round(m["accuracy"]   * 100, 1),
                "Bad Recall":    round(m.get("bad_recall",  tn/(tn+fp) if (tn+fp) else 0) * 100, 1),
                "Bad Precision": round(m.get("bad_precision", tn/(tn+fn) if (tn+fn) else 0) * 100, 1),
                "Good Recall":   round(m["recall"]     * 100, 1),
                "Good F1":       round(m["f1"]         * 100, 1),
                "TP": m["tp"], "TN": tn, "FP": fp, "FN": fn, "n": m["n"],
                "_rows": pd.DataFrame(ev["rows"]) if "rows" in ev else None,
            }
    return None

def mcnemar(df_a, df_b, bad_only=False):
    m = df_a.merge(df_b, on="id", suffixes=("_a","_b"))
    if bad_only:
        m = m[m["true_a"] == "bad"]
    n01   = int(( m["correct_a"] & ~m["correct_b"]).sum())
    n10   = int((~m["correct_a"] &  m["correct_b"]).sum())
    total = n01 + n10
    p = binomtest(n10, total, 0.5, alternative="two-sided").pvalue if total else 1.0
    return float(p), n01, n10, len(m)

all_data = load_all()
if not all_data:
    st.error("No result files found.")
    st.stop()

# ── Collect all metrics ────────────────────────────────────────────────────────

groq_a1_k10  = get_metrics(all_data["groq_k10"], "Agent1")
groq_rag_k10 = get_metrics(all_data["groq_k10"], "RAG")
groq_a1_k5   = get_metrics(all_data["groq_k5"],  "Agent1")
groq_rag_k5  = get_metrics(all_data["groq_k5"],  "RAG")
gpt_a1_k10   = get_metrics(all_data["gpt_k10"],  "Agent1")
gpt_rag_k10  = get_metrics(all_data["gpt_k10"],  "RAG")
gpt_a1_k5    = get_metrics(all_data["gpt_k5"],   "Agent1")
gpt_rag_k5   = get_metrics(all_data["gpt_k5"],   "RAG")

# McNemar — bad recall focus (k=10 is the main comparison)
p_groq_bad, n01_groq, n10_groq, n_groq = mcnemar(
    groq_a1_k10["_rows"], groq_rag_k10["_rows"], bad_only=True)
p_gpt_bad, n01_gpt, n10_gpt, n_gpt = mcnemar(
    gpt_a1_k10["_rows"], gpt_rag_k10["_rows"], bad_only=True)
p_groq_all, *_ = mcnemar(groq_a1_k10["_rows"], groq_rag_k10["_rows"], bad_only=False)
p_gpt_all,  *_ = mcnemar(gpt_a1_k10["_rows"],  gpt_rag_k10["_rows"],  bad_only=False)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("RAG vs Agent1 — Evaluation Results")
st.caption("GPT-4o-mini and Groq llama-3.1-8b-instant · Test set: 167 answers · Ground truth: Nigel-corrected")

# ── Main finding ───────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:#1e3a5f;border-radius:10px;padding:18px 24px;margin:12px 0 24px 0">
<span style="font-size:18px;font-weight:700;color:#ffffff">
Key finding: RAG improvement is model-dependent
</span><br>
<span style="color:#bfdbfe;font-size:14px">
RAG significantly improves bad-answer recall for <b style="color:#fff">GPT-4o-mini</b>
(64.2% → 81.1%, p = 0.012) but shows <b style="color:#fff">no significant improvement</b>
for <b style="color:#fff">Groq llama-3.1-8b</b> (41.5% → 47.2%, p = 0.629).
</span>
</div>
""", unsafe_allow_html=True)

# ── Model selector ─────────────────────────────────────────────────────────────

model_choice = st.radio("Select model to inspect:", ["Both", "GPT-mini", "Groq"],
                        horizontal=True)

# ── Helper: render one model panel ────────────────────────────────────────────

def render_model(name, color, a1_k5, a1_k10, rag_k5, rag_k10, p_all, p_bad,
                 n01_bad, n10_bad, n_bad):
    st.markdown(f"### {name}")

    # Metrics table
    table = pd.DataFrame([
        {"Approach": "Agent1 (baseline)", "k": "—",
         "Accuracy": f"{a1_k10['Accuracy']}%",
         "Bad Recall ★": f"{a1_k10['Bad Recall']}%",
         "Bad Precision": f"{a1_k10['Bad Precision']}%",
         "Good F1": f"{a1_k10['Good F1']}%",
         "TP": a1_k10["TP"], "TN": a1_k10["TN"],
         "FP": a1_k10["FP"], "FN": a1_k10["FN"]},
        {"Approach": "RAG", "k": "5",
         "Accuracy": f"{rag_k5['Accuracy']}%",
         "Bad Recall ★": f"{rag_k5['Bad Recall']}%",
         "Bad Precision": f"{rag_k5['Bad Precision']}%",
         "Good F1": f"{rag_k5['Good F1']}%",
         "TP": rag_k5["TP"], "TN": rag_k5["TN"],
         "FP": rag_k5["FP"], "FN": rag_k5["FN"]},
        {"Approach": "RAG", "k": "10",
         "Accuracy": f"{rag_k10['Accuracy']}%",
         "Bad Recall ★": f"{rag_k10['Bad Recall']}%",
         "Bad Precision": f"{rag_k10['Bad Precision']}%",
         "Good F1": f"{rag_k10['Good F1']}%",
         "TP": rag_k10["TP"], "TN": rag_k10["TN"],
         "FP": rag_k10["FP"], "FN": rag_k10["FN"]},
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    # Bar charts — Bad Recall and Bad Precision side by side
    labels    = ["Agent1 (baseline)", "RAG k=5", "RAG k=10"]
    rc_values = [a1_k10["Bad Recall"],    rag_k5["Bad Recall"],    rag_k10["Bad Recall"]]
    pr_values = [a1_k10["Bad Precision"], rag_k5["Bad Precision"], rag_k10["Bad Precision"]]
    r, g, b   = _hex_to_rgb(color)
    bar_colors = [
        "#374151",                        # Agent1: dark charcoal, clearly readable
        f"rgba({r},{g},{b},0.60)",        # RAG k=5: visible tint of model colour
        color,                             # RAG k=10: full model colour
    ]

    def make_bar(values, title, subtitle):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=bar_colors,
            marker_line=dict(color="#e5e7eb", width=1),
            text=[f"{v}%" for v in values],
            textposition="outside",
            textfont=dict(size=13, color="#111827"),
        ))
        fig.update_layout(
            title=dict(text=f"{title}<br><sup>{subtitle}</sup>",
                       font=dict(size=13, color="#111827")),
            yaxis=dict(range=[0, 108], title="%", gridcolor="#e5e7eb",
                       tickfont=dict(color="#111827"),
                       title_font=dict(color="#111827")),
            height=300, margin=dict(t=60, b=10, l=10, r=10),
            plot_bgcolor="#f8fafc", paper_bgcolor="white",
            xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#111827")),
            font=dict(color="#111827"),
        )
        return fig

    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(
            make_bar(rc_values,
                     "Bad Recall (main metric)",
                     "How many truly bad answers were caught?"),
            use_container_width=True,
        )
    with ch2:
        st.plotly_chart(
            make_bar(pr_values,
                     "Bad Precision",
                     "Of all predicted bad — how many were truly bad?"),
            use_container_width=True,
        )

    # Note about Groq RAG k=5 precision collapse
    if "Groq" in name:
        st.info(
            "⚠️ Groq RAG k=5 shows higher bad recall (69.8%) than Agent1 (41.5%), "
            "but bad precision collapses to 46.8% — meaning over half of its "
            "'bad' predictions are actually good answers. "
            "The recall gain is offset by a large increase in false positives, "
            "so this result does not represent a genuine improvement."
        )

    # McNemar result
    sig_bad = p_bad < 0.05
    sig_all = p_all < 0.05
    bg_bad  = "#dcfce7" if sig_bad else "#fef2f2"
    txt_bad = "#166534" if sig_bad else "#991b1b"
    bg_all  = "#dcfce7" if sig_all else "#fef2f2"
    txt_all = "#166534" if sig_all else "#991b1b"
    icon_b  = "✓" if sig_bad else "✗"
    icon_a  = "✓" if sig_all else "✗"
    st.markdown(f"""
<div style="display:flex;gap:12px;margin-top:4px">
  <div style="flex:1;background:{bg_bad};border-radius:8px;padding:14px 16px">
    <div style="font-size:12px;color:{txt_bad};font-weight:600;text-transform:uppercase">
      McNemar — bad rows only (n={n_bad})</div>
    <div style="font-size:20px;font-weight:800;color:{txt_bad};margin:4px 0">
      {icon_b}  p = {p_bad:.4f}</div>
    <div style="font-size:12px;color:{txt_bad}">
      {'Significant improvement in bad recall' if sig_bad else 'No significant improvement in bad recall'}
      &nbsp;·&nbsp; n01={n01_bad} &nbsp; n10={n10_bad}</div>
  </div>
  <div style="flex:1;background:{bg_all};border-radius:8px;padding:14px 16px">
    <div style="font-size:12px;color:{txt_all};font-weight:600;text-transform:uppercase">
      McNemar — all rows (n=167)</div>
    <div style="font-size:20px;font-weight:800;color:{txt_all};margin:4px 0">
      {icon_a}  p = {p_all:.4f}</div>
    <div style="font-size:12px;color:{txt_all}">
      {'Significant overall improvement' if sig_all else 'No significant overall improvement'}</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ── Render panels ──────────────────────────────────────────────────────────────

if model_choice == "Both":
    col_gpt, col_groq = st.columns(2)
    with col_gpt:
        render_model("GPT-4o-mini", "#10b981",
                     gpt_a1_k5, gpt_a1_k10, gpt_rag_k5, gpt_rag_k10,
                     p_gpt_all, p_gpt_bad, n01_gpt, n10_gpt, n_gpt)
    with col_groq:
        render_model("Groq llama-3.1-8b", "#3b82f6",
                     groq_a1_k5, groq_a1_k10, groq_rag_k5, groq_rag_k10,
                     p_groq_all, p_groq_bad, n01_groq, n10_groq, n_groq)

elif model_choice == "GPT-mini":
    render_model("GPT-4o-mini", "#10b981",
                 gpt_a1_k5, gpt_a1_k10, gpt_rag_k5, gpt_rag_k10,
                 p_gpt_all, p_gpt_bad, n01_gpt, n10_gpt, n_gpt)

else:
    render_model("Groq llama-3.1-8b", "#3b82f6",
                 groq_a1_k5, groq_a1_k10, groq_rag_k5, groq_rag_k10,
                 p_groq_all, p_groq_bad, n01_groq, n10_groq, n_groq)

# ── Data note ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Data: 334 mystery shopper answers from Supabase llm_reviews table · "
    "Nigel agreement: 264/334 (79.0%) · "
    "Fixed train/test split (167/167) via final_split.json · "
    "McNemar test compares Agent1 vs RAG k=10 on the same test rows · "
    "★ Bad Recall = fraction of truly bad answers correctly identified"
)
