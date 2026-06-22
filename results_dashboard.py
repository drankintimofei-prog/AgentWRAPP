"""
results_dashboard.py — Thesis evaluation dashboard: RAG vs Agent1.

Loads all available result JSON files and displays:
  - Full metrics table (all configurations)
  - Comparative bar charts
  - Confusion matrices
  - McNemar statistical tests (for configurations with row-level data)

No API keys needed — reads only from local JSON files.

Usage:
    streamlit run results_dashboard.py
"""
import json, os, glob
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import binomtest

st.set_page_config(page_title="RAG Evaluation — Thesis", layout="wide",
                   initial_sidebar_state="expanded")

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a1f2e; }
    [data-testid="stSidebar"] * { color: #e0e6f0 !important; }
    .metric-card {
        background: #f7f9fc; border: 1px solid #e0e6ef;
        border-radius: 8px; padding: 16px 20px; margin-bottom: 8px;
    }
    .metric-card h4 { margin: 0 0 8px 0; color: #1a1f2e; font-size: 15px; }
    .big-num { font-size: 28px; font-weight: 700; color: #2563eb; }
    .sub-num { font-size: 13px; color: #64748b; margin-top: 2px; }
    h1 { color: #1a1f2e !important; }
    h2 { color: #1e3a5f !important; border-bottom: 2px solid #e0e6ef; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────

RESULT_FILES = {
    "Groq Agent1 k=5":        "results_Timofei_groq.json",
    "Groq RAG k=5":           "results_Timofei_groq.json",
    "Groq Agent1 k=10":       "results_Timofei_groq_k10.json",
    "Groq RAG k=10":          "results_Timofei_groq_k10.json",
    "GPT-mini Agent1 k=5":    "results_Timofei_gpt_mini_k5.json",
    "GPT-mini RAG k=5":       "results_Timofei_gpt_mini_k5.json",
    "GPT-mini Agent1 k=10":   "results_Timofei_gpt_mini_k10.json",
    "GPT-mini RAG k=10":      "results_Timofei_gpt_mini_k10.json",
}

@st.cache_data
def load_results():
    loaded = {}
    for path in set(RESULT_FILES.values()):
        full = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full):
            loaded[path] = json.load(open(full))
    return loaded

@st.cache_data
def build_table(loaded: dict) -> pd.DataFrame:
    rows = []
    for display_name, path in RESULT_FILES.items():
        if path not in loaded:
            continue
        data = loaded[path]
        # Find matching evaluator label
        for label, ev in data["evaluators"].items():
            is_rag    = "RAG" in label
            is_agent1 = "Agent1" in label
            if ("RAG" in display_name and is_rag) or ("Agent1" in display_name and is_agent1):
                m = ev["metrics"]
                rows.append({
                    "Configuration":   display_name,
                    "Model":           "GPT-mini" if "GPT" in display_name else "Groq",
                    "Type":            "RAG" if "RAG" in display_name else "Agent1",
                    "k":               int(display_name.split("k=")[1]),
                    "n":               m["n"],
                    "Accuracy":        round(m["accuracy"] * 100, 1),
                    "Good Recall":     round(m["recall"] * 100, 1),
                    "Good Precision":  round(m["precision"] * 100, 1),
                    "Good F1":         round(m["f1"] * 100, 1),
                    "Bad Recall":      round(m["bad_recall"] * 100, 1),
                    "Bad Precision":   round(m["bad_precision"] * 100, 1),
                    "TP": m["tp"], "TN": m["tn"], "FP": m["fp"], "FN": m["fn"],
                    "_has_rows": "rows" in ev,
                    "_label":    label,
                    "_path":     path,
                })
                break
    return pd.DataFrame(rows)

loaded = load_results()
df_all = build_table(loaded)

if df_all.empty:
    st.error("No result files found. Run eval_artifact.py first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## RAG Evaluation")
    st.markdown("**Thesis artifact** — Timofei Drankin")
    st.markdown("---")
    st.markdown("**Filter configurations**")
    sel_model = st.multiselect("Model", ["Groq", "GPT-mini"],
                               default=["Groq", "GPT-mini"])
    sel_type  = st.multiselect("Type",  ["Agent1", "RAG"],
                               default=["Agent1", "RAG"])
    sel_k     = st.multiselect("k (RAG examples)", [5, 10], default=[5, 10])
    st.markdown("---")
    primary_metric = st.selectbox(
        "Primary metric for charts",
        ["Bad Recall", "Accuracy", "Good F1", "Bad Precision"],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px;color:#8899aa'>"
        "Data: Supabase llm_reviews<br>"
        "Split: final_split.json (167/167)<br>"
        "Ground truth: Nigel-corrected</div>",
        unsafe_allow_html=True,
    )

# Apply filters
df = df_all[
    df_all["Model"].isin(sel_model) &
    df_all["Type"].isin(sel_type)   &
    df_all["k"].isin(sel_k)
].copy()

# ── Title ─────────────────────────────────────────────────────────────────────

st.title("RAG vs Agent1 — Evaluation Results")
st.caption(
    "Mystery shopper answer quality assessment · GPT-4o-mini and Groq llama-3.1-8b-instant · "
    "Test set: 167 answers (Nigel-corrected ground truth)"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Results Table", "📈 Charts", "🔲 Confusion Matrices", "🧪 McNemar Tests"]
)

# ────────────────────────────────────────────────────────────────────────────
# Tab 1 — Full metrics table
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("All Metrics")

    display_cols = ["Configuration", "n", "Accuracy", "Bad Recall", "Bad Precision",
                    "Good Recall", "Good Precision", "Good F1", "TP", "TN", "FP", "FN"]

    def highlight_best(s):
        if s.name not in ("Accuracy", "Bad Recall", "Bad Precision",
                          "Good Recall", "Good Precision", "Good F1"):
            return [""] * len(s)
        best = s.max()
        return ["background-color: #dbeafe; font-weight: 700"
                if v == best else "" for v in s]

    styled = (
        df[display_cols]
        .reset_index(drop=True)
        .style
        .apply(highlight_best)
        .format({c: "{:.1f}%" for c in ["Accuracy", "Bad Recall", "Bad Precision",
                                         "Good Recall", "Good Precision", "Good F1"]})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Blue highlight = best value in each column")

    st.subheader("Nigel Agreement Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total reviewed answers", "334")
    c2.metric("Nigel agreed with LLM", "264  (79.0%)")
    c3.metric("Nigel disagreed (flipped)", "70  (21.0%)")

# ────────────────────────────────────────────────────────────────────────────
# Tab 2 — Charts
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"Comparison by {primary_metric}")

    COLOR_MAP = {
        ("Groq",    "Agent1"): "#3b82f6",
        ("Groq",    "RAG"):    "#1d4ed8",
        ("GPT-mini","Agent1"): "#10b981",
        ("GPT-mini","RAG"):    "#065f46",
    }
    df["Color"] = df.apply(
        lambda r: COLOR_MAP.get((r["Model"], r["Type"]), "#94a3b8"), axis=1
    )
    df["Label"] = df["Model"] + " " + df["Type"] + " k=" + df["k"].astype(str)

    fig = px.bar(
        df.sort_values(primary_metric, ascending=True),
        x=primary_metric, y="Label",
        orientation="h",
        color="Label",
        color_discrete_sequence=df.sort_values(primary_metric, ascending=True)["Color"].tolist(),
        text=primary_metric,
        template="plotly_white",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title=f"{primary_metric} (%)", range=[0, 105]),
        yaxis_title="",
        height=max(300, len(df) * 50 + 80),
        margin=dict(l=10, r=60, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All Metrics Side-by-Side")
    metrics_to_plot = ["Accuracy", "Bad Recall", "Bad Precision", "Good F1"]
    df_melt = df.melt(
        id_vars=["Label", "Color"],
        value_vars=metrics_to_plot,
        var_name="Metric", value_name="Score",
    )
    fig2 = px.bar(
        df_melt, x="Metric", y="Score", color="Label",
        barmode="group",
        text="Score",
        template="plotly_white",
        color_discrete_map={row["Label"]: row["Color"] for _, row in df.iterrows()},
    )
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(
        yaxis=dict(title="Score (%)", range=[0, 108]),
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# Tab 3 — Confusion matrices
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Confusion Matrices")

    n_cols = min(len(df), 4)
    for start in range(0, len(df), n_cols):
        chunk = df.iloc[start:start + n_cols]
        cols  = st.columns(len(chunk))
        for col, (_, row) in zip(cols, chunk.iterrows()):
            z = [[row["TN"], row["FP"]], [row["FN"], row["TP"]]]
            ann = [[f"TN\n{row['TN']}", f"FP\n{row['FP']}"],
                   [f"FN\n{row['FN']}", f"TP\n{row['TP']}"]]
            fig_cm = go.Figure(go.Heatmap(
                z=z, x=["Pred: bad", "Pred: good"], y=["True: bad", "True: good"],
                text=ann, texttemplate="%{text}",
                textfont=dict(size=13),
                colorscale=[[0, "#f0f7ff"], [1, row["Color"]]],
                showscale=False,
            ))
            fig_cm.update_layout(
                title=dict(text=row["Label"], x=0.5, font=dict(size=12)),
                height=240, margin=dict(t=40, b=5, l=5, r=5),
            )
            col.plotly_chart(fig_cm, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# Tab 4 — McNemar tests
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("McNemar Test — Paired Comparison of Classifiers")
    st.markdown(
        "Tests whether two classifiers differ significantly on the **same** test set. "
        "Uses the exact binomial test on discordant pairs (n01, n10). "
        "Only configurations with row-level data can be tested."
    )

    # Build row-level lookup: display_name → rows DataFrame
    row_data = {}
    for _, r in df_all[df_all["_has_rows"]].iterrows():
        data   = loaded[r["_path"]]
        ev_rows = data["evaluators"][r["_label"]]["rows"]
        row_data[r["Configuration"]] = pd.DataFrame(ev_rows)

    available = list(row_data.keys())

    if len(available) < 2:
        st.info("Need at least two configurations with row-level data to run McNemar tests.")
    else:
        col_a, col_b, col_mode = st.columns(3)
        with col_a:
            cfg_a = st.selectbox("Configuration A", available, index=0)
        with col_b:
            cfg_b = st.selectbox("Configuration B", available,
                                 index=min(1, len(available) - 1))
        with col_mode:
            test_mode = st.radio("Subset", ["All rows", "Bad rows only"],
                                 help="'Bad rows only' isolates bad-recall performance")

        if cfg_a == cfg_b:
            st.warning("Select two different configurations.")
        else:
            df_a = row_data[cfg_a]
            df_b = row_data[cfg_b]
            m    = df_a.merge(df_b, on="id", suffixes=("_a", "_b"))
            if test_mode == "Bad rows only":
                m = m[m["true_a"] == "bad"]

            n01   = int(( m["correct_a"] & ~m["correct_b"]).sum())
            n10   = int((~m["correct_a"] &  m["correct_b"]).sum())
            total = n01 + n10
            p     = binomtest(n10, total, 0.5, alternative="two-sided").pvalue if total else 1.0

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("n (subset)", len(m))
            r2.metric(f"A✓ B✗  (n01 — favours {cfg_a})", n01)
            r3.metric(f"A✗ B✓  (n10 — favours {cfg_b})", n10)
            r4.metric("Discordant pairs", total)

            sig = p < 0.05
            color = "#166534" if sig else "#7f1d1d"
            bg    = "#dcfce7" if sig else "#fee2e2"
            st.markdown(
                f"<div style='background:{bg};border-radius:8px;padding:16px 20px;"
                f"margin-top:12px'>"
                f"<span style='font-size:20px;font-weight:700;color:{color}'>"
                f"p = {p:.4f} — {'Statistically significant (p < 0.05) ✓' if sig else 'Not significant (p ≥ 0.05) ✗'}"
                f"</span><br>"
                f"<span style='color:{color};font-size:13px'>"
                f"{'The two classifiers differ significantly on this test set.' if sig else 'No significant difference detected between the two classifiers.'}"
                f"</span></div>",
                unsafe_allow_html=True,
            )

        st.subheader("All Pairwise Tests")
        tab_all, tab_bad = st.tabs(["All rows", "Bad rows only"])

        def run_all_pairs(bad_only: bool):
            results = []
            names = available
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a_name, b_name = names[i], names[j]
                    m = row_data[a_name].merge(row_data[b_name], on="id",
                                               suffixes=("_a", "_b"))
                    if bad_only:
                        m = m[m["true_a"] == "bad"]
                    n01   = int(( m["correct_a"] & ~m["correct_b"]).sum())
                    n10   = int((~m["correct_a"] &  m["correct_b"]).sum())
                    total = n01 + n10
                    p     = binomtest(n10, total, 0.5, alternative="two-sided").pvalue if total else 1.0
                    results.append({
                        "A":            a_name,
                        "B":            b_name,
                        "n":            len(m),
                        "n01 (A✓ B✗)": n01,
                        "n10 (A✗ B✓)": n10,
                        "Discordant":   total,
                        "p-value":      round(p, 4),
                        "Significant?": "✓  p<0.05" if p < 0.05 else "✗  n.s.",
                    })
            df_res = pd.DataFrame(results)

            def style_sig(val):
                return "color: #166534; font-weight:700" if "✓" in str(val) else "color: #9ca3af"

            st.dataframe(
                df_res.style.applymap(style_sig, subset=["Significant?"]),
                use_container_width=True, hide_index=True,
            )

        with tab_all:
            run_all_pairs(bad_only=False)
        with tab_bad:
            run_all_pairs(bad_only=True)
