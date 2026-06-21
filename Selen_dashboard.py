"""
Selen_dashboard.py — Compare Agent1, Reviewer, and Prosecutor on the Selen test set.

Usage:
    streamlit run Selen_dashboard.py
"""
import json, os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Selen Dashboard", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────

FILES = {
    "Agent1":     "agent1_Selen.json",
    "Reviewer":   "multi_agent_Selen.json",
    "Prosecutor": "selen_prosecutor.json",
}
COLORS = {"Agent1": "#4C72B0", "Reviewer": "#DD8452", "Prosecutor": "#55A868"}
CACHE_PREFIXES = {"Agent1": "a1", "Reviewer": "ma", "Prosecutor": "pros"}

@st.cache_data
def load_all():
    results = {}
    for name, path in FILES.items():
        if not os.path.exists(path):
            continue
        data  = json.load(open(path))
        label = list(data["evaluators"].keys())[0]
        ev    = data["evaluators"][label]
        results[name] = {
            "metrics": ev["metrics"],
            "rows":    pd.DataFrame(ev["rows"]),
        }
    cache = json.load(open("cache_selen.json")) if os.path.exists("cache_selen.json") else {}
    return results, cache

results, cache = load_all()
NAMES = list(results.keys())

if not results:
    st.error("No result files found. Run Selen_flow.py first.")
    st.stop()

# ── Title ─────────────────────────────────────────────────────────────────────

st.title("Selen — Multi-Agent Evaluation Dashboard")
st.caption("Agent1 vs Reviewer vs Prosecutor · GPT-4o-mini · test set n=167")

# ── 1. Key metrics ────────────────────────────────────────────────────────────

st.header("Key Metrics")
metric_cols = st.columns(len(NAMES))
for col, name in zip(metric_cols, NAMES):
    m = results[name]["metrics"]
    with col:
        st.markdown(f"#### {name}")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy",      f"{m['accuracy']*100:.1f}%")
        c2.metric("Bad Recall",    f"{m['bad_recall']*100:.1f}%")
        c1.metric("Bad Precision", f"{m['bad_precision']*100:.1f}%")
        c2.metric("Good F1",       f"{m['f1']*100:.1f}%")
        st.markdown(
            f"<div style='font-size:13px;color:#666;margin-top:4px'>"
            f"TP={m['tp']} &nbsp; TN={m['tn']} &nbsp; FP={m['fp']} &nbsp; FN={m['fn']}</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── 2. Bar chart ──────────────────────────────────────────────────────────────

st.header("Metrics Comparison")

METRIC_LABELS = {
    "Accuracy":       "accuracy",
    "Bad Recall":     "bad_recall",
    "Bad Precision":  "bad_precision",
    "Good Recall":    "recall",
    "Good Precision": "precision",
    "Good F1":        "f1",
}

fig_bar = go.Figure()
for name in NAMES:
    m = results[name]["metrics"]
    vals = [m[v] * 100 for v in METRIC_LABELS.values()]
    fig_bar.add_trace(go.Bar(
        name=name,
        x=list(METRIC_LABELS.keys()),
        y=vals,
        marker_color=COLORS[name],
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=11),
    ))

fig_bar.update_layout(
    barmode="group",
    yaxis=dict(title="Score (%)", range=[0, 110]),
    height=430,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=20, b=10),
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── 3. Confusion matrices ─────────────────────────────────────────────────────

st.header("Confusion Matrices")
cm_cols = st.columns(len(NAMES))
for col, name in zip(cm_cols, NAMES):
    m = results[name]["metrics"]
    z     = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    annot = [[f"TN={m['tn']}", f"FP={m['fp']}"], [f"FN={m['fn']}", f"TP={m['tp']}"]]
    fig_cm = go.Figure(go.Heatmap(
        z=z,
        x=["Pred: bad", "Pred: good"],
        y=["True: bad", "True: good"],
        text=annot,
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale=[[0, "#f0f4ff"], [1, COLORS[name]]],
        showscale=False,
    ))
    fig_cm.update_layout(
        title=dict(text=name, x=0.5),
        height=280,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    col.plotly_chart(fig_cm, use_container_width=True)

st.divider()

# ── 6. Per-row explorer ───────────────────────────────────────────────────────

st.header("Per-row Prediction Explorer")

base = results[NAMES[0]]["rows"][["id", "true", "predicted", "correct"]].rename(
    columns={"predicted": f"pred_{NAMES[0]}", "correct": f"correct_{NAMES[0]}"}
)
merged = base.copy()
for name in NAMES[1:]:
    r = results[name]["rows"][["id", "predicted", "correct"]].rename(
        columns={"predicted": f"pred_{name}", "correct": f"correct_{name}"}
    )
    merged = merged.merge(r, on="id", how="outer")

for name in NAMES:
    prefix = CACHE_PREFIXES[name]
    merged[f"reasoning_{name}"] = merged["id"].apply(
        lambda rid, p=prefix: cache.get(f"{p}_{int(rid)}", {}).get("raw", "")
    )

f1, f2, f3 = st.columns(3)
with f1:
    true_filter = st.selectbox("True verdict", ["All", "good", "bad"])
with f2:
    disagree_only = st.checkbox("Disagreements only", value=True)
with f3:
    show_reasoning = st.checkbox("Show reasoning", value=False)

view = merged.copy()
if true_filter != "All":
    view = view[view["true"] == true_filter]
if disagree_only:
    pred_cols = [f"pred_{n}" for n in NAMES if f"pred_{n}" in view.columns]
    if len(pred_cols) >= 2:
        view = view[view[pred_cols].nunique(axis=1) > 1]

display_cols = ["id", "true"] + [f"pred_{n}" for n in NAMES if f"pred_{n}" in view.columns]
if show_reasoning:
    display_cols += [f"reasoning_{n}" for n in NAMES if f"reasoning_{n}" in view.columns]

pred_style_cols = [c for c in display_cols if c.startswith("pred_")]

def color_pred(val):
    if val == "good":
        return "background-color: #d4edda; color: #155724"
    if val == "bad":
        return "background-color: #f8d7da; color: #721c24"
    return ""

styled = (
    view[display_cols]
    .reset_index(drop=True)
    .style.map(color_pred, subset=pred_style_cols)
)
st.dataframe(styled, use_container_width=True, height=420)
st.caption(f"Showing {len(view)} of {len(merged)} rows")
