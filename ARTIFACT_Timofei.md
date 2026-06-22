# Thesis Artifact — RAG for Mystery Shopper Answer Evaluation
**Author:** Timofei Drankin

This artifact accompanies the thesis section on evaluating Retrieval-Augmented Generation (RAG) for automated quality assessment of mystery shopper answers at WRAPP.

---

## 1. Streamlit App — Nigel's Review Interface (`streamlit_app.py`)

A Streamlit web application used by Nigel (WRAPP's owner) to review and rate LLM-generated verdicts on mystery shopper answers.

**How it works:**
- Nigel logs in by entering his name. Other users get a standard view without rating controls.
- The app loads visit data from CSV files (`answer_rows.csv`, `qaas_questions_rows.csv`, `visit_rows.csv`) and processes it using the GPT-4o-mini evaluator (`agent1gpt.py`).
- For each visit, the LLM evaluates each shopper answer as **good** or **bad** based on the WRAPP guidelines (detailed in `Agent1.py`).
- In Nigel mode, each verdict is shown with an **agree / disagree** radio button. Selections are saved to the Supabase `llm_reviews` table in real time.
- The app also displays receipt and questionnaire photos per visit.

**To run:**
```
streamlit run streamlit_app.py
```
Requires Streamlit secrets: `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`.

---

## 2. SQL Table — `llm_reviews` (`sql_scores.py`)

The `llm_reviews` table in Supabase stores every LLM verdict that Nigel has reviewed. Each row represents one shopper answer.

**Key columns:**

| Column | Description |
|---|---|
| `id` | Auto-increment primary key |
| `visit_id` | The mystery shop visit this answer belongs to |
| `qaas_question_id` | The question ID |
| `question_text` | The question asked to the shopper |
| `shopper_answer` | The shopper's text answer |
| `llm_verdict` | LLM's verdict: `"good"` or `"bad"` |
| `llm_reason` | LLM's one-sentence reasoning |
| `nigel_rating` | Nigel's review: `"agree"` or `"disagree"` |
| `nigel_comment` | Optional correction note from Nigel |
| `reviewed_at` | Timestamp of Nigel's rating |

**Ground-truth construction:** Where Nigel disagreed (`nigel_rating = "disagree"`), the true verdict is the **opposite** of `llm_verdict`. This corrected label is the evaluation target.

**Nigel agreement rate:** 264 / 334 = **79.0%** agreement with the LLM.

**To retrieve and export scores:**
```
python3 sql_scores.py
```
Fill in `SUPABASE_URL` and `SUPABASE_KEY` at the top of the file.
Outputs `llm_nigel_scores.csv`.

---

## 3. RAG Evaluation Flow (`eval_artifact.py`)

Evaluates the WRAPP answer quality classifier using two approaches on a fixed held-out test set of 167 answers:

- **Agent1:** Single LLM call with the WRAPP system prompt (baseline).
- **RAG:** Same prompt augmented with *k* labelled examples retrieved from the training set using TF-IDF cosine similarity.

**Data split:** `final_split.json` defines a fixed 167/167 train/test split across 334 total reviewed answers. The split is held constant across all experiments.

**Retrieval:** A TF-IDF index (unigrams + bigrams, sublinear TF) is built over the training set. At inference, the *k* most similar training examples are prepended to the system prompt as few-shot demonstrations with their Nigel-corrected verdicts.

**Supported backends:**

| BACKEND | Model | Notes |
|---|---|---|
| `"groq"` | llama-3.1-8b-instant | Free via Groq API |
| `"gpt-mini"` | gpt-4o-mini | Via OpenAI or UvA LiteLLM proxy |

**To run:**

1. Open `eval_artifact.py` and set `BACKEND`, `K`, and API keys at the top.
2. Run:
   ```
   python3 eval_artifact.py
   ```
3. Results are saved to `results_artifact_{backend}_k{k}.json`.

The script is **resume-safe**: if interrupted, re-running picks up from the cache file. API errors are not cached and will be retried.

**Evaluated configurations (pre-computed results included):**

| File | Backend | k |
|---|---|---|
| `results_Timofei_groq.json` | Groq | 5 |
| `results_Timofei_groq_k10.json` | Groq | 10 |
| `results_Timofei_gpt_mini_k5.json` | GPT-mini | 5 |
| `results_Timofei_gpt_mini_k10.json` | GPT-mini | 10 |

Missing rows (API errors during collection) are imputed as correct predictions, consistent with the analysis in the thesis.

---

## 4. Results Dashboard (`results_dashboard.py`)

A Streamlit dashboard that reads the pre-computed result JSON files and visualises:

- Full metrics table (Accuracy, Bad Recall, Bad Precision, Good F1, TP/TN/FP/FN) with best-value highlighting
- Horizontal bar charts and grouped comparison charts
- Confusion matrices for every configuration
- McNemar statistical tests (pairwise, all-rows and bad-rows-only)

No API keys required — the dashboard reads only local JSON files.

**To run:**
```
streamlit run results_dashboard.py
```

---

## Files for Submission

| File | Purpose |
|---|---|
| `streamlit_app.py` | Nigel's review interface (Streamlit app) |
| `agent1gpt.py` | LLM evaluator class + WRAPP system prompt |
| `Agent1.py` | WRAPP system prompt and evaluation logic |
| `eval_artifact.py` | RAG vs Agent1 evaluation script (configurable) |
| `sql_scores.py` | Retrieve and export LLM + Nigel scores from Supabase |
| `results_dashboard.py` | Results visualisation dashboard (Streamlit) |
| `final_split.json` | Fixed 167/167 train/test split |
| `results_Timofei_groq.json` | Groq results, k=5 |
| `results_Timofei_groq_k10.json` | Groq results, k=10 (with row-level data) |
| `results_Timofei_gpt_mini_k5.json` | GPT-mini results, k=5 (with row-level data) |
| `results_Timofei_gpt_mini_k10.json` | GPT-mini results, k=10 (with row-level data) |
| `ARTIFACT.md` | This documentation file |

**Dependencies:** `openai`, `groq`, `supabase`, `scikit-learn`, `pandas`, `numpy`, `streamlit`, `plotly`, `scipy`
