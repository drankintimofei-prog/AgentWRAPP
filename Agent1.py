import pandas as pd
from abc import ABC, abstractmethod
from dotenv import load_dotenv
load_dotenv()

pd.set_option("display.max_colwidth", None)

GOOD_THRESHOLD = 0.70  # fraction of questions that must be good for visit to pass

# ── Evaluators ────────────────────────────────────────────────────────────────

class AnswerEvaluator(ABC):
    @abstractmethod
    def evaluate_text(self, question: str, answer: str) -> bool:
        """Return True if the answer is adequate for the question."""
        pass


class GroqEvaluator(AnswerEvaluator):
    """Free LLM via Groq (console.groq.com — no credit card needed).
    Get a free key at: https://console.groq.com/keys
    """

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model

    def evaluate_text(self, question: str, answer: str) -> bool:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Je beoordeelt of een antwoord adequaat is voor de gegeven vraag. "
                        "Een slecht antwoord is te kort, irrelevant, of geeft geen echte informatie. "
                        "Antwoord alleen met het woord 'goed' of 'slecht', niets anders."
                    ),
                },
                {"role": "user", "content": f"Vraag: {question}\nAntwoord: {answer}"},
            ],
            max_tokens=5,
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower() == "goed"


class HuggingFaceEvaluator(AnswerEvaluator):
    """Zero-shot classification via HuggingFace Inference API (free with token).
    Get a free token at: https://huggingface.co/settings/tokens
    """

    def __init__(self, hf_token: str):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=hf_token)
        self.model = "joeddav/xlm-roberta-large-xnli"

    def evaluate_text(self, question: str, answer: str) -> bool:
        text = f"Vraag: {question}\nAntwoord: {answer}"
        result = self.client.zero_shot_classification(
            text=text,
            candidate_labels=["passend antwoord", "onvoldoende antwoord"],
            model=self.model,
        )
        return result[0].label == "passend antwoord"


class OpenAIEvaluator(AnswerEvaluator):
    """GPT evaluator — swap in when you have your university API key."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def evaluate_text(self, question: str, answer: str) -> bool:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Je beoordeelt of een antwoord adequaat is voor de gegeven vraag. "
                        "Een slecht antwoord is te kort, irrelevant, of geeft geen echte informatie. "
                        "Antwoord alleen met het woord 'goed' of 'slecht', niets anders."
                    ),
                },
                {"role": "user", "content": f"Vraag: {question}\nAntwoord: {answer}"},
            ],
            max_tokens=5,
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower() == "goed"


# ── Per-question evaluation ───────────────────────────────────────────────────

RULE_BASED_TYPES = {"Getal", "Schaal", "Datum", "Checkbox", "Foto upload"}
FOLLOW_UP_LABELS = {
    "toelichting",
    "licht toe hoe dit verliep",
    "licht je score toe",
    "licht je antwoord toe",
}


def evaluate_answer(row: pd.Series, evaluator: AnswerEvaluator,
                    parent_question: str | None = None) -> str:
    q_type = row["question_type_label"]
    text = row["answer_text"]
    value = row["answer_value"]
    has_text = pd.notna(text) and str(text).strip() != ""
    has_value = pd.notna(value)

    # Structured types: just check something was provided
    if q_type in RULE_BASED_TYPES:
        return "good" if (has_text or has_value) else "bad"

    # Text type: quick length check first, then LLM
    if q_type == "Tekst":
        if not has_text or len(str(text).split()) < 2:
            return "bad"
        # For generic follow-up questions, prepend the parent question as context
        label = str(row["description"]).strip().lower().rstrip(".")
        if label in FOLLOW_UP_LABELS and parent_question:
            question = f'[toelichting op: "{parent_question}"]\n{row["description"]}'
        else:
            question = row["description"]
        return "good" if evaluator.evaluate_text(question, str(text)) else "bad"

    return "good"  # unknown type: benefit of the doubt


# ── Visit-level verdict ───────────────────────────────────────────────────────

def evaluate_visit(visit_id: int, df: pd.DataFrame, evaluator: AnswerEvaluator) -> dict:
    rows = df[df["visit_id"] == visit_id].copy()

    # Sort by section first, then by question order within each section
    rows = rows.sort_values(["section_id", "question_order"]).reset_index(drop=True)

    # Attach parent question text to every follow-up ("Toelichting") row,
    # walking back past any consecutive follow-ups to find the real question.
    descriptions = rows["description"].tolist()
    parent_questions = []
    for i, desc in enumerate(descriptions):
        if desc.strip().lower().rstrip(".") in FOLLOW_UP_LABELS and i > 0:
            parent = None
            for j in range(i - 1, -1, -1):
                if descriptions[j].strip().lower().rstrip(".") not in FOLLOW_UP_LABELS:
                    parent = descriptions[j]
                    break
            parent_questions.append(parent)
        else:
            parent_questions.append(None)
    rows["parent_question"] = parent_questions

    rows["verdict"] = rows.apply(
        lambda r: evaluate_answer(r, evaluator, r["parent_question"]), axis=1
    )

    total = len(rows)
    good = (rows["verdict"] == "good").sum()
    pct = good / total if total > 0 else 0

    return {
        "visit_id": visit_id,
        "total_questions": total,
        "good": int(good),
        "bad": int(total - good),
        "pct_good": round(pct * 100, 1),
        "verdict": "PASS" if pct >= GOOD_THRESHOLD else "FAIL",
        "details": rows[["qaas_question_id", "question_order", "description",
                          "parent_question", "question_type_label",
                          "answer_text", "answer_value", "verdict"]],
    }


def print_visit_report(report: dict) -> None:
    v = report
    print("=" * 80)
    print(f"Visit {v['visit_id']}  |  {v['good']}/{v['total_questions']} good "
          f"({v['pct_good']}%)  |  {v['verdict']}")
    print("=" * 80)
    for _, row in v["details"].iterrows():
        icon = "✓" if row["verdict"] == "good" else "✗"
        ans = row["answer_text"] if pd.notna(row["answer_text"]) else row["answer_value"]
        q = str(row["description"])[:72]
        context = f'  [re: "{str(row["parent_question"])[:60]}"]' if pd.notna(row["parent_question"]) else ""
        print(f"  {icon} Q{row['qaas_question_id']:>3} [{row['question_type_label']}]: {q}{context}")
        print(f"         -> {ans}")
    print()


# ── Accuracy evaluation ───────────────────────────────────────────────────────

def run_accuracy_test(df: pd.DataFrame, visits: pd.DataFrame,
                      evaluator: AnswerEvaluator, n: int = 20) -> None:
    import json, os

    # Only visits with a clear human verdict
    clear = visits[visits["visit_status"].isin(["Goedgekeurd", "Afgekeurd"])].copy()
    # Include all 4 Afgekeurd + fill up to n with Goedgekeurd
    declined = clear[clear["visit_status"] == "Afgekeurd"]
    approved = clear[clear["visit_status"] == "Goedgekeurd"].head(n - len(declined))
    sample = pd.concat([declined, approved]).sort_values("id").head(n)

    # Cache results to avoid re-calling the API on reruns
    cache_path = "verdict_cache.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    results = []
    for _, visit in sample.iterrows():
        vid = str(visit["id"])
        if vid in cache:
            llm_verdict = cache[vid]
        else:
            report = evaluate_visit(int(visit["id"]), df, evaluator)
            llm_verdict = report["verdict"]
            cache[vid] = llm_verdict
            json.dump(cache, open(cache_path, "w"))

        ground_truth = "PASS" if visit["visit_status"] == "Goedgekeurd" else "FAIL"
        match = llm_verdict == ground_truth
        results.append({
            "visit_id": visit["id"],
            "date": visit["date"],
            "ground_truth": ground_truth,
            "llm_verdict": llm_verdict,
            "correct": match,
        })
        status = "✓" if match else "✗"
        print(f"  {status} Visit {visit['id']:>3}  GT: {ground_truth:<4}  LLM: {llm_verdict}")

    correct = sum(r["correct"] for r in results)
    total = len(results)
    print()
    print("=" * 40)
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    tp = sum(1 for r in results if r["ground_truth"] == "PASS" and r["llm_verdict"] == "PASS")
    tn = sum(1 for r in results if r["ground_truth"] == "FAIL" and r["llm_verdict"] == "FAIL")
    fp = sum(1 for r in results if r["ground_truth"] == "FAIL" and r["llm_verdict"] == "PASS")
    fn = sum(1 for r in results if r["ground_truth"] == "PASS" and r["llm_verdict"] == "FAIL")
    print(f"  Correct PASS (TP): {tp}  |  Correct FAIL (TN): {tn}")
    print(f"  Wrong   PASS (FP): {fp}  |  Wrong   FAIL (FN): {fn}")
    print("=" * 40)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    questions = pd.read_csv("qaas_questions_rows.csv")
    answers = pd.read_csv("answer_rows.csv")
    visits = pd.read_csv("visit_rows.csv")

    df = (
        answers
        .merge(
            questions[["id", "description", "category", "question_type_label",
                        "section_id", "question_order"]],
            left_on="qaas_question_id",
            right_on="id",
            suffixes=("_answer", "_question"),
        )
        .merge(
            visits[["id", "date", "visit_status", "visitor_id", "assignment_location_id"]],
            left_on="visit_id",
            right_on="id",
            suffixes=("", "_visit"),
        )
    )

    cols = [
        "visit_id", "date", "visit_status", "visitor_id", "assignment_location_id",
        "qaas_question_id", "section_id", "question_order", "description", "category",
        "question_type_label", "answer_text", "answer_value", "answer_list",
    ]
    return df[cols]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    df = load_data()
    visits = pd.read_csv("visit_rows.csv")

    # ── Choose your evaluator ──────────────────────────────────────────────────
    # Option 1 (NOW):   Groq — free, get key at https://console.groq.com/keys
    # Option 2 (LATER): OpenAIEvaluator(api_key=os.environ["OPENAI_API_KEY"])
    # Option 3:         HuggingFaceEvaluator(hf_token=os.environ["HF_TOKEN"])
    # ──────────────────────────────────────────────────────────────────────────
    evaluator = GroqEvaluator(api_key=os.environ["GROQ_API_KEY"])

    #VISIT_ID = 43
    VISIT_ID = 38
    report = evaluate_visit(VISIT_ID, df, evaluator)
    print_visit_report(report)

    # ── Accuracy test (uncomment to run) ──────────────────────────────────────
    # run_accuracy_test(df, visits, evaluator, n=20)
