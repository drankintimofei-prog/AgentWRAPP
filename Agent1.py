import pandas as pd
from abc import ABC, abstractmethod

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


def evaluate_answer(row: pd.Series, evaluator: AnswerEvaluator) -> str:
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
        return "good" if evaluator.evaluate_text(row["description"], str(text)) else "bad"

    return "good"  # unknown type: benefit of the doubt


# ── Visit-level verdict ───────────────────────────────────────────────────────

def evaluate_visit(visit_id: int, df: pd.DataFrame, evaluator: AnswerEvaluator) -> dict:
    rows = df[df["visit_id"] == visit_id].copy()
    rows["verdict"] = rows.apply(lambda r: evaluate_answer(r, evaluator), axis=1)

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
        "details": rows[["qaas_question_id", "description", "question_type_label",
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
        print(f"  {icon} Q{row['qaas_question_id']:>3} [{row['question_type_label']}]: {q}")
        print(f"         -> {ans}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    df = pd.read_csv("qa_joined.csv")

    # ── Choose your evaluator ──────────────────────────────────────────────────
    # Option 1 (NOW):   Groq — free, get key at https://console.groq.com/keys
    #                   Set env var:  export GROQ_API_KEY="gsk_..."
    # Option 2 (LATER): OpenAIEvaluator(api_key=os.environ["OPENAI_API_KEY"])
    # Option 3:         HuggingFaceEvaluator(hf_token=os.environ["HF_TOKEN"])
    # ──────────────────────────────────────────────────────────────────────────
    evaluator = GroqEvaluator(api_key=os.environ["GROQ_API_KEY"])

    VISIT_ID = 43
    report = evaluate_visit(VISIT_ID, df, evaluator)
    print_visit_report(report)
