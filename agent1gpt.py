import pandas as pd
from abc import ABC, abstractmethod
from dotenv import load_dotenv
load_dotenv()

pd.set_option("display.max_colwidth", None)

TEXT_THRESHOLD = 0.70       # min fraction of text questions that must be good
STRUCTURED_THRESHOLD = 0.90 # min fraction of structured questions that must be answered

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
- Toelichting eten (positief): "Beide pita's waren goed op smaak en een lekkere mix van ingrediënten. De frietjes waren ietwat aan de zoute kant maar hier hou ik persoonlijk wel van." → goed
- Toelichting eten (negatief): "De zoete aardappelfriet had teveel zout. De kipshoarma was alleen onderin te proeven, geen saus of uitje." → goed
- Toelichting ingrediëntvraag: "Ik vroeg wat za'atar was. De medewerker gaf aan dat dit een kruid was en liet dit ruiken." → goed
- Toelichting wachttijd: "We werden binnen een minuut na binnenkomst geholpen. Er waren geen andere klanten." → goed
- Toelichting uiterlijk eten: "De avocado was niet netjes gesneden en werd er gewoon opgegooid." → goed
- Toelichting toilet: "Toiletbril stond omhoog. De zeep was op." → goed
- Toelichting hygiëne medewerkers: "Alle medewerkers hadden werkkleding en handschoenen aan." → goed
- Was drukte mogelijk: "Ja, er zaten maximaal 10 mensen in de zaak dus een praatje maken was zeker mogelijk." → goed
- Compliment: "Dank voor de snelle bediening en het heerlijke eten." → goed

SLECHTE ANTWOORDEN:
- Vloeken in welke taal dan ook → slecht
- Onzinnige woorden: "asdfsda", "weasdfsd", "no is bike seven house tree" → slecht
- Antwoord van 1 woord als uitleg gevraagd wordt: "ja", "goed", "prima" → slecht
- Antwoord tegenstrijdig met het vorige ja/nee antwoord → slecht

Antwoord ALTIJD in dit formaat (max 1 zin reden):
goed: <korte reden>
of
slecht: <korte reden>"""

# ── Evaluators ────────────────────────────────────────────────────────────────

class AnswerEvaluator(ABC):
    @abstractmethod
    def evaluate_text(self, question: str, answer: str) -> tuple[bool, str]:
        """Return (is_good, reason)."""
        pass


class GroqEvaluator(AnswerEvaluator):
    """Free LLM via Groq. Get a free key at: https://console.groq.com/keys"""

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model

    def evaluate_text(self, question: str, answer: str) -> tuple[bool, str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Vraag: {question}\nAntwoord: {answer}"},
            ],
            max_tokens=60,
            temperature=0,
        )
        return _parse_verdict(response.choices[0].message.content)


class HuggingFaceEvaluator(AnswerEvaluator):
    """Zero-shot classification via HuggingFace Inference API (free with token).
    Get a free token at: https://huggingface.co/settings/tokens
    """

    def __init__(self, hf_token: str):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=hf_token)
        self.model = "joeddav/xlm-roberta-large-xnli"

    def evaluate_text(self, question: str, answer: str) -> tuple[bool, str]:
        text = f"Vraag: {question}\nAntwoord: {answer}"
        result = self.client.zero_shot_classification(
            text=text,
            candidate_labels=["passend antwoord", "onvoldoende antwoord"],
            model=self.model,
        )
        is_good = result[0].label == "passend antwoord"
        return is_good, ""


class OpenAIEvaluator(AnswerEvaluator):
    """OpenAI-compatible evaluator. Works with OpenAI directly or UvA LiteLLM proxy.
    UvA base URL: https://aichat.uva.nl/v1  (requires valid workspace key)
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def evaluate_text(self, question: str, answer: str) -> tuple[bool, str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Vraag: {question}\nAntwoord: {answer}"},
            ],
            max_tokens=60,
            temperature=0,
        )
        return _parse_verdict(response.choices[0].message.content)


def _parse_verdict(raw: str) -> tuple[bool, str]:
    """Parse 'goed: reason' or 'slecht: reason' from LLM output."""
    text = raw.strip().lower()
    is_good = text.startswith("goed")
    reason = raw.strip().split(":", 1)[1].strip() if ":" in raw else raw.strip()
    return is_good, reason


# ── Per-question evaluation ───────────────────────────────────────────────────

RULE_BASED_TYPES = {"Getal", "Schaal", "Datum", "Checkbox", "Foto upload"}
FOLLOW_UP_LABELS = {
    "toelichting",
    "licht toe hoe dit verliep",
    "licht je score toe",
    "licht je antwoord toe",
}


def evaluate_answer(row: pd.Series, evaluator: AnswerEvaluator,
                    parent_question: str | None = None,
                    parent_answer: str | None = None) -> tuple[str, str]:
    """Return (verdict, reason). verdict is 'good' or 'bad'."""
    q_type = row["question_type_label"]
    text = row["answer_text"]
    value = row["answer_value"]
    has_text = pd.notna(text) and str(text).strip() != ""
    has_value = pd.notna(value)

    if q_type in RULE_BASED_TYPES:
        return ("good", "") if (has_text or has_value) else ("bad", "Geen antwoord gegeven")

    if q_type == "Tekst":
        if not has_text:
            return ("bad", "Geen antwoord gegeven")

        label = str(row["description"]).strip().lower().rstrip(".")
        if label in FOLLOW_UP_LABELS and parent_question:
            # Include previous question + its answer as full context
            ctx = f'[toelichting op: "{parent_question}"'
            if parent_answer:
                ctx += f' → antwoord: "{parent_answer}"'
            ctx += f']\n{row["description"]}'
            question = ctx
        else:
            question = row["description"]

        is_good, reason = evaluator.evaluate_text(question, str(text))
        return ("good" if is_good else "bad"), reason

    return ("good", "")


# ── Visit-level verdict ───────────────────────────────────────────────────────

def evaluate_visit(visit_id: int, df: pd.DataFrame, evaluator: AnswerEvaluator) -> dict:
    rows = df[df["visit_id"] == visit_id].copy()
    rows = rows.sort_values(["section_id", "question_order"]).reset_index(drop=True)

    # Build parent question + parent answer for every follow-up row
    descriptions = rows["description"].tolist()
    raw_answers = rows.apply(
        lambda r: str(r["answer_text"]) if pd.notna(r["answer_text"])
                  else (str(r["answer_value"]) if pd.notna(r["answer_value"]) else ""),
        axis=1
    ).tolist()

    parent_questions, parent_answers = [], []
    for i, desc in enumerate(descriptions):
        if desc.strip().lower().rstrip(".") in FOLLOW_UP_LABELS and i > 0:
            parent_q, parent_a = None, None
            for j in range(i - 1, -1, -1):
                if descriptions[j].strip().lower().rstrip(".") not in FOLLOW_UP_LABELS:
                    parent_q = descriptions[j]
                    parent_a = raw_answers[j]
                    break
            parent_questions.append(parent_q)
            parent_answers.append(parent_a)
        else:
            parent_questions.append(None)
            parent_answers.append(None)

    rows["parent_question"] = parent_questions
    rows["parent_answer"] = parent_answers

    results = rows.apply(
        lambda r: evaluate_answer(r, evaluator, r["parent_question"], r["parent_answer"]),
        axis=1
    )
    rows["verdict"] = results.apply(lambda x: x[0])
    rows["reason"]  = results.apply(lambda x: x[1])

    text_rows       = rows[rows["question_type_label"] == "Tekst"]
    structured_rows = rows[rows["question_type_label"] != "Tekst"]

    text_good   = (text_rows["verdict"] == "good").sum()
    text_total  = len(text_rows)
    text_pct    = text_good / text_total if text_total > 0 else 1.0

    struct_good  = (structured_rows["verdict"] == "good").sum()
    struct_total = len(structured_rows)
    struct_pct   = struct_good / struct_total if struct_total > 0 else 1.0

    passed = text_pct >= TEXT_THRESHOLD and struct_pct >= STRUCTURED_THRESHOLD

    return {
        "visit_id":       visit_id,
        "total_questions": len(rows),
        "good":           int((rows["verdict"] == "good").sum()),
        "bad":            int((rows["verdict"] == "bad").sum()),
        "text_good":      int(text_good),
        "text_total":     int(text_total),
        "text_pct":       round(text_pct * 100, 1),
        "struct_good":    int(struct_good),
        "struct_total":   int(struct_total),
        "struct_pct":     round(struct_pct * 100, 1),
        "verdict":        "PASS" if passed else "FAIL",
        "details":        rows[["qaas_question_id", "question_order", "description",
                                 "parent_question", "parent_answer", "question_type_label",
                                 "answer_text", "answer_value", "verdict", "reason"]],
    }


def print_visit_report(report: dict) -> None:
    v = report
    print("=" * 80)
    print(f"Visit {v['visit_id']}  |  {v['verdict']}")
    print(f"  Text:       {v['text_good']}/{v['text_total']} good ({v['text_pct']}%)  [threshold 70%]")
    print(f"  Structured: {v['struct_good']}/{v['struct_total']} answered ({v['struct_pct']}%)  [threshold 90%]")
    print("=" * 80)
    for _, row in v["details"].iterrows():
        icon = "✓" if row["verdict"] == "good" else "✗"
        ans = row["answer_text"] if pd.notna(row["answer_text"]) else row["answer_value"]
        q = str(row["description"])[:72]
        is_text = row["question_type_label"] == "Tekst"

        if pd.notna(row["parent_question"]):
            parent_a = f' → "{str(row["parent_answer"])}"' if pd.notna(row["parent_answer"]) and str(row["parent_answer"]) else ""
            context = f'  [re: "{str(row["parent_question"])[:50]}"{parent_a}]'
        else:
            context = ""

        print(f"  {icon} Q{row['qaas_question_id']:>3} [{row['question_type_label']}]: {q}{context}")
        print(f"         -> {ans}")
        if is_text and row["reason"]:
            print(f"         ℹ  {row['reason']}")
    print()


# ── Accuracy evaluation ───────────────────────────────────────────────────────

def run_accuracy_test(df: pd.DataFrame, visits: pd.DataFrame,
                      evaluator: AnswerEvaluator, n: int = 20) -> None:
    import json, os

    clear    = visits[visits["visit_status"].isin(["Goedgekeurd", "Afgekeurd"])].copy()
    declined = clear[clear["visit_status"] == "Afgekeurd"]
    approved = clear[clear["visit_status"] == "Goedgekeurd"].head(n - len(declined))
    sample   = pd.concat([declined, approved]).sort_values("id").head(n)

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
        results.append({"visit_id": visit["id"], "ground_truth": ground_truth,
                         "llm_verdict": llm_verdict, "correct": match})
        print(f"  {'✓' if match else '✗'} Visit {visit['id']:>3}  GT: {ground_truth:<4}  LLM: {llm_verdict}")

    correct = sum(r["correct"] for r in results)
    total   = len(results)
    print(f"\n{'='*40}\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
    tp = sum(1 for r in results if r["ground_truth"]=="PASS" and r["llm_verdict"]=="PASS")
    tn = sum(1 for r in results if r["ground_truth"]=="FAIL" and r["llm_verdict"]=="FAIL")
    fp = sum(1 for r in results if r["ground_truth"]=="FAIL" and r["llm_verdict"]=="PASS")
    fn = sum(1 for r in results if r["ground_truth"]=="PASS" and r["llm_verdict"]=="FAIL")
    print(f"  Correct PASS (TP): {tp}  |  Correct FAIL (TN): {tn}")
    print(f"  Wrong   PASS (FP): {fp}  |  Wrong   FAIL (FN): {fn}\n{'='*40}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    questions = pd.read_csv("qaas_questions_rows.csv")
    answers   = pd.read_csv("answer_rows.csv")
    visits    = pd.read_csv("visit_rows.csv")

    df = (
        answers
        .merge(
            questions[["id", "description", "category", "question_type_label",
                        "section_id", "question_order"]],
            left_on="qaas_question_id", right_on="id",
            suffixes=("_answer", "_question"),
        )
        .merge(
            visits[["id", "date", "visit_status", "visitor_id", "assignment_location_id"]],
            left_on="visit_id", right_on="id",
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

    df     = load_data()
    visits = pd.read_csv("visit_rows.csv")

    # ── Choose your evaluator ──────────────────────────────────────────────────
    # Active: GPT-4o via OpenAI-compatible endpoint. Same LLM as the ReAct
    # version (Agent_ReAct.py) so the comparison isolates the agent paradigm.
    # Reads OPENAI_API_KEY (required) and UVA_API_BASE (optional — set when
    # using a UvA workspace key against the LiteLLM proxy; leave unset to
    # talk to api.openai.com directly).
    #
    # Fallback options (commented):
    # - GroqEvaluator(api_key=os.environ["GROQ_API_KEY"])    # free Groq
    # ──────────────────────────────────────────────────────────────────────────
    evaluator = OpenAIEvaluator(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("UVA_API_BASE") or None,
        model="gpt-4o",
    )

    VISIT_ID = 43
    report = evaluate_visit(VISIT_ID, df, evaluator)
    print_visit_report(report)

    # ── Accuracy test (uncomment to run) ──────────────────────────────────────
    # run_accuracy_test(df, visits, evaluator, n=20)
