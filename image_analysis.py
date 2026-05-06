import os
import base64
import glob
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

PHOTOS_ROOT = "receipts_and_questionnaire_photos"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_data():
    questions  = pd.read_csv("qaas_questions_rows.csv")
    assignments = pd.read_csv("assignment_rows.csv")
    answers    = pd.read_csv("answer_rows.csv")
    visits     = pd.read_csv("visit_rows.csv")

    assignments["date_from"] = pd.to_datetime(assignments["date_from"])
    assignments["date_to"]   = pd.to_datetime(assignments["date_to"])
    visits["date"]           = pd.to_datetime(visits["date"])

    return questions, assignments, answers, visits


def find_assignment(visit_id, questions, assignments, answers, visits):
    """Return the assignment row for a given visit via questionnaire_id + date."""
    visit = visits[visits["id"] == visit_id].iloc[0]

    q_ids     = answers[answers["visit_id"] == visit_id]["qaas_question_id"].unique()
    quest_ids = questions[questions["id"].isin(q_ids)]["questionnaire_id"].unique()

    matches = assignments[
        assignments["qaas_questionnaire_id"].isin(quest_ids) &
        (assignments["date_from"] <= visit["date"]) &
        (assignments["date_to"]   >= visit["date"])
    ]

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f"  Warning: {len(matches)} assignment matches for visit {visit_id}, using first")
    return matches.iloc[0]


def find_photos(visit_id, subfolder):
    """Find all photos for a visit_id.
    Receipts:            {uuid}/{visit_id}.ext
    Questionnaire photos: {uuid}/{visit_id}/*.ext
    """
    if subfolder == "receipts":
        pattern = os.path.join(PHOTOS_ROOT, subfolder, "**", f"{visit_id}.*")
        return glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(PHOTOS_ROOT, subfolder, "**", str(visit_id), "*.*")
        return glob.glob(pattern, recursive=True)


# ── Vision LLM ────────────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {"jpg": "image/jpeg", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")


def analyse_photo(photo_path: str, assignment: pd.Series,
                  visit: pd.Series, photo_type: str) -> str:
    """Send one photo to Groq vision model and return the analysis text."""
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    b64    = encode_image(photo_path)
    mime   = image_media_type(photo_path)

    if photo_type == "receipt":
        prompt = f"""Je controleert of een mystery shopper zijn opdracht correct heeft afgerond voor WRAPP.

OPDRACHT DETAILS:
- Restaurant: {assignment['title']}
- Vereiste producten: {assignment['products']}
- Type bezoek: {assignment['assignment_type']}
- Tijdvenster: {assignment['time_from']} – {assignment['time_to']}
- Groepsgrootte: {assignment['group_size']} persoon/personen
- Bezoekdatum (volgens systeem): {visit['date'].strftime('%d %B %Y')}
- Besteed bedrag (volgens systeem): {visit['spent_money'] if pd.notna(visit['spent_money']) else 'onbekend'}

Dit is een betalingsbewijs (bankapp screenshot of kassabon). Controleer elk punt en vermeld altijd BEIDE waarden: wat verwacht wordt én wat op de foto staat.

1. RESTAURANT — Verwacht: {assignment['title']} | Gevonden op foto: ?
2. DATUM — Verwacht: {visit['date'].strftime('%d %B %Y')} | Gevonden op foto: ?
3. TIJD — Verwacht: binnen {assignment['time_from']} – {assignment['time_to']} | Gevonden op foto: ?
4. BEDRAG — Verwacht: {visit['spent_money'] if pd.notna(visit['spent_money']) else 'onbekend'} | Gevonden op foto: ?
5. PRODUCTEN — Verwacht: {assignment['products']} | Gevonden op foto: ?

Geef per punt: ✓ OK / ✗ NIET OK / ? NIET ZICHTBAAR
Vervang de ? door wat je daadwerkelijk op de foto ziet. Als iets niet zichtbaar is, schrijf "niet zichtbaar".

Sluit af met een CONCLUSIE in dit formaat:
CONCLUSIE:
- Vereist: restaurant={assignment['title']}, datum={visit['date'].strftime('%d %B %Y')}, tijdvenster={assignment['time_from']}–{assignment['time_to']}, producten={assignment['products']}, groep={assignment['group_size']} persoon/personen
- Gevonden op foto: [beschrijf wat zichtbaar is: restaurantnaam, datum, tijd, bedrag, producten]
- Oordeel: [CORRECT AFGEROND / TWIJFELACHTIG / NIET CORRECT] — [één zin reden]"""

    else:  # questionnaire photo
        assignment_type = assignment["assignment_type"]
        if assignment_type == "Hier opeten":
            type_check = "Opdracht vereist TER PLAATSE eten. Het eten moet geserveerd zijn op een bord of in een kom (servies). Als het in een takeaway doos, papieren zak of wegwerpverpakking zit → NIET OK."
        elif assignment_type == "Afhalen":
            type_check = "Opdracht vereist AFHALEN (takeaway). Het eten moet in een takeaway verpakking, doos of zak zitten. Als het geserveerd is op borden/servies → NIET OK."
        else:  # Op locatie
            type_check = "Opdracht is een bezoek op locatie. Controleer of de foto overeenkomt met een bezoek aan de fysieke locatie."

        prompt = f"""Je controleert een foto van een mystery shopper bij {assignment['title']}.

OPDRACHT DETAILS:
- Vereiste producten: {assignment['products']}
- Type bezoek: {assignment['assignment_type']}

CONTROLEER:
1. PRODUCTEN: Zijn de vereiste producten zichtbaar op de foto?
2. TYPE BEZOEK: {type_check}

Geef per punt: ✓ OK / ✗ NIET OK / ? ONDUIDELIJK — en een korte reden.

CONCLUSIE:
- Gevonden op foto: [beschrijf wat je ziet]
- Oordeel: [PASSEND / NIET PASSEND / ONDUIDELIJK] — [één zin reden]"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=600,
        temperature=0,
    )
    return response.choices[0].message.content


# ── Per-visit analysis ────────────────────────────────────────────────────────

def analyse_visit_photos(visit_id: int,
                         questions, assignments, answers, visits) -> dict:
    visit      = visits[visits["id"] == visit_id].iloc[0]
    assignment = find_assignment(visit_id, questions, assignments, answers, visits)

    if assignment is None:
        return {"visit_id": visit_id, "error": "No matching assignment found"}

    receipt_photos      = find_photos(visit_id, "receipts")
    questionnaire_photos = find_photos(visit_id, "questionnaire-photos")

    results = {
        "visit_id":   visit_id,
        "assignment": assignment["title"],
        "products":   assignment["products"],
        "receipts":   [],
        "questionnaire_photos": [],
    }

    for path in receipt_photos:
        analysis = analyse_photo(path, assignment, visit, "receipt")
        results["receipts"].append({"file": os.path.basename(path), "analysis": analysis})

    for path in questionnaire_photos:
        analysis = analyse_photo(path, assignment, visit, "questionnaire")
        results["questionnaire_photos"].append({"file": os.path.basename(path), "path": path, "analysis": analysis})

    return results


def print_analysis(result: dict) -> None:
    print("=" * 80)
    print(f"Visit {result['visit_id']}  |  {result.get('assignment', 'UNKNOWN')}")
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  Required products: {result['products']}")
    print()

    if result["receipts"]:
        print("── RECEIPT PHOTOS ──")
        for r in result["receipts"]:
            print(f"  File: {r['file']}")
            print(r["analysis"])
            print()
    else:
        print("  No receipt photo found.")

    if result["questionnaire_photos"]:
        print("── QUESTIONNAIRE PHOTOS ──")
        for r in result["questionnaire_photos"]:
            print(f"  File: {r['file']}")
            print(r["analysis"])
            print()

    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    questions, assignments, answers, visits = load_all_data()

    VISIT_ID = 109
    result = analyse_visit_photos(VISIT_ID, questions, assignments, answers, visits)
    print_analysis(result)
