import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 1. Insert a test row ──────────────────────────────────────────────────────
print("Inserting test row...")
insert_result = client.table("llm_reviews").insert({
    "visit_id": 9999,
    "qaas_question_id": 0,
    "question_text": "TEST QUESTION",
    "parent_question": "TEST PARENT",
    "parent_answer": "Ja",
    "shopper_answer": "Test antwoord",
    "llm_verdict": "good",
    "llm_reason": "Test reden",
    "nigel_rating": "agree",
    "nigel_comment": "Test comment from Nigel",
}).execute()

inserted = insert_result.data[0]
test_id = inserted["id"]
print(f"Inserted row with id={test_id} ✓")

# ── 2. Read it back ───────────────────────────────────────────────────────────
print("\nReading back...")
read_result = client.table("llm_reviews").select("*").eq("id", test_id).execute()
row = read_result.data[0]
print(f"  id:           {row['id']}")
print(f"  visit_id:     {row['visit_id']}")
print(f"  question:     {row['question_text']}")
print(f"  llm_verdict:  {row['llm_verdict']}")
print(f"  nigel_rating: {row['nigel_rating']}")
print(f"  reviewed_at:  {row['reviewed_at']}")
print("Read OK ✓")

# ── 3. Delete test row ────────────────────────────────────────────────────────
print("\nCleaning up test row...")
client.table("llm_reviews").delete().eq("id", test_id).execute()
print("Deleted ✓")

print("\nAll tests passed — Supabase connection is working correctly.")
