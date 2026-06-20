"""
job_finisher.py — continues the interrupted final_test_Timofei.py run.

Rows 1-118 are hardcoded from the interrupted run.
Rows 119-167 (IDs 261-352) are evaluated fresh using Groq.

Run:
    python3 job_finisher.py
"""

from __future__ import annotations

import json, os, time
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL  = "llama-3.3-70b-versatile"
K_EXAMPLES  = 5
GROQ_DELAY  = 3.0
SPLIT_FILE  = "final_split.json"

from Agent1 import AnswerEvaluator, GroqEvaluator
from Rag    import RAGIndex, RAGEvaluator, load_reviews, _build_question_text

# ── Results from the interrupted run (rows 1–118) ────────────────────────────
# Format: (id, true_verdict, a1_predicted, rag_predicted)
COMPLETED = [
    (7,   "good", "good", "good"),
    (8,   "good", "good", "good"),
    (10,  "bad",  "bad",  "bad"),
    (11,  "good", "good", "good"),
    (25,  "bad",  "good", "bad"),
    (26,  "bad",  "good", "bad"),
    (27,  "good", "good", "good"),
    (28,  "good", "good", "bad"),
    (31,  "good", "good", "good"),
    (33,  "good", "good", "bad"),
    (34,  "good", "good", "good"),
    (35,  "good", "good", "good"),
    (38,  "good", "good", "good"),
    (41,  "good", "good", "good"),
    (43,  "good", "good", "good"),
    (45,  "good", "good", "good"),
    (48,  "good", "good", "good"),
    (49,  "good", "good", "good"),
    (52,  "bad",  "good", "bad"),
    (55,  "bad",  "bad",  "bad"),
    (57,  "good", "good", "good"),
    (59,  "good", "good", "good"),
    (60,  "good", "good", "good"),
    (62,  "bad",  "bad",  "good"),
    (63,  "good", "good", "good"),
    (66,  "good", "good", "good"),
    (67,  "bad",  "bad",  "good"),
    (71,  "good", "good", "good"),
    (73,  "bad",  "good", "good"),
    (74,  "good", "good", "bad"),
    (75,  "good", "good", "good"),
    (76,  "good", "good", "good"),
    (78,  "good", "good", "good"),
    (79,  "good", "bad",  "good"),
    (80,  "bad",  "bad",  "bad"),
    (81,  "good", "good", "good"),
    (84,  "good", "good", "good"),
    (86,  "bad",  "bad",  "bad"),
    (87,  "good", "good", "good"),
    (89,  "good", "good", "good"),
    (90,  "bad",  "good", "bad"),
    (91,  "good", "good", "good"),
    (92,  "good", "good", "good"),
    (93,  "bad",  "bad",  "bad"),
    (99,  "good", "good", "good"),
    (100, "good", "good", "good"),
    (101, "good", "good", "good"),
    (103, "good", "good", "good"),
    (106, "bad",  "good", "bad"),
    (108, "good", "good", "good"),
    (111, "bad",  "bad",  "bad"),
    (116, "bad",  "bad",  "good"),
    (117, "bad",  "good", "good"),
    (119, "bad",  "bad",  "bad"),
    (121, "good", "bad",  "good"),
    (122, "good", "bad",  "good"),
    (128, "good", "bad",  "good"),
    (129, "bad",  "bad",  "bad"),
    (130, "bad",  "bad",  "good"),
    (131, "bad",  "good", "bad"),
    (133, "good", "good", "good"),
    (135, "good", "good", "good"),
    (140, "good", "good", "good"),
    (142, "bad",  "good", "good"),
    (145, "good", "good", "good"),
    (146, "good", "good", "good"),
    (147, "good", "good", "good"),
    (149, "bad",  "bad",  "bad"),
    (151, "bad",  "bad",  "bad"),
    (153, "good", "good", "good"),
    (154, "good", "good", "good"),
    (156, "good", "good", "good"),
    (159, "good", "good", "good"),
    (161, "bad",  "good", "good"),
    (166, "bad",  "bad",  "good"),
    (169, "bad",  "good", "good"),
    (173, "good", "good", "good"),
    (174, "bad",  "bad",  "bad"),
    (175, "good", "good", "bad"),
    (176, "good", "bad",  "bad"),
    (180, "bad",  "good", "good"),
    (181, "bad",  "good", "bad"),
    (183, "good", "good", "bad"),
    (184, "bad",  "good", "bad"),
    (186, "bad",  "bad",  "good"),
    (187, "good", "good", "good"),
    (188, "good", "bad",  "good"),
    (189, "good", "good", "good"),
    (191, "bad",  "good", "good"),
    (192, "good", "good", "good"),
    (193, "good", "good", "good"),
    (194, "good", "good", "good"),
    (197, "bad",  "good", "good"),
    (198, "bad",  "bad",  "bad"),
    (207, "bad",  "good", "bad"),
    (213, "good", "bad",  "bad"),
    (215, "good", "good", "good"),
    (216, "good", "good", "good"),
    (217, "bad",  "bad",  "bad"),
    (218, "bad",  "bad",  "bad"),
    (219, "bad",  "bad",  "good"),
    (222, "good", "good", "bad"),
    (223, "good", "good", "good"),
    (226, "good", "good", "good"),
    (227, "good", "good", "good"),
    (230, "bad",  "good", "bad"),
    (232, "good", "good", "good"),
    (235, "good", "good", "good"),
    (236, "good", "good", "good"),
    (238, "good", "good", "bad"),
    (241, "good", "good", "good"),
    (244, "good", "good", "good"),
    (245, "good", "good", "good"),
    (246, "good", "good", "good"),
    (255, "bad",  "good", "good"),
    (257, "good", "good", "good"),
    (258, "bad",  "bad",  "bad"),
    (259, "good", "good", "good"),
    # ── rows 119–153 (second key) ─────────────────────────────────────────────
    (261, "good", "good", "good"),
    (262, "good", "good", "bad"),
    (263, "good", "good", "bad"),
    (268, "good", "good", "good"),
    (271, "bad",  "good", "bad"),
    (273, "good", "good", "good"),
    (274, "good", "good", "good"),
    (275, "good", "good", "good"),
    (276, "good", "good", "good"),
    (278, "bad",  "good", "good"),
    (280, "good", "good", "good"),
    (286, "good", "good", "good"),
    (287, "good", "good", "good"),
    (288, "good", "bad",  "good"),
    (289, "good", "good", "good"),
    (293, "good", "good", "good"),
    (294, "good", "good", "bad"),
    (295, "good", "good", "bad"),
    (296, "bad",  "good", "good"),
    (297, "good", "good", "good"),
    (298, "good", "bad",  "bad"),
    (299, "good", "good", "bad"),
    (300, "good", "good", "good"),
    (301, "good", "good", "good"),
    (302, "good", "good", "good"),
    (303, "good", "good", "bad"),
    (304, "good", "good", "good"),
    (305, "good", "good", "good"),
    (309, "good", "good", "good"),
    (311, "bad",  "bad",  "bad"),
    (314, "good", "good", "bad"),
    (316, "good", "good", "good"),
    (317, "bad",  "good", "good"),
    (319, "good", "good", "good"),
    (321, "bad",  "good", "good"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _eval_row(row: pd.Series, evaluator: AnswerEvaluator) -> str:
    is_good, _ = evaluator.evaluate_text(
        _build_question_text(row), str(row["shopper_answer"])
    )
    return "good" if is_good else "bad"


def _metrics(res_a1: list[dict], res_rag: list[dict], label_a1: str, label_rag: str):
    for results, label in [(res_a1, label_a1), (res_rag, label_rag)]:
        n  = len(results)
        ok = sum(r["correct"] for r in results)
        tp = sum(1 for r in results if r["true"] == "good" and r["predicted"] == "good")
        tn = sum(1 for r in results if r["true"] == "bad"  and r["predicted"] == "bad")
        fp = sum(1 for r in results if r["true"] == "bad"  and r["predicted"] == "good")
        fn = sum(1 for r in results if r["true"] == "good" and r["predicted"] == "bad")
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"\n{'─'*62}")
        print(f"  {label}")
        print(f"{'─'*62}")
        print(f"  Accuracy : {ok}/{n} = {ok/n*100:.1f}%")
        print(f"  Precision: {prec*100:.1f}%  Recall: {rec*100:.1f}%  F1: {f1*100:.1f}%")
        print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"{'─'*62}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    # Build result lists from completed rows
    res_a1  = [{"id": rid, "true": tv, "predicted": a1p, "correct": a1p == tv}
               for rid, tv, a1p, _    in COMPLETED]
    res_rag = [{"id": rid, "true": tv, "predicted": rp,  "correct": rp  == tv}
               for rid, tv, _,   rp   in COMPLETED]

    done_ids = {r["id"] for r in res_a1}

    # Load full test set and find remaining rows
    print("Loading llm_reviews from Supabase ...")
    df = load_reviews()
    split    = json.load(open(SPLIT_FILE))
    train_df = df[df["id"].isin(split["train_ids"])].reset_index(drop=True)
    test_df  = df[df["id"].isin(split["test_ids"])].reset_index(drop=True)

    remaining = test_df[~test_df["id"].isin(done_ids)].reset_index(drop=True)
    total     = len(test_df)
    print(f"Already done : {len(done_ids)}/{total}")
    print(f"Remaining    : {len(remaining)}/{total}")
    print(f"\nBuilding TF-IDF index ...")
    index = RAGIndex(train_df)

    a1_eval  = GroqEvaluator(api_key=groq_key, model=GROQ_MODEL)
    rag_eval = RAGEvaluator(groq_api_key=groq_key, index=index, k=K_EXAMPLES, model=GROQ_MODEL)

    lbl_a1  = f"Agent1+{GROQ_MODEL}"
    lbl_rag = f"RAG+{GROQ_MODEL}(k={K_EXAMPLES})"

    print(f"\nRunning remaining {len(remaining)} rows ({lbl_a1} vs {lbl_rag}) ...\n")

    offset = len(done_ids)
    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        rid = str(row["id"])
        tv  = row["true_verdict"]

        try:
            a1p = _eval_row(row, a1_eval)
        except Exception as e:
            print(f"    [!] a1 row {rid}: {e} — defaulting to 'good'")
            a1p = "good"
        time.sleep(GROQ_DELAY)

        try:
            rp = _eval_row(row, rag_eval)
        except Exception as e:
            print(f"    [!] rag row {rid}: {e} — defaulting to 'good'")
            rp = "good"
        time.sleep(GROQ_DELAY)

        res_a1.append( {"id": int(row["id"]), "true": tv, "predicted": a1p, "correct": a1p == tv})
        res_rag.append({"id": int(row["id"]), "true": tv, "predicted": rp,  "correct": rp  == tv})

        a1_ok  = "+" if a1p  == tv else "x"
        rag_ok = "+" if rp   == tv else "x"
        print(f"  {offset+i:>3}/{total}  id={rid:>4}  true={tv:<4}  "
              f"a1={a1p:<4}{a1_ok}  rag={rp:<4}{rag_ok}")

    # ── Final metrics ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  FINAL RESULTS — Timofei  ({GROQ_MODEL})")
    print(f"{'='*62}")
    _metrics(res_a1, res_rag, lbl_a1, lbl_rag)

    # Delta summary
    ok1 = sum(r["correct"] for r in res_a1)
    ok2 = sum(r["correct"] for r in res_rag)
    n   = len(res_a1)
    print(f"\n  Agent1 accuracy : {ok1}/{n} = {ok1/n*100:.1f}%")
    print(f"  RAG    accuracy : {ok2}/{n} = {ok2/n*100:.1f}%")
    print(f"  Delta           : {(ok2-ok1)/n*100:+.1f}%")

    fixed = sum(1 for a, b in zip(res_a1, res_rag) if not a["correct"] and b["correct"])
    broke = sum(1 for a, b in zip(res_a1, res_rag) if a["correct"] and not b["correct"])
    print(f"  RAG fixed {fixed} Agent1 mistakes, broke {broke}")
    print(f"{'='*62}")

    json.dump(
        {"res_a1": res_a1, "res_rag": res_rag},
        open("results_Timofei_groq_final.json", "w"), indent=2,
    )
    print(f"\n  Saved → results_Timofei_groq_final.json")


if __name__ == "__main__":
    main()
