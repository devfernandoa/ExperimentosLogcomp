import json
from pathlib import Path
from collections import Counter

JUDGE_PATH = Path("judgments.jsonl")

def main():
    rows = []
    with JUDGE_PATH.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    total = len(rows)
    correct = sum(1 for r in rows if r["model_bool"] == r["label"])
    acc = correct / total if total else 0.0

    # Confusion matrix
    # TP = predicted True and label True
    # FP = predicted True and label False
    # TN = predicted False and label False
    # FN = predicted False and label True
    confusion = Counter()
    for r in rows:
        confusion[(r["label"], r["model_bool"])] += 1

    TP = confusion[(True, True)]
    FP = confusion[(False, True)]
    TN = confusion[(False, False)]
    FN = confusion[(True, False)]

    # Metrics
    precision_true = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_true = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_true = (
        2 * precision_true * recall_true / (precision_true + recall_true)
        if (precision_true + recall_true) > 0 else 0.0
    )

    precision_false = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    recall_false = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_false = (
        2 * precision_false * recall_false / (precision_false + recall_false)
        if (precision_false + recall_false) > 0 else 0.0
    )

    # Report
    print("=== MODEL EVALUATION ===")
    print(f"Total pairs: {total}")
    print(f"Accuracy: {acc:.3f}\n")

    print("Confusion matrix (gold -> model):")
    for gold in [True, False]:
        for pred in [True, False]:
            print(f"  gold={gold:5} pred={pred:5}: {confusion[(gold, pred)]}")

    print("\n--- Metrics for predicting True ---")
    print(f"Precision: {precision_true:.3f}")
    print(f"Recall:    {recall_true:.3f}")
    print(f"F1-score:  {f1_true:.3f}")

    print("\n--- Metrics for predicting False ---")
    print(f"Precision: {precision_false:.3f}")
    print(f"Recall:    {recall_false:.3f}")
    print(f"F1-score:  {f1_false:.3f}")

    print("\nWrong cases:\n")
    for r in rows:
        if r["model_bool"] != r["label"]:
            print("test_id        :", r["test_id"])
            print("expected_error :", r["expected_error"])
            print("student_error  :", r["student_error"])
            print("gold label     :", r["label"])
            print("model_output   :", r["model_output"])
            print("---")

if __name__ == "__main__":
    main()
