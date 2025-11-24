import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SYNTH_PATH = Path("../data/synthetic.jsonl")

def load_pairs():
    rows = []
    with SYNTH_PATH.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def embed_texts(model, texts):
    # returns np.array [N, dim]
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)

def eval_threshold(sims, labels, thresh):
    preds = sims >= thresh
    acc = accuracy_score(labels, preds)
    # For consistency with your report, compute per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[True, False],
        zero_division=0
    )
    metrics = {
        "thresh": thresh,
        "accuracy": acc,
        "precision_true": precision[0],
        "recall_true": recall[0],
        "f1_true": f1[0],
        "precision_false": precision[1],
        "recall_false": recall[1],
        "f1_false": f1[1],
    }
    return metrics, preds

def main():
    rows = load_pairs()
    expected_errors = [r["expected_error"] for r in rows]
    student_errors  = [r["student_error"]  for r in rows]
    gold_labels     = np.array([r["label"] for r in rows], dtype=bool)

    # ---- timing: model load
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    t1 = time.time()
    load_secs = t1 - t0

    # ---- timing: encode
    t2 = time.time()
    exp_embs = embed_texts(model, expected_errors)
    stu_embs = embed_texts(model, student_errors)
    t3 = time.time()
    encode_secs = t3 - t2
    encode_per_pair_ms = (encode_secs / len(rows)) * 1000.0

    # cosine similarities
    sims = (exp_embs * stu_embs).sum(axis=1)  # since we normalized, dot == cosine
    # alternatively:
    # sims = cosine_similarity(exp_embs, stu_embs).diagonal()

    # sweep thresholds from 0.5 to 0.95
    all_metrics = []
    best = None
    for thresh in np.linspace(0.5, 0.95, 10):
        metrics, preds = eval_threshold(sims, gold_labels, thresh)
        all_metrics.append(metrics)
        if best is None or metrics["accuracy"] > best["accuracy"]:
            best = {**metrics, "preds": preds}

    # confusion matrix for best threshold
    preds = best["preds"]
    TP = np.sum((gold_labels == True)  & (preds == True))
    FN = np.sum((gold_labels == True)  & (preds == False))
    FP = np.sum((gold_labels == False) & (preds == True))
    TN = np.sum((gold_labels == False) & (preds == False))

    print("=== SENTENCE-TRANSFORMER EVAL ===")
    print(f"Model: all-MiniLM-L6-v2")
    print(f"Total pairs: {len(rows)}")
    print(f"Best threshold: {best['thresh']:.3f}")
    print(f"Accuracy: {best['accuracy']:.3f}")
    print()
    print("Confusion matrix (gold -> model):")
    print(f"  gold=    1 pred=    1: {TP}")
    print(f"  gold=    1 pred=    0: {FN}")
    print(f"  gold=    0 pred=    1: {FP}")
    print(f"  gold=    0 pred=    0: {TN}")
    print()
    print("--- Metrics for predicting True ---")
    print(f"Precision: {best['precision_true']:.3f}")
    print(f"Recall:    {best['recall_true']:.3f}")
    print(f"F1-score:  {best['f1_true']:.3f}")
    print()
    print("--- Metrics for predicting False ---")
    print(f"Precision: {best['precision_false']:.3f}")
    print(f"Recall:    {best['recall_false']:.3f}")
    print(f"F1-score:  {best['f1_false']:.3f}")
    print()
    print("--- Timing ---")
    print(f"Model load time (s): {load_secs:.3f}")
    print(f"Total encode time (s): {encode_secs:.3f}")
    print(f"Avg encode time per pair (ms): {encode_per_pair_ms:.3f}")

    # (optional) print worst disagreements for debugging
    print("\nSome mismatches:\n")
    shown = 0
    for r, sim, pred, gold in sorted(
        zip(rows, sims, preds, gold_labels),
        key=lambda x: abs(x[1] - best["thresh"])
    ):
        if pred != gold:
            print("test_id        :", r["test_id"])
            print("similarity     :", f"{sim:.3f}")
            print("expected_error :", r["expected_error"])
            print("student_error  :", r["student_error"])
            print("gold label     :", gold)
            print("pred label     :", bool(pred))
            print("---")
            shown += 1
        if shown >= 10:
            break

if __name__ == "__main__":
    main()
