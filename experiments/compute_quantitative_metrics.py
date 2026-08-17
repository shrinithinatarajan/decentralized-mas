"""Compute quantitative metrics from gold standard traces.

Metrics:
  - Accuracy, Balanced Accuracy
  - AUROC, AUPRC
  - Spearman rho (confidence vs. correct)
  - Cohen's Kappa
  - Per-class Precision, Recall, F1
  - Confusion matrix

Usage:
    PYTHONPATH=. python experiments/compute_quantitative_metrics.py
    PYTHONPATH=. python experiments/compute_quantitative_metrics.py --traces experiments/results/traces_gold_standard.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

RESULTS = Path("experiments/results")


def load_traces(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def verdict_to_score(verdict: str, confidence: float) -> float:
    """Convert verdict + confidence to a RESISTANT probability score for AUROC/AUPRC."""
    if verdict == "RESISTANT":
        return confidence
    elif verdict == "SENSITIVE":
        return 1.0 - confidence
    else:  # UNCERTAIN
        return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default=str(RESULTS / "traces_gold_standard.jsonl"))
    args = parser.parse_args()

    records = load_traces(Path(args.traces))
    print(f"Loaded {len(records)} traces from {args.traces}")

    # --- Build arrays ---
    y_true_bin = []   # 1=RESISTANT, 0=SENSITIVE
    y_pred_bin = []   # predicted binary (UNCERTAIN mapped to wrong)
    y_score = []      # continuous RESISTANT probability for AUROC/AUPRC
    y_correct = []    # 1=correct, 0=wrong/uncertain
    confidences = []
    labels_raw = []
    verdicts_raw = []

    uncertain_count = 0
    for r in records:
        true_label = r["true_label"]
        verdict = r["final_verdict"]
        conf = r["final_confidence"]

        true_bin = 1 if true_label == "RESISTANT" else 0
        if verdict == "UNCERTAIN":
            uncertain_count += 1
            pred_bin = 1 - true_bin  # count as wrong
        else:
            pred_bin = 1 if verdict == "RESISTANT" else 0

        y_true_bin.append(true_bin)
        y_pred_bin.append(pred_bin)
        y_score.append(verdict_to_score(verdict, conf))
        y_correct.append(1 if r["correct"] else 0)
        confidences.append(conf)
        labels_raw.append(true_label)
        verdicts_raw.append(verdict)

    y_true_bin = np.array(y_true_bin)
    y_pred_bin = np.array(y_pred_bin)
    y_score = np.array(y_score)
    y_correct = np.array(y_correct)
    confidences = np.array(confidences)

    # --- Exclude uncertain for some metrics ---
    mask_decided = np.array([v != "UNCERTAIN" for v in verdicts_raw])
    y_true_decided = y_true_bin[mask_decided]
    y_pred_decided = y_pred_bin[mask_decided]

    # --- Metrics ---
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    auroc = roc_auc_score(y_true_bin, y_score)
    auprc = average_precision_score(y_true_bin, y_score)
    kappa = cohen_kappa_score(y_true_bin, y_pred_bin)
    kappa_decided = cohen_kappa_score(y_true_decided, y_pred_decided) if len(y_true_decided) > 0 else float("nan")

    spearman_rho, spearman_p = stats.spearmanr(confidences, y_correct)

    prec_s = precision_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    rec_s = recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    f1_s = f1_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)

    prec_r = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    rec_r = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    f1_r = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])

    n_sensitive = int((y_true_bin == 0).sum())
    n_resistant = int((y_true_bin == 1).sum())

    print()
    print("=" * 60)
    print(f"  Dataset: {len(records)} cases  "
          f"({n_sensitive} SENSITIVE / {n_resistant} RESISTANT)")
    print(f"  Uncertain predictions: {uncertain_count}")
    print("=" * 60)
    print()
    print("  Core Accuracy")
    print(f"    Accuracy:             {accuracy:.4f}  ({int(accuracy*len(records))}/{len(records)})")
    print(f"    Balanced Accuracy:    {bal_acc:.4f}")
    print()
    print("  Ranking Metrics")
    print(f"    AUROC:                {auroc:.4f}")
    print(f"    AUPRC:                {auprc:.4f}")
    print()
    print("  Correlation")
    print(f"    Spearman rho:         {spearman_rho:.4f}  (p={spearman_p:.4f})")
    print()
    print("  Agreement")
    print(f"    Cohen's Kappa:        {kappa:.4f}  (all cases, uncertain=wrong)")
    print(f"    Cohen's Kappa:        {kappa_decided:.4f}  (decided cases only, n={mask_decided.sum()})")
    print()
    print("  Per-class (SENSITIVE as positive)")
    print(f"    Precision:            {prec_s:.4f}")
    print(f"    Recall:               {rec_s:.4f}")
    print(f"    F1:                   {f1_s:.4f}")
    print()
    print("  Per-class (RESISTANT as positive)")
    print(f"    Precision:            {prec_r:.4f}")
    print(f"    Recall:               {rec_r:.4f}")
    print(f"    F1:                   {f1_r:.4f}")
    print()
    print("  Confusion Matrix (rows=true, cols=pred)")
    print(f"                 Pred-S   Pred-R")
    print(f"    True-S:      {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"    True-R:      {cm[1,0]:5d}    {cm[1,1]:5d}")
    print("=" * 60)

    output = {
        "n_cases": len(records),
        "n_sensitive": n_sensitive,
        "n_resistant": n_resistant,
        "n_uncertain": uncertain_count,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "spearman_rho": round(spearman_rho, 4),
        "spearman_p": round(spearman_p, 4),
        "cohen_kappa_all": round(kappa, 4),
        "cohen_kappa_decided": round(kappa_decided, 4),
        "sensitive_precision": round(prec_s, 4),
        "sensitive_recall": round(rec_s, 4),
        "sensitive_f1": round(f1_s, 4),
        "resistant_precision": round(prec_r, 4),
        "resistant_recall": round(rec_r, 4),
        "resistant_f1": round(f1_r, 4),
        "confusion_matrix": cm.tolist(),
    }
    out_path = RESULTS / "quantitative_metrics.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
