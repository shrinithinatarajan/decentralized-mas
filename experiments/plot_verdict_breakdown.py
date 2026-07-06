"""Plot verdict distribution for all models by reading saved trace JSONL files.
No API calls, no orchestrator re-runs.
"""
import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("experiments/results")
OUT     = Path("experiments/figures/verdict_breakdown.pdf")

SKIP = {"mixtral-8x7b", "gemma-3n-4b"}


def load_traces(label: str) -> list[dict]:
    path = RESULTS / f"traces_{label}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def compute_counts(traces: list[dict]) -> dict:
    counts = {"correct": 0, "wrong": 0, "uncertain": 0,
              "pred_S": 0, "pred_R": 0, "pred_U": 0}
    for r in traces:
        pred = r["final_verdict"]
        true = r["true_label"]
        counts[f"pred_{pred[0]}"] += 1
        if pred == "UNCERTAIN":
            counts["uncertain"] += 1
        elif pred == true:
            counts["correct"] += 1
        else:
            counts["wrong"] += 1
    return counts


def main():
    # Find all trace files
    trace_files = sorted(RESULTS.glob("traces_*.jsonl"))
    model_data = {}
    for f in trace_files:
        label = f.stem.replace("traces_", "")
        if label in SKIP:
            continue
        traces = load_traces(label)
        if traces:
            model_data[label] = compute_counts(traces)
            print(f"  {label}: correct={model_data[label]['correct']} "
                  f"wrong={model_data[label]['wrong']} uncertain={model_data[label]['uncertain']}")

    if not model_data:
        print("No trace files found. Run compare_models.py first.")
        return

    labels = list(model_data.keys())
    n = 40
    x = np.arange(len(labels))
    w = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Verdict Breakdown Across Models (n=40 cases)", fontsize=13, fontweight="bold")

    # Left: stacked verdict distribution
    ax = axes[0]
    pred_S = [model_data[l]["pred_S"] for l in labels]
    pred_R = [model_data[l]["pred_R"] for l in labels]
    pred_U = [model_data[l]["pred_U"] for l in labels]

    ax.bar(x, pred_S, w, label="SENSITIVE",  color="#4CAF50")
    ax.bar(x, pred_R, w, bottom=pred_S,      label="RESISTANT", color="#F44336")
    ax.bar(x, pred_U, w,
           bottom=[s+r for s,r in zip(pred_S, pred_R)],
           label="UNCERTAIN", color="#9E9E9E")
    ax.axhline(21, color="#4CAF50", linestyle="--", linewidth=1, alpha=0.6, label="True SENSITIVE (21)")
    ax.axhline(19, color="#F44336", linestyle="--", linewidth=1, alpha=0.6, label="True RESISTANT (19)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cases")
    ax.set_title("Predicted Verdict Distribution")
    ax.set_ylim(0, n + 2)
    ax.legend(fontsize=8, loc="upper right")

    # Right: correct / wrong / uncertain
    ax2 = axes[1]
    correct   = [model_data[l]["correct"]   for l in labels]
    wrong     = [model_data[l]["wrong"]     for l in labels]
    uncertain = [model_data[l]["uncertain"] for l in labels]

    ax2.bar(x, correct,   w, label="Correct",   color="#2196F3")
    ax2.bar(x, wrong,     w, bottom=correct,     label="Wrong",     color="#FF9800")
    ax2.bar(x, uncertain, w,
            bottom=[c+w_ for c,w_ in zip(correct, wrong)],
            label="Uncertain", color="#9E9E9E")

    for i, l in enumerate(labels):
        c, w_ = model_data[l]["correct"], model_data[l]["wrong"]
        denom = c + w_
        if denom > 0:
            ax2.text(i, n + 0.5, f"{c/denom:.0%}", ha="center", va="bottom", fontsize=8, color="#333")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Cases")
    ax2.set_title("Prediction Outcomes (accuracy excl. uncertain shown above)")
    ax2.set_ylim(0, n + 3)
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=150)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
