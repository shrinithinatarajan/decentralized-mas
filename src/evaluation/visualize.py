from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.evaluation.metrics import EvaluationMetrics
from src.protocols.debate_engine import ConsensusResult


def plot_roc_curve(
    y_true: list[int],
    y_score: list[float],
    label: str = "Framework",
) -> matplotlib.figure.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{label} (AUC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return fig


def plot_pr_curve(
    y_true: list[int],
    y_score: list[float],
    label: str = "Framework",
) -> matplotlib.figure.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"{label} (AUC={auprc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    return fig


def plot_ablation_comparison(
    metrics: dict[str, EvaluationMetrics],
) -> matplotlib.figure.Figure:
    labels = list(metrics.keys())
    aurocs  = [m.auroc for m in metrics.values()]
    kappas  = [m.cohens_kappa for m in metrics.values()]

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    bars1 = ax.bar([i - width/2 for i in x], aurocs, width, label="AUROC", color="#2196F3")
    bars2 = ax.bar([i + width/2 for i in x], kappas, width, label="Cohen's κ", color="#FF9800")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axhline(0.5, color="#2196F3", linewidth=0.6, linestyle=":", alpha=0.5, label="AUROC=0.5 (chance)")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        offset = 0.01 if h >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width()/2, h + offset, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — AUROC & Cohen's κ")
    ax.set_ylim(-0.3, 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_debate_convergence(results: list[ConsensusResult]) -> matplotlib.figure.Figure:
    rounds = [r.rounds_taken for r in results]
    fig, ax = plt.subplots()
    if rounds:
        ax.hist(rounds, bins=range(min(rounds), max(rounds) + 2), align="left", rwidth=0.8)
    ax.set_xlabel("Rounds to Consensus")
    ax.set_ylabel("Count")
    ax.set_title("Debate Convergence")
    return fig


def plot_axiom_frequency(results: list[ConsensusResult]) -> matplotlib.figure.Figure:
    counts: Counter = Counter(
        entry["axiom_applied"]
        for r in results
        for entry in r.trace
    )
    fig, ax = plt.subplots()
    if counts:
        ax.bar(list(counts.keys()), list(counts.values()))
        ax.set_ylabel("Invocations")
        ax.set_title("Axiom Invocation Frequency")
        plt.xticks(rotation=30, ha="right")
    return fig
