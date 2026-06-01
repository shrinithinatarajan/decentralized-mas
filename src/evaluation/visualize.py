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
    aurocs = [m.auroc for m in metrics.values()]
    fig, ax = plt.subplots()
    ax.bar(labels, aurocs)
    ax.set_ylabel("AUROC")
    ax.set_title("Ablation Comparison")
    ax.set_ylim(0, 1)
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
