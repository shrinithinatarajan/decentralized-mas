import matplotlib
matplotlib.use("Agg")  # no display needed in tests
import matplotlib.figure
import pytest

from src.evaluation.metrics import EvaluationMetrics
from src.protocols.debate_engine import ConsensusResult
from src.schemas.evidence_pack import Verdict
from src.evaluation.visualize import (
    plot_roc_curve,
    plot_pr_curve,
    plot_ablation_comparison,
    plot_debate_convergence,
    plot_axiom_frequency,
)


def _metrics(auroc=0.8, auprc=0.7, rho=0.6, kappa=0.5, n=20):
    return EvaluationMetrics(auroc=auroc, auprc=auprc, spearman_rho=rho, cohens_kappa=kappa, n_evaluated=n)


def _result(verdict="SENSITIVE", rounds=1, forced=False, trace=None):
    return ConsensusResult(
        final_verdict=Verdict(verdict),
        final_confidence=0.8,
        cell_line="A375",
        drug="Vemurafenib",
        winning_agent="genomics_agent",
        rounds_taken=rounds,
        forced=forced,
        dissenting_agents=[],
        trace=trace or [],
    )


# --- plot_roc_curve ---

def test_plot_roc_curve_returns_figure():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.8, 0.2, 0.1]
    fig = plot_roc_curve(y_true, y_score)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_roc_curve_has_one_axes():
    fig = plot_roc_curve([1, 0], [0.9, 0.1])
    assert len(fig.axes) == 1


def test_plot_roc_curve_axes_labelled():
    fig = plot_roc_curve([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    ax = fig.axes[0]
    assert "False Positive Rate" in ax.get_xlabel()
    assert "True Positive Rate" in ax.get_ylabel()


def test_plot_roc_curve_has_diagonal_reference():
    # Diagonal chance line + the ROC curve = at least 2 lines
    fig = plot_roc_curve([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    ax = fig.axes[0]
    assert len(ax.lines) >= 2


# --- plot_pr_curve ---

def test_plot_pr_curve_returns_figure():
    fig = plot_pr_curve([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_pr_curve_axes_labelled():
    fig = plot_pr_curve([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    ax = fig.axes[0]
    assert "Recall" in ax.get_xlabel()
    assert "Precision" in ax.get_ylabel()


# --- plot_ablation_comparison ---

def test_plot_ablation_comparison_returns_figure():
    data = {
        "Framework": _metrics(auroc=0.82),
        "No Debate": _metrics(auroc=0.71),
        "No Axioms": _metrics(auroc=0.68),
    }
    fig = plot_ablation_comparison(data)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_ablation_comparison_bar_count_matches_variants():
    data = {
        "Framework": _metrics(auroc=0.82),
        "No Debate": _metrics(auroc=0.71),
        "No Axioms": _metrics(auroc=0.68),
    }
    fig = plot_ablation_comparison(data)
    ax = fig.axes[0]
    assert len(ax.patches) == 3


def test_plot_ablation_comparison_ylabel_is_auroc():
    data = {"A": _metrics(), "B": _metrics()}
    fig = plot_ablation_comparison(data)
    assert "AUROC" in fig.axes[0].get_ylabel()


# --- plot_debate_convergence ---

def test_plot_debate_convergence_returns_figure():
    results = [_result(rounds=1), _result(rounds=2), _result(rounds=1), _result(rounds=3)]
    fig = plot_debate_convergence(results)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_debate_convergence_xlabel_is_rounds():
    results = [_result(rounds=1), _result(rounds=2)]
    fig = plot_debate_convergence(results)
    assert "Round" in fig.axes[0].get_xlabel()


# --- plot_axiom_frequency ---

def test_plot_axiom_frequency_returns_figure():
    results = [
        _result(trace=[{"round": 1, "axiom_applied": "T1_STRUCTURAL", "winning_agent": "g", "verdict": "SENSITIVE"}]),
        _result(trace=[{"round": 1, "axiom_applied": "T2_TRANSCRIPTIONAL_GATE", "winning_agent": "t", "verdict": "RESISTANT"}]),
        _result(trace=[{"round": 1, "axiom_applied": "T1_STRUCTURAL", "winning_agent": "g", "verdict": "SENSITIVE"}]),
    ]
    fig = plot_axiom_frequency(results)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_axiom_frequency_bar_count_matches_unique_axioms():
    results = [
        _result(trace=[{"round": 1, "axiom_applied": "T1_STRUCTURAL", "winning_agent": "g", "verdict": "S"}]),
        _result(trace=[{"round": 1, "axiom_applied": "T2_TRANSCRIPTIONAL_GATE", "winning_agent": "t", "verdict": "R"}]),
        _result(trace=[{"round": 1, "axiom_applied": "T1_STRUCTURAL", "winning_agent": "g", "verdict": "S"}]),
    ]
    fig = plot_axiom_frequency(results)
    ax = fig.axes[0]
    assert len(ax.patches) == 2  # T1 and T2 only


def test_plot_axiom_frequency_no_trace_returns_empty_figure():
    results = [_result(trace=[]), _result(trace=[])]
    fig = plot_axiom_frequency(results)
    assert isinstance(fig, matplotlib.figure.Figure)
