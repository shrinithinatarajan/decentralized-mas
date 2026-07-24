import pytest
from src.protocols.debate_engine import ConsensusResult
from src.schemas.evidence_pack import Verdict
from src.data.loader import Case
from src.evaluation.metrics import EvaluationMetrics, evaluate


def _result(cell_line, drug, verdict, confidence):
    return ConsensusResult(
        final_verdict=Verdict(verdict),
        final_confidence=confidence,
        cell_line=cell_line,
        drug=drug,
        winning_agent="genomics_agent",
        rounds_taken=1,
        forced=False,
        dissenting_agents=[],
    )


def _case(cell_line, drug, label):
    return Case(cell_line=cell_line, drug=drug, label=label)


def test_evaluate_returns_evaluation_metrics():
    results = [
        _result("A375", "Vemurafenib", "SENSITIVE", 0.9),
        _result("HCT116", "Oxaliplatin", "RESISTANT", 0.8),
    ]
    cases = [
        _case("A375", "Vemurafenib", "SENSITIVE"),
        _case("HCT116", "Oxaliplatin", "RESISTANT"),
    ]
    metrics = evaluate(results, cases)
    assert isinstance(metrics, EvaluationMetrics)


def test_evaluate_perfect_predictions_auroc_1():
    results = [
        _result("A", "D1", "SENSITIVE", 0.9),
        _result("B", "D2", "RESISTANT", 0.9),
        _result("C", "D3", "SENSITIVE", 0.8),
        _result("D", "D4", "RESISTANT", 0.8),
    ]
    cases = [
        _case("A", "D1", "SENSITIVE"),
        _case("B", "D2", "RESISTANT"),
        _case("C", "D3", "SENSITIVE"),
        _case("D", "D4", "RESISTANT"),
    ]
    metrics = evaluate(results, cases)
    assert metrics.auroc == pytest.approx(1.0)


def test_evaluate_perfect_predictions_auprc_1():
    results = [
        _result("A", "D1", "SENSITIVE", 0.9),
        _result("B", "D2", "RESISTANT", 0.9),
        _result("C", "D3", "SENSITIVE", 0.8),
        _result("D", "D4", "RESISTANT", 0.8),
    ]
    cases = [
        _case("A", "D1", "SENSITIVE"),
        _case("B", "D2", "RESISTANT"),
        _case("C", "D3", "SENSITIVE"),
        _case("D", "D4", "RESISTANT"),
    ]
    metrics = evaluate(results, cases)
    assert metrics.auprc == pytest.approx(1.0)


def test_evaluate_perfect_agreement_kappa_1():
    results = [
        _result("A", "D1", "SENSITIVE", 0.9),
        _result("B", "D2", "RESISTANT", 0.8),
    ]
    cases = [
        _case("A", "D1", "SENSITIVE"),
        _case("B", "D2", "RESISTANT"),
    ]
    metrics = evaluate(results, cases)
    assert metrics.cohens_kappa == pytest.approx(1.0)


def test_evaluate_n_evaluated_counts_matched_cases():
    results = [
        _result("A", "D1", "SENSITIVE", 0.9),
        _result("B", "D2", "RESISTANT", 0.8),
        _result("C", "D3", "SENSITIVE", 0.7),
    ]
    cases = [
        _case("A", "D1", "SENSITIVE"),
        _case("B", "D2", "RESISTANT"),
        _case("C", "D3", "SENSITIVE"),
    ]
    metrics = evaluate(results, cases)
    assert metrics.n_total == 3


def test_evaluate_spearman_rho_monotone_near_1():
    # Confident SENSITIVE predictions paired with SENSITIVE labels → high rank correlation
    results = [
        _result("A", "D1", "SENSITIVE", 0.95),
        _result("B", "D2", "SENSITIVE", 0.85),
        _result("C", "D3", "RESISTANT", 0.85),
        _result("D", "D4", "RESISTANT", 0.95),
    ]
    cases = [
        _case("A", "D1", "SENSITIVE"),
        _case("B", "D2", "SENSITIVE"),
        _case("C", "D3", "RESISTANT"),
        _case("D", "D4", "RESISTANT"),
    ]
    metrics = evaluate(results, cases)
    assert metrics.spearman_rho > 0.5


def test_evaluate_uncertain_excluded_from_binary_metrics():
    # UNCERTAIN predictions are excluded from AUROC/AUPRC/Spearman
    results = [
        _result("A", "D1", "SENSITIVE", 0.9),
        _result("B", "D2", "RESISTANT", 0.9),
        _result("C", "D3", "UNCERTAIN", 0.5),
    ]
    cases = [
        _case("A", "D1", "SENSITIVE"),
        _case("B", "D2", "RESISTANT"),
        _case("C", "D3", "SENSITIVE"),
    ]
    metrics = evaluate(results, cases)
    # UNCERTAIN result excluded from binary metrics — n_total still counts all matched
    assert metrics.n_total == 3
    assert metrics.n_decisive == 2
    assert metrics.auroc == pytest.approx(1.0)  # only 2 non-uncertain used


def test_evaluate_unmatched_results_ignored():
    # Results with no matching ground-truth case are silently dropped
    results = [
        _result("A", "D1", "SENSITIVE", 0.9),
        _result("UNKNOWN", "D99", "SENSITIVE", 0.8),
    ]
    cases = [_case("A", "D1", "SENSITIVE")]
    metrics = evaluate(results, cases)
    assert metrics.n_total == 1
