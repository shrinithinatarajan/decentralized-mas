from dataclasses import dataclass

from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score
from scipy.stats import spearmanr

from src.data.loader import Case
from src.protocols.debate_engine import ConsensusResult
from src.schemas.evidence_pack import Verdict


@dataclass
class EvaluationMetrics:
    auroc: float
    auprc: float
    spearman_rho: float
    cohens_kappa: float
    n_evaluated: int


def _sensitivity_score(r: ConsensusResult) -> float:
    return r.final_confidence if r.final_verdict == Verdict.SENSITIVE else 1.0 - r.final_confidence


def evaluate(results: list[ConsensusResult], ground_truth: list[Case]) -> EvaluationMetrics:
    gt_map = {(c.cell_line, c.drug): c.label for c in ground_truth}

    matched = [(r, gt_map[key]) for r in results if (key := (r.cell_line, r.drug)) in gt_map]

    binary = [
        (r, lbl) for r, lbl in matched
        if r.final_verdict != Verdict.UNCERTAIN and lbl != "UNCERTAIN"
    ]

    scores = [_sensitivity_score(r) for r, _ in binary]
    labels = [1 if lbl == "SENSITIVE" else 0 for _, lbl in binary]
    unique_labels = set(labels)

    auroc = roc_auc_score(labels, scores) if len(unique_labels) == 2 else float("nan")
    auprc = average_precision_score(labels, scores) if len(unique_labels) == 2 else float("nan")
    spearman_rho = spearmanr(scores, labels).statistic if len(binary) >= 2 and len(unique_labels) == 2 else float("nan")

    pred_labels = [r.final_verdict.value for r, _ in matched]
    true_labels = [lbl for _, lbl in matched]
    all_labels = set(true_labels) | set(pred_labels)
    cohens_kappa = cohen_kappa_score(true_labels, pred_labels) if len(all_labels) >= 2 else float("nan")

    return EvaluationMetrics(
        auroc=auroc,
        auprc=auprc,
        spearman_rho=spearman_rho,
        cohens_kappa=cohens_kappa,
        n_evaluated=len(matched),
    )
