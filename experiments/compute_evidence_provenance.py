"""Tag each of the 100 gold-standard cases as RETRIEVAL vs PIECING_TOGETHER
and compute AUROC separately for each bucket.

RETRIEVAL  — verdict is explained by a direct database hit:
  - Winning agent is genomics AND any key_finding has data_source == 'CIViC'
    with a civic_description mentioning the tested drug (direct lookup)
  - OR winning agent is pharmacology_agent (IC50 z-score lookup)
  - OR winning agent is genomics AND OncoKB highestSensitiveLevel /
    highestResistanceLevel was set (direct clinical annotation lookup)

PIECING_TOGETHER — no single direct hit; agent had to combine signals:
  - Wild-type gene + RPPA / Chronos / pathway bypass
  - Multiple modalities debated (RESOLVER_TIEBREAK or CONSENSUS_R2 where
    genomics did NOT have a direct CIViC or OncoKB hit)
  - Pathway agent or transcriptomics agent drove the verdict
"""
from __future__ import annotations
import json, re
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

TRACES = Path("experiments/results/traces_gold_standard.jsonl")
OUT    = Path("experiments/results/evidence_provenance.json")

LABEL_MAP = {"SENSITIVE": 1, "RESISTANT": 0}
SCORE_MAP  = {"SENSITIVE": 1.0, "RESISTANT": 0.0, "UNCERTAIN": 0.5}


def has_civic_hit(agent: dict, drug: str) -> bool:
    """True if any key_finding cites CIViC with a drug-specific description."""
    drug_lower = drug.lower().replace("-", "").replace(" ", "")
    for kf in agent.get("key_findings") or []:
        if (kf.get("data_source") or "").upper() == "CIVIC":
            return True
    # Also check raw_evidence mutations for civic_description mentioning this drug
    raw = agent.get("raw_evidence") or {}
    for mut in raw.get("mutations") or []:
        desc = (mut.get("civic_description") or "").lower().replace("-", "").replace(" ", "")
        if desc and drug_lower in desc:
            return True
    return False


def has_oncokb_decisive(agent: dict) -> bool:
    """True if OncoKB returned a drug-level sensitive/resistant annotation."""
    raw = agent.get("raw_evidence") or {}
    for ann in raw.get("oncokb_annotations") or []:
        if ann.get("highestSensitiveLevel") or ann.get("highestResistanceLevel"):
            return True
    return False


def classify_case(record: dict) -> str:
    drug = record.get("drug", "")
    winning_agent_id = record.get("winning_agent", "")
    r1_agents = record.get("r1_agents") or []
    agents_by_id = {a["agent_id"]: a for a in r1_agents if "agent_id" in a}

    # Pharmacology agent winning = direct IC50 retrieval
    if winning_agent_id == "pharmacology_agent":
        return "RETRIEVAL"

    # Genomics agent winning with direct CIViC or OncoKB hit = retrieval
    if winning_agent_id == "genomics_agent":
        gen = agents_by_id.get("genomics_agent", {})
        if has_civic_hit(gen, drug) or has_oncokb_decisive(gen):
            return "RETRIEVAL"

    # Everything else = piecing together
    return "PIECING_TOGETHER"


def compute_metrics(cases: list[dict]) -> dict:
    if len(cases) < 2:
        return {"n": len(cases), "note": "too few cases"}
    y_true, y_score = [], []
    for c in cases:
        label = LABEL_MAP.get(c["true_label"])
        score = SCORE_MAP.get(c["final_verdict"], 0.5)
        if label is None:
            continue
        y_true.append(label)
        y_score.append(score)
    if len(set(y_true)) < 2:
        return {"n": len(cases), "note": "only one class present"}
    correct = sum(
        1 for c in cases
        if c["final_verdict"] == c["true_label"]
    )
    return {
        "n": len(cases),
        "accuracy": round(correct / len(cases), 4),
        "auroc": round(float(roc_auc_score(y_true, y_score)), 4),
        "auprc": round(float(average_precision_score(y_true, y_score)), 4),
    }


def main():
    records = [json.loads(l) for l in TRACES.open() if l.strip()]
    print(f"Loaded {len(records)} traces")

    retrieval, piecing = [], []
    tags = []
    for r in records:
        tag = classify_case(r)
        tags.append({"case_id": r["case_id"], "cell_line": r["cell_line"],
                     "drug": r["drug"], "tag": tag,
                     "winning_agent": r.get("winning_agent"),
                     "resolution_method": r.get("resolution_method"),
                     "true_label": r["true_label"],
                     "final_verdict": r["final_verdict"],
                     "correct": r.get("correct")})
        (retrieval if tag == "RETRIEVAL" else piecing).append(r)

    ret_metrics  = compute_metrics(retrieval)
    piece_metrics = compute_metrics(piecing)

    print(f"\n{'='*60}")
    print(f"  RETRIEVAL cases     : {len(retrieval)}")
    print(f"  PIECING_TOGETHER    : {len(piecing)}")
    print(f"\n  RETRIEVAL  AUROC    : {ret_metrics.get('auroc', 'N/A')}")
    print(f"  RETRIEVAL  accuracy : {ret_metrics.get('accuracy', 'N/A')}")
    print(f"\n  PIECING    AUROC    : {piece_metrics.get('auroc', 'N/A')}")
    print(f"  PIECING    accuracy : {piece_metrics.get('accuracy', 'N/A')}")
    print(f"{'='*60}")

    out = {
        "retrieval": {"metrics": ret_metrics, "cases": [t for t in tags if t["tag"] == "RETRIEVAL"]},
        "piecing_together": {"metrics": piece_metrics, "cases": [t for t in tags if t["tag"] == "PIECING_TOGETHER"]},
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"  Output: {OUT}")


if __name__ == "__main__":
    main()
