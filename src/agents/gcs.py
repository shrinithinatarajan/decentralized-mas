import json
from src.schemas.evidence_pack import EvidenceTier, Finding

_TIER_WEIGHTS: dict[EvidenceTier, float] = {
    EvidenceTier.T1_STRUCTURAL: 1.0,
    EvidenceTier.T2_TRANSCRIPTIONAL: 0.8,
    EvidenceTier.T3_PATHWAY: 0.6,
    EvidenceTier.T4_PHARMACOLOGICAL: 0.4,
    EvidenceTier.T5_STATISTICAL: 0.2,
}


def tier_weight(tier: EvidenceTier) -> float:
    return _TIER_WEIGHTS[tier]


def compute_h(key_findings: list[Finding], evidence_raw: dict) -> float:
    """Fraction of cited biomarkers that appear in MCP evidence text."""
    if not key_findings:
        return 0.0
    evidence_text = json.dumps(evidence_raw).lower()
    found = sum(1 for f in key_findings if f.biomarker.lower() in evidence_text)
    return found / len(key_findings)


def _has_data(value) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        return any(_has_data(v) for v in value.values())
    return True


def compute_e(evidence_raw: dict) -> float:
    """1.0 if MCP returned any non-empty data point, else 0.0."""
    return 1.0 if _has_data(evidence_raw) else 0.0


def compute_gcs(
    key_findings_or_h: "list[Finding] | float",
    evidence_raw: dict,
    tier: EvidenceTier,
    signal_strength: float,
) -> float:
    # Accepts either a pre-computed h float (from agent _compute_h override)
    # or the raw key_findings list (legacy path, computes h internally)
    if isinstance(key_findings_or_h, float):
        h = max(0.0, min(1.0, key_findings_or_h))
    else:
        h = compute_h(key_findings_or_h, evidence_raw)
    e = compute_e(evidence_raw)
    t = tier_weight(tier)
    s = max(0.0, min(1.0, signal_strength))
    return round((h * 0.40) + (e * 0.30) + (t * 0.20) + (s * 0.10), 6)
