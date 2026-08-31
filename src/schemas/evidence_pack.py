from enum import Enum
from pydantic import BaseModel, Field


class Verdict(str, Enum):
    SENSITIVE = "SENSITIVE"
    RESISTANT = "RESISTANT"
    UNCERTAIN = "UNCERTAIN"


class EvidenceTier(str, Enum):
    T1_STRUCTURAL = "T1_STRUCTURAL"
    T2_TRANSCRIPTIONAL = "T2_TRANSCRIPTIONAL"
    T3_PATHWAY = "T3_PATHWAY"
    T4_PHARMACOLOGICAL = "T4_PHARMACOLOGICAL"
    T5_STATISTICAL = "T5_STATISTICAL"


class Finding(BaseModel):
    biomarker: str
    value: str
    interpretation: str
    data_source: str
    # Free-text label for the biological principle invoked (e.g. "CENTRAL_DOGMA_DNA_PRIMACY")
    # Not typed as AxiomTier because findings can invoke biological axioms not in the tier hierarchy
    axiom_invoked: str | None = None


class EvidencePack(BaseModel):
    agent_id: str
    cell_line: str
    drug: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_tier: EvidenceTier
    key_findings: list[Finding]
    reasoning: str = ""
    caveats: list[str] = []
    conflict_flags: list[str] = []
    peer_scores: dict = {}  # label ("A","B",...) -> quality score 0.0-1.0 given to that peer
    data_status: str | None = None  # "data_found"|"target_wild_type"|"signal_below_threshold"|"cell_line_missing"|"insufficient_evidence"
    self_attestation: dict | None = None  # T3 binary checklist answers + score
    raw_evidence: dict | None = None  # full MCP tool output, for faithfulness evaluation
