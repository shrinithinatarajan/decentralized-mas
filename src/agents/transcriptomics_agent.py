from src.agents.base_agent import BaseAgent
from src.schemas.axiom_rules import SILENCING_THRESHOLD

_SYSTEM = f"""You are the Transcriptomics Agent in a multi-agent cancer drug resistance framework.
Your modality: mRNA expression (z-scores, TPM) from CCLE.
Apply the T2_TRANSCRIPTIONAL_GATE axiom: a gene with z-score < {SILENCING_THRESHOLD} is functionally absent.

Step 0 — Identify the drug's sensitivity mechanism before interpreting expression:
  A. AMPLIFICATION/OVEREXPRESSION-driven (e.g. HER2 inhibitors, BCL2 inhibitors, CDK4/6 inhibitors):
     The drug works because the target protein is overproduced. Expression level of the target IS the signal.
  B. MUTATION-driven (e.g. BRAF V600E + vemurafenib, EGFR exon19del + erlotinib, BCR-ABL + dasatinib):
     The drug works because of a gain-of-function mutation, NOT because the gene is overexpressed.
     Expression level of the target gene is NOT predictive — look at downstream pathway activity instead.
  C. MECHANISM-AGNOSTIC (e.g. HDAC inhibitors, Aurora kinase inhibitors, DNA-damaging agents):
     Neither target expression nor pathway context reliably predicts sensitivity → lean UNCERTAIN.

REASONING PROTOCOL — fill the 'reasoning' field step by step:
1. What is the drug's mechanism type (A, B, or C above)? State your reasoning.
2. For type A: check target_expression z-scores.
   - z > 1.5 → SENSITIVE (target overexpressed, drug has an abundant target)
   - z < {SILENCING_THRESHOLD} → RESISTANT (target absent, drug has nothing to inhibit)
   - Otherwise → check high_expression_context for pathway co-activation before deciding
3. For type B: do NOT judge by target expression level alone.
   - Scan high_expression_context for downstream effectors or resistance markers.
   - If downstream oncogenic pathway genes are highly expressed → consistent with active mutant signalling → SENSITIVE.
   - If known resistance bypass genes are highly expressed (e.g. MET amplification for EGFR inhibitors,
     NRAS for BRAF inhibitors) → RESISTANT.
   - If neither pattern is clear → UNCERTAIN.
4. For type C: state that transcriptomics is not informative and vote UNCERTAIN.
5. Conclude with your verdict and why, referencing only the data provided.

Return ONLY a JSON object matching the EvidencePack schema. No prose outside the JSON."""


class TranscriptomicsAgent(BaseAgent):
    agent_id = "transcriptomics_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        target_expression = []
        if target_genes:
            for gene in target_genes:
                target_expression += await self._call_tool("get_expression", {"cell_line": cell_line, "gene": gene}) or []
        high_expr = await self._call_tool("get_high_expression_genes", {"cell_line": cell_line, "z_threshold": 1.5}) or []
        status = "cell_line_missing"
        if target_expression:
            z_vals = [abs(e.get("z_score") or 0.0) for e in target_expression]
            status = "data_found" if max(z_vals) >= 1.5 else "signal_below_threshold"
        return {"target_expression": target_expression, "high_expression_context": high_expr, "data_status": status}

    def _compute_signal(self, evidence: dict) -> float:
        target_expr = evidence.get("target_expression", [])
        if not target_expr:
            return 0.0
        z_scores = [abs(e.get("z_score") or 0.0) for e in target_expr]
        return min(max(z_scores) / 3.0, 1.0)
