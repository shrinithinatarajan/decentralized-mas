from src.agents.base_agent import BaseAgent
from src.schemas.axiom_rules import SILENCING_THRESHOLD

_SYSTEM = f"""You are the Transcriptomics Agent in a multi-agent cancer drug resistance framework.
Your modality: mRNA expression (z-scores, TPM).
Apply the T2_TRANSCRIPTIONAL_GATE axiom: a gene with z-score < {SILENCING_THRESHOLD} is functionally absent.

REASONING PROTOCOL — fill the 'reasoning' field step by step before deciding the verdict:
1. What expression levels did MCP return for the drug target gene(s)? State each gene's z-score.
2. High expression (z > 1.5) of a drug target oncogene → strong SENSITIVE signal.
3. Silenced expression (z < {SILENCING_THRESHOLD}) of a drug target → gene is functionally absent;
   targeted drugs are unlikely to work → RESISTANT signal.
4. Moderate expression (z between {SILENCING_THRESHOLD} and 1.5) → UNCERTAIN unless other evidence clarifies.
5. If no expression data exists for the target gene, state that and lean UNCERTAIN.
6. Conclude with your verdict and why.

Return ONLY a JSON object matching the EvidencePack schema. No prose outside the JSON."""


class TranscriptomicsAgent(BaseAgent):
    agent_id = "transcriptomics_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        if target_genes:
            expression = []
            for gene in target_genes:
                expression += await self._call_tool("get_expression", {"cell_line": cell_line, "gene": gene}) or []
        else:
            expression = await self._call_tool("get_high_expression_genes", {"cell_line": cell_line, "z_threshold": 1.0}) or []
        return {"expression": expression}

    def _compute_signal(self, evidence: dict) -> float:
        expression = evidence.get("expression", [])
        if not expression:
            return 0.0
        z_scores = [abs(e.get("z_score") or 0.0) for e in expression]
        return min(max(z_scores) / 3.0, 1.0)
