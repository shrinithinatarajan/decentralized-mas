from src.agents.base_agent import BaseAgent
from src.schemas.axiom_rules import SILENCING_THRESHOLD

_SYSTEM = f"""You are the Transcriptomics Agent in a multi-agent cancer drug resistance framework.
Your modality: mRNA expression (z-scores, TPM).
Apply the T2_TRANSCRIPTIONAL_GATE axiom: a gene with z-score < {SILENCING_THRESHOLD} is functionally absent.
Return ONLY a JSON object matching the EvidencePack schema. No prose."""


class TranscriptomicsAgent(BaseAgent):
    agent_id = "transcriptomics_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str) -> dict:
        expression = await self._call_tool("get_expression", {"cell_line": cell_line})
        return {"expression": expression}
