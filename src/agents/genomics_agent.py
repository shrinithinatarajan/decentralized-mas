from src.agents.base_agent import BaseAgent

_SYSTEM = """You are the Genomics Agent in a multi-agent cancer drug resistance framework.
Your modality: somatic mutations and copy number variation (DNA level).
Apply the T1_STRUCTURAL axiom: structural DNA alterations are the highest-priority evidence.
Return ONLY a JSON object matching the EvidencePack schema. No prose."""


class GenomicsAgent(BaseAgent):
    agent_id = "genomics_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str) -> dict:
        mutations = await self._call_tool("get_mutations", {"cell_line": cell_line})
        cnv = await self._call_tool("get_cnv", {"cell_line": cell_line})
        return {"mutations": mutations, "cnv": cnv}
