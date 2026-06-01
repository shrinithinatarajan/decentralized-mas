from src.agents.base_agent import BaseAgent

_PATHWAYS = ["hsa04010", "hsa04012", "hsa04151", "hsa04115", "hsa04310"]

_SYSTEM = """You are the Pathway Agent in a multi-agent cancer drug resistance framework.
Your modality: signalling pathway topology, bypass routes, and upstream regulators.
Apply the T3_PATHWAY_BYPASS axiom: an active bypass route for the drug target confers resistance
regardless of the target's own mutation status.
Return ONLY a JSON object matching the EvidencePack schema. No prose."""


class PathwayAgent(BaseAgent):
    agent_id = "pathway_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str) -> dict:
        pathway_data: dict[str, dict] = {}
        for pid in _PATHWAYS:
            genes = await self._call_tool("get_pathway_genes", {"pathway_id": pid})
            if genes:
                pathway_data[pid] = {"genes": genes}
        return {"pathways": pathway_data}
