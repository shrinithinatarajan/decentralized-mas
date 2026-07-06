from src.agents.base_agent import BaseAgent

_SYSTEM = """You are the Pharmacology Agent in a multi-agent cancer drug resistance framework.
Your modality: historical IC50 data and drug target information.
Apply the T4_PHARMACOLOGICAL_PRIOR axiom: IC50 provides a population-level prior that can be
overridden by higher-tier molecular evidence (T1-T3).
Return ONLY a JSON object matching the EvidencePack schema. No prose."""


class PharmacologyAgent(BaseAgent):
    agent_id = "pharmacology_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        ic50 = await self._call_tool("get_ic50", {"cell_line": cell_line, "drug": drug})
        drug_info = await self._call_tool("get_drug_info", {"drug": drug})
        return {"ic50": ic50, "drug_info": drug_info}
