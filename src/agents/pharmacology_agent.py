from src.agents.base_agent import BaseAgent

_SYSTEM = """You are the Pharmacology Agent in a multi-agent cancer drug resistance framework.
Your modality: historical IC50 data and drug target information.
Apply the T4_PHARMACOLOGICAL_PRIOR axiom: IC50 provides a population-level prior that can be
overridden by higher-tier molecular evidence (T1-T3).

REASONING PROTOCOL — fill the 'reasoning' field step by step before deciding the verdict:
1. What did the IC50 data return? State the ln_ic50, z_score, and label field explicitly.
2. The 'label' field is pre-computed from z_score: SENSITIVE (z < -0.5) or RESISTANT (z > 0.5).
   Anchor your verdict to this label — it is your primary evidence.
3. If IC50 label is SENSITIVE: vote SENSITIVE unless strong molecular evidence (T1-T3) says otherwise.
4. If IC50 label is RESISTANT: vote RESISTANT as prior, but note higher-tier agents may override.
5. If no IC50 data: state that and vote UNCERTAIN.
6. Conclude with your verdict and why.

Return ONLY a JSON object matching the EvidencePack schema. No prose outside the JSON."""


class PharmacologyAgent(BaseAgent):
    agent_id = "pharmacology_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        ic50 = await self._call_tool("get_ic50", {"cell_line": cell_line, "drug": drug})
        drug_info = await self._call_tool("get_drug_info", {"drug": drug})
        return {"ic50": ic50, "drug_info": drug_info}

    def _compute_signal(self, evidence: dict) -> float:
        ic50 = evidence.get("ic50") or {}
        z_score = ic50.get("z_score", 0.0) if isinstance(ic50, dict) else 0.0
        return min(abs(z_score) / 3.0, 1.0)
