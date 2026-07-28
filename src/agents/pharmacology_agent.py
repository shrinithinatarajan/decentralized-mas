from src.agents.base_agent import BaseAgent

_SYSTEM = """You are the Pharmacology Agent in a multi-agent cancer drug resistance framework.
Your modality: historical IC50 data and drug target information.
Apply the T4_PHARMACOLOGICAL_PRIOR axiom: IC50 provides a population-level prior that can be
overridden by higher-tier molecular evidence (T1-T3).

REASONING PROTOCOL — fill the 'reasoning' field step by step before deciding the verdict:
1. What did the IC50 data return? State the ln_ic50, z_score, auc, and label field explicitly.
2. The 'label' field is pre-computed from z_score: SENSITIVE (z < -0.5) or RESISTANT (z > 0.5).
   Use both z_score AND auc to form your verdict:
   - z_score measures selective sensitivity relative to other cell lines for this drug.
   - auc measures absolute response: low AUC (< 0.8) = strong dose-response curve = drug is working.
   - When both agree: high confidence. When they conflict: flag the discrepancy and moderate confidence.
3. Conflict resolution when z_score and auc disagree:
   - z_score SENSITIVE (z < -0.5) but auc HIGH (> 0.9): the cell line is relatively sensitive but
     the absolute response curve is weak. This is a borderline case — cap confidence at 0.60.
   - z_score RESISTANT (z > 0.5) but auc LOW (< 0.7): drug has activity in absolute terms despite
     population-relative resistance. Do not dismiss — note the conflict and moderate toward UNCERTAIN.
4. If IC50 label is SENSITIVE and auc < 0.8: vote SENSITIVE with higher confidence (dual signal).
5. If IC50 label is RESISTANT and auc > 0.85: vote RESISTANT with higher confidence (dual signal).
6. If no IC50 data: state that and vote UNCERTAIN.
7. Conclude with your verdict and why, citing both z_score and auc values.

CONFIDENCE CALIBRATION — apply these caps before reporting confidence:
- z_score between -0.75 and +0.75 AND auc between 0.8 and 0.9 (both borderline): cap at 0.60.
- z_score between -1.5 and -0.75 or between +0.75 and +1.5 (moderate z): cap at 0.72.
- Only exceed confidence 0.75 when |z_score| > 1.5 OR auc < 0.75 (strong absolute response).
- If no IC50 data: confidence must be ≤ 0.50.

Return ONLY a JSON object matching the EvidencePack schema. No prose outside the JSON."""


class PharmacologyAgent(BaseAgent):
    agent_id = "pharmacology_agent"

    @property
    def system_prompt(self) -> str:
        return _SYSTEM

    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        ic50 = await self._call_tool("get_ic50", {"cell_line": cell_line, "drug": drug})
        drug_info = await self._call_tool("get_drug_info", {"drug": drug})
        ic50_data = ic50 if isinstance(ic50, dict) else {}
        status = "data_found" if ic50_data and ic50_data.get("z_score") is not None else "cell_line_missing"
        return {"ic50": ic50, "drug_info": drug_info, "data_status": status}

    def _compute_signal(self, evidence: dict) -> float:
        ic50 = evidence.get("ic50") or {}
        if not isinstance(ic50, dict):
            return 0.0
        z_score = ic50.get("z_score") or 0.0
        auc = ic50.get("auc") or 1.0
        z_signal = min(abs(z_score) / 3.0, 1.0)
        auc_signal = max(0.0, (1.0 - auc) / 0.5)  # auc=0.5 → 1.0, auc=1.0 → 0.0
        return max(z_signal, min(auc_signal, 1.0))
