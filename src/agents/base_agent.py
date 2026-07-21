import json
import re
from abc import ABC, abstractmethod

from fastmcp import Client

from src.agents.gcs import compute_gcs
from src.llm.client import LLMClient
from src.schemas.evidence_pack import EvidencePack, EvidenceTier, Verdict

def _truncate_evidence(evidence: dict, max_items: int = 20) -> dict:
    result = {}
    for k, v in evidence.items():
        if isinstance(v, list):
            result[k] = v[:max_items]
        elif isinstance(v, dict):
            result[k] = _truncate_evidence(v, max_items)
        else:
            result[k] = v
    return result


_SCHEMA_EXAMPLE = json.dumps({
    "agent_id": "<your_agent_id e.g. genomics_agent>",
    "cell_line": "<cell_line name>",
    "drug": "<drug name>",
    "reasoning": "<step-by-step: (1) what evidence did MCP return? (2) what does it imply for drug sensitivity? (3) are there contradictions? (4) final verdict rationale>",
    "verdict": "SENSITIVE or RESISTANT or UNCERTAIN",
    "confidence": 0.85,
    "evidence_tier": "T1_STRUCTURAL or T2_TRANSCRIPTIONAL or T3_PATHWAY or T4_PHARMACOLOGICAL or T5_STATISTICAL",
    "key_findings": [
        {"biomarker": "GENE", "value": "alteration", "interpretation": "biological effect", "data_source": "db_name"}
    ],
    "caveats": [],
    "conflict_flags": []
})


class BaseAgent(ABC):
    agent_id: str

    def __init__(self, mcp_app, llm_client: LLMClient | None = None) -> None:
        self.mcp_app = mcp_app
        self.llm = llm_client or LLMClient()

    async def analyze(
        self,
        cell_line: str,
        drug: str,
        target_genes: list[str] | None = None,
        *,
        run_logger=None,
        case_id: str | None = None,
    ) -> EvidencePack:
        self._run_logger = run_logger
        self._case_id = case_id
        evidence = await self._fetch_evidence(cell_line, drug, target_genes)
        prompt = self._build_prompt(cell_line, drug, evidence)
        raw = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            run_logger=run_logger,
            agent_id=self.agent_id,
            case_id=case_id,
        )
        pack = self._parse_pack(raw, cell_line, drug)
        pack.confidence = compute_gcs(
            pack.key_findings, evidence, pack.evidence_tier, self._compute_signal(evidence)
        )
        if pack.verdict == Verdict.UNCERTAIN:
            pack.confidence = min(pack.confidence, 0.5)
        pack = pack.model_copy(update={"data_status": evidence.get("data_status")})
        if run_logger:
            run_logger.log_agent_decision(
                case_id=case_id, agent_id=self.agent_id, round_num=1,
                verdict=pack.verdict.value, confidence=pack.confidence, reasoning=pack.reasoning,
            )
        return pack

    async def critique(
        self,
        original_pack: EvidencePack,
        peer_packs: list[EvidencePack],
        *,
        run_logger=None,
        case_id: str | None = None,
    ) -> EvidencePack:
        """Round-2 peer review: verify peers' factual claims via binary checklist; verdict LOCKED."""
        labels = [chr(ord('A') + i) for i in range(len(peer_packs))]

        CHECKLISTS = {
            "T1_STRUCTURAL": [
                "Did this peer name a specific mutation or CNV in the drug target gene (not just 'no mutation found')?",
                "Did this peer cite a CIViC/OncoKB entry or known functional consequence linking the variant to drug response?",
                "Did this peer confirm the variant is somatic (not germline or unclassified)?",
                "Does the variant's functional direction (gain-of-function / loss-of-function) logically support the verdict given?",
            ],
            "T2_TRANSCRIPTIONAL": [
                "Did this peer report a specific numeric z-score for the target gene expression?",
                "Is the expression level (overexpressed / silenced / normal) consistent with the verdict direction?",
                "Did this peer explicitly account for whether this drug is expression-driven vs mutation-driven?",
                "Did this peer name at least one specific gene with its z-score value (not just 'expression is high')?",
            ],
            "T3_PATHWAY": [
                "Did this peer name at least one specific bypass gene (not just 'bypass_exists=true')?",
                "Did this peer report the pathway activity score or fraction of expressed pathway genes?",
                "Does the bypass gene mechanistically connect to resistance for this specific drug class?",
                "Is the bypass gene distinct from the primary drug target gene?",
            ],
            "T4_PHARMACOLOGICAL": [
                "Did this peer report a specific IC50 z-score value (not just 'no data' or 'uncertain')?",
                "Did this peer explicitly state the pre-computed IC50 label (SENSITIVE/RESISTANT/UNCERTAIN)?",
                "Does the historical IC50 label match or logically support the verdict given?",
                "Did this peer acknowledge that T4 pharmacological evidence can be overridden by T1-T3 molecular evidence?",
            ],
        }

        peer_text = ""
        for label, peer in zip(labels, peer_packs):
            tier_key = peer.evidence_tier.value if hasattr(peer.evidence_tier, "value") else str(peer.evidence_tier)
            checklist = CHECKLISTS.get(tier_key, CHECKLISTS["T4_PHARMACOLOGICAL"])
            checklist_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(checklist))
            peer_text += (
                f"\n--- Peer {label} (tier: {tier_key}) ---\n"
                f"Verdict: {peer.verdict.value}  Confidence: {peer.confidence:.2f}\n"
                f"Reasoning: {peer.reasoning}\n"
                f"Key findings: {json.dumps([f.model_dump() for f in peer.key_findings])}\n"
                f"Checklist to verify (answer yes/no for each based on the reasoning above):\n{checklist_str}\n"
            )

        peer_schema_parts = []
        for label in labels:
            peer_schema_parts.append(
                f'"peer_{label}_checks": [<bool q1>, <bool q2>, <bool q3>, <bool q4>], '
                f'"peer_{label}_score": <float 0.0-1.0 = sum(checks)/4>'
            )
        peer_schema = ", ".join(peer_schema_parts)

        prompt = (
            f"Cell line: {original_pack.cell_line}\nDrug: {original_pack.drug}\n\n"
            f"YOUR VERDICT (LOCKED — do NOT change): {original_pack.verdict.value}\n"
            f"Your confidence: {original_pack.confidence:.2f}\n"
            f"Your reasoning: {original_pack.reasoning}\n\n"
            f"PEER ANALYSES TO REVIEW:\n{peer_text}\n"
            "TASK: For each peer, answer the checklist questions (true/false) based strictly on "
            "what their reasoning text actually states — not what you think is biologically correct. "
            "A claim scores true only if the peer explicitly made it in their reasoning.\n"
            "Score = number of true answers / 4.\n\n"
            "You may adjust your own confidence by at most ±0.10 based on what peers found.\n"
            "Your verdict is FINAL — do not change it.\n\n"
            "Respond with ONLY this JSON (no prose, no markdown):\n"
            f'{{"verdict": "{original_pack.verdict.value}", '
            f'"confidence": <float>, '
            f'{peer_schema}, '
            f'"peer_review_reasoning": "<one sentence per peer explaining scores>"}}'
        )
        raw = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            run_logger=run_logger,
            agent_id=self.agent_id,
            case_id=case_id,
        )
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        data: dict = {}
        for candidate in [cleaned] + [m.group(0) for m in re.finditer(r"\{.*?\}", raw, re.DOTALL)]:
            try:
                data = json.loads(candidate)
                break
            except Exception:
                continue
        new_conf = float(data.get("confidence", original_pack.confidence))
        new_conf = max(0.0, min(1.0, new_conf))
        if original_pack.verdict == Verdict.UNCERTAIN:
            new_conf = min(new_conf, 0.5)
        peer_score_map = {}
        for label in labels:
            score = data.get(f"peer_{label}_score")
            if score is not None:
                peer_score_map[label] = float(score)
            else:
                checks = data.get(f"peer_{label}_checks", [])
                if checks:
                    peer_score_map[label] = sum(1 for c in checks if c) / 4.0
                else:
                    peer_score_map[label] = 0.5
        critiqued = original_pack.model_copy(update={
            "confidence": new_conf,
            "peer_scores": peer_score_map,
        })
        if run_logger:
            run_logger.log_agent_decision(
                case_id=case_id, agent_id=self.agent_id, round_num=2,
                verdict=critiqued.verdict.value, confidence=critiqued.confidence,
                reasoning=data.get("peer_review_reasoning", ""),
            )
        return critiqued

    @abstractmethod
    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        ...

    @abstractmethod
    def _compute_signal(self, evidence: dict) -> float:
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    def _build_prompt(self, cell_line: str, drug: str, evidence: dict) -> str:
        return (
            f"Cell line: {cell_line}\nDrug: {drug}\n\n"
            f"Evidence data:\n{json.dumps(_truncate_evidence(evidence), indent=2)}\n\n"
            f"Respond with ONLY a JSON object in this exact format (no prose, no markdown):\n{_SCHEMA_EXAMPLE}"
        )

    def _parse_pack(self, raw: str, cell_line: str = "unknown", drug: str = "unknown") -> EvidencePack:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        candidates = [cleaned] if cleaned else []
        for m in re.finditer(r"\{.*?\}", raw, re.DOTALL):
            candidates.append(m.group(0))
        for candidate in candidates:
            try:
                data = json.loads(candidate)
                pack = EvidencePack.model_validate(data)
                pack.reasoning = data.get("reasoning", "")
                if "self_attestation" in data:
                    pack.self_attestation = data["self_attestation"]
                return pack
            except Exception:
                continue
        return EvidencePack(
            agent_id=self.agent_id,
            cell_line=cell_line,
            drug=drug,
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            evidence_tier=EvidenceTier.T5_STATISTICAL,
            key_findings=[],
            caveats=[f"parse_failed: {raw[:200]}"],
        )

    async def _call_tool(self, tool: str, args: dict) -> list | dict | None:
        async with Client(self.mcp_app) as client:
            result = await client.call_tool(tool, args)
        data = json.loads(result.data)
        run_logger = getattr(self, "_run_logger", None)
        if run_logger:
            run_logger.log_tool_call(
                case_id=getattr(self, "_case_id", None), agent_id=self.agent_id,
                tool=tool, args=args, result=data,
            )
        return data
