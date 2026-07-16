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

    async def analyze(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> EvidencePack:
        evidence = await self._fetch_evidence(cell_line, drug, target_genes)
        prompt = self._build_prompt(cell_line, drug, evidence)
        raw = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
        )
        pack = self._parse_pack(raw, cell_line, drug)
        pack.confidence = compute_gcs(
            pack.key_findings, evidence, pack.evidence_tier, self._compute_signal(evidence)
        )
        return pack

    async def critique(
        self,
        original_pack: EvidencePack,
        peer_packs: list[EvidencePack],
    ) -> EvidencePack:
        """Round-2 peer review: score peers' reasoning quality; verdict is LOCKED from round 1."""
        labels = [chr(ord('A') + i) for i in range(len(peer_packs))]
        peer_text = ""
        for label, peer in zip(labels, peer_packs):
            peer_text += (
                f"\n--- Peer {label} ---\n"
                f"Verdict: {peer.verdict.value}  Confidence: {peer.confidence:.2f}\n"
                f"Reasoning: {peer.reasoning}\n"
                f"Key findings: {json.dumps([f.model_dump() for f in peer.key_findings])}\n"
            )

        scores_schema = ", ".join(f'"peer_{l}": <int 1-5>' for l in labels)
        prompt = (
            f"Cell line: {original_pack.cell_line}\nDrug: {original_pack.drug}\n\n"
            f"YOUR VERDICT (LOCKED — do NOT change): {original_pack.verdict.value}\n"
            f"Your confidence: {original_pack.confidence:.2f}\n"
            f"Your reasoning: {original_pack.reasoning}\n\n"
            f"PEER ANALYSES (identities anonymized):\n{peer_text}\n"
            "TASK: You are a scientific peer reviewer. Rate each peer's reasoning quality\n"
            "on a scale of 1-5 (1=unsupported/flawed, 5=rigorous/well-evidenced).\n"
            "Criteria: Is the evidence specific and data-driven? Does it address the drug's\n"
            "mechanism of action? Are there logical gaps or unsupported leaps?\n\n"
            "You may adjust your confidence by at most ±0.10 based on what peers found.\n"
            "Your verdict is FINAL — do not change SENSITIVE/RESISTANT/UNCERTAIN.\n\n"
            "Respond with ONLY this JSON (no prose, no markdown):\n"
            f'{{"verdict": "{original_pack.verdict.value}", '
            f'"confidence": <float>, '
            f'"peer_scores": {{{scores_schema}}}, '
            f'"peer_review_reasoning": "<one sentence per peer>"}}'  
        )
        raw = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
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
        raw_scores = data.get("peer_scores", {})
        peer_score_map = {
            labels[i]: float(raw_scores.get(f"peer_{labels[i]}", 3.0))
            for i in range(len(labels))
        }
        return original_pack.model_copy(update={
            "confidence": new_conf,
            "peer_scores": peer_score_map,
        })

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
        return json.loads(result.data)
