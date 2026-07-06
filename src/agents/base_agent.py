import json
import re
from abc import ABC, abstractmethod

from fastmcp import Client

from src.llm.client import LLMClient
from src.schemas.evidence_pack import EvidencePack, EvidenceTier, Verdict

_SCHEMA_EXAMPLE = json.dumps({
    "agent_id": "<your_agent_id e.g. genomics_agent>",
    "cell_line": "<cell_line name>",
    "drug": "<drug name>",
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
        return self._parse_pack(raw, cell_line, drug)

    @abstractmethod
    async def _fetch_evidence(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> dict:
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    def _build_prompt(self, cell_line: str, drug: str, evidence: dict) -> str:
        return (
            f"Cell line: {cell_line}\nDrug: {drug}\n\n"
            f"Evidence data:\n{json.dumps(evidence, indent=2)}\n\n"
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
                return EvidencePack.model_validate(json.loads(candidate))
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
