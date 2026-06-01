import json
import re
from abc import ABC, abstractmethod

from fastmcp import Client

from src.llm.client import LLMClient
from src.schemas.evidence_pack import EvidencePack


class BaseAgent(ABC):
    agent_id: str

    def __init__(self, mcp_app, llm_client: LLMClient | None = None) -> None:
        self.mcp_app = mcp_app
        self.llm = llm_client or LLMClient()

    async def analyze(self, cell_line: str, drug: str) -> EvidencePack:
        evidence = await self._fetch_evidence(cell_line, drug)
        prompt = self._build_prompt(cell_line, drug, evidence)
        raw = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
        )
        return self._parse_pack(raw)

    @abstractmethod
    async def _fetch_evidence(self, cell_line: str, drug: str) -> dict:
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    def _build_prompt(self, cell_line: str, drug: str, evidence: dict) -> str:
        return (
            f"Cell line: {cell_line}\nDrug: {drug}\n\n"
            f"Evidence data:\n{json.dumps(evidence, indent=2)}\n\n"
            "Respond with a single JSON object matching the EvidencePack schema."
        )

    def _parse_pack(self, raw: str) -> EvidencePack:
        # strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        return EvidencePack.model_validate(json.loads(cleaned))

    async def _call_tool(self, tool: str, args: dict) -> list | dict | None:
        async with Client(self.mcp_app) as client:
            result = await client.call_tool(tool, args)
        return json.loads(result.data)
