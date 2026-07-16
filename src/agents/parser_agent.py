"""ParserAgent — extract (cell_line, drug) from natural language queries.

Accepts free-text like "Will BT-483 respond to Foretinib?" and returns canonical
names matched against the pharmacology DB, ready to pass to the Orchestrator.
"""
import asyncio
import json
import re
import sqlite3
from pathlib import Path

from src.llm.client import LLMClient, make_rate_limiter

_DB_PHARM = Path("src/data/processed/pharmacology.db")

_SYSTEM = """You are a biomedical query parser. Extract the cell line name and drug name from the user's query.

Return ONLY a JSON object with exactly two keys:
  "cell_line": the cancer cell line name (e.g. "BT-483", "MCF7", "A375")
  "drug": the drug or compound name (e.g. "Foretinib", "Imatinib", "Dabrafenib")

If either is absent or ambiguous, set its value to null.
No prose, no markdown — JSON only."""


def _load_catalog() -> tuple[list[str], list[str]]:
    """Return (cell_lines, drugs) from the pharmacology DB."""
    if not _DB_PHARM.exists():
        return [], []
    conn = sqlite3.connect(_DB_PHARM)
    cls  = [r[0] for r in conn.execute("SELECT DISTINCT cell_line FROM drug_response").fetchall()]
    drgs = [r[0] for r in conn.execute("SELECT DISTINCT drug FROM drug_info").fetchall()]
    conn.close()
    return cls, drgs


def _strip(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _fuzzy_match(raw: str, catalog: list[str]) -> str | None:
    """Return the closest catalog entry or None if no reasonable match."""
    if not raw or not catalog:
        return None
    # Exact match (case-insensitive)
    lower = raw.lower()
    for c in catalog:
        if c.lower() == lower:
            return c
    # Stripped match (ignore punctuation)
    s_raw = _strip(raw)
    by_stripped = {_strip(c): c for c in catalog}
    if s_raw in by_stripped:
        return by_stripped[s_raw]
    # Prefix / contains match
    candidates = [c for c in catalog if _strip(c).startswith(s_raw) or s_raw.startswith(_strip(c))]
    if len(candidates) == 1:
        return candidates[0]
    # Substring match (for short queries)
    if len(s_raw) >= 4:
        sub = [c for c in catalog if s_raw in _strip(c) or _strip(c) in s_raw]
        if len(sub) == 1:
            return sub[0]
    return None


class ParserAgent:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client or LLMClient()

    async def parse(self, query: str) -> dict:
        """Parse a natural language query.

        Returns:
            {"cell_line": str, "drug": str, "cell_line_matched": bool, "drug_matched": bool,
             "raw_cell_line": str, "raw_drug": str}
        """
        raw = await self.llm.complete(
            messages=[{"role": "user", "content": query}],
            system=_SYSTEM,
        )
        extracted = self._parse_json(raw)
        raw_cl   = extracted.get("cell_line") or ""
        raw_drug = extracted.get("drug") or ""

        cell_lines, drugs = _load_catalog()
        matched_cl   = _fuzzy_match(raw_cl,   cell_lines)
        matched_drug = _fuzzy_match(raw_drug, drugs)

        return {
            "cell_line":        matched_cl or raw_cl,
            "drug":             matched_drug or raw_drug,
            "cell_line_matched": matched_cl is not None,
            "drug_matched":      matched_drug is not None,
            "raw_cell_line":    raw_cl,
            "raw_drug":         raw_drug,
        }

    @staticmethod
    def _parse_json(raw: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            for m in re.finditer(r"\{.*?\}", raw, re.DOTALL):
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    continue
        return {}
