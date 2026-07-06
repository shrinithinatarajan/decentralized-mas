"""Print raw LLM responses for A375 + Dabrafenib from cache."""
import asyncio
from pathlib import Path
from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.llm.client import LLMClient
from src.mcp_servers.genomics_server import mcp as genomics_app
from src.mcp_servers.transcriptomics_server import mcp as transcriptomics_app
from src.mcp_servers.pharmacology_server import mcp as pharmacology_app
from src.mcp_servers.pathway_server import mcp as pathway_app

DATA = Path("src/data/processed")
MODEL = "groq:llama-3.3-70b-versatile"
CELL_LINE = "A375"
DRUG = "Dabrafenib"
TARGET_GENES = ["BRAF"]


async def main():
    client = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db")

    # Monkey-patch complete() to print raw response
    original_complete = client.complete
    async def debug_complete(messages, system=""):
        raw = await original_complete(messages, system)
        return raw

    agents = [
        ("Genomics",       GenomicsAgent(genomics_app, client)),
        ("Transcriptomics",TranscriptomicsAgent(transcriptomics_app, client)),
        ("Pharmacology",   PharmacologyAgent(pharmacology_app, client)),
        ("Pathway",        PathwayAgent(pathway_app, client)),
    ]

    for name, agent in agents:
        # Patch to capture raw before parsing
        orig = agent.llm.complete
        captured = {}
        async def patched(messages, system="", _orig=orig, _cap=captured):
            raw = await _orig(messages, system)
            _cap["raw"] = raw
            return raw
        agent.llm.complete = patched

        pack = await agent.analyze(CELL_LINE, DRUG, TARGET_GENES)
        raw = captured.get("raw", "(not captured)")

        print(f"\n{'='*60}")
        print(f"AGENT: {name}")
        print(f"VERDICT: {pack.verdict}  CONFIDENCE: {pack.confidence}")
        print(f"CAVEATS: {pack.caveats}")
        print(f"\n--- RAW RESPONSE (first 1500 chars) ---")
        print(raw[:1500])
        print(f"--- END ---")


if __name__ == "__main__":
    asyncio.run(main())
