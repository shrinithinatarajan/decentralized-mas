"""Print per-case verdict breakdown and diagnose kappa."""
import asyncio
from pathlib import Path
from collections import Counter

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import load_cases
from src.llm.client import LLMClient
from src.orchestrator import Orchestrator
from src.mcp_servers.genomics_server import mcp as genomics_app
from src.mcp_servers.transcriptomics_server import mcp as transcriptomics_app
from src.mcp_servers.pharmacology_server import mcp as pharmacology_app
from src.mcp_servers.pathway_server import mcp as pathway_app

DATA = Path("src/data/processed")
MODEL = "groq:llama-3.3-70b-versatile"


async def main():
    cases = load_cases(Path("cases.yaml"))
    client = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db")
    agents = [
        GenomicsAgent(genomics_app, client),
        TranscriptomicsAgent(transcriptomics_app, client),
        PharmacologyAgent(pharmacology_app, client),
        PathwayAgent(pathway_app, client),
    ]
    orch = Orchestrator(agents=agents)

    results = await orch.run_all(cases)

    print(f"{'Cell Line':<20} {'Drug':<25} {'True':<10} {'Pred':<10} {'Conf':>6} {'Match'}")
    print("-" * 80)

    pred_counts = Counter()
    true_counts = Counter()
    correct = wrong = uncertain = 0

    for case, result in zip(cases, results):
        pred = result.final_verdict.value
        true = case.label
        pred_counts[pred] += 1
        true_counts[true] += 1
        if pred == "UNCERTAIN":
            match = "?"
            uncertain += 1
        elif pred == true:
            match = "✓"
            correct += 1
        else:
            match = "✗"
            wrong += 1
        print(f"{case.cell_line:<20} {case.drug:<25} {true:<10} {pred:<10} {result.final_confidence:>6.2f} {match}")

    print("\n--- Predicted verdict distribution ---")
    for v, n in sorted(pred_counts.items()):
        print(f"  {v}: {n}")

    print("\n--- True label distribution ---")
    for v, n in sorted(true_counts.items()):
        print(f"  {v}: {n}")

    total = len(cases)
    print(f"\nCorrect: {correct}/{total}  Wrong: {wrong}/{total}  Uncertain: {uncertain}/{total}")
    print(f"Accuracy (excl. uncertain): {correct/(correct+wrong):.3f}" if correct+wrong else "")


if __name__ == "__main__":
    asyncio.run(main())
