"""Natural language query CLI for the drug resistance system.

Usage:
    python experiments/query.py "Will BT-483 respond to Foretinib?"
    python experiments/query.py "What is the sensitivity of MCF7 to Tamoxifen?"
    python experiments/query.py --cell-line A375 --drug Dabrafenib

Outputs a verdict with confidence, winning evidence tier, and key findings.
"""
import argparse
import asyncio
import os
from pathlib import Path

from src.agents.parser_agent import ParserAgent
from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.gene_aliases import GENE_ALIASES, SKIP_TARGETS
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator, _normalize_targets
from src.protocols.debate_engine import DebateEngine


def _get_drug_target(drug: str) -> list[str] | None:
    """Look up putative target for a drug from pharmacology DB."""
    import sqlite3
    db = Path("src/data/processed/pharmacology.db")
    if not db.exists():
        return None
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT target_genes FROM drug_info WHERE LOWER(drug)=LOWER(?)", (drug,)
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    raw = [t.strip() for t in row[0].split(",") if t.strip()]
    return _normalize_targets(raw)


async def run_query(query: str | None, cell_line: str | None, drug: str | None) -> None:
    model = os.getenv("LLM_MODEL", "vertex:gemini-3.1-flash-lite")
    limiter = make_rate_limiter()
    llm = LLMClient(model=model, rate_limiter=limiter)

    # Parse natural language if no structured input
    if query and not (cell_line and drug):
        print(f"Parsing query: {query!r}")
        parser = ParserAgent(llm_client=llm)
        parsed = await parser.parse(query)
        cell_line = cell_line or parsed["cell_line"]
        drug      = drug      or parsed["drug"]
        if not parsed["cell_line_matched"]:
            print(f"  [!] Cell line {parsed['raw_cell_line']!r} not found in DB — proceeding anyway")
        if not parsed["drug_matched"]:
            print(f"  [!] Drug {parsed['raw_drug']!r} not found in DB — proceeding anyway")

    if not cell_line or not drug:
        print("Error: could not extract cell line and drug from query.")
        return

    print(f"\nRunning analysis: {cell_line} + {drug}")

    from src.mcp_servers.genomics_server import mcp as genomics_app
    from src.mcp_servers.transcriptomics_server import mcp as transcriptomics_app
    from src.mcp_servers.pharmacology_server import mcp as pharmacology_app
    from src.mcp_servers.pathway_server import mcp as pathway_app

    agents = [
        GenomicsAgent(genomics_app, llm),
        TranscriptomicsAgent(transcriptomics_app, llm),
        PharmacologyAgent(pharmacology_app, llm),
        PathwayAgent(pathway_app, llm),
    ]
    orchestrator = Orchestrator(agents, DebateEngine())
    target_genes = _get_drug_target(drug)

    result = await orchestrator.run_case(cell_line, drug, target_genes=target_genes)

    print(f"\n{'='*60}")
    print(f"  Verdict:    {result.final_verdict.value}")
    print(f"  Confidence: {result.final_confidence:.2f}")
    print(f"  Evidence:   {result.winning_agent} ({result.rounds_taken} debate round(s))")
    if result.dissenting_agents:
        print(f"  Dissent:    {', '.join(result.dissenting_agents)}")
    print(f"{'='*60}")
    if result.trace:
        print("\nReasoning trace:")
        for step in result.trace[-3:]:  # last 3 steps
            print(f"  {step}")


def main():
    p = argparse.ArgumentParser(description="Query drug resistance system in natural language")
    p.add_argument("query", nargs="?", help="Free-text query (e.g. 'Will BT-483 respond to Foretinib?')")
    p.add_argument("--cell-line", "-c", help="Cell line name (bypasses parser)")
    p.add_argument("--drug", "-d", help="Drug name (bypasses parser)")
    args = p.parse_args()

    if not args.query and not (args.cell_line and args.drug):
        p.error("Provide a query string or both --cell-line and --drug")

    asyncio.run(run_query(args.query, args.cell_line, args.drug))


if __name__ == "__main__":
    main()
