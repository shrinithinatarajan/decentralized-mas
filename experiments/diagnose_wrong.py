"""For every wrong prediction from llama-70b, show what evidence each agent
received from the MCP servers and what verdict they returned.
All from cache — no API calls.
"""
import asyncio, json
from pathlib import Path
from fastmcp import Client

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import load_cases
from src.llm.client import LLMClient
from src.mcp_servers.genomics_server import mcp as genomics_app
from src.mcp_servers.transcriptomics_server import mcp as transcriptomics_app
from src.mcp_servers.pharmacology_server import mcp as pharmacology_app
from src.mcp_servers.pathway_server import mcp as pathway_app
from src.orchestrator import Orchestrator
from src.schemas.evidence_pack import EvidencePack

DATA = Path("src/data/processed")
MODEL = "nim:meta/llama-3.3-70b-instruct"


async def fetch_raw_evidence(cell_line, drug, targets):
    """Directly query MCP servers to show what raw data agents receive."""
    evidence = {}

    async with Client(genomics_app) as c:
        for gene in (targets or []):
            muts = json.loads((await c.call_tool("get_mutations", {"cell_line": cell_line, "gene": gene})).data)
            cnv  = json.loads((await c.call_tool("get_cnv",       {"cell_line": cell_line, "gene": gene})).data)
            evidence["genomics"] = {"mutations": muts, "cnv": cnv}

    async with Client(transcriptomics_app) as c:
        expr = []
        for gene in (targets or []):
            expr += json.loads((await c.call_tool("get_expression", {"cell_line": cell_line, "gene": gene})).data)
        evidence["transcriptomics"] = {"expression": expr}

    async with Client(pharmacology_app) as c:
        ic50     = json.loads((await c.call_tool("get_ic50",      {"cell_line": cell_line, "drug": drug})).data)
        drug_inf = json.loads((await c.call_tool("get_drug_info", {"drug": drug})).data)
        evidence["pharmacology"] = {"ic50": ic50, "drug_info": drug_inf}

    evidence["pathway"] = "(KEGG pathway topology — not shown for brevity)"
    return evidence


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

    wrong_cases = []
    all_packs: dict[str, list[EvidencePack]] = {}

    # Monkey-patch orchestrator to capture per-agent packs
    original_run_case = orch.run_case
    captured = {}
    async def capturing_run_case(cell_line, drug, target_genes=None):
        packs = list(await asyncio.gather(*[a.analyze(cell_line, drug, target_genes) for a in agents]))
        captured[f"{cell_line}|{drug}"] = packs
        return orch.engine.run(packs)
    orch.run_case = capturing_run_case

    for case in cases:
        targets = [g.strip() for g in case.putative_target.split(",")] if case.putative_target else None
        result = await orch.run_case(case.cell_line, case.drug, target_genes=targets)
        if result.final_verdict.value != "UNCERTAIN" and result.final_verdict.value != case.label:
            wrong_cases.append((case, result, captured.get(f"{case.cell_line}|{case.drug}", [])))

    print(f"\n{'='*72}")
    print(f"WRONG PREDICTIONS — llama-3.3-70b ({len(wrong_cases)} cases)")
    print(f"{'='*72}")

    # Tally failure patterns
    patterns = {"no_genomics_data": 0, "no_transcriptomics_data": 0,
                "pathway_override": 0, "pharmacology_conflict": 0, "agent_disagreement": 0}

    for case, result, packs in wrong_cases:
        targets = [g.strip() for g in case.putative_target.split(",")] if case.putative_target else []
        raw = await fetch_raw_evidence(case.cell_line, case.drug, targets)

        print(f"\n─ {case.cell_line} + {case.drug}")
        print(f"  True: {case.label} | Predicted: {result.final_verdict.value} | "
              f"Axiom tier: {case.axiom_tier} | Notes: {case.notes}")
        print(f"  Winning agent: {result.winning_agent}  |  Rounds: {result.rounds_taken}")

        # Raw MCP data
        g = raw.get("genomics", {})
        muts = g.get("mutations", [])
        cnv  = g.get("cnv", [])
        expr = raw.get("transcriptomics", {}).get("expression", [])
        ic50 = raw.get("pharmacology", {}).get("ic50", {})

        print(f"  MCP data:")
        print(f"    Mutations : {muts if muts else '(none)'}")
        print(f"    CNV       : {[{k: v for k,v in r.items() if k in ('gene','cnv_value','status')} for r in cnv] if cnv else '(none)'}")
        print(f"    Expression: {[{k: v for k,v in e.items() if k in ('gene','z_score','tpm')} for e in expr] if expr else '(none)'}")
        print(f"    IC50      : {ic50}")

        # Agent verdicts
        print(f"  Agent verdicts:")
        verdicts = set()
        for p in packs:
            findings = "; ".join(f"{f.biomarker}={f.value}" for f in p.key_findings[:2]) or "(none)"
            print(f"    {p.agent_id:<32} {p.verdict.value:<12} conf={p.confidence:.2f}  findings: {findings}")
            verdicts.add(p.verdict.value)

        # Pattern diagnosis
        if not muts and not expr:
            patterns["no_genomics_data"] += 1
            print(f"  ⚠ DIAGNOSIS: No genomic or expression data for target genes {targets}")
        elif len(verdicts) > 1:
            if any(p.agent_id == "pathway_agent" and p.verdict.value != case.label for p in packs):
                patterns["pathway_override"] += 1
                print(f"  ⚠ DIAGNOSIS: Pathway agent voted against ground truth and influenced resolution")
            elif any(p.agent_id == "pharmacology_agent" and p.verdict.value != case.label for p in packs):
                patterns["pharmacology_conflict"] += 1
                print(f"  ⚠ DIAGNOSIS: Pharmacology agent conflicted with ground truth")
            else:
                patterns["agent_disagreement"] += 1
                print(f"  ⚠ DIAGNOSIS: Agent disagreement not clearly attributable to one source")
        else:
            patterns["agent_disagreement"] += 1
            print(f"  ⚠ DIAGNOSIS: All agents agreed on wrong answer — likely data quality issue")

    print(f"\n{'='*72}")
    print("FAILURE PATTERN SUMMARY")
    print(f"{'='*72}")
    for k, v in patterns.items():
        print(f"  {k:<35}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
