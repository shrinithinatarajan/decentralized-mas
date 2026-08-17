"""Run all 20 cases from each of 5 CTRP sets (100 total) with per-case debate traces.

Target genes are looked up from drug_info so the pathway agent is fully active.
Traces are saved to experiments/results/traces_full_set_{1..5}.jsonl.
Full per-agent reasoning is captured in logs/run_<id>.log via RunLogger.

Usage:
    PYTHONPATH=. python experiments/run_mini_validation.py
"""
import asyncio
import json
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import load_ctrp_cases
from src.evaluation.metrics import evaluate
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator, _normalize_targets
from src.run_logger import RunLogger

DATA   = Path("src/data/processed")
RESULTS = Path("experiments/results")
MODEL  = "vertex:gemini-3.1-flash-lite"
N_PER_SET = 100

SETS = [
    (1, Path("data/cases/cases_gold_standard.yaml")),
]


def _mcp_apps():
    from src.mcp_servers.genomics_server import mcp as g
    from src.mcp_servers.transcriptomics_server import mcp as t
    from src.mcp_servers.pharmacology_server import mcp as p
    from src.mcp_servers.pathway_server import mcp as pw
    return g, t, p, pw


def _get_target_genes(drug: str) -> list[str] | None:
    """Look up target genes from drug_info and normalize via gene alias table."""
    conn = sqlite3.connect(DATA / "pharmacology.db")
    row = conn.execute("SELECT target_genes FROM drug_info WHERE drug=?", (drug,)).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    raw = [g.strip() for g in row[0].split(",")]
    return _normalize_targets(raw)


async def run_set(set_num: int, yaml_path: Path, orch: Orchestrator, run_logger: RunLogger) -> list[dict]:
    all_cases = load_ctrp_cases(yaml_path)
    cases = all_cases[:N_PER_SET]
    records = []

    out_path = RESULTS / f"traces_gold_standard.jsonl" if yaml_path.name == "cases_gold_standard.yaml" else RESULTS / f"traces_full_set_{set_num}.jsonl"
    out_path.unlink(missing_ok=True)

    print(f"\n=== Set {set_num}: {yaml_path.name} — {len(cases)} cases ===")
    for i, case in enumerate(cases, 1):
        target_genes = _get_target_genes(case.drug)
        case_id = f"mini_{set_num}:{i}:{case.cell_line}:{case.drug}"
        print(f"  [{i}/{len(cases)}] {case.cell_line} + {case.drug}  targets={target_genes}", flush=True)

        result = await orch.run_case(
            case.cell_line, case.drug,
            target_genes=target_genes,
            run_logger=run_logger,
            case_id=case_id,
        )
        record = {
            "set": set_num,
            "case_id": case_id,
            "cell_line": case.cell_line,
            "drug": case.drug,
            "target_genes": target_genes,
            "true_label": case.label,
            "final_verdict": result.final_verdict.value,
            "final_confidence": result.final_confidence,
            "resolution_method": result.resolution_method,
            "winning_agent": result.winning_agent,
            "contributing_agents": result.contributing_agents,
            "rounds_taken": result.rounds_taken,
            "forced": result.forced,
            "correct": result.final_verdict.value == case.label,
            "winning_agent_is_t3": "pathway" in result.winning_agent.lower(),
            "r1_agents": result.r1_agents,
            "dissenting_agents": result.dissenting_agents,
            "trace": result.trace,
        }
        records.append(record)
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        match = "✓" if record["correct"] else ("?" if result.final_verdict.value == "UNCERTAIN" else "✗")
        print(f"         {result.final_verdict.value:<12} {result.resolution_method:<20} gt={case.label}  {match}")

    return records


async def main():
    RESULTS.mkdir(exist_ok=True)
    limiter  = make_rate_limiter()
    client   = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db", rate_limiter=limiter)
    g_app, t_app, p_app, pw_app = _mcp_apps()

    agents = [
        GenomicsAgent(g_app, client),
        TranscriptomicsAgent(t_app, client),
        PharmacologyAgent(p_app, client),
        PathwayAgent(pw_app, client, transcriptomics_mcp=t_app),
    ]
    orch = Orchestrator(agents=agents)
    run_logger = RunLogger()
    print(f"run_id={run_logger.run_id}  log={run_logger.log_path}")

    all_records: list[dict] = []
    for set_num, yaml_path in SETS:
        records = await run_set(set_num, yaml_path, orch, run_logger)
        all_records.extend(records)

    run_logger.close()

    # Summary
    correct   = sum(1 for r in all_records if r["correct"])
    uncertain = sum(1 for r in all_records if r["final_verdict"] == "UNCERTAIN")
    t3_wins   = sum(1 for r in all_records if r.get("winning_agent_is_t3"))
    methods   = {}
    for r in all_records:
        methods[r["resolution_method"]] = methods.get(r["resolution_method"], 0) + 1

    # Per-agent winning breakdown
    agent_wins: dict[str, int] = {}
    for r in all_records:
        if r["final_verdict"] != "UNCERTAIN":
            agent_wins[r["winning_agent"]] = agent_wins.get(r["winning_agent"], 0) + 1

    print(f"\n{'='*60}")
    print(f"  Model: {MODEL}  |  Cases: {len(all_records)}")
    print(f"  Correct:   {correct}/{len(all_records)} ({correct/len(all_records):.0%})")
    print(f"  Uncertain: {uncertain}/{len(all_records)} ({uncertain/len(all_records):.0%})")
    print(f"  T3 (pathway) decisive: {t3_wins}/{len(all_records)} ({t3_wins/len(all_records):.0%})")
    print(f"  Resolution methods: {methods}")
    print(f"  Agent win breakdown: {agent_wins}")
    print(f"{'='*60}")
    print(f"  Traces: {RESULTS}/traces_mini_set_{{1,2,3}}.jsonl")
    print(f"  Full log: {run_logger.log_path}")


if __name__ == "__main__":
    asyncio.run(main())
