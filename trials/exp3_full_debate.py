"""Experiment 3 — same 4 agents, same dataset, full debate protocol.

Uses the real DebateEngine: Round-1 consensus check -> peer critique on genuine conflict ->
Round-2 consensus check -> AxiomResolver tiebreak as last resort. This measures the complete
system as designed, on the same 20 cases used in Experiments 1 and 2.

Note: as of this run, src/protocols/debate_engine.py still has open bugs (single-decisive-agent
R1 auto-win with no escalation, and a T3 self-attestation gate that zeroes missing — not just
low — scores) documented in docs/fixes_needed.html. Results here reflect current behavior,
not the system's fixed ceiling.

Usage:
    PYTHONPATH=. python trials/exp3_full_debate.py
"""
import asyncio
import json

from src.orchestrator import Orchestrator
from trials.common import (
    RESULTS, load_fixed_dataset, load_target_genes_map, make_client, make_agents,
    score_records, print_summary,
)


async def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    dataset = load_fixed_dataset()
    target_genes_map = load_target_genes_map()
    client = make_client()
    agents = make_agents(client)
    orch = Orchestrator(agents=agents)  # default engine = full DebateEngine

    out_path = RESULTS / "exp3_full_debate.jsonl"
    out_path.unlink(missing_ok=True)

    records = []
    for set_id, case_id, case in dataset:
        target_genes = target_genes_map[case_id]
        print(f"[{case_id}] {case.cell_line} + {case.drug}  targets={target_genes}", flush=True)
        result = await orch.run_case(case.cell_line, case.drug, target_genes=target_genes, case_id=case_id)
        record = {
            "set": set_id,
            "case_id": case_id,
            "cell_line": case.cell_line,
            "drug": case.drug,
            "true_label": case.label,
            "verdict": result.final_verdict.value,
            "confidence": result.final_confidence,
            "winning_agent": result.winning_agent,
            "resolution_method": result.resolution_method,
            "rounds_taken": result.rounds_taken,
            "forced": result.forced,
            "dissenting_agents": result.dissenting_agents,
        }
        records.append(record)
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        match = "OK" if record["verdict"] == case.label else ("?" if record["verdict"] == "UNCERTAIN" else "X")
        print(f"    {record['verdict']:<12} conf={record['confidence']:.2f}  {record['resolution_method']:<20} winner={record['winning_agent']:<24} {match}")

    summary = score_records(records)
    print_summary("EXP3 — full debate protocol (axiom resolver + peer critique)", summary)

    methods = {}
    wins = {}
    for r in records:
        methods[r["resolution_method"]] = methods.get(r["resolution_method"], 0) + 1
        wins[r["winning_agent"]] = wins.get(r["winning_agent"], 0) + 1
    print(f"\n  Resolution methods: {methods}")
    print(f"  Winning-agent breakdown: {wins}")

    (RESULTS / "exp3_summary.json").write_text(json.dumps(
        {"summary": summary, "resolution_methods": methods, "winning_agents": wins}, indent=2
    ))
    print(f"\nTraces:  {out_path}")
    print(f"Summary: {RESULTS / 'exp3_summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
