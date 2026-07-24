"""Experiment 2 — same 4 agents, same dataset, but aggregated by majority-vote consensus.

Each agent still analyzes independently (no cross-agent visibility during analysis), but
instead of reporting each agent's verdict separately, the four EvidencePacks are combined
via majority vote (src.evaluation.ablation_runner.NoDebateEngine) with NO debate/critique
round and NO axiom hierarchy. This isolates the effect of aggregation alone, before adding
structured debate.

Usage:
    PYTHONPATH=. python trials/exp2_consensus_no_debate.py
"""
import asyncio
import json

from src.evaluation.ablation_runner import NoDebateEngine
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
    orch = Orchestrator(agents=agents, engine=NoDebateEngine())

    out_path = RESULTS / "exp2_consensus_no_debate.jsonl"
    out_path.unlink(missing_ok=True)

    records = []
    for set_id, case_id, case in dataset:
        target_genes = target_genes_map[case_id]
        print(f"[{case_id}] {case.cell_line} + {case.drug}  targets={target_genes}", flush=True)
        result = await orch.run_case(case.cell_line, case.drug, target_genes=target_genes)
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
            "r1_agents": result.r1_agents,
        }
        records.append(record)
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        match = "OK" if record["verdict"] == case.label else ("?" if record["verdict"] == "UNCERTAIN" else "X")
        print(f"    {record['verdict']:<12} conf={record['confidence']:.2f}  winner={record['winning_agent']:<24} {match}")

    summary = score_records(records)
    print_summary("EXP2 — majority-vote consensus (no debate)", summary)

    wins = {}
    for r in records:
        wins[r["winning_agent"]] = wins.get(r["winning_agent"], 0) + 1
    print(f"\n  Winning-agent breakdown: {wins}")

    (RESULTS / "exp2_summary.json").write_text(json.dumps({"summary": summary, "winning_agents": wins}, indent=2))
    print(f"\nTraces:  {out_path}")
    print(f"Summary: {RESULTS / 'exp2_summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
