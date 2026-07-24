"""Experiment 1 — each of the 4 agents run individually and in isolation.

No consensus, no voting, no debate: each agent sees only its own MCP evidence and
produces its own verdict. This measures each specialist's standalone predictive power
before any cross-agent aggregation is applied.

Usage:
    PYTHONPATH=. python trials/exp1_isolated_agents.py
"""
import asyncio
import json

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

    out_path = RESULTS / "exp1_isolated_agents.jsonl"
    out_path.unlink(missing_ok=True)

    per_agent_records: dict[str, list[dict]] = {a.agent_id: [] for a in agents}

    for set_id, case_id, case in dataset:
        target_genes = target_genes_map[case_id]
        print(f"[{case_id}] {case.cell_line} + {case.drug}  targets={target_genes}", flush=True)
        for agent in agents:
            pack = await agent.analyze(case.cell_line, case.drug, target_genes)
            record = {
                "set": set_id,
                "case_id": case_id,
                "agent_id": agent.agent_id,
                "cell_line": case.cell_line,
                "drug": case.drug,
                "true_label": case.label,
                "verdict": pack.verdict.value,
                "confidence": pack.confidence,
                "evidence_tier": pack.evidence_tier.value if pack.evidence_tier else None,
                "reasoning": pack.reasoning,
                "data_status": pack.data_status,
                "self_attestation": pack.self_attestation,
            }
            per_agent_records[agent.agent_id].append(record)
            with out_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            match = "OK" if record["verdict"] == case.label else ("?" if record["verdict"] == "UNCERTAIN" else "X")
            print(f"    {agent.agent_id:<24} {record['verdict']:<12} conf={record['confidence']:.2f}  {match}")

    summaries = {}
    for agent_id, records in per_agent_records.items():
        summary = score_records(records)
        print_summary(f"EXP1 — {agent_id} (isolated)", summary)
        summaries[agent_id] = summary

    ranked = sorted(summaries.items(), key=lambda kv: kv[1]["accuracy_definitive"], reverse=True)
    print(f"\n{'=' * 60}\n  RANKING (by accuracy on definitive predictions)\n{'=' * 60}")
    for agent_id, s in ranked:
        print(f"  {agent_id:<24} acc_def={s['accuracy_definitive']:.1%}  auroc={s['auroc']:.3f}  uncertain={s['uncertain_rate']:.1%}")

    (RESULTS / "exp1_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nTraces:  {out_path}")
    print(f"Summary: {RESULTS / 'exp1_summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
