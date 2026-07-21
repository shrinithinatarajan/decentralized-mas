"""Run balanced validation: 3 sets of 10 cases each, covering mutation-driven (Set A),
pathway/expression-driven (Set B), and mechanism-agnostic (Set C) drug-cell pairs.

All cases drawn from CTRPv2 held-out sets 4 & 5 (not used in mini validation).
Traces saved to experiments/results/traces_balanced_set_{a,b,c}.jsonl.
Metrics saved to experiments/results/balanced_validation.json.
Full reasoning logs written to artifacts/ via RunLogger.

Usage:
    PYTHONPATH=. python experiments/run_balanced_validation.py          # all sets
    PYTHONPATH=. python experiments/run_balanced_validation.py c        # set C only
    PYTHONPATH=. python experiments/run_balanced_validation.py a b      # sets A and B

Resume: already-completed cases (present in traces file) are skipped automatically.
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

DATA    = Path("src/data/processed")
RESULTS = Path("experiments/results")
MODEL   = "vertex:gemini-3.1-flash-lite"

SETS = [
    ("a", "Set A: Mutation-driven",          Path("data/cases/cases_balanced_a.yaml")),
    ("b", "Set B: Pathway/expression-driven", Path("data/cases/cases_balanced_b.yaml")),
    ("c", "Set C: Mechanism-agnostic",        Path("data/cases/cases_balanced_c.yaml")),
]


def _mcp_apps():
    from src.mcp_servers.genomics_server import mcp as g
    from src.mcp_servers.transcriptomics_server import mcp as t
    from src.mcp_servers.pharmacology_server import mcp as p
    from src.mcp_servers.pathway_server import mcp as pw
    return g, t, p, pw


def _get_target_genes(drug: str, case_targets: list[str] | None) -> list[str] | None:
    """Use explicit target_genes from YAML if present, else look up from drug_info DB."""
    if case_targets:
        return _normalize_targets(case_targets)
    conn = sqlite3.connect(DATA / "pharmacology.db")
    row = conn.execute("SELECT target_genes FROM drug_info WHERE drug=?", (drug,)).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    raw = [g.strip() for g in row[0].split(",")]
    return _normalize_targets(raw)


def _load_completed(traces_path: Path) -> dict[str, dict]:
    """Return {case_id: record} for cases already written to the traces file."""
    if not traces_path.exists():
        return {}
    done: dict[str, dict] = {}
    for line in traces_path.read_text().splitlines():
        try:
            r = json.loads(line)
            done[r["case_id"]] = r
        except Exception:
            continue
    return done


async def run_set(
    set_id: str, label: str, yaml_path: Path,
    orch: Orchestrator, run_logger: RunLogger
) -> list[dict]:
    cases = load_ctrp_cases(yaml_path)
    out_path = RESULTS / f"traces_balanced_set_{set_id}.jsonl"

    completed = _load_completed(out_path)
    records: list[dict] = list(completed.values())

    print(f"\n=== {label} ({yaml_path.name}) — {len(cases)} cases ===")
    if completed:
        print(f"  Resuming: {len(completed)} already done, {len(cases) - len(completed)} remaining")

    for i, case in enumerate(cases, 1):
        case_targets = getattr(case, "target_genes", None)
        target_genes = _get_target_genes(case.drug, case_targets)
        case_id = f"balanced_{set_id}:{i}:{case.cell_line}:{case.drug}"

        if case_id in completed:
            r = completed[case_id]
            match = "✓" if r["correct"] else ("?" if r["final_verdict"] == "UNCERTAIN" else "✗")
            print(f"  [{i}/{len(cases)}] {case.cell_line} + {case.drug}  [cached {match}]")
            continue

        print(f"  [{i}/{len(cases)}] {case.cell_line} + {case.drug}  targets={target_genes}", flush=True)

        result = await orch.run_case(
            case.cell_line, case.drug,
            target_genes=target_genes,
            run_logger=run_logger,
            case_id=case_id,
        )
        record = {
            "set": set_id,
            "set_label": label,
            "case_id": case_id,
            "cell_line": case.cell_line,
            "drug": case.drug,
            "target_genes": target_genes,
            "true_label": case.label,
            "final_verdict": result.final_verdict.value,
            "final_confidence": result.final_confidence,
            "resolution_method": result.resolution_method,
            "winning_agent": result.winning_agent,
            "rounds_taken": result.rounds_taken,
            "correct": result.final_verdict.value == case.label,
        }
        records.append(record)
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        match = "✓" if record["correct"] else ("?" if result.final_verdict.value == "UNCERTAIN" else "✗")
        print(f"         {result.final_verdict.value:<12} {result.resolution_method:<20} gt={case.label}  {match}")

    return records


def _print_set_summary(set_id: str, label: str, records: list[dict]) -> dict:
    correct   = sum(1 for r in records if r["correct"])
    uncertain = sum(1 for r in records if r["final_verdict"] == "UNCERTAIN")
    methods: dict[str, int] = {}
    agents:  dict[str, int] = {}
    for r in records:
        methods[r["resolution_method"]] = methods.get(r["resolution_method"], 0) + 1
        if r["winning_agent"]:
            agents[r["winning_agent"]] = agents.get(r["winning_agent"], 0) + 1
    n = len(records)
    print(f"  {label}")
    print(f"    Correct: {correct}/{n} ({correct/n:.0%})  Uncertain: {uncertain}/{n}")
    print(f"    Methods: {methods}")
    print(f"    Winning agents: {agents}")
    return {
        "set": set_id, "label": label, "n": n,
        "correct": correct, "uncertain": uncertain,
        "accuracy": correct / n,
        "resolution_methods": methods,
        "winning_agents": agents,
    }


async def main():
    import sys
    RESULTS.mkdir(exist_ok=True)

    # Optional set filter: python run_balanced_validation.py c  OR  a b
    requested = {s.lower() for s in sys.argv[1:]} if len(sys.argv) > 1 else {"a", "b", "c"}
    active_sets = [(sid, lbl, p) for sid, lbl, p in SETS if sid in requested]
    if not active_sets:
        print(f"No matching sets for {sys.argv[1:]}. Valid: a b c"); return

    out_metrics = RESULTS / "balanced_validation.json"
    existing_metrics: dict = json.loads(out_metrics.read_text()) if out_metrics.exists() else {}

    limiter = make_rate_limiter()
    client  = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db", rate_limiter=limiter)
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
    print(f"Model: {MODEL}  Sets: {[s[0] for s in active_sets]}")

    all_records: list[dict] = []
    set_summaries: list[dict] = []

    for set_id, label, yaml_path in active_sets:
        records = await run_set(set_id, label, yaml_path, orch, run_logger)
        all_records.extend(records)

    print(f"\n{'='*65}")
    print(f"  BALANCED VALIDATION SUMMARY  |  Model: {MODEL}")
    print(f"{'='*65}")
    for set_id, label, _ in active_sets:
        recs = [r for r in all_records if r["set"] == set_id]
        summary = _print_set_summary(set_id, label, recs)
        set_summaries.append(summary)

    total     = len(all_records)
    correct   = sum(1 for r in all_records if r["correct"])
    uncertain = sum(1 for r in all_records if r["final_verdict"] == "UNCERTAIN")
    print(f"{'='*65}")
    print(f"  OVERALL: {correct}/{total} correct ({correct/total:.0%})  |  Uncertain: {uncertain}/{total}")
    print(f"{'='*65}")
    print(f"  Traces: {RESULTS}/traces_balanced_set_{{a,b,c}}.jsonl")
    print(f"  Full log: {run_logger.log_path}")

    # Merge per-set summaries into existing metrics file
    existing_metrics["sets"] = existing_metrics.get("sets", {})
    for summary in set_summaries:
        existing_metrics["sets"][summary["set"]] = summary
    out_metrics.write_text(json.dumps(existing_metrics, indent=2))
    print(f"  Metrics saved to {out_metrics}")

    run_logger.close()


if __name__ == "__main__":
    asyncio.run(main())
