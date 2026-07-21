"""Mini ablation study: 10 cases from each of sets 1-3 (30 total), multiple variants.

Variants:
  full_system       -- all 4 agents + full debate (baseline)
  no_debate         -- all 4 agents, majority vote only
  no_axioms         -- all 4 agents, confidence-only resolver
  no_genomics       -- drop T1 (genomics agent)
  no_transcriptomics-- drop T2 (transcriptomics agent)
  no_pathway        -- drop T3 (pathway agent)
  no_pharmacology   -- drop T4 (pharmacology agent)

Usage:
    PYTHONPATH=. python experiments/run_mini_ablation.py

Outputs:
    experiments/results/ablation_mini.json
"""
import asyncio
import json
import math
import os
import sqlite3
import statistics
from pathlib import Path

os.environ.setdefault("VERTEX_PROJECT", "project-d3bf2d5b-3451-46fd-8f3")

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.agents.monolithic_agent import MonolithicAgent
from src.data.loader import load_ctrp_cases
from src.evaluation.ablation_runner import AblationVariant, make_engine
from src.evaluation.metrics import evaluate, EvaluationMetrics
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator, _normalize_targets
from src.protocols.debate_engine import DebateEngine

MODEL     = "vertex:gemini-3.1-flash-lite"
DATA      = Path("src/data/processed")
RESULTS   = Path("experiments/results")
N_PER_SET = 10
SETS      = [
    (1, Path("data/cases/cases_held_out_ctrp_1.yaml")),
    (2, Path("data/cases/cases_held_out_ctrp_2.yaml")),
    (3, Path("data/cases/cases_held_out_ctrp_3.yaml")),
]


def _mcp_apps():
    from src.mcp_servers.genomics_server import mcp as g
    from src.mcp_servers.transcriptomics_server import mcp as t
    from src.mcp_servers.pharmacology_server import mcp as p
    from src.mcp_servers.pathway_server import mcp as pw
    return g, t, p, pw


def _get_target_genes(drug: str) -> list[str] | None:
    conn = sqlite3.connect(DATA / "pharmacology.db")
    row = conn.execute("SELECT target_genes FROM drug_info WHERE drug=?", (drug,)).fetchone()
    conn.close()
    if not row or not row[0]:
        return None
    return _normalize_targets([g.strip() for g in row[0].split(",")])


def _make_agents(apps, client, exclude: str | None = None):
    g_app, t_app, p_app, pw_app = apps
    all_agents = [
        ("genomics",       GenomicsAgent(g_app, client)),
        ("transcriptomics",TranscriptomicsAgent(t_app, client)),
        ("pharmacology",   PharmacologyAgent(p_app, client)),
        ("pathway",        PathwayAgent(pw_app, client, transcriptomics_mcp=t_app)),
    ]
    return [a for name, a in all_agents if name != exclude]


async def run_variant(
    label: str, agents, engine, apps, client,
) -> tuple[EvaluationMetrics, list[dict]]:
    _, t_app, _, _ = apps
    all_results, all_cases = [], []

    for set_num, yaml_path in SETS:
        cases = load_ctrp_cases(yaml_path)[N_PER_SET:N_PER_SET * 2]
        orch  = Orchestrator(agents=agents, engine=engine)
        for case in cases:
            tg = _get_target_genes(case.drug)
            r  = await orch.run_case(case.cell_line, case.drug, target_genes=tg)
            all_results.append(r)
            all_cases.append(case)

    m = evaluate(all_results, all_cases)
    return m


async def main():
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "ablation_mini.json"
    saved = json.loads(out_path.read_text()) if out_path.exists() else {}

    limiter = make_rate_limiter()
    client  = LLMClient(model=MODEL, cache_db=DATA / "llm_cache.db", rate_limiter=limiter)
    apps    = _mcp_apps()

    g_app, t_app, p_app, pw_app = apps
    variants = [
        ("full_system",           _make_agents(apps, client),                    DebateEngine()),
        ("no_debate",             _make_agents(apps, client),                    make_engine(AblationVariant.NO_DEBATE)),
        ("no_axioms",             _make_agents(apps, client),                    make_engine(AblationVariant.NO_AXIOMS)),
        ("no_genomics",           _make_agents(apps, client, exclude="genomics"),        DebateEngine()),
        ("no_transcriptomics",    _make_agents(apps, client, exclude="transcriptomics"), DebateEngine()),
        ("no_pathway",            _make_agents(apps, client, exclude="pathway"),         DebateEngine()),
        ("no_pharmacology",       _make_agents(apps, client, exclude="pharmacology"),    DebateEngine()),
        ("single_agent_multiomics", [MonolithicAgent(g_app, t_app, p_app, pw_app, client)], make_engine(AblationVariant.NO_DEBATE)),
    ]

    rows = []
    for label, agents, engine in variants:
        if label in saved:
            m = EvaluationMetrics(**saved[label])
            print(f"  {label:<22} AUROC={m.auroc:.3f}  κ={m.cohens_kappa:.3f}  cov={m.coverage:.0%}  [cached]")
        else:
            print(f"  {label:<22} running {N_PER_SET * len(SETS)} cases...", flush=True)
            m = await run_variant(label, agents, engine, apps, client)
            saved[label] = {
                "auroc": m.auroc, "auprc": m.auprc,
                "cohens_kappa": m.cohens_kappa, "spearman_rho": m.spearman_rho,
                "n_total": m.n_total, "n_decisive": m.n_decisive, "coverage": m.coverage,
            }
            out_path.write_text(json.dumps(saved, indent=2))
            print(f"  {label:<22} AUROC={m.auroc:.3f}  AUPRC={m.auprc:.3f}  κ={m.cohens_kappa:.3f}  ρ={m.spearman_rho:.3f}  cov={m.coverage:.0%}")
        rows.append((label, m))

    print(f"\n{'='*70}")
    print(f"  {'Variant':<22} {'AUROC':>7} {'AUPRC':>7} {'κ':>7} {'ρ':>7} {'Cov':>6}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for label, m in rows:
        marker = " ◀ baseline" if label == "full_system" else ""
        print(f"  {label:<22} {m.auroc:>7.3f} {m.auprc:>7.3f} {m.cohens_kappa:>7.3f} {m.spearman_rho:>7.3f} {m.coverage:>5.0%}{marker}")
    print(f"{'='*70}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
