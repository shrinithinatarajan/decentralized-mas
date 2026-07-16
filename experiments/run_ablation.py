"""Ablation study runner for the multi-agent drug resistance system.

Variants
--------
Agent contribution (remove one at a time):
  no-genomics, no-transcriptomics, no-pharmacology, no-pathway

Single-agent baselines:
  only-genomics, only-transcriptomics, only-pharmacology, only-pathway

Protocol ablations:
  no-debate          First-round axiom resolution only, no challenge rounds
  majority-vote      Majority verdict wins instead of axiom hierarchy
  no-civic           Genomics agent with CIViC descriptions stripped

Usage:
    set -a && source .env && set +a
    PYTHONPATH=. LLM_CALLS_PER_MINUTE=8 python experiments/run_ablation.py <variant>
    PYTHONPATH=. LLM_CALLS_PER_MINUTE=8 python experiments/run_ablation.py --table

All results accumulate in experiments/results/ablation_results.json.
"""
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from src.agents.genomics_agent import GenomicsAgent
from src.agents.transcriptomics_agent import TranscriptomicsAgent
from src.agents.pharmacology_agent import PharmacologyAgent
from src.agents.pathway_agent import PathwayAgent
from src.data.loader import load_cases
from src.evaluation.metrics import evaluate
from src.llm.client import LLMClient, make_rate_limiter
from src.orchestrator import Orchestrator
from src.protocols.axiom_resolver import AxiomResolver, ResolutionResult
from src.protocols.debate_engine import ConsensusResult, DebateEngine
from src.schemas.axiom_rules import AXIOM_HIERARCHY, EVIDENCE_TO_AXIOM_TIER
from src.schemas.evidence_pack import EvidencePack, Verdict

CASES_FILE = Path("data/cases/cases_held_out_v2.yaml")
RESULTS    = Path("experiments/results")
OUT_PATH   = RESULTS / "ablation_results.json"
CACHE      = Path("src/data/processed/llm_cache.db")
MODEL      = "vertex:gemini-3.1-flash-lite"

VARIANTS = [
    "no-genomics", "no-transcriptomics", "no-pharmacology", "no-pathway",
    "only-genomics", "only-transcriptomics", "only-pharmacology", "only-pathway",
    "no-debate", "majority-vote", "no-civic",
]


# ── Protocol ablation: majority-vote resolver ──────────────────────────

class MajorityVoteResolver(AxiomResolver):
    """Resolve by majority verdict; break ties by confidence."""
    def resolve(self, packs: list[EvidencePack]) -> ResolutionResult:
        counts = Counter(p.verdict for p in packs if p.verdict != Verdict.UNCERTAIN)
        if not counts:
            counts = Counter(p.verdict for p in packs)
        majority = counts.most_common(1)[0][0]
        # among majority voters pick highest confidence
        winners = [p for p in packs if p.verdict == majority]
        winner  = max(winners, key=lambda p: p.confidence)
        return ResolutionResult(
            verdict=majority,
            winning_agent=winner.agent_id,
            axiom_applied="majority_vote",
            adjusted_packs=packs,
        )


# ── Protocol ablation: no-debate engine ────────────────────────────

class NoDebateEngine(DebateEngine):
    """Skip debate rounds — resolve directly from first-round agent outputs."""
    def run(self, packs: list[EvidencePack]) -> ConsensusResult:
        resolution = self._resolver.resolve(packs)
        avg_conf = sum(p.confidence for p in packs) / len(packs)
        return ConsensusResult(
            final_verdict=resolution.verdict,
            final_confidence=avg_conf,
            cell_line=packs[0].cell_line,
            drug=packs[0].drug,
            winning_agent=resolution.winning_agent,
            rounds_taken=0,
            forced=False,
            dissenting_agents=[
                p.agent_id for p in packs
                if p.agent_id != resolution.winning_agent and p.verdict != resolution.verdict
            ],
            trace=[],
        )


# ── Evidence ablation: strip CIViC from genomics evidence ─────────────

class NoCivicGenomicsAgent(GenomicsAgent):
    async def _fetch_evidence(self, cell_line, drug, target_genes=None):
        evidence = await super()._fetch_evidence(cell_line, drug, target_genes)
        # Strip civic_description from every mutation row
        for m in evidence.get("mutations", []):
            m.pop("civic_description", None)
        return evidence


# ── MCP apps (shared) ──────────────────────────────────────────

def _apps():
    from src.mcp_servers.genomics_server import mcp as gen
    from src.mcp_servers.transcriptomics_server import mcp as tran
    from src.mcp_servers.pharmacology_server import mcp as pharm
    from src.mcp_servers.pathway_server import mcp as path
    return gen, tran, pharm, path


def _make_agents(variant: str, llm: LLMClient) -> tuple[list, DebateEngine]:
    gen, tran, pharm, path = _apps()
    all_agents = {
        "genomics":       GenomicsAgent(gen, llm),
        "transcriptomics": TranscriptomicsAgent(tran, llm),
        "pharmacology":   PharmacologyAgent(pharm, llm),
        "pathway":        PathwayAgent(path, llm),
    }
    engine = DebateEngine()  # default

    if variant.startswith("no-") and variant[3:] in all_agents:
        skip = variant[3:]
        agents = [v for k, v in all_agents.items() if k != skip]
    elif variant.startswith("only-") and variant[5:] in all_agents:
        keep = variant[5:]
        agents = [all_agents[keep]]
    elif variant == "no-debate":
        agents = list(all_agents.values())
        engine = NoDebateEngine()
    elif variant == "majority-vote":
        agents = list(all_agents.values())
        engine = DebateEngine(resolver=MajorityVoteResolver())
    elif variant == "no-civic":
        agents = [
            NoCivicGenomicsAgent(gen, llm),
            TranscriptomicsAgent(tran, llm),
            PharmacologyAgent(pharm, llm),
            PathwayAgent(path, llm),
        ]
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Choose from: {VARIANTS}")

    return agents, engine


# ── Runner ──────────────────────────────────────────────────────

def _print_table(results: dict) -> None:
    # Group by category
    groups = {
        "Full system (baseline)": ["full-system"],
        "Remove one agent":       ["no-genomics","no-transcriptomics","no-pharmacology","no-pathway"],
        "Single agent only":      ["only-genomics","only-transcriptomics","only-pharmacology","only-pathway"],
        "Protocol ablation":      ["no-debate","majority-vote","no-civic"],
    }
    print("\n" + "="*70)
    print(f"  {'Variant':<28}  {'AUROC':>6}  {'AUPRC':>6}  {'κ':>6}  {'n':>4}")
    print("  " + "-"*60)
    for group, keys in groups.items():
        printed_header = False
        for key in keys:
            if key not in results:
                continue
            if not printed_header:
                print(f"  {group}")
                printed_header = True
            m = results[key]
            if "error" in m:
                print(f"    {key:<26}  ERROR")
            else:
                print(f"    {key:<26}  {m['auroc']:>6.3f}  {m['auprc']:>6.3f}  {m['cohens_kappa']:>6.3f}  {m.get('n_evaluated',20):>4}")
    print("="*70)


async def run_variant(variant: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = load_cases(CASES_FILE)

    all_results: dict = {}
    if OUT_PATH.exists():
        all_results = json.loads(OUT_PATH.read_text())

    # Seed full-system baseline from held-out v2 metrics
    v2 = Path("experiments/results/held_out_v2_metrics.json")
    if v2.exists():
        v2_data = json.loads(v2.read_text())
        if "gemini-3.1-flash-lite" in v2_data:
            all_results.setdefault("full-system", v2_data["gemini-3.1-flash-lite"])

    if variant in all_results and "error" not in all_results[variant]:
        print(f"[{variant}] already done — skipping.")
        _print_table(all_results)
        return

    print(f"=== Ablation: {variant} — {len(cases)} cases ===", flush=True)
    limiter = make_rate_limiter()
    try:
        llm    = LLMClient(model=MODEL, cache_db=CACHE, rate_limiter=limiter)
        agents, engine = _make_agents(variant, llm)
        orch   = Orchestrator(agents=agents, engine=engine)
        preds  = await orch.run_all(
            cases,
            traces_path=RESULTS / f"traces_ablation_{variant}.jsonl",
        )
        m = evaluate(preds, cases)
        all_results[variant] = {
            "auroc": round(m.auroc, 3),
            "auprc": round(m.auprc, 3),
            "cohens_kappa": round(m.cohens_kappa, 3),
            "n_evaluated": m.n_evaluated,
        }
        print(f"  AUROC={m.auroc:.3f}  AUPRC={m.auprc:.3f}  κ={m.cohens_kappa:.3f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results[variant] = {"error": str(e)}

    OUT_PATH.write_text(json.dumps(all_results, indent=2))
    _print_table(all_results)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--table":
        if OUT_PATH.exists():
            _print_table(json.loads(OUT_PATH.read_text()))
        else:
            print(f"No results yet. Run: python experiments/run_ablation.py <variant>")
            print(f"Variants: {VARIANTS}")
        sys.exit(0)
    asyncio.run(run_variant(sys.argv[1]))
