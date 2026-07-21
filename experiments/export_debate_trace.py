"""Export a full debate trace for a single case to a readable text file.

Picks the first case where the debate engine ran at least one round
(i.e. there was genuine conflict). Outputs to experiments/results/debate_trace.txt
"""
import asyncio
import json
from pathlib import Path

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
from src.protocols.debate_engine import ConsensusResult
from src.schemas.evidence_pack import EvidencePack

DATA = Path("src/data/processed")
OUT  = Path("experiments/results/debate_trace.txt")
OUT_T3 = Path("experiments/results/debate_trace_pathway_win.txt")
MODEL = "nim:meta/llama-3.3-70b-instruct"
GEMINI_MODEL = "gemini:gemini-2.5-flash"

# Override DebateEngine to capture per-round EvidencePacks
from src.protocols.debate_engine import DebateEngine
from src.protocols.axiom_resolver import AxiomResolver
from src.protocols.conflict_detector import ConflictDetector
from src.schemas.axiom_rules import MAX_DEBATE_ROUNDS
from dataclasses import dataclass, field
from src.schemas.evidence_pack import Verdict


@dataclass
class RichConsensusResult(ConsensusResult):
    round_snapshots: list[list[EvidencePack]] = field(default_factory=list)


class TracingDebateEngine(DebateEngine):
    def run(self, packs: list[EvidencePack]) -> RichConsensusResult:
        cell_line = packs[0].cell_line
        drug = packs[0].drug
        current = list(packs)
        trace = []
        dissenting = set()
        forced = False
        snapshots = [list(current)]

        for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
            if not self._detector.has_conflict(current):
                break
            resolution = self._resolver.resolve(current)
            trace.append({
                "round": round_num,
                "axiom_applied": resolution.axiom_applied,
                "winning_agent": resolution.winning_agent,
                "verdict": resolution.verdict.value,
            })
            for p in current:
                if p.agent_id != resolution.winning_agent and p.verdict != resolution.verdict:
                    dissenting.add(p.agent_id)
            current = self._update_packs(current, resolution)
            snapshots.append(list(current))
            if not self._detector.has_conflict(current):
                break
        else:
            forced = True

        final = self._resolver.resolve(current)
        avg_conf = sum(p.confidence for p in final.adjusted_packs) / len(final.adjusted_packs)

        return RichConsensusResult(
            final_verdict=final.verdict,
            final_confidence=avg_conf,
            cell_line=cell_line,
            drug=drug,
            winning_agent=final.winning_agent,
            rounds_taken=len(trace),
            forced=forced,
            dissenting_agents=sorted(dissenting),
            trace=trace,
            round_snapshots=snapshots,
        )


def _pack_summary(p: EvidencePack) -> str:
    findings = "; ".join(
        f"{f.biomarker}={f.value} ({f.interpretation})"
        for f in p.key_findings[:3]
    ) or "(no findings)"
    return (
        f"    {p.agent_id:<30} verdict={p.verdict.value:<12} "
        f"conf={p.confidence:.2f}  tier={p.evidence_tier.value}\n"
        f"      findings: {findings}\n"
        f"      caveats:  {'; '.join(p.caveats) or 'none'}"
    )


def write_trace(result: RichConsensusResult, case, out: Path) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append(f"DEBATE TRACE — {case.cell_line} + {case.drug}")
    lines.append(f"Ground truth label : {case.label}  (z={case.z_score})")
    lines.append(f"Notes              : {case.notes}")
    lines.append("=" * 72)

    lines.append("\n── ROUND 0: Independent Analysis (agents cannot see each other) ──")
    for p in result.round_snapshots[0]:
        lines.append(_pack_summary(p))

    for i, round_event in enumerate(result.trace, 1):
        lines.append(f"\n── ROUND {round_event['round']}: Axiom Resolution ──")
        lines.append(f"   Axiom applied  : {round_event['axiom_applied']}")
        lines.append(f"   Winning agent  : {round_event['winning_agent']}")
        lines.append(f"   Forced verdict : {round_event['verdict']}")
        if i < len(result.round_snapshots):
            lines.append("   Agent states after resolution:")
            for p in result.round_snapshots[i]:
                lines.append(_pack_summary(p))

    lines.append("\n" + "=" * 72)
    lines.append("FINAL VERDICT")
    lines.append("=" * 72)
    lines.append(f"  Verdict         : {result.final_verdict.value}")
    lines.append(f"  Confidence      : {result.final_confidence:.3f}")
    lines.append(f"  Winning agent   : {result.winning_agent}")
    lines.append(f"  Rounds taken    : {result.rounds_taken}")
    lines.append(f"  Forced          : {result.forced}")
    lines.append(f"  Dissenting      : {', '.join(result.dissenting_agents) or 'none'}")
    lines.append(f"  Correct         : {'YES' if result.final_verdict.value == case.label else 'NO'  }")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Trace written to {out}")


async def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "t1"
    use_model = GEMINI_MODEL if mode == "gemini" else MODEL
    cases = load_cases(Path("data/cases/cases.yaml"))
    client = LLMClient(model=use_model, cache_db=DATA / "llm_cache.db")
    engine = TracingDebateEngine()
    agents = [
        GenomicsAgent(genomics_app, client),
        TranscriptomicsAgent(transcriptomics_app, client),
        PharmacologyAgent(pharmacology_app, client),
        PathwayAgent(pathway_app, client, transcriptomics_mcp=transcriptomics_app),
    ]
    orch = Orchestrator(agents=agents, engine=engine)

    if mode == "gemini":
        # Pull HCC-827 + Gefitinib from Gemini cache — T1 structural win, 2 dissenters
        target_cases = [c for c in cases if c.cell_line == "HCC-827" and c.drug == "Gefitinib"]
        case = target_cases[0]
        targets = [g.strip() for g in case.putative_target.split(",")] if case.putative_target else None
        result = await orch.run_case(case.cell_line, case.drug, target_genes=targets)
        if isinstance(result, RichConsensusResult):
            print(f"Selected: {case.cell_line} + {case.drug} (Gemini 2.5 Flash)")
            write_trace(result, case, Path("experiments/results/debate_trace_gemini.txt"))
        return
    elif mode == "clean":
        # Find a case with SENSITIVE vs RESISTANT conflict (no UNCERTAIN agents)
        # NCI-H1975 + Osimertinib: genomics RESISTANT, transcriptomics/pharmacology SENSITIVE
        target = next((c for c in cases if c.cell_line == "NCI-H1975" and c.drug == "Osimertinib"), None)
        if target:
            tgts = [g.strip() for g in target.putative_target.split(",")] if target.putative_target else None
            result = await orch.run_case(target.cell_line, target.drug, target_genes=tgts)
            if isinstance(result, RichConsensusResult):
                print(f"Selected: {target.cell_line} + {target.drug}")
                write_trace(result, target, Path("experiments/results/debate_trace_clean.txt"))
        return
    elif mode == "t3":
        # Find a case where pathway agent wins (winning_agent == pathway_agent)
        t3_cases = [c for c in cases if c.axiom_tier == "T3_PATHWAY_BYPASS"]
        for case in t3_cases:
            targets = [g.strip() for g in case.putative_target.split(",")] if case.putative_target else None
            result = await orch.run_case(case.cell_line, case.drug, target_genes=targets)
            if isinstance(result, RichConsensusResult) and result.winning_agent == "pathway_agent":
                print(f"Selected: {case.cell_line} + {case.drug} (pathway agent won, {result.rounds_taken} rounds)")
                write_trace(result, case, OUT_T3)
                return
        # Fallback: just pick first T3 case with any debate
        for case in t3_cases:
            targets = [g.strip() for g in case.putative_target.split(",")] if case.putative_target else None
            result = await orch.run_case(case.cell_line, case.drug, target_genes=targets)
            if isinstance(result, RichConsensusResult) and result.rounds_taken >= 1:
                print(f"Selected (fallback): {case.cell_line} + {case.drug} ({result.rounds_taken} rounds)")
                write_trace(result, case, OUT_T3)
                return
        print("No T3 case with pathway win found.")
    else:
        # Default: first case with at least 1 debate round
        for case in cases:
            targets = [g.strip() for g in case.putative_target.split(",")] if case.putative_target else None
            result = await orch.run_case(case.cell_line, case.drug, target_genes=targets)
            if isinstance(result, RichConsensusResult) and result.rounds_taken >= 1:
                print(f"Selected: {case.cell_line} + {case.drug} ({result.rounds_taken} debate rounds)")
                write_trace(result, case, OUT)
                return
        print("No debate found — writing first case.")
        case = cases[0]
        targets = [g.strip() for g in case.putative_target.split(",")] if case.putative_target else None
        result = await orch.run_case(case.cell_line, case.drug, target_genes=targets)
        write_trace(result, case, OUT)


if __name__ == "__main__":
    asyncio.run(main())
