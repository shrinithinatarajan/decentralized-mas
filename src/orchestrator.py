import asyncio
import json
from pathlib import Path

from src.data.gene_aliases import GENE_ALIASES, SKIP_TARGETS
from src.protocols.debate_engine import ConsensusResult, DebateEngine
from src.schemas.evidence_pack import EvidencePack


def _normalize_targets(raw: list[str]) -> list[str] | None:
    result: list[str] = []
    for t in raw:
        t = t.strip()
        if t in SKIP_TARGETS:
            continue
        alias = GENE_ALIASES.get(t, t)
        if isinstance(alias, list):
            result.extend(alias)
        else:
            result.append(alias)
    return result if result else None

# Agents run concurrently per case; shared rate limiter keeps API quota safe.
SEQUENTIAL_AGENTS = False


class Orchestrator:
    def __init__(self, agents: list, engine: DebateEngine | None = None) -> None:
        self.agents = agents
        self.engine = engine or DebateEngine()

    async def run_case(self, cell_line: str, drug: str, target_genes: list[str] | None = None) -> ConsensusResult:
        if SEQUENTIAL_AGENTS:
            packs: list[EvidencePack] = []
            for agent in self.agents:
                packs.append(await agent.analyze(cell_line, drug, target_genes))
        else:
            packs = list(await asyncio.gather(*[a.analyze(cell_line, drug, target_genes) for a in self.agents]))
        return self.engine.run(packs)

    async def run_all(
        self,
        cases,
        output_path: Path | None = None,
        traces_path: Path | None = None,
    ) -> list[ConsensusResult]:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        if traces_path is not None:
            traces_path.parent.mkdir(parents=True, exist_ok=True)
            traces_path.write_text("")  # reset file each run

        results = []
        for i, case in enumerate(cases, 1):
            raw_targets = [g.strip() for g in case.putative_target.split(",")] if case.putative_target else None
            targets = _normalize_targets(raw_targets) if raw_targets else None
            print(f"  [{i}/{len(cases)}] {case.cell_line} + {case.drug}", flush=True)
            result = await self.run_case(case.cell_line, case.drug, target_genes=targets)
            results.append(result)
            record = _result_to_dict(result)
            record["true_label"] = case.label
            record["axiom_tier"] = case.axiom_tier
            if output_path is not None:
                with output_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
            if traces_path is not None:
                with traces_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")

        return results


def _result_to_dict(r: ConsensusResult) -> dict:
    return {
        "cell_line": r.cell_line,
        "drug": r.drug,
        "final_verdict": r.final_verdict.value,
        "final_confidence": r.final_confidence,
        "winning_agent": r.winning_agent,
        "rounds_taken": r.rounds_taken,
        "forced": r.forced,
        "dissenting_agents": r.dissenting_agents,
        "trace": r.trace,
    }
