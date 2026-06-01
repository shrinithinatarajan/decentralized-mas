import asyncio
import json
from pathlib import Path

from src.protocols.debate_engine import ConsensusResult, DebateEngine
from src.schemas.evidence_pack import EvidencePack


class Orchestrator:
    def __init__(self, agents: list, engine: DebateEngine | None = None) -> None:
        self.agents = agents
        self.engine = engine or DebateEngine()

    async def run_case(self, cell_line: str, drug: str) -> ConsensusResult:
        packs: list[EvidencePack] = list(
            await asyncio.gather(*[a.analyze(cell_line, drug) for a in self.agents])
        )
        return self.engine.run(packs)

    async def run_all(
        self,
        cases,
        output_path: Path | None = None,
    ) -> list[ConsensusResult]:
        results = []
        for case in cases:
            result = await self.run_case(case.cell_line, case.drug)
            results.append(result)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                for r in results:
                    f.write(json.dumps(_result_to_dict(r)) + "\n")

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
