import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from src.schemas.evidence_pack import EvidencePack, Verdict, EvidenceTier, Finding
from src.protocols.debate_engine import ConsensusResult
from src.orchestrator import Orchestrator


def _pack(agent_id: str, verdict: str = "SENSITIVE", tier: str = "T1_STRUCTURAL") -> EvidencePack:
    return EvidencePack(
        agent_id=agent_id,
        cell_line="A375",
        drug="Vemurafenib",
        verdict=Verdict(verdict),
        confidence=0.8,
        evidence_tier=EvidenceTier(tier),
        key_findings=[Finding(biomarker="BRAF", value="V600E", interpretation="driver", data_source="test")],
    )


def _mock_agent(agent_id: str, verdict: str = "SENSITIVE", tier: str = "T1_STRUCTURAL"):
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.analyze = AsyncMock(return_value=_pack(agent_id, verdict, tier))
    return agent


@pytest.mark.asyncio
async def test_run_case_returns_consensus_result():
    agents = [
        _mock_agent("genomics_agent", "SENSITIVE", "T1_STRUCTURAL"),
        _mock_agent("transcriptomics_agent", "SENSITIVE", "T2_TRANSCRIPTIONAL"),
    ]
    orch = Orchestrator(agents=agents)
    result = await orch.run_case("A375", "Vemurafenib")
    assert isinstance(result, ConsensusResult)
    assert result.final_verdict == Verdict.SENSITIVE
    assert result.cell_line == "A375"
    assert result.drug == "Vemurafenib"


@pytest.mark.asyncio
async def test_run_case_calls_all_agents():
    agents = [
        _mock_agent("genomics_agent"),
        _mock_agent("transcriptomics_agent"),
        _mock_agent("pharmacology_agent"),
        _mock_agent("pathway_agent"),
    ]
    orch = Orchestrator(agents=agents)
    await orch.run_case("A375", "Vemurafenib")
    for agent in agents:
        agent.analyze.assert_called_once_with("A375", "Vemurafenib")


@pytest.mark.asyncio
async def test_run_case_agents_called_concurrently():
    # verify asyncio.gather is used: both agents start before either finishes
    call_order = []

    def make_agent(aid):
        a = MagicMock()
        a.agent_id = aid

        async def _analyze(cl, dr):
            call_order.append(("start", aid))
            await asyncio.sleep(0)  # yield to event loop
            call_order.append(("end", aid))
            return _pack(aid)

        a.analyze = _analyze
        return a

    agents = [make_agent("genomics_agent"), make_agent("transcriptomics_agent")]
    orch = Orchestrator(agents=agents)
    await orch.run_case("A375", "Vemurafenib")
    # with gather: both starts appear before both ends
    starts = [e for e in call_order if e[0] == "start"]
    assert len(starts) == 2


@pytest.mark.asyncio
async def test_run_all_processes_multiple_cases():
    from src.data.loader import Case
    agents = [
        _mock_agent("genomics_agent"),
        _mock_agent("transcriptomics_agent"),
    ]
    cases = [
        Case(cell_line="A375", drug="Vemurafenib", label="SENSITIVE"),
        Case(cell_line="HCC827", drug="Gefitinib", label="RESISTANT"),
    ]
    # patch analyze to handle different cell lines
    for agent in agents:
        async def make_analyze(aid):
            async def _analyze(cl, dr):
                return _pack(aid)
            return _analyze
        agent.analyze = AsyncMock(side_effect=lambda cl, dr, _id=agent.agent_id: _pack(_id))

    orch = Orchestrator(agents=agents)
    results = await orch.run_all(cases)
    assert len(results) == 2
    assert all(isinstance(r, ConsensusResult) for r in results)


@pytest.mark.asyncio
async def test_run_all_writes_jsonl(tmp_path):
    from src.data.loader import Case
    agents = [_mock_agent("genomics_agent")]
    cases = [Case(cell_line="A375", drug="Vemurafenib", label="SENSITIVE")]
    out = tmp_path / "results.jsonl"

    orch = Orchestrator(agents=agents)
    await orch.run_all(cases, output_path=out)

    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["cell_line"] == "A375"
    assert record["drug"] == "Vemurafenib"
    assert "final_verdict" in record
