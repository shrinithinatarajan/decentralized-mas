import importlib
import json
import pytest
from unittest.mock import AsyncMock, patch

from src.schemas.evidence_pack import EvidencePack, Verdict, EvidenceTier


# --- helpers ---

def _pack_json(verdict="SENSITIVE", tier="T1_STRUCTURAL", agent_id="genomics_agent") -> str:
    return json.dumps({
        "agent_id": agent_id,
        "cell_line": "A375",
        "drug": "Vemurafenib",
        "verdict": verdict,
        "confidence": 0.85,
        "evidence_tier": tier,
        "key_findings": [
            {
                "biomarker": "BRAF_V600E",
                "value": "mutant",
                "interpretation": "Driver mutation; drug target active",
                "data_source": "genomics_db",
                "axiom_invoked": "CENTRAL_DOGMA_DNA_PRIMACY",
            }
        ],
        "caveats": [],
        "conflict_flags": [],
    })


# --- BaseAgent / parse robustness ---

@pytest.mark.asyncio
async def test_base_agent_parses_plain_json(genomics_db, monkeypatch):
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)
    from src.agents.genomics_agent import GenomicsAgent

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = AsyncMock(return_value=_pack_json())
        agent = GenomicsAgent(mcp_app=genomics_server.mcp)
        pack = await agent.analyze("A375", "Vemurafenib")

    assert isinstance(pack, EvidencePack)
    assert pack.verdict == Verdict.SENSITIVE


@pytest.mark.asyncio
async def test_base_agent_parses_json_in_markdown_code_block(genomics_db, monkeypatch):
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)
    from src.agents.genomics_agent import GenomicsAgent

    wrapped = f"```json\n{_pack_json()}\n```"
    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = AsyncMock(return_value=wrapped)
        agent = GenomicsAgent(mcp_app=genomics_server.mcp)
        pack = await agent.analyze("A375", "Vemurafenib")

    assert pack.verdict == Verdict.SENSITIVE


# --- GenomicsAgent ---

@pytest.mark.asyncio
async def test_genomics_agent_uses_t1_tier(genomics_db, monkeypatch):
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)
    from src.agents.genomics_agent import GenomicsAgent

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = AsyncMock(
            return_value=_pack_json(tier="T1_STRUCTURAL", agent_id="genomics_agent")
        )
        agent = GenomicsAgent(mcp_app=genomics_server.mcp)
        pack = await agent.analyze("A375", "Vemurafenib")

    assert pack.agent_id == "genomics_agent"
    assert pack.evidence_tier == EvidenceTier.T1_STRUCTURAL


@pytest.mark.asyncio
async def test_genomics_agent_prompt_includes_mutation_data(genomics_db, monkeypatch):
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)
    from src.agents.genomics_agent import GenomicsAgent

    captured_messages = []

    async def capture(messages, system=""):
        captured_messages.extend(messages)
        return _pack_json()

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = capture
        agent = GenomicsAgent(mcp_app=genomics_server.mcp)
        await agent.analyze("A375", "Vemurafenib")

    prompt_text = captured_messages[0]["content"]
    assert "BRAF" in prompt_text
    assert "V600E" in prompt_text


# --- TranscriptomicsAgent ---

@pytest.mark.asyncio
async def test_transcriptomics_agent_uses_t2_tier(transcriptomics_db, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOMICS_DB", str(transcriptomics_db))
    from src.mcp_servers import transcriptomics_server
    importlib.reload(transcriptomics_server)
    from src.agents.transcriptomics_agent import TranscriptomicsAgent

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = AsyncMock(
            return_value=_pack_json(tier="T2_TRANSCRIPTIONAL", agent_id="transcriptomics_agent")
        )
        agent = TranscriptomicsAgent(mcp_app=transcriptomics_server.mcp)
        pack = await agent.analyze("A375", "Vemurafenib")

    assert pack.agent_id == "transcriptomics_agent"
    assert pack.evidence_tier == EvidenceTier.T2_TRANSCRIPTIONAL


@pytest.mark.asyncio
async def test_transcriptomics_agent_prompt_includes_expression_data(transcriptomics_db, monkeypatch):
    monkeypatch.setenv("TRANSCRIPTOMICS_DB", str(transcriptomics_db))
    from src.mcp_servers import transcriptomics_server
    importlib.reload(transcriptomics_server)
    from src.agents.transcriptomics_agent import TranscriptomicsAgent

    captured_messages = []

    async def capture(messages, system=""):
        captured_messages.extend(messages)
        return _pack_json(tier="T2_TRANSCRIPTIONAL", agent_id="transcriptomics_agent")

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = capture
        agent = TranscriptomicsAgent(mcp_app=transcriptomics_server.mcp)
        await agent.analyze("A375", "Vemurafenib")

    prompt_text = captured_messages[0]["content"]
    assert "BRAF" in prompt_text or "EGFR" in prompt_text


# --- PharmacologyAgent ---

@pytest.mark.asyncio
async def test_pharmacology_agent_uses_t4_tier(pharmacology_db, monkeypatch):
    monkeypatch.setenv("PHARMACOLOGY_DB", str(pharmacology_db))
    from src.mcp_servers import pharmacology_server
    importlib.reload(pharmacology_server)
    from src.agents.pharmacology_agent import PharmacologyAgent

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = AsyncMock(
            return_value=_pack_json(tier="T4_PHARMACOLOGICAL", agent_id="pharmacology_agent")
        )
        agent = PharmacologyAgent(mcp_app=pharmacology_server.mcp)
        pack = await agent.analyze("A375", "Vemurafenib")

    assert pack.agent_id == "pharmacology_agent"
    assert pack.evidence_tier == EvidenceTier.T4_PHARMACOLOGICAL


@pytest.mark.asyncio
async def test_pharmacology_agent_prompt_includes_ic50_data(pharmacology_db, monkeypatch):
    monkeypatch.setenv("PHARMACOLOGY_DB", str(pharmacology_db))
    from src.mcp_servers import pharmacology_server
    importlib.reload(pharmacology_server)
    from src.agents.pharmacology_agent import PharmacologyAgent

    captured_messages = []

    async def capture(messages, system=""):
        captured_messages.extend(messages)
        return _pack_json(tier="T4_PHARMACOLOGICAL", agent_id="pharmacology_agent")

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = capture
        agent = PharmacologyAgent(mcp_app=pharmacology_server.mcp)
        await agent.analyze("A375", "Vemurafenib")

    prompt_text = captured_messages[0]["content"]
    assert "Vemurafenib" in prompt_text
    assert "ln_ic50" in prompt_text or "ic50" in prompt_text.lower()


# --- PathwayAgent ---

@pytest.mark.asyncio
async def test_pathway_agent_uses_t3_tier(pathways_db, monkeypatch):
    monkeypatch.setenv("PATHWAYS_DB", str(pathways_db))
    from src.mcp_servers import pathway_server
    importlib.reload(pathway_server)
    from src.agents.pathway_agent import PathwayAgent

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = AsyncMock(
            return_value=_pack_json(tier="T3_PATHWAY", agent_id="pathway_agent")
        )
        agent = PathwayAgent(mcp_app=pathway_server.mcp)
        pack = await agent.analyze("A375", "Vemurafenib")

    assert pack.agent_id == "pathway_agent"
    assert pack.evidence_tier == EvidenceTier.T3_PATHWAY


@pytest.mark.asyncio
async def test_pathway_agent_prompt_includes_pathway_data(pathways_db, monkeypatch):
    monkeypatch.setenv("PATHWAYS_DB", str(pathways_db))
    from src.mcp_servers import pathway_server
    importlib.reload(pathway_server)
    from src.agents.pathway_agent import PathwayAgent

    captured_messages = []

    async def capture(messages, system=""):
        captured_messages.extend(messages)
        return _pack_json(tier="T3_PATHWAY", agent_id="pathway_agent")

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = capture
        agent = PathwayAgent(mcp_app=pathway_server.mcp)
        await agent.analyze("A375", "Vemurafenib")

    prompt_text = captured_messages[0]["content"]
    assert "bypass_check" in prompt_text or "pathway_membership" in prompt_text
