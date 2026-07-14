"""RED phase: tests for chain-of-thought prompting."""
import json
import importlib
import pytest
from unittest.mock import AsyncMock, patch


# ── schema includes reasoning field ─────────────────────────────────────────

def test_schema_example_has_reasoning_field():
    """Prompt schema must include a reasoning field to elicit chain-of-thought."""
    from src.agents.base_agent import _SCHEMA_EXAMPLE
    schema = json.loads(_SCHEMA_EXAMPLE)
    assert "reasoning" in schema, "_SCHEMA_EXAMPLE must include 'reasoning' for CoT"


def test_reasoning_field_comes_before_verdict_in_schema():
    """reasoning must appear before verdict so LLM thinks before deciding."""
    from src.agents.base_agent import _SCHEMA_EXAMPLE
    keys = list(json.loads(_SCHEMA_EXAMPLE).keys())
    assert keys.index("reasoning") < keys.index("verdict")


# ── parse_pack handles reasoning field ────────────────────────────────────────

@pytest.mark.asyncio
async def test_parse_pack_ignores_reasoning_field(genomics_db, monkeypatch):
    """analyze() must succeed when LLM response includes a reasoning field."""
    monkeypatch.setenv("GENOMICS_DB", str(genomics_db))
    from src.mcp_servers import genomics_server
    importlib.reload(genomics_server)
    from src.agents.genomics_agent import GenomicsAgent
    from src.schemas.evidence_pack import Verdict

    pack_with_reasoning = json.dumps({
        "agent_id": "genomics_agent",
        "cell_line": "A375",
        "drug": "Vemurafenib",
        "reasoning": "BRAF V600E is a known driver mutation. The drug targets BRAF. Therefore verdict is SENSITIVE.",
        "verdict": "SENSITIVE",
        "confidence": 0.9,
        "evidence_tier": "T1_STRUCTURAL",
        "key_findings": [
            {"biomarker": "BRAF", "value": "V600E", "interpretation": "driver", "data_source": "db"}
        ],
        "caveats": [],
        "conflict_flags": [],
    })

    with patch("src.agents.base_agent.LLMClient") as MockLLM:
        MockLLM.return_value.complete = AsyncMock(return_value=pack_with_reasoning)
        agent = GenomicsAgent(mcp_app=genomics_server.mcp)
        pack = await agent.analyze("A375", "Vemurafenib")

    assert pack.verdict == Verdict.SENSITIVE


# ── system prompts contain CoT instructions ─────────────────────────────────

def test_genomics_system_prompt_has_cot_instructions():
    """Genomics agent system prompt must include step-by-step reasoning instructions."""
    from src.agents import genomics_agent
    prompt = genomics_agent._SYSTEM.lower()
    assert "reasoning" in prompt or "step" in prompt, \
        "Genomics system prompt must instruct chain-of-thought reasoning"


def test_genomics_system_prompt_addresses_wildtype_bias():
    """Genomics prompt must warn that wild-type does NOT always mean RESISTANT."""
    from src.agents import genomics_agent
    prompt = genomics_agent._SYSTEM.lower()
    assert "wild-type" in prompt or "wild type" in prompt or "absence" in prompt, \
        "Genomics prompt must address wild-type vs resistant conflation"


def test_transcriptomics_system_prompt_has_cot_instructions():
    from src.agents import transcriptomics_agent
    prompt = transcriptomics_agent._SYSTEM.lower()
    assert "reasoning" in prompt or "step" in prompt


def test_pharmacology_system_prompt_has_cot_instructions():
    from src.agents import pharmacology_agent
    prompt = pharmacology_agent._SYSTEM.lower()
    assert "reasoning" in prompt or "step" in prompt


def test_pathway_system_prompt_has_cot_instructions():
    from src.agents import pathway_agent
    prompt = pathway_agent._SYSTEM.lower()
    assert "reasoning" in prompt or "step" in prompt


def test_pharmacology_prompt_anchors_to_ic50():
    """Pharmacology agent must be instructed to anchor its verdict to IC50 data."""
    from src.agents import pharmacology_agent
    prompt = pharmacology_agent._SYSTEM.lower()
    assert "ic50" in prompt or "z_score" in prompt or "label" in prompt, \
        "Pharmacology prompt must reference IC50 as primary anchor"
