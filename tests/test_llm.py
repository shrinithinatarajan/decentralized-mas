import hashlib
import json
import sqlite3
import pytest
from unittest.mock import AsyncMock, patch
from pathlib import Path

from src.llm.cache import ResponseCache
from src.llm.client import LLMClient


# --- ResponseCache ---

def test_cache_miss_returns_none(tmp_path):
    cache = ResponseCache(tmp_path / "cache.db")
    assert cache.get("model", [{"role": "user", "content": "hi"}]) is None


def test_cache_set_then_get_returns_value(tmp_path):
    cache = ResponseCache(tmp_path / "cache.db")
    messages = [{"role": "user", "content": "what is BRAF?"}]
    cache.set("gpt-4o", messages, "BRAF is a kinase.")
    assert cache.get("gpt-4o", messages) == "BRAF is a kinase."


def test_cache_different_models_are_independent(tmp_path):
    cache = ResponseCache(tmp_path / "cache.db")
    messages = [{"role": "user", "content": "hello"}]
    cache.set("gpt-4o", messages, "openai response")
    assert cache.get("claude-sonnet-4-6", messages) is None


def test_cache_different_messages_are_independent(tmp_path):
    cache = ResponseCache(tmp_path / "cache.db")
    cache.set("gpt-4o", [{"role": "user", "content": "msg1"}], "r1")
    assert cache.get("gpt-4o", [{"role": "user", "content": "msg2"}]) is None


def test_cache_overwrites_existing_entry(tmp_path):
    cache = ResponseCache(tmp_path / "cache.db")
    messages = [{"role": "user", "content": "x"}]
    cache.set("gpt-4o", messages, "first")
    cache.set("gpt-4o", messages, "second")
    assert cache.get("gpt-4o", messages) == "second"


# --- LLMClient ---

@pytest.mark.asyncio
async def test_llm_client_calls_openai_and_returns_text(tmp_path):
    with patch("src.llm.client.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_openai_response("SENSITIVE"))
        MockOpenAI.return_value = mock_client

        client = LLMClient(model="gpt-4o", cache_db=tmp_path / "cache.db")
        result = await client.complete(
            messages=[{"role": "user", "content": "predict resistance"}],
            system="You are a genomics agent.",
        )
    assert result == "SENSITIVE"


@pytest.mark.asyncio
async def test_llm_client_caches_response(tmp_path):
    with patch("src.llm.client.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_openai_response("RESISTANT"))
        MockOpenAI.return_value = mock_client

        client = LLMClient(model="gpt-4o", cache_db=tmp_path / "cache.db")
        messages = [{"role": "user", "content": "predict"}]
        await client.complete(messages=messages, system="sys")
        await client.complete(messages=messages, system="sys")  # second call

    # API should be called only once — second hit served from cache
    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_llm_client_calls_anthropic_when_model_is_claude(tmp_path):
    with patch("src.llm.client.AsyncAnthropic") as MockAnthropic:
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=_anthropic_response("SENSITIVE"))
        MockAnthropic.return_value = mock_client

        client = LLMClient(model="claude-sonnet-4-6", cache_db=tmp_path / "cache.db")
        result = await client.complete(
            messages=[{"role": "user", "content": "predict"}],
            system="You are an agent.",
        )
    assert result == "SENSITIVE"


@pytest.mark.asyncio
async def test_llm_client_system_prompt_included_in_cache_key(tmp_path):
    with patch("src.llm.client.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[_openai_response("A"), _openai_response("B")]
        )
        MockOpenAI.return_value = mock_client

        client = LLMClient(model="gpt-4o", cache_db=tmp_path / "cache.db")
        messages = [{"role": "user", "content": "same"}]
        r1 = await client.complete(messages=messages, system="system A")
        r2 = await client.complete(messages=messages, system="system B")

    assert r1 == "A"
    assert r2 == "B"
    assert mock_client.chat.completions.create.call_count == 2


# --- helpers ---

def _openai_response(text: str):
    from types import SimpleNamespace
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _anthropic_response(text: str):
    from types import SimpleNamespace
    return SimpleNamespace(content=[SimpleNamespace(text=text)])
