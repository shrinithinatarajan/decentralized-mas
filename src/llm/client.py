import os
from pathlib import Path

from src.llm.cache import ResponseCache

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

try:
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore

_ANTHROPIC_PREFIXES = ("claude",)


class LLMClient:
    def __init__(self, model: str | None = None, cache_db: Path | None = None) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o")
        self._cache = ResponseCache(cache_db or Path("src/data/processed/llm_cache.db"))

    def _is_anthropic(self) -> bool:
        return any(self.model.startswith(p) for p in _ANTHROPIC_PREFIXES)

    async def complete(self, messages: list[dict], system: str = "") -> str:
        cached = self._cache.get(self.model, messages, system)
        if cached is not None:
            return cached

        response = await self._call_api(messages, system)
        self._cache.set(self.model, messages, response, system)
        return response

    async def _call_api(self, messages: list[dict], system: str) -> str:
        if self._is_anthropic():
            return await self._call_anthropic(messages, system)
        return await self._call_openai(messages, system)

    async def _call_openai(self, messages: list[dict], system: str) -> str:
        all_messages = ([{"role": "system", "content": system}] if system else []) + messages
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = await client.chat.completions.create(model=self.model, messages=all_messages)
        return resp.choices[0].message.content

    async def _call_anthropic(self, messages: list[dict], system: str) -> str:
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        kwargs = {"model": self.model, "max_tokens": 4096, "messages": messages}
        if system:
            kwargs["system"] = system
        resp = await client.messages.create(**kwargs)
        return resp.content[0].text
