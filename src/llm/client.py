import asyncio
import os
import time
from pathlib import Path

from src.llm.cache import ResponseCache


class RateLimiter:
    """Token-bucket rate limiter: max `calls` API calls per `period` seconds."""
    def __init__(self, calls: int = 10, period: float = 60.0) -> None:
        self._calls = calls
        self._period = period
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < self._period]
            if len(self._timestamps) >= self._calls:
                wait = self._period - (now - self._timestamps[0])
                if wait > 0:
                    await asyncio.sleep(wait)
            self._timestamps.append(time.monotonic())


def make_rate_limiter(calls_per_minute: int | None = None) -> RateLimiter:
    n = calls_per_minute or int(os.getenv("LLM_CALLS_PER_MINUTE", "2"))
    return RateLimiter(calls=n, period=60.0)

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

try:
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore

_ANTHROPIC_PREFIXES = ("claude",)

_OPENAI_COMPATIBLE_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
    },
    "routeway": {
        "base_url": "https://api.routeway.ai/v1",
        "key_env": "ROUTEWAY_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "nim": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NIM_API_KEY",
    },
}


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        cache_db: Path | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o")
        self._cache = ResponseCache(cache_db or Path("src/data/processed/llm_cache.db"))
        self._limiter = rate_limiter  # None = no rate limiting (tests); pass make_rate_limiter() for production

    def _provider(self) -> str | None:
        if ":" in self.model:
            prefix = self.model.split(":", 1)[0]
            if prefix in _OPENAI_COMPATIBLE_PROVIDERS:
                return prefix
        return None

    def _api_model_name(self) -> str:
        if self._provider() is not None:
            return self.model.split(":", 1)[1]
        return self.model

    def _is_anthropic(self) -> bool:
        return any(self.model.startswith(p) for p in _ANTHROPIC_PREFIXES)

    async def complete(self, messages: list[dict], system: str = "") -> str:
        cached = self._cache.get(self.model, messages, system)
        if cached is not None:
            return cached

        response = await self._call_api(messages, system)
        if response:  # don't cache empty responses (token limit fallback)
            self._cache.set(self.model, messages, response, system)
        return response

    async def _call_api(self, messages: list[dict], system: str) -> str:
        if self._limiter:
            await self._limiter.acquire()
        for attempt in range(5):
            try:
                if self._is_anthropic():
                    coro = self._call_anthropic(messages, system)
                elif self.model.startswith("vertex:"):
                    coro = self._call_vertex(messages, system)
                elif self.model.startswith("gemini:"):
                    coro = self._call_gemini_native(messages, system)
                else:
                    coro = self._call_openai(messages, system)
                return await asyncio.wait_for(coro, timeout=120.0)
            except asyncio.TimeoutError:
                if attempt < 4:
                    await asyncio.sleep(10 * (attempt + 1))  # 10s, 20s, 30s, 40s
                    continue
                raise RuntimeError("LLM API timed out after 5 attempts")
            except Exception as e:
                status = getattr(e, "status_code", None) or getattr(e, "code", None)
                is_conn = "connection" in str(e).lower() or "nodename" in str(e).lower()
                is_quota = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
                is_transient = "ServiceUnavailable" in str(e) or "DeadlineExceeded" in str(e)
                if (status in (429, 500, 503) or is_conn or is_quota or is_transient) and attempt < 4:
                    wait = 2 ** attempt * 15  # 15s, 30s, 60s, 120s
                    await asyncio.sleep(wait)
                    continue
                if status == 400:
                    return ""  # context_length_exceeded — parse_pack will return UNCERTAIN
                raise

    async def _call_openai(self, messages: list[dict], system: str) -> str:
        all_messages = ([{"role": "system", "content": system}] if system else []) + messages

        provider = self._provider()
        if provider:
            cfg = _OPENAI_COMPATIBLE_PROVIDERS[provider]
            client = AsyncOpenAI(api_key=os.getenv(cfg["key_env"]), base_url=cfg["base_url"])
        else:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        kwargs: dict = {"model": self._api_model_name(), "messages": all_messages, "max_tokens": 1024}
        provider = self._provider()
        # Only add json_object mode for providers/models known to support it.
        # Mixtral and some older models silently fail or return empty responses with this flag.
        model_name = self._api_model_name().lower()
        supports_json_mode = provider in ("groq", "nim") and "mixtral" not in model_name
        if supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    async def _call_anthropic(self, messages: list[dict], system: str) -> str:
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        kwargs = {"model": self.model, "max_tokens": 4096, "messages": messages}
        if system:
            kwargs["system"] = system
        resp = await client.messages.create(**kwargs)
        return resp.content[0].text

    async def _call_gemini_native(self, messages: list[dict], system: str) -> str:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = self._api_model_name()
        contents = [types.Content(role=m["role"], parts=[types.Part(text=m["content"])]) for m in messages]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            response_mime_type="application/json",
            max_output_tokens=1024,
        )
        resp = await asyncio.to_thread(
            client.models.generate_content, model=model_name, contents=contents, config=config
        )
        return resp.text

    async def _call_vertex(self, messages: list[dict], system: str) -> str:
        """Google Cloud Vertex AI via REST API with Application Default Credentials.
        Model string format: vertex:gemini-2.5-flash
        Requires: gcloud auth application-default login
        """
        import subprocess
        import json as _json
        try:
            import aiohttp
        except ImportError:
            import subprocess as _sp
            _sp.run(["pip", "install", "aiohttp", "-q"], check=True)
            import aiohttp

        project = os.getenv("VERTEX_PROJECT")
        model_id = self.model.split(":", 1)[1]  # strip "vertex:" prefix

        # Get bearer token via gcloud
        token = await asyncio.to_thread(
            lambda: subprocess.check_output(
                ["gcloud", "auth", "application-default", "print-access-token"],
                text=True
            ).strip()
        )

        url = (
            f"https://aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/global/publishers/google/models/{model_id}:generateContent"
        )

        # Build request body in native Gemini format
        contents = [{"role": m["role"], "parts": [{"text": m["content"]}]} for m in messages]
        body: dict = {
            "contents": contents,
            "generationConfig": {
                "responseMimeType": "application/json",
                "maxOutputTokens": 1024,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers) as resp:
                data = await resp.json()
                if resp.status == 400:
                    return ""  # token limit or invalid argument — parse_pack returns UNCERTAIN
                if resp.status != 200:
                    raise RuntimeError(f"Vertex API error {resp.status}: {data}")
                return data["candidates"][0]["content"]["parts"][0]["text"]
