"""Model connection for the free-tier providers (OpenAI-compatible pattern from the W44 starter).

Cohere uses PydanticAI's native Cohere model, because its OpenAI-compatible API rejects
tool_choice.

Two retry layers exist: the OpenAI client retries 429/5xx itself (`max_retries`), while
PydanticAI's `request_limit` only counts agent turns. HttpCounter makes the transport visible.
"""

import asyncio
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import httpx2
from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.cohere import CohereProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from scout.config import Settings


@dataclass
class HttpCounter:
    """Counts every HTTP attempt to the model API, including the client's own retries."""

    attempts: int = 0
    statuses: Counter[int] = field(default_factory=Counter)

    async def on_request(self, request: httpx.Request | httpx2.Request) -> None:
        self.attempts += 1

    async def on_response(self, response: httpx.Response | httpx2.Response) -> None:
        self.statuses[response.status_code] += 1

    @property
    def rate_limited(self) -> int:
        return self.statuses[429]


# Cohere's trial key allows 20 calls/minute; these waits (60 s in total) cover a full window.
RETRY_WAITS = (5.0, 15.0, 40.0)
RETRY_STATUSES = {429, 500, 502, 503, 504}


class RateLimitRetryTransport(httpx.AsyncBaseTransport):
    """Waits and retries on 429/5xx (Retry-After if given). Used for the cohere SDK's httpx client,
    whose own retries do not wait long enough for a per-minute limit."""

    def __init__(
        self,
        wrapped: httpx.AsyncBaseTransport,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.wrapped = wrapped
        self.sleep = sleep

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        for wait in (*RETRY_WAITS, None):
            response = await self.wrapped.handle_async_request(request)
            if response.status_code not in RETRY_STATUSES or wait is None:
                return response
            header = response.headers.get("Retry-After", "")
            await response.aclose()
            await self.sleep(float(header) if header.isdigit() else wait)
        raise AssertionError("unreachable")  # pragma: no cover

    async def aclose(self) -> None:
        await self.wrapped.aclose()


@asynccontextmanager
async def open_model(settings: Settings, counter: HttpCounter) -> AsyncIterator[Model]:
    hooks: dict[str, list[Callable[..., Any]]] = {
        "request": [counter.on_request],
        "response": [counter.on_response],
    }
    if settings.provider == "cohere":
        # The cohere SDK uses httpx. The counter's hooks see one attempt per SDK request;
        # waits for rate limits happen inside RateLimitRetryTransport.
        transport = RateLimitRetryTransport(httpx.AsyncHTTPTransport())
        async with httpx.AsyncClient(
            timeout=settings.timeout, event_hooks=hooks, transport=transport
        ) as cohere_http:
            provider = CohereProvider(api_key=settings.api_key, http_client=cohere_http)
            yield CohereModel(settings.model, provider=provider)
        return
    async with (
        httpx2.AsyncClient(timeout=settings.timeout, event_hooks=hooks) as http,
        AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=settings.max_retries,
            timeout=settings.timeout,
            http_client=http,
        ) as client,
    ):
        yield OpenAIChatModel(settings.model, provider=OpenAIProvider(openai_client=client))


def model_settings(settings: Settings) -> ModelSettings:
    """Run settings shared by the measurement and the agent (one tool call at a time, as in W44)."""
    result: ModelSettings = {"temperature": settings.temperature}
    if settings.supports_parallel_tool_calls:
        result["parallel_tool_calls"] = False
    return result
