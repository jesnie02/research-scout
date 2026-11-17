"""Model connection for the free-tier providers (OpenAI-compatible pattern from the W44 starter).

Cohere uses PydanticAI's native Cohere model, because its OpenAI-compatible API rejects
tool_choice.

Two retry layers exist: the OpenAI client retries 429/5xx itself (`max_retries`), while
PydanticAI's `request_limit` only counts agent turns. HttpCounter makes the transport visible.
"""

from collections import Counter
from collections.abc import AsyncIterator, Callable
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


@asynccontextmanager
async def open_model(settings: Settings, counter: HttpCounter) -> AsyncIterator[Model]:
    hooks: dict[str, list[Callable[..., Any]]] = {
        "request": [counter.on_request],
        "response": [counter.on_response],
    }
    if settings.provider == "cohere":
        # The cohere SDK uses httpx; its own retries are not configurable here, so the
        # counter shows every attempt instead.
        async with httpx.AsyncClient(timeout=settings.timeout, event_hooks=hooks) as cohere_http:
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
