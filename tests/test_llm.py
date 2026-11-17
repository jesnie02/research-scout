import asyncio

import httpx
import httpx2
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.models.openai import OpenAIChatModel

from scout.config import Settings
from scout.llm import RETRY_WAITS, HttpCounter, RateLimitRetryTransport, model_settings, open_model

SETTINGS = Settings(model="some-model", base_url="https://api.example.com/v1", api_key="k")


def test_counter_counts_attempts_and_status_codes() -> None:
    counter = HttpCounter()
    request = httpx2.Request("POST", "https://api.example.com/v1/chat/completions")

    async def exchange(status: int) -> None:
        await counter.on_request(request)
        await counter.on_response(httpx2.Response(status, request=request))

    async def run() -> None:
        await exchange(429)
        await exchange(200)

    asyncio.run(run())
    assert counter.attempts == 2
    assert counter.statuses == {429: 1, 200: 1}
    assert counter.rate_limited == 1


def test_model_settings_send_parallel_tool_calls_only_when_supported() -> None:
    assert model_settings(SETTINGS) == {"temperature": 0.2, "parallel_tool_calls": False}
    cohere_like = Settings(
        model="m",
        base_url="https://api.example.com/v1",
        api_key="k",
        supports_parallel_tool_calls=False,
    )
    assert model_settings(cohere_like) == {"temperature": 0.2}


def test_open_model_builds_the_native_cohere_model() -> None:
    cohere = Settings(
        model="command-a-plus-05-2026",
        base_url="https://api.cohere.com",
        api_key="k",
        provider="cohere",
        supports_parallel_tool_calls=False,
    )

    async def run() -> str:
        async with open_model(cohere, HttpCounter()) as model:
            assert isinstance(model, CohereModel)
            return model.model_name

    assert asyncio.run(run()) == "command-a-plus-05-2026"


def test_open_model_builds_an_openai_compatible_model() -> None:
    async def run() -> str:
        async with open_model(SETTINGS, HttpCounter()) as model:
            assert isinstance(model, OpenAIChatModel)
            return model.model_name

    assert asyncio.run(run()) == "some-model"


def run_transport(statuses: list[tuple[int, dict[str, str]]]) -> tuple[int, list[float], int]:
    """Send one request through the retry transport; returns final status, waits, attempts."""
    replies = list(statuses)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        status, headers = replies.pop(0)
        return httpx.Response(status, headers=headers, json={"ok": status == 200})

    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    transport = RateLimitRetryTransport(httpx.MockTransport(handler), sleep=fake_sleep)

    async def run() -> int:
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.post("https://api.example.com/chat", json={"q": 1})
            return response.status_code

    return asyncio.run(run()), waits, attempts


def test_retry_transport_waits_on_rate_limits_then_succeeds() -> None:
    status, waits, attempts = run_transport([(429, {}), (503, {}), (200, {})])
    assert (status, attempts) == (200, 3)
    assert waits == list(RETRY_WAITS[:2])


def test_retry_transport_uses_retry_after_when_given() -> None:
    _, waits, _ = run_transport([(429, {"Retry-After": "7"}), (200, {})])
    assert waits == [7.0]


def test_retry_transport_gives_up_and_returns_the_last_response() -> None:
    status, waits, attempts = run_transport([(429, {})] * (len(RETRY_WAITS) + 1))
    assert status == 429
    assert attempts == len(RETRY_WAITS) + 1
    assert waits == list(RETRY_WAITS)


def test_retry_transport_does_not_retry_other_errors() -> None:
    status, waits, attempts = run_transport([(400, {})])
    assert (status, waits, attempts) == (400, [], 1)
