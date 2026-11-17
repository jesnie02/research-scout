import asyncio

import httpx2
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.models.openai import OpenAIChatModel

from scout.config import Settings
from scout.llm import HttpCounter, model_settings, open_model

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
