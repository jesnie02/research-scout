"""Step 0: measure tool calling + structured output on a free-tier model.

Each run gives the model a request and one tool that returns fixed, synthetic candidates
(no OpenAlex). Exactly one candidate fits, or none does (the model must then answer null).
Python checks the answer; the model's own claims are not trusted.

  uv run python -m scripts.check_model --preset groq --list-models
  uv run python -m scripts.check_model --preset cohere
  uv run python -m scripts.check_model --preset groq --model <id from the list>
  uv run python -m scripts.check_model --from-env --runs 1   # the agent's LLM_* settings

Keys come from .env (COHERE_API_KEY / GROQ_API_KEY). Results go to scripts/results/.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel
from pydantic_ai import Agent, PromptedOutput, RunContext, capture_run_messages
from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import UsageLimits

from scout.config import PRESETS, Settings, load_dotenv
from scout.llm import HttpCounter, model_settings, open_model

RESULTS = Path(__file__).resolve().parent / "results"

INSTRUCTIONS = """You help find research papers.
Always call lookup_papers first. Choose the ONE candidate that matches the topic AND every
year and citation constraint in the request. Use only the candidates the tool returned.
If no candidate satisfies everything, return paper_id null and explain why.
Never invent a paper or an ID."""


class Pick(BaseModel):
    paper_id: str | None
    explanation: str


@dataclass
class Task:
    request: str
    candidates: list[dict[str, Any]]
    correct: str | None
    tool_calls: int = 0


def paper(pid: str, title: str, year: int, citations: int) -> dict[str, Any]:
    return {"id": pid, "title": title, "year": year, "citations": citations}


# Synthetic candidates for a controlled test; they are not real papers.
TASKS: list[tuple[str, list[dict[str, Any]], str | None]] = [
    (
        "Find a paper about LLM agents for code generation published after 2022 "
        "with at least 100 citations.",
        [
            paper("P1", "LLM agents that write and repair code", 2021, 300),
            paper("P2", "Autonomous LLM agents for code generation", 2023, 150),
            paper("P3", "Tool-using agents for program synthesis", 2024, 40),
            paper("P4", "Transformers for image segmentation", 2023, 500),
        ],
        "P2",
    ),
    (
        "Find a paper about retrieval-augmented question answering published before 2021 "
        "with more than 500 citations.",
        [
            paper("P1", "Retrieval-augmented generation for open-domain QA", 2020, 900),
            paper("P2", "Scaling retrieval-augmented question answering", 2022, 2000),
            paper("P3", "Dense retrieval for question answering", 2019, 120),
            paper("P4", "End-to-end speech recognition", 2018, 3000),
        ],
        "P1",
    ),
    (
        "Find a paper about language models that learn to use tools, published in 2023.",
        [
            paper("P1", "Language models calling external APIs", 2022, 800),
            paper("P2", "Teaching language models to use tools", 2023, 1500),
            paper("P3", "Tool learning with foundation models", 2024, 400),
            paper("P4", "Protein structure prediction at scale", 2023, 2500),
        ],
        "P2",
    ),
    (
        "Find a paper about reinforcement learning for robot manipulation "
        "with at most 50 citations.",
        [
            paper("P1", "Reinforcement learning for dialogue systems", 2020, 10),
            paper("P2", "Sample-efficient RL for robotic grasping", 2022, 30),
            paper("P3", "Deep RL for robot manipulation", 2019, 200),
            paper("P4", "Offline RL for robot arms", 2021, 51),
        ],
        "P2",
    ),
    (
        "Find a paper about quantum chemistry simulations with more than 10000 citations.",
        [
            paper("P1", "Quantum chemistry on near-term hardware", 2019, 900),
            paper("P2", "Machine learning for quantum chemistry", 2020, 1200),
            paper("P3", "Density functional theory benchmarks", 2015, 4000),
        ],
        None,
    ),
]


def build_agent() -> Agent[Task, Pick]:
    agent: Agent[Task, Pick] = Agent(
        deps_type=Task,
        # Prompted JSON output (as in W44) for both providers: no forced tool_choice, which
        # Cohere rejects, and the same output mode makes the comparison fair.
        output_type=PromptedOutput(Pick),
        instructions=INSTRUCTIONS,
        retries={"output": 2},
    )

    @agent.tool(sequential=True)
    def lookup_papers(ctx: RunContext[Task], query: str) -> list[dict[str, Any]]:
        """Search for candidate papers. Returns id, title, year and citation count."""
        ctx.deps.tool_calls += 1
        return ctx.deps.candidates

    return agent


async def one_run(agent: Agent[Task, Pick], settings: Settings, number: int) -> dict[str, Any]:
    request, candidates, correct = TASKS[number % len(TASKS)]
    task = Task(request, candidates, correct)
    counter = HttpCounter()
    record: dict[str, Any] = {"run": number + 1, "task": number % len(TASKS) + 1}
    started = time.monotonic()
    with capture_run_messages() as messages:
        try:
            async with open_model(settings, counter) as model:
                result = await agent.run(
                    request,
                    model=model,
                    deps=task,
                    usage_limits=UsageLimits(request_limit=8, tool_calls_limit=3),
                    model_settings=model_settings(settings),
                )
            record["picked"] = result.output.paper_id
            record["explanation"] = result.output.explanation
            record["correct"] = task.tool_calls > 0 and result.output.paper_id == correct
        except Exception as error:  # a measurement records every failure type
            record["correct"] = False
            record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
    record["expected"] = correct
    record["tool_calls"] = task.tool_calls
    record["model_requests"] = sum(isinstance(m, ModelResponse) for m in messages)
    record["http_attempts"] = counter.attempts
    record["http_statuses"] = dict(counter.statuses)
    record["seconds"] = round(time.monotonic() - started, 2)
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    seconds = [r["seconds"] for r in records]
    return {
        "runs": len(records),
        "correct": sum(r["correct"] for r in records),
        "errors": sum("error" in r for r in records),
        "model_requests_total": sum(r["model_requests"] for r in records),
        "http_attempts_total": sum(r["http_attempts"] for r in records),
        "http_429_total": sum(r["http_statuses"].get(429, 0) for r in records),
        "http_5xx_total": sum(
            n for r in records for code, n in r["http_statuses"].items() if code >= 500
        ),
        "seconds_median": statistics.median(seconds),
        "seconds_max": max(seconds),
    }


def list_models(preset: str) -> None:
    load_dotenv()
    chosen = PRESETS[preset]
    if chosen.provider != "openai_compatible":
        raise SystemExit(f"--list-models only works for OpenAI-compatible presets, not {preset}.")
    key = os.getenv(chosen.key_variable, "")
    if not key:
        raise SystemExit(f"{chosen.key_variable} is empty. Add it to .env.")
    response = requests.get(
        f"{chosen.base_url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20
    )
    if response.status_code != 200:
        raise SystemExit(f"HTTP {response.status_code}: {response.text[:300]}")
    for model_id in sorted(m["id"] for m in response.json().get("data", [])):
        print(model_id)


async def measure(settings: Settings, runs: int, pause: float) -> list[dict[str, Any]]:
    agent = build_agent()
    records = []
    for number in range(runs):
        record = await one_run(agent, settings, number)
        records.append(record)
        status = "OK  " if record["correct"] else "FAIL"
        detail = record.get("error") or f"picked {record['picked']} (expected {record['expected']})"
        print(
            f"{status} run {record['run']:>2} task {record['task']}: "
            f"{record['model_requests']} model req, {record['http_attempts']} HTTP, "
            f"{record['tool_calls']} tool calls, {record['seconds']}s | {detail}"
        )
        if number < runs - 1:
            await asyncio.sleep(pause)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=sorted(PRESETS))
    source.add_argument(
        "--from-env", action="store_true", help="measure the agent's LLM_* settings from .env"
    )
    parser.add_argument("--model", help="model id (required for groq; see --list-models)")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between runs")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        if not args.preset:
            parser.error("--list-models needs --preset")
        list_models(args.preset)
        return
    label = args.preset or "env"
    try:
        settings = (
            Settings.from_env() if args.from_env else Settings.from_preset(args.preset, args.model)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    records = asyncio.run(measure(settings, args.runs, args.pause))
    summary = summarize(records)
    print("\n" + json.dumps(summary, indent=1))

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"model_{label}_{settings.model.replace('/', '_')}_{stamp}.json"
    meta = {
        "preset": label,
        "model": settings.model,
        "host": urlsplit(settings.base_url).hostname,
        "temperature": settings.temperature,
        "max_retries": settings.max_retries,
        "date": stamp,
    }
    path.write_text(json.dumps({"meta": meta, "summary": summary, "runs": records}, indent=1))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
