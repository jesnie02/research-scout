# Research Scout Agent

A PydanticAI agent that finds research papers matching a topic, year and citation constraints,
using OpenAlex as its data source and a free-tier LLM. Work in progress (Compulsory Assignment #2).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync                 # installs dependencies from uv.lock
cp .env.example .env    # then fill in your keys; .env is git-ignored
uv run pytest           # tests run offline, no keys needed
```

## Run the agent

Set `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` and `OPENALEX_API_KEY` in `.env` (see `.env.example`).
The measured choice is Cohere Command A+ (`LLM_BASE_URL=https://api.cohere.com`,
`LLM_MODEL=command-a-plus-05-2026`).

```bash
uv run python main.py "Find a paper about retrieval-augmented generation published before 2021 with more than 500 citations. Summarize its contribution in 5-7 sentences."
uv run python main.py "..." --json   # full structured result
```

## Evaluation cases

Cases live in `eval/cases.json` (format: see `eval/cases.example.json`). Write the expectations
before running the agent. Check the file with:

```bash
uv run python -m eval.check_cases
```

## Step 0: measurements

```bash
uv run python -m scripts.check_openalex                       # needs OPENALEX_API_KEY
uv run python -m scripts.check_model --preset cohere          # needs COHERE_API_KEY
uv run python -m scripts.check_model --preset groq --list-models
uv run python -m scripts.check_model --preset groq --model <model-id>
```

Results are saved in `scripts/results/` (git-ignored).

## Development

```bash
uv run ruff format . && uv run ruff check .
uv run mypy --strict
```
