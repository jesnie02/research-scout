"""Research Scout CLI: uv run python main.py "Find a paper about ..." [--json]"""

import argparse
import asyncio
import functools
from datetime import date

from scout.agent import run_scout
from scout.config import Settings, openalex_api_key
from scout.llm import HttpCounter, model_settings, open_model
from scout.models import ScoutResult
from scout.openalex import search_papers


async def answer(prompt: str) -> tuple[ScoutResult, HttpCounter]:
    settings = Settings.from_env()
    search = functools.partial(search_papers, api_key=openalex_api_key())
    counter = HttpCounter()
    async with open_model(settings, counter) as model:
        result = await run_scout(
            prompt, model, search=search, today=date.today(), settings=model_settings(settings)
        )
    return result, counter


def show(result: ScoutResult) -> None:
    print(f"Outcome: {result.outcome}\n{result.message}\n")
    if result.request:
        r = result.request
        print(f"Interpreted topic: {r.topic}")
        if r.note:
            print(f"Interpretation note: {r.note}")
    if result.intervals:
        print(f"Constraints: year {result.intervals.year}, citations {result.intervals.citations}")
        for note in result.intervals.notes:
            print(f"  {note}")
    paper = result.paper
    if paper:
        authors = ", ".join(paper.authors[:5]) + (" et al." if len(paper.authors) > 5 else "")
        print(f"\nTitle:      {paper.title}")
        print(f"Authors:    {authors}")
        print(f"Year:       {paper.year}")
        print(f"Citations:  {paper.citations}")
        print(f"Source:     {paper.citation_source}")
        print(f"Link:       {paper.url}")
        print(f"\n{paper.answer}")
        if paper.uncertainty:
            print(f"\nUncertainty: {paper.uncertainty}")
    print(
        f"\n[{len(result.searches)} search(es), {result.tool_calls} tool call(s), "
        f"{result.model_requests} model request(s)]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Find a research paper that meets constraints.")
    parser.add_argument("prompt")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()
    try:
        result, counter = asyncio.run(answer(args.prompt))
    except ValueError as error:  # missing keys or settings
        raise SystemExit(str(error)) from error
    if args.json:
        print(result.model_dump_json(indent=1, exclude={"raw_responses"}))
    else:
        show(result)
        print(f"[{counter.attempts} HTTP attempt(s) to the model API]")


if __name__ == "__main__":
    main()
