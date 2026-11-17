"""Run the evaluation set and grade it.

  uv run python -m eval.run_eval run [--runs 3] [--pause 10] [--only C01,C02] [--out DIR]
  uv run python -m eval.run_eval report DIR      # after filling in the manual grades

Every result is saved as JSON (with the raw OpenAlex responses and abstracts) so grading can be
checked later. An interrupted run resumes: finished runs are read from disk, not repeated.
"""

import argparse
import asyncio
import functools
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests
from pydantic import TypeAdapter

from eval.cases import Case, load_cases
from eval.grading import Lookup, RunRecord, grade_run, render_report
from scout.agent import run_scout
from scout.config import Settings, openalex_api_key
from scout.llm import HttpCounter, model_settings, open_model
from scout.models import ScoutResult
from scout.openalex import search_papers

ROOT = Path(__file__).resolve().parent
ScoutFn = Callable[[str], Awaitable[ScoutResult]]
_RECORDS = TypeAdapter(list[RunRecord])


def _save(out_dir: Path, records: list[RunRecord]) -> None:
    (out_dir / "grading.json").write_text(
        json.dumps(_RECORDS.dump_python(records, mode="json"), indent=1)
    )
    (out_dir / "RESULTS.md").write_text(render_report(records))


def _load_records(out_dir: Path) -> list[RunRecord]:
    path = out_dir / "grading.json"
    return _RECORDS.validate_json(path.read_text()) if path.exists() else []


async def run_cases(
    cases: list[Case],
    scout: ScoutFn,
    out_dir: Path,
    *,
    runs: int,
    lookup: Lookup,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    pause: float = 10.0,
) -> list[RunRecord]:
    """Run every case `runs` times (pausing between live runs) and grade each result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = {(r.case_id, r.run): r for r in _load_records(out_dir)}
    records: list[RunRecord] = []
    for case in cases:
        for run in range(1, runs + 1):
            path = out_dir / f"{case.id}_r{run}.json"
            if path.exists():
                result = ScoutResult.model_validate_json(path.read_text())
            else:
                result = await scout(case.prompt)
                path.write_text(result.model_dump_json(indent=1))
                print(f"{case.id} run {run}: {result.outcome}", flush=True)
                await sleep(pause)
            record = grade_run(case, result, run=run, lookup=lookup, result_file=path.name)
            if (old := previous.get((case.id, run))) is not None:
                record = record.model_copy(update={"manual": old.manual})
            records.append(record)
            _save(out_dir, records)
    return records


def openalex_lookup(api_key: str, pid: str) -> dict[str, Any]:
    response = requests.get(
        f"https://api.openalex.org/works/{pid}",
        params={"select": "id,title,publication_year"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=20,
    )
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


async def live(cases: list[Case], out_dir: Path, runs: int, pause: float) -> list[RunRecord]:
    settings = Settings.from_env()
    key = openalex_api_key()
    search = functools.partial(search_papers, api_key=key)
    async with open_model(settings, HttpCounter()) as model:

        async def scout(prompt: str) -> ScoutResult:
            return await run_scout(
                prompt,
                model,
                search=search,
                today=date.today(),
                settings=model_settings(settings),
            )

        return await run_cases(
            cases,
            scout,
            out_dir,
            runs=runs,
            lookup=functools.partial(openalex_lookup, key),
            pause=pause,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--cases", type=Path, default=ROOT / "cases.json")
    run.add_argument("--runs", type=int, default=3)
    run.add_argument("--pause", type=float, default=10.0, help="seconds between live runs")
    run.add_argument("--only", help="comma-separated case IDs, e.g. C01,C03")
    run.add_argument("--out", type=Path, help="results folder (reuse one to resume)")
    report = sub.add_parser("report")
    report.add_argument("out", type=Path)
    args = parser.parse_args()

    if args.command == "report":
        records = _load_records(args.out)
        if not records:
            raise SystemExit(f"No grading.json in {args.out}.")
        _save(args.out, records)
        print(f"Wrote {args.out / 'RESULTS.md'}")
        return

    cases = load_cases(args.cases)
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.id in wanted]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or ROOT / "results" / stamp
    try:
        asyncio.run(live(cases, out_dir, args.runs, args.pause))
    except ValueError as error:  # missing keys or settings
        raise SystemExit(str(error)) from error
    print(f"Results: {out_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
