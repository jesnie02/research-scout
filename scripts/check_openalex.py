"""Step 0: check OpenAlex search on the assignment's three example requests.

Run: uv run python -m scripts.check_openalex   (needs OPENALEX_API_KEY in .env)
Raw responses are saved in scripts/results/ for the README.
"""

import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests

from scout.config import openalex_api_key
from scout.constraints import openalex_filter, to_intervals
from scout.models import CitationConstraint, Intervals, YearConstraint

API = "https://api.openalex.org/works"
SELECT = "id,doi,title,publication_year,cited_by_count,type,authorships,abstract_inverted_index"
RESULTS = Path(__file__).resolve().parent / "results"

# (name, search words, year constraints, citation constraints, recent)
EXAMPLES: list[tuple[str, str, list[YearConstraint], list[CitationConstraint], bool]] = [
    (
        "llm-agents-se",
        "LLM agents for software engineering",
        [YearConstraint(op="after", value=2022)],
        [CitationConstraint(op="at_least", value=100)],
        False,
    ),
    (
        "rag",
        "retrieval-augmented generation",
        [YearConstraint(op="before", value=2021)],
        [CitationConstraint(op="over", value=500)],
        False,
    ),
    ("tool-agents", "AI agents using tools", [], [], True),
]


def check(name: str, query: str, intervals: Intervals, key: str) -> dict[str, Any]:
    params = {"search": query, "filter": openalex_filter(intervals), "per_page": "8"}
    started = time.monotonic()
    response = requests.get(
        API,
        params={**params, "select": SELECT},
        headers={"Authorization": f"Bearer {key}"},
        timeout=20,
    )
    seconds = round(time.monotonic() - started, 2)
    print(f"\n=== {name}: search={query!r} filter={params['filter']!r}")
    print(f"HTTP {response.status_code} in {seconds}s")
    record: dict[str, Any] = {"name": name, "params": params, "status": response.status_code}
    try:
        body = response.json()
    except ValueError:
        print(response.text[:300])
        record["body"] = response.text[:2000]
        return record
    record["body"] = body
    if response.status_code != 200:
        print(json.dumps(body)[:300])
        return record
    meta = body.get("meta", {})
    print(f"total matches: {meta.get('count')}  cost_usd: {meta.get('cost_usd')}")
    violations = 0
    for work in body.get("results", []):
        year, cites = work.get("publication_year"), work.get("cited_by_count")
        ok = intervals.year.contains(year) and intervals.citations.contains(cites)
        violations += not ok
        flag = "  " if ok else "!!"
        abstract = "abstract" if work.get("abstract_inverted_index") else "NO abstract"
        title = (work.get("title") or "")[:80]
        print(f"{flag} {year} | {cites:>6} cites | {work.get('type')} | {abstract} | {title}")
    record["violations"] = violations
    print("filters respected" if violations == 0 else f"{violations} result(s) break the filters")
    return record


def main() -> None:
    try:
        key = openalex_api_key()
    except ValueError as error:
        raise SystemExit(str(error)) from error
    records = []
    for name, query, years, citations, recent in EXAMPLES:
        intervals = to_intervals(years, citations, recent=recent, today=date.today())
        records.append(check(name, query, intervals, key))
        time.sleep(1)
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"openalex_{stamp}.json"
    path.write_text(json.dumps(records, indent=1))
    print(f"\nSaved raw responses to {path}")


if __name__ == "__main__":
    main()
