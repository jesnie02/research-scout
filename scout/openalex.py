"""OpenAlex search tool. Deterministic code: fetch, parse, filter, deduplicate, store.

A failed search raises SearchError. It never turns into an empty result, so the agent
cannot report "no match" when the search did not actually run.
"""

import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import requests

from scout.constraints import openalex_filter
from scout.models import Intervals, Paper

API_URL = "https://api.openalex.org/works"
SELECT = "id,doi,title,publication_year,cited_by_count,type,authorships,abstract_inverted_index"
PER_PAGE = 25
MAX_NEW_CANDIDATES = 8
ABSTRACT_LIMIT = 1000
TIMEOUT_SECONDS = 20
RETRIES = 2
MAX_RETRY_AFTER = 10.0
RETRY_STATUSES = {429, 500, 502, 503, 504}
# Research output only; e.g. book reviews, datasets and software are dropped.
# book-chapter is kept because many conference proceedings are typed that way.
RESEARCH_TYPES = {"article", "preprint", "conference-paper", "review", "book-chapter"}

Getter = Callable[..., Any]


class SearchError(RuntimeError):
    """The OpenAlex search could not be completed."""


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not index:
        return None
    positions = sorted((pos, word) for word, places in index.items() for pos in places)
    return " ".join(word for _, word in positions)


def shorten(text: str, limit: int = ABSTRACT_LIMIT) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…", True


def _short_id(openalex_id: str) -> str:
    return openalex_id.rstrip("/").rsplit("/", 1)[-1]


def parse_work(work: dict[str, Any], *, retrieved_at: datetime) -> Paper | None:
    """Turn one OpenAlex work into a Paper; None if title, year or id is missing."""
    raw_id, title, year = work.get("id"), work.get("title"), work.get("publication_year")
    if not raw_id or not title or year is None:
        return None
    short_id = _short_id(raw_id)
    openalex_url = f"https://openalex.org/{short_id}"
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    truncated = False
    if abstract is not None:
        abstract, truncated = shorten(abstract)
    authors = [
        name
        for authorship in work.get("authorships") or []
        if (name := (authorship.get("author") or {}).get("display_name"))
    ]
    return Paper(
        openalex_id=short_id,
        title=title,
        authors=authors,
        year=year,
        citations=work.get("cited_by_count"),
        doi=work.get("doi"),
        url=work.get("doi") or openalex_url,
        openalex_url=openalex_url,
        work_type=work.get("type"),
        abstract=abstract,
        abstract_truncated=truncated,
        retrieved_at=retrieved_at,
    )


def _retry_wait(response: Any, attempt: int) -> float:
    header = response.headers.get("Retry-After") if response is not None else None
    if header and header.replace(".", "", 1).isdigit():
        return min(float(header), MAX_RETRY_AFTER)
    return float(2**attempt)


def fetch_works(
    query: str,
    intervals: Intervals,
    *,
    api_key: str,
    get: Getter = requests.get,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """One OpenAlex search with the constraints as API filters. Retries 429/5xx/network errors."""
    params = {"search": query, "per_page": str(PER_PAGE), "select": SELECT}
    filter_value = openalex_filter(intervals)
    if filter_value:
        params["filter"] = filter_value
    problem = ""
    for attempt in range(RETRIES + 1):
        response = None
        try:
            response = get(
                API_URL,
                params=params,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            problem = f"{type(error).__name__}: {error}"
        else:
            if response.status_code == 200:
                try:
                    body = response.json()
                except ValueError as error:
                    raise SearchError("OpenAlex returned a response that is not JSON.") from error
                if not isinstance(body, dict) or not isinstance(body.get("results"), list):
                    raise SearchError("OpenAlex returned an unexpected response format.")
                return body
            problem = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in RETRY_STATUSES:
                break
        if attempt < RETRIES:
            sleep(_retry_wait(response, attempt))
    raise SearchError(f"OpenAlex search failed ({problem}).")


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())


@dataclass
class CandidateStore:
    """All candidates for one user request, keyed by OpenAlex ID. Entries are never replaced."""

    _papers: dict[str, Paper] = field(default_factory=dict)
    _titles: set[str] = field(default_factory=set)

    def is_known(self, paper: Paper) -> bool:
        return paper.openalex_id in self._papers or _title_key(paper.title) in self._titles

    def add(self, papers: list[Paper]) -> list[Paper]:
        """Add papers not seen before (same ID or same title), at most MAX_NEW_CANDIDATES."""
        added: list[Paper] = []
        for paper in papers:
            if len(added) == MAX_NEW_CANDIDATES:
                break
            if self.is_known(paper):
                continue
            self._papers[paper.openalex_id] = paper
            self._titles.add(_title_key(paper.title))
            added.append(paper)
        return added

    def get(self, openalex_id: str) -> Paper | None:
        return self._papers.get(openalex_id)

    def __iter__(self) -> Iterator[Paper]:
        return iter(self._papers.values())

    def __len__(self) -> int:
        return len(self._papers)


@dataclass
class SearchResult:
    query: str
    filter: str
    total_count: int | None
    new_candidates: list[Paper]
    rejected: int
    already_known: int
    raw: dict[str, Any]


def search_papers(
    query: str,
    intervals: Intervals,
    store: CandidateStore,
    *,
    api_key: str,
    get: Getter = requests.get,
    sleep: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
) -> SearchResult:
    """Search, then re-check every result locally before it can become a candidate."""
    body = fetch_works(query, intervals, api_key=api_key, get=get, sleep=sleep)
    retrieved_at = now or datetime.now(UTC)
    passing: list[Paper] = []
    rejected = 0
    for work in body["results"]:
        paper = parse_work(work, retrieved_at=retrieved_at)
        if (
            paper is None
            or paper.work_type not in RESEARCH_TYPES
            or not intervals.year.contains(paper.year)
            or not intervals.citations.contains(paper.citations)
        ):
            rejected += 1
            continue
        passing.append(paper)
    already_known = sum(store.is_known(p) for p in passing)
    added = store.add(passing)
    return SearchResult(
        query=query,
        filter=openalex_filter(intervals),
        total_count=(body.get("meta") or {}).get("count"),
        new_candidates=added,
        rejected=rejected,
        already_known=already_known,
        raw=body,
    )
