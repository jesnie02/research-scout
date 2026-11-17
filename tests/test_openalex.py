import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import requests

from scout.models import Interval, Intervals
from scout.openalex import (
    MAX_NEW_CANDIDATES,
    CandidateStore,
    SearchError,
    fetch_works,
    parse_work,
    reconstruct_abstract,
    search_papers,
    shorten,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "openalex_rag.json").read_text())
NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
RAG = Intervals(year=Interval(max=2020), citations=Interval(min=501), notes=[])
OPEN = Intervals(year=Interval(), citations=Interval(), notes=[])


def work(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1/x",
        "title": "A paper",
        "publication_year": 2020,
        "cited_by_count": 10,
        "type": "article",
        "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
    }
    data.update(changes)
    return data


class FakeResponse:
    def __init__(self, status: int, body: Any = None, headers: dict[str, str] | None = None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body) if body is not None else "not json"

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


class FakeGet:
    """Stands in for requests.get: returns queued responses or raises queued errors."""

    def __init__(self, *responses: FakeResponse | Exception):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def no_sleep(seconds: float) -> None:
    pass


# --- abstract and parsing ---------------------------------------------------------------


def test_reconstruct_abstract_orders_words_by_position() -> None:
    assert reconstruct_abstract({"world": [1], "Hello": [0], "again": [3], "hello": [2]}) == (
        "Hello world hello again"
    )


def test_reconstruct_missing_abstract() -> None:
    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_shorten_cuts_at_a_word_and_marks_it() -> None:
    text, cut = shorten("one two three four", limit=10)
    assert (text, cut) == ("one two…", True)
    assert shorten("short", limit=10) == ("short", False)


def test_parse_work_takes_metadata_only_from_openalex() -> None:
    paper = parse_work(FIXTURE["results"][0], retrieved_at=NOW)
    assert paper is not None
    assert paper.openalex_id == "W3027879771"
    assert paper.title.startswith("Retrieval-Augmented Generation")
    assert paper.year == 2020
    assert paper.citations == 10269
    assert paper.authors[0] == "Patrick Lewis"
    assert paper.url == "https://doi.org/10.48550/arxiv.2005.11401"
    assert paper.openalex_url == "https://openalex.org/W3027879771"
    assert paper.abstract is not None and len(paper.abstract) <= 1001
    assert paper.retrieved_at == NOW


def test_parse_work_without_doi_links_to_openalex() -> None:
    paper = parse_work(work(doi=None), retrieved_at=NOW)
    assert paper is not None
    assert paper.url == "https://openalex.org/W1"


def test_missing_citation_count_stays_unknown_not_zero() -> None:
    paper = parse_work(work(cited_by_count=None), retrieved_at=NOW)
    assert paper is not None
    assert paper.citations is None


@pytest.mark.parametrize("missing", ["id", "title", "publication_year"])
def test_work_without_checkable_metadata_is_skipped(missing: str) -> None:
    assert parse_work(work(**{missing: None}), retrieved_at=NOW) is None


def test_missing_abstract_is_marked() -> None:
    paper = parse_work(work(abstract_inverted_index=None), retrieved_at=NOW)
    assert paper is not None
    assert paper.abstract is None


# --- fetching ---------------------------------------------------------------------------


def test_fetch_sends_filters_and_key_in_header() -> None:
    get = FakeGet(FakeResponse(200, FIXTURE))
    body = fetch_works("retrieval-augmented generation", RAG, api_key="k", get=get)
    assert body is FIXTURE
    call = get.calls[0]
    assert call["params"]["search"] == "retrieval-augmented generation"
    assert call["params"]["filter"] == "publication_year:<2021,cited_by_count:>500"
    assert call["headers"] == {"Authorization": "Bearer k"}
    assert "api_key" not in call["params"]
    assert call["timeout"] == 20


def test_fetch_without_filters_sends_no_filter_param() -> None:
    get = FakeGet(FakeResponse(200, FIXTURE))
    fetch_works("anything", OPEN, api_key="k", get=get)
    assert "filter" not in get.calls[0]["params"]


def test_fetch_retries_rate_limits_and_server_errors() -> None:
    waits: list[float] = []
    get = FakeGet(
        FakeResponse(429, {"error": "slow down"}, {"Retry-After": "3"}),
        FakeResponse(503, {"error": "busy"}),
        FakeResponse(200, FIXTURE),
    )
    assert fetch_works("q", OPEN, api_key="k", get=get, sleep=waits.append) is FIXTURE
    assert waits == [3.0, 2.0]


def test_fetch_retries_network_errors() -> None:
    get = FakeGet(requests.Timeout("slow"), FakeResponse(200, FIXTURE))
    assert fetch_works("q", OPEN, api_key="k", get=get, sleep=no_sleep) is FIXTURE


def test_fetch_gives_up_after_retries_with_an_error_not_an_empty_list() -> None:
    get = FakeGet(*(FakeResponse(503, {"error": "busy"}) for _ in range(3)))
    with pytest.raises(SearchError, match="503"):
        fetch_works("q", OPEN, api_key="k", get=get, sleep=no_sleep)
    assert len(get.calls) == 3


def test_fetch_does_not_retry_a_bad_request() -> None:
    get = FakeGet(FakeResponse(400, {"error": "bad filter"}))
    with pytest.raises(SearchError, match="400"):
        fetch_works("q", OPEN, api_key="k", get=get, sleep=no_sleep)
    assert len(get.calls) == 1


def test_fetch_rejects_a_body_that_is_not_a_result_list() -> None:
    with pytest.raises(SearchError):
        fetch_works("q", OPEN, api_key="k", get=FakeGet(FakeResponse(200, None)), sleep=no_sleep)
    with pytest.raises(SearchError):
        fetch_works("q", OPEN, api_key="k", get=FakeGet(FakeResponse(200, {"x": 1})))


def test_network_errors_become_search_errors() -> None:
    get = FakeGet(*(requests.ConnectionError("down") for _ in range(3)))
    with pytest.raises(SearchError, match="ConnectionError"):
        fetch_works("q", OPEN, api_key="k", get=get, sleep=no_sleep)


# --- candidate store and full search ----------------------------------------------------


def test_search_filters_noise_and_duplicates_from_a_real_response() -> None:
    store = CandidateStore()
    result = search_papers(
        "retrieval-augmented generation",
        RAG,
        store,
        api_key="k",
        get=FakeGet(FakeResponse(200, FIXTURE)),
        now=NOW,
    )
    ids = [p.openalex_id for p in result.new_candidates]
    # The conference version of the RAG paper has the same title as the preprint: kept once.
    assert "W3098425262" not in ids
    # A book review is not a research paper.
    assert "W1532325895" not in ids
    assert ids[0] == "W3027879771"
    assert all(RAG.year.contains(p.year) and RAG.citations.contains(p.citations) for p in store)
    assert result.total_count == FIXTURE["meta"]["count"]
    assert result.raw is FIXTURE


def test_local_filter_rejects_results_the_api_should_not_have_returned() -> None:
    body = {"meta": {"count": 2}, "results": [work(publication_year=2024), work(id="W2")]}
    result = search_papers(
        "q", RAG, CandidateStore(), api_key="k", get=FakeGet(FakeResponse(200, body)), now=NOW
    )
    assert result.new_candidates == []
    assert result.rejected == 2


def test_unknown_citation_count_cannot_meet_a_citation_requirement() -> None:
    body = {"meta": {"count": 1}, "results": [work(cited_by_count=None, publication_year=2019)]}
    result = search_papers(
        "q", RAG, CandidateStore(), api_key="k", get=FakeGet(FakeResponse(200, body)), now=NOW
    )
    assert result.new_candidates == []


def test_store_never_overwrites_an_earlier_candidate() -> None:
    store = CandidateStore()
    first = parse_work(work(title="Original"), retrieved_at=NOW)
    clash = parse_work(work(title="Different paper, same id"), retrieved_at=NOW)
    assert first is not None and clash is not None
    assert store.add([first]) == [first]
    assert store.add([clash]) == []
    assert store.get("W1") == first


def test_store_caps_new_candidates_per_search_and_keeps_order() -> None:
    store = CandidateStore()
    papers = [
        p
        for i in range(12)
        if (p := parse_work(work(id=f"W{i}", title=f"Paper {i}"), retrieved_at=NOW))
    ]
    added = store.add(papers)
    assert [p.openalex_id for p in added] == [f"W{i}" for i in range(MAX_NEW_CANDIDATES)]
    assert len(store) == MAX_NEW_CANDIDATES
    assert store.get("W99") is None


def test_second_search_adds_only_new_papers() -> None:
    store = CandidateStore()
    body = {"meta": {"count": 2}, "results": [work(id="W1"), work(id="W2", title="Another")]}
    search_papers("q", OPEN, store, api_key="k", get=FakeGet(FakeResponse(200, body)), now=NOW)
    again = search_papers(
        "q2", OPEN, store, api_key="k", get=FakeGet(FakeResponse(200, body)), now=NOW
    )
    assert again.new_candidates == []
    assert again.already_known == 2
    assert len(store) == 2
