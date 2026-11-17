import asyncio
import json
from datetime import UTC, date, datetime
from typing import Any

import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from scout.agent import MAX_SEARCHES, run_scout
from scout.models import Intervals, Paper, ScoutResult
from scout.openalex import CandidateStore, SearchError, SearchResult

models.ALLOW_MODEL_REQUESTS = False

TODAY = date(2026, 11, 17)
NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def paper(pid: str, year: int = 2020, citations: int | None = 900, abstract: str = "") -> Paper:
    return Paper(
        openalex_id=pid,
        title=f"Paper {pid}",
        authors=["Ada Lovelace", "Alan Turing"],
        year=year,
        citations=citations,
        doi=None,
        url=f"https://openalex.org/{pid}",
        openalex_url=f"https://openalex.org/{pid}",
        work_type="article",
        abstract=abstract or None,
        abstract_truncated=False,
        retrieved_at=NOW,
    )


class FakeSearch:
    """Each call returns the next queued batch of papers (or raises SearchError)."""

    def __init__(self, *batches: list[Paper] | SearchError):
        self.batches = list(batches)
        self.calls: list[tuple[str, Intervals]] = []

    def __call__(self, query: str, intervals: Intervals, store: CandidateStore) -> SearchResult:
        self.calls.append((query, intervals))
        batch = self.batches.pop(0)
        if isinstance(batch, SearchError):
            raise batch
        added = store.add(batch)
        return SearchResult(
            query=query,
            filter="f",
            total_count=len(batch),
            new_candidates=added,
            rejected=0,
            already_known=len(batch) - len(added),
            raw={"results": [p.openalex_id for p in batch]},
        )


def extraction(**fields: Any) -> str:
    data: dict[str, Any] = {
        "topic": "retrieval-augmented generation",
        "year": [{"op": "before", "value": 2021}],
        "citations": [{"op": "over", "value": 500}],
        "recent": False,
        "answer_requirements": None,
        "clarifying_question": None,
        "note": "",
    }
    data.update(fields)
    return json.dumps(data)


def selection(openalex_id: str | None, answer: str = "It matches.", **extra: Any) -> str:
    return json.dumps({"openalex_id": openalex_id, "answer": answer, "uncertainty": "", **extra})


Step = str | tuple[str, dict[str, Any]]


def scripted(
    extract: str, select_steps: list[Step]
) -> tuple[FunctionModel, list[list[ModelMessage]]]:
    """Agent A gets `extract`; agent B replays `select_steps`: text, or (tool name, args)."""
    steps = list(select_steps)
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(t.name == "search_papers" for t in info.function_tools):
            return ModelResponse(parts=[TextPart(extract)])
        seen.append(list(messages))
        step = steps.pop(0)
        if isinstance(step, tuple):
            return ModelResponse(parts=[ToolCallPart(step[0], step[1])])
        return ModelResponse(parts=[TextPart(step)])

    return FunctionModel(respond), seen


def run(model: FunctionModel, search: FakeSearch, prompt: str = "Find a RAG paper") -> ScoutResult:
    return asyncio.run(run_scout(prompt, model, search=search, today=TODAY))


SEARCH: Step = ("search_papers", {"query": "retrieval augmented generation"})


def retry_texts(seen: list[list[ModelMessage]]) -> list[str]:
    """Validator feedback sent back to the model. For text output PydanticAI 2.61 sends it as
    a user prompt marked <retry_feedback>; tool retries use RetryPromptPart."""
    return [
        str(part.content)
        for messages in seen
        for m in messages
        if isinstance(m, ModelRequest)
        for part in m.parts
        if isinstance(part, RetryPromptPart)
        or (isinstance(part, UserPromptPart) and "retry_feedback" in str(part.content))
    ]


# --- the seven outcomes -----------------------------------------------------------------


def test_match_builds_the_answer_from_stored_metadata_only() -> None:
    model, _ = scripted(extraction(), [SEARCH, selection("W1", "Introduces RAG.")])
    result = run(model, FakeSearch([paper("W1", abstract="We introduce RAG.")]))
    assert result.outcome == "match"
    assert result.paper is not None
    assert result.paper.title == "Paper W1"
    assert result.paper.year == 2020
    assert result.paper.citations == 900
    assert result.paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert "OpenAlex" in result.paper.citation_source
    assert "https://openalex.org/W1" in result.paper.citation_source
    assert result.paper.answer == "Introduces RAG."
    assert result.tool_calls == 1
    assert result.model_requests == 3  # 1 extraction + 2 selection turns
    assert [p.openalex_id for p in result.candidates] == ["W1"]
    assert result.raw_responses == [{"results": ["W1"]}]


def test_full_openalex_url_as_id_is_accepted() -> None:
    model, _ = scripted(extraction(), [SEARCH, selection("https://openalex.org/W1")])
    result = run(model, FakeSearch([paper("W1")]))
    assert result.outcome == "match"


def test_no_candidates_after_successful_searches_is_no_match() -> None:
    model, _ = scripted(extraction(), [SEARCH, selection(None, "Nothing came back.")])
    result = run(model, FakeSearch([]))
    assert result.outcome == "no_match"
    assert "in these searches" in result.message
    assert "does not exist" not in result.message


def test_model_rejecting_all_candidates_is_rejected_on_relevance() -> None:
    model, _ = scripted(extraction(), [SEARCH, selection(None, "All are off-topic.")])
    result = run(model, FakeSearch([paper("W1")]))
    assert result.outcome == "rejected_on_relevance"
    assert "All are off-topic." in result.message


@pytest.mark.parametrize(
    "fields",
    [
        {"topic": None},
        {"topic": "  "},
        {"clarifying_question": "Which field do you mean?"},
    ],
)
def test_missing_or_vague_topic_asks_instead_of_searching(fields: dict[str, Any]) -> None:
    model, seen = scripted(extraction(**fields), [])
    search = FakeSearch()
    result = run(model, search)
    assert result.outcome == "needs_clarification"
    assert result.message.endswith("?")
    assert search.calls == [] and seen == []


def test_conflicting_constraints_stop_before_any_search() -> None:
    years = [{"op": "before", "value": 2010}, {"op": "after", "value": 2020}]
    model, seen = scripted(extraction(year=years), [])
    search = FakeSearch()
    result = run(model, search)
    assert result.outcome == "conflicting_constraints"
    assert "before 2010" in result.message
    assert search.calls == [] and seen == []


def test_a_failed_search_never_becomes_no_match() -> None:
    model, _ = scripted(
        extraction(), [SEARCH, ("search_papers", {"query": "rag"}), selection(None, "None.")]
    )
    result = run(model, FakeSearch(SearchError("OpenAlex search failed (HTTP 503)."), []))
    assert result.outcome == "search_error"
    assert result.searches[0].error is not None
    assert result.searches[1].error is None


def test_invalid_output_twice_more_is_an_agent_error() -> None:
    model, _ = scripted(extraction(), [SEARCH, "not json", "still not json", "nope"])
    result = run(model, FakeSearch([paper("W1")]))
    assert result.outcome == "agent_error"


def test_extraction_failure_is_an_agent_error() -> None:
    model, _ = scripted("not json", [])
    result = run(model, FakeSearch())
    assert result.outcome == "agent_error"
    assert result.request is None


# --- retries and guards -----------------------------------------------------------------


def test_unknown_id_is_sent_back_with_the_allowed_ids() -> None:
    model, seen = scripted(extraction(), [SEARCH, selection("W999"), selection("W1")])
    result = run(model, FakeSearch([paper("W1")]))
    assert result.outcome == "match"
    assert any("W999" in t and "W1" in t for t in retry_texts(seen))


def test_answer_without_searching_is_sent_back() -> None:
    model, seen = scripted(extraction(), [selection(None), SEARCH, selection("W1")])
    result = run(model, FakeSearch([paper("W1")]))
    assert result.outcome == "match"
    assert any("search_papers" in t for t in retry_texts(seen))


def test_wrong_sentence_count_is_sent_back() -> None:
    requirements = {"task": "summarize_contribution", "sentences_min": 2, "sentences_max": 3}
    model, seen = scripted(
        extraction(answer_requirements=requirements),
        [SEARCH, selection("W1", "One sentence."), selection("W1", "One. Two. Three.")],
    )
    result = run(model, FakeSearch([paper("W1")]))
    assert result.outcome == "match"
    assert any("2-3 sentences" in t for t in retry_texts(seen))


def test_rejection_reason_is_not_held_to_the_sentence_count() -> None:
    requirements = {"task": "summarize_contribution", "sentences_min": 5, "sentences_max": 7}
    model, _ = scripted(
        extraction(answer_requirements=requirements), [SEARCH, selection(None, "Off-topic.")]
    )
    assert run(model, FakeSearch([paper("W1")])).outcome == "rejected_on_relevance"


def test_search_limit_is_enforced_by_code() -> None:
    queries = [("search_papers", {"query": f"q{i}"}) for i in range(MAX_SEARCHES + 1)]
    model, seen = scripted(extraction(), [*queries, selection("W0")])
    search = FakeSearch(*([paper(f"W{i}")] for i in range(MAX_SEARCHES)))
    result = run(model, search)
    assert len(search.calls) == MAX_SEARCHES
    assert result.outcome == "match"
    last_tool_reply = [
        p.content
        for m in seen[-1]
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ][-1]
    assert "limit" in str(last_tool_reply).lower()


def test_tool_shows_the_model_only_what_openalex_returned() -> None:
    model, seen = scripted(extraction(), [SEARCH, selection("W1")])
    run(model, FakeSearch([paper("W1", abstract="We introduce RAG."), paper("W2")]))
    tool_reply = next(
        p.content
        for m in seen[1]
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    )
    text = json.dumps(tool_reply)
    assert "We introduce RAG." in text
    assert "no abstract available" in text  # W2 has none, and the model is told so


def test_constraints_reach_the_tool_without_the_model_setting_them() -> None:
    model, _ = scripted(extraction(), [SEARCH, selection("W1")])
    search = FakeSearch([paper("W1")])
    run(model, search)
    _, intervals = search.calls[0]
    assert intervals.year.max == 2020
    assert intervals.citations.min == 501


def test_final_check_rejects_a_paper_outside_the_constraints() -> None:
    # Defence in depth: even if a search let a wrong paper through, it is never returned.
    model, _ = scripted(extraction(), [SEARCH, selection("W1")])
    result = run(model, FakeSearch([paper("W1", year=2024)]))
    assert result.outcome == "agent_error"
    assert result.paper is None


def test_the_original_request_and_answer_requirements_reach_the_selector() -> None:
    requirements = {"task": "assess_usefulness", "sentences_min": None, "sentences_max": None}
    model, seen = scripted(extraction(answer_requirements=requirements), [SEARCH, selection("W1")])
    run(model, FakeSearch([paper("W1")]), prompt="Would a RAG paper help my chatbot team?")
    first_prompt = json.dumps([str(p) for p in seen[0][0].parts])
    assert "Would a RAG paper help my chatbot team?" in first_prompt
    assert "useful" in first_prompt.lower()


def test_whole_run_has_a_time_limit() -> None:
    def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise TimeoutError

    result = asyncio.run(
        run_scout("x", FunctionModel(slow), search=FakeSearch(), today=TODAY, timeout=0.01)
    )
    assert result.outcome == "agent_error"


def test_slow_model_hits_the_overall_timeout() -> None:
    async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(1)
        return ModelResponse(parts=[TextPart(extraction())])

    result = asyncio.run(
        run_scout("x", FunctionModel(slow), search=FakeSearch(), today=TODAY, timeout=0.05)
    )
    assert result.outcome == "agent_error"
    assert "time" in result.message.lower()


def test_failed_search_then_broken_output_is_reported_as_a_search_error() -> None:
    model, _ = scripted(extraction(), [SEARCH, "not json", "not json", "not json"])
    result = run(model, FakeSearch(SearchError("OpenAlex search failed (HTTP 503).")))
    assert result.outcome == "search_error"


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"year": [], "citations": []}, ["no year limit", "no citations limit"]),
        (
            {"year": [{"op": "from", "value": 2019}, {"op": "until", "value": 2021}]},
            ["year 2019-2021"],
        ),
        (
            {"answer_requirements": {"task": "summarize_contribution", "sentences_min": 3}},
            ["at least 3 sentences"],
        ),
        (
            {
                "answer_requirements": {
                    "task": "explain_relevance",
                    "sentences_min": 3,
                    "sentences_max": 3,
                }
            },
            ["Write 3 sentences."],
        ),
    ],
)
def test_selector_prompt_states_the_resolved_limits(
    fields: dict[str, Any], expected: list[str]
) -> None:
    model, seen = scripted(extraction(**fields), [SEARCH, selection(None, "None fit.")])
    run(model, FakeSearch([]))
    first_prompt = " ".join(str(p.content) for p in seen[0][0].parts if hasattr(p, "content"))
    for text in expected:
        assert text in first_prompt


# --- numbers in the answer must come from the data ---------------------------------------


def test_number_not_in_the_data_is_sent_back() -> None:
    model, seen = scripted(
        extraction(),
        [SEARCH, selection("W1", "It has 5000 citations."), selection("W1", "It has 900.")],
    )
    result = run(model, FakeSearch([paper("W1", citations=900)]))
    assert result.outcome == "match"
    assert result.paper is not None and result.paper.answer == "It has 900."
    assert any("5000" in t for t in retry_texts(seen))


def test_numbers_from_title_abstract_metadata_and_request_are_allowed() -> None:
    answer = (
        "Published in 2020 with 10,269 citations, it gains 3.5 points on paper W1. "
        "It fits the request for over 500 citations."
    )
    model, seen = scripted(extraction(), [SEARCH, selection("W1", answer)])
    found = paper("W1", citations=10269, abstract="Gains of 3.5 points.")
    result = run(model, FakeSearch([found]), prompt="Find a RAG paper with over 500 citations")
    assert result.outcome == "match"
    assert retry_texts(seen) == []


def test_numbers_in_a_rejection_are_not_checked() -> None:
    model, _ = scripted(extraction(), [SEARCH, selection(None, "None has 9999 citations.")])
    assert run(model, FakeSearch([paper("W1")])).outcome == "rejected_on_relevance"


# --- choosing between equally relevant candidates ------------------------------------------


def test_selector_is_told_to_prefer_reviewed_and_cited_work_on_ties() -> None:
    model, seen = scripted(extraction(), [SEARCH, selection("W1")])
    run(model, FakeSearch([paper("W1")]))
    first = seen[0][0]
    assert isinstance(first, ModelRequest)
    instructions = (first.instructions or "").lower()
    assert "peer-reviewed" in instructions
    assert "citations" in instructions
