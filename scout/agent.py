"""Research Scout workflow.

LLM (agent A):  read the request -> topic, constraints, answer requirements.
CODE:           ask for clarification / resolve intervals / reject contradictions.
LLM (agent B):  search_papers (ReAct: search, read results, search again or choose),
                then pick ONE stored candidate by ID (or null) and write the answer text.
CODE:           the search tool filters and stores candidates; the output validator checks
                the ID, that a search happened and the sentence count; the final answer is
                built from stored OpenAlex data only; the outcome is decided by code.
"""

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from scout.constraints import ConflictingConstraintsError, count_sentences, to_intervals
from scout.models import (
    AnswerRequirements,
    ExtractedRequest,
    Interval,
    Intervals,
    Outcome,
    PaperAnswer,
    ScoutResult,
    SearchLog,
    Selection,
)
from scout.openalex import CandidateStore, SearchError, SearchResult

MAX_SEARCHES = 3
TOTAL_TIMEOUT = 180.0
EXTRACT_LIMITS = UsageLimits(request_limit=2)
# One extra tool call is allowed so the model can be told the search limit was reached.
SELECT_LIMITS = UsageLimits(request_limit=8, tool_calls_limit=MAX_SEARCHES + 1)
AUTHORS_SHOWN = 3

SearchFn = Callable[[str, Intervals, CandidateStore], SearchResult]

EXTRACT_INSTRUCTIONS = """You read a user's request for a research paper and extract search
constraints. Extract only what the request says; never add constraints of your own.

- topic: a short English search phrase for the subject. null if no researchable subject is named.
- year: list of {"op", "value"}. before = strictly earlier than the year, after = strictly later,
  in = that exact year, from = that year or later ("since"), until = that year or earlier.
  "between A and B" = from A and until B.
- citations: list of {"op", "value"}. over = more than, at_least, under = fewer than, at_most,
  about = approximately.
- recent: true only if the request asks for recent/latest/new work without giving a year.
- answer_requirements: what the user wants said about the paper, or null if they only ask to
  find one. task: explain_relevance, summarize_contribution or assess_usefulness (whether it
  would help for a purpose the user describes). sentences_min / sentences_max only if a length
  is stated ("5-7 sentences" = 5 and 7, "in 3 sentences" = 3 and 3), otherwise null.
- clarifying_question: only if the request is too vague to search (no subject); else null.
- note: how you interpreted anything vague, or "".
"""

SELECT_INSTRUCTIONS = """You find ONE research paper for a user's request.

1. Call search_papers with a concise English query for the topic. The tool already enforces the
   year and citation limits; every candidate it returns satisfies them.
2. Read the candidates. If none is clearly about the topic, you may search again with other
   words (at most 3 searches in total).
3. Choose the single most relevant candidate by its openalex_id. Use only IDs the tool returned.
   If several candidates are equally relevant, prefer peer-reviewed work (article,
   conference-paper) and more citations over preprints with few citations, unless the user asks
   for the newest work. If no candidate is relevant to the topic, set openalex_id to null and
   explain why in answer.
4. Write answer from the candidate's title and abstract ONLY. Do not add facts from memory.
   If the abstract is missing, say so and keep to what the title supports. Title, authors, year
   and citation count are added from OpenAlex automatically; do not repeat them. Any number you
   write must appear in the candidate's data or in the user's request.
5. uncertainty: any doubt, e.g. a missing abstract or how you read a vague request; else "".
"""

TASK_TEXT = {
    "explain_relevance": "Explain why the paper matches the request.",
    "summarize_contribution": "Summarize the paper's contribution.",
    "assess_usefulness": (
        "Assess whether the paper would be useful for the purpose described in the request."
    ),
}


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    """Numbers in a text, with thousands separators removed (10,269 -> 10269)."""
    return {match.replace(",", "") for match in _NUMBER.findall(text)}


@dataclass
class RunState:
    """Per-request state passed to the tool and validator (deps), as DevSession in W45."""

    intervals: Intervals
    requirements: AnswerRequirements | None
    search: SearchFn
    prompt: str = ""
    store: CandidateStore = field(default_factory=CandidateStore)
    searches: list[SearchLog] = field(default_factory=list)
    raw_responses: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0


@dataclass
class Trace:
    """What has happened so far, so a timeout can still report it."""

    request: ExtractedRequest | None = None
    intervals: Intervals | None = None
    state: RunState | None = None
    model_requests: int = 0

    def result(self, outcome: Outcome, message: str, **extra: Any) -> ScoutResult:
        state = self.state
        return ScoutResult(
            outcome=outcome,
            message=message,
            request=self.request,
            intervals=self.intervals,
            searches=state.searches if state else [],
            candidates=list(state.store) if state else [],
            raw_responses=state.raw_responses if state else [],
            model_requests=self.model_requests,
            tool_calls=state.tool_calls if state else 0,
            **extra,
        )


def _count_responses(messages: list[ModelMessage]) -> int:
    return sum(isinstance(m, ModelResponse) for m in messages)


def _sentence_rule(req: AnswerRequirements | None) -> tuple[int, int | None] | None:
    if req is None or (req.sentences_min is None and req.sentences_max is None):
        return None
    return req.sentences_min or 1, req.sentences_max


def _describe_sentences(low: int, high: int | None) -> str:
    if high is None:
        return f"at least {low} sentences"
    return f"{low} sentences" if low == high else f"{low}-{high} sentences"


def _describe(interval: Interval, unit: str) -> str:
    if interval.min is None and interval.max is None:
        return f"no {unit} limit"
    if interval.min is not None and interval.max is not None:
        return f"{unit} {interval.min}-{interval.max}"
    if interval.min is not None:
        return f"{unit} {interval.min} or more"
    return f"{unit} {interval.max} or less"


def _selection_prompt(prompt: str, intervals: Intervals, req: AnswerRequirements | None) -> str:
    task = TASK_TEXT[req.task if req else "explain_relevance"]
    rule = _sentence_rule(req)
    length = f" Write {_describe_sentences(*rule)}." if rule else ""
    notes = " ".join(intervals.notes)
    return (
        f"User request: {prompt}\n\n"
        f"Limits already enforced by the search tool: {_describe(intervals.year, 'year')}; "
        f"{_describe(intervals.citations, 'citations')}. {notes}\n"
        f"What the answer must do: {task}{length}"
    )


extract_agent: Agent[None, ExtractedRequest] = Agent(
    output_type=PromptedOutput(ExtractedRequest),
    instructions=EXTRACT_INSTRUCTIONS,
    retries={"output": 1},
)

select_agent: Agent[RunState, Selection] = Agent(
    deps_type=RunState,
    output_type=PromptedOutput(Selection),
    instructions=SELECT_INSTRUCTIONS,
    retries={"output": 2},
)


@select_agent.tool(sequential=True)
def search_papers(ctx: RunContext[RunState], query: str) -> dict[str, Any]:
    """Search OpenAlex for research papers about `query` (a short English search phrase).

    Returns new candidates with openalex_id, title, year, citations, authors and abstract.
    """
    state = ctx.deps
    state.tool_calls += 1
    if len(state.searches) >= MAX_SEARCHES:
        return {"error": f"Search limit reached ({MAX_SEARCHES}). Choose a candidate or null."}
    try:
        result = state.search(query, state.intervals, state.store)
    except SearchError as error:
        state.searches.append(SearchLog(query=query, error=str(error)))
        return {"error": f"{error} You may try the search again."}
    state.raw_responses.append(result.raw)
    state.searches.append(
        SearchLog(
            query=query,
            filter=result.filter,
            total_count=result.total_count,
            new_ids=[p.openalex_id for p in result.new_candidates],
            rejected=result.rejected,
            already_known=result.already_known,
        )
    )
    candidates = [
        {
            "openalex_id": p.openalex_id,
            "title": p.title,
            "year": p.year,
            "citations": p.citations,
            "type": p.work_type,
            "authors": p.authors[:AUTHORS_SHOWN] + (["et al."] if len(p.authors) > 3 else []),
            "abstract": p.abstract or "(no abstract available)",
        }
        for p in result.new_candidates
    ]
    return {
        "new_candidates": candidates,
        "already_shown_earlier": result.already_known,
        "removed_by_filters": result.rejected,
    }


@select_agent.output_validator
def check_selection(ctx: RunContext[RunState], output: Selection) -> Selection:
    state = ctx.deps
    if not state.searches:
        raise ModelRetry("Call search_papers before you answer.")
    if output.openalex_id is None:
        return output
    short_id = output.openalex_id.strip().rstrip("/").rsplit("/", 1)[-1]
    if state.store.get(short_id) is None:
        allowed = ", ".join(p.openalex_id for p in state.store) or "none"
        raise ModelRetry(
            f"{output.openalex_id} is not one of the candidates. Allowed IDs: {allowed}. "
            "Choose one of them or use null."
        )
    paper = state.store.get(short_id)
    assert paper is not None
    sources = " ".join(
        [paper.title, paper.abstract or "", str(paper.year), str(paper.citations), state.prompt]
    )
    unsupported = sorted(_numbers(output.answer) - _numbers(sources))
    if unsupported:
        raise ModelRetry(
            f"The answer mentions {', '.join(unsupported)}, which is not in the candidate's "
            "title, abstract or metadata. Remove it or use only numbers from the data."
        )
    rule = _sentence_rule(state.requirements)
    if rule is not None:
        low, high = rule
        count = count_sentences(output.answer)
        if count < low or (high is not None and count > high):
            raise ModelRetry(
                f"The answer must be {_describe_sentences(low, high)}; it has {count}. Rewrite it."
            )
    return output.model_copy(update={"openalex_id": short_id})


def _finish(trace: Trace, selection: Selection) -> ScoutResult:
    state = trace.state
    assert state is not None and trace.intervals is not None
    failed = any(s.error for s in state.searches)
    if selection.openalex_id is None:
        if failed:
            return trace.result(
                "search_error",
                "At least one search failed, so the agent cannot say whether a matching paper "
                "exists. Try again later.",
            )
        if len(state.store) == 0:
            return trace.result(
                "no_match",
                "No paper matching the constraints was found in these searches. One may still "
                "exist that the searches did not reach.",
            )
        return trace.result(
            "rejected_on_relevance",
            f"No suitable paper was found in these searches. All {len(state.store)} candidates "
            f"were rejected as not relevant: {selection.answer}",
        )
    paper = state.store.get(selection.openalex_id)
    assert paper is not None  # guaranteed by check_selection
    if not (
        trace.intervals.year.contains(paper.year)
        and trace.intervals.citations.contains(paper.citations)
    ):
        return trace.result(
            "agent_error",
            f"The selected paper {paper.openalex_id} does not meet the constraints, so it is "
            "not returned.",
        )
    answer = PaperAnswer(
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        citations=paper.citations,
        citation_source=(
            f"OpenAlex cited_by_count for {paper.openalex_id}, retrieved "
            f"{paper.retrieved_at:%Y-%m-%d %H:%M} UTC: {paper.openalex_url}"
        ),
        url=paper.url,
        openalex_url=paper.openalex_url,
        answer=selection.answer,
        uncertainty=selection.uncertainty,
    )
    return trace.result("match", "Found a paper that meets the constraints.", paper=answer)


async def _run(
    trace: Trace,
    prompt: str,
    model: Model,
    search: SearchFn,
    today: date,
    settings: ModelSettings | None,
) -> ScoutResult:
    extract_settings: ModelSettings = {**(settings or {}), "temperature": 0.0}
    with capture_run_messages() as messages:
        try:
            extracted = await extract_agent.run(
                prompt, model=model, usage_limits=EXTRACT_LIMITS, model_settings=extract_settings
            )
        except Exception as error:  # any model failure ends the run as an agent error
            trace.model_requests += _count_responses(messages)
            return trace.result(
                "agent_error", f"Could not read the request ({type(error).__name__})."
            )
    trace.model_requests += _count_responses(messages)
    request = trace.request = extracted.output

    if request.clarifying_question or not (request.topic and request.topic.strip()):
        question = request.clarifying_question or "Which research topic should I search for?"
        return trace.result("needs_clarification", question)
    try:
        intervals = to_intervals(
            request.year, request.citations, recent=request.recent, today=today
        )
    except ConflictingConstraintsError as error:
        return trace.result("conflicting_constraints", str(error))
    trace.intervals = intervals
    state = trace.state = RunState(intervals, request.answer_requirements, search, prompt)

    with capture_run_messages() as messages:
        try:
            selected = await select_agent.run(
                _selection_prompt(prompt, intervals, request.answer_requirements),
                model=model,
                deps=state,
                usage_limits=SELECT_LIMITS,
                model_settings=settings,
            )
        except Exception as error:  # budget, invalid output, HTTP failure after retries
            trace.model_requests += _count_responses(messages)
            if any(s.error for s in state.searches) and len(state.store) == 0:
                return trace.result("search_error", "The search could not be completed.")
            return trace.result(
                "agent_error",
                f"The agent could not give a valid answer ({type(error).__name__}).",
            )
    trace.model_requests += _count_responses(messages)
    return _finish(trace, selected.output)


async def run_scout(
    prompt: str,
    model: Model,
    *,
    search: SearchFn,
    today: date,
    settings: ModelSettings | None = None,
    timeout: float = TOTAL_TIMEOUT,
) -> ScoutResult:
    """Answer one request. Never raises for model or search problems; see ScoutResult.outcome."""
    trace = Trace()
    try:
        return await asyncio.wait_for(_run(trace, prompt, model, search, today, settings), timeout)
    except TimeoutError:
        return trace.result(
            "agent_error", f"The agent did not finish within the time limit ({timeout:.0f} s)."
        )
