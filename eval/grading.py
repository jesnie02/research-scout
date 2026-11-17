"""Grade one evaluation run against expectations written before the agent ran.

Part A: did the agent understand the request? (intervals, answer requirements, topic)
Part B: did the answer fulfil it? (outcome, year, citations, source, metadata, length,
        relevance, explanation, grounding)
Automatic checks compare with the EXPECTED values in cases.json, never with the agent's own
reading. Manual checks start as None (to be graded) or "n/a" (does not apply to this outcome).
"""

from collections import Counter
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel

from eval.cases import Case
from scout.constraints import count_sentences
from scout.models import ScoutResult

Check = Literal["pass", "fail", "n/a"]
Verdict = Literal["pass", "fail", "pending", "error"]
Lookup = Callable[[str], dict[str, Any]]

ERROR_OUTCOMES = {"search_error", "agent_error"}
SYMBOL = {"pass": "✓", "fail": "✗", "n/a": "–", None: "?"}


class AutoGrade(BaseModel):
    outcome: Check
    intervals: Check
    answer_requirements: Check
    year: Check
    citations: Check
    source: Check
    metadata: Check
    sentences: Check


class ManualGrade(BaseModel):
    topic: Check | None = None
    relevance: Check | None = None
    explanation: Check | None = None
    grounded: Check | None = None
    note: str = ""


class RunRecord(BaseModel):
    case_id: str
    category: str
    run: int
    outcome: str
    expected_outcomes: list[str]
    paper_id: str | None
    paper_title: str | None
    model_requests: int
    tool_calls: int
    searches: int
    result_file: str
    auto: AutoGrade
    manual: ManualGrade
    notes: str = ""


def _ok(condition: bool) -> Check:
    return "pass" if condition else "fail"


def _source_check(pid: str, title: str, year: int, url: str, lookup: Lookup) -> tuple[Check, str]:
    """Look the paper up again in OpenAlex: the ID must resolve to the same title and year."""
    try:
        work = lookup(pid)
    except Exception as error:  # any lookup failure means the source was not confirmed
        return "fail", f"source lookup failed: {error}"
    same = work.get("title") == title and work.get("publication_year") == year and bool(url)
    return _ok(same), "" if same else f"OpenAlex now says {work.get('title')!r}"


def grade_run(
    case: Case, result: ScoutResult, *, run: int, lookup: Lookup, result_file: str
) -> RunRecord:
    expected = case.expected
    paper = result.paper
    pid = paper.openalex_url.rsplit("/", 1)[-1] if paper else None
    notes: list[str] = []

    # Part A: compared with the expected values, not with what the agent produced.
    if result.intervals is None:
        intervals: Check = "fail" if result.outcome in {"match", "no_match"} else "n/a"
    else:
        intervals = _ok(
            result.intervals.year == expected.year
            and result.intervals.citations == expected.citations
        )
    if result.request is None or result.intervals is None:
        requirements: Check = "n/a"
    else:
        requirements = _ok(result.request.answer_requirements == expected.answer_requirements)

    # Part B: only a returned paper can be checked against the data.
    year = citations = source = metadata = sentences = "n/a"
    if paper is not None and pid is not None:
        year = _ok(expected.year.contains(paper.year))
        citations = _ok(expected.citations.contains(paper.citations))
        source, note = _source_check(pid, paper.title, paper.year, paper.url, lookup)
        if note:
            notes.append(note)
        stored = next((c for c in result.candidates if c.openalex_id == pid), None)
        metadata = _ok(
            stored is not None
            and (stored.title, stored.authors, stored.year, stored.citations, stored.url)
            == (paper.title, paper.authors, paper.year, paper.citations, paper.url)
        )
        req = expected.answer_requirements
        if req and (req.sentences_min is not None or req.sentences_max is not None):
            count = count_sentences(paper.answer)
            low, high = req.sentences_min or 1, req.sentences_max
            sentences = _ok(count >= low and (high is None or count <= high))
            notes.append(f"{count} sentences")

    errored = result.outcome in ERROR_OUTCOMES
    has_topic = result.request is not None and bool(result.request.topic)
    manual = ManualGrade(
        topic=None if has_topic and not errored else "n/a",
        relevance=None
        if result.outcome in {"match", "no_match", "rejected_on_relevance"}
        else "n/a",
        explanation=None if result.outcome == "match" else "n/a",
        grounded=None if result.outcome == "match" else "n/a",
    )
    return RunRecord(
        case_id=case.id,
        category=case.category,
        run=run,
        outcome=result.outcome,
        expected_outcomes=list(expected.outcomes),
        paper_id=pid,
        paper_title=paper.title if paper else None,
        model_requests=result.model_requests,
        tool_calls=result.tool_calls,
        searches=len(result.searches),
        result_file=result_file,
        auto=AutoGrade(
            outcome=_ok(result.outcome in expected.outcomes),
            intervals=intervals,
            answer_requirements=requirements,
            year=year,
            citations=citations,
            source=source,
            metadata=metadata,
            sentences=sentences,
        ),
        manual=manual,
        notes="; ".join(notes),
    )


def _manual_values(record: RunRecord) -> list[Check | None]:
    m = record.manual
    return [m.topic, m.relevance, m.explanation, m.grounded]


def verdict(record: RunRecord) -> Verdict:
    """A run passes only when every check that applies to it passed (plan: bestået-regel)."""
    if record.outcome in ERROR_OUTCOMES:
        return "error"
    if "fail" in record.auto.model_dump().values() or "fail" in _manual_values(record):
        return "fail"
    if None in _manual_values(record):
        return "pending"
    return "pass"


def summarize(records: list[RunRecord]) -> dict[str, Any]:
    verdicts = Counter(verdict(r) for r in records)
    papers: dict[str, set[str | None]] = {}
    all_pass: dict[str, bool] = {}
    for r in records:
        papers.setdefault(r.case_id, set()).add(r.paper_id)
        all_pass[r.case_id] = all_pass.get(r.case_id, True) and verdict(r) == "pass"
    total = len(records)
    return {
        "runs": total,
        "verdicts": dict(verdicts),
        "pass_rate": verdicts["pass"] / total if total else 0.0,
        "outcomes": dict(Counter(r.outcome for r in records)),
        "model_requests_total": sum(r.model_requests for r in records),
        "model_requests_mean": sum(r.model_requests for r in records) / total if total else 0.0,
        "tool_calls_mean": sum(r.tool_calls for r in records) / total if total else 0.0,
        "same_paper_every_run": {case: len(ids) == 1 for case, ids in papers.items()},
        # pass^k (tau-bench): the case passes only if every one of its k runs passed.
        "all_runs_pass": all_pass,
    }


def render_report(records: list[RunRecord]) -> str:
    s = summarize(records)
    v = s["verdicts"]
    lines = [
        "# Evaluation results",
        "",
        f"- Runs: {s['runs']}. Pass: {v.get('pass', 0)}, fail: {v.get('fail', 0)}, "
        f"error (search/agent): {v.get('error', 0)}, pending: {v.get('pending', 0)}.",
        f"- Pass rate (errors count as not passed): {s['pass_rate']:.0%}.",
        f"- Outcomes: {s['outcomes']}.",
        f"- Model requests: {s['model_requests_total']} in total, "
        f"{s['model_requests_mean']:.1f} per run; tool calls {s['tool_calls_mean']:.1f} per run.",
    ]
    if v.get("pending", 0):
        lines.append(
            f"- **{v['pending']} run(s) still need manual grading** (grading.json), "
            "then run `uv run python -m eval.run_eval report <dir>`."
        )
    k = max((r.run for r in records), default=0)
    passed_all = sum(s["all_runs_pass"].values())
    lines += [
        f"- pass^{k} (every run of a case passed): "
        f"{passed_all} of {len(s['all_runs_pass'])} cases.",
        "",
        "Per case: all runs passed / same paper in every run:",
        "",
    ]
    lines += [
        f"- {case}: {'yes' if s['all_runs_pass'][case] else 'no'} / {'yes' if same else 'no'}"
        for case, same in s["same_paper_every_run"].items()
    ]
    lines += [
        "",
        "✓ pass, ✗ fail, – not applicable, ? not graded yet. A = understood the request, "
        "B = fulfilled it. Year/citations are checked against the expected intervals.",
        "",
        "| Case | Run | Category | Outcome | Paper | A: intervals | A: requirements | A: topic "
        "| B: outcome | B: year | B: citations | B: source | B: metadata | B: sentences "
        "| B: relevance | B: explanation | B: grounded | Verdict | Notes |",
        "|" + "---|" * 19,
    ]
    for r in records:
        a, m = r.auto, r.manual
        cells = [
            r.case_id,
            str(r.run),
            r.category,
            r.outcome,
            (r.paper_title or "")[:50].replace("|", "/"),
            SYMBOL[a.intervals],
            SYMBOL[a.answer_requirements],
            SYMBOL[m.topic],
            SYMBOL[a.outcome],
            SYMBOL[a.year],
            SYMBOL[a.citations],
            SYMBOL[a.source],
            SYMBOL[a.metadata],
            SYMBOL[a.sentences],
            SYMBOL[m.relevance],
            SYMBOL[m.explanation],
            SYMBOL[m.grounded],
            verdict(r),
            "; ".join(x for x in (r.notes, m.note) if x).replace("|", "/"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
