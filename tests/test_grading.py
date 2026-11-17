import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eval.cases import Case
from eval.grading import (
    ManualGrade,
    RunRecord,
    grade_run,
    render_report,
    summarize,
    verdict,
)
from eval.run_eval import run_cases
from scout.models import (
    AnswerRequirements,
    ExtractedRequest,
    Interval,
    Intervals,
    Paper,
    PaperAnswer,
    ScoutResult,
)

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def make_case(**expected: Any) -> Case:
    data: dict[str, Any] = {
        "topic": "retrieval-augmented generation",
        "year": {"min": None, "max": 2020},
        "citations": {"min": 501, "max": None},
        "answer_requirements": {
            "task": "summarize_contribution",
            "sentences_min": 2,
            "sentences_max": 3,
        },
        "outcomes": ["match"],
        "no_match_evidence": None,
    }
    data.update(expected)
    return Case.model_validate(
        {"id": "C01", "category": "year_before", "prompt": "Find a RAG paper ...", "expected": data}
    )


def candidate(pid: str = "W1", year: int = 2020, citations: int = 900) -> Paper:
    return Paper(
        openalex_id=pid,
        title=f"Paper {pid}",
        authors=["Ada Lovelace"],
        year=year,
        citations=citations,
        doi=None,
        url=f"https://openalex.org/{pid}",
        openalex_url=f"https://openalex.org/{pid}",
        work_type="article",
        abstract="An abstract.",
        abstract_truncated=False,
        retrieved_at=NOW,
    )


def make_result(
    outcome: str = "match",
    *,
    paper: Paper | None = None,
    answer: str = "One. Two.",
    intervals: Intervals | None | str = "default",
    requirements: AnswerRequirements | None = None,
    **changes: Any,
) -> ScoutResult:
    found = paper if paper is not None else candidate()
    request = ExtractedRequest(
        topic="retrieval-augmented generation",
        answer_requirements=requirements
        or AnswerRequirements(task="summarize_contribution", sentences_min=2, sentences_max=3),
    )
    answer_obj = None
    if outcome == "match":
        answer_obj = PaperAnswer(
            title=found.title,
            authors=found.authors,
            year=found.year,
            citations=found.citations,
            citation_source="OpenAlex ...",
            url=found.url,
            openalex_url=found.openalex_url,
            answer=answer,
            uncertainty="",
        )
    data: dict[str, Any] = {
        "outcome": outcome,
        "message": "m",
        "request": request,
        "intervals": (
            Intervals(year=Interval(max=2020), citations=Interval(min=501), notes=[])
            if intervals == "default"
            else intervals
        ),
        "paper": answer_obj,
        "candidates": [found],
        "model_requests": 4,
        "tool_calls": 1,
    }
    data.update(changes)
    return ScoutResult(**data)


def lookup_ok(pid: str) -> dict[str, Any]:
    return {"id": f"https://openalex.org/{pid}", "title": f"Paper {pid}", "publication_year": 2020}


def grade(case: Case, result: ScoutResult, lookup: Any = lookup_ok) -> RunRecord:
    return grade_run(case, result, run=1, lookup=lookup, result_file="C01_r1.json")


ALL_MANUAL_PASS = ManualGrade(topic="pass", relevance="pass", explanation="pass", grounded="pass")


# --- automatic checks ------------------------------------------------------------------


def test_correct_match_passes_every_automatic_check_and_waits_for_manual_grading() -> None:
    record = grade(make_case(), make_result())
    auto = record.auto
    assert auto.model_dump() == {
        "outcome": "pass",
        "intervals": "pass",
        "answer_requirements": "pass",
        "year": "pass",
        "citations": "pass",
        "source": "pass",
        "metadata": "pass",
        "sentences": "pass",
    }
    assert verdict(record) == "pending"
    assert verdict(record.model_copy(update={"manual": ALL_MANUAL_PASS})) == "pass"


def test_year_is_checked_against_the_expected_interval_not_the_agents_own() -> None:
    wrong_reading = Intervals(year=Interval(max=2022), citations=Interval(min=501), notes=[])
    record = grade(make_case(), make_result(paper=candidate(year=2022), intervals=wrong_reading))
    assert record.auto.intervals == "fail"
    assert record.auto.year == "fail"
    assert verdict(record.model_copy(update={"manual": ALL_MANUAL_PASS})) == "fail"


def test_citations_below_the_expected_minimum_fail() -> None:
    assert grade(make_case(), make_result(paper=candidate(citations=400))).auto.citations == "fail"


def test_misread_answer_requirements_fail_part_a() -> None:
    other = AnswerRequirements(task="explain_relevance")
    assert grade(make_case(), make_result(requirements=other)).auto.answer_requirements == "fail"


def test_source_must_resolve_to_the_same_title_and_year() -> None:
    def moved(pid: str) -> dict[str, Any]:
        return {"title": "Something else", "publication_year": 2020}

    def broken(pid: str) -> dict[str, Any]:
        raise RuntimeError("HTTP 404")

    assert grade(make_case(), make_result(), lookup=moved).auto.source == "fail"
    record = grade(make_case(), make_result(), lookup=broken)
    assert record.auto.source == "fail"
    assert "404" in record.notes


def test_metadata_must_equal_the_stored_candidate() -> None:
    result = make_result()
    assert result.paper is not None
    tampered = result.model_copy(
        update={"paper": result.paper.model_copy(update={"citations": 5000})}
    )
    assert grade(make_case(), tampered).auto.metadata == "fail"


def test_sentence_count_is_checked_against_the_expected_bounds() -> None:
    assert grade(make_case(), make_result(answer="Only one.")).auto.sentences == "fail"
    no_bounds = make_case(answer_requirements=None)
    record = grade(no_bounds, make_result(requirements=None))
    assert record.auto.sentences == "n/a"


def test_unexpected_outcome_fails() -> None:
    record = grade(make_case(), make_result("no_match", candidates=[]))
    assert record.auto.outcome == "fail"
    assert verdict(record) == "fail"


# --- rejections, clarifications and errors ---------------------------------------------


def test_expected_rejection_only_needs_the_relevant_manual_checks() -> None:
    case = make_case(
        outcomes=["no_match", "rejected_on_relevance"],
        no_match_evidence="OpenAlex search=... filter=... gave 0 results on 2026-11-17",
    )
    record = grade(case, make_result("rejected_on_relevance"))
    assert record.auto.outcome == "pass"
    assert record.auto.year == "n/a" and record.auto.source == "n/a"
    assert record.manual.explanation == "n/a" and record.manual.grounded == "n/a"
    assert record.manual.relevance is None  # was the rejection justified? graded by hand
    graded = record.model_copy(
        update={"manual": record.manual.model_copy(update={"topic": "pass", "relevance": "pass"})}
    )
    assert verdict(graded) == "pass"


def test_conflict_and_clarification_are_graded_on_outcome_alone() -> None:
    conflict = make_case(
        year={"min": 2021, "max": 2009},
        outcomes=["conflicting_constraints"],
        answer_requirements=None,
    )
    record = grade(conflict, make_result("conflicting_constraints", intervals=None))
    record = record.model_copy(
        update={"manual": record.manual.model_copy(update={"topic": "pass"})}
    )
    assert record.auto.intervals == "n/a"
    assert verdict(record) == "pass"

    vague = make_case(topic=None, outcomes=["needs_clarification"], answer_requirements=None)
    asked = make_result("needs_clarification", intervals=None, request=None)
    record = grade(vague, asked)
    assert record.manual.topic == "n/a"
    assert verdict(record) == "pass"


@pytest.mark.parametrize("outcome", ["search_error", "agent_error"])
def test_errors_are_their_own_verdict(outcome: str) -> None:
    record = grade(make_case(), make_result(outcome, intervals=None))
    assert verdict(record) == "error"


def test_a_manual_fail_fails_the_run() -> None:
    record = grade(make_case(), make_result())
    graded = record.model_copy(
        update={"manual": ALL_MANUAL_PASS.model_copy(update={"grounded": "fail"})}
    )
    assert verdict(graded) == "fail"


# --- report ------------------------------------------------------------------------------


def records_for_report() -> list[RunRecord]:
    case = make_case()
    passing = grade(case, make_result()).model_copy(update={"manual": ALL_MANUAL_PASS})
    second = grade_run(case, make_result(), run=2, lookup=lookup_ok, result_file="C01_r2.json")
    error = grade_run(
        case, make_result("agent_error", intervals=None), run=3, lookup=lookup_ok, result_file="x"
    )
    return [passing, second, error]


def test_summary_counts_errors_separately_and_in_the_failure_rate() -> None:
    summary = summarize(records_for_report())
    assert summary["runs"] == 3
    assert summary["verdicts"] == {"pass": 1, "pending": 1, "error": 1}
    assert summary["pass_rate"] == pytest.approx(1 / 3)
    assert summary["model_requests_total"] == 12
    assert summary["same_paper_every_run"] == {"C01": False}
    # pass^k as in tau-bench: a case counts only if every one of its runs passed.
    assert summary["all_runs_pass"] == {"C01": False}
    assert summarize(records_for_report()[:1])["all_runs_pass"] == {"C01": True}


def test_report_has_one_row_per_run_and_says_when_grading_is_unfinished() -> None:
    text = render_report(records_for_report())
    assert text.count("| C01 |") == 3
    assert "1 run(s) still need manual grading" in text
    assert "pass^3" in text


# --- runner ------------------------------------------------------------------------------


def test_runner_saves_every_result_and_resumes_without_rerunning(tmp_path: Path) -> None:
    calls: list[str] = []

    async def scout(prompt: str) -> ScoutResult:
        calls.append(prompt)
        return make_result()

    async def no_sleep(seconds: float) -> None:
        pass

    def go() -> list[RunRecord]:
        return asyncio.run(
            run_cases([make_case()], scout, tmp_path, runs=2, lookup=lookup_ok, sleep=no_sleep)
        )

    first = go()
    assert len(calls) == 2
    assert sorted(p.name for p in tmp_path.glob("C01_r*.json")) == ["C01_r1.json", "C01_r2.json"]
    assert json.loads((tmp_path / "C01_r1.json").read_text())["outcome"] == "match"
    assert (tmp_path / "RESULTS.md").exists()

    # Grades written by hand survive a re-run, and finished runs are not repeated.
    grading = tmp_path / "grading.json"
    data = json.loads(grading.read_text())
    data[0]["manual"] = ALL_MANUAL_PASS.model_dump()
    grading.write_text(json.dumps(data))
    second = go()
    assert len(calls) == 2
    assert second[0].manual == ALL_MANUAL_PASS
    assert [r.result_file for r in second] == [r.result_file for r in first]
