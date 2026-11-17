import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from eval.cases import REQUIRED_CATEGORIES, Case, coverage_problems, load_cases
from eval.check_cases import main

EXAMPLE = Path(__file__).resolve().parent.parent / "eval" / "cases.example.json"


def case(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "C01",
        "category": "year_after",
        "prompt": "Find a paper about protein folding published after 2020.",
        "expected": {
            "topic": "protein folding",
            "year": {"min": 2021, "max": None},
            "citations": {"min": None, "max": None},
            "answer_requirements": None,
            "outcomes": ["match"],
            "no_match_evidence": None,
        },
        "notes": "",
    }
    expected_changes = changes.pop("expected", {})
    data.update(changes)
    data["expected"].update(expected_changes)
    return data


def full_set() -> list[dict[str, Any]]:
    """14 valid cases covering every required category, three with answer requirements."""
    categories = sorted(REQUIRED_CATEGORIES) + ["combined"] * (14 - len(REQUIRED_CATEGORIES))
    cases = []
    for number, category in enumerate(categories, start=1):
        expected: dict[str, Any] = {}
        if number <= 3:
            expected["answer_requirements"] = {
                "task": "summarize_contribution",
                "sentences_min": 5,
                "sentences_max": 7,
            }
        if category == "no_result":
            expected["outcomes"] = ["no_match", "rejected_on_relevance"]
            expected["no_match_evidence"] = "OpenAlex search=... filter=... returned 0 (2026-11-17)"
        cases.append(case(id=f"C{number:02d}", category=category, expected=expected))
    return cases


def write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    return path


def test_example_file_is_valid() -> None:
    cases = load_cases(EXAMPLE)
    assert len(cases) == 1
    # The example alone is not a complete evaluation set.
    assert coverage_problems(cases)


def test_complete_set_has_no_problems(tmp_path: Path) -> None:
    assert coverage_problems(load_cases(write(tmp_path, full_set()))) == []


def test_bad_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Case.model_validate(case(id="case-1"))


def test_unknown_outcome_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Case.model_validate(case(expected={"outcomes": ["search_error"]}))


def test_no_match_needs_evidence() -> None:
    with pytest.raises(ValidationError, match="no_match_evidence"):
        Case.model_validate(case(expected={"outcomes": ["no_match"]}))


def test_match_needs_a_topic() -> None:
    with pytest.raises(ValidationError, match="topic"):
        Case.model_validate(case(expected={"topic": None}))


def test_clarification_case_may_have_no_topic() -> None:
    parsed = Case.model_validate(
        case(category="ambiguous", expected={"topic": None, "outcomes": ["needs_clarification"]})
    )
    assert parsed.expected.topic is None


def test_reversed_interval_only_allowed_for_conflicting_constraints() -> None:
    reversed_years = {"year": {"min": 2021, "max": 2009}}
    with pytest.raises(ValidationError, match="min > max"):
        Case.model_validate(case(expected=reversed_years))
    Case.model_validate(
        case(
            category="conflicting",
            expected={**reversed_years, "outcomes": ["conflicting_constraints"]},
        )
    )


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="C01"):
        load_cases(write(tmp_path, [case(), case()]))


def test_coverage_problems_are_reported(tmp_path: Path) -> None:
    problems = coverage_problems(load_cases(write(tmp_path, [case()])))
    text = "\n".join(problems)
    assert "14" in text
    assert "no_result" in text
    assert "answer_requirements" in text


def test_cli_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(write(tmp_path, full_set()))]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_reports_problems(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(write(tmp_path, [case()]))]) == 1
    assert "14" in capsys.readouterr().out


def test_cli_reports_invalid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(write(tmp_path, [case(id="bad")]))]) == 1
    assert "id" in capsys.readouterr().out


def test_cli_reports_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "missing.json")]) == 1
    assert "not found" in capsys.readouterr().out
