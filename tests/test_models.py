import pytest
from pydantic import ValidationError

from scout.models import AnswerRequirements, CitationConstraint, Interval, YearConstraint


def test_unknown_operator_is_rejected() -> None:
    with pytest.raises(ValidationError):
        YearConstraint.model_validate({"op": "around", "value": 2020})


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        YearConstraint.model_validate({"op": "in", "value": 2020, "month": 5})


def test_negative_citation_count_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CitationConstraint.model_validate({"op": "over", "value": -1})


def test_sentence_bounds_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        AnswerRequirements(task="summarize_contribution", sentences_min=7, sentences_max=5)


def test_answer_requirements_accept_open_bounds() -> None:
    req = AnswerRequirements(task="assess_usefulness")
    assert req.sentences_min is None and req.sentences_max is None


@pytest.mark.parametrize(
    ("interval", "value", "expected"),
    [
        (Interval(min=2019, max=2021), 2019, True),
        (Interval(min=2019, max=2021), 2021, True),
        (Interval(min=2019, max=2021), 2022, False),
        (Interval(min=100), 99, False),
        (Interval(max=100), 100, True),
        # An unknown value (e.g. a missing citation count) never satisfies a bound.
        (Interval(min=100), None, False),
        (Interval(), None, True),
    ],
)
def test_interval_contains(interval: Interval, value: int | None, expected: bool) -> None:
    assert interval.contains(value) is expected
