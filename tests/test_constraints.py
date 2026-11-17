from datetime import date

import pytest

from scout.constraints import (
    ConflictingConstraintsError,
    count_sentences,
    openalex_filter,
    to_intervals,
)
from scout.models import CitationConstraint, Interval, Intervals, YearConstraint

TODAY = date(2026, 11, 17)


def years(*pairs: tuple[str, int]) -> list[YearConstraint]:
    return [YearConstraint.model_validate({"op": op, "value": v}) for op, v in pairs]


def cites(*pairs: tuple[str, int]) -> list[CitationConstraint]:
    return [CitationConstraint.model_validate({"op": op, "value": v}) for op, v in pairs]


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("before", Interval(min=None, max=2020)),
        ("after", Interval(min=2022, max=None)),
        ("in", Interval(min=2021, max=2021)),
        ("from", Interval(min=2021, max=None)),
        ("until", Interval(min=None, max=2021)),
    ],
)
def test_each_year_operator(op: str, expected: Interval) -> None:
    result = to_intervals(years((op, 2021)), [], recent=False, today=TODAY)
    assert result.year == expected
    assert result.citations == Interval()


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("over", Interval(min=501, max=None)),
        ("at_least", Interval(min=500, max=None)),
        ("under", Interval(min=None, max=499)),
        ("at_most", Interval(min=None, max=500)),
        ("about", Interval(min=400, max=600)),
    ],
)
def test_each_citation_operator(op: str, expected: Interval) -> None:
    result = to_intervals([], cites((op, 500)), recent=False, today=TODAY)
    assert result.citations == expected


def test_about_rounds_inwards() -> None:
    # 0.8 * 333 = 266.4 -> 267 and 1.2 * 333 = 399.6 -> 399
    result = to_intervals([], cites(("about", 333)), recent=False, today=TODAY)
    assert result.citations == Interval(min=267, max=399)
    assert any("267" in note and "399" in note for note in result.notes)


def test_constraints_on_same_field_are_merged_to_the_strictest() -> None:
    result = to_intervals(
        years(("after", 2018), ("before", 2022)),
        cites(("at_least", 100), ("over", 200)),
        recent=False,
        today=TODAY,
    )
    assert result.year == Interval(min=2019, max=2021)
    assert result.citations == Interval(min=201, max=None)


def test_recent_means_the_last_three_calendar_years() -> None:
    result = to_intervals([], [], recent=True, today=TODAY)
    assert result.year == Interval(min=2024, max=2026)
    assert any("2024" in note and "2026" in note for note in result.notes)


def test_recent_combines_with_an_explicit_year() -> None:
    result = to_intervals(years(("before", 2025)), [], recent=True, today=TODAY)
    assert result.year == Interval(min=2024, max=2024)


def test_no_constraints_give_unbounded_intervals() -> None:
    result = to_intervals([], [], recent=False, today=TODAY)
    assert result == Intervals(year=Interval(), citations=Interval(), notes=[])


def test_contradicting_years_are_rejected_with_an_explanation() -> None:
    with pytest.raises(ConflictingConstraintsError) as error:
        to_intervals(years(("before", 2010), ("after", 2020)), [], recent=False, today=TODAY)
    message = str(error.value)
    assert "before 2010" in message
    assert "after 2020" in message


def test_exact_year_and_after_same_year_conflict() -> None:
    with pytest.raises(ConflictingConstraintsError):
        to_intervals(years(("in", 2020), ("after", 2020)), [], recent=False, today=TODAY)


def test_contradicting_citations_are_rejected() -> None:
    with pytest.raises(ConflictingConstraintsError) as error:
        to_intervals([], cites(("over", 500), ("under", 100)), recent=False, today=TODAY)
    assert "over 500" in str(error.value)


def test_fewer_than_zero_citations_is_impossible() -> None:
    with pytest.raises(ConflictingConstraintsError):
        to_intervals([], cites(("under", 0)), recent=False, today=TODAY)


@pytest.mark.parametrize(
    ("year", "citations", "expected"),
    [
        (Interval(), Interval(), ""),
        (Interval(min=2024, max=2024), Interval(), "publication_year:2024"),
        (Interval(min=2019, max=2021), Interval(), "publication_year:2019-2021"),
        (Interval(min=2023), Interval(), "publication_year:>2022"),
        (Interval(max=2020), Interval(), "publication_year:<2021"),
        (Interval(), Interval(min=100), "cited_by_count:>99"),
        (Interval(), Interval(min=80, max=120), "cited_by_count:80-120"),
        (Interval(), Interval(max=499), "cited_by_count:<500"),
        (Interval(), Interval(min=0), ""),
        (
            Interval(min=2023),
            Interval(min=100),
            "publication_year:>2022,cited_by_count:>99",
        ),
    ],
)
def test_openalex_filter(year: Interval, citations: Interval, expected: str) -> None:
    assert openalex_filter(Intervals(year=year, citations=citations, notes=[])) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("   ", 0),
        ("One sentence.", 1),
        ("One. Two! Three?", 3),
        ("No final full stop", 1),
        ("Lewis et al. proposed RAG. It combines retrieval and generation.", 2),
        ("It uses tools, e.g. search. It also plans, i.e. it reasons.", 2),
        ("Accuracy rose to 3.5 points. That is a lot.", 2),
        ("See Fig. 2 for details. It shows the pipeline.", 2),
    ],
)
def test_count_sentences(text: str, expected: int) -> None:
    assert count_sentences(text) == expected
