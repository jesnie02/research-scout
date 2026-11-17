"""Turn extracted constraints into inclusive intervals. Deterministic code, no LLM."""

import math
import re
from datetime import date

from scout.models import CitationConstraint, Interval, Intervals, YearConstraint

ABOUT_TOLERANCE = 0.2
RECENT_YEARS = 3

Bounds = tuple[int | None, int | None]


class ConflictingConstraintsError(ValueError):
    """The constraints cannot all hold, so no search is made."""


def _year_bounds(c: YearConstraint) -> Bounds:
    match c.op:
        case "before":
            return None, c.value - 1
        case "after":
            return c.value + 1, None
        case "in":
            return c.value, c.value
        case "from":
            return c.value, None
        case "until":
            return None, c.value


def _citation_bounds(c: CitationConstraint) -> Bounds:
    match c.op:
        case "over":
            return c.value + 1, None
        case "at_least":
            return c.value, None
        case "under":
            return None, c.value - 1
        case "at_most":
            return None, c.value
        case "about":
            low = math.ceil(round(c.value * (1 - ABOUT_TOLERANCE), 6))
            high = math.floor(round(c.value * (1 + ABOUT_TOLERANCE), 6))
            return low, high


def _merge(bounds: list[Bounds]) -> Interval:
    mins = [low for low, _ in bounds if low is not None]
    maxs = [high for _, high in bounds if high is not None]
    return Interval(min=max(mins) if mins else None, max=min(maxs) if maxs else None)


def _describe(op: str, value: int) -> str:
    return f"{op.replace('_', ' ')} {value}"


def to_intervals(
    year: list[YearConstraint],
    citations: list[CitationConstraint],
    *,
    recent: bool,
    today: date,
) -> Intervals:
    """Merge all constraints per field into one inclusive interval (strictest bound wins)."""
    notes: list[str] = []
    year_bounds = [_year_bounds(c) for c in year]
    year_terms = [_describe(c.op, c.value) for c in year]
    if recent:
        first = today.year - (RECENT_YEARS - 1)
        year_bounds.append((first, today.year))
        year_terms.append("recent")
        notes.append(f"'recent' was read as published {first}-{today.year}.")

    citation_bounds = [_citation_bounds(c) for c in citations]
    for c in citations:
        if c.op == "about":
            low, high = _citation_bounds(c)
            notes.append(f"'about {c.value}' was read as {low}-{high} citations (+/-20 %).")

    year_interval = _merge(year_bounds)
    citation_interval = _merge(citation_bounds)

    if (
        year_interval.min is not None
        and year_interval.max is not None
        and year_interval.min > year_interval.max
    ):
        raise ConflictingConstraintsError(
            f"The year constraints contradict each other ({', '.join(year_terms)}): "
            "no publication year satisfies all of them."
        )
    citation_max = citation_interval.max
    citation_min = citation_interval.min if citation_interval.min is not None else 0
    if citation_max is not None and citation_min > citation_max:
        terms = ", ".join(_describe(c.op, c.value) for c in citations)
        raise ConflictingConstraintsError(
            f"The citation constraints contradict each other ({terms}): "
            "no citation count satisfies all of them."
        )
    return Intervals(year=year_interval, citations=citation_interval, notes=notes)


def _filter_part(field: str, low: int | None, high: int | None) -> str | None:
    # Syntax seen working against the API (range) or documented by OpenAlex (> and <).
    if low is None and high is None:
        return None
    if low is not None and high is not None:
        return f"{field}:{low}" if low == high else f"{field}:{low}-{high}"
    if low is not None:
        return f"{field}:>{low - 1}"
    return f"{field}:<{high + 1}" if high is not None else None


def openalex_filter(intervals: Intervals) -> str:
    """Build the OpenAlex `filter=` value. A citation minimum of 0 is no constraint."""
    citation_min = intervals.citations.min
    parts = [
        _filter_part("publication_year", intervals.year.min, intervals.year.max),
        _filter_part(
            "cited_by_count",
            citation_min if citation_min is not None and citation_min > 0 else None,
            intervals.citations.max,
        ),
    ]
    return ",".join(part for part in parts if part)


_ABBREVIATIONS = re.compile(r"\b(et al|e\.g|i\.e|vs|etc|fig|eq|approx|cf)\.", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def count_sentences(text: str) -> int:
    """Count sentences ending in . ! or ? plus whitespace; common abbreviations do not split."""
    cleaned = _ABBREVIATIONS.sub(lambda m: m.group(0)[:-1], text.strip())
    return len([part for part in _SENTENCE_END.split(cleaned) if part.strip()])
