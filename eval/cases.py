"""Format of the evaluation cases. Expectations are written before the agent is run."""

import json
from collections import Counter
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from scout.models import AnswerRequirements, Interval

MIN_CASES = 14
MIN_ANSWER_REQUIREMENT_CASES = 3

Category = Literal[
    "broad_topic",
    "narrow_topic",
    "year_before",
    "year_after",
    "year_exact",
    "citations_min",
    "citations_max",
    "citations_about",
    "combined",
    "recent",
    "no_result",
    "conflicting",
    "ambiguous",
]
# The prompt types the assignment lists explicitly.
REQUIRED_CATEGORIES: frozenset[str] = frozenset(
    {
        "broad_topic",
        "narrow_topic",
        "year_before",
        "year_after",
        "year_exact",
        "citations_min",
        "no_result",
        "ambiguous",
    }
)
# search_error and agent_error are never an expected outcome.
Outcome = Literal[
    "match",
    "no_match",
    "rejected_on_relevance",
    "needs_clarification",
    "conflicting_constraints",
]


class ExpectedCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str | None
    year: Interval
    citations: Interval
    answer_requirements: AnswerRequirements | None
    outcomes: list[Outcome] = Field(min_length=1)
    no_match_evidence: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcomes contains duplicates")
        if "no_match" in self.outcomes and not self.no_match_evidence:
            raise ValueError(
                "no_match_evidence is required when 'no_match' is accepted: "
                "record the direct OpenAlex query, its result count and the date"
            )
        if "match" in self.outcomes and not self.topic:
            raise ValueError("topic is required when 'match' is an accepted outcome")
        if self.outcomes != ["conflicting_constraints"]:
            for name, interval in (("year", self.year), ("citations", self.citations)):
                if (
                    interval.min is not None
                    and interval.max is not None
                    and interval.min > interval.max
                ):
                    raise ValueError(
                        f"{name}: min > max is only allowed for conflicting_constraints"
                    )
        return self


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^C\d{2}$")
    category: Category
    prompt: str = Field(min_length=10)
    expected: ExpectedCase
    notes: str = ""


_CASE_LIST = TypeAdapter(list[Case])


def load_cases(path: Path) -> list[Case]:
    """Validate every case and reject duplicate IDs."""
    cases = _CASE_LIST.validate_python(json.loads(path.read_text()))
    duplicates = sorted(i for i, n in Counter(c.id for c in cases).items() if n > 1)
    if duplicates:
        raise ValueError(f"Duplicate case id(s): {', '.join(duplicates)}")
    return cases


def coverage_problems(cases: list[Case]) -> list[str]:
    """What the set still lacks compared with the assignment and the plan."""
    problems = []
    if len(cases) < MIN_CASES:
        problems.append(f"Only {len(cases)} case(s); the plan needs at least {MIN_CASES}.")
    missing = sorted(REQUIRED_CATEGORIES - {c.category for c in cases})
    if missing:
        problems.append(f"Missing required categories: {', '.join(missing)}.")
    with_requirements = sum(c.expected.answer_requirements is not None for c in cases)
    if with_requirements < MIN_ANSWER_REQUIREMENT_CASES:
        problems.append(
            f"Only {with_requirements} case(s) have answer_requirements; "
            f"at least {MIN_ANSWER_REQUIREMENT_CASES} needed."
        )
    return problems
