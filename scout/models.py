"""Typed data shared by the agent, the search tool and the evaluation."""

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

YearOp = Literal["before", "after", "in", "from", "until"]
CitationOp = Literal["over", "at_least", "under", "at_most", "about"]
AnswerTask = Literal["explain_relevance", "summarize_contribution", "assess_usefulness"]


class YearConstraint(BaseModel):
    """`before`/`after` are exclusive; `in`/`from`/`until` are inclusive."""

    model_config = ConfigDict(extra="forbid")
    op: YearOp
    value: int = Field(ge=1500, le=2100)


class CitationConstraint(BaseModel):
    """`over`/`under` are exclusive; `at_least`/`at_most` are inclusive; `about` is +/-20 %."""

    model_config = ConfigDict(extra="forbid")
    op: CitationOp
    value: int = Field(ge=0)


class Interval(BaseModel):
    """Inclusive bounds. None means unbounded on that side."""

    model_config = ConfigDict(extra="forbid")
    min: int | None = None
    max: int | None = None

    def contains(self, value: int | None) -> bool:
        if value is None:
            # A missing value (e.g. unknown citation count) never satisfies a bound.
            return self.min is None and self.max is None
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)


class Intervals(BaseModel):
    """The resolved search constraints, plus notes on how vague terms were interpreted."""

    model_config = ConfigDict(extra="forbid")
    year: Interval
    citations: Interval
    notes: list[str]


class AnswerRequirements(BaseModel):
    """What the answer must contain. None in a request means a relevance explanation."""

    model_config = ConfigDict(extra="forbid")
    task: AnswerTask
    sentences_min: int | None = Field(default=None, ge=1)
    sentences_max: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> Self:
        if (
            self.sentences_min is not None
            and self.sentences_max is not None
            and self.sentences_min > self.sentences_max
        ):
            raise ValueError("sentences_min must not exceed sentences_max")
        return self


class Paper(BaseModel):
    """One candidate exactly as OpenAlex returned it. The LLM never writes these fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    openalex_id: str
    title: str
    authors: list[str]
    year: int
    citations: int | None
    doi: str | None
    url: str
    openalex_url: str
    work_type: str | None
    abstract: str | None
    abstract_truncated: bool
    retrieved_at: datetime
