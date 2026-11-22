"""Pydantic contracts for structured student AI responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


class CourseTopicOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    skill_group: str = Field(default="Course topics", max_length=100)
    document_name: str = Field(default="", max_length=255)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_topic_string(cls, value):
        if isinstance(value, str):
            return {"name": value}
        return value


class CourseTopicsOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    topics: list[CourseTopicOutput] = Field(default_factory=list, max_length=12)


class StudyPlanEntryOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    day: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    topic: str = Field(min_length=1, max_length=100)
    activity: str = Field(min_length=1, max_length=500)


class StudyPlanOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    summary: str = Field(default="", max_length=1000)
    plan: list[StudyPlanEntryOutput] = Field(min_length=7, max_length=7)

    @model_validator(mode="after")
    def validate_days(self):
        days = [entry.day for entry in self.plan]
        expected = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if set(days) != set(expected):
            raise ValueError("The plan must contain one entry for each weekday.")
        return self


class StudentExamQuestionOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: StrictInt = Field(ge=0, le=3)
    explanation: str = Field(min_length=1, max_length=1200)
    # The service fills missing metadata from the validated request. Keeping
    # these optional avoids repeating a costly generation for a sound question.
    topic: str = Field(default="", max_length=100)
    difficulty: str = Field(default="", max_length=20)

    @field_validator("options")
    @classmethod
    def validate_distinct_options(cls, options):
        if any(not option.strip() for option in options):
            raise ValueError("answer options must not be empty")
        if len({option.strip().casefold() for option in options}) != 4:
            raise ValueError("answer options must be distinct")
        return [option.strip() for option in options]


class StudentExamOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=255)
    questions: list[StudentExamQuestionOutput] = Field(min_length=1, max_length=20)
