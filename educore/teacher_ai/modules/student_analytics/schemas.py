
from typing import Optional

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TopicPerformance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str
    awarded_marks: float = Field(ge=0)
    max_marks: float = Field(gt=0)


class StudentAssessmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_id: str
    student_name: str
    assessment_title: str
    subject: str
    grade_level: str

    awarded_marks: float = Field(ge=0)
    max_marks: float = Field(gt=0)

    assessment_date: Optional[str] = None

    topic_performance: list[TopicPerformance] = Field(
        default_factory=list
    )

    @field_validator("assessment_date")
    @classmethod
    def validate_assessment_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not value.strip():
            return None
        normalized = value.strip()
        try:
            date.fromisoformat(normalized)
        except ValueError:
            try:
                datetime.fromisoformat(normalized.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("assessment_date must be an ISO date or datetime") from exc
        return normalized


class AnalyticsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    grade_level: str

    pass_mark: float = Field(default=50, ge=0, le=100)

    records: list[StudentAssessmentRecord] = Field(min_length=1)


class StudentInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_id: str
    student_name: str

    assessment_count: int = Field(ge=1)
    average_percentage: float = Field(ge=0, le=100)
    latest_percentage: float = Field(ge=0, le=100)

    performance_status: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class TopicInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str
    average_percentage: float = Field(ge=0, le=100)
    students_count: int = Field(ge=1)
    performance_level: str


class AnalyticsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    grade_level: str

    student_count: int = Field(ge=1)
    assessment_count: int = Field(ge=1)

    class_average: float = Field(ge=0, le=100)
    pass_rate: float = Field(ge=0, le=100)

    highest_average: float = Field(ge=0, le=100)
    lowest_average: float = Field(ge=0, le=100)

    students: list[StudentInsight]
    topics: list[TopicInsight]
    recommendations: list[str] = Field(default_factory=list)
