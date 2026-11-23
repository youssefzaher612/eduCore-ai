
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_title: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    grade_level: str = Field(min_length=1)

    student_id: Optional[str] = None
    student_name: Optional[str] = None

    include_recommendations: bool = True


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str = Field(min_length=1)
    content: str = Field(min_length=1)


class StudentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_title: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    grade_level: str = Field(min_length=1)

    student_id: str = Field(min_length=1)
    student_name: str = Field(min_length=1)

    average_percentage: float = Field(ge=0, le=100)
    performance_status: str = Field(min_length=1)

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    sections: list[ReportSection] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ClassReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_title: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    grade_level: str = Field(min_length=1)

    student_count: int = Field(ge=1)
    class_average: float = Field(ge=0, le=100)
    pass_rate: float = Field(ge=0, le=100)

    sections: list[ReportSection] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class CommunicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipient_type: str = Field(min_length=1)
    recipient_name: str = Field(min_length=1)

    student_name: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    grade_level: str = Field(min_length=1)

    purpose: str = Field(min_length=1)
    tone: str = Field(default="professional", min_length=1)

    additional_context: Optional[str] = None


class CommunicationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipient_type: str = Field(min_length=1)
    recipient_name: str = Field(min_length=1)

    subject_line: str = Field(min_length=1)
    message: str = Field(min_length=1)

    tone: str = Field(min_length=1)
    language: str = Field(default="English", min_length=1)
