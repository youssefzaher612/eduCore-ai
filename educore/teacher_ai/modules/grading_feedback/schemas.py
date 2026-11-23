
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator


class GradingQuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    PROBLEM_SOLVING = "problem_solving"
    ESSAY = "essay"


class GradingLanguage(str, Enum):
    ARABIC = "Arabic"
    ENGLISH = "English"


class MarkingCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    max_marks: float = Field(gt=0)


class GradingQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question_text: str = Field(min_length=1)
    question_type: GradingQuestionType
    max_marks: float = Field(gt=0)
    expected_answer: Optional[str] = None
    marking_criteria: list[MarkingCriterion] = Field(default_factory=list)


class StudentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    answer: str


class GradingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1)
    grade_level: str = Field(min_length=1)
    language: GradingLanguage = GradingLanguage.ENGLISH
    teacher_instructions: Optional[str] = None
    course_material: Optional[str] = Field(default=None, max_length=12000)
    questions: list[GradingQuestion] = Field(min_length=1)
    student_answers: list[StudentAnswer] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_question_and_answer_ids(self):
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question IDs must be unique")

        answer_ids = [answer.question_id for answer in self.student_answers]
        if len(answer_ids) != len(set(answer_ids)):
            raise ValueError("student answers must contain at most one answer per question")

        unknown_ids = set(answer_ids) - set(question_ids)
        if unknown_ids:
            raise ValueError(
                "student answers reference unknown question IDs: "
                + ", ".join(sorted(unknown_ids))
            )

        for question in self.questions:
            criterion_ids = [item.criterion_id for item in question.marking_criteria]
            if len(criterion_ids) != len(set(criterion_ids)):
                raise ValueError(f"marking criterion IDs must be unique for question {question.question_id}")
            if question.marking_criteria:
                criteria_max = sum(item.max_marks for item in question.marking_criteria)
                if criteria_max > question.max_marks + 0.01:
                    raise ValueError(
                        f"marking criteria for question {question.question_id} exceed its maximum marks"
                    )
        return self


class CriterionGrade(BaseModel):
    model_config = ConfigDict(extra="ignore")

    criterion_id: str
    awarded_marks: float = Field(ge=0)
    feedback: str


class QuestionGrade(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: str
    awarded_marks: float = Field(ge=0)
    max_marks: float = Field(gt=0)
    is_fully_correct: bool
    feedback: str
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    criterion_grades: list[CriterionGrade] = Field(default_factory=list)


class GradingResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_awarded_marks: float = Field(ge=0)
    total_max_marks: float = Field(gt=0)
    percentage: float = Field(ge=0, le=100)
    overall_feedback: str
    question_grades: list[QuestionGrade]
