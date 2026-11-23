
from enum import Enum
from functools import lru_cache
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, TypeAdapter, create_model, field_validator, model_validator


class AssessmentType(str, Enum):
    QUIZ = "quiz"
    HOMEWORK = "homework"
    EXAM = "exam"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    MIXED = "mixed"


class QuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    PROBLEM_SOLVING = "problem_solving"
    ESSAY = "essay"


class Language(str, Enum):
    ARABIC = "Arabic"
    ENGLISH = "English"


class AssessmentRequest(BaseModel):
    subject: str = Field(min_length=1)
    grade: str = Field(min_length=1)
    topic: str = Field(min_length=1)

    assessment_type: AssessmentType
    difficulty: Difficulty

    question_types: list[QuestionType] = Field(min_length=1)
    question_count: int = Field(ge=1, le=100)
    total_marks: float = Field(gt=0, le=1000)
    duration_minutes: int = Field(ge=1, le=300)

    language: Language = Language.ENGLISH
    additional_instructions: str = ""

    @field_validator("question_types")
    @classmethod
    def question_types_must_be_unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("question_types must not contain duplicates")
        return value


class AnswerOption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class MarkingPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1)
    marks: float = Field(gt=0)


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1)


class _GeneratedQuestion(BaseModel):
    """Common generated content; marks and IDs are assigned deterministically."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question_id: str | None = None
    question_number: int | None = Field(default=None, ge=1)
    topic: str | None = None
    question_text: str = Field(min_length=1)
    difficulty: Literal[Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]


class GeneratedMCQ(_GeneratedQuestion):
    question_type: Literal[QuestionType.MCQ]
    options: list[AnswerOption] = Field(min_length=2)
    correct_answer: StrictStr = Field(min_length=1)
    explanation: str = Field(min_length=1)

    @field_validator("options")
    @classmethod
    def distinct_options(cls, options):
        ids = [item.id.strip().casefold() for item in options]
        texts = [item.text.strip().casefold() for item in options]
        if len(ids) != len(set(ids)) or len(texts) != len(set(texts)):
            raise ValueError("MCQ option IDs and texts must be unique")
        return options


class GeneratedTrueFalse(_GeneratedQuestion):
    question_type: Literal[QuestionType.TRUE_FALSE]
    correct_answer: StrictBool
    explanation: str = Field(min_length=1)

    @field_validator("correct_answer", mode="before")
    @classmethod
    def normalize_boolean_answer(cls, value):
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "t", "yes", "1", "\u0635\u062d", "\u0635\u062d\u064a\u062d"}:
                return True
            if normalized in {"false", "f", "no", "0", "\u062e\u0637\u0623", "\u062e\u0627\u0637\u0626"}:
                return False
        return value


class GeneratedEssay(_GeneratedQuestion):
    question_type: Literal[QuestionType.ESSAY]
    model_answer: str = Field(min_length=1)
    grading_rubric: list[RubricCriterion] = Field(min_length=1)


class GeneratedShortAnswer(_GeneratedQuestion):
    question_type: Literal[QuestionType.SHORT_ANSWER]
    expected_answer: str = Field(min_length=1)
    acceptable_alternatives: list[str]
    explanation: str = Field(min_length=1)


class GeneratedProblemSolving(_GeneratedQuestion):
    question_type: Literal[QuestionType.PROBLEM_SOLVING]
    expected_answer: str = Field(min_length=1)
    solution_steps: list[str] = Field(min_length=1)
    explanation: str = Field(min_length=1)


GeneratedQuestion = Annotated[
    Union[GeneratedMCQ, GeneratedTrueFalse, GeneratedEssay, GeneratedShortAnswer, GeneratedProblemSolving],
    Field(discriminator="question_type"),
]
generated_question_adapter = TypeAdapter(GeneratedQuestion)
_GENERATED_VARIANTS = {
    QuestionType.MCQ: GeneratedMCQ,
    QuestionType.TRUE_FALSE: GeneratedTrueFalse,
    QuestionType.ESSAY: GeneratedEssay,
    QuestionType.SHORT_ANSWER: GeneratedShortAnswer,
    QuestionType.PROBLEM_SOLVING: GeneratedProblemSolving,
}


@lru_cache(maxsize=16)
def assessment_output_schema(question_types: tuple[str, ...]) -> type[BaseModel]:
    """Create a JSON Schema envelope restricted to the requested question types."""
    variants = [_GENERATED_VARIANTS[QuestionType(value)] for value in question_types]
    if len(variants) == 1:
        question_field = list[variants[0]]
    else:
        question_union = Annotated[Union[tuple(variants)], Field(discriminator="question_type")]
        question_field = list[question_union]
    return create_model(
        "TeacherAssessmentOutput_" + "_".join(question_types),
        __config__=ConfigDict(extra="forbid"),
        title=(str, Field(min_length=1)),
        instructions=(list[str], ...),
        questions=(question_field, Field(min_length=1)),
        duration_minutes=(int, Field(gt=0)),
    )


class AssessmentQuestion(BaseModel):
    question_number: int = Field(ge=1)
    question_type: QuestionType
    difficulty: Difficulty

    question_text: str = Field(min_length=1)
    marks: float = Field(gt=0)
    question_id: str = ""
    topic: str = ""
    maximum_score: float | None = None

    options: list[AnswerOption] = Field(default_factory=list)
    correct_answer: StrictStr | StrictBool | None = None
    explanation: str = ""

    marking_scheme: list[MarkingPoint] = Field(default_factory=list)
    model_answer: str | None = None
    grading_rubric: list[MarkingPoint] = Field(default_factory=list)
    expected_answer: str | None = None
    acceptable_alternatives: list[str] = Field(default_factory=list)


class AssessmentResult(BaseModel):
    title: str = Field(min_length=1)
    instructions: list[str] = Field(default_factory=list)

    questions: list[AssessmentQuestion] = Field(min_length=1)

    total_marks: float = Field(gt=0)
    duration_minutes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_assessment(self):
        if len(self.questions) == 0:
            raise ValueError("Assessment must contain questions.")

        actual_marks = sum(q.marks for q in self.questions)

        if abs(actual_marks - self.total_marks) > 0.001:
            raise ValueError(
                f"Question marks sum ({actual_marks}) "
                f"does not match total marks ({self.total_marks})."
            )

        for question in self.questions:
            if question.question_type == QuestionType.MCQ:
                if len(question.options) < 2:
                    raise ValueError(
                        f"Question {question.question_number}: "
                        "MCQ must have at least two options."
                    )

                option_ids = [option.id for option in question.options]

                if len(option_ids) != len(set(option_ids)):
                    raise ValueError(
                        f"Question {question.question_number}: "
                        "Duplicate option IDs."
                    )

                if (question.correct_answer is not None and question.correct_answer not in option_ids):
                    raise ValueError(
                        f"Question {question.question_number}: "
                        "Correct answer must match one option ID."
                    )
            elif question.options:
                raise ValueError(
                    f"Question {question.question_number}: non-MCQ questions cannot have MCQ options."
                )

        return self


class GeneratedAssessment(BaseModel):
    student_version: AssessmentResult
    teacher_version: AssessmentResult

    @model_validator(mode="after")
    def validate_versions(self):
        student = self.student_version
        teacher = self.teacher_version

        if len(student.questions) != len(teacher.questions):
            raise ValueError("Student and teacher versions must match.")

        if abs(student.total_marks - teacher.total_marks) > 0.001:
            raise ValueError("Assessment versions must have equal total marks.")

        for question in teacher.questions:
            if question.question_type == QuestionType.TRUE_FALSE and not isinstance(question.correct_answer, bool):
                raise ValueError(f"True/False question {question.question_number} needs a boolean answer.")
            if question.question_type == QuestionType.ESSAY:
                if not question.model_answer or not question.grading_rubric:
                    raise ValueError(f"Essay question {question.question_number} needs a model answer and rubric.")
                if abs(sum(point.marks for point in question.grading_rubric) - question.marks) > 0.001:
                    raise ValueError(f"Essay rubric for question {question.question_number} must add to its marks.")
            if question.question_type == QuestionType.SHORT_ANSWER and not question.expected_answer:
                raise ValueError(f"Short-answer question {question.question_number} needs an expected answer.")
            if question.question_type == QuestionType.PROBLEM_SOLVING and not question.expected_answer:
                raise ValueError(f"Problem-solving question {question.question_number} needs an expected answer.")

        for question in student.questions:
            if (question.correct_answer is not None or question.model_answer or question.expected_answer
                    or question.acceptable_alternatives or question.grading_rubric):
                raise ValueError(f"Student question {question.question_number} exposes answer data.")

        return self
