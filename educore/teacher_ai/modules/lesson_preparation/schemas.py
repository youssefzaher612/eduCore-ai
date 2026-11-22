"""
Schemas for the Lesson Preparation module (Pydantic v2).

* LessonRequest        - what the teacher asks for (unchanged field names/limits).
* GeneratedLesson      - the structure the AI must return, with nested models.
* parse_generated_lesson(data) -> dict
                         validates structure and returns a normalised plain dict
                         (strings stripped, numbers as ints, optional gaps filled).
* check_lesson_against_request(lesson, request) -> RequestCheck
                         checks that need the request: practice-question count,
                         lesson-flow timing, preparation-mode consistency.

Validation philosophy
---------------------
Core fields that a teacher cannot work without (titles, questions, answers,
solution steps, durations) are required and non-empty. Secondary descriptive
fields default to "" so one missing sentence does not discard an otherwise useful
lesson. Structure is validated here; educational correctness is NOT - that is the
job of the verification and review stages.
"""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

# ---------------------------------------------------------------------------
# 1. Request (public interface, unchanged)
# ---------------------------------------------------------------------------


def _none_to_empty(value):
    return "" if value is None else value


class LessonRequest(BaseModel):

    model_config = ConfigDict(str_strip_whitespace=True)

    # Basic lesson information
    subject: str = Field(min_length=2, max_length=100)
    grade: str = Field(min_length=1, max_length=50)
    topic: str = Field(min_length=3, max_length=200)

    # Lesson preparation preferences
    preparation_mode: Literal[
        "Prepare my lesson",
        "Understand and prepare"
    ] = "Prepare my lesson"

    language: Literal[
        "English",
        "Arabic",
        "Egyptian Arabic"
    ] = "English"

    duration_minutes: int = Field(default=45, ge=15, le=180)

    # Classroom context
    student_level: Literal[
        "Beginner",
        "Intermediate",
        "Advanced",
        "Mixed"
    ] = "Mixed"

    teaching_style: Literal[
        "Interactive",
        "Visual",
        "Practical",
        "Traditional",
        "Mixed"
    ] = "Interactive"

    # Assessment preferences
    question_count: int = Field(default=3, ge=1, le=20)

    # Optional teacher instructions (treated as untrusted text by the prompts)
    extra_instructions: Annotated[
        str, BeforeValidator(_none_to_empty), Field(max_length=1000)
    ] = ""


# ---------------------------------------------------------------------------
# 2. Reusable field types
# ---------------------------------------------------------------------------

DIFFICULTY_LABELS = ("Basic", "Intermediate", "Advanced")


def _scalar_text(value):
    """Accept text; accept plain numbers as text (answers are often numeric)."""
    if isinstance(value, bool) or value is None:
        raise ValueError("must be text")
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        raise ValueError(f"must be text, not {type(value).__name__}")
    return value.strip()


def _optional_text(value):
    return "" if value is None else _scalar_text(value)


def _text_list(value):
    """A list of text items. A lone string becomes a one-item list; blank items are dropped."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"must be a list of text items, not {type(value).__name__}")
    items = []
    for index, item in enumerate(value):
        if item is None:
            continue
        if isinstance(item, (dict, list, tuple, bool)):
            raise ValueError(f"item {index + 1} must be text, not {type(item).__name__}")
        text = _scalar_text(item)
        if text:
            items.append(text)
    return items


def _minutes(value):
    """Whole, positive minutes. Booleans, strings and fractional values are rejected."""
    if isinstance(value, bool):
        raise ValueError("must be a whole number of minutes, not true/false")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("must be a whole number of minutes")
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"must be a whole number of minutes, not {type(value).__name__}")
    if value <= 0:
        raise ValueError("must be greater than zero")
    return value


def _difficulty(value):
    if isinstance(value, str):
        cleaned = value.strip().capitalize()
        if cleaned in DIFFICULTY_LABELS:
            return cleaned
    raise ValueError("must be exactly one of: Basic, Intermediate, Advanced")


def _normalized_difficulty(value, default="Basic"):
    if isinstance(value, str):
        aliases = {
            "basic": "Basic", "easy": "Basic", "beginner": "Basic",
            "intermediate": "Intermediate", "medium": "Intermediate",
            "advanced": "Advanced", "hard": "Advanced", "challenging": "Advanced",
        }
        return aliases.get(value.strip().casefold(), default)
    return default


def _strict_bool(value):
    if isinstance(value, bool):
        return value
    raise ValueError("must be true or false")


Text = Annotated[str, BeforeValidator(_scalar_text), Field(min_length=1)]
OptText = Annotated[str, BeforeValidator(_optional_text)]
TextList = Annotated[list[Text], BeforeValidator(_text_list)]
RequiredTextList = Annotated[list[Text], BeforeValidator(_text_list), Field(min_length=1)]
Minutes = Annotated[int, BeforeValidator(_minutes)]
Difficulty = Annotated[Literal["Basic", "Intermediate", "Advanced"], BeforeValidator(_difficulty)]
Flag = Annotated[bool, BeforeValidator(_strict_bool)]


class _Model(BaseModel):
    # Unknown keys are kept (not silently dropped) so nothing the AI wrote disappears.
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# 3. Generated-lesson models
# ---------------------------------------------------------------------------

class LessonOverview(_Model):
    central_idea: Text
    why_it_matters: OptText = ""
    big_picture: OptText = ""
    expected_outcome: Text


class LearningObjective(_Model):
    objective: Text
    success_criteria: OptText = ""


class Prerequisite(_Model):
    knowledge: Text
    why_needed: OptText = ""
    quick_check: OptText = ""


class Foundation(_Model):
    concept: Text
    deep_explanation: Text
    why_it_works: OptText = ""
    important_connections: OptText = ""


class TeacherExample(_Model):
    problem_or_example: Text
    solution: Text
    reasoning: OptText = ""


class TeacherLearning(_Model):
    include: Flag
    foundations: list[Foundation] = Field(default_factory=list)
    teacher_examples: list[TeacherExample] = Field(default_factory=list)
    knowledge_warnings: TextList = Field(default_factory=list)


class TeachingStrategy(_Model):
    recommended_approach: Text
    pedagogical_reason: OptText = ""
    lesson_narrative: OptText = ""
    engagement_strategy: OptText = ""


class TeacherScriptEntry(_Model):
    say: Text
    ask_students: OptText = ""
    expected_response: OptText = ""
    teacher_follow_up: OptText = ""

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, data):
        return {"say": data} if isinstance(data, str) else data


class LessonStage(_Model):
    stage: Text
    duration_minutes: Minutes
    purpose: OptText = ""
    teacher_actions: TextList = Field(default_factory=list)
    teacher_script: list[TeacherScriptEntry] = Field(default_factory=list)
    student_actions: TextList = Field(default_factory=list)
    transition: OptText = ""


class StepExplanation(_Model):
    step: Text
    teacher_explanation: Text
    reasoning: OptText = ""


class StudentCheck(_Model):
    question: Text
    expected_answer: Text
    what_wrong_answers_reveal: OptText = ""


class ConceptExplanation(_Model):
    concept: Text
    teaching_goal: OptText = ""
    introduction: OptText = ""
    step_by_step_explanation: list[StepExplanation] = Field(default_factory=list)
    teacher_script: OptText = ""
    board_work: TextList = Field(default_factory=list)
    student_check: Optional[StudentCheck] = None

    @model_validator(mode="after")
    def _needs_content(self):
        if not self.teacher_script and not self.step_by_step_explanation:
            raise ValueError("needs a teacher_script or at least one step_by_step_explanation")
        return self


class SolutionStep(_Model):
    step: OptText = ""
    work: Text
    teacher_explanation: OptText = ""
    reasoning: OptText = ""

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, data):
        return {"work": data} if isinstance(data, str) else data


class WorkedExample(_Model):
    title: Text
    difficulty: Difficulty
    problem: Text
    teaching_purpose: OptText = ""
    solution_steps: Annotated[list[SolutionStep], Field(min_length=1)]
    final_answer: Text
    common_student_error: OptText = ""
    how_to_address_it: OptText = ""


class InteractionScenario(_Model):
    situation: Text
    student_response: Text
    teacher_response: Text
    teaching_purpose: OptText = ""


class ClassroomActivity(_Model):
    activity_name: Text
    learning_goal: OptText = ""
    duration_minutes: Minutes
    instructions: TextList = Field(default_factory=list)
    student_task: OptText = ""
    expected_result: OptText = ""
    teacher_guidance: OptText = ""

    @model_validator(mode="after")
    def _needs_content(self):
        if not self.instructions and not self.student_task:
            raise ValueError("needs instructions or a student_task")
        return self


class Misconception(_Model):
    misconception: Text
    why_it_happens: OptText = ""
    how_to_detect: OptText = ""
    how_to_correct: Text


class FormativeCheckpoint(_Model):
    checkpoint: OptText = ""
    question_or_task: Text
    correct_response: Text
    interpretation: OptText = ""
    next_teacher_action: OptText = ""


class PracticeQuestion(_Model):
    question: Text
    difficulty: Difficulty
    step_by_step_solution: RequiredTextList
    correct_answer: Text
    explanation: Text
    common_mistake: OptText = ""
    feedback: OptText = ""


class LessonSummary(_Model):
    key_takeaways: RequiredTextList
    teacher_closing_script: OptText = ""
    final_check_question: OptText = ""
    expected_answer: OptText = ""


class HomeworkTask(_Model):
    question: Text
    purpose: OptText = ""
    difficulty: Difficulty
    answer_or_guidance: Text


class Differentiation(_Model):
    students_needing_support: TextList = Field(default_factory=list)
    students_at_expected_level: TextList = Field(default_factory=list)
    advanced_students: TextList = Field(default_factory=list)

    @model_validator(mode="after")
    def _needs_content(self):
        if not (self.students_needing_support or self.students_at_expected_level
                or self.advanced_students):
            raise ValueError("needs at least one adaptation")
        return self


class GeneratedLesson(_Model):
    """All 19 top-level sections are required; the ones a lesson cannot work without are non-empty."""

    lesson_title: Text
    lesson_overview: LessonOverview
    learning_objectives: Annotated[list[LearningObjective], Field(min_length=1)]
    prerequisites: list[Prerequisite]
    teacher_learning: TeacherLearning
    teaching_strategy: TeachingStrategy
    lesson_flow: Annotated[list[LessonStage], Field(min_length=1)]
    concept_explanations: Annotated[list[ConceptExplanation], Field(min_length=1)]
    worked_examples: Annotated[list[WorkedExample], Field(min_length=1)]
    classroom_interaction: list[InteractionScenario]
    classroom_activities: list[ClassroomActivity]
    common_misconceptions: list[Misconception]
    formative_assessment: list[FormativeCheckpoint]
    practice_questions: Annotated[list[PracticeQuestion], Field(min_length=1)]
    lesson_summary: LessonSummary
    homework: list[HomeworkTask]
    differentiation: Differentiation
    teacher_notes: TextList
    post_lesson_reflection: TextList


class LessonSectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class LessonActivityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activity_title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    estimated_duration: int = Field(ge=1)
    learning_objective: str = Field(min_length=1)


class LessonAssessmentQuestionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str = Field(min_length=1)
    question_type: Literal["short_answer"] = "short_answer"
    question_text: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    explanation: str = ""


class LessonPlan(BaseModel):
    """API/export lesson plan shape; the detailed teacher lesson remains the UI contract."""
    model_config = ConfigDict(extra="forbid")
    lesson_title: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    grade_level: str = Field(min_length=1)
    learning_objectives: list[str] = Field(min_length=1)
    prerequisites: list[str] = Field(default_factory=list)
    estimated_duration: int = Field(ge=1)
    introduction: str = Field(min_length=1)
    lesson_sections: list[LessonSectionOutput] = Field(min_length=1)
    teaching_activities: list[LessonActivityOutput] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    assessment_questions: list[LessonAssessmentQuestionOutput] = Field(default_factory=list)
    homework: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)


def adapt_generated_lesson_to_plan(lesson: dict, request: LessonRequest) -> LessonPlan:
    """Map the rich lesson contract to a concise, independently validated plan."""
    objectives = [item.get("objective", "") for item in lesson.get("learning_objectives", [])]
    stages = lesson.get("lesson_flow", [])
    first_stage = stages[0] if stages else {}
    scripts = first_stage.get("teacher_script", [])
    introduction = (scripts[0].get("say", "") if scripts else "") or " ".join(first_stage.get("teacher_actions", []))

    sections = []
    for concept in lesson.get("concept_explanations", []):
        steps = concept.get("step_by_step_explanation", [])
        key_points = [
            (item.get("teacher_explanation", "") if isinstance(item, dict) else str(item))
            for item in steps
        ]
        explanation = concept.get("teacher_script", "") or " ".join(key_points)
        if not explanation:
            continue
        sections.append(LessonSectionOutput(
            title=concept.get("concept", "Concept"), explanation=explanation,
            key_points=[item for item in key_points if item],
            examples=[item for item in concept.get("board_work", []) if item],
        ))

    objective = objectives[0] if objectives else "Apply the lesson's central idea."
    activities = []
    raw_activities = lesson.get("classroom_activities", [])
    if raw_activities:
        for item in raw_activities:
            description = item.get("student_task", "") or " ".join(item.get("instructions", []))
            activities.append(LessonActivityOutput(
                activity_title=item.get("activity_name", "Class activity"),
                description=description or item.get("expected_result", "Practice the target skill."),
                estimated_duration=max(1, int(item.get("duration_minutes", 1))),
                learning_objective=item.get("learning_goal", "") or objective,
            ))
    else:
        for stage in stages:
            description = " ".join(stage.get("teacher_actions", []) + stage.get("student_actions", []))
            activities.append(LessonActivityOutput(
                activity_title=stage.get("stage", "Lesson activity"),
                description=description or stage.get("purpose", "Practice the target skill."),
                estimated_duration=max(1, int(stage.get("duration_minutes", 1))),
                learning_objective=objective,
            ))

    examples = [
        f"{item.get('title', 'Example')}: {item.get('problem', '')} Answer: {item.get('final_answer', '')}"
        for item in lesson.get("worked_examples", [])
    ]
    questions = [
        LessonAssessmentQuestionOutput(
            question_id=f"lesson_question_{index}", question_text=item.get("question", ""),
            expected_answer=item.get("correct_answer", ""), explanation=item.get("explanation", ""),
        )
        for index, item in enumerate(lesson.get("practice_questions", []), start=1)
    ]
    homework = [item.get("question", "") for item in lesson.get("homework", []) if item.get("question")]
    takeaways = lesson.get("lesson_summary", {}).get("key_takeaways", [])
    return LessonPlan(
        lesson_title=lesson.get("lesson_title", request.topic), subject=request.subject,
        grade_level=request.grade, learning_objectives=objectives,
        prerequisites=[item.get("knowledge", "") for item in lesson.get("prerequisites", []) if item.get("knowledge")],
        estimated_duration=request.duration_minutes, introduction=introduction or request.topic,
        lesson_sections=sections, teaching_activities=activities, examples=examples,
        assessment_questions=questions, homework=homework, summary=" ".join(takeaways),
    )


REQUIRED_SECTIONS = tuple(GeneratedLesson.model_fields.keys())


# ---------------------------------------------------------------------------
# 4. Error reporting
# ---------------------------------------------------------------------------

def path_to_text(path) -> str:
    """('practice_questions', 2, 'correct_answer') -> 'practice_questions[2].correct_answer'"""
    text = ""
    for part in path:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += ("." if text else "") + str(part)
    return text


@dataclass
class SchemaProblem:
    path: tuple
    message: str

    @property
    def section(self) -> str:
        return str(self.path[0]) if self.path else ""

    def __str__(self) -> str:
        where = path_to_text(self.path)
        return f"{where}: {self.message}" if where else self.message


class LessonSchemaError(ValueError):
    """The generated lesson does not match the required structure."""

    def __init__(self, problems):
        self.problems = list(problems)
        shown = "; ".join(str(p) for p in self.problems[:8])
        extra = f" (+{len(self.problems) - 8} more)" if len(self.problems) > 8 else ""
        super().__init__(f"Invalid lesson structure: {shown}{extra}")

    @property
    def sections(self) -> list:
        seen = []
        for problem in self.problems:
            if problem.section and problem.section not in seen:
                seen.append(problem.section)
        return seen


_FRIENDLY = {
    "missing": "is required but missing",
    "string_too_short": "must not be empty",
    "too_short": "must contain at least one item",
    "list_type": "must be a list",
    "dict_type": "must be an object",
    "model_type": "must be an object",
}


def format_validation_error(exc: ValidationError) -> list:
    problems = []
    for error in exc.errors():
        loc = tuple(error.get("loc", ()))
        kind = error.get("type", "")
        message = _FRIENDLY.get(kind) or str(error.get("msg", "is invalid"))
        if message.startswith("Value error, "):
            message = message[len("Value error, "):]
        problems.append(SchemaProblem(loc, message))
    return problems


def _normalize_lesson_dict(data: dict) -> dict:
    if not isinstance(data, dict):
        return data

    if "lesson_title" not in data or not data["lesson_title"]:
        data["lesson_title"] = data.get("title") or (data.get("lesson_overview") or {}).get("title") or "Lesson Plan"

    if "lesson_overview" not in data or not isinstance(data["lesson_overview"], dict):
        data["lesson_overview"] = {
            "central_idea": str(data.get("topic") or "Core Topic Concepts"),
            "expected_outcome": "Students will understand and apply the topic concepts.",
        }
    else:
        lo = data["lesson_overview"]
        if not lo.get("central_idea"):
            lo["central_idea"] = str(data.get("topic") or "Core Topic Concepts")
        if not lo.get("expected_outcome"):
            lo["expected_outcome"] = "Students will understand and apply the topic concepts."

    if not data.get("learning_objectives"):
        data["learning_objectives"] = [{"objective": f"Understand the fundamentals of {data.get('lesson_title', 'the topic')}."}]

    if "prerequisites" not in data or not isinstance(data["prerequisites"], list):
        data["prerequisites"] = []

    if "teacher_learning" not in data or not isinstance(data["teacher_learning"], dict):
        data["teacher_learning"] = {
            "include": True,
            "foundations": [],
            "teacher_examples": [],
            "knowledge_warnings": [],
        }
    else:
        tl = data["teacher_learning"]
        if "include" not in tl:
            tl["include"] = True
        if "foundations" not in tl or not isinstance(tl["foundations"], list):
            tl["foundations"] = []
        if "teacher_examples" not in tl or not isinstance(tl["teacher_examples"], list):
            tl["teacher_examples"] = []
        if "knowledge_warnings" not in tl or not isinstance(tl["knowledge_warnings"], list):
            tl["knowledge_warnings"] = []

    if "teaching_strategy" not in data or not isinstance(data["teaching_strategy"], dict):
        data["teaching_strategy"] = {
            "recommended_approach": "Guided instruction with interactive checks",
            "pedagogical_reason": "Allows students to grasp fundamental concepts step by step.",
        }
    else:
        ts = data["teaching_strategy"]
        if "recommended_approach" not in ts or not ts["recommended_approach"]:
            ts["recommended_approach"] = ts.get("approach") or "Guided instruction with interactive checks"

    # Normalize lesson_flow
    stages = data.get("lesson_flow")
    if not stages or not isinstance(stages, list):
        data["lesson_flow"] = [
            {"stage": "Introduction", "duration_minutes": 10, "teacher_actions": ["Introduce key concepts."], "student_actions": ["Listen and take notes."]},
            {"stage": "Core Teaching & Practice", "duration_minutes": 25, "teacher_actions": ["Explain concepts and guide practice."], "student_actions": ["Engage and solve examples."]},
            {"stage": "Wrap Up & Review", "duration_minutes": 10, "teacher_actions": ["Summarize key takeaways."], "student_actions": ["Review and ask questions."]},
        ]
    else:
        for s in stages:
            if not isinstance(s, dict):
                continue
            if "stage" not in s or not s["stage"]:
                s["stage"] = s.get("stage_name") or "Lesson Stage"
            if "duration_minutes" not in s or not s["duration_minutes"]:
                s["duration_minutes"] = 10
            for act_field in ("teacher_actions", "student_actions"):
                val = s.get(act_field)
                if isinstance(val, str):
                    s[act_field] = [val]
                elif not isinstance(val, list):
                    s[act_field] = ["Participate actively."]

    # Normalize concept_explanations
    concepts = data.get("concept_explanations")
    if not concepts or not isinstance(concepts, list):
        data["concept_explanations"] = [
            {"concept": str(data.get("lesson_title") or "Core Topic"), "teacher_script": "Explain the core ideas and definitions clearly."}
        ]
    else:
        for c in concepts:
            if not isinstance(c, dict):
                continue
            if "concept" not in c or not c["concept"]:
                c["concept"] = c.get("concept_name") or "Core Concept"
            if not c.get("teacher_script") and not c.get("step_by_step_explanation"):
                c["teacher_script"] = c.get("explanation") or "Explain the concept thoroughly to the students."

    # Normalize worked_examples
    examples = data.get("worked_examples")
    if not examples or not isinstance(examples, list):
        data["worked_examples"] = [
            {
                "title": "Worked Demonstration",
                "difficulty": "Basic",
                "problem": "Sample problem demonstrating the key concept.",
                "solution_steps": [{"work": "Identify the key components."}, {"work": "Apply the rule step by step."}],
                "final_answer": "Complete worked solution.",
            }
        ]
    else:
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            if "title" not in ex or not ex["title"]:
                ex["title"] = ex.get("example_title") or "Worked Example"
            ex["difficulty"] = _normalized_difficulty(ex.get("difficulty"))
            if "problem" not in ex or not ex["problem"]:
                ex["problem"] = ex.get("problem_statement") or "Sample problem."
            if not ex.get("solution_steps"):
                ex["solution_steps"] = [{"work": "Step 1: Understand problem."}, {"work": "Step 2: Apply concept."}]
            if "final_answer" not in ex or not ex["final_answer"]:
                ex["final_answer"] = "Final solution derived from steps."

    for list_field in (
        "classroom_interaction", "classroom_activities", "common_misconceptions",
        "formative_assessment", "homework", "teacher_notes", "post_lesson_reflection"
    ):
        if list_field not in data or not isinstance(data[list_field], list):
            data[list_field] = []

    # Normalize practice_questions
    pqs = data.get("practice_questions")
    if not pqs or not isinstance(pqs, list):
        data["practice_questions"] = [
            {
                "question": "Explain the main concept in your own words with an example.",
                "difficulty": "Basic",
                "correct_answer": "Accurate explanation demonstrating understanding.",
                "explanation": "Tests core conceptual understanding.",
                "step_by_step_solution": ["State definition", "Provide an example"],
            }
        ]
    else:
        for q in pqs:
            if not isinstance(q, dict):
                continue
            if "question" not in q or not q["question"]:
                q["question"] = q.get("question_text") or "Practice problem on the topic."
            q["difficulty"] = _normalized_difficulty(q.get("difficulty"))
            if not q.get("correct_answer"):
                q["correct_answer"] = "Model answer showing correct application."
            if not q.get("explanation"):
                q["explanation"] = "Standard explanation for the correct answer."
            if not q.get("step_by_step_solution"):
                q["step_by_step_solution"] = ["Step 1: Read carefully", "Step 2: Solve step by step"]
            elif isinstance(q["step_by_step_solution"], list):
                normalized_steps = []
                for step in q["step_by_step_solution"]:
                    if not isinstance(step, dict):
                        normalized_steps.append(step)
                        continue
                    parts = [
                        str(step[key]).strip()
                        for key in ("step", "work", "teacher_explanation", "reasoning")
                        if isinstance(step.get(key), (str, int, float))
                        and not isinstance(step.get(key), bool)
                        and str(step[key]).strip()
                    ]
                    normalized_steps.append(" — ".join(parts) if parts else step)
                q["step_by_step_solution"] = normalized_steps

    if "lesson_summary" not in data or not isinstance(data["lesson_summary"], dict):
        data["lesson_summary"] = {
            "key_takeaways": ["Understand definition and main components.", "Review practical examples."],
            "teacher_closing_script": "Great job today, review your notes for next time.",
        }
    else:
        ls = data["lesson_summary"]
        if not ls.get("key_takeaways"):
            ls["key_takeaways"] = ["Review core lecture concepts."]

    for homework in data.get("homework", []):
        if isinstance(homework, dict):
            homework["difficulty"] = _normalized_difficulty(homework.get("difficulty"))

    if "differentiation" not in data or not isinstance(data["differentiation"], dict):
        data["differentiation"] = {
            "students_needing_support": ["Provide additional worked examples."],
            "students_at_expected_level": ["Complete standard practice questions."],
            "advanced_students": ["Explore deeper extension problems."],
        }
    else:
        diff = data["differentiation"]
        if not diff.get("students_needing_support") and not diff.get("students_at_expected_level") and not diff.get("advanced_students"):
            diff["students_at_expected_level"] = ["Complete standard practice questions."]

    return data


def parse_generated_lesson(data: Any) -> dict:
    """Validate ``data`` and return the normalised lesson as a plain dict."""
    if not isinstance(data, dict):
        raise LessonSchemaError([SchemaProblem((), "The generated lesson must be a JSON object.")])
    data = _normalize_lesson_dict(data)
    try:
        model = GeneratedLesson.model_validate(data)
    except ValidationError as exc:
        raise LessonSchemaError(format_validation_error(exc)) from None
    return model.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# 5. Checks that depend on the request
# ---------------------------------------------------------------------------

@dataclass
class RequestIssue:
    section: str
    message: str

    def __str__(self) -> str:
        return f"{self.section}: {self.message}"


@dataclass
class RequestCheck:
    errors: list = field(default_factory=list)    # block the lesson
    warnings: list = field(default_factory=list)  # teacher attention, non-blocking


def check_lesson_against_request(lesson: dict, request) -> RequestCheck:
    """Request-dependent consistency checks on an already schema-valid lesson."""
    result = RequestCheck()
    err, warn = result.errors, result.warnings

    # Practice-question count
    count = len(lesson.get("practice_questions", []))
    if count != request.question_count:
        err.append(RequestIssue(
            "practice_questions",
            f"expected exactly {request.question_count} practice questions but received {count}."))

    # Lesson-flow timing
    stages = lesson.get("lesson_flow", [])
    total = sum(stage["duration_minutes"] for stage in stages)
    if total != request.duration_minutes:
        err.append(RequestIssue(
            "lesson_flow",
            f"stage durations add up to {total} minutes but the lesson is "
            f"{request.duration_minutes} minutes."))

    # Preparation-mode consistency
    learning = lesson.get("teacher_learning", {})
    deep = request.preparation_mode == "Understand and prepare"
    if learning.get("include") is not deep:
        err.append(RequestIssue(
            "teacher_learning",
            f"include is {learning.get('include')} but the selected mode "
            f"'{request.preparation_mode}' requires {deep}."))
    elif deep and not learning.get("foundations"):
        err.append(RequestIssue(
            "teacher_learning",
            "'Understand and prepare' requires at least one foundation concept."))
    elif not deep and (learning.get("foundations") or learning.get("teacher_examples")
                       or learning.get("knowledge_warnings")):
        warn.append(RequestIssue(
            "teacher_learning",
            "'Prepare my lesson' should leave teacher_learning content empty; it will not be shown."))

    # Classroom feasibility (warnings)
    activities = lesson.get("classroom_activities", [])
    activity_total = sum(a["duration_minutes"] for a in activities)
    if activity_total > request.duration_minutes:
        warn.append(RequestIssue(
            "classroom_activities",
            f"activities total {activity_total} minutes, more than the {request.duration_minutes}-minute lesson."))
    longest_stage = max((s["duration_minutes"] for s in stages), default=0)
    for index, activity in enumerate(activities, start=1):
        if activity["duration_minutes"] > longest_stage:
            warn.append(RequestIssue(
                "classroom_activities",
                f"activity {index} ('{activity['activity_name']}') lasts {activity['duration_minutes']} "
                f"minutes, longer than any lesson stage ({longest_stage})."))
    for index, stage in enumerate(stages, start=1):
        if not stage.get("teacher_script"):
            warn.append(RequestIssue(
                "lesson_flow", f"stage {index} ('{stage['stage']}') has no teacher script."))
    for index, concept in enumerate(lesson.get("concept_explanations", []), start=1):
        if not concept.get("student_check"):
            warn.append(RequestIssue(
                "concept_explanations",
                f"concept {index} ('{concept['concept']}') has no check for understanding."))
    if len(lesson.get("worked_examples", [])) == 1:
        warn.append(RequestIssue("worked_examples", "only one worked example was provided."))
    for key in ("prerequisites", "classroom_interaction", "classroom_activities",
                "common_misconceptions", "formative_assessment", "homework",
                "teacher_notes", "post_lesson_reflection"):
        if not lesson.get(key):
            warn.append(RequestIssue(key, "this section is empty."))
    return result
