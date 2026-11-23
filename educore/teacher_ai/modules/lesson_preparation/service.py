"""
Lesson preparation service: generate -> verify -> review -> (safe) repair -> re-review.

    LessonPreparationService(generator=None, reviewer=None, auto_repair=True).prepare(request)
        -> PreparedLesson

A lesson with critical verification failures is still RETURNED (the teacher needs to see
what is wrong); ``PreparedLesson.report.passed`` is False and the UI must say so. Only a
failure to generate a usable lesson raises (``LessonGenerationError`` or an ``LLMError``).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .generator import GenerationResult, LessonGenerator
from .quality_reviewer import QualityReport, QualityReviewer, RepairResult
from .schemas import LessonPlan, adapt_generated_lesson_to_plan

logger = logging.getLogger(__name__)


@dataclass
class PreparedLesson:
    request: object
    lesson: dict
    report: QualityReport
    metadata: dict = field(default_factory=dict)
    repair: Optional[RepairResult] = None
    lesson_plan: Optional[LessonPlan] = None

    @property
    def passed(self) -> bool:
        return self.report.passed

    @property
    def was_repaired(self) -> bool:
        return bool(self.repair and self.repair.changed)

    def to_dict(self) -> dict:
        return {
            "lesson": self.lesson,
            "lesson_plan": (self.lesson_plan.model_dump(mode="json")
                            if self.lesson_plan is not None else None),
            "report": self.report.to_dict(),
            "metadata": dict(self.metadata),
            "repair": None if self.repair is None else {
                "attempted": self.repair.attempted, "accepted": list(self.repair.accepted),
                "rejected": [list(r) for r in self.repair.rejected], "message": self.repair.message},
        }


class LessonPreparationService:
    def __init__(self, generator: Optional[LessonGenerator] = None,
                 reviewer: Optional[QualityReviewer] = None, auto_repair: bool = True):
        self.generator = generator or LessonGenerator()
        self.reviewer = reviewer or QualityReviewer()
        self.auto_repair = auto_repair

    def prepare(self, request, source_material: str = "") -> PreparedLesson:
        generated: GenerationResult = self.generator.generate(
            request, source_material=source_material
        )
        lesson = generated.lesson
        report = self.reviewer.review(lesson, request, source_material=source_material)
        repair = None

        if self.auto_repair and report.verification.has_critical:
            repair = self.reviewer.repair(
                lesson, request, report, source_material=source_material
            )
            if repair.changed:
                lesson = repair.lesson
                report = self.reviewer.review(
                    lesson, request, source_material=source_material
                )

        # Keep the full lesson document used by the existing teacher UI and also
        # provide the standalone LessonPlan contract for exports/integrations.
        lesson_plan = adapt_generated_lesson_to_plan(lesson, request)
        return PreparedLesson(request=request, lesson=lesson, report=report,
                              metadata=generated.metadata, repair=repair,
                              lesson_plan=lesson_plan)
