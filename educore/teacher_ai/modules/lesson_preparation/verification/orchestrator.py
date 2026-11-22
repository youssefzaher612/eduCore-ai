"""
Verification orchestration: runs the deterministic verifiers that apply to a lesson and
merges their reports without duplicating any of their logic.

Public interface
----------------
    LessonVerifier(verifiers=None).verify(lesson, request=None) -> VerificationOutcome
    verify_lesson(lesson, request=None) -> VerificationOutcome        (default verifiers)

Rules
-----
* Applicability. A verifier that defines ``applies_to(request)`` decides for itself
  (ChemistryVerifier does). The math verifier has no such method, so the subject domains
  from prompts.detect_subject_domains decide: arithmetic/algebra checks run for
  mathematics, physics, chemistry, integrated science and computer science, and are skipped
  for history, languages, biology etc. so prose cannot cause false accusations.
  With ``request=None`` every verifier runs.
* Nothing is hidden. Every issue keeps its severity, code, path and suggestion; item
  records keep their status; limitations are merged without repeats; skipped verifiers
  are listed with the reason.
* A verifier that crashes never passes silently and never takes the pipeline down: it is
  reported as a WARNING issue (``verification.verifier_error``) and the lesson is NOT
  presented as fully verified.
* Deterministic CRITICAL issues are final. Nothing downstream (AI review, repair
  attempts) can remove or downgrade them; ``VerificationOutcome`` is the single source of
  truth for "was this lesson found wrong by computation".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .chemistry_verifier import ChemistryVerifier
from .math_verifier import MathVerifier
from .models import CRITICAL, FAILED, INFO, WARNING, Issue, VerificationReport

logger = logging.getLogger(__name__)

# Subject domains (from prompts.detect_subject_domains) for which the arithmetic /
# algebra checks of MathVerifier are meaningful.
MATH_APPLICABLE_DOMAINS = frozenset(
    {"mathematics", "physics", "chemistry", "integrated_science", "computer_science"}
)

# Overall deterministic status values.
STATUS_FAILED = "failed"                  # at least one critical issue / failed item
STATUS_ATTENTION = "needs_attention"      # warnings, a crashed verifier, or nothing machine-checked
STATUS_PASSED = "passed"                  # checks ran, found nothing wrong
STATUS_NOT_APPLICABLE = "not_applicable"  # no verifier applies to this subject


def _domains(request) -> set:
    from ..prompts import detect_subject_domains
    return set(detect_subject_domains(request))


def verifier_applies(verifier, request) -> bool:
    """Whether ``verifier`` should run for ``request`` (None -> always)."""
    if request is None:
        return True
    own = getattr(verifier, "applies_to", None)
    if callable(own):
        return bool(own(request))
    if isinstance(verifier, MathVerifier):
        return bool(MATH_APPLICABLE_DOMAINS & _domains(request))
    return True  # unknown verifier without applies_to: run it rather than skip checks


@dataclass
class VerificationOutcome:
    """Merged result of all verifiers that ran for one lesson."""

    report: VerificationReport = field(default_factory=VerificationReport)
    ran: list = field(default_factory=list)          # verifier names that ran
    skipped: dict = field(default_factory=dict)      # verifier name -> reason
    errors: dict = field(default_factory=dict)       # verifier name -> short error text

    # -- convenience ---------------------------------------------------------
    @property
    def issues(self) -> list:
        return self.report.issues

    @property
    def critical_issues(self) -> list:
        return [i for i in self.report.issues if i.severity == CRITICAL]

    @property
    def has_critical(self) -> bool:
        return bool(self.critical_issues) or self.report.status_counts().get(FAILED, 0) > 0

    @property
    def status(self) -> str:
        if self.has_critical:
            return STATUS_FAILED
        if not self.ran:
            return STATUS_NOT_APPLICABLE
        if self.errors or self.report.count(WARNING):
            return STATUS_ATTENTION
        return STATUS_PASSED

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "ran": list(self.ran),
            "skipped": dict(self.skipped),
            "errors": dict(self.errors),
            "counts": {
                CRITICAL: self.report.count(CRITICAL),
                WARNING: self.report.count(WARNING),
                INFO: self.report.count(INFO),
            },
            "item_status_counts": self.report.status_counts(),
            "issues": [issue.to_dict() for issue in self.report.issues],
            "records": [record.to_dict() for record in self.report.records],
            "limitations": list(self.report.limitations),
        }


class LessonVerifier:
    """Runs every applicable verifier and merges the results."""

    def __init__(self, verifiers: Optional[list] = None):
        self.verifiers = list(verifiers) if verifiers is not None else [MathVerifier(), ChemistryVerifier()]

    def verify(self, lesson: dict, request=None) -> VerificationOutcome:
        outcome = VerificationOutcome()
        for verifier in self.verifiers:
            name = getattr(verifier, "name", type(verifier).__name__)
            try:
                applies = verifier_applies(verifier, request)
            except Exception as exc:  # a broken applicability check must not hide the verifier
                logger.warning("applicability check failed for %s: %s", name, type(exc).__name__)
                applies = True
            if not applies:
                outcome.skipped[name] = "not applicable to this subject"
                continue
            try:
                partial = verifier.verify(lesson, request)
            except Exception as exc:
                logger.warning("lesson verification_task=%s stage=failed error_category=%s",
                               name, type(exc).__name__)
                outcome.ran.append(name)
                outcome.errors[name] = f"{type(exc).__name__}: {exc}"[:300]
                outcome.report.add(Issue(
                    WARNING, "verification.verifier_error",
                    f"The {name} verifier could not complete, so its checks did not run. "
                    "Treat this lesson's calculations as unverified.", (),
                    source="deterministic"))
                continue
            outcome.ran.append(name)
            outcome.report.extend(partial)

        if outcome.skipped:
            outcome.report.limitations.append(
                "No deterministic checker applies to: " + ", ".join(sorted(outcome.skipped))
                + " (subject-specific content in this lesson was not machine-verified by them).")
        if not outcome.ran:
            outcome.report.limitations.append(
                "No deterministic verifier applies to this subject: the lesson content is not "
                "machine-verified and relies on the teacher's own review.")
        return outcome


def verify_lesson(lesson: dict, request=None) -> VerificationOutcome:
    return LessonVerifier().verify(lesson, request)
