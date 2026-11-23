"""Data objects shared by all verifiers and by the quality reviewer."""

from dataclasses import dataclass, field
from typing import Optional

CRITICAL = "critical"   # content must NOT be presented as verified
WARNING = "warning"     # content needs teacher attention
INFO = "info"           # optional improvement or scope note

# Item-level verification status (what a deterministic check could establish)
VERIFIED = "verified"                    # the answer itself was checked by computation
PARTIAL = "partially_verified"           # only intermediate steps/arithmetic were checked
UNVERIFIED = "unverified"                # nothing machine-checkable was found/readable
FAILED = "failed"                        # a deterministic check found a contradiction


def path_text(path) -> str:
    text = ""
    for part in path:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += ("." if text else "") + str(part)
    return text


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    path: tuple = ()
    source: str = "deterministic"      # deterministic | structure | ai_review | model_marker
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": list(self.path),
            "path_text": path_text(self.path),
            "source": self.source,
            "suggestion": self.suggestion,
        }

    def key(self):
        return (self.severity, self.code, self.path, self.message)


@dataclass
class ItemRecord:
    """Verification coverage for one question/example/answer."""

    path: tuple
    label: str
    status: str = UNVERIFIED
    passed: list = field(default_factory=list)    # human-readable checks that succeeded
    notes: list = field(default_factory=list)     # why something could not be checked

    def to_dict(self) -> dict:
        return {"path": list(self.path), "path_text": path_text(self.path), "label": self.label,
                "status": self.status, "passed": list(self.passed), "notes": list(self.notes)}


@dataclass
class VerificationReport:
    issues: list = field(default_factory=list)
    records: list = field(default_factory=list)
    limitations: list = field(default_factory=list)

    def add(self, issue: Issue) -> None:
        if all(existing.key() != issue.key() for existing in self.issues):
            self.issues.append(issue)

    def extend(self, other: "VerificationReport") -> None:
        for issue in other.issues:
            self.add(issue)
        self.records.extend(other.records)
        for note in other.limitations:
            if note not in self.limitations:
                self.limitations.append(note)

    def count(self, severity: str) -> int:
        return sum(1 for issue in self.issues if issue.severity == severity)

    def status_counts(self) -> dict:
        counts = {VERIFIED: 0, PARTIAL: 0, UNVERIFIED: 0, FAILED: 0}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts
