"""Student analytics and planning over topics discovered from course RAG files."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


BKT = {"p_init": 0.20, "p_learn": 0.08, "p_guess": 0.20, "p_slip": 0.10}
MIN_ATTEMPTS = 3


def bkt_update(prior: float, correct: bool) -> float:
    """Apply one Bayesian Knowledge Tracing observation."""
    if correct:
        numerator = prior * (1 - BKT["p_slip"])
        denominator = numerator + (1 - prior) * BKT["p_guess"]
    else:
        numerator = prior * BKT["p_slip"]
        denominator = numerator + (1 - prior) * (1 - BKT["p_guess"])
    posterior = numerator / denominator if denominator else prior
    return posterior + (1 - posterior) * BKT["p_learn"]


def _topic_catalog(course_topics: Iterable[dict | str]) -> list[dict[str, str]]:
    topics = []
    seen = set()
    for item in course_topics:
        if isinstance(item, str):
            name, skill = item.strip(), "المقرر"
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("topic") or "").strip()
            skill = str(item.get("skill_group") or item.get("skill") or "المقرر").strip()
        else:
            continue
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            topics.append({"name": name[:100], "skill_group": (skill or "المقرر")[:100]})
    return topics


def build_mastery(
    attempts: Iterable,
    course_topics: Iterable[dict | str],
) -> list[dict]:
    """Calculate accuracy and BKT mastery for this course's RAG topics."""
    catalog = _topic_catalog(course_topics)
    names = {item["name"] for item in catalog}
    grouped: dict[str, list] = defaultdict(list)
    for attempt in attempts:
        topic = getattr(attempt, "topic", None)
        if topic in names:
            grouped[topic].append(attempt)

    rows = []
    for topic in catalog:
        name = topic["name"]
        topic_attempts = sorted(
            grouped[name],
            key=lambda item: getattr(item, "created_at", None) or 0,
        )
        mastery = BKT["p_init"]
        for attempt in topic_attempts:
            mastery = bkt_update(mastery, bool(attempt.is_correct))

        count = len(topic_attempts)
        accuracy = (
            round(100 * sum(bool(item.is_correct) for item in topic_attempts) / count, 1)
            if count else None
        )
        avg_time = (
            round(sum(int(item.time_taken_sec or 0) for item in topic_attempts) / count, 1)
            if count else None
        )

        if count < MIN_ATTEMPTS:
            status = "Insufficient data"
        elif accuracy < 50 and 100 * mastery < 60:
            status = "Needs support"
        elif accuracy < 75:
            status = "Developing"
        else:
            status = "On track"

        rows.append({
            "topic": name,
            "skill": topic["skill_group"],
            "attempts_count": count,
            "observed_accuracy_pct": accuracy,
            # The BKT prior is a model initialization, not evidence about this
            # student's actual level. Keep the metric empty until an attempt exists.
            "bkt_mastery_pct": round(100 * mastery, 1) if count else None,
            "avg_time_sec": avg_time,
            "status": status,
        })
    return rows


def create_study_plan(
    attempts: Iterable,
    course_topics: Iterable[dict | str],
    weekly_hours: int = 4,
) -> dict:
    """Create a weekly schedule from topics found in the indexed course files."""
    hours = max(1, min(int(weekly_hours), 40))
    mastery_rows = build_mastery(attempts, course_topics)
    weak_rows = sorted(
        (row for row in mastery_rows if row["attempts_count"] >= MIN_ATTEMPTS),
        key=lambda row: (row["bkt_mastery_pct"] or 0, row["observed_accuracy_pct"] or 0),
    )
    weak_names = {row["topic"] for row in weak_rows}
    remaining_rows = sorted(
        (row for row in mastery_rows if row["topic"] not in weak_names),
        key=lambda row: (
            row["attempts_count"] >= MIN_ATTEMPTS,
            row["bkt_mastery_pct"] or 0,
            row["attempts_count"],
        ),
    )
    # Put weak areas first, then use other course topics so one weak topic does
    # not occupy every day of the weekly plan.
    topics = [row["topic"] for row in weak_rows + remaining_rows]

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    base_minutes, extra_minutes = divmod(hours * 60, len(days))
    activities = [
        "فهم المفاهيم الأساسية",
        "تدريب على أسئلة قصيرة",
        "مراجعة النقاط الصعبة",
        "حل تطبيقات من المحاضرة",
        "استرجاع المفاهيم من الذاكرة",
        "حل أسئلة متنوعة",
        "مراجعة أسبوعية سريعة",
    ]
    return {
        "weekly_hours": hours,
        "plan": [
            {
                "day": day,
                "topic": topics[index % len(topics)],
                "activity": activities[index],
                "minutes": base_minutes + int(index < extra_minutes),
            }
            for index, day in enumerate(days)
        ] if topics else [],
    }
