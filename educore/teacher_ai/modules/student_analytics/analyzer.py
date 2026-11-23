
from collections import defaultdict
from datetime import date, datetime, timezone

from .schemas import (
    AnalyticsRequest,
    AnalyticsResult,
    StudentInsight,
    TopicInsight,
)


def _percentage(awarded: float, maximum: float) -> float:
    return round((awarded / maximum) * 100, 2)


def _record_order(record, input_index: int) -> tuple:
    """Use assessment dates when present; retain input order as a fallback."""
    value = record.assessment_date
    if not value:
        return (0, input_index)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed_date = date.fromisoformat(value)
        parsed = datetime.combine(parsed_date, datetime.min.time(), tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (1, parsed.timestamp())


def analyze_student_performance(
    request: AnalyticsRequest,
) -> AnalyticsResult:
    if not isinstance(request, AnalyticsRequest):
        request = AnalyticsRequest.model_validate(request)

    student_records = defaultdict(list)
    topic_records = defaultdict(list)

    for input_index, record in enumerate(request.records):
        if record.awarded_marks > record.max_marks:
            raise ValueError(
                f"Student {record.student_id}: awarded marks "
                "cannot exceed maximum marks."
            )

        percentage = _percentage(
            record.awarded_marks,
            record.max_marks,
        )

        student_records[record.student_id].append(
            {
                "name": record.student_name,
                "percentage": percentage,
                "record": record,
                "input_index": input_index,
            }
        )

        for topic in record.topic_performance:
            if topic.awarded_marks > topic.max_marks:
                raise ValueError(
                    f"Student {record.student_id}, topic {topic.topic}: "
                    "awarded marks cannot exceed maximum marks."
                )

            topic_records[topic.topic].append(
                (record.student_id, topic.awarded_marks, topic.max_marks)
            )

    student_insights = []

    for student_id, records in student_records.items():
        latest_record = max(records, key=lambda item: _record_order(item["record"], item["input_index"]))

        awarded_total = sum(item["record"].awarded_marks for item in records)
        maximum_total = sum(item["record"].max_marks for item in records)
        average = _percentage(awarded_total, maximum_total)
        latest = latest_record["percentage"]

        if average >= 85:
            status = "Excellent"
        elif average >= request.pass_mark:
            status = "Satisfactory"
        else:
            status = "Needs Support"

        student_insights.append(
            StudentInsight(
                student_id=student_id,
                student_name=latest_record["name"],
                assessment_count=len(records),
                average_percentage=average,
                latest_percentage=latest,
                performance_status=status,
            )
        )

    topic_insights = []

    for topic_name, marks in topic_records.items():
        awarded_total = sum(awarded for _, awarded, _ in marks)
        maximum_total = sum(maximum for _, _, maximum in marks)
        average = _percentage(awarded_total, maximum_total)

        if average >= 85:
            level = "Strong"
        elif average >= 60:
            level = "Moderate"
        else:
            level = "Needs Improvement"

        topic_insights.append(
            TopicInsight(
                topic=topic_name,
                average_percentage=average,
                students_count=len({student_id for student_id, _, _ in marks}),
                performance_level=level,
            )
        )

    student_averages = [
        student.average_percentage
        for student in student_insights
    ]

    class_average = round(
        sum(student_averages) / len(student_averages),
        2,
    )

    pass_count = sum(
        student.average_percentage >= request.pass_mark
        for student in student_insights
    )

    pass_rate = round(
        (pass_count / len(student_insights)) * 100,
        2,
    )

    return AnalyticsResult(
        subject=request.subject,
        grade_level=request.grade_level,
        student_count=len(student_insights),
        assessment_count=len(request.records),
        class_average=class_average,
        pass_rate=pass_rate,
        highest_average=max(student_averages),
        lowest_average=min(student_averages),
        students=student_insights,
        topics=topic_insights,
    )
