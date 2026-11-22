
from .schemas import (
    ReportRequest,
    StudentReport,
    ClassReport,
    ReportSection,
)


def generate_student_report(
    request: ReportRequest,
    analytics_result,
) -> StudentReport:
    if not isinstance(request, ReportRequest):
        request = ReportRequest.model_validate(request)

    if not request.student_id:
        raise ValueError("student_id is required to generate an individual student report.")

    student = next(
        (
            item
            for item in analytics_result.students
            if item.student_id == request.student_id
        ),
        None,
    )

    if student is None:
        raise ValueError(
            f"Student not found: {request.student_id}"
        )

    sections = [
        ReportSection(
            heading="Academic Performance",
            content=(
                f"{student.student_name} achieved an average of "
                f"{student.average_percentage}% across "
                f"{student.assessment_count} assessment(s). "
                f"Performance status: {student.performance_status}."
            ),
        ),
        ReportSection(
            heading="Strengths",
            content=" ".join(student.strengths) or "No strengths recorded.",
        ),
        ReportSection(
            heading="Areas for Improvement",
            content=" ".join(student.weaknesses) or "No weaknesses recorded.",
        ),
    ]

    recommendations = (
        student.weaknesses + student.strengths
        if request.include_recommendations
        else []
    )

    return StudentReport(
        report_title=request.report_title,
        subject=request.subject,
        grade_level=request.grade_level,
        student_id=student.student_id,
        student_name=student.student_name,
        average_percentage=student.average_percentage,
        performance_status=student.performance_status,
        strengths=student.strengths,
        weaknesses=student.weaknesses,
        sections=sections,
        recommendations=recommendations,
    )


def generate_class_report(
    request: ReportRequest,
    analytics_result,
) -> ClassReport:
    if not isinstance(request, ReportRequest):
        request = ReportRequest.model_validate(request)

    sections = [
        ReportSection(
            heading="Class Overview",
            content=(
                f"The class contains {analytics_result.student_count} "
                f"students with an average performance of "
                f"{analytics_result.class_average}%."
            ),
        ),
        ReportSection(
            heading="Pass Rate",
            content=(
                f"{analytics_result.pass_rate}% of students "
                "achieved the configured pass mark."
            ),
        ),
    ]

    recommendations = (
        analytics_result.recommendations
        if request.include_recommendations
        else []
    )

    return ClassReport(
        report_title=request.report_title,
        subject=request.subject,
        grade_level=request.grade_level,
        student_count=analytics_result.student_count,
        class_average=analytics_result.class_average,
        pass_rate=analytics_result.pass_rate,
        sections=sections,
        recommendations=recommendations,
    )
