
from .schemas import StudentReport, ClassReport, CommunicationResult


def get_student_report_data(report: StudentReport) -> dict:
    return {
        "report_title": report.report_title,
        "subject": report.subject,
        "grade_level": report.grade_level,
        "student_id": report.student_id,
        "student_name": report.student_name,
        "average_percentage": report.average_percentage,
        "performance_status": report.performance_status,
        "strengths": report.strengths,
        "weaknesses": report.weaknesses,
        "sections": [
            {
                "heading": section.heading,
                "content": section.content,
            }
            for section in report.sections
        ],
        "recommendations": report.recommendations,
    }


def get_class_report_data(report: ClassReport) -> dict:
    return {
        "report_title": report.report_title,
        "subject": report.subject,
        "grade_level": report.grade_level,
        "student_count": report.student_count,
        "class_average": report.class_average,
        "pass_rate": report.pass_rate,
        "sections": [
            {
                "heading": section.heading,
                "content": section.content,
            }
            for section in report.sections
        ],
        "recommendations": report.recommendations,
    }


def get_communication_data(
    result: CommunicationResult,
) -> dict:
    return {
        "recipient_type": result.recipient_type,
        "recipient_name": result.recipient_name,
        "subject_line": result.subject_line,
        "message": result.message,
        "tone": result.tone,
        "language": result.language,
    }
