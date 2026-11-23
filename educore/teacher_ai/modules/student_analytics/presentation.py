
from .schemas import AnalyticsResult


def get_analytics_summary(result: AnalyticsResult) -> dict:
    return {
        "subject": result.subject,
        "grade_level": result.grade_level,
        "student_count": result.student_count,
        "assessment_count": result.assessment_count,
        "class_average": result.class_average,
        "pass_rate": result.pass_rate,
        "highest_average": result.highest_average,
        "lowest_average": result.lowest_average,
    }


def get_student_ranking(result: AnalyticsResult) -> list[dict]:
    ranked_students = sorted(
        result.students,
        key=lambda student: student.average_percentage,
        reverse=True,
    )

    return [
        {
            "rank": index + 1,
            "student_id": student.student_id,
            "student_name": student.student_name,
            "average_percentage": student.average_percentage,
            "performance_status": student.performance_status,
        }
        for index, student in enumerate(ranked_students)
    ]


def get_student_details(result: AnalyticsResult) -> list[dict]:
    return [
        {
            "student_id": student.student_id,
            "student_name": student.student_name,
            "assessment_count": student.assessment_count,
            "average_percentage": student.average_percentage,
            "latest_percentage": student.latest_percentage,
            "performance_status": student.performance_status,
            "strengths": student.strengths,
            "weaknesses": student.weaknesses,
        }
        for student in result.students
    ]


def get_topic_analysis(result: AnalyticsResult) -> list[dict]:
    return [
        {
            "topic": topic.topic,
            "average_percentage": topic.average_percentage,
            "students_count": topic.students_count,
            "performance_level": topic.performance_level,
        }
        for topic in sorted(
            result.topics,
            key=lambda topic: topic.average_percentage,
        )
    ]


def get_analytics_report(result: AnalyticsResult) -> dict:
    return {
        "summary": get_analytics_summary(result),
        "student_ranking": get_student_ranking(result),
        "student_details": get_student_details(result),
        "topic_analysis": get_topic_analysis(result),
        "recommendations": result.recommendations,
    }
