
from .schemas import AnalyticsResult


def generate_student_insights(
    result: AnalyticsResult,
) -> AnalyticsResult:
    if not isinstance(result, AnalyticsResult):
        result = AnalyticsResult.model_validate(result)

    for student in result.students:
        strengths = []
        weaknesses = []

        if student.average_percentage >= 85:
            strengths.append("Consistently strong overall performance.")
        elif student.average_percentage >= 70:
            strengths.append("Good overall academic performance.")

        if student.latest_percentage > student.average_percentage:
            strengths.append("Recent performance is improving.")
        elif student.latest_percentage < student.average_percentage:
            weaknesses.append("Recent performance has declined.")

        if student.average_percentage < 50:
            weaknesses.append("Overall performance requires attention.")

        if student.average_percentage < 70:
            weaknesses.append("Needs additional practice and support.")

        if not strengths:
            strengths.append("Completed the recorded assessments.")

        if not weaknesses:
            weaknesses.append("Continue practicing to maintain progress.")

        student.strengths = strengths
        student.weaknesses = weaknesses

    recommendations = []

    if result.pass_rate < 60:
        recommendations.append(
            "Review the main learning objectives and provide "
            "additional support to students below the pass mark."
        )
    elif result.pass_rate < 80:
        recommendations.append(
            "Provide targeted revision sessions to improve "
            "the class pass rate."
        )
    else:
        recommendations.append(
            "Maintain the current teaching approach while "
            "challenging high-performing students."
        )

    weak_topics = [
        topic
        for topic in result.topics
        if topic.average_percentage < 60
    ]

    for topic in weak_topics:
        recommendations.append(
            f"Provide additional lessons and practice exercises "
            f"for the topic: {topic.topic}."
        )

    support_students = [
        student
        for student in result.students
        if student.performance_status == "Needs Support"
    ]

    if support_students:
        recommendations.append(
            f"Arrange individual follow-up for "
            f"{len(support_students)} student(s) who need support."
        )

    if not weak_topics and not support_students:
        recommendations.append(
            "Continue monitoring student progress through "
            "regular assessments."
        )

    result.recommendations = recommendations

    return result
