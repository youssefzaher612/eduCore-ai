
import re
from fractions import Fraction

import sympy as sp

from .schemas import AssessmentQuestion, QuestionType


def _parse_linear_equation(question_text: str):
    """
    Parse a simple linear equation in one variable.

    Supported examples:
        Solve for x: 3x - 7 = 2x + 5
        2x + 3 = 11
        4y - 8 = 12

    Unsupported expressions return None.
    """

    text = question_text.strip()

    text = re.sub(
        r"^(solve\s+for\s+[a-z]\s*:\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = (
        text.replace("−", "-")
        .replace("×", "*")
        .replace("–", "-")
    )

    if text.count("=") != 1:
        return None

    left, right = text.split("=", 1)

    variable_matches = re.findall(
        r"[a-zA-Z]",
        left + right,
    )

    variables = set(variable_matches)

    if len(variables) != 1:
        return None

    variable = next(iter(variables))

    allowed = re.compile(r"^[0-9a-zA-Z+\-*/().\s]+$")

    if not allowed.fullmatch(left) or not allowed.fullmatch(right):
        return None

    symbol = sp.Symbol(variable)

    try:
        for side_name, expression in [
            ("left", left),
            ("right", right),
        ]:
            expression = re.sub(
                r"(\d)([a-zA-Z])",
                r"\1*\2",
                expression,
            )
            expression = re.sub(
                r"(\d)\s*\(",
                r"\1*(",
                expression,
            )
            expression = re.sub(
                r"\)\s*\(",
                r")*(",
                expression,
            )

            if side_name == "left":
                left = expression
            else:
                right = expression

        lhs = sp.sympify(
            left,
            locals={variable: symbol},
            rational=True,
        )

        rhs = sp.sympify(
            right,
            locals={variable: symbol},
            rational=True,
        )

        equation = sp.expand(lhs - rhs)

        if equation.has(sp.Pow):
            return None

        polynomial = sp.Poly(equation, symbol)

        if polynomial.degree() > 1 or polynomial.degree() < 0:
            return None

        solutions = sp.solve(equation, symbol)

        if len(solutions) != 1:
            return None

        solution = solutions[0]

        if solution.is_real is not True:
            return None

        return solution

    except Exception:
        return None


def _parse_numeric_option(
    value: str,
    variable: str | None = None,
):
    text = value.strip()

    text = (
        text.replace("ΓêÆ", "-")
        .replace("−", "-")
        .replace("–", "-")
    )

    # Support assignments such as "x = 3".
    if variable:
        assignment = re.fullmatch(
            rf"\s*{re.escape(variable)}\s*=\s*(.+?)\s*",
            text,
            flags=re.IGNORECASE,
        )

        if assignment:
            text = assignment.group(1)

    match = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?)\s*",
        text,
    )

    if not match:
        return None

    try:
        return sp.Rational(
            str(Fraction(match.group(1).replace(" ", "")))
        )
    except (ValueError, ZeroDivisionError):
        return None


def validate_linear_mcq(
    question: AssessmentQuestion,
) -> list[str]:
    """
    Validate supported linear-equation MCQs.

    Unsupported questions are intentionally skipped.
    """

    if question.question_type != QuestionType.MCQ:
        return []

    solution = _parse_linear_equation(question.question_text)

    if solution is None:
        return []

    # Extract the variable from the equation, not the instruction text.
    equation_text = question.question_text.strip()

    equation_text = re.sub(
        r"^solve\s+for\s+[a-z]\s*:\s*",
        "",
        equation_text,
        flags=re.IGNORECASE,
    )

    if equation_text.count("=") != 1:
        return []

    left, right = equation_text.split("=", 1)

    variables = set(
        re.findall(r"[a-zA-Z]", left + right)
    )

    if len(variables) != 1:
        return []

    variable = next(iter(variables))

    matching_options = []

    for option in question.options:
        option_value = _parse_numeric_option(
            option.text,
            variable,
        )

        if (
            option_value is not None
            and sp.simplify(option_value - solution) == 0
        ):
            matching_options.append(option.id)

    if len(matching_options) != 1:
        options_debug = [
            f"{option.id}: {option.text}"
            for option in question.options
        ]

        return [
            f"Question {question.question_number}: "
            "Linear equation does not have exactly one matching "
            "numeric option. "
            f"Equation: {question.question_text!r}. "
            f"Calculated solution: {sp.sstr(solution)}. "
            f"Options: {options_debug}. "
            f"Model correct_answer: {question.correct_answer!r}."
        ]

    if question.correct_answer != matching_options[0]:
        return [
            f"Question {question.question_number}: "
            "Incorrect mathematical answer. "
            f"Expected option {matching_options[0]}, "
            f"got {question.correct_answer!r}."
        ]

    return []
