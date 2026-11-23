"""
Deterministic mathematical verification (SymPy).

What is checked, and how
------------------------
A. Solutions        The equation in a question is solved symbolically (polynomials of
                    degree 1-4 in one variable, real solutions) and the answer key is
                    compared with the true solution set by substitution.
B. Key vs steps     The conclusion of the written solution must agree with the answer key.
C. Arithmetic       Every purely numeric chain such as '4 + 32 = 36' is evaluated.
D. Discriminant     For a single quadratic: the stated value of b² - 4ac, and claims such as
                    'discriminant is zero' or 'two distinct roots', are checked against the
                    computed discriminant.
E. Equivalence      Consecutive equations in a worked solution (for example completing the
                    square) must have the same solution set; a claimed root must satisfy the
                    previous equation.
F. Identities       Chains like (x-2)(x+3) = x² + 3x - 2x - 6 = x² + x - 6 must be true
                    polynomial identities.
G. Roots -> equation 'Form the equation with roots 2 and -3' is checked by solving the
                    answer's equation and comparing roots.
H. Evaluate         'Calculate 12 / 4 + 3' style questions are evaluated and compared.

Everything is parsed with safe_math (no eval/sympify). Anything that cannot be read
reliably is reported as UNVERIFIED, never as correct.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import sympy as sp

from .extract import Chain, find_chains, find_expression_runs, glue_is_union, sentences
from .models import (CRITICAL, FAILED, INFO, PARTIAL, UNVERIFIED, VERIFIED, WARNING,
                     Issue, ItemRecord, VerificationReport, path_text)
from .safe_math import MathParseError, is_numeric, normalize_math, parse_expression

_CHECK_PREFIX = re.compile(r"^\s*(?:check|verify|verification|checking|substitut|التحقق|تحقق|تأكد)", re.I)
_RESTRICTED_DOMAIN = re.compile(
    r"\b(integers?|positive|negative|natural|whole|rational|length|width|age|time|non-?negative"
    r"|reject|discard|extraneous|not valid|invalid|cannot be negative)\b", re.I)
_COMPLEX_WORDS = re.compile(r"\b(?:complex|imaginary)\b", re.I)
_IMAGINARY_UNIT = re.compile(r"(?<![A-Za-z])i(?![A-Za-z])")
# Tasks such as "find the error in 2 + 3 = 6" contain false statements on purpose.
_ERROR_HUNT = re.compile(r"\b(?:error|mistake|wrong|incorrect|false|spot|misconception)\b", re.I)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _is_check(text: str) -> bool:
    return bool(_CHECK_PREFIX.match(text or ""))


def _parse(text) -> Optional[sp.Expr]:
    try:
        return parse_expression(text)
    except (MathParseError, RecursionError, ValueError, TypeError):
        return None


def _parse_all(parts) -> Optional[list]:
    exprs = [_parse(p) for p in parts]
    return None if any(e is None for e in exprs) else exprs


def fmt(value) -> str:
    return str(value).replace("**", "^")


def fmt_set(values) -> str:
    try:
        ordered = sorted(values, key=lambda v: float(sp.N(v)))
    except (TypeError, ValueError):
        ordered = list(values)
    return "{" + ", ".join(fmt(v) for v in ordered) + "}" if ordered else "no real solutions"


def same_value(a, b) -> bool:
    try:
        if sp.simplify(a - b) == 0:
            return True
        return abs(complex(sp.N(a - b))) < 1e-9
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def sets_equal(left: list, right: list) -> bool:
    if len(left) != len(right):
        return False
    remaining = list(right)
    for value in left:
        for index, other in enumerate(remaining):
            if same_value(value, other):
                del remaining[index]
                break
        else:
            return False
    return True


def contains_value(values: list, target) -> bool:
    return any(same_value(v, target) for v in values)


def _dedupe(values: list) -> list:
    out = []
    for v in values:
        if not contains_value(out, v):
            out.append(v)
    return out


def _decimals(text: str) -> int:
    found = re.findall(r"\d+\.(\d+)", text)
    return max((len(f) for f in found), default=0)


def numbers_match(left_text: str, right_text: str, a, b) -> bool:
    """Exact equality, or equality up to rounding when one side is a plain decimal."""
    if same_value(a, b):
        return True
    for text in (left_text, right_text):
        if re.fullmatch(r"\s*-?\d+\.\d+\s*", text):
            tolerance = 0.5 * 10 ** (-_decimals(text)) * 1.0001
            try:
                return abs(float(sp.N(a - b))) <= tolerance
            except (TypeError, ValueError):
                return False
    return False


@dataclass
class Equation:
    text: str
    lhs: sp.Expr
    rhs: sp.Expr
    var: sp.Symbol
    roots: Optional[list]
    degree: int


def real_roots(lhs, rhs, var) -> Optional[list]:
    """Real solutions of a polynomial equation of degree 1-4, or None if out of scope."""
    expr = sp.expand(lhs - rhs)
    try:
        degree = sp.Poly(expr, var).degree()
    except (sp.PolynomialError, sp.GeneratorsNeeded, ValueError):
        return None
    if degree < 1 or degree > 4:
        return None
    try:
        solutions = sp.solveset(expr, var, domain=sp.S.Reals)
    except (NotImplementedError, ValueError, TypeError):
        return None
    if isinstance(solutions, sp.FiniteSet):
        return list(solutions)
    if solutions == sp.S.EmptySet:
        return []
    return None


def _equation_from_chain(chain: Chain) -> Optional[Equation]:
    exprs = _parse_all([chain.parts[0], chain.parts[-1]])
    if exprs is None:
        return None
    lhs, rhs = exprs
    symbols = lhs.free_symbols | rhs.free_symbols
    if len(symbols) != 1:
        return None
    var = next(iter(symbols))
    roots = real_roots(lhs, rhs, var)
    try:
        degree = sp.Poly(sp.expand(lhs - rhs), var).degree()
    except (sp.PolynomialError, sp.GeneratorsNeeded, ValueError):
        degree = -1
    return Equation(f"{chain.parts[0]} = {chain.parts[-1]}", lhs, rhs, var, roots, degree)


def problem_equation(text: str):
    """Return (Equation | None, reason)."""
    candidates = []
    for chain in find_chains(text):
        if chain.alt_group is not None or len(chain.parts) != 2:
            continue
        equation = _equation_from_chain(chain)
        if equation is not None:
            candidates.append(equation)
    unique = []
    for eq in candidates:
        key = sp.srepr(sp.expand(eq.lhs - eq.rhs))
        if all(sp.srepr(sp.expand(u.lhs - u.rhs)) != key for u in unique):
            unique.append(eq)
    if not unique:
        return None, "no single-variable equation could be read from the question"
    if len(unique) > 1:
        return None, "the question contains several equations"
    if unique[0].roots is None:
        return None, "the equation is outside the supported scope (degree 1-4 polynomial)"
    return unique[0], ""


# ---------------------------------------------------------------------------
# claims (answers) parsing
# ---------------------------------------------------------------------------

@dataclass
class Claims:
    kind: str                      # "values" | "none_real"
    values: list = field(default_factory=list)


_NO_REAL = re.compile(
    r"\bno\s+(?:real\s+)?(?:solutions?|roots?)\b|\b(?:does\s+not|doesn't|do\s+not|don't)\s+have\s+(?:any\s+)?real\b"
    r"|لا يوجد", re.I)


def claims_from_chains(text: str, var) -> Optional[Claims]:
    """'x = 3 or x = -1' style claims about ``var`` (chains only, no fallbacks)."""
    values = []
    for chain in find_chains(text):
        if len(chain.parts) != 2:
            continue
        exprs = _parse_all(chain.parts)
        if exprs is None:
            continue
        left, right = exprs
        if left == var and is_numeric(right):
            values.append(right)
        elif right == var and is_numeric(left):
            values.append(left)
    return Claims("values", _dedupe(values)) if values else None


def parse_claims(text: str, var) -> Optional[Claims]:
    """Read an answer key as a set of values for ``var`` (or 'no real solutions')."""
    claims = claims_from_chains(text, var)
    if claims:
        return claims
    if _NO_REAL.search(text or ""):
        return Claims("none_real")
    cleaned = normalize_math(text or "").lower()
    cleaned = re.sub(r"\b(or|and)\b|أو|و", ",", cleaned)
    cleaned = re.sub(rf"\b(the|roots?|solutions?|answers?|are|is|values?|set|{re.escape(str(var).lower())})\b", " ", cleaned)
    cleaned = re.sub(r"[{}\[\]=]", " ", cleaned)
    pieces = [p.strip() for p in re.split(r"[,;]", cleaned) if p.strip()]
    values = []
    for piece in pieces:
        if "±" in piece:
            variants = [piece.replace("±", "+", 1), piece.replace("±", "-", 1)]
        else:
            variants = [piece]
        for variant in variants:
            expr = _parse(variant)
            if expr is None or not is_numeric(expr):
                return None
            values.append(expr)
    return Claims("values", _dedupe(values)) if values else None


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------

@dataclass
class Item:
    path: tuple
    label: str
    problem: str = ""
    answer: str = ""
    steps: list = field(default_factory=list)    # (path, text, is_check) in order
    extras: list = field(default_factory=list)   # (path, text)
    record: bool = True


def _strings(obj, path=()):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _strings(value, path + (key,))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _strings(value, path + (index,))


def collect_items(lesson: dict) -> list:
    items = []
    for i, q in enumerate(lesson.get("practice_questions", [])):
        base = ("practice_questions", i)
        steps = [(base + ("step_by_step_solution", k), s, _is_check(s))
                 for k, s in enumerate(q.get("step_by_step_solution", []))]
        items.append(Item(base, f"Practice question {i + 1}", q.get("question", ""),
                          q.get("correct_answer", ""), steps,
                          [(base + ("explanation",), q.get("explanation", ""))]))
    for i, ex in enumerate(lesson.get("worked_examples", [])):
        base = ("worked_examples", i)
        steps = []
        for k, step in enumerate(ex.get("solution_steps", [])):
            check = _is_check(step.get("work", "")) or _is_check(step.get("step", ""))
            for fld in ("work", "teacher_explanation", "reasoning"):
                if step.get(fld):
                    steps.append((base + ("solution_steps", k, fld), step[fld], check))
        items.append(Item(base, f"Worked example {i + 1}", ex.get("problem", ""),
                          ex.get("final_answer", ""), steps))
    for i, hw in enumerate(lesson.get("homework", [])):
        items.append(Item(("homework", i), f"Homework {i + 1}", hw.get("question", ""),
                          hw.get("answer_or_guidance", "")))
    for i, te in enumerate(lesson.get("teacher_learning", {}).get("teacher_examples", [])):
        base = ("teacher_learning", "teacher_examples", i)
        sol = te.get("solution", "")
        items.append(Item(base, f"Teacher example {i + 1}", te.get("problem_or_example", ""), "",
                          [(base + ("solution",), sol, False)]))
    for i, fa in enumerate(lesson.get("formative_assessment", [])):
        items.append(Item(("formative_assessment", i), f"Checkpoint {i + 1}",
                          fa.get("question_or_task", ""), fa.get("correct_response", "")))
    for i, concept in enumerate(lesson.get("concept_explanations", [])):
        check = concept.get("student_check")
        if check:
            items.append(Item(("concept_explanations", i, "student_check"),
                              f"Check for understanding ({concept.get('concept', i + 1)})",
                              check.get("question", ""), check.get("expected_answer", "")))
    summary = lesson.get("lesson_summary", {})
    if summary.get("final_check_question"):
        items.append(Item(("lesson_summary",), "Final check question",
                          summary.get("final_check_question", ""), summary.get("expected_answer", "")))
    return items


# ---------------------------------------------------------------------------
# the verifier
# ---------------------------------------------------------------------------

class MathVerifier:
    name = "mathematics"

    def verify(self, lesson: dict, request=None) -> VerificationReport:
        report = VerificationReport()
        arithmetic = self._arithmetic_sweep(lesson, report)
        for item in collect_items(lesson):
            self._verify_item(item, report, arithmetic)
        report.limitations.append(
            "Mathematics is checked by computation only where an equation, an arithmetic chain or a "
            "numeric answer can be read reliably from the text (single-variable polynomials up to "
            "degree 4, arithmetic, discriminants, identities). Other content is not machine-verified.")
        return report

    # -- C. arithmetic over every string in the lesson ------------------------
    def _arithmetic_sweep(self, lesson: dict, report: VerificationReport) -> dict:
        checked = {}
        for path, text in _strings(lesson):
            if len(text) > 4000 or "=" not in text or _ERROR_HUNT.search(text):
                continue
            for chain in find_chains(text):
                parsed = _parse_all(chain.parts)
                if parsed is None or not all(is_numeric(e) for e in parsed):
                    continue
                for k in range(len(parsed) - 1):
                    ok = numbers_match(chain.parts[k], chain.parts[k + 1], parsed[k], parsed[k + 1])
                    checked[path] = checked.get(path, 0) + 1
                    if not ok:
                        report.add(Issue(
                            CRITICAL, "math.arithmetic_error",
                            f"Arithmetic error: '{chain.parts[k]} = {chain.parts[k + 1]}' is false "
                            f"({fmt(sp.simplify(parsed[k]))} ≠ {fmt(sp.simplify(parsed[k + 1]))}).",
                            path, suggestion=f"{chain.parts[k]} = {fmt(sp.simplify(parsed[k]))}"))
        return checked

    def _arith_count(self, arithmetic: dict, prefix: tuple) -> int:
        return sum(n for path, n in arithmetic.items() if path[:len(prefix)] == prefix)

    # -- per item --------------------------------------------------------------
    def _verify_item(self, item: Item, report: VerificationReport, arithmetic: dict) -> None:
        record = ItemRecord(item.path, item.label)
        before = len(report.issues)
        answer_level = False

        eq, reason = problem_equation(item.problem) if item.problem else (None, "")
        if item.problem and eq is None:
            record.notes.append(f"No equation check: {reason}.")

        if eq is not None:
            if item.answer:
                answer_level |= self._check_solutions(item, eq, report, record)
            self._check_equivalence(item, eq, report, record)
            if eq.degree == 2:
                self._check_discriminant(item, eq, report, record)

        if item.problem and item.answer:
            answer_level |= self._check_roots_to_equation(item, report, record)
            answer_level |= self._check_evaluate(item, report, record)
            if eq is None:
                self._check_key_vs_steps(item, None, report, record)
        self._check_identities(item, report, record)

        arith = self._arith_count(arithmetic, item.path)
        if arith:
            record.passed.append(f"{arith} arithmetic statement(s) evaluated")
        item_issues = report.issues[before:]
        failed = any(i.severity == CRITICAL for i in item_issues) or any(
            i.severity == CRITICAL and i.path[:len(item.path)] == item.path for i in report.issues)
        if failed:
            record.status = FAILED
        elif answer_level:
            record.status = VERIFIED
        elif record.passed:
            record.status = PARTIAL
        else:
            record.status = UNVERIFIED
            if not record.notes:
                record.notes.append("Nothing machine-checkable was found; this item is AI-reviewed only.")
        if item.record:
            report.records.append(record)

    # -- A + B ----------------------------------------------------------------
    def _check_solutions(self, item, eq: Equation, report, record) -> bool:
        problem_domain_limited = bool(_RESTRICTED_DOMAIN.search(item.problem))
        if (_COMPLEX_WORDS.search(item.problem) or _COMPLEX_WORDS.search(item.answer)
                or _IMAGINARY_UNIT.search(item.answer)):
            record.notes.append("Complex or imaginary solutions are outside the real-number check.")
            return False
        claims = parse_claims(item.answer, eq.var)
        if claims is None:
            record.notes.append("The answer key could not be read in a machine-checkable form.")
            return False
        answer_path = item.path + (self._answer_field(item),)
        problems = []
        if claims.kind == "none_real":
            if eq.roots:
                problems.append(f"the answer says there are no real solutions but {eq.text} has "
                                f"solutions {fmt_set(eq.roots)}")
        else:
            for value in claims.values:
                lhs = sp.simplify(eq.lhs.subs(eq.var, value))
                rhs = sp.simplify(eq.rhs.subs(eq.var, value))
                if not same_value(lhs, rhs):
                    problems.append(f"{eq.var} = {fmt(value)} does not satisfy {eq.text} "
                                    f"(substituting gives {fmt(lhs)} on the left and {fmt(rhs)} on the right)")
            if not problem_domain_limited:
                missing = [r for r in eq.roots if not contains_value(claims.values, r)]
                if missing:
                    problems.append(f"the answer omits solution(s) {fmt_set(missing)} of {eq.text}")
        if problems:
            report.add(Issue(
                CRITICAL, "math.answer_mismatch",
                "The answer key contradicts the equation: " + "; ".join(problems) + ".",
                answer_path, suggestion=f"Solutions of {eq.text}: {fmt_set(eq.roots)}"))
            return False
        record.passed.append(f"answer verified by substitution: {eq.text} has solutions {fmt_set(eq.roots)}")
        self._check_key_vs_steps(item, eq, report, record, claims)
        return True

    @staticmethod
    def _answer_field(item: Item) -> str:
        return {"practice_questions": "correct_answer", "worked_examples": "final_answer",
                "homework": "answer_or_guidance", "formative_assessment": "correct_response",
                "concept_explanations": "expected_answer", "lesson_summary": "expected_answer"
                }.get(item.path[0], "")

    def _check_key_vs_steps(self, item, eq, report, record, claims: Optional[Claims] = None) -> None:
        var = eq.var if eq is not None else None
        if var is None:
            # Without an equation, infer the variable from the answer key.
            for chain in find_chains(item.answer):
                exprs = _parse_all(chain.parts)
                if exprs and len(chain.parts) == 2 and len(exprs[0].free_symbols) == 1 and not exprs[1].free_symbols:
                    var = next(iter(exprs[0].free_symbols))
                    break
            if var is None:
                return
        key = claims or claims_from_chains(item.answer, var)
        if key is None or key.kind != "values":
            return
        concluding = None
        for path, text, is_check in reversed(item.steps):
            if is_check:
                continue
            concluding = claims_from_chains(text, var)
            if concluding:
                concluding_path = path
                break
        if not concluding:
            return
        if sets_equal(key.values, concluding.values):
            record.passed.append("the written solution's conclusion agrees with the answer key")
            return
        restricted = bool(_RESTRICTED_DOMAIN.search(item.problem + " " + " ".join(t for _, t, _ in item.steps)))
        report.add(Issue(
            WARNING if restricted else CRITICAL, "math.key_contradicts_steps",
            f"The answer key says {fmt_set(key.values)} but the written solution concludes "
            f"{fmt_set(concluding.values)}.",
            concluding_path,
            suggestion="Make the answer key and the final solution step identical."))

    # -- E. equivalence ---------------------------------------------------------
    def _groups(self, text: str) -> list:
        chains = sorted(find_chains(text), key=lambda c: (c.start, c.alt_group or 0))
        groups, current = [], []
        for chain in chains:
            if current and glue_is_union(text, current[-1], chain):
                current.append(chain)
            else:
                if current:
                    groups.append(current)
                current = [chain]
        if current:
            groups.append(current)
        return groups

    def _group_info(self, group: list):
        """-> (var, roots, claim_only, label) or None when the group cannot be interpreted."""
        var, roots, claim_only, labels = None, [], True, []
        for chain in group:
            eq = _equation_from_chain(chain)
            if eq is None or eq.roots is None:
                return None
            if var is not None and eq.var != var:
                return None
            var = eq.var
            roots.extend(eq.roots)
            labels.append(eq.text)
            if not (eq.lhs == var and is_numeric(eq.rhs)) and not (eq.rhs == var and is_numeric(eq.lhs)):
                claim_only = False
        return var, _dedupe(roots), claim_only, " or ".join(labels)

    def _check_equivalence(self, item, seed: Equation, report, record) -> None:
        previous = (seed.var, seed.roots, f"the original equation {seed.text}")
        compared = 0
        for path, text, is_check in item.steps:
            if is_check or "=" not in text:
                continue
            for group in self._groups(text):
                info = self._group_info(group)
                if info is None:
                    continue
                var, roots, claim_only, label = info
                if var != previous[0]:
                    continue
                compared += 1
                if claim_only:
                    wrong = [r for r in roots if not contains_value(previous[1], r)]
                    if wrong:
                        report.add(Issue(
                            CRITICAL, "math.equation_not_equivalent",
                            f"'{label}' states {fmt_set(wrong)} as solution(s), but they do not satisfy "
                            f"{previous[2]} (whose solutions are {fmt_set(previous[1])}).", path,
                            suggestion=f"Solutions are {fmt_set(previous[1])}."))
                    continue
                if not sets_equal(roots, previous[1]):
                    report.add(Issue(
                        CRITICAL, "math.equation_not_equivalent",
                        f"This step is not equivalent to the previous one: {previous[2]} has solutions "
                        f"{fmt_set(previous[1])} but '{label}' has solutions {fmt_set(roots)}. "
                        "Check the sign or operation applied in this step.", path,
                        suggestion=f"An equivalent step must keep the solutions {fmt_set(previous[1])}."))
                previous = (var, roots, f"'{label}'")
        if compared:
            record.passed.append(f"{compared} solution step(s) checked for equivalence")

    # -- D. discriminant --------------------------------------------------------
    _GENERIC = re.compile(
        r"\b(?:a|an|any|every|each|if|when|whenever|whether)\b[^.]{0,25}(?:discriminant|Δ|∆)"
        r"|\b(?:quadratics?|equations?)\s+(?:with|that|which|whose)\b|\bwhenever\b", re.I)
    _CLAIMS = {
        "two": re.compile(r"two\s+(?:distinct|different|unequal)\s+(?:real\s+)?(?:roots?|solutions?)"
                          r"|two\s+real\s+and\s+distinct", re.I),
        "one": re.compile(r"(?:one|a\s+single)\s+(?:repeated|double|real|equal)?\s*(?:root|solution)"
                          r"|(?:repeated|double|equal|coincident)\s+(?:real\s+)?(?:roots?|solutions?)"
                          r"|exactly\s+one\s+(?:real\s+)?(?:root|solution)", re.I),
        "none": re.compile(r"no\s+real\s+(?:roots?|solutions?)|(?:does\s+not|doesn't|do\s+not|don't|has\s+no|have\s+no)"
                           r"\s+(?:have\s+)?(?:any\s+)?real\s+(?:roots?|solutions?)", re.I),
    }
    _SIGN = {
        "pos": re.compile(r"(?:discriminant|Δ|∆)\s*(?:is\s+|=\s*)?(?:positive|greater\s+than\s+(?:zero|0)|>\s*0)", re.I),
        "neg": re.compile(r"(?:discriminant|Δ|∆)\s*(?:is\s+|=\s*)?(?:negative|less\s+than\s+(?:zero|0)|<\s*0)", re.I),
        "zero": re.compile(r"(?:discriminant|Δ|∆)\s*(?:is\s+|=\s*)?(?:zero|equal\s+to\s+(?:zero|0))", re.I),
    }
    _TRUTH = {1: ("two", "pos", "two distinct real roots"), 0: ("one", "zero", "one repeated root"),
              -1: ("none", "neg", "no real roots")}

    def _check_discriminant(self, item, eq: Equation, report, record) -> None:
        a, b, c = sp.Poly(sp.expand(eq.lhs - eq.rhs), eq.var).all_coeffs()
        true_d = sp.simplify(b ** 2 - 4 * a * c)
        sign = int(sp.sign(true_d))
        want_count, want_sign, meaning = self._TRUTH[sign]
        texts = [(p, t) for p, t, _ in item.steps] + list(item.extras)
        if item.answer:
            texts.append((item.path + (self._answer_field(item),), item.answer))
        verified = 0
        for path, text in texts:
            for sentence in sentences(text):
                low = sentence.lower()
                if re.search(r"discriminant|Δ|∆", sentence, re.I):
                    s2 = re.sub(r"(?i)\b(?:the\s+)?discriminant\b", "D", sentence).replace("Δ", "D").replace("∆", "D")
                    s2 = re.sub(r"\bD\s+(?:is|equals|will be|becomes)\s+", "D = ", s2)
                    for chain in find_chains(s2):
                        parts = list(chain.parts)
                        if parts[0].strip() == "D":
                            parts = parts[1:]
                        exprs = _parse_all(parts)
                        if not exprs or not all(is_numeric(e) for e in exprs):
                            continue
                        claimed = sp.simplify(exprs[-1])
                        if same_value(claimed, true_d):
                            verified += 1
                        else:
                            report.add(Issue(
                                CRITICAL, "math.discriminant_wrong",
                                f"The discriminant is stated as {fmt(claimed)} but b² - 4ac = "
                                f"{fmt(true_d)} for {eq.text} (a = {fmt(a)}, b = {fmt(b)}, c = {fmt(c)}).",
                                path, suggestion=f"Δ = b² - 4ac = {fmt(true_d)}"))
                if self._GENERIC.search(low):
                    continue
                counts = {k for k, rx in self._CLAIMS.items() if rx.search(sentence)}
                signs = {k for k, rx in self._SIGN.items() if rx.search(sentence)}
                if not counts and not signs:
                    continue
                if len(counts) > 1 or len(signs) > 1:
                    continue  # a list of cases, not a conclusion about this equation
                wrong = [k for k in counts if k != want_count] + [k for k in signs if k != want_sign]
                if wrong:
                    report.add(Issue(
                        CRITICAL, "math.discriminant_conclusion",
                        f"The text draws a conclusion that contradicts the discriminant: for {eq.text} "
                        f"b² - 4ac = {fmt(true_d)}, which means {meaning}.", path,
                        suggestion=f"With discriminant {fmt(true_d)}: {meaning}."))
                else:
                    verified += 1
        if verified:
            record.passed.append(f"discriminant statements checked (b² - 4ac = {fmt(true_d)})")

    # -- F. identity chains -----------------------------------------------------
    def _check_identities(self, item, report, record) -> None:
        texts = [(p, t) for p, t, _ in item.steps] + list(item.extras)
        checked = 0
        for path, text in texts:
            if "=" not in text:
                continue
            for chain in find_chains(text):
                if len(chain.parts) < 3 or chain.alt_group is not None:
                    continue
                exprs = _parse_all(chain.parts)
                if exprs is None or all(is_numeric(e) for e in exprs):
                    continue
                if len(set().union(*[e.free_symbols for e in exprs])) != 1:
                    continue
                if not self._identical(exprs[0], exprs[1]):
                    continue  # an equation chain, not an identity chain
                checked += 1
                for k in range(1, len(exprs) - 1):
                    if self._identical(exprs[k], exprs[k + 1]):
                        continue
                    if k == len(exprs) - 2 and is_numeric(exprs[k + 1]):
                        continue  # final '= 0' turns the expression into an equation
                    report.add(Issue(
                        CRITICAL, "math.identity_error",
                        f"'{chain.parts[k]}' is not equal to '{chain.parts[k + 1]}' as an expression "
                        "(expansion or simplification error).", path,
                        suggestion=f"Expanded/simplified form: {fmt(sp.expand(exprs[k]))}"))
                    break
        if checked:
            record.passed.append(f"{checked} expansion/factorisation chain(s) checked")

    @staticmethod
    def _identical(a, b) -> bool:
        try:
            return sp.expand(a - b) == 0 or sp.simplify(a - b) == 0
        except (TypeError, ValueError):
            return False

    # -- G. roots -> equation ---------------------------------------------------
    _BUILD = re.compile(r"\b(?:form|find|write|construct|determine|obtain|build|create)\b", re.I)
    _ROOTS_WORD = re.compile(r"\b(?:roots?|solutions?|zeros?)\b", re.I)

    def _check_roots_to_equation(self, item, report, record) -> bool:
        problem = item.problem
        if not (self._BUILD.search(problem) and re.search(r"\bequation\b", problem, re.I)):
            return False
        word = self._ROOTS_WORD.search(problem)
        if not word:
            return False
        tail = problem[word.end():word.end() + 90]
        values = []
        claims_chain = [c for c in find_chains(tail) if len(c.parts) == 2]
        for chain in claims_chain:
            exprs = _parse_all(chain.parts)
            if exprs and exprs[0].free_symbols and is_numeric(exprs[1]):
                values.append(exprs[1])
        if len(values) < 2:
            values = []
            for run in find_expression_runs(tail):
                expr = _parse(run.parts[0])
                if expr is not None and is_numeric(expr):
                    values.append(expr)
        values = values[:2]
        if len(values) != 2 or same_value(values[0], values[1]):
            return False
        claimed_poly = None
        for chain in find_chains(item.answer):
            exprs = _parse_all([chain.parts[0], chain.parts[-1]])
            if exprs and len(exprs[0].free_symbols | exprs[1].free_symbols) == 1:
                claimed_poly = (chain.parts[0] + " = " + chain.parts[-1], exprs[0], exprs[1])
                break
        if claimed_poly is None:
            expr = _parse(item.answer.strip().rstrip("."))
            if expr is not None and len(expr.free_symbols) == 1:
                claimed_poly = (f"{item.answer.strip()} = 0", expr, sp.Integer(0))
        if claimed_poly is None:
            record.notes.append("The constructed equation in the answer could not be read.")
            return False
        text, lhs, rhs = claimed_poly
        var = next(iter((lhs.free_symbols | rhs.free_symbols)))
        roots = real_roots(lhs, rhs, var)
        if roots is None:
            return False
        answer_path = item.path + (self._answer_field(item),)
        if not sets_equal(roots, values):
            report.add(Issue(
                CRITICAL, "math.roots_do_not_match_equation",
                f"The equation '{text}' has roots {fmt_set(roots)}, not the requested {fmt_set(values)}.",
                answer_path,
                suggestion=f"An equation with roots {fmt_set(values)}: "
                           f"{fmt(sp.expand((var - values[0]) * (var - values[1])))} = 0"))
            return False
        record.passed.append(f"the constructed equation has exactly the requested roots {fmt_set(values)}")
        return True

    # -- H. evaluate -------------------------------------------------------------
    _EVAL = re.compile(r"\b(?:evaluate|calculate|compute|simplify|find the value of|what is)\b\s*:?\s*(.+)", re.I)

    def _check_evaluate(self, item, report, record) -> bool:
        match = self._EVAL.search(item.problem)
        if not match:
            return False
        runs = find_expression_runs(match.group(1))
        if len(runs) != 1:
            return False
        expected = _parse(runs[0].parts[0])
        if expected is None or not is_numeric(expected) or not re.search(r"[-+*/]|\*\*", runs[0].parts[0][1:]):
            return False
        candidates = []
        for chain in find_chains(item.answer):
            exprs = _parse_all([chain.parts[-1]])
            if exprs and is_numeric(exprs[0]):
                candidates.append((chain.parts[-1], exprs[0]))
        if not candidates:
            for run in find_expression_runs(item.answer):
                expr = _parse(run.parts[0])
                if expr is not None and is_numeric(expr):
                    candidates.append((run.parts[0], expr))
        if not candidates:
            return False
        text, claimed = candidates[0]
        value = sp.simplify(expected)
        if numbers_match(runs[0].parts[0], text, value, claimed):
            record.passed.append(f"{runs[0].parts[0]} = {fmt(value)} verified")
            return True
        report.add(Issue(
            CRITICAL, "math.evaluation_mismatch",
            f"The answer is given as {text} but {runs[0].parts[0]} = {fmt(value)}.",
            item.path + (self._answer_field(item),), suggestion=f"Correct value: {fmt(value)}"))
        return False
