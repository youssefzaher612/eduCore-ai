"""
Safe parsing of mathematical text into SymPy objects.

AI-generated text is NEVER passed to eval(), exec() or sympy.sympify()/parse_expr().
Instead the text is normalised, tokenised against a strict whitelist and parsed with
a small recursive-descent parser that builds SymPy objects directly.

Allowed: numbers, single-letter variables, the constant pi, + - * / ** and
parentheses, implicit multiplication (2x, 3(x+1), (a)(b)), and the functions
sqrt() and abs(). Anything else raises MathParseError. Input size, number length
and exponent size are bounded so a hostile string cannot exhaust time or memory.
"""

import re

import sympy as sp

MAX_CHARS = 300
MAX_TOKENS = 160
MAX_DIGITS = 15
MAX_EXPONENT = 12

FUNCTIONS = {"sqrt": sp.sqrt, "abs": sp.Abs}
CONSTANTS = {"pi": sp.pi}


class MathParseError(ValueError):
    """The text is not a supported mathematical expression."""


_SUPERSCRIPT_TO_ASCII = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPERSCRIPT_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+")
_DIGIT_TRANSLATION = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_UNICODE_FRACTIONS = {"½": "(1/2)", "¼": "(1/4)", "¾": "(3/4)", "⅓": "(1/3)", "⅔": "(2/3)",
                      "⅕": "(1/5)", "⅛": "(1/8)"}
_SIMPLE_REPLACEMENTS = (
    ("−", "-"), ("–", "-"), ("—", "-"), ("‒", "-"),
    ("×", "*"), ("·", "*"), ("∙", "*"), ("⋅", "*"),
    ("÷", "/"), ("∕", "/"), ("⁄", "/"),
    ("π", "pi"), ("^", "**"),
)


def normalize_math(text) -> str:
    """Convert Unicode mathematical notation into the ASCII form the parser reads."""
    s = str(text).translate(_DIGIT_TRANSLATION).replace("،", ",")

    def superscript(match):
        digits = match.group(0).translate(_SUPERSCRIPT_TO_ASCII)
        return f"**({digits})" if digits[0] in "+-" else f"**{digits}"

    s = _SUPERSCRIPT_RUN.sub(superscript, s)
    for old, new in _SIMPLE_REPLACEMENTS:
        s = s.replace(old, new)
    for old, new in _UNICODE_FRACTIONS.items():
        s = s.replace(old, new)
    s = re.sub(r"√\s*\(", "sqrt(", s)
    s = re.sub(r"√\s*(\d+(?:\.\d+)?|[A-Za-z])", r"sqrt(\1)", s)
    return s


_TOKEN = re.compile(r"\s*(?:(?P<num>\d+(?:\.\d+)?)|(?P<name>[A-Za-z]+)|(?P<op>\*\*|[-+*/()=]))")


def _tokenize(text: str) -> list:
    s = normalize_math(text).strip()
    if not s:
        raise MathParseError("empty expression")
    if len(s) > MAX_CHARS:
        raise MathParseError("expression too long")
    tokens, pos = [], 0
    while pos < len(s):
        match = _TOKEN.match(s, pos)
        if not match or match.end() == pos:
            raise MathParseError(f"unsupported character {s[pos:pos + 1]!r}")
        pos = match.end()
        if match.group("num") is not None:
            digits = match.group("num").replace(".", "")
            if len(digits) > MAX_DIGITS:
                raise MathParseError("number too long")
            tokens.append(("num", match.group("num")))
        elif match.group("name") is not None:
            name = match.group("name")
            if name not in FUNCTIONS and name not in CONSTANTS and len(name) != 1:
                raise MathParseError(f"unsupported name {name!r}")
            tokens.append(("name", name))
        else:
            tokens.append(("op", match.group("op")))
        if len(tokens) > MAX_TOKENS:
            raise MathParseError("expression too complex")
    return tokens


def _value_end(token) -> bool:
    kind, value = token
    return kind == "num" or (kind == "name" and value not in FUNCTIONS) or (kind == "op" and value == ")")


def _value_start(token) -> bool:
    kind, value = token
    return kind in ("num", "name") or (kind == "op" and value == "(")


def _with_implicit_multiplication(tokens: list) -> list:
    out = []
    for token in tokens:
        if out and _value_end(out[-1]) and _value_start(token):
            prev = out[-1]
            if prev[0] == "num" and token[0] == "num":
                raise MathParseError("two numbers in a row")
            if prev[0] == "name" and token[0] == "num":
                raise MathParseError("a number cannot directly follow a variable")
            out.append(("op", "*"))
        out.append(token)
    return out


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = 0

    def _peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _take(self):
        token = self._peek()
        self.index += 1
        return token

    def _is_op(self, *values) -> bool:
        token = self._peek()
        return token is not None and token[0] == "op" and token[1] in values

    def parse(self):
        result = self._expression()
        if self._peek() is not None:
            raise MathParseError(f"unexpected {self._peek()[1]!r}")
        if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise MathParseError("undefined value (division by zero)")
        return result

    def _expression(self):
        value = self._term()
        while self._is_op("+", "-"):
            op = self._take()[1]
            right = self._term()
            value = value + right if op == "+" else value - right
        return value

    def _term(self):
        value = self._unary()
        while self._is_op("*", "/"):
            op = self._take()[1]
            right = self._unary()
            if op == "/":
                if right == 0:
                    raise MathParseError("division by zero")
                value = value / right
            else:
                value = value * right
        return value

    def _unary(self):
        if self._is_op("+", "-"):
            op = self._take()[1]
            operand = self._unary()
            return operand if op == "+" else -operand
        return self._power()

    def _power(self):
        base = self._atom()
        if self._is_op("**"):
            self._take()
            exponent = self._unary()
            if not exponent.is_Rational or abs(exponent) > MAX_EXPONENT:
                raise MathParseError("unsupported exponent")
            if base == 0 and exponent < 0:
                raise MathParseError("division by zero")
            return base ** exponent
        return base

    def _atom(self):
        token = self._take()
        if token is None:
            raise MathParseError("unexpected end of expression")
        kind, value = token
        if kind == "num":
            return sp.Rational(value)
        if kind == "name":
            if value in FUNCTIONS:
                if not self._is_op("("):
                    raise MathParseError(f"{value} needs parentheses")
                self._take()
                inner = self._expression()
                if not self._is_op(")"):
                    raise MathParseError("missing )")
                self._take()
                return FUNCTIONS[value](inner)
            if value in CONSTANTS:
                return CONSTANTS[value]
            return sp.Symbol(value, real=True)
        if value == "(":
            inner = self._expression()
            if not self._is_op(")"):
                raise MathParseError("missing )")
            self._take()
            return inner
        raise MathParseError(f"unexpected {value!r}")


def _parse_tokens(tokens: list):
    if not tokens:
        raise MathParseError("empty expression")
    return _Parser(_with_implicit_multiplication(tokens)).parse()


def parse_expression(text):
    """Parse text such as '(-1 + sqrt(13))/2' or 'x² - 2x - 3' into a SymPy expression."""
    tokens = _tokenize(text)
    if any(kind == "op" and value == "=" for kind, value in tokens):
        raise MathParseError("an expression cannot contain '='")
    return _parse_tokens(tokens)


def parse_equation(text):
    """Parse 'lhs = rhs' (exactly one '=') and return (lhs, rhs)."""
    tokens = _tokenize(text)
    positions = [i for i, (kind, value) in enumerate(tokens) if kind == "op" and value == "="]
    if len(positions) != 1:
        raise MathParseError("an equation needs exactly one '='")
    split = positions[0]
    return _parse_tokens(tokens[:split]), _parse_tokens(tokens[split + 1:])


def is_numeric(expr) -> bool:
    return not expr.free_symbols
