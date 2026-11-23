"""
Find mathematical equations inside free-form lesson text.

The extractor is deliberately CONSERVATIVE. A wrong extraction would create a false
accusation against correct content, so anything ambiguous is skipped and later
reported as "not machine-checkable" rather than "wrong". Runs of mathematical
tokens are discarded when they

* touch a word with no space (``4ac``, ``3rd``),
* start or end with a dangling operator,
* follow words such as "of", "times", "plus" (``20% of 50 = 10``),
* are followed by %, a degree sign, or a space-separated operand (units, prose),
* contain unbalanced parentheses or more than one ±.

What survives is split at '=' into a chain of parts, e.g. ``x² - 2x - 3 = 0`` gives
parts ['x**2 - 2*x - 3', '0'].
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .safe_math import CONSTANTS, FUNCTIONS, normalize_math

_MATH_KINDS = {"num", "var", "name", "op", "lp", "rp", "eq", "pm"}
_OPERATOR_WORDS = {"of", "times", "plus", "minus", "by", "from", "than", "over",
                   "squared", "cubed", "to", "per"}


@dataclass
class Tok:
    kind: str
    text: str
    start: int
    end: int
    space_before: bool


@dataclass
class Chain:
    parts: list
    start: int = 0
    end: int = 0
    alt_group: Optional[int] = None


def _tokenize_text(s: str) -> list:
    tokens, pos, n, spaced = [], 0, len(s), False
    while pos < n:
        ch = s[pos]
        if ch.isspace():
            pos += 1
            spaced = True
            continue
        number = re.match(r"\d+(?:\.\d+)?", s[pos:])
        letters = re.match(r"[^\W\d_]+", s[pos:])
        if number:
            kind, text = "num", number.group(0)
        elif letters:
            text = letters.group(0)
            if text in FUNCTIONS or text in CONSTANTS:
                kind = "name"
            elif len(text) == 1 and text.isascii():
                kind = "var"
            else:
                kind = "word"
        elif s.startswith("**", pos):
            kind, text = "op", "**"
        elif ch in "+-*/":
            kind, text = "op", ch
        elif ch == "(":
            kind, text = "lp", ch
        elif ch == ")":
            kind, text = "rp", ch
        elif ch == "=":
            kind, text = "eq", ch
        elif ch == "±":
            kind, text = "pm", ch
        else:
            kind, text = "punct", ch
        tokens.append(Tok(kind, text, pos, pos + len(text), spaced))
        pos += len(text)
        spaced = False
    return tokens


def _operand_end(t: Tok) -> bool:
    return t.kind in ("num", "var", "rp") or (t.kind == "name" and t.text in CONSTANTS)


def _operand_start(t: Tok) -> bool:
    return t.kind in ("num", "var", "lp", "name")


_MARKER_SHAPES = (("num", "rp"), ("var", "rp"), ("lp", "var", "rp"), ("lp", "num", "rp"))


def _is_list_marker(tokens: list, i0: int, i1: int) -> bool:
    return tuple(t.kind for t in tokens[i0:i1 + 1]) in _MARKER_SHAPES


def _runs(tokens: list) -> list:
    runs, start = [], None
    for index, token in enumerate(tokens):
        if token.kind in _MATH_KINDS:
            if start is None:
                start = index
            elif (_operand_end(tokens[index - 1]) and _operand_start(token) and token.space_before):
                runs.append((start, index - 1))
                start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(tokens) - 1))
    return runs


def _clean_run(tokens: list, i: int, j: int, prev_run: Optional[tuple]):
    """Return trimmed (i, j) or None when the run is not safe to interpret."""
    trimmed_left = trimmed_right = False
    while i <= j and tokens[i].kind == "eq":
        i += 1
        trimmed_left = True
    while j >= i and tokens[j].kind == "eq":
        j -= 1
        trimmed_right = True
    if i > j:
        return None

    prev = tokens[i - 1] if i > 0 else None
    nxt = tokens[j + 1] if j + 1 < len(tokens) else None
    first, last = tokens[i], tokens[j]

    anchored_left = (trimmed_left or prev is None or prev.kind == "punct"
                     or (prev_run is not None and _is_list_marker(tokens, *prev_run)))
    if first.kind == "op" and first.text in ("*", "/", "**"):
        return None
    if not anchored_left:
        if first.kind == "op":
            second = tokens[i + 1] if i + 1 <= j else None
            if first.text != "-" or not (first.space_before and second is not None
                                         and not second.space_before):
                return None
        if prev.kind == "word":
            if not first.space_before or prev.text.lower() in _OPERATOR_WORDS:
                return None
        else:
            return None

    if last.kind in ("op", "pm"):
        return None
    anchored_right = trimmed_right or nxt is None or (nxt.kind == "punct" and nxt.text not in "%°")
    if not anchored_right:
        if nxt.kind == "word" and nxt.space_before:
            pass
        else:
            return None

    depth = 0
    for token in tokens[i:j + 1]:
        if token.kind == "lp":
            depth += 1
        elif token.kind == "rp":
            depth -= 1
            if depth < 0:
                return None
    if depth != 0 or sum(1 for t in tokens[i:j + 1] if t.kind == "pm") > 1:
        return None
    return i, j


def _split_parts(s: str, tokens: list, i: int, j: int):
    parts, begin = [], i
    for k in range(i, j + 2):
        if k == j + 1 or tokens[k].kind == "eq":
            if k - 1 < begin:
                return None  # empty part, e.g. '= =' or leading '='
            parts.append(s[tokens[begin].start:tokens[k - 1].end])
            begin = k + 1
    return parts


@dataclass
class Run:
    parts: list
    start: int
    end: int


def find_runs(text: str) -> list:
    """All clean runs of mathematical tokens (with or without '=')."""
    s = normalize_math(text)
    tokens = _tokenize_text(s)
    out, prev = [], None
    for i0, i1 in _runs(tokens):
        cleaned = _clean_run(tokens, i0, i1, prev)
        prev = (i0, i1)
        if cleaned is None:
            continue
        i, j = cleaned
        parts = _split_parts(s, tokens, i, j)
        if parts:
            out.append(Run(parts, tokens[i].start, tokens[j].end))
    return out


def find_chains(text: str) -> list:
    """Equation chains (>= 2 parts). A single ± yields two alternative chains sharing alt_group."""
    chains, group = [], 0
    for run in find_runs(text):
        if len(run.parts) < 2:
            continue
        joined = " = ".join(run.parts)
        if "±" not in joined:
            chains.append(Chain(run.parts, run.start, run.end))
            continue
        if joined.count("±") != 1:
            continue
        group += 1
        for sign in ("+", "-"):
            parts = [p.replace("±", sign) for p in run.parts]
            chains.append(Chain(parts, run.start, run.end, alt_group=group))
    return chains


def find_expression_runs(text: str) -> list:
    """Runs without '=' (single expressions), e.g. the numbers in 'roots 2 and -3'."""
    return [run for run in find_runs(text) if len(run.parts) == 1]


def sentences(text: str) -> list:
    """Split into sentence-like segments without breaking decimals such as 4.5."""
    pieces = re.split(r"(?<=[.;!?])\s+|\n+", str(text))
    return [p.strip() for p in pieces if p and p.strip()]


_UNION_GLUE = re.compile(r"^\s*(?:or|and|,|;|،|أو|و)?\s*$", re.IGNORECASE)


def glue_is_union(text: str, left: Chain, right: Chain) -> bool:
    """True when two chains are joined by 'or'/'and'/a comma (alternatives in one statement)."""
    if left.alt_group is not None and left.alt_group == right.alt_group:
        return True
    between = normalize_math(text)[left.end:right.start]
    return bool(_UNION_GLUE.match(between))
