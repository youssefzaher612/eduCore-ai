"""
Deterministic chemistry check: atom and charge balance of written equations.

An equation such as '2H₂ + O₂ → 2H₂O' is parsed strictly (valid element symbols only,
optional state symbols, optional ionic charges written with superscripts). If any
species cannot be parsed the equation is SKIPPED, never guessed, so prose and
non-chemical arrows cannot cause false accusations.
"""

import re
from collections import Counter
from fractions import Fraction
from typing import Optional

from .models import CRITICAL, INFO, Issue, VerificationReport

ELEMENTS = set("""
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb
Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
""".split())

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_ARROW = re.compile(r"\s*(?:→|⟶|⇌|⇄|<->|->|=>)\s*")
_STATE = re.compile(r"\((?:s|l|g|aq)\)")
_CHARGE = re.compile(r"([⁰¹²³⁴⁵⁶⁷⁸⁹]*)([⁺⁻])$")
_TOKEN = re.compile(r"[A-Z][a-z]?|\d+|\(|\)")
_BOUNDARY = re.compile(r"[:;.!?\n]")


def parse_formula(text: str) -> Optional[Counter]:
    """Atom counts of a formula such as 'Ca(OH)2', or None when it is not a valid formula."""
    if not text or re.search(r"[^A-Za-z0-9()]", text):
        return None
    tokens = _TOKEN.findall(text)
    if "".join(tokens) != text:
        return None
    stack = [Counter()]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "(":
            stack.append(Counter())
            i += 1
        elif tok == ")":
            if len(stack) == 1:
                return None
            group = stack.pop()
            i += 1
            mult = 1
            if i < len(tokens) and tokens[i].isdigit():
                mult = int(tokens[i])
                i += 1
            for element, n in group.items():
                stack[-1][element] += n * mult
        elif tok.isdigit():
            return None
        else:
            if tok not in ELEMENTS:
                return None
            i += 1
            count = 1
            if i < len(tokens) and tokens[i].isdigit():
                count = int(tokens[i])
                i += 1
            stack[-1][tok] += count
    return stack[0] if len(stack) == 1 and stack[0] else None


def parse_species(raw: str):
    """-> (coefficient, atoms Counter, charge) or None."""
    text = _STATE.sub("", raw.translate(_SUBSCRIPTS)).strip().strip(".,")
    coefficient = Fraction(1)
    match = re.match(r"^(\d+(?:/\d+)?)\s*", text)
    if match and len(text) > match.end():
        try:
            coefficient = Fraction(match.group(1))
        except (ValueError, ZeroDivisionError):
            return None
        text = text[match.end():]
    charge = 0
    cm = _CHARGE.search(text)
    if cm:
        magnitude = int(cm.group(1).translate(_SUPERSCRIPTS)) if cm.group(1) else 1
        charge = magnitude if cm.group(2) == "⁺" else -magnitude
        text = text[:cm.start()]
    atoms = parse_formula(text.strip())
    return None if atoms is None else (coefficient, atoms, charge)


def _side(text: str, is_left: bool):
    pieces = re.split(r"\s+\+\s+|\s*\+\s+|\s+\+\s*", text.strip())
    if is_left:
        pieces = pieces[-6:] if len(pieces) > 6 else pieces
    species = []
    for index, piece in enumerate(pieces):
        parsed = parse_species(piece.translate(_SUBSCRIPTS))
        if parsed is None:
            edge = (is_left and index == 0) or (not is_left and index == len(pieces) - 1)
            words = piece.split()
            if not edge or not words:
                return None
            parsed = parse_species((words[-1] if is_left else words[0]).translate(_SUBSCRIPTS))
            if parsed is None:
                return None
        species.append(parsed)
    return species or None


def _totals(species):
    atoms, charge = Counter(), Fraction(0)
    for coefficient, counts, ion in species:
        for element, n in counts.items():
            atoms[element] += coefficient * n
        charge += coefficient * ion
    return atoms, charge


def _show(atoms: Counter, charge: Fraction) -> str:
    body = ", ".join(f"{el}:{int(n) if n == int(n) else float(n)}" for el, n in sorted(atoms.items()))
    return body + (f"; charge {charge}" if charge else "")


def find_equations(text: str) -> list:
    """Parsable chemical equations in ``text`` as (original_text, left_species, right_species)."""
    found = []
    pieces = _ARROW.split(text)
    spans = list(_ARROW.finditer(text))
    for k, arrow in enumerate(spans):
        left_raw = pieces[k]
        right_raw = pieces[k + 1]
        left_raw = _BOUNDARY.split(left_raw)[-1] if not re.search(r"[A-Z][a-z]?\d*\s*\+", left_raw[-1:]) else left_raw
        left_raw = _BOUNDARY.split(left_raw)[-1]
        right_raw = _BOUNDARY.split(right_raw)[0]
        right_raw = re.split(r",\s", right_raw)[0]
        left, right = _side(left_raw, True), _side(right_raw, False)
        if left and right:
            found.append((f"{left_raw.strip()} → {right_raw.strip()}", left, right))
    return found


class ChemistryVerifier:
    name = "chemistry"

    def applies_to(self, request) -> bool:
        if request is None:
            return True
        from ..prompts import detect_subject_domains
        return bool({"chemistry", "integrated_science"} & set(detect_subject_domains(request)))

    def verify(self, lesson: dict, request=None) -> VerificationReport:
        from .math_checks import _strings
        report = VerificationReport()
        balanced = 0
        for path, text in _strings(lesson):
            if not any(a in text for a in ("→", "->", "⇌", "⇄", "=>", "⟶")):
                continue
            for original, left, right in find_equations(text):
                la, lc = _totals(left)
                ra, rc = _totals(right)
                if la == ra and lc == rc:
                    balanced += 1
                else:
                    report.add(Issue(
                        CRITICAL, "chem.unbalanced",
                        f"The chemical equation '{original}' is not balanced: left side has "
                        f"[{_show(la, lc)}] but right side has [{_show(ra, rc)}].", path,
                        suggestion="Adjust the coefficients so every element (and the charge) matches."))
        if balanced:
            report.add(Issue(INFO, "chem.balanced",
                             f"{balanced} chemical equation(s) were checked for atom and charge balance "
                             "and are balanced.", (), suggestion=None))
        report.limitations.append(
            "Chemistry: only the atom/charge balance of written equations is checked by computation. "
            "Reaction feasibility, conditions, observations and safety are not machine-verified.")
        return report