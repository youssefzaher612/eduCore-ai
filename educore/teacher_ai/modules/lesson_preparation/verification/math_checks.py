"""
Compatibility shim (NEW file; no existing file was modified).

chemistry_verifier.py does ``from .math_checks import _strings`` but the math module that
defines ``_strings`` is ``math_verifier.py``. This re-export makes the existing import
resolve. Listed as an integration issue for the final review: the cleaner fix is to change
that one import line in chemistry_verifier.py (or share ``_strings`` from a common module),
after which this file can be deleted.
"""

from .math_verifier import _strings  # noqa: F401

__all__ = ["_strings"]
