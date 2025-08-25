from __future__ import annotations
import re
from typing import Set

_CANONICAL: Set[str] = {"IFF","IMPLIES","XOR","OR","AND","NOT","NONE"}

def _norm_operator(op_raw: str | None) -> str:
    if not op_raw:
        return ""
    s = str(op_raw).strip().upper()
    if s in _CANONICAL:
        return s
    syn = {
        "BICOND": "IFF",
        "BICONDITIONAL": "IFF",
        "BICOND.": "IFF",
        "EQUIVALENCE": "IFF",
        "ENTAILS": "IMPLIES",
        "IMPLY": "IMPLIES",
        "EXCLUSIVE OR": "XOR",
        "EXCL OR": "XOR",
        "X-OR": "XOR",

        "NEGATION": "NOT",
        "UNKNOWN": "NONE",
        "NULL": "NONE",
    }
    if s in syn:
        return syn[s]
    if "EXCLUSIVE" in s and "OR" in s:
        return "XOR"
    if "IF AND ONLY IF" in s or s == "IFF.":
        return "IFF"
    return s if s in _CANONICAL else ""

def _consistency_adjust(sentence: str, op: str) -> str:
    s = (sentence or "").lower()
    if op == "IFF":
        if not ("if and only if" in s or re.search(r"\biff\b", s)):
            if (re.search(r"\bif\b.+\bthen\b", s) or " only if " in s or s.strip().startswith("unless")):
                return "IMPLIES"
    if op == "XOR":
        if not ("exclusive or" in s or "exactly one" in s or "but not both" in s or re.search(r"\beither\b.+\bor\b", s)):
            if " or " in s:
                return "OR"
    return op

def _obvious_cue_operator(sentence: str) -> str:
    s = (sentence or "").lower().strip()
    if not s:
        return ""
    if "if and only if" in s or re.search(r"\biff\b", s):
        return "IFF"
    if re.search(r"\bif\b.+\bthen\b", s) or " only if " in s or s.startswith("unless"):
        return "IMPLIES"
    if re.search(r"\bneither\b.+\bnor\b", s):
        return "AND"
    if re.search(r"\bboth\b.+\band\b", s):
        return "AND"
    if "either" in s and " or " in s:
        return "XOR"
    if re.search(r"\bno\b|\bnone\b|\bnever\b|\bnot\b|n't\b|\bwithout\b", s):
        return "NOT"
    if " or " in s:
        return "OR"
    if " and " in s:
        return "AND"
    return ""

__all__ = [
    "_CANONICAL",
    "_norm_operator",
    "_consistency_adjust",
    "_obvious_cue_operator",
]
