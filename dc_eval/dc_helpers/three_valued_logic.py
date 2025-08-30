"""Three-valued logic aggregation helpers for dcneurosymbolic mode."""
from __future__ import annotations
from typing import List


def combine_operator(op: str, child_vals: List[str], error_token: str = "Error") -> str:
    op = (op or "NONE").upper()
    if not child_vals:
        return error_token
    if op in ("NONE"):
        return child_vals[0]
    if op == "AND":
        if any(v == "False" for v in child_vals): return "False"
        if all(v == "True" for v in child_vals): return "True"
        return "Uncertain"
    if op == "OR":
        if any(v == "True" for v in child_vals): return "True"
        if all(v == "False" for v in child_vals): return "False"
        return "Uncertain"
    if op == "NOT":
        if child_vals[0] == "True": return "False"
        if child_vals[0] == "False": return "True"
        if child_vals[0] == "Uncertain": return "Uncertain"
    if op == "IMPLIES":
        if len(child_vals) != 2: return child_vals[0]
        a, b = child_vals[0], child_vals[1]
        if a == "True" and b == "False": return "False"
        if a == "True" and b == "True": return "True"
        if a == "False": return "True"
        if b == "True": return "True"
        return "Uncertain"
    if op == "IFF":
        if len(child_vals) != 2:
            return child_vals[0]
        a, b = child_vals[0], child_vals[1]
        if a == "Uncertain" or b == "Uncertain": return "Uncertain"
        return "True" if a == b else "False"
    if op == "XOR":
        if len(child_vals) < 2: return child_vals[0]
        if any(v == "Uncertain" for v in child_vals): return "Uncertain"
        trues = sum(v == "True" for v in child_vals)
        return "True" if trues == 1 else "False"
    if op == "NOR":
        if len(child_vals) < 2: return child_vals[0]
        if any(v == "True" for v in child_vals): return "False"
        if all(v == "False" for v in child_vals): return "True"
        return "Uncertain"
    return child_vals[0]
