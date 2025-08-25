from __future__ import annotations
from typing import Dict

def _basic_checks(list_field):
    if not isinstance(list_field, list) or len(list_field) == 0:
        return False
    for c in list_field:
        if not isinstance(c, str) or not c.strip():
            return False
    return True

def validate_and_split(output: Dict) -> bool:
    if not isinstance(output, dict):
        return False
    if output.get("operator") != "AND":
        return False
    conj = output.get("conjuncts")
    neg = output.get("negated")
    if not _basic_checks(conj):
        return False
    # If negated missing, auto-fill False
    if not isinstance(neg, list):
        output["negated"] = [False]*len(conj)
        neg = output.get("negated")
    # If length mismatch, attempt truncate/extend with False
    if len(neg) != len(conj):
        if len(neg) < len(conj):
            neg.extend([False]*(len(conj)-len(neg)))
        else:
            output["negated"] = neg[:len(conj)]
            neg = output["negated"]
    bad = {"i think", "as an ai", "cannot"}
    for c in conj:
        if any(b in c.lower() for b in bad):
            return False
    return True

def validate_or_split(output: Dict) -> bool:
    if not isinstance(output, dict):
        return False
    if output.get("operator") != "OR":
        return False
    dis = output.get("disjuncts")
    neg = output.get("negated")
    if not _basic_checks(dis):
        return False
    if not isinstance(neg, list):
        output["negated"] = [False]*len(dis)
        neg = output.get("negated")
    if len(neg) != len(dis):
        if len(neg) < len(dis):
            neg.extend([False]*(len(dis)-len(neg)))
        else:
            output["negated"] = neg[:len(dis)]
            neg = output["negated"]
    return True
