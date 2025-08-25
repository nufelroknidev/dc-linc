from __future__ import annotations
import re
from typing import Dict, List

def _endify(c: str) -> str:
    c = c.strip()
    if not c.endswith('.'):
        c += '.'
    return c

def split_or_rule(sentence: str) -> Dict:
    """Deterministic OR splitter. Focus on simple top-level 'A or B' and comma lists.
    Returns empty disjuncts when exclusivity (XOR cues) appear."""
    if not sentence or not isinstance(sentence, str):
        return {"operator":"OR","disjuncts":[],"negated":[],"confidence":0.0,"rationale":"empty"}
    s = sentence.strip()
    low = s.lower()
    if any(k in low for k in ["but not both", "exactly one", "xor", "either one but not"]):
        return {"operator":"OR","disjuncts":[],"negated":[],"confidence":0.05,"rationale":"xor cue"}

    # Either A or B
    m = re.search(r'either\s+(.+?)\s+or\s+(.+)$', s, re.I)
    if m:
        a = m.group(1).strip().rstrip(' .')
        b = re.split(r'[.;]', m.group(2).strip())[0].rstrip(' .')
        return {"operator":"OR","disjuncts":[_endify(a), _endify(b)],"negated":[False, False],"confidence":0.8,"rationale":"either or"}

    # Comma list with final or: A, B, or C
    if ' or ' in low and ',' in s:
        # split last ' or '
        parts = re.split(r',\s*', s)
        last = parts[-1]
        if ' or ' in last.lower() and len(parts) <= 6:  # limit fan-out
            pre = parts[:-1]
            left, right = re.split(r'\s+or\s+', last, maxsplit=1, flags=re.I)
            items = pre + [left.strip(), right.strip()]
            items = [p.strip().rstrip(' .') for p in items if p.strip()]
            if len(items) >= 2:
                return {"operator":"OR","disjuncts":[_endify(i) for i in items],"negated":[False]*len(items),"confidence":0.75,"rationale":"comma or list"}

    # Simple binary A or B (take last top-level or)
    if ' or ' in low:
        idx = low.rfind(' or ')
        left = s[:idx].strip().rstrip(' .')
        right = s[idx+4:].strip().rstrip(' .')
        if left and right and len(left.split()) > 1:
            return {"operator":"OR","disjuncts":[_endify(left), _endify(right)],"negated":[False, False],"confidence":0.7,"rationale":"or"}

    return {"operator":"OR","disjuncts":[],"negated":[],"confidence":0.1,"rationale":"no rule split"}
