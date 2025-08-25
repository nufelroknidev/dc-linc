from __future__ import annotations
import re
from typing import Tuple, List, Dict, Any

# ---------------------------------------------------------------------------
# Quantifier heuristics (used to decide if a sentence should remain atomic)
# ---------------------------------------------------------------------------
_QUANTIFIER_PATTERNS = [
    r'for all', r'for every', r'for each', r'for any',
    r'there exists', r'there exist', r'there is', r'there are',
    r'there does not exist', r'there do not exist',
    r'no ', r'none', r'nobody', r'no one', r'at least one', r'at most', r'exactly one'
]
_Q_REGEX = re.compile("|".join(_QUANTIFIER_PATTERNS), re.I)

def _part_has_quantifier(text: str) -> bool:
    return bool(_Q_REGEX.search(text or ""))

def _has_leading_quantifier(sentence: str) -> bool:
    return bool(re.match(
        r'^\s*(for all|for every|for each|for any|there exists|there exist|there is|there are|there does not exist|there do not exist|no\b|none\b|nobody\b|no one\b|at least one|at most|exactly one)\b',
        (sentence or "").strip(), re.I))

# Heuristic detectors returning (operator, parts)

def detect_either_or_neither_nor(s: str) -> Tuple[str, list] | None:
    """Detect pattern: either X (and Y) or neither X nor Y -> top-level OR of two composite clauses.

    Example: "Ben is either yellow and ugly or neither yellow nor ugly." should yield OR.
    We only label OR (not IFF) to stay within existing operator set.
    """
    if not isinstance(s, str):
        return None
    low = s.lower()
    # Do NOT treat as top-level OR if the whole sentence is a conditional starting with 'if';
    # in that case we want an IMPLIES with a disjunctive antecedent handled elsewhere.
    if low.lstrip().startswith('if '):
        return None
    if 'either' in low and ' or neither ' in low and ' nor ' in low:
        # Rough split: after 'either' up to ' or neither ' is left; after that is right starting at 'neither'
        try:
            m = re.search(r'either\s+(.*?)\s+or\s+neither\s+(.*)', s, re.IGNORECASE)
            if m:
                left_raw = m.group(1).strip().rstrip(' .')
                right_tail = m.group(2).strip().rstrip(' .')  # starts after 'neither'
                right_raw = 'neither ' + right_tail
                if left_raw and right_tail:
                    return ("OR", [left_raw, right_raw])
        except Exception:
            return None
    return None


def detect_either_neither_xor(s: str) -> Tuple[str, list] | None:
    """Detect pattern: either (A and B) or neither A nor B  -> treat as XOR.

    Although logically (A ∧ B) ∨ (¬A ∧ ¬B) is an equivalence (XNOR), downstream
    splitting often expects an exclusive-style label to force later refinement.
    We therefore map to XOR for pragmatic branching.
    """
    if not isinstance(s, str):
        return None
    low = s.lower()
    if low.lstrip().startswith('if '):  # do not interfere with implication
        return None
    # Extract inner either ... or neither ... nor ... region
    m = re.search(r'either\s+(.+?)\s+or\s+neither\s+(.+?)\s+nor\s+(.+?)(?:[\.,;]|$)', s, re.I)
    if not m:
        return None
    left = m.group(1).strip().rstrip(' .,' )
    n1 = m.group(2).strip().rstrip(' .,' )
    n2 = m.group(3).strip().rstrip(' .,' )
    # Heuristic: left should contain an ' and ' joining two spans that correspond to n1 and n2 tokens order.
    if ' and ' not in left.lower():
        return None
    # Get the two conjuncts (take last top-level ' and ' to reduce noise similar to detect_and)
    l_low = left.lower()
    idx = l_low.rfind(' and ')
    left_a = left[:idx].strip()
    left_b = left[idx+5:].strip()
    # Compare against n1/n2 (case-insensitive substring containment)
    def approx_match(a: str, b: str) -> bool:
        return a.lower() in b.lower() or b.lower() in a.lower()
    if approx_match(left_a, n1) and approx_match(left_b, n2):
        part1 = f"{left_a} and {left_b}"
        part2 = f"neither {n1} nor {n2}"
        return ("XOR", [part1, part2])
    return None

def detect_either_or_conditional(s: str) -> Tuple[str, list] | None:
    """Detect pattern: either A or, if C, then D  -> top-level OR.

    We collapse the second branch to its consequent (D) for surface form,
    dropping the conditional guard, since downstream splitting will handle
    implications separately if needed. This avoids misclassifying as IMPLIES.
    """
    if not isinstance(s, str):
        return None
    low = s.lower()
    if 'either' not in low or ' or' not in low or ' if ' not in low or ' then ' not in low:
        return None
    # Require 'either' before first ' or'
    m = re.search(r'either\s+(.+?)\s+or[, ]+\s*if\s+(.+?)\s+then\s+(.+?)\.?\s*$', s, re.I)
    if not m:
        return None
    left = m.group(1).strip().rstrip(' .,' )
    # condition = m.group(2).strip()  # currently unused
    consequent = m.group(3).strip().rstrip(' .,' )
    if left and consequent:
        return ("OR", [left, consequent])
    return None

def detect_meta_not(s: str) -> Tuple[str, list] | None:
    """Detect meta-negation wrappers like:

    - It is not true that S
    - It's not true that S

    Returns a NOT operator over the inner clause S (kept verbatim), allowing
    downstream components to treat this as a standard negation rather than
    suppressing structure inside.
    """
    if not isinstance(s, str):
        return None
    m = re.match(r"^\s*(?:it\s+is|it's)\s+not\s+true\s+that\s+(.+)$", s, re.I)
    if m:
        inner = m.group(1).strip().rstrip(' .')
        if inner:
            return ("NOT", [inner])
    return None


def detect_leading_no_negation(s: str) -> Tuple[str, list] | None:
    """Deprecated (leading 'No'). 'No' no longer triggers top-level NOT."""
    return None

def detect_implies(s: str) -> Tuple[str, list] | None:
    """Detect a top-level IMPLIES pattern.

    Supports:
      - If A, then B.
      - If A, B. (no 'then')
      - B if A.
      - A only if B.  (re-ordered)
      - Unless A, B. / B unless A. (antecedent negated via textual 'not (...)')
    The function is intentionally generous so it should run BEFORE other
    disjunction / conjunction detectors.
    """
    if not isinstance(s, str):
        return None
    text = s.strip()
    low = text.lower()
    # Fast reject: must contain ' if ' at start or ' if ' / ' unless ' / ' only if '
    if not (low.startswith('if ') or ' if ' in low or low.startswith('unless ') or ' unless ' in low or ' only if ' in low):
        return None
    # 1. Leading If ... then ...
    m = re.search(r'^\s*if\s+(.+?),\s*then\s+(.+?)\.*\s*$', text, re.I)
    if m:
        return ("IMPLIES", [m.group(1).strip(), m.group(2).strip()])
    # 2. Leading If ... , ... (no explicit 'then'). Ensure second clause does NOT start with 'if' (avoid chaining)
    m = re.search(r'^\s*if\s+(.+?),\s*(?!if\b)(.+?)\.*\s*$', text, re.I)
    if m:
        return ("IMPLIES", [m.group(1).strip(), m.group(2).strip()])
    # 3. Trailing condition: B if A.
    m = re.search(r'^\s*(.+?),\s*if\s+(.+?)\.*\s*$', text, re.I)
    if m:
        return ("IMPLIES", [m.group(2).strip(), m.group(1).strip()])
    # 4. Only if pattern: A only if B  => B -> A
    m = re.search(r'^\s*(.+?)\s+only if\s+(.+?)\.*\s*$', text, re.I)
    if m:
        return ("IMPLIES", [m.group(2).strip(), m.group(1).strip()])
    # 5. Unless leading: Unless A, B. => not A -> B
    m = re.search(r'^\s*unless\s+(.+?),\s*(.+?)\.*\s*$', text, re.I)
    if m:
        A, B = m.group(1).strip(), m.group(2).strip()
        return ("IMPLIES", [f"not ({A})", B])
    # 6. Trailing unless: B unless A. => not A -> B
    m = re.search(r'^\s*(.+?)\s+unless\s+(.+?)\.*\s*$', text, re.I)
    if m:
        B, A = m.group(1).strip(), m.group(2).strip()
        return ("IMPLIES", [f"not ({A})", B])
    return None


def detect_iff(s: str) -> Tuple[str, list] | None:
    if not isinstance(s, str):
        return None
    if re.search(r"\biff\b|if and only if", s, re.I):
        # Crude split around phrase
        parts = re.split(r"\biff\b|if and only if", s, flags=re.I)
        if len(parts) >= 2:
            left = parts[0].strip().rstrip(' .')
            right = parts[1].strip().rstrip(' .')
            return ("IFF", [left, right])
        return ("IFF", [s.strip().rstrip(' .')])
    return None


def detect_xor(s: str) -> Tuple[str, list] | None:
    if not isinstance(s, str):
        return None
    # XOR if explicit exclusivity markers OR a simple 'either X or Y' pattern (two options) not followed by 'neither' variant.
    if re.search(r"\bbut not both\b|\bexactly one\b|\bmutually exclusive\b", s, re.I):
        m = re.search(r"either\s+(.+?)\s+or\s+(.+?)(?:,|\.|;|$)", s, re.I)
        if m:
            return ("XOR", [m.group(1).strip(), m.group(2).strip()])
        return ("XOR", [s.strip().rstrip('.')])
    # Simple either-or (two disjuncts) -> XOR (unless 'neither' appears making it handled elsewhere)
    if 'either' in s.lower() and ' neither ' not in s.lower():
        m2 = re.search(r"either\s+(.+?)\s+or\s+(.+?)(?:,|\.|;|$)", s, re.I)
        if m2:
            left, right = m2.group(1).strip(), m2.group(2).strip()
            # Avoid treating lists with internal ' or ' beyond the main one
            if left and right and ' or ' not in left.lower() and ' or ' not in right.lower():
                return ("XOR", [left.rstrip('.'), right.rstrip('.')])
    return None


def detect_neither_nor(s: str) -> Tuple[str, list] | None:
    if not isinstance(s, str):
        return None
    # If 'either' appears before 'nor', we treat it as a higher-level OR pattern handled upstream.
    low = s.lower()
    # Only treat 'either' as a blocker if it appears as a standalone token (not as part of 'neither').
    if re.search(r'\beither\b', low):
        idx_nor = low.find(' nor ')
        # find standalone either token position
        m_either_token = re.search(r'\beither\b', low)
        idx_either = m_either_token.start() if m_either_token else -1
        if idx_either != -1 and idx_nor != -1 and idx_either < idx_nor:
            return None
    # General pattern allowing 'neither' to appear mid-sentence, capturing the coordinated phrases.
    # Examples:
    #   "KiKi neither barks nor is a dog." -> not (barks), not (is a dog)
    #   "Dried Thai chilies are neither a product of X nor a bakery." -> not (a product of X), not (a bakery)
    m = re.search(r'\bneither\b\s+(.*?)\s+\bnor\b\s+(.*)', s, re.I)
    if m:
        part1 = m.group(1).strip().rstrip(' .,' )
        part2 = m.group(2).strip().rstrip(' .,' )
        # Avoid degenerate captures that accidentally include preceding subject tokens before 'neither'.
        # If part1 still contains another ' neither ' (rare) or is empty, bail.
        if part1 and part2:
            # Map directly to NOR (new operator) with explicit children parts.
            return ("NOR", [part1, part2])
    return None


def detect_or(s: str) -> Tuple[str, list] | None:
    if not isinstance(s, str):
        return None
    low = s.strip().lower()
    # Special exception: pattern "No <NP> ... or <Proper>" where the disjunction is a clean list of two proper-noun
    # like spans (e.g., channel names) – treat as OR so later stages can apply De Morgan if needed.
    # Heuristic: starts with 'No ' then contains exactly one ' or ' at top level and BOTH sides after split each
    # contain at least one capitalised token (to reduce risk of splitting quantifier scope like 'No student or teacher ...').
    if low.startswith('no '):
        # Do a light split on ' or ' (case-insensitive) ignoring cases where ' nor ' appears (handled elsewhere)
        if ' or ' in low and ' nor ' not in low:
            parts = re.split(r'\bor\b', s, flags=re.I)
            if len(parts) == 2:
                left_ctx = parts[0]
                right_ctx = parts[1]
                # Extract the last NP chunk before 'or' by taking last 6 tokens window
                left_tokens = left_ctx.strip().split()
                right_tokens = right_ctx.strip().split()
                # Capitalization test (proper noun heuristic)
                def has_cap(tokens):
                    return any(t[0].isupper() for t in tokens if t and t[0].isalpha())
                if has_cap(left_tokens[-5:]) and has_cap(right_tokens[:5]):
                    # Build disjunct strings: reuse trailing 5 tokens of left and leading 5 tokens of right to approximate
                    left_phrase = ' '.join(left_tokens[-5:]).strip(', .')
                    right_phrase = ' '.join(right_tokens[:5]).strip(', .')
                    if left_phrase and right_phrase:
                        return ("OR", [left_phrase, right_phrase])
    # Reserve 'either ... or ...' constructions for XOR detection; do not emit OR here.
    if re.search(r'\beither\b', s, re.I):
        return None
    parts = re.split(r"\s+or\s+", s, flags=re.I)
    # Allow leading negation in first part (we will later model negation flags in splitter)
    if len(parts) == 2:
        return ("OR", [parts[0].strip().rstrip('.'), parts[1].strip().rstrip('.')])
    return None


def detect_and(s: str) -> Tuple[str, list] | None:
    if not isinstance(s, str):
        return None
    low = s.lower()
    # Guard: meta-negation wrapper 'It is not true that ... and ...' should not force AND top-level; treat as NONE.
    if re.match(r'^\s*it\s+is\s+not\s+true\s+that\b', low):
        return None
    # Simple proper-noun coordination at sentence start: "Alice and Bob ..." -> treat as AND of two entities
    # This helps capture cases like "Butte and Pierre are in the same state." where previous heuristic skipped (single-word left side).
    m_simple_names = re.search(r'^\s*([A-Z][\w-]+)\s+and\s+([A-Z][\w-]+)\b', s)
    if m_simple_names:
        return ("AND", [m_simple_names.group(1).strip(), m_simple_names.group(2).strip()])
    # Handle 'and also' or 'as well as' explicitly (top-level)
    for marker in ["and also", "as well as", "not only", "but also"]:
        if marker in low:
            # Normalize not only ... but also separately, other markers treat as binary split
            if marker == "not only" and "but also" in low:
                m = re.search(r'not only (.+?) but also (.+)', s, re.I)
                if m:
                    return ("AND", [m.group(1).strip().rstrip('.'), m.group(2).strip().rstrip('.')])
            elif marker != "not only":
                # Choose earliest connecting marker of these types
                if marker in ["and also", "as well as"]:
                    parts = re.split(r"\b" + re.escape(marker) + r"\b", s, flags=re.I)
                    if len(parts) == 2:
                        left, right = parts[0].strip().rstrip('. ,'), parts[1].strip().rstrip('. ,')
                        if left and right:
                            return ("AND", [left, right])
    m = re.search(r'\bboth\s+(.+?)\s+and\s+(.+)', s, re.I)
    if m:
        return ("AND", [m.group(1).strip().rstrip('.'), m.group(2).strip().rstrip('.')])
    if ' and ' in low and ' if ' not in low:
        # Use the last top-level ' and ' to reduce impact of inner adjective coordinations
        idx = low.rfind(' and ')
        left = s[:idx].strip().rstrip('. ,')
        right = s[idx+5:].strip().rstrip('. ,')
        # Avoid pathological splits where either side empty
        if left and right and len(left.split()) > 1:
            return ("AND", [left, right])
    return None


def detect_not_only(s: str) -> Tuple[str, list] | None:
    """Detect bare negation only if NO other logical operator patterns present.

    Ignored negators: 'no', 'none' (treated as lexical, not top-level NOT now).
    Active negators: not, n't, never, without, cannot, doesn't, etc.
    If conjunction/disjunction/implication markers appear, we skip to allow
    those detectors to represent structure; negation will be handled downstream.
    """
    if not isinstance(s, str):
        return None
    low = s.lower()
    # If other operator markers present, abort
    if any(tok in low for tok in [' if ', ' unless ', ' either ', ' or ', ' and ', ' nor ', ' only if ', ' iff ', ' both ']):
        return None
    # If the sentence starts with a quantifier word, the negation is within
    # quantifier scope (e.g. "Some X are not Y" = ∃x(X∧¬Y)), not a top-level NOT.
    if re.match(r'^\s*(some|all|every|each|most|few|many|several|any)\b', low):
        return None
    # Detect negation (excluding 'no'/'none')
    if re.search(r"\bnever\b|\bnot\b|n't\b|\bcannot\b|\bcan't\b|\bwon't\b|\bdoesn't\b|\bdon't\b|\bisn't\b|\baren't\b|\bwasn't\b|\bweren't\b|\bhasn't\b|\bhaven't\b|\bhadn't\b|\bshouldn't\b|\bwouldn't\b|\bcouldn't\b|\bmustn't\b|\bwithout\b", low):
        return ("NOT", [s.strip().rstrip('.')])
    return None


def detect_operator_rule_based(sentence: str) -> Dict[str, Any]:
    """Return a dict with operator, confidence, and evidence based on heuristics.

    Ordering places IMPLIES first so that leading 'If/Unless/Only if' constructs are not
    misclassified as OR/AND/NOT due to internal coordination or negation.
    """
    s = (sentence or "").strip()
    # Updated precedence: implication & equivalence before composite disjunction pattern.
    for fn in (
        detect_either_or_conditional,  # must precede implies to avoid misclassification
        detect_implies,
        detect_iff,
        detect_meta_not,              # NEW: handle meta-negation before other splits
        detect_either_neither_xor,
        detect_either_or_neither_nor,
        detect_xor,
        detect_neither_nor,
        detect_or,
        detect_and,
        detect_not_only,              # fires only if none of the above triggered and negation markers present
    ):
        out = fn(s)
        if out:
            op, parts = out
            conf_map = {
                "IFF": 0.9,
                "IMPLIES": 0.85,  # slight boost due to earlier precision improvements
                "XOR": 0.75,
                "AND": 0.7,
                "OR": 0.6,
                "NOT": 0.55,
            }
            if fn is detect_either_or_neither_nor and op == 'OR':
                conf = 0.75  # slightly lower now that implication has priority
            elif fn is detect_meta_not and op == 'NOT':
                conf = 0.65  # meta-negation more precise than generic neg-only
            else:
                conf = conf_map.get(op, 0.5)
            # Quantifier scope guard: if a *single leading* quantifier appears to scope
            # over multiple predicates (AND/OR/XOR/IFF) and is not redundantly present
            # in every part, treat sentence as atomic (NONE) to avoid unsafe splits.
            if op in ("AND", "OR", "XOR", "IFF") and _has_leading_quantifier(s):
                part_q_counts = [1 if _part_has_quantifier(p) else 0 for p in (parts if isinstance(parts, list) else [])]
                if parts and isinstance(parts, list) and sum(part_q_counts) < len(parts):
                    return {
                        "operator": "NONE",
                        "confidence": 0.4,
                        "evidence": "quantifier_scope",
                    }
            return {
                "operator": op,
                "confidence": conf,
                "evidence": "; ".join(parts) if isinstance(parts, list) else str(parts),
            }
    return {"operator": "NONE", "confidence": 0.2, "evidence": "no pattern"}
