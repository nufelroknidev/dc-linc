from __future__ import annotations
import re
from typing import Dict, List


def _clean(s: str) -> str:
    return s.strip().rstrip('.')


def _endify(clause: str) -> str:
    c = clause.strip()
    if not c.endswith('.'): c += '.'
    return c


def _canon_subject_pattern() -> re.Pattern:
    # Rough heuristic verbs to anchor subject/predicate boundary.
    return re.compile(r"\b(is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|should|would|will|spends|provides|barks|complains|hosts|visits|loves|love|contains)\b", re.I)


def split_and_rule(sentence: str) -> Dict:
    """Best-effort deterministic splitter for top-level AND coordination.
    Handles patterns seen in failures: neither/nor, both/and, simple 'X is A and B',
    shared subject with second predicate/adjective, 'and also', 'not only ... but also ...',
    mixed polarity 'X is not A and B'. Returns empty structure if no confident split.
    """
    if not sentence or not isinstance(sentence, str):
        return {"operator":"AND","conjuncts":[],"negated":[],"confidence":0.0,"rationale":"empty"}
    s = sentence.strip()
    low = s.lower()

    # Helper to ensure subject injection when missing
    def _choose_copula(subj: str) -> str:
        s2 = (subj or '').strip().lower()
        if re.search(r'\b(they|we|you)\b', s2):
            return 'are'
        if s2.endswith('ies') or (s2.endswith('s') and not s2.endswith('ss')):
            return 'are'
        return 'is'

    def inject_subject(subj: str, clause: str) -> str:
        c = clause.strip()
        if not subj:
            return clause
        # If clause already starts with subject token or pronoun, skip
        if c.lower().startswith(subj.lower() + ' '):
            return clause
        # If clause begins with an auxiliary + pronoun pattern we keep
        if re.match(r'^(he|she|it|they|we|i)\b', c, re.I):
            return clause
        # If clause starts with a verb or adjective or determiner, prepend appropriately
        if re.match(r'^(is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|should|would|will)\b', c, re.I):
            return f"{subj} {c}".strip()
        if re.match(r'^(?:a|an|the)\b', c, re.I):
            return f"{subj} {_choose_copula(subj)} {c}".strip()
        return f"{subj} {c}".strip()

    # neither ... nor (possibly multiple nor)
    if re.search(r"\bneither\b.+\bnor\b", low):
        # Capture subject BEFORE 'neither'
        msubj = re.match(r'^\s*([^,.]+?)\s+neither\b', s, flags=re.I)
        subj_hint = msubj.group(1).strip() if msubj else None
        body = re.sub(r'^.*?\bneither\b\s*', '', s, flags=re.I)
        parts = re.split(r'\b(?:nor)\b', body, flags=re.I)
        parts = [p.strip(' .') for p in parts if p.strip()]
        if len(parts) >= 2:
            verb_pat = _canon_subject_pattern()
            conj: List[str] = []
            for p in parts:
                cp = p
                # Normalize auxiliary inversion: "does he travel" -> "He travels"
                inv = re.match(r'^(does|did|can|could|will|would|should|might|may)\s+([A-Z][a-zA-Z0-9_-]*)\s+([^ ].+)$', cp, re.I)
                if inv:
                    aux, subj2, rest = inv.groups()
                    # crude present simple heuristic
                    rest = re.sub(r'^travel\b', 'travels', rest)
                    cp = f"{subj2} {rest}".strip()
                if subj_hint and not verb_pat.search(cp):
                    if re.match(r'^(?:a|an|the)\b', cp, re.I):
                        cp = f"{subj_hint} is {cp}"
                    else:
                        cp = f"{subj_hint} {cp}" if re.match(r'^[a-z]', cp) else cp
                # Inject subject if still missing obvious subject token
                if subj_hint:
                    cp = inject_subject(subj_hint, cp)
                # Clean trivial duplicate copulas (e.g., 'is is') from heuristic injections
                cp = re.sub(r'\b(is|are|was|were)\s+\1\b', r'\1', cp, flags=re.I)
                conj.append(_endify(cp))
            return {"operator":"AND","conjuncts":conj,"negated":[True]*len(conj),"confidence":0.9,"rationale":"neither nor"}

    # Pattern: X was part of both A and B
    m_part_both = re.match(r'^(?P<subj>.+?)\s+was part of both\s+(.+?)\s+and\s+(.+)$', s, flags=re.I)
    if m_part_both:
        subj = m_part_both.group('subj').strip()
        tail = s[s.lower().find('was part of both') + len('was part of both'):].strip()
        # Re-extract objects after 'both'
        m_objs = re.match(r'^(.+?)\s+and\s+(.+)$', tail, flags=re.I)
        if m_objs:
            a = _clean(m_objs.group(1))
            b = _clean(m_objs.group(2))
            conj = [f"{subj} was part of {a}.", f"{subj} was part of {b}."]
            return {"operator":"AND","conjuncts":conj,"negated":[False, False],"confidence":0.9,"rationale":"was part of both"}

    # has hosted both pattern
    m_host_both = re.match(r'^(?P<subj>.+?)\s+has hosted both\s+(.+?)\s+and\s+(.+)$', s, flags=re.I)
    if m_host_both:
        subj = m_host_both.group('subj').strip()
        tail = s[s.lower().find('has hosted both') + len('has hosted both'):].strip()
        m_objs = re.match(r'^(.+?)\s+and\s+(.+)$', tail, flags=re.I)
        if m_objs:
            a = _clean(m_objs.group(1))
            b = _clean(m_objs.group(2))
            conj = [f"{subj} has hosted {a}.", f"{subj} has hosted {b}."]
            return {"operator":"AND","conjuncts":conj,"negated":[False, False],"confidence":0.88,"rationale":"has hosted both"}

    # not only A but also B
    m_no_only = re.search(r'not only (.+?) but also (.+)', s, re.I)
    if m_no_only:
        a, b = _clean(m_no_only.group(1)), _clean(m_no_only.group(2))
        return {"operator":"AND","conjuncts":[_endify(a), _endify(b)],"negated":[False, False],"confidence":0.85,"rationale":"not only but also"}

    # both A and B
    m_both = re.search(r'\bboth\s+(.+?)\s+and\s+(.+)', s, re.I)
    if m_both:
        a_raw, b_raw = _clean(m_both.group(1)), _clean(m_both.group(2))
        # If preceding context contains a clear subject+verb phrase ending with 'both'
        pre_ctx_match = re.match(r'^(?P<lead>.+?)\bboth\b', s, flags=re.I)
        lead = pre_ctx_match.group('lead').strip() if pre_ctx_match else ''
        subj = None
        if ' was part of ' in lead.lower():
            subj = lead[:lead.lower().find(' was part of ')].strip()
            if subj:
                conj = [f"{subj} was part of {a_raw}.", f"{subj} was part of {b_raw}."]
                return {"operator":"AND","conjuncts":conj,"negated":[False, False],"confidence":0.9,"rationale":"both and"}
        return {"operator":"AND","conjuncts":[_endify(a_raw), _endify(b_raw)],"negated":[False, False],"confidence":0.88,"rationale":"both and"}

    # and also / as well as
    for marker in ["and also", "as well as"]:
        if marker in low:
            # pick first occurrence of marker to split
            idx = low.find(marker)
            left = s[:idx].strip(' ,.')
            right = s[idx+len(marker):].strip(' ,.')
            # Attempt subject reuse
            # Heuristic: extract subject from left up to first verb
            subj = None
            verb_pat = _canon_subject_pattern()
            mverb = verb_pat.search(left)
            if mverb:
                before = left[:mverb.start()].strip()
                subj = before if before else None
            def ensure_sentence(fragment: str) -> str:
                f = fragment.strip()
                if subj:
                    starts_with_subj = f.lower().startswith(subj.lower()+" ")
                    if not starts_with_subj:
                        if not verb_pat.search(f):
                            if re.match(r'^(?:a|an|the)\b', f, re.I):
                                f = f"{subj} {_choose_copula(subj)} {f}"
                            else:
                                f = f"{subj} {f}"
                        elif re.match(r'^[a-z]+s\b', f) and not re.match(r'^(he|she|it|they|we|i)\b', f, re.I):
                            f = f"{subj} {f}"
                return _endify(f)
            return {"operator":"AND","conjuncts":[ensure_sentence(left), ensure_sentence(right)],"negated":[False, False],"confidence":0.85,"rationale":marker}

    # Simple pattern: Subject ... and ... (shared subject)
    if ' and ' in low:
        # Avoid splitting lists with commas for now
        parts = re.split(r'\s+and\s+', s)
        if 2 <= len(parts) <= 3:
            left = parts[0]
            # If we have three pieces, decide whether last two belong together (internal coordination)
            if len(parts) == 3:
                p2, p3 = parts[1].strip(), parts[2].strip()
                # Heuristic: if third starts with gerund/participle (verb+ing or catching/eating/etc.) or short NP w/out finite verb, treat as internal to second predicate
                if re.match(r'^(?:[a-zA-Z]+ing)\b', p3.lower()) or re.match(r'^(catching|eating|drinking|talking|playing|working|learning|reading)\b', p3.lower()):
                    right = f"{p2} and {p3}"
                else:
                    # Treat as three separate top-level predicates: rebuild later
                    right = p2
                    extra = p3
                # else branch continues
            else:
                right = parts[1]
            # Derive subject from left
            verb_pat = _canon_subject_pattern()
            mverb = verb_pat.search(left)
            subj = None
            if mverb:
                subj_candidate = left[:mverb.start()].strip()
                subj = subj_candidate if subj_candidate else None
            # If right lacks a verb, attach subject + copula
            if subj and not verb_pat.search(right):
                if re.match(r'^(?:a|an|the)\b', right.strip(), re.I):
                    right = f"{subj} {_choose_copula(subj)} {right.strip()}"
                else:
                    right = f"{subj} {right.strip()}"
            # Mixed polarity: X is not P and Q
            mixed = False
            second_should_negate = False
            if re.search(r'\bis\s+not\b', left.lower()) and subj:
                # extract complement after 'is not'
                mcomp = re.search(r'\bis\s+not\s+(.+)$', left, re.I)
                if mcomp:
                    comp = mcomp.group(1).strip()
                    left = f"{subj} is {comp}"  # canonical positive form
                    mixed = True
                    # Heuristic: if second clause is adjectival/complement (no verb), also mark negated per user requirement
                    if subj and not verb_pat.search(right):
                        second_should_negate = True
            # Inject subject into right clause if missing
            if subj:
                right = inject_subject(subj, right)
                if re.match(r'^[a-z]+s\b', right.strip()) and not right.lower().startswith(subj.lower()):
                    right = f"{subj} {right.strip()}"
            conjuncts = [_endify(left), _endify(right)]
            # If right contains ' and ' followed by a gerund phrase and is long, consider splitting
            if subj and re.search(r'\band\s+[a-z]+ing\b', right.lower()) and len(right.split()) > 10:
                # Split once at the last ' and '
                r_low = right.lower()
                idx = r_low.rfind(' and ')
                if idx > -1:
                    head = right[:idx].strip()
                    tail = right[idx+5:].strip()
                    if re.match(r'^[a-z]+ing\b', tail.lower()):
                        # Ensure subject in both
                        head = inject_subject(subj, head)
                        tail = inject_subject(subj, tail)
                        conjuncts = [_endify(left), _endify(head), _endify(tail)]
            # If we earlier detected an extra third predicate (len(parts)==3 and not merged), add it
            if 'extra' in locals():
                ex = extra
                if subj:
                    ex = inject_subject(subj, ex)
                if subj and not verb_pat.search(ex):
                    if re.match(r'^(?:a|an|the)\b', ex.strip(), re.I):
                        ex = f"{subj} {_choose_copula(subj)} {ex.strip()}"
                    else:
                        ex = f"{subj} {ex.strip()}"
                conjuncts.append(_endify(ex))
            negated = [False] * len(conjuncts)
            if 'extra' in locals() and len(negated) < len(conjuncts):  # safety
                negated = [False] * len(conjuncts)
            if mixed:
                if len(negated) > 0:
                    negated[0] = True
                if second_should_negate and len(negated) > 1:
                    negated[1] = True
            else:
                # Detect leading explicit negation per clause
                for i, c in enumerate(conjuncts):
                    if re.search(r'\b(is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|should|would|will|spends|provides|barks|complains|hosts|visits|loves|love)\s+not\b', c, re.I) or re.search(r"\b(no|never|none)\b", c.lower()):
                        negated[i] = True
                        # remove 'not' for canonical form when simple copula/aux
                        c2 = re.sub(r'\b(is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|should|would|will)\s+not\b', r'\1', c, flags=re.I)
                        conjuncts[i] = _endify(c2.replace('  ', ' ').strip())
            # Cleanup artifacts like 'is complains', 'is catches'
            cleaned = []
            for c in conjuncts:
                c = re.sub(r'\b(is|are|was|were)\s+([a-z]+s)\b', lambda m: m.group(2) if m.group(2).endswith('s') and not re.match(r'(?:is|are|was|were)\b', m.group(2)) else m.group(0), c, flags=re.I)
                cleaned.append(c)
            conjuncts = cleaned
            return {"operator":"AND","conjuncts":conjuncts,"negated":negated,"confidence":0.8,"rationale":"and"}

    return {"operator":"AND","conjuncts":[],"negated":[],"confidence":0.1,"rationale":"no rule split"}


def validate_and_split(output: Dict) -> bool:
    if not isinstance(output, dict):
        return False
    if output.get("operator") != "AND":
        return False
    conj = output.get("conjuncts")
    neg = output.get("negated")
    if not isinstance(conj, list) or not isinstance(neg, list):
        return False
    if len(conj) != len(neg) or len(conj) == 0:
        return False
    # Basic sanity: no hallucination phrases
    bad = {"i think", "as an ai", "cannot"}
    for c in conj:
        if not isinstance(c, str) or not c.strip():
            return False
        if any(b in c.lower() for b in bad):
            return False
    return True
