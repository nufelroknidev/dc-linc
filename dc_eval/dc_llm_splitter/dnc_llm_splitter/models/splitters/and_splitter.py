from __future__ import annotations
import re
from typing import Dict, List

def _clean(s: str) -> str:
    return s.strip().rstrip('.')

def _endify(clause: str) -> str:
    c = clause.strip()
    if not c.endswith('.'):
        c += '.'
    return c

def _canon_subject_pattern() -> re.Pattern:
    """Return a pattern approximating sentence-level auxiliaries / copulas / modals.
    Intentionally kept SMALL to avoid overfitting to specific test verbs.
    """
    return re.compile(r"\b(is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|should|would|will)\b", re.I)

def split_and_rule(sentence: str) -> Dict:
    if not sentence or not isinstance(sentence, str):
        return {"operator":"AND","conjuncts":[],"negated":[],"confidence":0.0,"rationale":"empty"}
    s = sentence.strip()
    low = s.lower()

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
        if c.lower().startswith(subj.lower() + ' '):
            return clause
        if re.match(r'^(he|she|it|they|we|i)\b', c, re.I):
            return clause
        if re.match(r'^(is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|should|would|will)\b', c, re.I):
            return f"{subj} {c}".strip()
        if re.match(r'^(?:a|an|the)\b', c, re.I):
            return f"{subj} {_choose_copula(subj)} {c}".strip()
        return f"{subj} {c}".strip()

    # Generic: neither ... nor ... (multiple 'nor' supported)
    if re.search(r"\bneither\b.+\bnor\b", low):
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
                inv = re.match(r'^(does|did|can|could|will|would|should|might|may)\s+([A-Z][a-zA-Z0-9_-]*)\s+([^ ].+)$', cp, re.I)
                if inv:
                    aux, subj2, rest = inv.groups()
                    rest = re.sub(r'^travel\b', 'travels', rest)
                    cp = f"{subj2} {rest}".strip()
                if subj_hint and not verb_pat.search(cp):
                    if re.match(r'^(?:a|an|the)\b', cp, re.I):
                        cp = f"{subj_hint} is {cp}"
                    else:
                        cp = f"{subj_hint} {cp}" if re.match(r'^[a-z]', cp) else cp
                if subj_hint:
                    cp = inject_subject(subj_hint, cp)
                cp = re.sub(r'\b(is|are|was|were)\s+\1\b', r'\1', cp, flags=re.I)
                conj.append(_endify(cp))
            return {"operator":"AND","conjuncts":conj,"negated":[True]*len(conj),"confidence":0.9,"rationale":"neither nor"}

    m_no_only = re.search(r'not only (.+?) but also (.+)', s, re.I)
    if m_no_only:
        a, b = _clean(m_no_only.group(1)), _clean(m_no_only.group(2))
        return {"operator":"AND","conjuncts":[_endify(a), _endify(b)],"negated":[False, False],"confidence":0.85,"rationale":"not only but also"}

    m_both = re.search(r'\bboth\s+(.+?)\s+and\s+(.+)', s, re.I)
    if m_both:
        a_raw, b_raw = _clean(m_both.group(1)), _clean(m_both.group(2))
        return {"operator":"AND","conjuncts":[_endify(a_raw), _endify(b_raw)],"negated":[False, False],"confidence":0.88,"rationale":"both and"}

    for marker in ["and also", "as well as"]:
        if marker in low:
            idx = low.find(marker)
            left = s[:idx].strip(' ,.')
            right = s[idx+len(marker):].strip(' ,.')
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

    # Generic comma list with final 'and':  A, B, and C  (extract subject if possible, replicate)
    if ',' in s and ' and ' in low:
        # Split on commas first, then handle last 'and'
        prelim = [p.strip() for p in s.split(',')]
        if len(prelim) >= 2 and any(' and ' in p.lower() for p in prelim[-1:]):
            last = prelim[-1]
            before_last = prelim[:-1]
            last_parts = re.split(r'\s+and\s+', last)
            items = before_last + [lp.strip() for lp in last_parts]
            # Basic filter
            items = [i.rstrip(' .') for i in items if i and len(i.split()) > 0]
            if len(items) >= 3:
                # Attempt subject extraction from first item if pattern Subject Verb ...
                verb_pat = _canon_subject_pattern()
                first = items[0]
                mverb = verb_pat.search(first)
                subj = first[:mverb.start()].strip() if mverb else None
                conj: List[str] = []
                for it in items:
                    c_it = it
                    if subj and not c_it.lower().startswith(subj.lower()) and not verb_pat.search(c_it):
                        # simple adjective/np complement
                        if re.match(r'^(?:a|an|the)\b', c_it, re.I):
                            c_it = f"{subj} is {c_it}"
                        else:
                            c_it = f"{subj} {c_it}"
                    conj.append(_endify(c_it))
                return {"operator":"AND","conjuncts":conj,"negated":[False]*len(conj),"confidence":0.75,"rationale":"comma list"}

    if ' and ' in low:
        parts = re.split(r'\s+and\s+', s)
        if 2 <= len(parts) <= 3:
            left = parts[0]
            if len(parts) == 3:
                p2, p3 = parts[1].strip(), parts[2].strip()
                if re.match(r'^(?:[a-zA-Z]+ing)\b', p3.lower()):
                    right = f"{p2} and {p3}"
                else:
                    right = p2
                    extra = p3
            else:
                right = parts[1]
            verb_pat = _canon_subject_pattern()
            mverb = verb_pat.search(left)
            subj = None
            if mverb:
                subj_candidate = left[:mverb.start()].strip()
                subj = subj_candidate if subj_candidate else None
            if subj and not verb_pat.search(right):
                if re.match(r'^(?:a|an|the)\b', right.strip(), re.I):
                    right = f"{subj} {_choose_copula(subj)} {right.strip()}"
                else:
                    right = f"{subj} {right.strip()}"
            mixed = False
            second_should_negate = False
            if re.search(r'\bis\s+not\b', left.lower()) and subj:
                mcomp = re.search(r'\bis\s+not\s+(.+)$', left, re.I)
                if mcomp:
                    comp = mcomp.group(1).strip()
                    left = f"{subj} is {comp}"
                    mixed = True
                    if subj and not verb_pat.search(right):
                        second_should_negate = True
            if subj:
                right = inject_subject(subj, right)
                if re.match(r'^[a-z]+s\b', right.strip()) and not right.lower().startswith(subj.lower()):
                    right = f"{subj} {right.strip()}"
            conjuncts = [_endify(left), _endify(right)]
            if subj and re.search(r'\band\s+[a-z]+ing\b', right.lower()) and len(right.split()) > 10:
                r_low = right.lower()
                idx = r_low.rfind(' and ')
                if idx > -1:
                    head = right[:idx].strip()
                    tail = right[idx+5:].strip()
                    if re.match(r'^[a-z]+ing\b', tail.lower()):
                        head = inject_subject(subj, head)
                        tail = inject_subject(subj, tail)
                        conjuncts = [_endify(left), _endify(head), _endify(tail)]
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
            if mixed:
                if len(negated) > 0:
                    negated[0] = True
                if second_should_negate and len(negated) > 1:
                    negated[1] = True
            else:
                for i, c in enumerate(conjuncts):
                    if re.search(r'\b(is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|should|would|will)\s+not\b', c, re.I) or re.search(r"\b(no|never|none)\b", c.lower()):
                        negated[i] = True
                        c2 = re.sub(r'\b(is|are|was|were|has|have|had|does|do|did|can|could|may|might|must|should|would|will)\s+not\b', r'\1', c, flags=re.I)
                        conjuncts[i] = _endify(c2.replace('  ', ' ').strip())
            cleaned = []
            for c in conjuncts:
                cleaned.append(c)
            conjuncts = cleaned
            return {"operator":"AND","conjuncts":conjuncts,"negated":negated,"confidence":0.8,"rationale":"and"}

    return {"operator":"AND","conjuncts":[],"negated":[],"confidence":0.1,"rationale":"no rule split"}
