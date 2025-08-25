from __future__ import annotations
from .utils import escape_and_fill

_SPLIT_IFF_BASE = """
You are a JSON function. Task: Extract a TOP-LEVEL biconditional (logical equivalence) if present.
Return STRICT JSON ONLY (no prose, no markdown). Schema:
{"operator":"IFF|NONE","children":["...","..."],"confidence":0.0,"rationale":"<10 words>"}

Decision rule:
- Output operator IFF ONLY if a clear marker ("iff" or "if and only if") connects two substantive clauses.
- Otherwise return: {"operator":"NONE","children":["<sentence>"],"confidence":<0.3,"rationale":"no iff"}.

Field notes:
- children: exactly two clauses (left, right) with marker removed; each capitalized and ending with a period.
- Keep original lexical forms; do NOT paraphrase.
- Preserve polarity as written (do NOT strip or invert negations) — unlike earlier versions we no longer output a negation boolean array.
- If one side contains embedded punctuation/coordination, include it verbatim (be conservative).
- If either side is missing or trivial (<2 tokens), return NONE.

Accepted markers (case-insensitive):
- "iff"
- "if and only if"

Normalization:
1. Remove the marker from between the clauses.
2. Trim surrounding whitespace and trailing commas.
3. Ensure final period on each child.

Examples:
Sentence: Authorized access occurs iff the presented token proves valid.
{"operator":"IFF","children":["Authorized access occurs.","The presented token proves valid."],"confidence":0.9,"rationale":"iff"}

Sentence: The build succeeds if and only if every test case passes.
{"operator":"IFF","children":["The build succeeds.","Every test case passes."],"confidence":0.9,"rationale":"if and only if"}

Sentence: Logging is enabled iff tracing is active.
{"operator":"IFF","children":["Logging is enabled.","Tracing is active."],"confidence":0.87,"rationale":"iff"}

Sentence: Feature X is active iff the configuration flag is true.
{"operator":"IFF","children":["Feature X is active.","The configuration flag is true."],"confidence":0.89,"rationale":"iff"}

Sentence: The system does not alert if and only if a threshold is breached.
{"operator":"IFF","children":["The system does not alert.","A threshold is breached."],"confidence":0.85,"rationale":"iff"}

Sentence: The server restarts quickly if and only if updates are not applied.
{"operator":"IFF","children":["The server restarts quickly.","Updates are not applied."],"confidence":0.82,"rationale":"iff"}

Sentence: The API does not respond if and only if the service is not working.
{"operator":"IFF","children":["The API does not respond.","The service is not working."],"confidence":0.84,"rationale":"iff"}

Sentence: {SENTENCE}
"""

SPLIT_IFF_PROMPT = lambda sentence: escape_and_fill(_SPLIT_IFF_BASE, sentence)

__all__ = ["SPLIT_IFF_PROMPT"]
