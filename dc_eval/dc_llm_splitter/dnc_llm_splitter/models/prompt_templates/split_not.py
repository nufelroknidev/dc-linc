from __future__ import annotations
from .utils import escape_and_fill

# Detailed NOT splitting instructions mirroring the style used for AND/OR.
_SPLIT_NOT_BASE = """
You are a JSON function. Task: Given a sentence whose TOP-LEVEL logical operator is a unary surface negation, output a POSITIVE canonical clause (negation removed) plus metadata.
Return STRICT JSON ONLY (no prose, no code fences). Schema:
{"operator":"NOT|NONE","children":["..."],"confidence":0.0,"rationale":"<10 words>"}

Decision rule:
- Output operator NOT only if there is a clear SINGLE top-level surface negation (not / no / never / cannot / won't / shouldn't / didn't / etc.).
- Otherwise return: {"operator":"NONE","children":["<sentence>"],"confidence":<0.3,"rationale":"<why>"}

Field notes:
- children: exactly ONE positive clause (capitalized, ends with a period) with all surface negation markers removed (not / never / no / cannot / doesn't / isn't / won't / shouldn't / etc.).
- Do NOT include any negation token inside the child text. The outer operator NOT encodes the negation.
- Preserve tense/aspect/modality (e.g., "cannot proceed" -> child: "can proceed.").
- Convert contracted/auxiliary negation forms to their positive forms (doesn't -> does; won't -> will; can't -> can; shouldn't -> should; didn't -> did; hadn't -> had).
- Existential: "There is no X." -> child: "There is X." (keep existential form, drop 'no').
- If scope of negation over coordination is ambiguous, still produce a single positive clause; lower confidence.
- Avoid fabricating entities beyond those present.

Rationale tags pattern: "top-level negation: <cue>" where <cue> is 'not', 'never', 'cannot', contraction, etc.
Confidence: <0.3 if sentence lacks a clear single top-level negation.

Examples:
Sentence: An object does not move.
{"operator":"NOT","children":["An object moves."],"confidence":0.88,"rationale":"top-level negation: not"}

Sentence: It never responds.
{"operator":"NOT","children":["It responds."],"confidence":0.83,"rationale":"top-level negation: never"}

Sentence: It isn't active.
{"operator":"NOT","children":["It is active."],"confidence":0.8,"rationale":"top-level negation: isn't"}

Sentence: This is not helpful and concise.
{"operator":"NOT","children":["This is helpful and concise."],"confidence":0.45,"rationale":"top-level negation: not"}

Sentence: It is not unhelpful.
{"operator":"NOT","children":["It is unhelpful."],"confidence":0.6,"rationale":"top-level negation: double negation, keep one"}

Sentence: An agent cannot proceed.
{"operator":"NOT","children":["An agent can proceed."],"confidence":0.78,"rationale":"top-level negation: cannot"}

Sentence: An item shouldn't fail validation.
{"operator":"NOT","children":["An item should fail validation."],"confidence":0.72,"rationale":"top-level negation: shouldn't"}

Sentence: A process wouldn't start.
{"operator":"NOT","children":["A process would start."],"confidence":0.7,"rationale":"top-level negation: wouldn't"}

Sentence: A task couldn't complete.
{"operator":"NOT","children":["A task could complete."],"confidence":0.7,"rationale":"top-level negation: couldn't"}

Sentence: Input mustn't exceed limits.
{"operator":"NOT","children":["Input must exceed limits."],"confidence":0.65,"rationale":"top-level negation: mustn't"}

Sentence: The system won't respond.
{"operator":"NOT","children":["The system will respond."],"confidence":0.73,"rationale":"top-level negation: won't"}

Sentence: The service doesn't log events.
{"operator":"NOT","children":["The service logs events."],"confidence":0.74,"rationale":"top-level negation: doesn't"}

Sentence: The experiment didn't yield results.
{"operator":"NOT","children":["The experiment yielded results."],"confidence":0.74,"rationale":"top-level negation: didn't"}

Sentence: The session hadn't expired.
{"operator":"NOT","children":["The session had expired."],"confidence":0.65,"rationale":"top-level negation: hadn't"}

Sentence: {SENTENCE}
"""

SPLIT_NOT_PROMPT = lambda sentence: escape_and_fill(_SPLIT_NOT_BASE, sentence)
