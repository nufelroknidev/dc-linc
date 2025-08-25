from __future__ import annotations
from .utils import escape_and_fill

_SPLIT_XOR_BASE = """
You are a JSON function. Task: Split the input sentence into its TOP-LEVEL EXCLUSIVE disjuncts (logical XOR generalised to EXACTLY ONE OF N). Emit a list of N≥2 alternatives when exactly/only/at-most one of the listed options holds.

Return STRICT JSON ONLY (no prose, no code fences). Schema:
{"operator":"XOR|NONE","disjuncts":["..."],"confidence":0.0,"rationale":"<10 words>"}

Exclusivity cues (accept any of these; also accept naturally exclusive pairs):
- "exactly one (of)", "only one (of)", "one of" + concrete list
- "mutually exclusive"
- "either A or B" (binary), esp. with clarifier "but not both"
- explicit clarifier: "(not both)", "but not both", "at most one"
- naturally exclusive pairs: on/off, true/false, even/odd, locked/unlocked, Canadian/Australian, etc.

Decision rule:
- Output XOR only if a clear exclusivity cue applies and there are 2–8 concrete, parallel, top-level alternatives.
- Otherwise return: {"operator":"NONE","disjuncts":["<sentence>"],"confidence":<0.3,"rationale":"no XOR cue"}.

Field notes:
- disjuncts are minimal grammatical clauses (capitalised, end with a period).
- Reuse shared subject/copula/aux so each clause stands alone.
- Do not fabricate entities, modifiers, numbers, or attributes.

Reject as NONE when:
- Options are comparative/interval thresholds ("<", "≥", "at least", ranges).
- List includes vague residuals ("etc.", "other", "something else").
- Ellipsis cannot be expanded unambiguously.
- Attachment is not top-level or options are not syntactically parallel.

Canonicalisation steps:
1) Detect exclusivity cue and its span.
2) Identify the options (after the cue or following colon/parentheses). Split on top-level commas/conjunctions.
3) Filter empties; ensure 2..8 remain.
4) Expand shared subject/copula/aux to form full clauses.
5) Strip scaffolding words (either, exactly, only one of, one of, mutually exclusive, not both, at most one).
6) Validate concreteness and parallelism; if violated → NONE.
7) Calibrate confidence.

Confidence guide:
- 0.90–0.95: strong cue (“but not both”, “exactly one”, “only one”, “mutually exclusive”) + clean options.
- 0.85–0.90: strong cue with 3–5 options, all concrete.
- 0.78–0.85: strong cue with 6–8 options, all concrete.
- 0.55–0.70: weak but unmistakable exclusivity with 2–3 options.
- <0.30: return NONE.

Output constraints:
- Accepted XOR: 2..8 disjuncts.
- Each clause ends with a period.
- Rationale: concise exclusivity label (e.g., "but-not-both exclusivity").

Format examples exactly: "Sentence:" line then a single JSON line. No extra commentary.

Examples:
Sentence: Either the lights are on or the alarm is triggered, but not both.
{"operator":"XOR","disjuncts":["The lights are on.","The alarm is triggered."],"confidence":0.93,"rationale":"but-not-both exclusivity"}

Sentence: Either Sarah will not complete the project or Michael will not handle it.
{"operator":"XOR","disjuncts":["Sarah will not complete the project.","Michael will nothandle the project."],"confidence":0.93,"rationale":"but-not-both exclusivity"}

Sentence: Either Charlie will attend the meeting or Dana will, but not both.
{"operator":"XOR","disjuncts":["Charlie will attend the meeting.","Dana will attend the meeting."],"confidence":0.93,"rationale":"but-not-both exclusivity"}

Sentence: Exactly one of the switches is flipped: red or blue.
{"operator":"XOR","disjuncts":["The red switch is flipped.","The blue switch is flipped."],"confidence":0.9,"rationale":"exactly-one exclusivity"}

Sentence: One of the servers is down: east or west.
{"operator":"XOR","disjuncts":["The east server is down.","The west server is down."],"confidence":0.9,"rationale":"one-of exclusivity"}

Sentence: Only one of the modules loads: auth, billing, or cache.
{"operator":"XOR","disjuncts":["The auth module loads.","The billing module loads.","The cache module loads."],"confidence":0.88,"rationale":"only-one exclusivity"}

Sentence: Exactly one of A, B, C, or D fails.
{"operator":"XOR","disjuncts":["A fails.","B fails.","C fails.","D fails."],"confidence":0.87,"rationale":"exactly-one exclusivity"}

Sentence: At most one of the sensors is offline: north, east, south.
{"operator":"XOR","disjuncts":["The north sensor is offline.","The east sensor is offline.","The south sensor is offline."],"confidence":0.86,"rationale":"at-most-one exclusivity"}

Sentence: Either the backup is not complete or the system shuts down (but not both).
{"operator":"XOR","disjuncts":["The backup is not complete.","The system shuts down."],"confidence":0.88,"rationale":"but-not-both exclusivity"}

Sentence: Exactly one of the features is enabled: logging, tracing, metrics, or caching.
{"operator":"XOR","disjuncts":["Logging is enabled.","Tracing is enabled.","Metrics are enabled.","Caching is enabled."],"confidence":0.88,"rationale":"exactly-one exclusivity"}

Sentence: Exactly one of these is true: the result is positive or the result is negative.
{"operator":"XOR","disjuncts":["The result is positive.","The result is negative."],"confidence":0.9,"rationale":"exactly-one exclusivity"}

Sentence: Robert Wilson was either Canadian or Australian.
{"operator":"XOR","disjuncts":["Robert Wilson was Canadian.","Robert Wilson was Australian."],"confidence":0.93,"rationale":"but-not-both exclusivity"}

Sentence: Emma either studies chemistry or works at the laboratory, but not both.
{"operator":"XOR","disjuncts":["Emma studies chemistry.","Emma works at the laboratory."],"confidence":0.93,"rationale":"but-not-both exclusivity"}

Sentence: At most one of these statements is true: the door is locked or the door is unlocked.
{"operator":"XOR","disjuncts":["The door is locked.","The door is unlocked."],"confidence":0.86,"rationale":"at-most-one exclusivity"}

Sentence: Either the email was sent or the server crashed.
{"operator":"XOR","disjuncts":["The email was sent.","The server crashed."],"confidence":0.93,"rationale":"but-not-both exclusivity"}

Sentence: {SENTENCE}
"""

SPLIT_XOR_PROMPT = lambda sentence: escape_and_fill(_SPLIT_XOR_BASE, sentence)
