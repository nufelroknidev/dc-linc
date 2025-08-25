from __future__ import annotations
from .utils import escape_and_fill

_DETECT_OPERATOR_BASE = """
You are a deterministic JSON function. Detect the TOP-LEVEL logical operator and list ALL operators present anywhere in the sentence.

Return STRICT SINGLE-LINE JSON ONLY:
{"top_level":"IMPLIES|IFF|XOR|OR|AND|NOT|NONE","present":["IMPLIES|IFF|XOR|OR|AND|NOT"],"confidence":0.00,"rationale":"<15 words>"}

Contract:
- Keys: exactly top_level, present, confidence, rationale.
- present: unique operators found anywhere (exclude NONE). Empty array if none.
- top_level: the outermost operator or NONE.
- Uppercase only. One line. No extra text.

Rules (brief):
1) If cross-scope binding/quantifiers/relatives/comparatives/anaphora span halves → top_level=NONE (but still list inner operators in present if clearly local).
2) Internal lists only (NP/PP) without two predicate clauses → don’t set top_level, but you may include local AND/OR in present.
3) Non-distributive/joint predicates (together/each other/between/same/different/jointly/respectively/average/total/ratio/union/intersection) → top_level=NONE.
4) Token hygiene: whole tokens only; punctuation isn’t an operator.
5) Copula “X is Y” is atomic; doesn’t create IMPLIES.
6) NOT: outer sentential negation with no other outer connective → top_level=NOT.
7) IFF cues: “iff/if and only if/necessary and sufficient/is equivalent to/exactly when”.
8) IMPLIES cues: “if … then …”, initial “if …, …”, or “therefore/thus/so/hence/consequently” linking two full clauses.
9) XOR only with explicit exclusivity: “exactly one/only one/but not both/mutually exclusive”.
10) Outer “either … or …” outranks inner conditionals for top_level=OR.

Confidence bins:
- 0.97 explicit unambiguous cue; 0.86 mild nesting; 0.72 slight ambiguity.

Few-shots examples:

Input: If the reactor overheats then the pumps activate.
Output: {"top_level":"IMPLIES","present":["IMPLIES"],"confidence":0.97,"rationale":"top-level if…then"}

Input: Either the cache is primed or latency spikes.
Output: {"top_level":"OR","present":["OR"],"confidence":0.92,"rationale":"outer either…or"}

Input: A process is valid iff its output is reproducible.
Output: {"top_level":"IFF","present":["IFF"],"confidence":0.97,"rationale":"explicit iff"}

Input: It is not the case that the server is reachable.
Output: {"top_level":"NOT","present":["NOT"],"confidence":0.97,"rationale":"outer sentential negation"}

Input: If algae bloom persists and oxygen drops, fish die and plants decay.
Output: {"top_level":"IMPLIES","present":["IMPLIES","AND"],"confidence":0.95,"rationale":"conditional spans internal ANDs"}

Input: All users with admin tokens in production were notified.
Output: {"top_level":"NONE","present":[],"confidence":0.96,"rationale":"quantifier/relative scope risk"}

prediction example:
Input: {sentence}
"""

DETECT_OPERATOR_PROMPT = lambda sentence: escape_and_fill(_DETECT_OPERATOR_BASE, sentence)
