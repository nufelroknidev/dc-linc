from __future__ import annotations
from .utils import escape_and_fill

# OR splitting prompt formatted to mirror the AND template structure for consistency.
_SPLIT_OR_BASE = """
You are a JSON function. Task: Split the input sentence into its TOP-LEVEL disjuncts whose operator is OR.
Return STRICT JSON ONLY (no prose, no code fences). Schema:
{"operator":"OR|NONE","disjuncts":["..."],"confidence":0.0,"rationale":"<10 words>"}

Decision rule:
- Output OR only if there is an unambiguous TOP-LEVEL OR coordination that can be safely expanded.
- Otherwise return: {"operator":"NONE","disjuncts":["<sentence>"],"confidence":<0.3,"rationale":"<why>"}.

Field notes:
- disjuncts: minimal grammatical clauses (each ends with a period).
- Expand shared subjects/predicates/objects/modifiers so each clause stands alone.
- Only split at TOP LEVEL.

Map to OR (coordination):
- A or B
- A, B, or C (lists)

Do NOT map (return NONE):
- Explicit exclusivity cues (e.g., "but not both", "exactly one") → treat as different operator.
- Nested OR that is not top-level.

Guidelines:
- Do not invent facts; only restructure.
- Keep entity names verbatim.
- Each clause must be capitalized and end with a period.
- If an item in a list is just an adjective/noun phrase sharing a copula/verb, reconstruct a full clause.
- If any clause would be ungrammatical or ambiguous after expansion, return NONE.

Format examples exactly: "Sentence:" line followed by plain JSON line. No extra commentary.

Examples:
Sentence: the weather is sunny or rainy.
{"operator":"OR","disjuncts":["The weather is sunny.","The weather is rainy."],"confidence":0.9,"rationale":"top-level OR disjunction"}

Sentence: The school is lovely or ugly.
{"operator":"OR","disjuncts":["The school is lovely.","The school is ugly."],"confidence":0.85,"rationale":"adjectival complements under OR"}

Sentence: The script logs warnings, errors, or critical failures.
{"operator":"OR","disjuncts":["The script logs warnings.","The script logs errors.","The script logs critical failures."],"confidence":0.9,"rationale":"comma list with OR"}

Sentence: Zoe Kardi debugs front-end or optimizes back-end services.
{"operator":"OR","disjuncts":["Zoe Kardi debugs front-end services.","Zoe Kardi optimizes back-end services."],"confidence":0.86,"rationale":"coordinated verb phrases under OR"}

Sentence: It is not raining or traffic is light.
{"operator":"OR","disjuncts":["It is not raining.","Traffic is light."],"confidence":0.84,"rationale":"mixed polarity; top-level OR"}

Sentence: The new model is not a car that comes with autonomous driving or offers excellent fuel economy.
{"operator":"OR","disjuncts":["The new model is not a car that comes with autonomous driving.","The new model is not a car that offers excellent fuel economy."],"confidence":0.78,"rationale":"shared subject; two complements under OR"}

Sentence: Choose sunrise or sunset photography.
{"operator":"OR","disjuncts":["Choose sunrise photography.","Choose sunset photography."],"confidence":0.8,"rationale":"imperative alternatives under OR"}

Sentence: Rafael is not a musician who recorded the album or one of the most streamed artists this year.
{"operator":"OR","disjuncts":["Rafael is not a musician who recorded the album.","Rafael is one of the most streamed artists this year."],"confidence":0.75,"rationale":"mixed polarity; shared subject"}

Sentence: The rover reboots or transmits telemetry or performs safe-mode recovery.
{"operator":"OR","disjuncts":["The rover reboots.","The rover transmits telemetry.","The rover performs safe-mode recovery."],"confidence":0.9,"rationale":"multi-item OR disjunction"}

Sentence: Sam is light or handsome.
{"operator":"OR","disjuncts":["Sam is light.","Sam is handsome."],"confidence":0.97,"rationale":"adjectival complements under OR"}

Sentence: The parser detects syntax errors or semantic drift.
{"operator":"OR","disjuncts":["The parser detects syntax errors.","The parser detects semantic drift."],"confidence":0.83,"rationale":"object alternatives under OR"}

Sentence: {SENTENCE}
"""

SPLIT_OR_PROMPT = lambda sentence: escape_and_fill(_SPLIT_OR_BASE, sentence)
