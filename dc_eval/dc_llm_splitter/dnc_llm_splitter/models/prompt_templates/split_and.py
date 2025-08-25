from __future__ import annotations
from .utils import escape_and_fill

# AND splitter: only emit AND for safe top-level coordination; otherwise NONE + echo original.
_SPLIT_AND_BASE = """
You are a JSON function. Task: Split the input sentence into its TOP-LEVEL conjuncts whose operator is AND (including synonymous surface forms).
Return STRICT JSON ONLY (no prose, no code fences). Schema:
{"operator":"AND|NONE","conjuncts":["..."],"confidence":0.0,"rationale":"<10 words>"}

Decision rule:
- Output AND only if there is an unambiguous TOP-LEVEL AND coordination that can be safely expanded.
- Otherwise return: {"operator":"NONE","conjuncts":["<sentence>"],"confidence":<0.3,"rationale":"<why>"}.
- Special cases that must return NONE (echo original sentence):
- Any sentence containing a clear 'neither ... nor' coordination — do NOT split such negation-coordination.
- Cases where coordinated adjectives or modifiers attach to a noun phrase and cannot be replicated unambiguously into separate grammatical sentences (e.g., "A portrait by Lena is colorful and moving.").
- Any coordination where expanding shared scaffolding would yield ungrammatical or semantically incoherent clauses.

Clause construction when AND:
- Each clause is a minimal grammatical sentence ending with a period.
- Replicate shared scaffolding (subject/verb/object/modifier/adjective/adverb) that appears before the split in EVERY clause.
- Keep negation INSIDE the clause text (no separate flags).
- Resolve pronouns/bound variables ONLY if the antecedent is explicit within the sentence; if not resolvable, return NONE.
- Do NOT split when a quantifier/determiner scopes over the coordination (all, every, no, some, most, few, many, at least/at most/exactly N, there is/are).

Top-level mapping to AND (treat as coordination):
- A and B
- both A and B
- lists: A, B, and C
- as well as; and also
- not only A but also B
- “A but B” where B is additive (conjunctive), not contrastive entailment-breaking

Guardrails:
- Only split at TOP LEVEL (ignore coordination inside subordinate/relative clauses).
- If coordination is inside a conditional, relative clause, or complement, return NONE.
- If shared scaffolding cannot be unambiguously replicated, return NONE.
- If any conjunct would be ungrammatical or semantically incomplete after expansion, return NONE.
- If the sentence contains negation coordination using 'neither'...'nor', return NONE (avoid splitting negation wrappers).

Formatting:
- Prefix each example with "Sentence:" then a newline containing JSON. No extra prose.

Examples:
Sentence: Both owls and hawks hunt at dusk.
{"operator":"AND","conjuncts":["Owls hunt at dusk.","Hawks hunt at dusk."],"confidence":0.9,"rationale":"shared subject split into two clauses"}

Sentence: Sofia is patient and resourceful.
{"operator":"AND","conjuncts":["Sofia is patient.","Sofia is resourceful."],"confidence":0.85,"rationale":"adjectival predicates coordinated"}

Sentence: Jonas fixed the hinge and also repaired the door.
{"operator":"AND","conjuncts":["Jonas fixed the hinge.","Jonas repaired the door."],"confidence":0.82,"rationale":"two coordinated verb phrases"}

Sentence: The international scientists gathered data, discussed findings, and stated outcomes.
{"operator":"AND","conjuncts":["The international scientists gathered data.","The international scientists discussed findings.","The international scientists stated outcomes."],"confidence":0.9,"rationale":"three-item coordinated list"}

Sentence: Not only the parser detects errors but also suggests fixes.
{"operator":"AND","conjuncts":["The parser detects errors.","The parser suggests fixes."],"confidence":0.86,"rationale":"not only…but also construction"}

Sentence: Aria and Peter own a spacious and modern studio.
{"operator":"AND","conjuncts":["Aria and Peter own a spacious studio.","Aria and Peter own a modern studio."],"confidence":0.8,"rationale":"shared noun phrase with two adjectives"}

Sentence: Dylan is not an extrovert and prefers quiet study.
{"operator":"AND","conjuncts":["Dylan is not an extrovert.","Dylan prefers quiet study."],"confidence":0.9,"rationale":"negative predicate plus positive predicate"}

Sentence: Priya compiles metrics as well as maintains release scripts.
{"operator":"AND","conjuncts":["Priya compiles metrics.","Priya maintains release scripts."],"confidence":0.84,"rationale":"‘as well as’ signals coordination"}

Sentence: Caleb compiled logs and also generated dashboards.
{"operator":"AND","conjuncts":["Caleb compiled logs.","Caleb generated dashboards."],"confidence":0.85,"rationale":"‘and also’ links two actions"}

Sentence: The rover collected samples and transmitted telemetry.
{"operator":"AND","conjuncts":["The rover collected samples.","The rover transmitted telemetry."],"confidence":0.88,"rationale":"two coordinated verbs with shared subject"}

Sentence: Evelyn drafts proposals but did not review budgets.
{"operator":"AND","conjuncts":["Evelyn drafts proposals.","Evelyn did not review budgets."],"confidence":0.82,"rationale":"additive use of ‘but’"}

Sentence: Marcus documents APIs, tests endpoints, and deploys hotfixes.
{"operator":"AND","conjuncts":["Marcus documents APIs.","Marcus tests endpoints.","Marcus deploys hotfixes."],"confidence":0.9,"rationale":"verb phrase list coordination"}

Sentence: Maya is not patient and nice to Katy.
{"operator":"AND","conjuncts":["Maya is not patient to Katy.","Maya is nice to Katy."],"confidence":0.83,"rationale":"two adjectives linked by ‘and’"}

Sentence: They are not attentive and helpful to the researchers.
{"operator":"AND","conjuncts":["They are not attentive to the researchers.","They are not helpful to the researchers."],"confidence":0.83,"rationale":"two coordinated adjectives with negation"}

Sentence: {SENTENCE}
"""

SPLIT_AND_PROMPT = lambda sentence: escape_and_fill(_SPLIT_AND_BASE, sentence)
