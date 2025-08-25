from __future__ import annotations
from .utils import escape_and_fill

_SPLIT_NOR_BASE = """
You are a JSON function. Task: If the input sentence has a TOP-LEVEL pattern "neither X nor Y" extract the TWO coordinated BASE clauses (positive form, WITHOUT embedding negation) under operator NOR.
Return STRICT JSON ONLY (no prose, no code fences). Schema:
{"operator":"NOR|NONE","children":["...","..."],"confidence":0.0,"rationale":"<10 words>"}

Decision rule:
- Output NOR only for an unambiguous TOP-LEVEL "neither ... nor ..." spanning exactly two coordinated parts.
- Otherwise return: {"operator":"NONE","children":["<sentence>"],"confidence":<0.3,"rationale":"<why>"}.

Field notes:
- children: Two FULL sentences (capitalized, final period) expressing the underlying POSITIVE propositions (e.g., "Alice sings.", "Alice dances."). Do NOT insert "not", "does not", "never", etc. The NOR operator itself implies both are false.
- Recover shared subject so each child stands alone. Duplicate any elided subject.
- Preserve tense/aspect/modality of the positive base form where applicable (e.g., "encrypts", "is clean").
- Do not expand or split nested coordination inside each part.
- Do not fabricate entities or details.
- If the original uses auxiliary verbs (e.g., "does not log"), produce the simple positive clause ("logs").

Examples:
Sentence: Alice neither sings nor dances.
{"operator":"NOR","children":["Alice sings.","Alice dances."],"confidence":0.9,"rationale":"neither nor"}

Sentence: The service neither logs errors nor raises alerts.
{"operator":"NOR","children":["The service logs errors.","The service raises alerts."],"confidence":0.88,"rationale":"neither nor"}

Sentence: The dataset is neither clean nor complete.
{"operator":"NOR","children":["The dataset is clean.","The dataset is complete."],"confidence":0.87,"rationale":"adjectival neither nor"}

Sentence: Storage is neither encrypted nor replicated.
{"operator":"NOR","children":["Storage is encrypted.","Storage is replicated."],"confidence":0.9,"rationale":"neither nor"}

Sentence: The API neither authenticates users nor validates payloads.
{"operator":"NOR","children":["The API authenticates users.","The API validates payloads."],"confidence":0.88,"rationale":"neither nor"}

Sentence: It is neither sunny nor windy.
{"operator":"NOR","children":["It is sunny.","It is windy."],"confidence":0.86,"rationale":"neither nor"}

Sentence: They neither confirmed nor denied the claim.
{"operator":"NOR","children":["They confirmed the claim.","They denied the claim."],"confidence":0.82,"rationale":"neither nor"}

Sentence: The protocol neither encrypts nor compresses traffic.
{"operator":"NOR","children":["The protocol encrypts traffic.","The protocol compresses traffic."],"confidence":0.88,"rationale":"neither nor"}

Sentence: The rover neither reboots nor transmits telemetry.
{"operator":"NOR","children":["The rover reboots.","The rover transmits telemetry."],"confidence":0.85,"rationale":"neither nor"}

Sentence: {SENTENCE}
"""

SPLIT_NOR_PROMPT = lambda sentence: escape_and_fill(_SPLIT_NOR_BASE, sentence)
