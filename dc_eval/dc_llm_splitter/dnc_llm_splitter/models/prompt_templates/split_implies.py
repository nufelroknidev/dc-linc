from __future__ import annotations
from .utils import escape_and_fill

_SPLIT_IMPLIES_BASE = """
You are a JSON function. Task: Extract a TOP-LEVEL implication (IF ... THEN ...) if present.
Return STRICT JSON ONLY with schema:
{"operator":"IMPLIES|NONE","children":["<antecedent>","<consequent>"],"confidence":0.0,"rationale":"<10 words>"}

Decision rule:
- Output IMPLIES only if you can identify exactly two substantive clauses forming condition → result via markers (if, when, whenever, only if, unless, provided, assuming, given that, in the event that, etc.).
- If no clear single implication pattern (ambiguous, fragment, multiple disjoint ifs without a unified consequent): return {"operator":"NONE","children":["<original sentence>"],"confidence":<0.3,"rationale":"no implies"}.

Field notes:
- children[0] = antecedent: the condition clause (remove leading markers: if, only if, unless, when, whenever, provided, assuming).
- children[1] = consequent: the result clause (remove leading 'then').
- Always output a full explicit subject in the consequent (do NOT replace with a pronoun like 'It' or 'They').
- Normalize both clauses to end with a period.
- 'A only if B'  => children = ["B.", "A."] (only-if reversal).
- 'Unless A, B'  => children = ["Not A.", "B."] (negate antecedent in text).
- 'B unless A'   => children = ["Not A.", "B."] (trailing unless).
- 'If A B.' without comma or 'then' => children = ["A.", "B."].
- 'B if A.' (clause-final condition) => children = ["A.", "B."].
- If ambiguous or no clear implication pattern: children=[], confidence<0.3.
- Preserve INTERNAL logical structure (AND / OR / EITHER / NEITHER / BOTH) *inside* each clause. There are ALWAYS exactly two children for IMPLIES.
- For patterns like "If X or if Y, then Z" normalize antecedent to "X or Y." (strip repeated 'if').
- For patterns like "If either X or Y, then Z" keep antecedent "Either X or Y." (or "X or Y." if 'either' is stylistic).
- For patterns like "If X and Y, then Z" keep "X and Y." in the antecedent.
- If antecedent contains a NEITHER ... NOR ... structure, keep it verbatim (e.g., "Neither X nor Y."). Do NOT expand into separate sentences.
- If consequent contains conjunction or disjunction, keep it as one clause (e.g., "The system logs an error and halts.").
- Expand pronouns in the consequent if they refer to a clear head noun from antecedent (it, they, this, that) → repeat the noun phrase.
- Do NOT aggressively shorten internal phrases; retain meaningful qualifiers (time/place/objects).

Guidelines:
- Keep original entity names verbatim.
- Capitalize first character of each clause.
- No extra keys besides the schema.

Examples:
Sentence: If a request exceeds quota, then the API returns an error.
{"operator":"IMPLIES","children":["A request exceeds quota.","The API returns an error."],"confidence":0.9,"rationale":"if–then implication"}

Sentence: A dashboard refresh occurs only if caching is disabled.
{"operator":"IMPLIES","children":["Caching is disabled.","A dashboard refresh occurs."],"confidence":0.87,"rationale":"only-if reversal"}

Sentence: Unless the index is corrupt, queries succeed.
{"operator":"IMPLIES","children":["The index is not corrupt.","Queries succeed."],"confidence":0.85,"rationale":"unless → negate antecedent"}

Sentence: Unless the config file is present, the service does not start.
{"operator":"IMPLIES","children":["The config file is not present.","The service does not start."],"confidence":0.84,"rationale":"unless → negate antecedent"}

Sentence: The cache is cleared unless the feature flag is set.
{"operator":"IMPLIES","children":["The feature flag is not set.","The cache is cleared."],"confidence":0.83,"rationale":"unless → negate antecedent"}

Sentence: Notifications are suppressed unless the user opts in.
{"operator":"IMPLIES","children":["The user does not opt in.","Notifications are suppressed."],"confidence":0.82,"rationale":"unless → negate antecedent"}

Sentence: The experiment, if approved, launches next quarter.
{"operator":"IMPLIES","children":["The experiment is approved.","The experiment launches next quarter."],"confidence":0.88,"rationale":"embedded if"}

Sentence: If temperature thresholds are exceeded servers throttle.
{"operator":"IMPLIES","children":["Temperature thresholds are exceeded.","Servers throttle."],"confidence":0.83,"rationale":"if–then (no comma)"}

Sentence: Alerts fire whenever latency spikes.
{"operator":"IMPLIES","children":["Latency spikes.","Alerts fire."],"confidence":0.82,"rationale":"whenever implies"}

Sentence: The rollout, if delayed, will impact Q4 targets.
{"operator":"IMPLIES","children":["The rollout is delayed.","The rollout will impact Q4 targets."],"confidence":0.84,"rationale":"embedded if with subject"}

Sentence: The contract, if signed, becomes active immediately.
{"operator":"IMPLIES","children":["The contract is signed.","The contract becomes active immediately."],"confidence":0.85,"rationale":"embedded if with subject"}

Sentence: If the sensor overheats, it shuts down automatically.
{"operator":"IMPLIES","children":["The sensor overheats.","The sensor shuts down automatically."],"confidence":0.84,"rationale":"pronoun expansion"}

Sentence: If the servers lose power, they restart in safe mode.
{"operator":"IMPLIES","children":["The servers lose power.","The servers restart in safe mode."],"confidence":0.83,"rationale":"plural pronoun expansion"}

Sentence: The system triggers a rollback if it fails.
{"operator":"IMPLIES","children":["The system fails.","The system triggers a rollback."],"confidence":0.83,"rationale":"clause-final condition"}

Sentence: If a node disconnects, this triggers resynchronization.
{"operator":"IMPLIES","children":["A node disconnects.","A node triggers resynchronization."],"confidence":0.82,"rationale":"demonstrative expansion"}

Sentence: If the release is delayed, that impacts revenue.
{"operator":"IMPLIES","children":["The release is delayed.","The release impacts revenue."],"confidence":0.83,"rationale":"that-pronoun expansion"}

Sentence: Unless the license is renewed, access is revoked.
{"operator":"IMPLIES","children":["The license is not renewed.","Access is revoked."],"confidence":0.84,"rationale":"unless → negate antecedent"}

Sentence: If storage is neither encrypted nor replicated, then the platform raises a compliance alarm.
{"operator":"IMPLIES","children":["Storage is neither encrypted nor replicated.","The platform raises a compliance alarm."],"confidence":0.87,"rationale":"if–then with neither–nor"}

Sentence: If a build passes tests and security scanning, then the pipeline deploys and updates release notes.
{"operator":"IMPLIES","children":["A build passes tests and security scanning.","The pipeline deploys and updates release notes."],"confidence":0.88,"rationale":"and in antecedent"}

Sentence: If there is no signal, there is no output.
{"operator":"IMPLIES","children":["There is no signal.","There is no output."],"confidence":0.86,"rationale":"negation kept in clauses"}

Sentence: If the engine is either overheating or starved of oil, then the safety subsystem activates and logs an alert.
{"operator":"IMPLIES","children":["The engine is either overheating or starved of oil.","The safety subsystem activates and logs an alert."],"confidence":0.87,"rationale":"either–or in antecedent"}

Sentence: If power fails, the device does not start.
{"operator":"IMPLIES","children":["Power fails.","The device does not start."],"confidence":0.85,"rationale":"negative consequent"}

Sentence: If a client authenticates with a token or signs in with a password, then the gateway issues a session key.
{"operator":"IMPLIES","children":["A client authenticates with a token or signs in with a password.","The gateway issues a session key."],"confidence":0.89,"rationale":"or in antecedent"}

Sentence: If latency spikes or if the queue overflows, then the autoscaler adds nodes.
{"operator":"IMPLIES","children":["Latency spikes or the queue overflows.","The autoscaler adds nodes."],"confidence":0.88,"rationale":"if–or–if normalised"}

Sentence: {SENTENCE}
"""

SPLIT_IMPLIES_PROMPT = lambda sentence: escape_and_fill(_SPLIT_IMPLIES_BASE, sentence)

