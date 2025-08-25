from __future__ import annotations
"""Aggregator module re-exporting modular prompt templates.

The original monolithic prompt definitions have been moved into
`prompt_templates/` for easier incremental editing by LLM tooling.
This file preserves the public API (`DETECT_OPERATOR_PROMPT`, `get_split_prompt`) to avoid breaking imports.
"""

from .prompt_templates.detect_operator import DETECT_OPERATOR_PROMPT  # type: ignore
from .prompt_templates.split_and import SPLIT_AND_PROMPT  # type: ignore
from .prompt_templates.split_or import SPLIT_OR_PROMPT  # type: ignore
from .prompt_templates.split_implies import SPLIT_IMPLIES_PROMPT  # type: ignore
from .prompt_templates.split_iff import SPLIT_IFF_PROMPT  # type: ignore
from .prompt_templates.split_xor import SPLIT_XOR_PROMPT  # type: ignore
from .prompt_templates.split_not import SPLIT_NOT_PROMPT  # type: ignore
from .prompt_templates.split_nor import SPLIT_NOR_PROMPT  # type: ignore

def get_operator_prompt(sentence: str) -> str:
    return DETECT_OPERATOR_PROMPT(sentence=sentence)

def get_split_prompt(operator: str, sentence: str) -> str:
    op = (operator or "").strip().upper()
    if op == "AND":
        return SPLIT_AND_PROMPT(sentence)
    if op == "OR":
        return SPLIT_OR_PROMPT(sentence)
    if op == "IMPLIES":
        return SPLIT_IMPLIES_PROMPT(sentence)
    if op == "IFF":
        return SPLIT_IFF_PROMPT(sentence)
    if op == "XOR":
        return SPLIT_XOR_PROMPT(sentence)
    if op == "NOT":
        return SPLIT_NOT_PROMPT(sentence)
    if op == "NOR":
        return SPLIT_NOR_PROMPT(sentence)
    return ""


