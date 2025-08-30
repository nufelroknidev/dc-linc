from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


from dc_eval.dc_llm_splitter.dnc_llm_splitter.io_utils import read_conclusions  # noqa: F401 (may be used elsewhere)
from dc_eval.dc_llm_splitter.dnc_llm_splitter.run_pipeline_enhanced import run_pipeline_sentence
from dc_eval.dc_llm_splitter.dnc_llm_splitter.config import load_settings  # noqa: F401 (left for backwards compat)


def normalize_split_result(r: Dict[str, Any], sentence: Optional[str] = None) -> tuple:
    """Normalize various legacy shapes returned by run_pipeline_sentence.

    Updated: negation flags removed from the pipeline. We now only return (operator, children).
    Any legacy fields related to negation are ignored.
    """
    op = r.get('operator') or r.get('detector', {}).get('operator')
    split_block = r.get('split', {}) or {}
    # unify legacy shapes for child list
    if 'children' not in split_block:
        for k in ('conjuncts', 'disjuncts', 'clauses', 'options'):
            if k in split_block and isinstance(split_block.get(k), list):
                split_block['children'] = split_block[k]
                break
    return op, split_block.get('children', [sentence])


def run_split(conclusion_text: str, cfg, llm, strategy: str = "rule") -> tuple:
    """Execute the splitter on `conclusion_text` and return (operator, children).

    Negation handling has been removed. For backward compatibility, callers expecting
    a third element should be updated. This function now returns only two values.
    """
    r = run_pipeline_sentence(llm, sentence=conclusion_text, strategy=strategy)
    op, children = normalize_split_result(r, conclusion_text)
    return op, children
