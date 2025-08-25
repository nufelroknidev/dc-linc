from __future__ import annotations
from typing import Dict, Any
from ..models.llm_client import BaseLLMClient
from ..models.prompts import get_split_prompt

def split_NOR(llm: BaseLLMClient | None, sentence: str, debug: bool = False) -> Dict[str, Any]:
    """LLM-only NOR splitter.

    Expected LLM JSON (via prompt): may use either 'children' or 'disjuncts' for the two negated clauses.
    Fallback returns NONE with original sentence (mirrors other splitter patterns).
    """
    s = (sentence or "").strip()

    def _make_base(msg: str = "") -> Dict[str, Any]:
        return {
            "operator": "NONE",
            "children": [s] if s else [""],
            "confidence": 1.0,
            "rationale": msg,
            "_method": "none",
        }

    if not s:
        return _make_base("empty sentence")
    if llm is None:
        return _make_base("no llm provided")

    prompt = get_split_prompt("NOR", s)
    try:
        out = llm.complete_json(prompt)
    except Exception as e:  # noqa: BLE001
        return _make_base(f"llm error: {e}")

    if not isinstance(out, dict):
        return _make_base("llm returned non-dict response")

    # Accept either key style from prompt.
    raw_children = []
    if isinstance(out.get("children"), list):
        raw_children = [c for c in out.get("children") if isinstance(c, str)]
    elif isinstance(out.get("disjuncts"), list):
        raw_children = [c for c in out.get("disjuncts") if isinstance(c, str)]

    # Normalize count; require exactly 2 for NOR else fallback NONE
    if len(raw_children) != 2:
        return _make_base("expected two children")

    # Confidence
    conf = out.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    rationale = out.get("rationale", "") or "neither nor"

    res: Dict[str, Any] = {
        "operator": "NOR",
        "children": [c.rstrip('.') + '.' for c in raw_children],
        "confidence": conf,
        "rationale": rationale,
        "_method": "llm",
    }

    if debug:
        res["_llm_prompt_full"] = prompt
        res["_llm_raw_full"] = getattr(llm, "last_raw", None)
    return res

__all__ = ["split_NOR"]
