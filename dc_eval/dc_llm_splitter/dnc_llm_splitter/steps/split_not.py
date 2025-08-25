from __future__ import annotations
from typing import Dict, Any
from ..models.llm_client import BaseLLMClient
from ..models.prompts import get_split_prompt

def split_NOT(llm: BaseLLMClient | None, sentence: str, debug: bool = False) -> Dict[str, Any]:
    """LLM NOT splitter matching style of split_OR / split_IMPLIES.

    Expects prompt to yield positive canonical clause (no internal negation tokens) in `children` (list length 1).
    Fallback returns NONE base with original sentence.
    """
    s = (sentence or "").strip()
    def _make_base(err: str = "") -> Dict[str, Any]:
        return {
            "operator": "NONE",
            "children": [s] if s else [""],
            "confidence": 1.0,
            "rationale": err,
            "_method": "none",
        }
    if not s:
        return _make_base("empty sentence")
    if llm is None:
        return _make_base("no llm provided")

    prompt = get_split_prompt("NOT", s)
    try:
        out = llm.complete_json(prompt)
    except Exception as e:  # noqa: BLE001
        return _make_base(f"llm error: {e}")
    if not isinstance(out, dict):
        return _make_base("llm returned non-dict response")

    # Normalize children (accept alt keys clause/statement)
    if "children" not in out and isinstance(out.get("clause"), str):
        out["children"] = [out.get("clause")]
    if isinstance(out.get("children"), str):
        out["children"] = [out.get("children")]
    children_raw = out.get("children") if isinstance(out.get("children"), list) else []
    children = [c for c in children_raw if isinstance(c, str) and c.strip()]
    if len(children) != 1:
        return _make_base("expected single child")

    # Confidence / rationale
    conf = out.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    rationale = out.get("rationale", "") or "top-level negation"
    child = children[0].strip()
    if child and not child.endswith('.'):
        child += '.'
    res: Dict[str, Any] = {
        "operator": "NOT",
        "children": [child],
        "confidence": conf,
        "rationale": rationale,
        "_method": "llm",
    }
    if debug:
        res["_llm_prompt_full"] = prompt
        res["_llm_raw_full"] = getattr(llm, "last_raw", None)
    return res

__all__ = ["split_NOT"]
