from __future__ import annotations
from typing import Dict, Any
from ..models.llm_client import BaseLLMClient
from ..models.prompts import get_split_prompt

def split_AND(llm: BaseLLMClient | None, sentence: str, debug: bool = False) -> Dict[str, Any]:
    """LLM AND splitter returning unified schema.

    Unified schema keys:
      operator: "AND"
      children: list[str]  (was conjuncts)
      confidence: float
      rationale: str
      _method: 'llm' | 'none'
    """
    def _make_base(err_msg: str = "") -> Dict[str, Any]:
        return {
            "operator": "NONE",
            "children": [sentence] if sentence else [""],
            "confidence": 1.0,
            "rationale": err_msg,
            "_method": "none",
        }

    base: Dict[str, Any] = _make_base("")
    if not sentence or llm is None:
        # return base (NONE) for missing inputs
        if not sentence: base["rationale"] = "empty sentence"
        else: base["rationale"] = "no llm provided"
        return base
    
    prompt = get_split_prompt("AND", sentence)
    try:
        out = llm.complete_json(prompt)
    except Exception as e:  # noqa: BLE001
        base["rationale"] = f"llm error: {e}"
        return base
    if not isinstance(out, dict):
        base["rationale"] = "llm returned non-dict response"
        return base
    out.setdefault("operator", "AND")
    # Accept legacy 'conjuncts'
    if "children" not in out and "conjuncts" in out:
        out["children"] = out.pop("conjuncts")

    # Normalize children
    children = out.get("children")
    if isinstance(children, str):
        children = [children]
    if not isinstance(children, list):
        children = []
    # Ensure children are strings
    children = [str(c) for c in children]

    # Confidence
    conf = out.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0

    rationale = out.get("rationale", "")
    if rationale is None:
        rationale = ""
    rationale = str(rationale)

    res: Dict[str, Any] = {
        "operator": out.get("operator", "AND"),
        "children": children,
        "confidence": conf,
        "rationale": rationale,
        "_method": "llm",
    }

    if debug:
        res["_llm_prompt_full"] = prompt
        res["_llm_raw_full"] = getattr(llm, "last_raw", None)

    return res

__all__ = ["split_AND"]
