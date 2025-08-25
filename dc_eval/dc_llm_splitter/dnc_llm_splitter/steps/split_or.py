from __future__ import annotations
from typing import Dict, Any
from ..models.llm_client import BaseLLMClient
from ..models.prompts import get_split_prompt

def split_OR(llm: BaseLLMClient | None, sentence: str, debug: bool = False) -> Dict[str, Any]:
    """LLM OR splitter returning unified schema (children come only from 'disjuncts')."""
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
        if not sentence:
            base["rationale"] = "empty sentence"
        else:
            base["rationale"] = "no llm provided"
        return base

    prompt = get_split_prompt("OR", sentence)
    try:
        out = llm.complete_json(prompt)
    except Exception as e:  # noqa: BLE001
        base["rationale"] = f"llm error: {e}"
        return base
    if not isinstance(out, dict):
        base["rationale"] = "llm returned non-dict response"
        return base
    out.setdefault("operator", "OR")
    # Accept 'disjuncts' (schema key) or 'children' (fallback if LLM uses wrong key)
    children = out.get("disjuncts") or out.get("children")
    if isinstance(children, str):
        children = [children]
    if not isinstance(children, list):
        children = []
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
        "operator": out.get("operator", "OR"),
        "children": children,
        "confidence": conf,
        "rationale": rationale,
        "_method": "llm",
    }

    # attach debug info if requested
    if debug:
        res["_llm_prompt_full"] = prompt
        res["_llm_raw_full"] = getattr(llm, "last_raw", None)

    return res

__all__ = ["split_OR"]
