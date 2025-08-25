from __future__ import annotations
from typing import Dict, Any
from ..models.llm_client import BaseLLMClient
from ..models.prompts import get_split_prompt

def split_IMPLIES(llm: BaseLLMClient | None, sentence: str, debug: bool = False) -> Dict[str, Any]:
    """IMPLIES splitter: returns children = [antecedent, consequent] or fallback NONE error base.

    Error base: {operator:NONE, children:[sentence], confidence:1.0, rationale:<msg>, _method:none}
    """
    def _make_base(err: str = "") -> Dict[str, Any]:
        return {
            "operator": "NONE",
            "children": [sentence] if sentence else [""],
            "confidence": 1.0,
            "rationale": err,
            "_method": "none",
        }
    base = _make_base()
    if not sentence or llm is None:
        base["rationale"] = "empty sentence" if not sentence else "no llm provided"
        return base
    prompt = get_split_prompt("IMPLIES", sentence)
    try:
        out = llm.complete_json(prompt)
    except Exception as e:  # noqa: BLE001
        base["rationale"] = f"llm error: {e}"
        return base
    if not isinstance(out, dict):
        base["rationale"] = "llm returned non-dict response"
        return base
    out.setdefault("operator", "IMPLIES")
    # Gather candidate antecedent/consequent from either children or explicit keys
    antecedent = ""
    consequent = ""
    if isinstance(out.get("children"), list) and out.get("children"):
        ch = [c for c in out.get("children") if isinstance(c, str)]
        if ch:
            antecedent = ch[0]
        if len(ch) > 1:
            consequent = ch[1]
    else:
        if isinstance(out.get("antecedent"), str):
            antecedent = out.get("antecedent")
        if isinstance(out.get("consequent"), str):
            consequent = out.get("consequent")
    children = [c for c in (antecedent, consequent) if c]
    # Normalize confidence / rationale
    conf = out.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    rationale = out.get("rationale", "") or ""
    res: Dict[str, Any] = {
        "operator": "IMPLIES" if len(children) == 2 else "NONE" if not children else "IMPLIES",
        "children": children if len(children) == 2 else [sentence],
        "confidence": conf if len(children) == 2 else 1.0 if not children else conf,
        "rationale": rationale if len(children) == 2 else ("missing antecedent/consequent" if children else "parse failure"),
        "_method": "llm" if len(children) == 2 else "none",
    }
    if debug:
        res["_llm_prompt_full"] = prompt
        res["_llm_raw_full"] = getattr(llm, "last_raw", None)
    return res

__all__ = ["split_IMPLIES"]
