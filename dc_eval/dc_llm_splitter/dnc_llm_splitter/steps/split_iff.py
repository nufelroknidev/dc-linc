from __future__ import annotations
from typing import Dict, Any
from ..models.llm_client import BaseLLMClient
from ..models.prompts import get_split_prompt

def split_IFF(llm: BaseLLMClient | None, sentence: str, debug: bool = False) -> Dict[str, Any]:
    """LLM IFF splitter aligned with other split_* functions.

    Success => {operator: IFF, children:[A,B], confidence, rationale, _method: llm}
    Failure => base NONE structure: original sentence as sole child.
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

    prompt = get_split_prompt("IFF", s)
    try:
        out = llm.complete_json(prompt)
    except Exception as e:  # noqa: BLE001
        return _make_base(f"llm error: {e}")
    if not isinstance(out, dict):
        return _make_base("llm returned non-dict response")
    out.setdefault("operator", "IFF")
    # Gather children.
    children: list[str] = []
    if isinstance(out.get("children"), list):
        children = [str(c).strip() for c in out.get("children") if isinstance(c, str)]

    # Normalize to exactly two.
    if len(children) > 2:
        children = children[:2]

    # Confidence / rationale
    conf = out.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    rationale = out.get("rationale", "") or ""

    if len(children) == 2 and all(children):
        res: Dict[str, Any] = {
            "operator": "IFF",
            "children": children,
            "confidence": conf,
            "rationale": rationale,
            "_method": "llm",
        }
    else:
        res = _make_base("missing sides" if children else "parse failure")

    if debug:
        res["_llm_prompt_full"] = prompt
        res["_llm_raw_full"] = getattr(llm, "last_raw", None)
    return res

__all__ = ["split_IFF"]
