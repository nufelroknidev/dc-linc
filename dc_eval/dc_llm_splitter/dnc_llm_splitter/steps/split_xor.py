from __future__ import annotations
from typing import Dict, Any
from ..models.llm_client import BaseLLMClient
from ..models.prompts import get_split_prompt


def split_XOR(llm: BaseLLMClient | None, sentence: str, debug: bool = False) -> Dict[str, Any]:
    """XOR splitter: expects exactly two alternative clauses; else returns NONE base."""
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
    prompt = get_split_prompt("XOR", sentence)
    try:
        out = llm.complete_json(prompt)
    except Exception as e:  # noqa: BLE001
        base["rationale"] = f"llm error: {e}"
        return base
    if not isinstance(out, dict):
        base["rationale"] = "llm returned non-dict response"
        return base
    out.setdefault("operator", "XOR")
    # Collect children from possible keys
    raw_children: list[str] = []
    if isinstance(out.get("children"), list):
        raw_children = [c for c in out.get("children") if isinstance(c, str)]
    else:
        for key in ("disjuncts", "options", "clauses"):
            if isinstance(out.get(key), list):
                raw_children = [c for c in out.get(key) if isinstance(c, str)]
                break
    # Ensure max two
    raw_children = raw_children[:2]
    conf = out.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    rationale = out.get("rationale", "") or ""
    if len(raw_children) == 2:
        res = {
            "operator": "XOR",
            "children": raw_children,
            "confidence": conf,
            "rationale": rationale,
            "_method": "llm",
        }
    else:
        res = {
            "operator": "NONE",
            "children": [sentence],
            "confidence": 1.0,
            "rationale": "not binary" if len(raw_children)==1 else "parse failure",
            "_method": "none",
        }
    if debug:
        res["_llm_prompt_full"] = prompt
        res["_llm_raw_full"] = getattr(llm, "last_raw", None)
    return res
