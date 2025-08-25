from __future__ import annotations
import re, json
from typing import Dict, Any

from ..models.llm_client import BaseLLMClient
from ..models.detectors import detect_operator_rule_based
from ..models.prompts import get_operator_prompt
# from .operator_core import (
#     _CANONICAL,
#     _norm_operator,
#     _consistency_adjust,
#     _obvious_cue_operator,
# )

def _extract_last_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    objs = []
    for m in re.finditer(r"\{[\s\S]*?\}", text):
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            objs.append(obj)
        except Exception:
            continue
    return objs[-1] if objs else {}

def detect_operator_rule(sentence: str) -> Dict[str, Any]:
    rb = detect_operator_rule_based(sentence)
    return {
        "operator": rb.get("operator", "NONE"),
        "method": "rule",
        "confidence": float(rb.get("confidence", 0.5)),
        "evidence": rb.get("evidence", ""),
    }

def detect_operator_llm(llm: BaseLLMClient | None, sentence: str, debug: bool = False) -> Dict[str, Any]:
    if llm is None:
        return {"operator": "NONE", "method": "llm", "confidence": 0.0, "rationale": "no client"}
    # Phase 1: minimal JSON
    prompt = get_operator_prompt(sentence)
    def _try_json(prompt: str) -> Dict[str, Any]:
        try:
            out = llm.complete_json(prompt) or {}
        except Exception:
            raw = llm.complete(prompt)
            out = _extract_last_json(raw)
        return out if isinstance(out, dict) else {}
    out = _try_json(prompt)
    if debug:
        print(f"[detect_operator_llm] raw output: {out}")

    # Normalise fields
    op_raw = str(out.get("operator", "")).strip().upper()
    # Extend accepted set to include IMPLIES, XOR, IFF, NOT for downstream splitting.
    if op_raw not in {"AND", "OR", "IMPLIES", "XOR", "IFF", "NOT"}:
        op_raw = "NONE"
    try:
        conf = float(out.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    # bound confidence
    if conf < 0:
        conf = 0.0
    if conf > 1:
        conf = 1.0

    result = {
        "operator": op_raw,
        "method": "llm",
        "confidence": conf,
        "rationale": out.get("rationale") or out.get("reason") or "",
        "raw": out,
    }

    # If the model produced nothing usable, optionally fall back to rule-based as a guard.
    if op_raw == "NONE" and not out:
        guard = detect_operator_rule(sentence)
        guard["method"] = "llm+ruleguard"
        if debug:
            print("[detect_operator_llm] empty LLM output; falling back to rule-based guard")
        return guard
    return result

def detect_operator(llm: BaseLLMClient | None, sentence: str, strategy: str = "auto", debug: bool = False) -> Dict[str, Any]:
    # Auto: prefer LLM when available, otherwise use rule-based detector.
    if strategy == "auto":
        if llm is None:
            return detect_operator_rule(sentence)
        try:
            return detect_operator_llm(llm, sentence, debug=debug)
        except Exception:
            return detect_operator_rule(sentence)
    if strategy == "rule":
        return detect_operator_rule(sentence)
    if strategy == "llm":
        return detect_operator_llm(llm, sentence, debug=debug)

__all__ = ["detect_operator", "detect_operator_llm", "detect_operator_rule"]