from __future__ import annotations
from typing import Dict, Any, List

from .config import load_settings
from .models.llm_client import MistralHFClient, BaseLLMClient, SharedHFClient
from .steps import detect_operator, split_AND, split_OR, split_IMPLIES, split_XOR, split_IFF, split_NOT, split_NOR
from .models.prompts import get_split_prompt  # still used indirectly if needed


def get_strategy(cfg) -> str:
    # allow either top-level 'strategy' or steps.strategy in YAML
    s = (getattr(cfg, 'strategy', None) or '').strip().lower() if getattr(cfg, 'strategy', None) else ''
    if not s:
        steps_cfg = getattr(cfg, 'steps', None) or {}
        s = str(steps_cfg.get('strategy', '')).strip().lower() if isinstance(steps_cfg, dict) else ''
    return s or 'auto'


def make_llm(cfg, *, shared_model=None, shared_tokenizer=None, device=None) -> BaseLLMClient:
    client_key = (cfg.client or "").lower().strip()
    if client_key in {"mistral", "mistralai", "hf", "huggingface"}:
        if shared_model is not None and shared_tokenizer is not None:
            print(
                f"[splitter] Using SharedHFClient (reusing in-memory model) temp={cfg.temperature} "
                f"max_new_tokens={getattr(cfg,'max_new_tokens', None) or getattr(cfg,'max_tokens', None)}"
            )
            return SharedHFClient(
                model=shared_model,
                tokenizer=shared_tokenizer,
                temperature=cfg.temperature,
                max_tokens=(cfg.max_new_tokens or (cfg.max_tokens or 512)),
                max_new_tokens=cfg.max_new_tokens,
                top_p=getattr(cfg, "top_p", None),
                top_k=getattr(cfg, "top_k", None),
                seed=getattr(cfg, "seed", None),
                max_generation=getattr(cfg, "max_generation", None),
                precision=getattr(cfg, "precision", None),
                device=device,
            )
        else:
            print(
                f"[splitter] Using MistralHFClient model={cfg.model} temp={cfg.temperature} "
                f"max_new_tokens={getattr(cfg,'max_new_tokens', None) or getattr(cfg,'max_tokens', None)} "
                f"precision={getattr(cfg,'precision', None)}"
            )
            return MistralHFClient(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=(cfg.max_new_tokens or (cfg.max_tokens or 512)),
                max_new_tokens=cfg.max_new_tokens,
                top_p=getattr(cfg, "top_p", None),
                top_k=getattr(cfg, "top_k", None),
                seed=getattr(cfg, "seed", None),
                max_generation=getattr(cfg, "max_generation", None),
                precision=getattr(cfg, "precision", None),
            )
    raise ValueError("[splitter] Invalid client specified. To use Mistral set client: mistral in config.yaml")

def run_pipeline_sentence(llm: BaseLLMClient, sentence: str, strategy: str = "auto", debug: bool = False) -> Dict[str, Any]:
    log: Dict[str, Any] = {"original": sentence, "strategy": strategy}
    # Step 1: detect top-level operator
    # Force LLM-based detection regardless of configured strategy per user request
    # Use purely rule-based detection as requested
    det = detect_operator(llm, sentence, strategy=strategy, debug=debug)
    log["detector"] = det
    log["operator"] = det.get("operator", "NONE")
    op = (log["operator"] or "").strip().upper()
    if op == "AND":
        log["split"] = split_AND(llm, sentence, debug=debug)
    elif op == "OR":
        log["split"] = split_OR(llm, sentence, debug=debug)
    elif op == "IMPLIES":
        log["split"] = split_IMPLIES(llm, sentence, debug=debug)
    elif op == "IFF":
        log["split"] = split_IFF(llm, sentence, debug=debug)
    elif op == "XOR":
        log["split"] = split_XOR(llm, sentence, debug=debug)
    elif op == "NOR":
        log["split"] = split_NOR(llm, sentence, debug=debug)
    elif op == "NOT":
        log["split"] = split_NOT(llm, sentence, debug=debug)
    elif op == "NONE":
        log["split"] = {"operator": "NONE", "children": [sentence]}
    return log

def run_on_sentences(cfg_path: str, sentences: List[str]) -> List[Dict[str, Any]]:
    cfg = load_settings(cfg_path)
    llm = make_llm(cfg)
    strategy = get_strategy(cfg)
    debug = bool(getattr(cfg, 'debug', False))
    results = []
    for s in sentences:
        results.append(run_pipeline_sentence(llm, s, strategy=strategy, debug=debug))
    return results
