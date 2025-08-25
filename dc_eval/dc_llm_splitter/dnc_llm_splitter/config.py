from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml

@dataclass
class Settings:
    strategy: Optional[str] = None
    client: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.1
    debug: bool = False
    # Legacy + new generation controls
    max_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    max_generation: Optional[int] = None
    precision: Optional[str] = None
    # Optional sections
    validators: Optional[Dict[str, Any]] = None
    steps: Optional[Dict[str, Any]] = None

def load_settings(path: str) -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Back-compat: allow only max_tokens, client handles fallback
    if "max_new_tokens" not in cfg and "max_tokens" in cfg:
        pass

    return Settings(**cfg)
