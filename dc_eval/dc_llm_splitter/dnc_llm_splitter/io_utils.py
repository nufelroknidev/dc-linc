from __future__ import annotations
from typing import List
from pathlib import Path

def read_conclusions(path: str | Path) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Conclusion:"):
                sent = line.split(":", 1)[1].strip()
                lines.append(sent)
    return lines
