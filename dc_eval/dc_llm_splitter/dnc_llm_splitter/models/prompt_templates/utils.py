from __future__ import annotations

def escape_and_fill(template: str, sentence: str) -> str:
    """Double braces to avoid .format collisions and insert sentence."""
    return template.replace('{SENTENCE}', sentence)
