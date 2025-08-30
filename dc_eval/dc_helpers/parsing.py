"""Parsing and evaluation helpers for dcneurosymbolic generations."""
from __future__ import annotations
from typing import List
from dc_eval.tasks.utils import evaluate


def parse_and_eval_generation(generation_text: str, prompt_text: str, stop_words: List[str], error_token: str) -> str:
    try:
        gen = generation_text
        if gen.startswith(prompt_text):
            gen = gen[len(prompt_text):].strip()
        for sw in stop_words or []:
            if sw in gen:
                gen = gen.split(sw)[0].strip()
        flag = "FOL:"
        parses = [line.replace(flag, "").strip() for line in gen.split("\n") if flag in line]
        if not parses:
            return error_token
        premises_fol, conclusion_fol = parses[:-1], parses[-1]
        resp = evaluate(premises_fol, conclusion_fol)
        if resp not in ["True", "False", "Uncertain"]:
            return error_token
        return resp
    except Exception:
        print(f"Error: {generation_text}")
        # print("Error in parsing and/or evaluating LLM output")
        return error_token
