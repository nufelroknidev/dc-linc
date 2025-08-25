from __future__ import annotations
from typing import Dict, Any, Optional
import json, re


class BaseLLMClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 800,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        max_generation: Optional[int] = None,
        precision: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_new_tokens if max_new_tokens is not None else max_tokens
        self._has_max_new_tokens = max_new_tokens is not None
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.max_generation = max_generation
        self.precision = precision
        self.last_raw: Optional[str] = None

    def complete_json(self, prompt: str) -> Dict[str, Any]:  # interface
        raise NotImplementedError

def _extract_first_json_obj(text: str) -> Dict[str, Any]:
    """Attempt to robustly extract the first JSON object from a model response.

    Handles arrow bullets (→), leading commentary, and truncation at unmatched braces.
    Returns empty dict on failure.
    """
    if not text:
        return {}
    # Common leading arrow or unicode bullets
    cleaned = text
    # Strip markdown fences if present
    if cleaned.strip().startswith('```'):
        cleaned = re.sub(r'^```[a-zA-Z0-9_-]*\n?', '', cleaned.strip())
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()
    # Drop any echoed 'Sentence:' markers that appear before JSON
    # Keep only portion after last 'Sentence:' if multiple
    if 'Sentence:' in cleaned:
        parts = cleaned.split('Sentence:')
        # Heuristic: JSON is usually after the last occurrence
        cleaned = parts[-1]
    cleaned = cleaned.lstrip("\n >\t").lstrip("→ ").strip()
    # Fix common double-brace pattern {{"operator": ... }}
    if cleaned.startswith('{{"operator"'):
        # Replace only the first '{{' and the last '}}'
        cleaned = '{' + cleaned[2:]
        if cleaned.endswith('}}'):
            cleaned = cleaned[:-2] + '}'
    # Heuristic: find first '{' and last '}' that yields parseable json when slicing
    first = cleaned.find('{')
    last = cleaned.rfind('}')
    if first == -1 or last == -1 or last <= first:
        return {}
    candidate = cleaned[first:last+1]
    # Secondary fix: if candidate still starts with '{{', collapse
    if candidate.startswith('{{') and candidate.endswith('}}'):
        candidate = candidate[1:-1]
    # Remove obvious trailing junk after final JSON (sometimes model repeats prompt)
    # Try incremental shrink if braces mismatch
    for end in range(len(candidate), 0, -1):
        frag = candidate[:end]
        try:
            return json.loads(frag)
        except Exception:
            continue
    return {}


class MistralHFClient(BaseLLMClient):
    _pipe = None
    _tokenizer = None

    def _lazy_init(self) -> None:
        if self.__class__._pipe is not None:
            return
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # type: ignore
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("transformers not installed; install transformers torch") from e

        dtype = None
        if self.precision:
            p = str(self.precision).lower().strip()
            try:
                import torch  # type: ignore
                if p in {"fp32", "float32"}:
                    dtype = torch.float32
                elif p in {"fp16", "float16"}:
                    dtype = torch.float16
                elif p in {"bf16", "bfloat16"}:
                    dtype = torch.bfloat16
            except Exception:
                pass

        tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        mdl_kwargs = {"trust_remote_code": True}
        if dtype is not None:
            mdl_kwargs["torch_dtype"] = dtype
        mdl = AutoModelForCausalLM.from_pretrained(self.model, **mdl_kwargs)
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            tok.pad_token_id = tok.eos_token_id
        try:
            ml = getattr(tok, "model_max_length", None)
            if ml is None or (isinstance(ml, int) and ml > 100000):
                tok.model_max_length = self.max_generation or 8192
        except Exception:
            pass
        self.__class__._pipe = pipeline("text-generation", model=mdl, tokenizer=tok)  # type: ignore
        self.__class__._tokenizer = tok

    def _gen(self, prompt: str, *, stop_at_json: bool = False) -> str:
        self._lazy_init()
        tok = self.__class__._tokenizer
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            messages = [
                {"role": "user", "content": prompt.strip()},
            ]
            try:
                full_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except ValueError as e:
                if "Invalid format specifier" in str(e):
                    sanitized = prompt.replace('{', '{{').replace('}', '}}')
                    full_prompt = f"{sanitized}\n"
                else:
                    raise
        else:
            full_prompt = f"{prompt.strip()}\n"
        gen_kwargs: Dict[str, Any] = {
            "do_sample": self.temperature > 0,
            "temperature": self.temperature,
            "eos_token_id": getattr(tok, "eos_token_id", None),
            "pad_token_id": getattr(tok, "pad_token_id", None),
            "return_full_text": False,
            "truncation": True,
        }
        if self._has_max_new_tokens:
            gen_kwargs["max_new_tokens"] = self.max_tokens
        elif self.max_generation is not None:
            gen_kwargs["max_length"] = self.max_generation
        else:
            gen_kwargs["max_new_tokens"] = self.max_tokens
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p
        if self.top_k is not None and self.top_k > 0:
            gen_kwargs["top_k"] = self.top_k
        # Seed only once per class instance lifecycle to avoid identical outputs on every call.
        if self.seed is not None and not getattr(self.__class__, "_seeded", False):
            try:
                import random, numpy as np, torch  # type: ignore
                random.seed(self.seed)
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():  # type: ignore
                    torch.cuda.manual_seed_all(self.seed)  # type: ignore
                self.__class__._seeded = True  # mark so we don't reseed each generation
            except Exception:
                pass
        # Optional early stop once the first balanced top-level JSON object is complete
        if stop_at_json:
            try:  # soft-fail if transformers API changes
                from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore

                class _FirstJSONStop(StoppingCriteria):
                    def __init__(self, tokenizer, prompt_token_len: int):
                        self.tok = tokenizer
                        self.prompt_token_len = prompt_token_len
                        self.started = False
                        self.depth = 0
                        self._processed_chars = 0  # number of generated chars already scanned

                    def __call__(self, input_ids, scores, **kwargs):  # type: ignore
                        # input_ids[0] contains prompt + generated
                        if input_ids is None:
                            return False
                        if len(input_ids) == 0:
                            return False
                        full_seq = input_ids[0]
                        gen_ids = full_seq[self.prompt_token_len:]
                        # gen_ids is a tensor; check length safely
                        try:
                            if gen_ids.numel() == 0:  # type: ignore[attr-defined]
                                return False
                        except Exception:
                            pass
                        # Convert to python list for decode stability
                        try:
                            gen_list = gen_ids.tolist()  # type: ignore[attr-defined]
                        except Exception:
                            gen_list = gen_ids
                        try:
                            gen_txt = self.tok.decode(gen_list, skip_special_tokens=True)
                        except Exception:
                            return False
                        # Only scan newly appended characters since last call
                        new_segment = gen_txt[self._processed_chars:]
                        if not new_segment:
                            return False
                        for ch in new_segment:
                            if not self.started:
                                if ch == '{':
                                    self.started = True
                                    self.depth = 1
                            else:
                                if ch == '{':
                                    self.depth += 1
                                elif ch == '}':
                                    self.depth -= 1
                                    if self.depth == 0:
                                        # Update processed count to avoid duplicate triggers (not strictly needed once True)
                                        self._processed_chars += len(new_segment)
                                        return True
                        # Update processed count after scanning
                        self._processed_chars += len(new_segment)
                        return False

                prompt_ids = tok(full_prompt).input_ids
                # Some tokenizers return list[int]; some list[list[int]]
                if isinstance(prompt_ids[0], list):  # type: ignore[index]
                    prompt_token_len = len(prompt_ids[0])  # type: ignore[arg-type]
                else:
                    prompt_token_len = len(prompt_ids)  # type: ignore[arg-type]
                sc = _FirstJSONStop(tok, prompt_token_len)
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([sc])
            except Exception:
                pass  # fallback: no early stop

        out = self.__class__._pipe(full_prompt, **gen_kwargs)
        return out[0]["generated_text"].strip()

    def complete_json(self, prompt: str) -> Dict[str, Any]:
        # Use early stopping tailored for first JSON object
        content = self._gen(prompt, stop_at_json=True)
        # print(f"[MistralHFClient] raw output: {content}")
        self.last_raw = content
        # Fast path
        try:
            return json.loads(content)
        except Exception:
            pass
        # Regex fallback
        m = re.search(r"(\{[\s\S]*?\})", content, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # Heuristic salvage
        return _extract_first_json_obj(content)

    def complete(self, prompt: str) -> str:
        return self._gen(prompt)

class SharedHFClient(BaseLLMClient):
    """Adapter that reuses an existing Transformers model+tokenizer.

    This avoids loading a second copy of the LLM for the splitter pipeline.
    The provided model is used as-is (no device move). Caller is responsible
    for device placement and dtype. Generation runs with standard HF kwargs.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        max_generation: Optional[int] = None,
        precision: Optional[str] = None,
        device=None,
    ) -> None:
        super().__init__(
            model=getattr(model, "name_or_path", "hf-shared"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_generation=max_generation,
            precision=precision,
        )
        self._model = model
        self._tok = tokenizer
        self._device = device  # optional explicit device tensor placement

        # Ensure tokenizer has pad_token_id
        if getattr(self._tok, "pad_token_id", None) is None and getattr(self._tok, "eos_token_id", None) is not None:
            try:
                self._tok.pad_token_id = self._tok.eos_token_id
            except Exception:
                pass

    def _prepare_prompt(self, prompt: str) -> str:
        tok = self._tok
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            messages = [
                {"role": "user", "content": prompt.strip()},
            ]
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except ValueError as e:
                if "Invalid format specifier" in str(e):
                    sanitized = prompt.replace('{', '{{').replace('}', '}}')
                    return f"{sanitized}\n"
                raise
        return f"{prompt.strip()}\n"

    def _gen(self, prompt: str, *, stop_at_json: bool = False) -> str:
        import torch  # local import to avoid hard dep at import time
        mdl = self._model
        tok = self._tok
        mdl.eval()

        full_prompt = self._prepare_prompt(prompt)
        enc = tok(full_prompt, return_tensors="pt")
        input_ids = enc.input_ids
        attn = enc.get("attention_mask", None)

        # Place on the same device as model parameters unless device provided
        try:
            if self._device is not None:
                dev = self._device
            else:
                dev = next(mdl.parameters()).device
            input_ids = input_ids.to(dev)
            if attn is not None:
                attn = attn.to(dev)
        except Exception:
            pass

        gen_kwargs: Dict[str, Any] = {
            "do_sample": self.temperature > 0,
            "temperature": self.temperature,
            "eos_token_id": getattr(tok, "eos_token_id", None),
            "pad_token_id": getattr(tok, "pad_token_id", None),
        }
        if self._has_max_new_tokens:
            gen_kwargs["max_new_tokens"] = self.max_tokens
        elif self.max_generation is not None:
            gen_kwargs["max_length"] = self.max_generation
        else:
            gen_kwargs["max_new_tokens"] = self.max_tokens
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p
        if self.top_k is not None and self.top_k > 0:
            gen_kwargs["top_k"] = self.top_k

        # Optional: early stop on first balanced JSON object
        if stop_at_json:
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore

                class _FirstJSONStop(StoppingCriteria):
                    def __init__(self, tokenizer, prompt_token_len: int):
                        self.tok = tokenizer
                        self.prompt_token_len = prompt_token_len
                        self.started = False
                        self.depth = 0
                        self._processed_chars = 0

                    def __call__(self, input_ids, scores, **kwargs):  # type: ignore
                        if input_ids is None or len(input_ids) == 0:
                            return False
                        full_seq = input_ids[0]
                        gen_ids = full_seq[self.prompt_token_len:]
                        try:
                            if gen_ids.numel() == 0:  # type: ignore[attr-defined]
                                return False
                            gen_txt = self.tok.decode(gen_ids.tolist(), skip_special_tokens=True)  # type: ignore[attr-defined]
                        except Exception:
                            return False
                        new_segment = gen_txt[self._processed_chars:]
                        if not new_segment:
                            return False
                        for ch in new_segment:
                            if not self.started:
                                if ch == '{':
                                    self.started = True
                                    self.depth = 1
                            else:
                                if ch == '{':
                                    self.depth += 1
                                elif ch == '}':
                                    self.depth -= 1
                                    if self.depth == 0:
                                        self._processed_chars += len(new_segment)
                                        return True
                        self._processed_chars += len(new_segment)
                        return False

                prompt_ids = tok(full_prompt).input_ids
                prompt_len = len(prompt_ids[0]) if isinstance(prompt_ids[0], list) else len(prompt_ids)
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_FirstJSONStop(tok, prompt_len)])
            except Exception:
                pass

        with torch.no_grad():
            out = mdl.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
        # When return_full_text=False isn't supported here, strip the prompt manually
        try:
            # Decode only the generated continuation
            cont = out[0][input_ids.shape[1]:]
            text = tok.decode(cont, skip_special_tokens=True)
        except Exception:
            text = tok.decode(out[0], skip_special_tokens=True)
        return text.strip()

    def complete_json(self, prompt: str) -> Dict[str, Any]:
        content = self._gen(prompt, stop_at_json=True)
        self.last_raw = content
        try:
            return json.loads(content)
        except Exception:
            pass
        m = re.search(r"(\{[\s\S]*?\})", content, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return _extract_first_json_obj(content)

    def complete(self, prompt: str) -> str:
        return self._gen(prompt, stop_at_json=False)

__all__ = ["BaseLLMClient", "MistralHFClient", "SharedHFClient"]
