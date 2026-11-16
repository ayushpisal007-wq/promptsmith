"""Transformer LLM backend (GPT-style decoder) used by the judge, the rewriter and the runner.

Backends (chosen with the LLM_BACKEND env var, default "local"):
  local  - runs an open Hugging Face model on CPU/GPU with transformers + PyTorch.
           Model id from LLM_MODEL (default Qwen/Qwen2.5-1.5B-Instruct). No API key needed.
  hf_api - Hugging Face Inference Providers (needs HF_TOKEN secret). Faster on free CPU Spaces.
  openai - any OpenAI-compatible endpoint, e.g. Groq (OPENAI_BASE_URL + OPENAI_API_KEY secrets).
  mock   - deterministic fake responses, only for unit tests.
Keys are read from environment variables / Space secrets, never from code.
"""
from __future__ import annotations

import os
import re
import threading
import time

DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_lock = threading.Lock()
_local = {}


def backend_name() -> str:
    return os.getenv("LLM_BACKEND", "local").lower()


def model_name() -> str:
    b = backend_name()
    if b == "local":
        return os.getenv("LLM_MODEL", DEFAULT_LOCAL_MODEL)
    if b == "hf_api":
        return os.getenv("HF_API_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    if b == "openai":
        return os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")
    return "mock"


def _load_local():
    if "model" in _local:
        return _local["tok"], _local["model"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = os.getenv("LLM_MODEL", DEFAULT_LOCAL_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    _local.update(tok=tok, model=model, device=device)
    return tok, model


def _chat_local(messages, max_new_tokens, temperature):
    import torch

    tok, model = _load_local()
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(_local["device"]) for k, v in inputs.items()}
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new = out[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip()


def _chat_hf_api(messages, max_new_tokens, temperature):
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=os.environ["HF_TOKEN"])
    r = client.chat_completion(messages=messages, model=model_name(), max_tokens=max_new_tokens,
                               temperature=max(temperature, 0.01))
    return r.choices[0].message.content.strip()


def _chat_openai(messages, max_new_tokens, temperature):
    import requests

    base = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": model_name(), "messages": messages, "max_tokens": max_new_tokens, "temperature": temperature},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _chat_mock(messages, max_new_tokens, temperature):
    """Deterministic stand-in for tests: mimics the judge / rewriter output formats."""
    system = messages[0]["content"].lower()
    user = messages[-1]["content"]
    if "examiner" in system:
        from .rubric import rule_score

        m = re.search(r"PROMPT TO GRADE.*?:\s*(.*)", user, re.S)
        s = rule_score(m.group(1) if m else user)
        lines = [f"{c}: {v.score}" for c, v in s.items()]
        return "\n".join(lines + ["feedback: Add the missing parts of the rubric."])
    if "rewrite" in system:
        return ("You are an experienced teacher. The reader is a beginner who wants a clear overview. "
                f"{user.strip().rstrip('.')} in 3 sections with headings. Keep it under 200 words and avoid jargon.")
    return f"[mock answer to: {user[:60]}]"


_BACKENDS = {"local": _chat_local, "hf_api": _chat_hf_api, "openai": _chat_openai, "mock": _chat_mock}


def chat(messages: list[dict], max_new_tokens: int = 256, temperature: float = 0.0) -> tuple[str, float]:
    """Return (reply_text, latency_seconds)."""
    fn = _BACKENDS[backend_name()]
    t0 = time.perf_counter()
    with _lock:  # one generation at a time on a small CPU host
        text = fn(messages, max_new_tokens, temperature)
    return text, time.perf_counter() - t0
