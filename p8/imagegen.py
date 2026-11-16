"""U4 - Text-to-image generation so original and rewritten image prompts can be compared.

IMAGE_BACKEND=local (default): stabilityai/sd-turbo with diffusers + PyTorch, 1 step, 512x512 (works on CPU).
IMAGE_BACKEND=hf_api: Hugging Face Inference Providers text_to_image (needs HF_TOKEN).
The same seed is used for both prompts so only the wording differs.
"""
from __future__ import annotations

import os
import threading
import time

_lock = threading.Lock()
_cache = {}


def _pipe():
    if "pipe" not in _cache:
        import torch
        from diffusers import AutoPipelineForText2Image

        name = os.getenv("IMAGE_MODEL", "stabilityai/sd-turbo")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(name, dtype=dtype).to(device)
        pipe.set_progress_bar_config(disable=True)
        _cache.update(pipe=pipe, device=device)
    return _cache["pipe"]


def generate(prompt: str, seed: int = 42, size: int = 512):
    """Return (PIL.Image, latency_seconds)."""
    t0 = time.perf_counter()
    with _lock:
        if os.getenv("IMAGE_BACKEND", "local") == "hf_api":
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=os.environ["HF_TOKEN"])
            img = client.text_to_image(prompt, model=os.getenv("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
                                       seed=seed, width=size, height=size)
        else:
            import torch

            pipe = _pipe()
            g = torch.Generator(device=_cache["device"]).manual_seed(seed)
            img = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0,
                       height=size, width=size, generator=g).images[0]
    return img, time.perf_counter() - t0
