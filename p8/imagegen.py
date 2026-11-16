"""U4 - Text-to-image generation so original and rewritten image prompts can be compared.

IMAGE_BACKEND (the same seed is used for both prompts, so only the wording differs):
  pollinations - free hosted FLUX via image.pollinations.ai, no key (default for the Streamlit deployment)
  hf_api       - Hugging Face Inference Providers text_to_image (needs HF_TOKEN)
  local        - stabilityai/sd-turbo with diffusers + PyTorch, 1 step (used for the Colab evaluation)
"""
from __future__ import annotations

import io
import os
import threading
import time

_lock = threading.Lock()
_cache = {}


def backend() -> str:
    return os.getenv("IMAGE_BACKEND", "local").lower()


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


def _pollinations(prompt, seed, size):
    import requests
    from urllib.parse import quote
    from PIL import Image

    url = f"https://image.pollinations.ai/prompt/{quote(prompt)}"
    params = {"width": size, "height": size, "seed": seed, "nologo": "true",
              "model": os.getenv("POLLINATIONS_MODEL", "flux")}
    for attempt in range(4):  # anonymous tier allows about one request per 15 s
        r = requests.get(url, params=params, timeout=180)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(16)
            continue
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    raise RuntimeError(f"image service busy (HTTP {r.status_code}); try again in a minute")


def generate(prompt: str, seed: int = 42, size: int = 512):
    """Return (PIL.Image, latency_seconds)."""
    t0 = time.perf_counter()
    with _lock:
        b = backend()
        if b == "pollinations":
            img = _pollinations(prompt, seed, size)
        elif b == "hf_api":
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=os.environ["HF_TOKEN"])
            img = client.text_to_image(prompt, model=os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell"),
                                       seed=seed, width=size, height=size)
        else:
            import torch

            pipe = _pipe()
            g = torch.Generator(device=_cache["device"]).manual_seed(seed)
            img = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0,
                       height=size, width=size, generator=g).images[0]
    return img, time.perf_counter() - t0
