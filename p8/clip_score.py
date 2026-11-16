"""U3 - CLIP score for image prompts: how well an image matches a text (cosine x 100, floored at 0)."""
from __future__ import annotations

import os

_cache = {}


def _load():
    if "model" not in _cache:
        from transformers import CLIPModel, CLIPProcessor

        name = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
        _cache["model"] = CLIPModel.from_pretrained(name).eval()
        _cache["proc"] = CLIPProcessor.from_pretrained(name)
    return _cache["model"], _cache["proc"]


def clip_score(image, text: str) -> float:
    import torch

    model, proc = _load()
    inputs = proc(text=[text], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.inference_mode():
        out = model(**inputs)
    img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return round(max(100.0 * float((img * txt).sum()), 0.0), 2)
