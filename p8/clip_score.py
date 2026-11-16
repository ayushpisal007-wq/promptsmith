"""U3 - CLIP score for image prompts: how well an image matches a text (cosine x 100, floored at 0)."""
from __future__ import annotations

import os

_cache = {}


def _load():
    if "model" not in _cache:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        name = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
        dtype = torch.bfloat16 if os.getenv("LOW_MEMORY") == "1" else torch.float32
        _cache["model"] = CLIPModel.from_pretrained(name, dtype=dtype).eval()
        _cache["proc"] = CLIPProcessor.from_pretrained(name)
    return _cache["model"], _cache["proc"]


def clip_score(image, text: str) -> float:
    import torch

    model, proc = _load()
    inputs = proc(text=[text], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.inference_mode():
        out = model(**inputs)
    img, txt = out.image_embeds.float(), out.text_embeds.float()
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    return round(max(100.0 * float((img * txt).sum()), 0.0), 2)
