"""U3 - BERT prompt-quality classifier (discriminative model) with an attention view.

Loads the fine-tuned model from BERT_MODEL (a local folder or a Hugging Face Hub repo id).
Default: ./models/bert-prompt-quality  (created by train_bert.py or the Colab notebook).
"""
from __future__ import annotations

import os

LABELS = ["weak", "average", "strong"]
_cache = {}


def model_path() -> str:
    return os.getenv("BERT_MODEL", "models/bert-prompt-quality")


def available() -> bool:
    """True if a local model folder exists, or BERT_MODEL points at a Hub repo id."""
    return os.path.isdir(model_path()) or "BERT_MODEL" in os.environ


def _load():
    if "model" in _cache:
        return _cache["tok"], _cache["model"]
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    p = model_path()
    tok = AutoTokenizer.from_pretrained(p)
    # eager attention so the attention weights can be returned for the explanation view
    dtype = torch.bfloat16 if os.getenv("LOW_MEMORY") == "1" else torch.float32  # half the RAM on small hosts
    model = AutoModelForSequenceClassification.from_pretrained(p, attn_implementation="eager", dtype=dtype).eval()
    torch.set_grad_enabled(False)
    _cache.update(tok=tok, model=model)
    return tok, model


def predict(prompt: str) -> dict:
    """Return class probabilities and per-word attention from [CLS] (last layer, mean over heads)."""
    import torch

    tok, model = _load()
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=128)
    out = model(**enc, output_attentions=True)
    probs = torch.softmax(out.logits[0].float(), -1).tolist()
    id2label = model.config.id2label
    probs = {id2label.get(i, id2label.get(str(i), LABELS[i])): float(p) for i, p in enumerate(probs)}

    att = out.attentions[-1][0].float().mean(0)[0]  # [seq] attention from CLS
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    words, weights = [], []
    for t, w in zip(tokens, att.tolist()):
        if t in (tok.cls_token, tok.sep_token, tok.pad_token):
            continue
        if t.startswith("##") and words:
            words[-1] += t[2:]
            weights[-1] = max(weights[-1], w)
        else:
            words.append(t)
            weights.append(w)
    mx = max(weights) if weights else 1.0
    attention = [(w, round(a / mx, 3)) for w, a in zip(words, weights)]
    return {"probs": probs, "label": max(probs, key=probs.get), "attention": attention}
