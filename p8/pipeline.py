"""End-to-end flow shared by the Gradio app and evaluate.py.

safety (U5) -> rule rubric (U1) -> LLM judge (U2) -> BERT (U3) -> rewrite -> re-score
-> run original vs rewritten (U4: text via LLM, image via sd-turbo + CLIP score).
"""
from __future__ import annotations

from . import classifier, judge
from .rubric import CRITERIA, band, detect_kind, rule_score, total
from .safety import safety_report


def score_prompt(prompt: str, kind: str, few_shot: bool = True, use_llm: bool = True) -> dict:
    rule = rule_score(prompt, kind)
    out = {
        "rule": {c: v.score for c, v in rule.items()},
        "rule_evidence": {c: v.evidence for c, v in rule.items()},
        "rule_total": total(rule),
    }
    out["rule_band"] = band(out["rule_total"])
    if use_llm:
        try:
            j = judge.judge(prompt, kind, few_shot)
            out.update(judge=j["scores"], judge_total=j["total"], judge_band=band(j["total"]),
                       judge_feedback=j["feedback"], judge_latency=j["latency"], judge_parsed=j["parsed_ok"])
        except Exception as e:  # keep the app usable if the LLM backend fails
            out.update(judge=None, judge_error=f"{type(e).__name__}: {e}")
    if classifier.available():
        try:
            b = classifier.predict(prompt)
            out.update(bert_probs=b["probs"], bert_label=b["label"], bert_attention=b["attention"])
        except Exception as e:
            out.update(bert_error=f"{type(e).__name__}: {e}")
    return out


def analyze(prompt: str, kind: str = "auto", few_shot: bool = True, judge_rewritten: bool = True) -> dict:
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Please enter a prompt.")
    kind = detect_kind(prompt) if kind in (None, "", "auto") else kind
    safety = safety_report(prompt)
    clean = safety["redacted"]  # guardrail: only redacted text reaches the models
    before = score_prompt(clean, kind, few_shot)
    scores = before.get("judge") or before["rule"]
    missing = [c for c in CRITERIA if (scores.get(c) or 0) < 2]
    rw = judge.rewrite(clean, kind, missing)
    after = score_prompt(rw["prompt"], kind, few_shot, use_llm=judge_rewritten)
    after_safety = safety_report(rw["prompt"])
    return {"kind": kind, "original": prompt, "clean": clean, "safety": safety, "before": before,
            "rewritten": rw["prompt"], "rewrite_latency": rw["latency"], "after": after,
            "after_safety": after_safety, "missing": missing}


def run_both(original: str, rewritten: str, kind: str, seed: int = 42, max_new_tokens: int = 300) -> dict:
    """U4 - run original and rewritten prompts side by side."""
    original = safety_report(original)["redacted"]
    if kind == "image":
        from .clip_score import clip_score
        from .imagegen import generate

        img_a, t_a = generate(original, seed=seed)
        img_b, t_b = generate(rewritten, seed=seed)
        return {
            "kind": "image", "image_original": img_a, "image_rewritten": img_b,
            "latency_original": t_a, "latency_rewritten": t_b,
            # intent alignment: both images vs the user's ORIGINAL request
            "clip_original_vs_intent": clip_score(img_a, original),
            "clip_rewritten_vs_intent": clip_score(img_b, original),
            # prompt adherence: rewritten image vs its own detailed prompt
            "clip_rewritten_vs_own": clip_score(img_b, rewritten),
            "clip_original_vs_rewritten_text": clip_score(img_a, rewritten),
        }
    a = judge.run_text_prompt(original, max_new_tokens)
    b = judge.run_text_prompt(rewritten, max_new_tokens)
    return {"kind": "text", "output_original": a["output"], "output_rewritten": b["output"],
            "latency_original": a["latency"], "latency_rewritten": b["latency"],
            "words_original": a["words"], "words_rewritten": b["words"]}
