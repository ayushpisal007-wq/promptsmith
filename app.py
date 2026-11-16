"""P8 - Prompt Grader & Rewriter (Gradio app for Hugging Face Spaces).

Ayush Pisal | Roll No. 08 | Generative AI CA mini-project
"""
from __future__ import annotations

import os
import threading

import gradio as gr
import pandas as pd

from p8 import classifier, llm
from p8.pipeline import analyze, run_both
from p8.rubric import CRITERIA, RUBRIC

TITLE = "P8 · Prompt Grader & Rewriter"
EXAMPLES = [
    ["explain machine learning", "auto"],
    ["write email to boss about leave", "auto"],
    ["write a job ad for a young energetic salesman, applicants can contact rahul.k@gmail.com or 9876543210", "auto"],
    ["a cat", "image"],
    ["city at night", "image"],
]


def _safety_md(s: dict) -> str:
    lines = []
    if s["pii"]:
        kinds = ", ".join(sorted({p["type"] for p in s["pii"]}))
        lines.append(f"🔒 **Personal data found and redacted before any model saw it:** {kinds}")
    else:
        lines.append("✅ **Personal data:** none found")
    if s["bias"]:
        lines.append("⚠️ **Biased wording:**")
        lines += [f"- “{b['term']}” ({b['category']}) → try *{b['suggestion']}*" for b in s["bias"]]
    else:
        lines.append("✅ **Biased wording:** none found")
    return "\n".join(lines)


def _table(before: dict, after: dict) -> pd.DataFrame:
    rows = []
    for c in CRITERIA:
        jb = (before.get("judge") or {}).get(c)
        ja = (after.get("judge") or {}).get(c)
        rows.append([c, before["rule"][c], "-" if jb is None else jb, after["rule"][c], "-" if ja is None else ja])
    rows.append(["TOTAL /10", before["rule_total"], before.get("judge_total", "-"),
                 after["rule_total"], after.get("judge_total", "-")])
    return pd.DataFrame(rows, columns=["criterion", "rules: before", "LLM judge: before", "rules: after", "LLM judge: after"])


def _summary_md(r: dict) -> str:
    b, a = r["before"], r["after"]
    md = [f"**Prompt type:** {r['kind']}  ·  **LLM:** `{llm.model_name()}`"]
    if b.get("judge_error"):
        md.append(f"LLM judge failed: `{b['judge_error']}`. The rule-based scores are still shown.")
    else:
        md.append(f"**Judge feedback:** {b.get('judge_feedback', '')}")
        md.append(f"**Band:** {b.get('judge_band', b['rule_band'])} → {a.get('judge_band', a['rule_band'])}")
    if "bert_label" in b:
        md.append(f"**BERT classifier:** {b['bert_label']} (P(strong)={b['bert_probs'].get('strong', 0):.2f}) → "
                  f"{a.get('bert_label', '-')} (P(strong)={a.get('bert_probs', {}).get('strong', 0):.2f})")
    elif not classifier.available():
        md.append("_The BERT classifier isn't loaded. Set the `BERT_MODEL` secret to your Hub repo, or add `models/bert-prompt-quality`._")
    if not r["after_safety"]["pii_pass"] or not r["after_safety"]["bias_pass"]:
        md.append("⚠️ The rewritten prompt still triggers a safety check. Review it before using it.")
    return "\n\n".join(md)


def do_grade(prompt, kind, few_shot, judge_after):
    try:
        r = analyze(prompt, kind, few_shot=few_shot, judge_rewritten=judge_after)
    except ValueError as e:
        raise gr.Error(str(e))
    b = r["before"]
    probs = b.get("bert_probs") or {}
    attention = b.get("bert_attention") or []
    return (_safety_md(r["safety"]), _table(b, r["after"]), _summary_md(r), probs,
            attention, r["rewritten"], r["kind"])


def do_run(prompt, rewritten, kind):
    if not rewritten:
        raise gr.Error("Grade the prompt first, so a rewritten version exists.")
    kind = kind or "text"
    r = run_both(prompt, rewritten, kind, max_new_tokens=int(os.getenv("RUN_MAX_TOKENS", "200")))
    if r["kind"] == "image":
        md = (f"**CLIP score vs your original intent:** original {r['clip_original_vs_intent']} → "
              f"rewritten {r['clip_rewritten_vs_intent']}  \n"
              f"**CLIP score vs the detailed rewritten prompt:** original image {r['clip_original_vs_rewritten_text']} → "
              f"rewritten image {r['clip_rewritten_vs_own']}  \n"
              f"Same seed for both images · {r['latency_original']:.1f}s / {r['latency_rewritten']:.1f}s")
        return "", "", r["image_original"], r["image_rewritten"], md
    md = (f"Words: {r['words_original']} → {r['words_rewritten']} · "
          f"latency {r['latency_original']:.1f}s / {r['latency_rewritten']:.1f}s")
    return r["output_original"], r["output_rewritten"], None, None, md


def load_batch_results():
    path = "results/before_after.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({"info": ["Run `python evaluate.py` (or the Colab notebook) to create results/before_after.csv"]})


ABOUT = f"""
### How it works
1. **U5 guardrail:** personal data (email, phone, Aadhaar, PAN, card, IP, address, name) is redacted, and biased wording is flagged.
2. **U1 rubric:** five criteria scored 0-2 each (total out of 10) by transparent keyword rules.
3. **U2 LLM judge:** `{llm.model_name()}` grades the same rubric, either zero-shot or with 3 graded few-shot examples.
4. **U3 BERT:** a fine-tuned `bert-base-uncased` classifier predicts weak / average / strong. The attention view shows which words it relied on. For image prompts, a **CLIP score** measures how well the image matches the text.
5. **Rewrite:** the LLM adds whatever parts of the rubric are missing, using generic assumptions and never inventing personal facts.
6. **U4 run both:** text prompts run through the LLM; image prompts run through `stabilityai/sd-turbo` with the same seed.

**Generative vs discriminative:** the LLM and sd-turbo *generate* new text and images (they model p(x)), while BERT *discriminates* between quality classes (it models p(y|x)).

| criterion | text prompt | image prompt |
|---|---|---|
""" + "\n".join(f"| {c} | {RUBRIC[c]['text']} | {RUBRIC[c]['image']} |" for c in CRITERIA) + """

**Licences:** Qwen2.5 (Apache-2.0), BERT (Apache-2.0), CLIP (MIT), sd-turbo (Stability AI Non-Commercial Research Community licence: academic, non-commercial use only). The app code is MIT-licensed. Don't paste confidential data.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\nScore a prompt on a 5-part rubric, rewrite it, then compare the outputs of both versions. "
                    "*Ayush Pisal · Roll No. 08 · Generative AI CA*")
        kind_state = gr.State("text")
        with gr.Tab("Grade & Rewrite"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(label="Your prompt", lines=4, placeholder="e.g. explain machine learning")
                    with gr.Row():
                        kind = gr.Radio(["auto", "text", "image"], value="auto", label="Prompt type")
                        few_shot = gr.Checkbox(value=True, label="Few-shot judge (off = zero-shot)")
                        judge_after = gr.Checkbox(value=True, label="LLM-judge the rewrite too")
                    grade_btn = gr.Button("1 · Grade & rewrite", variant="primary")
                    gr.Examples(EXAMPLES, inputs=[prompt, kind])
                with gr.Column(scale=2):
                    safety = gr.Markdown(label="Responsible-AI checks")
                    summary = gr.Markdown()
            scores = gr.Dataframe(label="Rubric scores (0-2 per criterion)", interactive=False)
            with gr.Row():
                bert = gr.Label(label="BERT quality prediction (original prompt)", num_top_classes=3)
                attn = gr.HighlightedText(label="BERT attention from [CLS] (darker = more weight)")
            rewritten = gr.Textbox(label="Rewritten prompt (you can edit it)", lines=4)
            run_btn = gr.Button("2 · Run original vs rewritten", variant="secondary")
            run_md = gr.Markdown()
            with gr.Row():
                out_a = gr.Textbox(label="Output: original prompt", lines=12)
                out_b = gr.Textbox(label="Output: rewritten prompt", lines=12)
            with gr.Row():
                img_a = gr.Image(label="Image: original prompt", type="pil")
                img_b = gr.Image(label="Image: rewritten prompt", type="pil")

            grade_btn.click(do_grade, [prompt, kind, few_shot, judge_after],
                            [safety, scores, summary, bert, attn, rewritten, kind_state])
            run_btn.click(do_run, [prompt, rewritten, kind_state], [out_a, out_b, img_a, img_b, run_md])
        with gr.Tab("Results: 5 weak prompts"):
            gr.Markdown("Before/after scores for the 5 weak prompts, created by `evaluate.py`.")
            table = gr.Dataframe(value=load_batch_results, interactive=False)
            gr.Button("Reload").click(load_batch_results, None, table)
        with gr.Tab("About & model card"):
            gr.Markdown(ABOUT)
    return demo


def _warmup():
    """Load models in the background so the first click is faster."""
    try:
        llm.chat([{"role": "user", "content": "hi"}], max_new_tokens=1)
        if classifier.available():
            classifier.predict("warm up")
    except Exception as e:
        print("warm-up skipped:", e)


if __name__ == "__main__":
    if os.getenv("WARMUP", "1") == "1":
        threading.Thread(target=_warmup, daemon=True).start()
    build().queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", theme=gr.themes.Soft())
