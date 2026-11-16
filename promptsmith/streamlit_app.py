"""P8 - Prompt Grader & Rewriter (Streamlit Community Cloud version).

Ayush Pisal | Roll No. 08 | Generative AI CA mini-project
Secrets (Streamlit -> App settings -> Secrets):  GROQ_API_KEY = "gsk_..."
"""
from __future__ import annotations

import html
import os

import streamlit as st

# ---- configuration: Streamlit secrets -> environment, small-host defaults (before importing p8)
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:  # no secrets file locally
    pass
if os.getenv("GROQ_API_KEY"):
    os.environ.setdefault("LLM_BACKEND", "openai")
    os.environ.setdefault("OPENAI_API_KEY", os.environ["GROQ_API_KEY"])
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
os.environ.setdefault("BERT_MODEL", "ayushpisal/p8-bert-prompt-quality")
os.environ.setdefault("IMAGE_BACKEND", "pollinations")
os.environ.setdefault("LOW_MEMORY", "1")

import pandas as pd  # noqa: E402

from p8 import classifier, llm  # noqa: E402
from p8.pipeline import analyze, run_both  # noqa: E402
from p8.rubric import CRITERIA, RUBRIC  # noqa: E402

st.set_page_config(page_title="P8 · Prompt Grader & Rewriter", page_icon="📝", layout="wide")

EXAMPLES = {
    "explain machine learning": "auto",
    "write email to boss about leave": "auto",
    "write a job ad for a young energetic salesman, applicants can contact rahul.k@gmail.com or 9876543210": "auto",
    "a cat": "image",
    "city at night": "image",
}


def safety_md(s: dict) -> str:
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


def scores_df(before: dict, after: dict) -> pd.DataFrame:
    rows = []
    for c in CRITERIA:
        jb = (before.get("judge") or {}).get(c)
        ja = (after.get("judge") or {}).get(c)
        rows.append([c, before["rule"][c], jb, after["rule"][c], ja])
    rows.append(["TOTAL /10", before["rule_total"], before.get("judge_total"), after["rule_total"], after.get("judge_total")])
    return pd.DataFrame(rows, columns=["criterion", "rules: before", "LLM judge: before", "rules: after", "LLM judge: after"])


def attention_html(att: list) -> str:
    spans = [f'<span style="background: rgba(255,140,0,{w:.2f}); padding: 2px 3px; margin: 1px; '
             f'border-radius: 3px; line-height: 2">{html.escape(t)}</span>' for t, w in att]
    return "<div>" + " ".join(spans) + "</div>"


# ------------------------------------------------------------------ layout
st.title("📝 P8 · Prompt Grader & Rewriter")
st.caption("Score a prompt on a 5-part rubric, rewrite it, then compare the outputs of both versions · "
           "Ayush Pisal · Roll No. 08 · Generative AI CA")

with st.sidebar:
    st.header("Settings")
    few_shot = st.checkbox("Few-shot judge (off = zero-shot)", value=True)
    judge_after = st.checkbox("LLM-judge the rewrite too", value=True)
    st.divider()
    st.markdown(f"**LLM:** `{llm.model_name()}` ({llm.backend_name()})  \n"
                f"**Classifier:** `{os.environ['BERT_MODEL']}`  \n"
                f"**Images:** {os.environ['IMAGE_BACKEND']}")
    if llm.backend_name() == "local":
        st.warning("No GROQ_API_KEY secret found, so the app is trying to run a local LLM, which is too big for this host. "
                   "Add the key under App settings → Secrets.")

tab_main, tab_results, tab_about = st.tabs(["Grade & Rewrite", "Results: 5 weak prompts", "About & model card"])

with tab_main:
    col_in, col_ex = st.columns([3, 2])
    with col_ex:
        st.markdown("**Examples**")
        for ex, k in EXAMPLES.items():
            if st.button(ex[:60] + ("…" if len(ex) > 60 else ""), key=f"ex_{ex}", width="stretch"):
                st.session_state["prompt"] = ex
                st.session_state["kind"] = k
    with col_in:
        prompt = st.text_area("Your prompt", key="prompt", height=110, placeholder="e.g. explain machine learning")
        kind = st.radio("Prompt type", ["auto", "text", "image"], key="kind", horizontal=True)
        grade = st.button("1 · Grade & rewrite", type="primary")

    if grade:
        if not (prompt or "").strip():
            st.error("Please enter a prompt.")
        else:
            with st.spinner("Checking safety, grading, rewriting and re-grading…"):
                try:
                    st.session_state["result"] = analyze(prompt, kind, few_shot=few_shot, judge_rewritten=judge_after)
                    st.session_state["rewritten"] = st.session_state["result"]["rewritten"]
                    st.session_state.pop("run", None)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    r = st.session_state.get("result")
    if r:
        b, a = r["before"], r["after"]
        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("Responsible-AI checks (U5)")
            st.markdown(safety_md(r["safety"]))
            if not r["after_safety"]["pii_pass"] or not r["after_safety"]["bias_pass"]:
                st.warning("The rewritten prompt still triggers a safety check:\n\n" + safety_md(r["after_safety"]))
        with c2:
            st.subheader("Rubric scores, 0-2 per criterion (U1, U2)")
            st.dataframe(scores_df(b, a), hide_index=True, width="stretch")
            if b.get("judge_error"):
                st.error(f"LLM judge failed: {b['judge_error']}")
            else:
                st.markdown(f"**Judge feedback:** {b.get('judge_feedback', '')}")

        st.subheader("BERT quality classifier (U3)")
        if "bert_probs" in b:
            c3, c4 = st.columns([1, 2])
            with c3:
                probs = pd.DataFrame({"before": b["bert_probs"], "after": a.get("bert_probs", {})}).reindex(
                    ["weak", "average", "strong"])
                st.dataframe(probs.style.format("{:.3f}"), width="stretch")
                st.caption(f"Prediction: **{b['bert_label']} → {a.get('bert_label', '-')}**")
            with c4:
                st.markdown("Attention from [CLS] on the original prompt (darker = more weight)")
                st.markdown(attention_html(b["bert_attention"]), unsafe_allow_html=True)
        else:
            st.info(b.get("bert_error", "BERT classifier is not available."))

        st.subheader("Rewritten prompt")
        rewritten = st.text_area("You can edit it before running", key="rewritten", height=130)
        if st.button("2 · Run original vs rewritten (U4)"):
            with st.spinner("Running both prompts… (images can take up to a minute)"):
                try:
                    st.session_state["run"] = run_both(prompt, rewritten, r["kind"],
                                                       max_new_tokens=int(os.getenv("RUN_MAX_TOKENS", "400")))
                except Exception as e:
                    st.error(f"Run failed: {e}")
        out = st.session_state.get("run")
        if out:
            ca, cb = st.columns(2)
            if out["kind"] == "image":
                ca.image(out["image_original"], caption="Original prompt")
                cb.image(out["image_rewritten"], caption="Rewritten prompt")
                st.markdown(
                    f"**CLIP score vs your original intent:** {out['clip_original_vs_intent']} → {out['clip_rewritten_vs_intent']}  \n"
                    f"**CLIP score vs the detailed rewritten prompt:** original image {out['clip_original_vs_rewritten_text']} → "
                    f"rewritten image {out['clip_rewritten_vs_own']}  \nSame seed (42) for both images.")
            else:
                ca.markdown("**Output: original prompt**")
                ca.markdown(out["output_original"])
                cb.markdown("**Output: rewritten prompt**")
                cb.markdown(out["output_rewritten"])
                st.caption(f"Words {out['words_original']} → {out['words_rewritten']} · "
                           f"latency {out['latency_original']:.1f}s / {out['latency_rewritten']:.1f}s")

with tab_results:
    st.markdown("Before/after results for the 5 weak prompts from the evaluation run "
                "(Colab T4, Qwen2.5-1.5B-Instruct + sd-turbo; `evaluate.py`).")
    if os.path.exists("results/before_after.csv"):
        df = pd.read_csv("results/before_after.csv")
        st.dataframe(df, hide_index=True, width="stretch")
        imgs = [p for p in ["W4_original", "W4_rewritten", "W5_original", "W5_rewritten"]
                if os.path.exists(f"results/outputs/{p}.png")]
        if imgs:
            cols = st.columns(len(imgs))
            for c, p in zip(cols, imgs):
                c.image(f"results/outputs/{p}.png", caption=p.replace("_", " "))
    if os.path.exists("results/summary.md"):
        with st.expander("Full results (summary.md)"):
            st.markdown(open("results/summary.md", encoding="utf-8").read())

with tab_about:
    st.markdown(f"""
### How it works
1. **U5 guardrail:** personal data is redacted before any model sees the prompt, and biased wording is flagged. The rewrite is checked again.
2. **U1 rubric:** role, context, task, format and constraints are each scored 0-2 by transparent keyword rules.
3. **U2 LLM judge:** `{llm.model_name()}` grades the same rubric, with 3 graded few-shot examples or zero-shot.
4. **U3 BERT + CLIP:** a fine-tuned `bert-base-uncased` predicts weak / average / strong, and the attention view shows which words it used. CLIP scores how well images match the text.
5. **U4 run both:** text prompts run on the LLM; image prompts are rendered with the same seed.

**Generative vs discriminative:** the LLM and the image model *generate* new text and images, while BERT *classifies* quality.

**Deployment note:** the evaluation in the report ran Qwen2.5-1.5B and sd-turbo locally on a GPU. This free host has about 1-2.7 GB of RAM, so the deployed app calls a hosted LLM (Groq) and a hosted image model (Pollinations FLUX). BERT and CLIP still run inside the app with PyTorch.

**Licences:** BERT (Apache-2.0), CLIP (MIT), gpt-oss (Apache-2.0); hosted images are for non-commercial use. The code is MIT-licensed. Don't paste confidential data.
""")
    st.table(pd.DataFrame([[c, RUBRIC[c]["text"], RUBRIC[c]["image"]] for c in CRITERIA],
                          columns=["criterion", "text prompt", "image prompt"]))
