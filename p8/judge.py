"""U2 - LLM-as-judge grading (zero-shot vs few-shot) and the prompt rewriter.

The judge sees the rubric and (in few-shot mode) three graded example prompts, then
returns one score per criterion in a fixed line format that is parsed with regex.
"""
from __future__ import annotations

import re

from . import llm
from .rubric import CRITERIA, RUBRIC

FEW_SHOT = {
    "text": [
        ("explain python",
         "role: 0\ncontext: 0\ntask: 1\nformat: 0\nconstraints: 0\n"
         "feedback: Clear verb but no role, audience, format or limits; say who it is for and how long."),
        ("You are a maths tutor. Explain derivatives for my class 12 exam using 3 bullet points.",
         "role: 2\ncontext: 1\ntask: 1\nformat: 2\nconstraints: 0\n"
         "feedback: Good role and format; add constraints such as length, tone or what to avoid."),
        ("Act as a senior HR manager. I'm a fresher applying to startups and want to stand out. "
         "Write a cover letter that highlights my two internships and one measurable result. "
         "Format it as a 4-paragraph email with a subject line. Keep it under 200 words and do not invent facts.",
         "role: 2\ncontext: 2\ntask: 2\nformat: 2\nconstraints: 2\n"
         "feedback: Complete prompt: specific role, context, scoped task, structure and limits."),
    ],
    "image": [
        ("a dog",
         "role: 0\ncontext: 0\ntask: 0\nformat: 0\nconstraints: 0\n"
         "feedback: Only a vague subject; add a style, setting, composition and exclusions."),
        ("oil painting of a fishing boat at sunset",
         "role: 1\ncontext: 1\ntask: 1\nformat: 0\nconstraints: 0\n"
         "feedback: Style and setting given; add composition (shot type, aspect) and exclusions."),
        ("cinematic photograph, 35mm film, of an old man reading a newspaper in a busy Mumbai cafe, warm morning light, "
         "close-up, shallow depth of field, no text, muted tones",
         "role: 2\ncontext: 2\ntask: 2\nformat: 2\nconstraints: 2\n"
         "feedback: Complete image prompt with style, scene, subject, composition and exclusions."),
    ],
}


def _system(kind: str) -> str:
    lines = "\n".join(f"- {c}: {RUBRIC[c][kind]}" for c in CRITERIA)
    what = "text-to-image prompt" if kind == "image" else "prompt for a chat LLM"
    return (
        f"You are a strict prompt-engineering examiner. Grade the student's {what} on five criteria, "
        f"each 0, 1 or 2:\n{lines}\n\n"
        "Grade only what is written; do not reward things that are missing. Reply in exactly this format and nothing else:\n"
        "role: <0-2>\ncontext: <0-2>\ntask: <0-2>\nformat: <0-2>\nconstraints: <0-2>\nfeedback: <one sentence>"
    )


def judge_messages(prompt: str, kind: str = "text", few_shot: bool = True) -> list[dict]:
    msgs = [{"role": "system", "content": _system(kind)}]
    if few_shot:
        for ex, graded in FEW_SHOT[kind]:
            msgs.append({"role": "user", "content": f"PROMPT TO GRADE:\n{ex}"})
            msgs.append({"role": "assistant", "content": graded})
    msgs.append({"role": "user", "content": f"PROMPT TO GRADE:\n{prompt}"})
    return msgs


def parse_judgement(text: str) -> dict:
    scores = {}
    for c in CRITERIA:
        m = re.search(rf"{c}\s*[:=\-]\s*\**\s*([0-2])\b", text, re.I)
        scores[c] = int(m.group(1)) if m else None
    fb = re.search(r"feedback\s*[:=\-]\s*(.+)", text, re.I)
    return {"scores": scores, "feedback": fb.group(1).strip() if fb else "", "parsed_ok": all(v is not None for v in scores.values())}


def judge(prompt: str, kind: str = "text", few_shot: bool = True) -> dict:
    raw, latency = llm.chat(judge_messages(prompt, kind, few_shot), max_new_tokens=90)
    res = parse_judgement(raw)
    res.update(raw=raw, latency=latency, total=sum(v or 0 for v in res["scores"].values()))
    return res


# ---------------------------------------------------------------- rewriter
REWRITE_TEXT = (
    "You rewrite weak prompts into strong prompts for a chat LLM. Keep the user's original intent. "
    "The rewritten prompt must contain: (1) a specific role for the model, (2) context - who it is for and why, "
    "(3) a precise task with scope, (4) the output format and structure, (5) constraints such as length, tone and "
    "what to avoid. If a detail is unknown, state a sensible generic assumption (e.g. 'Assume the reader is a "
    "beginner') - never invent names, numbers, contact details or personal facts. Replace biased wording with "
    "neutral, inclusive wording and keep redaction tags such as [EMAIL] or [PHONE] exactly as they are. "
    "Reply with only the rewritten prompt, written to the model in second person, under 120 words."
)
REWRITE_IMAGE = (
    "You rewrite weak text-to-image prompts into strong ones. Keep the original subject. Write ONE comma-separated "
    "prompt with: art style or medium, the subject with attributes and action, setting and lighting, composition "
    "(shot type, lens or aspect), and exclusions such as 'no text, no watermark'. Avoid living artists' names and "
    "brand logos. Reply with only the prompt, at most 55 words."
)


def _clean(text: str) -> str:
    text = re.sub(r"^```\w*|```$", "", text.strip()).strip()
    text = re.sub(r"^(?:\*\*)?(?:rewritten|improved|new) prompt(?:\*\*)?\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip()


def rewrite(prompt: str, kind: str = "text", missing: list[str] | None = None) -> dict:
    system = REWRITE_IMAGE if kind == "image" else REWRITE_TEXT
    hint = f"\nThe weakest criteria are: {', '.join(missing)}." if missing else ""
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Weak prompt:\n{prompt}{hint}"}]
    raw, latency = llm.chat(msgs, max_new_tokens=110 if kind == "image" else 200)
    return {"prompt": _clean(raw), "latency": latency}


def run_text_prompt(prompt: str, max_new_tokens: int = 300) -> dict:
    """U4 - actually run a prompt so original and rewritten outputs can be compared."""
    out, latency = llm.chat([{"role": "user", "content": prompt}], max_new_tokens=max_new_tokens, temperature=0.0)
    return {"output": out, "latency": latency, "words": len(out.split())}
