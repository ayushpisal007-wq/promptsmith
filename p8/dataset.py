"""Synthetic, rubric-labelled prompt dataset for training the BERT quality classifier (U3).

Each prompt is assembled from five rubric components. Every component is sampled at
level 0 (absent), 1 (partial) or 2 (strong) from varied phrasing pools, so the label
is known by construction: quality = sum of levels (0-10) -> band weak/average/strong.

    python -m p8.dataset --n 3000 --out data/synthetic_prompts.csv
"""
from __future__ import annotations

import argparse
import csv
import random

from .rubric import CRITERIA, band

DOMAINS = {
    "programming": {
        "topics": ["recursion", "quicksort", "REST APIs", "Python decorators", "SQL joins", "Git branching",
                   "memory leaks in JavaScript", "unit testing", "Docker containers", "binary search"],
        "aspects": ["time complexity", "common mistakes", "real-world use", "a worked example", "edge cases"],
        "role2": ["You are a senior Python developer with 10 years of experience.",
                  "Act as an experienced software engineering mentor.",
                  "You are a principal backend engineer who reviews code for a fintech startup.",
                  "Imagine you are a computer science professor known for clear explanations."],
        "aud2": ["I'm a second-year CSE student preparing for a technical interview next week.",
                 "This is for a workshop for first-year students who have never coded; the goal is to build confidence.",
                 "Our team is migrating a legacy app and I need to explain this to junior developers so that they can review PRs."],
        "aud1": ["for beginners", "I am a student.", "for my team"],
    },
    "data": {
        "topics": ["linear regression", "overfitting", "a confusion matrix", "k-means clustering",
                   "data cleaning in pandas", "A/B testing", "feature scaling", "precision vs recall"],
        "aspects": ["assumptions", "how to detect it", "a small numeric example", "when not to use it", "key metrics"],
        "role2": ["You are a data scientist who teaches analytics to MBA students.",
                  "Act as a senior machine learning engineer at a healthcare company.",
                  "You are an experienced statistics tutor for university students."],
        "aud2": ["I'm a final-year student building my first ML project and my guide asked me to justify the model choice.",
                 "The audience is non-technical managers who need to decide whether to fund the project."],
        "aud1": ["for a presentation", "I'm new to data science.", "for non-technical readers"],
    },
    "marketing": {
        "topics": ["a launch post for a streetwear brand", "an Instagram caption for a cafe", "a product description for wireless earbuds",
                   "a newsletter for a fitness app", "a tagline for an eco-friendly bottle", "ad copy for a coding bootcamp"],
        "aspects": ["the target customer", "a call to action", "three key benefits", "the brand voice", "an offer"],
        "role2": ["You are an award-winning copywriter for D2C fashion brands.",
                  "Act as a social media strategist with 8 years of experience in Indian markets.",
                  "You are a senior marketing manager at a consumer-electronics company."],
        "aud2": ["Our customers are college students aged 18-24 in Pune and the goal is to drive pre-orders.",
                 "We are a small startup launching next Monday, so the copy must build hype before the drop."],
        "aud1": ["for our website", "for Instagram", "for young customers"],
    },
    "education": {
        "topics": ["photosynthesis", "Newton's third law", "the French Revolution", "compound interest",
                   "the water cycle", "climate change", "the structure of an atom", "supply and demand"],
        "aspects": ["a simple analogy", "key terms", "a real-life example", "common misconceptions", "a quick quiz"],
        "role2": ["You are a high-school physics teacher who uses everyday examples.",
                  "Act as a patient primary-school science tutor.",
                  "You are a history professor who writes engaging study notes."],
        "aud2": ["My 12-year-old brother has an exam tomorrow and finds textbooks boring, so it should feel like a story.",
                 "I'm preparing revision notes for a class of 40 tenth-grade students before their board exams."],
        "aud1": ["for kids", "for my exam", "for students"],
    },
    "career": {
        "topics": ["a cover letter for a frontend internship", "my resume summary", "answers to 'tell me about yourself'",
                   "a LinkedIn headline", "a follow-up email after an interview", "a salary negotiation script"],
        "aspects": ["my projects", "measurable achievements", "the company's values", "a confident closing line"],
        "role2": ["You are a technical recruiter who has hired 200+ engineers.",
                  "Act as an experienced career coach for fresh graduates.",
                  "You are an HR manager at a product company reviewing intern applications."],
        "aud2": ["I'm a final-year B.Tech student applying to product startups and I have two internships in AI and frontend.",
                 "The role is a junior data analyst position and the hiring manager values clear communication."],
        "aud1": ["for an interview", "I'm a fresher.", "for a recruiter"],
    },
    "writing": {
        "topics": ["a short story about a lost robot", "a poem about monsoon", "a blog post on productivity",
                   "a speech for a college fest", "a horror story set in a hostel", "an essay on social media"],
        "aspects": ["a twist ending", "vivid imagery", "a clear argument", "dialogue", "a strong opening hook"],
        "role2": ["You are a published fantasy novelist.", "Act as a professional speechwriter.",
                  "You are an experienced blog editor who writes for young professionals."],
        "aud2": ["It will be read aloud at our college annual day to an audience of 500 students, so it must be energetic.",
                 "This is for a children's magazine aimed at 8-10 year olds who love adventure."],
        "aud1": ["for a competition", "for my blog", "for children"],
    },
    "business": {
        "topics": ["an email asking my manager for leave", "meeting minutes", "a project status update",
                   "a complaint to a vendor about a late delivery", "a proposal for a new feature", "a weekly report"],
        "aspects": ["dates", "the reason", "next steps", "the impact", "a clear request"],
        "role2": ["You are a professional business communication coach.",
                  "Act as an experienced project manager at an IT services company.",
                  "You are an executive assistant who writes concise corporate emails."],
        "aud2": ["My manager is strict about deadlines and I need leave from 12 to 14 March for a family function.",
                 "The update goes to senior leadership who only read the first three lines."],
        "aud1": ["to my boss", "for the client", "for my team"],
    },
    "travel": {
        "topics": ["a 3-day trip to Goa", "a bike ride from Pune to Lonavala", "a budget trip to Manali",
                   "a weekend in Hampi", "a packing list for a trek", "a road trip across Rajasthan"],
        "aspects": ["budget", "places to eat", "safety tips", "the best time to visit", "travel time between stops"],
        "role2": ["You are an experienced travel planner who specialises in budget trips in India.",
                  "Act as a motorcycle touring guide who has ridden across the Western Ghats."],
        "aud2": ["We are four college friends with a total budget of 12,000 rupees and we travel by train.",
                 "I'm riding solo for the first time next month, so safety matters more than speed."],
        "aud1": ["for friends", "on a budget", "for a family"],
    },
}

ROLE1 = ["You are an expert.", "Act as an assistant.", "You are a helpful AI.", "As an expert,", "Act as a teacher."]
VERB1 = ["Explain", "Write about", "Describe", "Tell me about", "Summarize", "Create", "Write"]
VAGUE = ["{t}", "{t} pls", "help me with {t}", "something about {t}", "{t}??", "i need {t}", "can u do {t}", "{t} stuff", "idk, {t}"]
TASK2 = [
    "Explain {t}, covering {a1} and {a2}.",
    "Write {t} that focuses on {a1} and {a2}.",
    "Create a step-by-step guide to {t}, including {a1}.",
    "Explain {t} with {a1}, then list {n} key points about {a2}.",
    "Compare two approaches to {t} in terms of {a1} and {a2}.",
    "Draft {t} that highlights {a1} and ends with {a2}.",
]
FORMAT1 = ["Use bullet points.", "Give it as a table.", "Format it as an email.", "Write it in paragraphs.",
           "Answer in a numbered list.", "Return JSON.", "Make it a short summary."]
FORMAT2 = ["Structure it as {n} sections with headings, each under 80 words.",
           "Return a markdown table with columns: Concept, Example, Why it matters.",
           "Give {n} numbered steps, followed by a one-line summary.",
           "Format: a title, {n} bullet points, then a closing call to action.",
           "Use {n} short paragraphs and end with a 3-question quiz.",
           "Output JSON with keys 'summary', 'key_points' and 'example'."]
CONS1 = ["Keep it short.", "Use a friendly tone.", "Avoid jargon.", "Keep it under 150 words.", "Make it formal.",
         "Do not use emojis.", "Use simple language."]
CONS2 = ["Keep it under {w} words, use simple language and avoid jargon.",
         "Use a professional but warm tone, do not exceed {w} words, and include one real example.",
         "Avoid clichés, keep every sentence under 20 words, and don't use emojis.",
         "Use only Indian examples, keep it under {w} words and make sure the tone is encouraging.",
         "Do not invent statistics, cite sources where possible, and stay under {w} words."]

# ------------------------------------------------------------- image slots
IMG_SUBJECTS = [("a cat", "a ginger cat curled up asleep on a stack of old books"),
                ("a car", "a vintage red convertible parked beside a sea cliff"),
                ("a city", "a futuristic city skyline with flying trams between glass towers"),
                ("a dog", "a golden retriever puppy chasing bubbles in a garden"),
                ("a castle", "a crumbling stone castle on a hill covered with wildflowers"),
                ("a robot", "a small friendly robot watering plants on a balcony"),
                ("a temple", "an ancient stone temple with carved pillars and oil lamps"),
                ("a motorcycle", "a matte black touring motorcycle on a winding ghat road"),
                ("a girl reading", "a girl reading a book under a banyan tree"),
                ("food", "a steaming plate of misal pav with lemon and chopped onions"),
                ("a tree", "a lone cherry blossom tree beside a frozen lake"),
                ("a hoodie", "an oversized black hoodie with a constellation print on a mannequin")]
IMG_STYLE1 = ["photo", "painting", "anime", "digital art", "illustration", "3d render", "sketch"]
IMG_STYLE2 = ["cinematic photograph, 35mm film", "soft watercolour painting with ink outlines",
              "highly detailed digital painting, concept art", "studio ghibli inspired anime illustration",
              "photorealistic product photograph, studio quality", "isometric low poly 3d render"]
IMG_CTX1 = ["at night", "at sunset", "in the rain", "in a forest", "on a beach", "in a studio"]
IMG_CTX2 = ["in a misty forest at sunrise, soft golden light", "on a neon-lit street at night in the rain, moody atmosphere",
            "in a cozy room with warm lamp light", "on a snowy mountain at dusk, dramatic sky",
            "in a bustling market at noon, vibrant and lively"]
IMG_FMT1 = ["close-up", "wide shot", "16:9", "square", "aerial view", "full body"]
IMG_FMT2 = ["close-up, 85mm, shallow depth of field", "wide-angle shot, rule of thirds, 16:9",
            "top-down view, centered composition, 4k", "low-angle shot, sharp focus, highly detailed"]
IMG_CON1 = ["no text", "no watermark", "pastel palette", "monochrome", "white background"]
IMG_CON2 = ["no text, no watermark, muted earth tones", "only two people, no logos, teal and orange palette",
            "plain white background, no shadows, limited colour palette", "no blur, no distortion, warm tones"]


def _text_prompt(rng: random.Random, levels: dict) -> str:
    dom = DOMAINS[rng.choice(list(DOMAINS))]
    t = rng.choice(dom["topics"])
    a1, a2 = rng.sample(dom["aspects"], 2)
    n, w = rng.choice([3, 4, 5]), rng.choice([150, 200, 250, 300])
    parts = {}
    parts["role"] = {0: "", 1: rng.choice(ROLE1), 2: rng.choice(dom["role2"])}[levels["role"]]
    parts["context"] = {0: "", 1: rng.choice(dom["aud1"]), 2: rng.choice(dom["aud2"])}[levels["context"]]
    if levels["task"] == 0:
        task = rng.choice(VAGUE).format(t=t) + ("." if rng.random() < 0.5 else "")
    elif levels["task"] == 1:
        task = f"{rng.choice(VERB1)} {t}."
    else:
        task = rng.choice(TASK2).format(t=t, a1=a1, a2=a2, n=n)
    parts["task"] = task
    parts["format"] = {0: "", 1: rng.choice(FORMAT1), 2: rng.choice(FORMAT2).format(n=n)}[levels["format"]]
    parts["constraints"] = {0: "", 1: rng.choice(CONS1), 2: rng.choice(CONS2).format(w=w)}[levels["constraints"]]

    ctx = parts["context"]
    # short audience phrases ("for beginners") attach to the task sentence
    if ctx and not ctx.endswith("."):
        parts["task"] = parts["task"].rstrip(".?") + f" {ctx}."
        parts["context"] = ""
    order = ["role", "context", "task", "format", "constraints"]
    if rng.random() < 0.3:
        tail = order[3:]
        rng.shuffle(tail)
        order = order[:3] + tail
    if rng.random() < 0.2 and parts["context"]:
        order = ["role", "task", "context", "format", "constraints"]
    text = " ".join(parts[k] for k in order if parts[k])
    if rng.random() < 0.15:
        text = text.lower()
    if rng.random() < 0.1:
        text = text.replace(". ", ".\n")
    return text.strip()


def _image_prompt(rng: random.Random, levels: dict) -> str:
    vague, rich = rng.choice(IMG_SUBJECTS)
    subj = {0: vague.split()[-1], 1: vague, 2: rich}[levels["task"]]
    bits = [subj]
    pools = {"role": (IMG_STYLE1, IMG_STYLE2), "context": (IMG_CTX1, IMG_CTX2),
             "format": (IMG_FMT1, IMG_FMT2), "constraints": (IMG_CON1, IMG_CON2)}
    for c in ["role", "context", "format", "constraints"]:
        lv = levels[c]
        if lv:
            bits.append(rng.choice(pools[c][lv - 1]))
    if levels["role"] and rng.random() < 0.5:  # style first: "watercolour painting of ..."
        style = bits.pop(1)
        bits[0] = f"{style} of {bits[0]}"
    text = ", ".join(bits)
    if rng.random() < 0.3:
        text = rng.choice(["Generate an image of ", "Draw ", "Picture of ", ""]) + text
    return text


def sample_levels(rng: random.Random, target_band: str) -> dict:
    lo, hi = {"weak": (0, 3), "average": (4, 6), "strong": (7, 10)}[target_band]
    while True:
        lv = {c: rng.choice([0, 1, 2]) for c in CRITERIA}
        if lo <= sum(lv.values()) <= hi:
            return lv


def generate(n: int = 3000, seed: int = 42, image_share: float = 0.3) -> list[dict]:
    rng = random.Random(seed)
    rows, seen = [], set()
    bands = ["weak", "average", "strong"]
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        b = bands[len(rows) % 3]
        lv = sample_levels(rng, b)
        kind = "image" if rng.random() < image_share else "text"
        text = _image_prompt(rng, lv) if kind == "image" else _text_prompt(rng, lv)
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        score = sum(lv.values())
        rows.append({"prompt": text, "kind": kind, **lv, "total": score, "label": band(score)})
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_prompts.csv")
    args = ap.parse_args()
    rows = generate(args.n, args.seed)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
