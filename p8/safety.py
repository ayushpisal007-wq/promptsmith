"""U5 - Responsible-AI rubric checks: personal data (PII) and biased wording.

Guardrail: PII is detected and redacted *before* a prompt is sent to any LLM.
Bias check: a transparent lexicon of exclusionary / stereotyping wording with
inclusive suggestions, plus a regex for sweeping group generalisations.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------- PII
PII_PATTERNS = {
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "AADHAAR": r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b",
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "CARD": r"\b(?:\d[ -]?){13,19}\b",
    "PHONE": r"(?:\+?91[ -]?)?\b[6-9]\d{4}[ -]?\d{5}\b|\+\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}",
    "IP": r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
    "DOB": r"\b(?:dob|date of birth|born on)\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    "NAME": r"\b(?i:my name is|i am|i'm|name:)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\b",
    "ADDRESS": r"\b\d{1,4},?\s+[A-Za-z0-9 ]{2,40}\b(?:road|rd|street|st|lane|nagar|colony|society|apartments?|sector)\b",
}
_NOT_NAMES = {"Student", "Teacher", "Engineer", "Developer", "Beginner", "Writing", "Looking", "Trying", "Not", "A", "An", "The"}


def _luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    s = 0
    for i, n in enumerate(reversed(d)):
        if i % 2:
            n *= 2
            if n > 9:
                n -= 9
        s += n
    return s % 10 == 0


def find_pii(text: str) -> list[dict]:
    found = []
    taken: list[tuple[int, int]] = []
    order = ["EMAIL", "DOB", "CARD", "AADHAAR", "PAN", "PHONE", "IP", "ADDRESS", "NAME"]
    for kind in order:
        for m in re.finditer(PII_PATTERNS[kind], text, flags=0 if kind in ("PAN", "NAME") else re.I):
            start, end = (m.start(1), m.end(1)) if kind == "NAME" else (m.start(), m.end())
            value = text[start:end]
            if kind == "CARD" and not _luhn_ok(value):
                continue
            if kind == "NAME" and value.split()[0] in _NOT_NAMES:
                continue
            if any(s < end and start < e for s, e in taken):
                continue
            taken.append((start, end))
            found.append({"type": kind, "value": value, "start": start, "end": end})
    return sorted(found, key=lambda x: x["start"])


def redact(text: str) -> tuple[str, list[dict]]:
    """Replace every PII span with a [TYPE] tag. Returns (redacted_text, findings)."""
    items = find_pii(text)
    out = text
    for it in sorted(items, key=lambda x: -x["start"]):
        out = out[: it["start"]] + f"[{it['type']}]" + out[it["end"]:]
    return out, items


# -------------------------------------------------------------------- bias
BIAS_LEXICON = [
    # (pattern, category, suggestion)
    (r"\bchairman\b", "gender", "chairperson / chair"),
    (r"\b(?:sales|business|police|fire|camera|delivery|fore|work)m[ae]n\b", "gender", "gender-neutral job title (salesperson, police officer, firefighter...)"),
    (r"\bmankind\b", "gender", "humankind / people"),
    (r"\bmanpower\b", "gender", "workforce / staff"),
    (r"\bman[- ]hours?\b", "gender", "person-hours / work hours"),
    (r"\bstewardess\b", "gender", "flight attendant"),
    (r"\b(?:waitress|actress|hostess)\b", "gender", "server / actor / host"),
    (r"\bguys\b", "gender", "everyone / folks / team"),
    (r"\bfemale (?:engineer|doctor|pilot|ceo|boss|driver|scientist)\b", "gender", "drop the gender marker unless it is relevant"),
    (r"\b(?:young(?: and|,)? (?:energetic|dynamic|fresh)|youthful|digital natives?)\b", "age", "describe the skill needed (e.g. 'motivated', 'comfortable with digital tools')"),
    (r"\b(?:recent graduates? only|under \d{2}|below \d{2} years|max(?:imum)? age)\b", "age", "remove age limits unless legally required"),
    (r"\b(?:boomers?|old people|oldies|senile)\b", "age", "older adults"),
    (r"\b(?:aged?|age group|age range)\s*(?:between|of|from)?\s*\d{2}\s*(?:-|–|to|and)\s*\d{2}\b|\b\d{2}\s*(?:-|–|to)\s*\d{2}\s*(?:years?|yrs)(?: old)?\b", "age", "remove the age range; describe the experience or skills needed"),
    (r"\byoung (?:professionals?|candidates?|people|talent)\b", "age", "early-career professionals, or describe the skills needed"),
    (r"\bnative (?:english )?speakers?\b", "nationality/language", "fluent in English"),
    (r"\b(?:fair[- ]skinned|fair complexion|good looking|attractive)\b", "appearance", "remove appearance requirements"),
    (r"\b(?:able[- ]bodied)\b", "disability", "state the actual physical requirement"),
    (r"\b(?:wheelchair[- ]bound|confined to a wheelchair)\b", "disability", "wheelchair user"),
    (r"\bsuffers? from\b", "disability", "has / lives with"),
    (r"\b(?:crazy|insane|lame|psycho|retarded|dumb)\b", "disability", "a neutral word (wild, unreasonable, poor...)"),
    (r"\b(?:culture fit)\b", "exclusion", "'values alignment' with stated values"),
    (r"\b(?:third[- ]world)\b", "nationality", "low- and middle-income countries"),
    (r"\b(?:illegals?|illegal aliens?)\b", "nationality", "undocumented people"),
]

GROUPS = (
    r"women|men|girls|boys|females|males|muslims|hindus|christians|sikhs|jews|indians|chinese|americans|"
    r"africans|asians|europeans|immigrants|old people|elderly|millennials|gen z|poor people|rich people|"
    r"gay people|disabled people|north indians|south indians|biharis|marathis|punjabis|tamils"
)
GENERALISATION = re.compile(
    r"\b(?:all|every|most|typical)?\s*(?P<group>" + GROUPS + r")\s+(?:are|can't|cannot|don't|always|never|"
    r"should(?:n't)?|tend to|are bad at|are good at|lack)\b",
    re.I,
)
GENDER_ASSUMPTION = re.compile(
    r"\b(?:the |a |an )?(?:doctor|engineer|ceo|pilot|manager|programmer|scientist|boss|surgeon)\b[^.?!]{0,60}\bhe\b|"
    r"\b(?:the |a |an )?(?:nurse|secretary|receptionist|teacher|maid|homemaker)\b[^.?!]{0,60}\bshe\b",
    re.I,
)


def find_bias(text: str) -> list[dict]:
    hits = []
    for pat, cat, sugg in BIAS_LEXICON:
        for m in re.finditer(pat, text, re.I):
            hits.append({"term": m.group(0), "category": cat, "suggestion": sugg})
    for m in GENERALISATION.finditer(text):
        hits.append({"term": m.group(0).strip(), "category": "stereotype / generalisation",
                     "suggestion": "avoid claims about whole groups; describe individuals or evidence"})
    for m in GENDER_ASSUMPTION.finditer(text):
        hits.append({"term": m.group(0).strip(), "category": "gender assumption",
                     "suggestion": "use 'they' or the role name instead of assuming gender"})
    seen, out = set(), []
    for h in hits:
        key = h["term"].lower()
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


def safety_report(text: str) -> dict:
    redacted, pii = redact(text)
    bias = find_bias(text)
    return {
        "redacted": redacted,
        "pii": pii,
        "bias": bias,
        "pii_pass": not pii,
        "bias_pass": not bias,
    }
