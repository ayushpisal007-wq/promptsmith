"""U1 - The prompt-quality rubric and a transparent rule-based scorer.

Five criteria, each scored 0 (missing), 1 (partial) or 2 (strong):
    role, context, task, format, constraints  -> total out of 10.

Image prompts use the same five slots, re-interpreted for text-to-image:
    role -> style / medium, context -> setting / lighting / mood,
    task -> subject + action, format -> composition / camera / aspect ratio,
    constraints -> exclusions / palette / "no text".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CRITERIA = ["role", "context", "task", "format", "constraints"]
BANDS = ["weak", "average", "strong"]

RUBRIC = {
    "role": {
        "text": "Does the prompt give the model a role or persona? "
                "0 = none, 1 = generic ('you are an assistant/expert'), "
                "2 = specific role with domain or experience ('You are a senior Python reviewer').",
        "image": "Is an art style / medium given? 0 = none, 1 = one style word "
                 "('photo'), 2 = clear style plus qualifier ('cinematic photograph, 35mm film').",
    },
    "context": {
        "text": "Does it say who it is for, why, or the background? "
                "0 = none, 1 = one detail (audience OR purpose), 2 = audience AND purpose/background.",
        "image": "Is a setting, lighting or mood described? 0 = none, 1 = one, 2 = two or more.",
    },
    "task": {
        "text": "Is the task a clear action? 0 = vague ('help with X', just a topic), "
                "1 = clear verb ('explain X'), 2 = clear verb + specific scope (aspects, number, depth).",
        "image": "Is the subject clear? 0 = one or two vague words, 1 = clear subject, "
                 "2 = subject with action/attributes.",
    },
    "format": {
        "text": "Is the output format stated? 0 = none, 1 = a format type (list, table, email), "
                "2 = detailed structure (sections, columns, length per part).",
        "image": "Is composition stated? 0 = none, 1 = one (close-up, 16:9, wide shot), "
                 "2 = two or more (shot type + lens/aspect/detail level).",
    },
    "constraints": {
        "text": "Are limits given? 0 = none, 1 = one (length OR tone OR avoid X), 2 = two or more.",
        "image": "Are exclusions or palette given? 0 = none, 1 = one ('no text'), 2 = two or more.",
    },
}

# ---------------------------------------------------------------- lexicons
ROLE_NOUNS = (
    "teacher|tutor|professor|lecturer|instructor|mentor|coach|developer|programmer|engineer|"
    "architect|scientist|analyst|researcher|writer|copywriter|editor|journalist|author|novelist|"
    "poet|marketer|strategist|consultant|designer|recruiter|manager|lawyer|doctor|nutritionist|"
    "trainer|accountant|advisor|adviser|planner|historian|photographer|chef|reviewer|examiner|"
    "specialist|expert|assistant|translator|interviewer|counsellor|counselor|pm|product manager|"
    "data scientist|seo|hr|guide|storyteller|comedian|critic|auditor|officer|founder|ceo"
)
ROLE_TRIGGER = re.compile(
    r"\b(you are|you're|act as|acting as|pretend (?:to be|you are)|imagine you(?:'re| are)|"
    r"take the role of|your role is|role:|as (?:an?|my) (?:\w+ ){0,3}(?:" + ROLE_NOUNS + r"))\b",
    re.I,
)
GENERIC_ROLE = re.compile(r"\b(?:you are|act as) (?:an? )?(?:helpful |smart |good )?(?:ai|assistant|chatbot|expert|bot)\b[.,!]?", re.I)
ROLE_SPECIFIC = re.compile(
    r"\b(senior|junior|experienced|expert|professional|certified|award[- ]winning|veteran|"
    r"\d+\+? years?|specialis(?:ed|t)|specializ(?:ed|ing)|phd|seasoned|lead|principal|high[- ]school|"
    r"university|primary[- ]school|corporate|technical|python|java|marketing|finance|medical|legal|"
    r"data|ux|ui|seo|career|fitness|travel|physics|chemistry|biology|math(?:s|ematics)?|history|english)\b",
    re.I,
)

CONTEXT_SIGNALS = [
    r"\bfor (?:a |an |my |our |the )?(?:beginners?|students?|kids|children|class|team|clients?|customers?|"
    r"audience|readers?|manager|boss|interview|exam|presentation|workshop|website|blog|newsletter|"
    r"startup|company|non[- ]technical|first[- ]year|college|school|parents|seniors|recruiters?)\b",
    r"\b(?:audience|readers?|users?) (?:is|are)\b",
    r"\bto (?:my |the |a )?(?:boss|manager|teacher|professor|hr|client|customer|landlord|team|colleague|principal)\b",
    r"\b(?:i am|i'm|i have|i've|we are|we're|my|our)\b",
    r"\b(?:because|so that|in order to|the goal is|goal:|purpose|context:|background:|currently|"
    r"preparing|deadline|tomorrow|next week|upcoming)\b",
    r"\bwho (?:are|is|have|has|know|don't)\b",
]
IMAGE_CONTEXT = re.compile(
    r"\b(at (?:night|dawn|dusk|sunset|sunrise)|sunset|sunrise|golden hour|night|morning|evening|rain(?:y)?|"
    r"fog(?:gy)?|snow(?:y)?|forest|city|street|beach|mountains?|desert|ocean|studio|kitchen|room|cafe|"
    r"market|space|underwater|background|lighting|lit|light|neon|moody|mood|atmosphere|cozy|dramatic|"
    r"serene|vibrant|warm|cold|soft light|backlit|in a|on a)\b",
    re.I,
)
TASK_VERBS = re.compile(
    r"\b(write|explain|summari[sz]e|create|generate|list|compare|translate|draft|design|analy[sz]e|"
    r"classify|debug|fix|review|plan|outline|describe|convert|calculate|suggest|recommend|rewrite|"
    r"build|code|implement|find|give|make|prepare|evaluate|critique|teach|solve|answer|propose|"
    r"brainstorm|edit|proofread|extract|rank|identify|predict|draw|paint|illustrate|render|compose|"
    r"optimi[sz]e|refactor|test|grade|score|define|map|schedule|estimate)\b",
    re.I,
)
VAGUE_TASK = re.compile(r"\b(help(?: me)?(?: with)?|something about|stuff|things|anything|do (?:it|this|my)|idk)\b", re.I)
SCOPE_SIGNALS = re.compile(
    r"(\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b|covering|including|focus(?:ing)? on|"
    r"in terms of|step[- ]by[- ]step|pros and cons|advantages|differences?|examples?|each|both|"
    r"specifically|\bvs\.?\b|versus|key|main)",
    re.I,
)
FORMAT_WORDS = re.compile(
    r"\b(bullet(?:s| points?)?|list|numbered|table|columns?|rows?|json|csv|yaml|markdown|headings?|"
    r"sections?|paragraphs?|steps|outline|essay|email|e-mail|letter|tweet|thread|slides?|code block|"
    r"script|summary|report|faq|q&a|template|checklist|diagram|poem|story|words?|sentences?|lines?|"
    r"title|subject line|format)\b",
    re.I,
)
FORMAT_DETAIL = re.compile(
    r"(\d+\s*(?:bullet|point|section|paragraph|step|column|row|heading|line|sentence|word|item|slide|tip)s?|"
    r"columns?:|with columns|with headings|with (?:a )?(?:title|subject line)|each (?:section|point|step)|"
    r"followed by|then (?:a|an)|structure|format:)",
    re.I,
)
IMAGE_FORMAT = re.compile(
    r"(\b\d{1,2}:\d{1,2}\b|\bsquare\b|\bportrait (?:orientation|format)\b|\blandscape (?:orientation|format)\b|"
    r"close[- ]up|wide[- ]angle|wide shot|full[- ]body|macro|aerial|bird'?s[- ]eye|top[- ]down|"
    r"low[- ]angle|isometric|\d{2,3}\s?mm|f/\d|bokeh|depth of field|rule of thirds|centered|symmetr\w+|"
    r"\b[48]k\b|high(?:ly)? detailed|ultra[- ]detailed|sharp focus|composition|framing|shot)",
    re.I,
)
IMAGE_STYLE = re.compile(
    r"\b(photo(?:graph)?|photorealistic|realistic|oil painting|watercolou?r|digital (?:art|painting)|"
    r"illustration|3d render|render|anime|manga|pixel art|sketch|pencil|charcoal|cinematic|concept art|"
    r"vector|flat design|low poly|cartoon|comic|surreal|minimalist|art nouveau|impressionist|ukiyo-e|"
    r"studio ghibli|in the style of|style|film|polaroid|vintage|retro|cyberpunk|steampunk|fantasy)\b",
    re.I,
)
CONSTRAINT_SIGNALS = [
    r"\b(?:under|below|less than|fewer than|no more than|at most|maximum|max|within|up to|limit(?:ed)? to|"
    r"exactly|at least|minimum)\s+\d+",
    r"\b(?:avoid|don't|do not|never|without|no jargon|exclude|must not|shouldn't)\b",
    r"\b(?:tone|formal|informal|casual|friendly|professional|polite|simple (?:language|words|english)|"
    r"plain (?:language|english)|encouraging|neutral|persuasive|humou?rous)\b",
    r"\b(?:must|only|make sure|ensure|required?|cite|include|use (?:only|python|british|american|examples?))\b",
    r"\b(?:keep it (?:short|brief|simple|concise)|short|brief|concise)\b",
    r"\b(?:budget|deadline|by (?:monday|tuesday|wednesday|thursday|friday|tomorrow))\b",
]
IMAGE_CONSTRAINTS = [
    r"\b(?:no|without|avoid) (?:text|words|letters|watermarks?|logos?|people|humans|blur|distortion|extra)\b",
    r"\bnegative prompt\b",
    r"\b(?:colou?r palette|palette|pastel|monochrome|black and white|muted|earth tones|limited colou?rs?|"
    r"teal and orange|warm tones|cool tones)\b",
    r"\bonly\b",
    r"\bplain background|white background|transparent background\b",
]


@dataclass
class CriterionResult:
    score: int
    evidence: list[str] = field(default_factory=list)


def _find_all(pattern, text):
    if isinstance(pattern, str):
        pattern = re.compile(pattern, re.I)
    return list(dict.fromkeys(m.group(0).strip() for m in pattern.finditer(text)))


def detect_kind(prompt: str) -> str:
    """Guess whether a prompt is for text-to-image generation."""
    p = prompt.lower()
    if re.search(r"\b(image|picture|photo(?:graph)?|drawing|painting|illustration|render|wallpaper|logo|poster art)\b"
                 r"|\b(draw|paint|illustrate|render)\b", p):
        if not re.search(r"\b(write|explain|essay|email|code|list)\b", p):
            return "image"
    words = re.findall(r"[a-z']+", p)
    if TASK_VERBS.search(p) or VAGUE_TASK.search(p) or p.strip().endswith("?"):
        return "text"
    if IMAGE_STYLE.search(p) or IMAGE_FORMAT.search(p) or len(words) <= 8:
        return "image"
    return "text"


def _score_role(p, kind):
    if kind == "image":
        ev = _find_all(IMAGE_STYLE, p)
        return CriterionResult(min(2, len(ev)), ev)
    m = ROLE_TRIGGER.search(p)
    if not m:
        return CriterionResult(0)
    sentence = re.split(r"[.!?\n]", p[m.start():], maxsplit=1)[0]
    if GENERIC_ROLE.fullmatch(sentence.strip() + "") or (GENERIC_ROLE.search(sentence) and not ROLE_SPECIFIC.search(sentence)):
        return CriterionResult(1, [sentence.strip()])
    if ROLE_SPECIFIC.search(sentence) or len(sentence.split()) >= 7:
        return CriterionResult(2, [sentence.strip()])
    return CriterionResult(1, [sentence.strip()])


def _score_context(p, kind):
    if kind == "image":
        ev = _find_all(IMAGE_CONTEXT, p)
        return CriterionResult(0 if not ev else (1 if len(ev) == 1 else 2), ev)
    ev, hits = [], 0
    for pat in CONTEXT_SIGNALS:
        found = _find_all(pat, p)
        if found:
            hits += 1
            ev += found
    return CriterionResult(min(2, hits), ev)


def _score_task(p, kind):
    words = re.findall(r"[A-Za-z']+", p)
    if kind == "image":
        content = [w for w in words if len(w) > 2 and w.lower() not in {"the", "and", "with", "image", "picture", "photo", "of"}]
        n = len(content)
        return CriterionResult(0 if n < 2 else (1 if n < 5 else 2), [" ".join(content[:6])])
    verbs = _find_all(TASK_VERBS, p)
    vague = VAGUE_TASK.search(p)
    if not verbs:
        if p.strip().endswith("?") and len(words) >= 4:
            return CriterionResult(1, ["question"])
        return CriterionResult(0, [vague.group(0)] if vague else [])
    scope = _find_all(SCOPE_SIGNALS, p)
    if scope and len(words) >= 8:
        return CriterionResult(2, verbs + scope[:3])
    if vague and len(words) < 6:
        return CriterionResult(0, verbs)
    return CriterionResult(1, verbs)


def _score_format(p, kind):
    if kind == "image":
        ev = _find_all(IMAGE_FORMAT, p)
        return CriterionResult(min(2, len(ev)), ev)
    words = _find_all(FORMAT_WORDS, p)
    detail = _find_all(FORMAT_DETAIL, p)
    if not words and not detail:
        return CriterionResult(0)
    if detail or len(words) >= 3:
        return CriterionResult(2, words + detail)
    return CriterionResult(1, words)


def _score_constraints(p, kind):
    pats = IMAGE_CONSTRAINTS if kind == "image" else CONSTRAINT_SIGNALS
    ev, hits = [], 0
    for pat in pats:
        found = _find_all(pat, p)
        if found:
            hits += 1
            ev += found
    return CriterionResult(min(2, hits), ev)


_SCORERS = {
    "role": _score_role,
    "context": _score_context,
    "task": _score_task,
    "format": _score_format,
    "constraints": _score_constraints,
}


def rule_score(prompt: str, kind: str | None = None) -> dict[str, CriterionResult]:
    """Score a prompt on every rubric criterion with transparent keyword rules."""
    kind = kind or detect_kind(prompt)
    return {c: _SCORERS[c](prompt, kind) for c in CRITERIA}


def total(scores: dict) -> int:
    vals = [s.score if isinstance(s, CriterionResult) else int(s) for s in scores.values()]
    return int(sum(vals))


def band(total_score: int) -> str:
    """0-3 weak, 4-6 average, 7-10 strong (same thresholds used to label training data)."""
    if total_score <= 3:
        return "weak"
    if total_score <= 6:
        return "average"
    return "strong"
