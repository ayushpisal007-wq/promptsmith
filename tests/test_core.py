"""Fast tests that need no model downloads:  LLM_BACKEND=mock pytest -q"""
import os

os.environ.setdefault("LLM_BACKEND", "mock")

from p8.dataset import generate
from p8.judge import parse_judgement
from p8.rubric import band, detect_kind, rule_score, total
from p8.safety import find_bias, find_pii, redact


def test_weak_and_strong_prompts_are_separated():
    weak = total(rule_score("explain machine learning"))
    strong = total(rule_score(
        "You are a senior Python developer with 10 years of experience. I am a first-year student preparing for my "
        "exam next week. Explain how quicksort works, covering the partition step and time complexity. Use 3 sections "
        "with headings. Keep it under 300 words and avoid jargon."))
    assert band(weak) == "weak" and band(strong) == "strong"


def test_detect_kind():
    assert detect_kind("a cat") == "image"
    assert detect_kind("explain machine learning") == "text"
    assert detect_kind("cinematic photograph of a fox in a forest, close-up") == "image"


def test_pii_redaction():
    text, found = redact("mail rahul.k@gmail.com or call 9876543210, PAN ABCDE1234F")
    assert "[EMAIL]" in text and "[PHONE]" in text and "[PAN]" in text
    assert {f["type"] for f in found} == {"EMAIL", "PHONE", "PAN"}
    assert not find_pii("Version 2.4.1 was released on 12 March")


def test_card_needs_luhn():
    assert find_pii("card 4111 1111 1111 1111")
    assert not find_pii("order 1234 5678 is delayed")


def test_bias_terms():
    terms = {b["term"].lower() for b in find_bias("hire a young energetic salesman, native English speakers only")}
    assert "salesman" in terms and "young energetic" in terms and "native english speakers" in terms
    assert not find_bias("The nurse finished their shift")


def test_judge_parser():
    p = parse_judgement("role: 2\ncontext: 1\ntask: 2\nformat: 0\nconstraints: 1\nfeedback: Add a format.")
    assert p["parsed_ok"] and p["scores"]["role"] == 2 and p["feedback"] == "Add a format."
    assert not parse_judgement("I think it's good")["parsed_ok"]


def test_dataset_balanced_and_labelled():
    rows = generate(300, seed=1)
    labels = [r["label"] for r in rows]
    assert {labels.count(l) for l in ["weak", "average", "strong"]} == {100}
    assert all(band(r["total"]) == r["label"] for r in rows)


def test_pipeline_with_mock_llm():
    from p8.pipeline import analyze

    r = analyze("write a job ad for a salesman, contact a@b.com", "text")
    assert "a@b.com" not in r["clean"] and r["safety"]["bias"]
    assert r["after"]["rule_total"] >= r["before"]["rule_total"]
