# Model card: P8 BERT prompt-quality classifier (and the models used around it)

*Format follows Mitchell et al. (2019), "Model Cards for Model Reporting".*

## Model details
- **Name:** `p8-bert-prompt-quality`
- **Developer:** Ayush Pisal (Roll No. 08), MIT ADT University, Generative AI CA mini-project, 2026
- **Type:** `bert-base-uncased` (110M parameters) fine-tuned for 3-class sequence classification: `weak` / `average` / `strong`
- **Training:** PyTorch, AdamW, learning rate 3e-5, 3 epochs, batch size 16, max length 128, linear warm-up and decay, seed 42 (`train_bert.py`)
- **Licence:** Apache-2.0, inherited from BERT

## Intended use
- **Primary:** feedback for students learning prompt engineering. The classifier predicts how complete a prompt is against the P8 rubric (role, context, task, format, constraints). It is shown alongside the rule-based scores and the LLM judge, never alone.
- **Out of scope:** grading students for marks without a human check; judging whether a prompt is *factually* correct or safe; non-English prompts.

## Training data
- 3,000 **synthetic** prompts (`p8/dataset.py`), balanced across the three classes, about 70% text and 30% image prompts, built from 8 domains (programming, data, marketing, education, career, writing, business, travel).
- Each prompt is assembled from five rubric parts, each sampled at level 0, 1 or 2 from phrasing pools. The label comes from the construction: total 0–3 = weak, 4–6 = average, 7–10 = strong.
- The split is 80 / 10 / 10 (train / validation / test).

## Evaluation
| Test set | Accuracy | Macro-F1 |
|---|---|---|
| Synthetic test split (300) | 0.940 | 0.940 |
| 30 hand-labelled gold prompts (out-of-distribution) | 0.800 | 0.709 |

Trained in 113 s on a T4 GPU; validation accuracy 0.890 → 0.903 → 0.933 over the 3 epochs. On the gold set, 4 of 6 *average* prompts were predicted *weak*.

Band accuracy of the other graders on the same 30 gold prompts: rule-based 0.867, few-shot LLM judge 0.633, zero-shot LLM judge 0.400 (`results/judge_comparison.csv`).

**Counterfactual bias test:** across 10 name/gender-swapped pairs, BERT's P(strong) changed by 0.000 and the rule scores did not change. The LLM judge changed in 3 of 10 pairs, by up to 3 points.

## Limitations and failure modes
- **Synthetic-template bias:** the model learns phrasing patterns from the generator (e.g. "You are a…", "Use bullet points"). Unusual but good prompts can be under-rated, and keyword-stuffed prompts over-rated.
- **Completeness, not quality:** a prompt can be "strong" on the rubric and still ask for something harmful or wrong.
- The 30 gold labels were written by one person, the author, so they are subjective.

## Ethical considerations
- **Personal data:** the app redacts emails, phone numbers, Aadhaar, PAN, card numbers, IPs, addresses and self-stated names *before* any model sees a prompt. Nothing is stored. Regex detection misses spelled-out or partial data (see `results/safety_misses.csv`).
- **Bias:** a lexicon flags gendered job titles, age requirements, ableist words and group generalisations. `evaluate.py` runs a counterfactual test (only a name or gender is swapped) and reports how much the scores change.
- **IP:** the image rewriter is told not to add living artists' names or brand logos. Generated images may still resemble training data. sd-turbo outputs are for non-commercial academic use.

## Other models used (not trained by me)
| Model | Role | Licence |
|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | LLM judge, rewriter, text runner | Apache-2.0 |
| stabilityai/sd-turbo | image generation (1 step) | Stability AI Non-Commercial Research Community |
| openai/clip-vit-base-patch32 | CLIP score | MIT |
| bert-base-uncased | base for the classifier | Apache-2.0 |

## Environmental note
Fine-tuning takes about 3 minutes on one T4 GPU. Inference runs on a free 2-vCPU Hugging Face Space.
