---
title: P8 Prompt Grader & Rewriter
emoji: 📝
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Grade a prompt on a rubric, rewrite it, compare outputs
---

# P8 · Prompt Grader & Rewriter

**Generative AI CA mini-project · Ayush Pisal · Roll No. 08 · MIT ADT University (B.Tech CSE AI & Analytics)**

The app scores a student's prompt on a five-part rubric (**role, context, task, format, constraints**), rewrites it, and shows the output of the original and the rewritten prompt side by side, for both **text** and **image** prompts.

- 🔗 **Live app (Streamlit Community Cloud):** https://<your-app-name>.streamlit.app
- 🔗 **Fine-tuned BERT:** https://huggingface.co/ayushpisal/p8-bert-prompt-quality

![block diagram](docs/block_diagram.png)

## What each unit adds

| Unit | Requirement in P8 | Where in the code |
|---|---|---|
| **U1** Foundations | Rubric (role, context, task, format, constraints), with a transparent rule-based scorer; zero-shot vs few-shot judge compared; BERT class probabilities as the probability view; lifecycle diagram | `p8/rubric.py`, `docs/lifecycle.png` |
| **U2** LLMs | LLM judge (Qwen2.5-1.5B-Instruct) that grades with **few-shot graded examples**; prompt-based learning compared with BERT fine-tuning; BERT attention explanation | `p8/judge.py`, `p8/llm.py` |
| **U3** GPT · BERT · multimodal | **BERT classifier** fine-tuned to predict weak/average/strong; **CLIP score** for image prompts | `train_bert.py`, `p8/classifier.py`, `p8/clip_score.py` |
| **U4** Tools & generation | Runs the original and rewritten prompts: text on the LLM, images on **sd-turbo** with the same seed (PyTorch) | `p8/pipeline.py`, `p8/imagegen.py` |
| **U5** Responsible AI | Rubric checks for **biased wording** and **personal data**, with PII redacted before any model sees the prompt; measured detector precision/recall; counterfactual bias test of the graders | `p8/safety.py`, `evaluate.py` |

## Run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py          # main app (http://localhost:8501)
LLM_BACKEND=mock pytest -q tests        # fast tests, no model downloads
```
`app.py` is an equivalent Gradio UI for GPU hosts (it needs `gradio`, `diffusers` and `accelerate` too).

### Train BERT and produce the results (Google Colab, free T4 GPU)
Open `notebooks/P8_train_and_evaluate.ipynb` in Colab and choose **Runtime → Run all**. It fine-tunes `bert-base-uncased` (about 2 min), runs `evaluate.py` with Qwen2.5-1.5B and sd-turbo on the GPU (writing every table to `results/`), and pushes the model to the Hugging Face Hub.

## Deploy on Streamlit Community Cloud (free)
1. Go to share.streamlit.io, sign in with GitHub, then **Create app → Deploy a public app from GitHub**.
2. Repo `ayushpisal007-wq/promptsmith`, branch `main`, main file `streamlit_app.py`.
3. **Advanced settings:** Python 3.12. Under **Secrets**, add `GROQ_API_KEY = "gsk_..."` (a free key from console.groq.com).
4. Click **Deploy**.

The free host has only about 1-2.7 GB of RAM, so the deployed app calls Groq (`openai/gpt-oss-20b`) for the LLM and Pollinations (FLUX) for images. The fine-tuned BERT (`ayushpisal/p8-bert-prompt-quality`) and CLIP run inside the app with PyTorch, in bfloat16 to save memory. No key is ever stored in the repo.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | - | Streamlit secret; switches the LLM to Groq |
| `LLM_BACKEND` | `local` | `local` (transformers), `hf_api`, `openai` (any OpenAI-compatible API, e.g. Groq), `mock` (tests) |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | local judge, rewriter and runner |
| `BERT_MODEL` | `models/bert-prompt-quality` | local folder or Hub repo id |
| `IMAGE_BACKEND` | `pollinations` on Streamlit, `local` (sd-turbo) elsewhere | text-to-image |
| `LOW_MEMORY` | `1` on Streamlit | load BERT and CLIP in bfloat16 |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | CLIP score |

## Repo layout
```
streamlit_app.py    Streamlit UI (deployed)      app.py  Gradio UI
p8/rubric.py        U1 rubric + rule scorer      p8/judge.py       U2 judge + rewriter
p8/classifier.py    U3 BERT + attention          p8/clip_score.py  U3 CLIP score
p8/imagegen.py      U4 sd-turbo                  p8/safety.py      U5 PII + bias checks
p8/dataset.py       synthetic training data      p8/pipeline.py    end-to-end flow
train_bert.py       fine-tuning                  evaluate.py       all measured results
data/               gold (30), weak (5), safety (38) and counterfactual (10) sets
MODEL_CARD.md       model card                   docs/             diagrams
```

## Results (Colab T4 run, 16 Nov 2026)
| | |
|---|---|
| BERT accuracy: synthetic test / hand-labelled gold | 0.940 / 0.800 |
| LLM-judge MAE vs gold: zero-shot → few-shot | 2.80 → 1.47 |
| 5 weak prompts: mean rule score before → after | 1.8 → 5.2 /10 |
| 5 weak prompts: mean LLM-judge score before → after | 2.0 → 9.0 /10 |
| PII / bias detector recall (precision 1.0 for both) | 0.846 / 0.867 |

Full tables: `results/summary.md`.

## Limitations (summary)
The BERT classifier is trained on synthetic, template-built prompts, so it can over-reward keywords. The rule scorer is keyword-based. The 1.5B judge can be inconsistent, and sd-turbo at 1 step gives modest image quality. See `MODEL_CARD.md` and the report for the measured failure cases.

## Licence
Code: MIT. Models keep their own licences: Qwen2.5 (Apache-2.0), BERT (Apache-2.0), CLIP (MIT), and sd-turbo (Stability AI Non-Commercial Research Community licence, **non-commercial use only**).
