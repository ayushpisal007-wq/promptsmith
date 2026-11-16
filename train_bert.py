"""U3 - Fine-tune BERT (bert-base-uncased) to predict prompt quality: weak / average / strong.

Plain PyTorch training loop on the synthetic rubric-labelled dataset, then evaluation on
(a) the held-out synthetic test split and (b) the 30 hand-labelled gold prompts (out-of-distribution).

    python train_bert.py                       # GPU (Colab T4): ~3 min. CPU: ~25-40 min.
    python train_bert.py --push your-hf-username/p8-bert-prompt-quality
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from p8.dataset import generate
from p8.rubric import CRITERIA, band

LABELS = ["weak", "average", "strong"]
L2I = {l: i for i, l in enumerate(LABELS)}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def batches(tok, texts, labels, bs, shuffle):
    idx = list(range(len(texts)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([texts[k] for k in j], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc["labels"] = torch.tensor([labels[k] for k in j])
        yield enc


@torch.no_grad()
def predict(model, tok, texts, device, bs=64):
    model.eval()
    preds, probs = [], []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        p = torch.softmax(model(**enc).logits, -1).cpu()
        probs += p.tolist()
        preds += p.argmax(-1).tolist()
    return preds, probs


def report(name, y, p):
    acc, f1 = accuracy_score(y, p), f1_score(y, p, average="macro")
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    print(f"\n== {name}: accuracy={acc:.3f} macro-F1={f1:.3f}")
    print(classification_report(y, p, labels=[0, 1, 2], target_names=LABELS, zero_division=0))
    print(pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS]))
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "confusion_matrix": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--out", default="models/bert-prompt-quality")
    ap.add_argument("--push", default="", help="optional Hugging Face Hub repo id to upload the model to")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.DataFrame(generate(args.n, args.seed))
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_prompts.csv", index=False)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_tr, n_va = int(0.8 * len(df)), int(0.1 * len(df))
    tr, va, te = df[:n_tr], df[n_tr:n_tr + n_va], df[n_tr + n_va:]
    print(f"train={len(tr)} val={len(va)} test={len(te)} device={device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=L2I,
        ignore_mismatched_sizes=True).to(device)

    Xtr, ytr = tr.prompt.tolist(), tr.label.map(L2I).tolist()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(Xtr) + args.bs - 1) // args.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        losses = []
        for b in batches(tok, Xtr, ytr, args.bs, True):
            b = b.to(device)
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        vp, _ = predict(model, tok, va.prompt.tolist(), device)
        vacc = accuracy_score(va.label.map(L2I), vp)
        history.append({"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 4), "val_acc": round(vacc, 4)})
        print(history[-1])
    train_time = time.time() - t0

    metrics = {"base_model": args.model, "epochs": args.epochs, "train_size": len(tr),
               "train_seconds": round(train_time, 1), "history": history}
    tp, _ = predict(model, tok, te.prompt.tolist(), device)
    metrics["synthetic_test"] = report("synthetic test split", te.label.map(L2I).tolist(), tp)

    gold = pd.read_csv("data/gold_prompts.csv")
    gold["gold_band"] = gold[CRITERIA].sum(axis=1).map(band)
    gp, gprob = predict(model, tok, gold.prompt.tolist(), device)
    metrics["gold_set"] = report("hand-labelled gold set (out-of-distribution)", gold.gold_band.map(L2I).tolist(), gp)
    gold["bert_pred"] = [LABELS[i] for i in gp]
    gold["bert_p_strong"] = [round(p[2], 3) for p in gprob]
    os.makedirs("results", exist_ok=True)
    gold[["id", "kind", "prompt", "gold_band", "bert_pred", "bert_p_strong"]].to_csv("results/bert_gold_predictions.csv", index=False)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open("results/bert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, key, title in [(axes[0], "synthetic_test", "Synthetic test"), (axes[1], "gold_set", "Gold set (OOD)")]:
            cm = np.array(metrics[key]["confusion_matrix"])
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(3), LABELS)
            ax.set_yticks(range(3), LABELS)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"{title}: acc {metrics[key]['accuracy']:.2f}")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig("results/bert_confusion_matrices.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(f"\nSaved model to {args.out}; metrics to results/bert_metrics.json")

    if args.push:
        model.push_to_hub(args.push)
        tok.push_to_hub(args.push)
        print(f"Pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
