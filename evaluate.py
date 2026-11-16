"""Measured results for the report (run on Colab GPU, or locally once models download).

    python evaluate.py              # everything
    python evaluate.py --no-images  # skip sd-turbo / CLIP (faster)

Writes to results/:
  judge_comparison.csv  zero-shot vs few-shot LLM judge vs rules vs BERT on the 30 gold prompts  (U1/U2)
  gold_scores.csv       per-prompt scores from every grader
  before_after.csv      the 5 weak prompts before vs after rewriting (P8 deliverable)  (U4)
  outputs/              text outputs and images for the original and rewritten prompts
  safety_eval.csv       PII / bias detector precision and recall + the misses  (U5)
  counterfactual.csv    bias test: score change when only a name or gender is swapped  (U5)
  failure_cases.csv     the prompts with the largest grading errors
  summary.md            all tables in markdown, ready to paste into the report
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from sklearn.metrics import precision_score, recall_score

from p8 import classifier, judge
from p8.pipeline import analyze, run_both
from p8.rubric import CRITERIA, band, rule_score, total
from p8.safety import find_bias, find_pii

R = "results"


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)


def eval_graders(gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gold = gold.copy()
    gold["gold_total"] = gold[CRITERIA].sum(axis=1)
    gold["gold_band"] = gold.gold_total.map(band)
    rows = []
    for _, g in gold.iterrows():
        r = {"id": g.id, "kind": g.kind, "gold_total": g.gold_total, "gold_band": g.gold_band}
        rs = rule_score(g.prompt, g.kind)
        r["rule_total"] = total(rs)
        for mode, fs in [("zero", False), ("few", True)]:
            j = judge.judge(g.prompt, g.kind, few_shot=fs)
            r[f"{mode}_total"] = j["total"]
            r[f"{mode}_parsed"] = j["parsed_ok"]
            r[f"{mode}_latency"] = round(j["latency"], 2)
            for c in CRITERIA:
                r[f"{mode}_{c}"] = j["scores"][c]
        if classifier.available():
            b = classifier.predict(g.prompt)
            r["bert_band"] = b["label"]
            r["bert_p_strong"] = round(b["probs"].get("strong", 0), 3)
        rows.append(r)
        print(g.id, r["gold_total"], r["rule_total"], r["zero_total"], r["few_total"], r.get("bert_band"))
    s = pd.DataFrame(rows)
    comp = []
    for name, col in [("Rule-based rubric (U1)", "rule_total"), ("LLM judge, zero-shot (U2)", "zero_total"),
                      ("LLM judge, few-shot (U2)", "few_total")]:
        comp.append({
            "grader": name,
            "MAE total (/10)": round((s[col] - s.gold_total).abs().mean(), 2),
            "band accuracy": round((s[col].map(band) == s.gold_band).mean(), 3),
            "Pearson r": round(s[col].corr(s.gold_total), 3),
            "parse rate": 1.0 if col == "rule_total" else round(s[col.replace("total", "parsed")].mean(), 3),
            "mean latency s": 0.0 if col == "rule_total" else round(s[col.replace("total", "latency")].mean(), 2),
        })
    if "bert_band" in s:
        comp.append({"grader": "BERT classifier (U3)", "MAE total (/10)": "-",
                     "band accuracy": round((s.bert_band == s.gold_band).mean(), 3), "Pearson r": "-",
                     "parse rate": 1.0, "mean latency s": "-"})
    # per-criterion MAE for the judge (where does it disagree with the rubric?)
    for mode in ["zero", "few"]:
        for c in CRITERIA:
            s[f"{mode}_{c}_err"] = (s[f"{mode}_{c}"].fillna(0) - gold[c].values).abs()
    return s, pd.DataFrame(comp)


def eval_before_after(weak: pd.DataFrame, images: bool) -> pd.DataFrame:
    os.makedirs(f"{R}/outputs", exist_ok=True)
    rows = []
    for _, w in weak.iterrows():
        t0 = time.perf_counter()
        a = analyze(w.prompt, w.kind, few_shot=True, judge_rewritten=True)
        b, af = a["before"], a["after"]
        row = {
            "id": w.id, "kind": w.kind, "original": w.prompt, "rewritten": a["rewritten"],
            "PII found": ", ".join(p["type"] for p in a["safety"]["pii"]) or "-",
            "bias found": ", ".join(x["term"] for x in a["safety"]["bias"]) or "-",
            "rule before": b["rule_total"], "rule after": af["rule_total"],
            "judge before": b.get("judge_total"), "judge after": af.get("judge_total"),
            "BERT before": b.get("bert_label"), "BERT after": af.get("bert_label"),
            "P(strong) before": round(b.get("bert_probs", {}).get("strong", float("nan")), 3),
            "P(strong) after": round(af.get("bert_probs", {}).get("strong", float("nan")), 3),
            "rewrite still flagged": (not a["after_safety"]["pii_pass"]) or (not a["after_safety"]["bias_pass"]),
        }
        if w.kind == "text" or images:
            out = run_both(w.prompt, a["rewritten"], w.kind)
            if out["kind"] == "image":
                out["image_original"].save(f"{R}/outputs/{w.id}_original.png")
                out["image_rewritten"].save(f"{R}/outputs/{w.id}_rewritten.png")
                row.update({"CLIP vs intent before": out["clip_original_vs_intent"],
                            "CLIP vs intent after": out["clip_rewritten_vs_intent"],
                            "CLIP vs rewritten text: orig img": out["clip_original_vs_rewritten_text"],
                            "CLIP vs rewritten text: new img": out["clip_rewritten_vs_own"]})
            else:
                for tag in ["original", "rewritten"]:
                    with open(f"{R}/outputs/{w.id}_{tag}.txt", "w", encoding="utf-8") as f:
                        f.write(out[f"output_{tag}"])
                row.update({"words before": out["words_original"], "words after": out["words_rewritten"]})
        row["seconds"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(row["id"], row["rule before"], "->", row["rule after"], "| judge", row["judge before"], "->", row["judge after"])
    return pd.DataFrame(rows)


def eval_safety(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["pred_pii"] = [int(bool(find_pii(t))) for t in df.text]
    df["pred_bias"] = [int(bool(find_bias(t))) for t in df.text]
    m = []
    for name, y, p in [("PII detector", "has_pii", "pred_pii"), ("Bias-wording detector", "has_bias", "pred_bias")]:
        m.append({"check": name, "precision": round(precision_score(df[y], df[p], zero_division=0), 3),
                  "recall": round(recall_score(df[y], df[p], zero_division=0), 3),
                  "missed": int(((df[y] == 1) & (df[p] == 0)).sum()),
                  "false alarms": int(((df[y] == 0) & (df[p] == 1)).sum()), "n": len(df)})
    misses = df[(df.has_pii != df.pred_pii) | (df.has_bias != df.pred_bias)]
    return pd.DataFrame(m), misses


def eval_counterfactual(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in pairs.iterrows():
        ja, jb = judge.judge(p.prompt_a, "text"), judge.judge(p.prompt_b, "text")
        r = {"id": p.id, "prompt_a": p.prompt_a, "prompt_b": p.prompt_b,
             "judge_a": ja["total"], "judge_b": jb["total"], "judge_diff": abs(ja["total"] - jb["total"]),
             "rule_diff": abs(total(rule_score(p.prompt_a, "text")) - total(rule_score(p.prompt_b, "text")))}
        if classifier.available():
            r["bert_p_strong_diff"] = round(abs(classifier.predict(p.prompt_a)["probs"]["strong"]
                                                - classifier.predict(p.prompt_b)["probs"]["strong"]), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--skip-gold", action="store_true")
    args = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    out = ["# P8 results\n"]

    safety_m, misses = eval_safety(pd.read_csv("data/safety_testset.csv"))
    safety_m.to_csv(f"{R}/safety_eval.csv", index=False)
    misses.to_csv(f"{R}/safety_misses.csv", index=False)
    out += ["## Safety checks (U5)\n", md(safety_m), "\nMissed / wrong:\n", md(misses[["id", "text", "has_pii", "pred_pii", "has_bias", "pred_bias"]])]

    if not args.skip_gold:
        scores, comp = eval_graders(pd.read_csv("data/gold_prompts.csv"))
        scores.to_csv(f"{R}/gold_scores.csv", index=False)
        comp.to_csv(f"{R}/judge_comparison.csv", index=False)
        out += ["\n## Graders vs 30 hand-labelled gold prompts (U1, U2, U3)\n", md(comp)]
        fails = scores.assign(err=(scores.few_total - scores.gold_total).abs()).sort_values("err", ascending=False).head(5)
        gold = pd.read_csv("data/gold_prompts.csv").set_index("id")
        fails["prompt"] = [gold.loc[i, "prompt"] for i in fails.id]
        cols = ["id", "prompt", "gold_total", "rule_total", "zero_total", "few_total"] + (["bert_band"] if "bert_band" in fails else [])
        fails[cols].to_csv(f"{R}/failure_cases.csv", index=False)
        out += ["\n## Largest grading errors (failure cases)\n", md(fails[cols])]

    ba = eval_before_after(pd.read_csv("data/weak_prompts.csv"), images=not args.no_images)
    ba.to_csv(f"{R}/before_after.csv", index=False)
    out += ["\n## Before / after for the 5 weak prompts (P8 deliverable)\n", md(ba.drop(columns=["original", "rewritten"])),
            "\n### Rewritten prompts\n"] + [f"- **{r.id}** `{r.original}` → {r.rewritten}" for r in ba.itertuples()]

    cf = eval_counterfactual(pd.read_csv("data/counterfactual_pairs.csv"))
    cf.to_csv(f"{R}/counterfactual.csv", index=False)
    out += ["\n## Counterfactual bias test (U5): swap only a name or gender\n",
            f"Mean |Δ judge score| = {cf.judge_diff.mean():.2f} /10 · pairs with any change: {(cf.judge_diff > 0).sum()}/{len(cf)}"
            + (f" · mean |Δ BERT P(strong)| = {cf.bert_p_strong_diff.mean():.3f}" if "bert_p_strong_diff" in cf else ""),
            "", md(cf.drop(columns=["prompt_a", "prompt_b"]))]

    with open(f"{R}/summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
