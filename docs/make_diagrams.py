"""Draw the block diagram and the GenAI lifecycle diagram used in the README and report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C = {"U1": "#DCEAF7", "U2": "#E3F1DF", "U3": "#FBE9D0", "U4": "#EDE3F6", "U5": "#F8DCDC", "io": "#F2F2F2"}


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#444", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#333"))


def block_diagram(path="docs/block_diagram.png"):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    box(ax, 0.2, 2.6, 1.6, 1.0, "User prompt\n(text or image)", C["io"])
    box(ax, 2.3, 2.6, 1.9, 1.0, "U5 guardrail\nPII redaction +\nbias-wording check", C["U5"])
    box(ax, 4.8, 4.6, 2.2, 1.0, "U1 rule-based rubric\nrole · context · task ·\nformat · constraints", C["U1"])
    box(ax, 4.8, 2.6, 2.2, 1.0, "U2 LLM judge\n(Qwen2.5, zero-/few-shot)", C["U2"])
    box(ax, 4.8, 0.6, 2.2, 1.0, "U3 BERT classifier\nweak / average / strong\n+ attention view", C["U3"])
    box(ax, 7.6, 2.6, 1.8, 1.0, "LLM rewriter\n(fills missing\ncriteria)", C["U2"])
    box(ax, 9.9, 4.2, 1.9, 1.1, "U4 text: run original\nvs rewritten on LLM", C["U4"])
    box(ax, 9.9, 1.0, 1.9, 1.3, "U4 image: sd-turbo\n(same seed) +\nU3 CLIP score", C["U4"])
    box(ax, 7.6, 0.2, 1.8, 0.9, "Scores before → after\n(re-graded)", C["io"], 8)
    arrow(ax, 1.8, 3.1, 2.3, 3.1)
    for y in (5.1, 3.1, 1.1):
        arrow(ax, 4.2, 3.1, 4.8, y)
    for y in (5.1, 3.1, 1.1):
        arrow(ax, 7.0, y, 7.6, 3.1)
    arrow(ax, 9.4, 3.3, 9.9, 4.6)
    arrow(ax, 9.4, 2.9, 9.9, 1.8)
    arrow(ax, 8.5, 2.6, 8.5, 1.1)
    ax.text(6, 6.05, "P8 · Prompt Grader & Rewriter: block diagram", ha="center", fontsize=12, weight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")


def lifecycle(path="docs/lifecycle.png"):
    steps = ["1. Problem\nweak student prompts", "2. Data\nsynthetic labelled\nprompts + 30 gold",
             "3. Model choice\nQwen2.5 (generative)\nBERT (discriminative)", "4. Adapt\nfew-shot prompting\n+ BERT fine-tuning",
             "5. Evaluate\nMAE, accuracy, CLIP,\nbias & PII tests", "6. Deploy\nGradio on\nHF Spaces",
             "7. Monitor\nfailure cases,\nfeedback → data"]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 13.3)
    ax.set_ylim(-0.6, 2.6)
    ax.axis("off")
    colors = [C["io"], C["U3"], C["U2"], C["U2"], C["U1"], C["U4"], C["U5"]]
    for i, (s, col) in enumerate(zip(steps, colors)):
        x = 0.15 + i * 1.88
        box(ax, x, 0.6, 1.6, 1.3, s, col, 8)
        if i:
            arrow(ax, x - 0.28, 1.25, x, 1.25)
    ax.add_patch(FancyArrowPatch((12.3, 0.6), (1.0, 0.6), connectionstyle="arc3,rad=-0.12",
                                 arrowstyle="-|>", mutation_scale=12, lw=1, color="#777", ls="--"))
    ax.text(6.65, -0.35, "feedback loop: failure cases become new training data", ha="center", fontsize=8, color="#555")
    ax.text(6.65, 2.4, "Generative-AI project lifecycle applied to P8", ha="center", fontsize=11, weight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    block_diagram()
    lifecycle()
    print("saved docs/block_diagram.png and docs/lifecycle.png")
