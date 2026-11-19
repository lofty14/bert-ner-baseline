"""Figures for the domain-adaptation experiment (needs matplotlib).

    python -m ner.plot_dapt [--runs_dir runs/v2] [--dapt_dir runs_dapt] [--out_dir dapt/figs]

mlm_valid_loss.png  validation masked-LM loss vs. step for the 1- and 3-epoch DAPT runs (dapt.json)
mlm_valid_loss_slide.png  the same without batch size / step counts in the labels (for the methods slide)
ner_f1_paired.png   test strict F1 per seed, lines join the same seed, mean ± 1 sample std
ner_pr_paired.png   the same for test strict precision and recall (two panels, same y span)

All numbers are read from runs_dapt/<name>/dapt.json and runs/v2/<label>/seed<k>/metrics.json
(`test` = the dev-selected epoch).
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUPS = [("bert", "Direct fine-tuning"), ("bert+dapt1ep", "+DAPT 1 epoch"), ("bert+dapt3ep", "+DAPT 3 epochs")]
DAPT_RUNS = [("bert_dapt_1ep", "DAPT 1 epoch", "#4C72B0", (4, -22), "left"),
             ("bert_dapt_3ep", "DAPT 3 epochs", "#C44E52", (0, 20), "right")]
SEED_COLORS = {1: "#7F7F7F", 2: "#8C8CB0", 3: "#A08C70"}

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 12,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False})


def load_scores(runs_dir):
    """-> {label: {seed: test strict {precision, recall, f1} in %}}"""
    out = {}
    for label, _ in GROUPS:
        out[label] = {}
        for run in sorted((Path(runs_dir) / label).glob("seed*")):
            m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            out[label][m["seed"]] = {k: 100 * v for k, v in m["test"]["strict"].items()}
    seeds = set.intersection(*(set(v) for v in out.values()))
    if not seeds:
        raise SystemExit("no seed common to all groups")
    return {lab: {s: v[s] for s in sorted(seeds)} for lab, v in out.items()}, sorted(seeds)


def paired_panel(ax, scores, seeds, metric, ylim, ylabel):
    xs = range(len(GROUPS))
    for seed in seeds:
        ys = [scores[lab][seed][metric] for lab, _ in GROUPS]
        ax.plot(xs, ys, "-o", color=SEED_COLORS.get(seed, "#999999"), lw=1, ms=6, alpha=0.9,
                label=f"seed {seed}", zorder=2)
    for i, (lab, _) in enumerate(GROUPS):
        vals = [scores[lab][s][metric] for s in seeds]
        mean, sd = statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0
        ax.errorbar(i + 0.18, mean, yerr=sd, fmt="D", color="black", ms=7, capsize=5, lw=1.5, zorder=3,
                    label="mean ± 1 SD" if i == 0 else None)
        ax.annotate(f"{mean:.2f} ± {sd:.2f}", (i + 0.18, mean), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=11)
    ax.set_xticks(list(xs), [name for _, name in GROUPS])
    ax.set_xlim(-0.35, len(GROUPS) - 0.35)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)


def plot_f1(scores, seeds, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    paired_panel(ax, scores, seeds, "f1", (83.5, 87.0), "Test strict F1 (%)")
    ax.set_title(f"NER test strict F1, seeds {', '.join(map(str, seeds))}")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_pr(scores, seeds, out):
    span = 4.5  # same y span on both panels so the two differences are visually comparable
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, name in zip(axes, ("precision", "recall"), ("precision", "recall")):
        v = [scores[lab][s][metric] for lab, _ in GROUPS for s in seeds]
        if max(v) - min(v) > span - 1.5:
            raise SystemExit(f"{metric}: y span {span} too small for the data")
        mid = (min(v) + max(v)) / 2
        lo = math.floor((mid - span / 2) * 2) / 2
        paired_panel(ax, scores, seeds, metric, (lo, lo + span), f"Test strict {name} (%)")
        ax.set_title(f"Test strict {name}")
    axes[0].legend(loc="lower left", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_mlm(dapt_dir, out, slide=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ppl_initial = None
    for name, label, color, offset, ha in DAPT_RUNS:
        d = json.loads((Path(dapt_dir) / name / "dapt.json").read_text(encoding="utf-8"))
        hist = d["valid_loss_history"]
        steps, losses = [h["step"] for h in hist], [h["loss"] for h in hist]
        ppl = d["valid_perplexity"]
        ppl_initial = ppl["initial"]
        ax.plot(steps, losses, "-o", color=color, lw=2, ms=5,
                label=(f"{label}: final PPL {ppl['final']:.2f}" if slide else
                       f"{label} ({d['steps']:,} steps): final PPL {ppl['final']:.2f}"))
        ax.annotate(f"PPL {ppl['final']:.2f}", (steps[-1], losses[-1]), xytext=offset,
                    textcoords="offset points", color=color, fontsize=12, va="center", ha=ha)
    ax.annotate(f"start: loss {math.log(ppl_initial):.3f}, PPL {ppl_initial:.2f}", (0, math.log(ppl_initial)),
                xytext=(12, 0), textcoords="offset points", fontsize=12, va="center")
    sec = ax.secondary_yaxis("right", functions=(lambda x: x, lambda x: x))
    lo, hi = ax.get_ylim()
    ticks = [p for p in (3.0, 3.5, 4.0, 5.0, 6.0, 7.0) if lo <= math.log(p) <= hi]
    sec.set_yticks([math.log(p) for p in ticks], [f"{p:g}" for p in ticks])
    sec.set_ylabel("Perplexity, exp(loss)")
    ax.set_xlabel("Continued pre-training step" if slide else
                  "Continued pre-training step (batch 32 x 256 tokens)")
    ax.set_ylabel("Validation masked-LM loss")
    ax.set_title("DAPT on PubMed abstracts: validation MLM loss")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.04)
    ax.grid(color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir", default="runs/v2")
    p.add_argument("--dapt_dir", default="runs_dapt")
    p.add_argument("--out_dir", default="dapt/figs")
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores, seeds = load_scores(args.runs_dir)
    for lab, _ in GROUPS:
        f1 = [scores[lab][s]["f1"] for s in seeds]
        print(f"{lab}: test strict F1 {statistics.mean(f1):.2f} ± {statistics.stdev(f1):.2f}")
    plot_mlm(args.dapt_dir, out / "mlm_valid_loss.png")
    plot_mlm(args.dapt_dir, out / "mlm_valid_loss_slide.png", slide=True)
    plot_f1(scores, seeds, out / "ner_f1_paired.png")
    plot_pr(scores, seeds, out / "ner_pr_paired.png")
    print(f"wrote {out}/mlm_valid_loss.png, mlm_valid_loss_slide.png, ner_f1_paired.png, ner_pr_paired.png")


if __name__ == "__main__":
    main()
