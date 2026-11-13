"""Summarise runs/**/metrics.json as mean ± std (sample std, n-1) markdown tables.

    python -m ner.aggregate runs/ [--out results.md]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ORDER = ["bert", "biobert", "pubmedbert"]


def fmt(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{100 * mean:.2f} ± {100 * std:.2f}"


def aggregate(runs_dir):
    groups = defaultdict(list)
    for path in sorted(Path(runs_dir).glob("**/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        groups[(m["model"], m["data"]["limited"])].append(m)
    if not groups:
        return f"No metrics.json found under {runs_dir}"

    keys = sorted(groups, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[0], k[1]))
    lines = ["Scores are percentages, mean ± sample std over seeds. "
             "`default` = seqeval default mode, `strict` = mode='strict', scheme=IOB2.", ""]
    for split in ("dev", "test"):
        lines += [f"### {split}", "",
                  "| model | n seeds | P (default) | R (default) | F1 (default) | P (strict) | R (strict) "
                  "| F1 (strict) | token acc |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key in keys:
            runs = groups[key]
            name = key[0] + (" (limited/smoke)" if key[1] else "")
            cells = [fmt([r[split][mode][metric] for r in runs])
                     for mode in ("default", "strict") for metric in ("precision", "recall", "f1")]
            cells.append(fmt([r[split]["token_accuracy"] for r in runs]))
            seeds = ",".join(str(r["seed"]) for r in sorted(runs, key=lambda r: r["seed"]))
            lines.append(f"| {name} | {len(runs)} ({seeds}) | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["### model selection", "", "| model | best dev epoch per seed |", "|---|---|"]
    for key in keys:
        epochs = ", ".join(f"s{r['seed']}:{r['selection']['best_epoch']:g}"
                           for r in sorted(groups[key], key=lambda r: r["seed"]))
        lines.append(f"| {key[0]}{' (limited/smoke)' if key[1] else ''} | {epochs} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs_dir", nargs="?", default="runs")
    p.add_argument("--out", default=None, help="also write the markdown to this file")
    args = p.parse_args(argv)
    table = aggregate(args.runs_dir)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
