"""Summarise runs/**/metrics.json as mean ± std (sample std, n-1) markdown tables.

    python -m ner.aggregate runs/ [--out results.md]

Runs are grouped by protocol (training precision + dev selection metric) and then by model.
Two test protocols are reported, both with seqeval strict (IOB2) and default scores:
  * dev-selected: the epoch picked on dev by the selection metric (metrics.json "test");
  * fixed last epoch: the final evaluated epoch (metrics.json "test_history"[-1]); runs written
    before test_history existed are skipped there and listed.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ORDER = ["bert", "biobert", "pubmedbert"]
SCORE_HEADER = ("| P (default) | R (default) | F1 (default) | P (strict) | R (strict) | F1 (strict) | token acc |",
                "|---:|---:|---:|---:|---:|---:|---:|")


def fmt(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{100 * mean:.2f} ± {100 * std:.2f}"


def protocol(m):
    """(precision, select_metric) of a run; v1 runs (before these fields existed) are labelled as such."""
    precision = m.get("precision") or "unrecorded (v1: bf16 if the CUDA GPU reported bf16, else fp32)"
    select = m["selection"]["metric"]
    if select == "dev f1 (seqeval default)":  # v1 wording
        select = "default_f1"
    return precision, select


def score_cells(scores):
    cells = [fmt([s[mode][metric] for s in scores])
             for mode in ("default", "strict") for metric in ("precision", "recall", "f1")]
    cells.append(fmt([s["token_accuracy"] for s in scores]))
    return cells


def seeds_cell(runs):
    return f"{len(runs)} ({','.join(str(r['seed']) for r in runs)})"


def model_key(k):
    return ORDER.index(k[0]) if k[0] in ORDER else 99, k[0], k[1]


def label(key):
    return key[0] + (" (limited/smoke)" if key[1] else "")


def protocol_section(groups):
    keys = sorted(groups, key=model_key)
    lines = []
    for split, title in (("dev", "dev (selected epoch)"), ("test", "test, dev-selected epoch")):
        lines += [f"#### {title}", "", "| model | n seeds " + SCORE_HEADER[0], "|---|---:" + SCORE_HEADER[1]]
        for key in keys:
            runs = groups[key]
            lines.append(f"| {label(key)} | {seeds_cell(runs)} | " + " | ".join(score_cells([r[split] for r in runs])) + " |")
        lines.append("")

    lines += ["#### test, fixed last epoch", ""]
    if any(r.get("test_history") for runs in groups.values() for r in runs):
        lines += ["| model | n seeds | last epoch " + SCORE_HEADER[0], "|---|---:|---:" + SCORE_HEADER[1]]
    skipped = []
    for key in keys:
        runs = [r for r in groups[key] if r.get("test_history")]
        skipped += [f"{label(key)} s{r['seed']}" for r in groups[key] if not r.get("test_history")]
        if not runs:
            continue
        last = [r["test_history"][-1] for r in runs]
        epochs = ",".join(f"{h['epoch']:g}" for h in last)
        lines.append(f"| {label(key)} | {seeds_cell(runs)} | {epochs} | " + " | ".join(score_cells(last)) + " |")
    if skipped:
        lines += ([""] if lines[-1] else []) + [
            f"No `test_history` (run predates per-epoch test scoring), skipped: {', '.join(skipped)}."]
    lines.append("")

    lines += ["#### model selection", "", "| model | best dev epoch per seed | epochs run |", "|---|---|---|"]
    for key in keys:
        runs = groups[key]
        best = ", ".join(f"s{r['seed']}:{r['selection']['best_epoch']:g}" for r in runs)
        done = ", ".join(f"s{r['seed']}:{r['selection']['epochs_completed']:g}" for r in runs)
        lines.append(f"| {label(key)} | {best} | {done} |")
    lines.append("")
    return lines


def aggregate(runs_dir):
    by_protocol = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(runs_dir).glob("**/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        by_protocol[protocol(m)][(m["model"], m["data"]["limited"])].append(m)
    if not by_protocol:
        return f"No metrics.json found under {runs_dir}"
    for groups in by_protocol.values():
        for runs in groups.values():
            runs.sort(key=lambda r: r["seed"])

    lines = ["Scores are percentages, mean ± sample std over seeds. "
             "`default` = seqeval default mode, `strict` = mode='strict', scheme=IOB2.", ""]
    if len(by_protocol) > 1:
        lines += [f"**WARNING: {runs_dir} mixes {len(by_protocol)} protocols (precision / selection metric); "
                  "they are reported separately below and must not be pooled.**", ""]
    for precision, select in sorted(by_protocol):
        lines += [f"### precision: {precision}; dev selection metric: {select}", ""]
        lines += protocol_section(by_protocol[(precision, select)])
    return "\n".join(lines).rstrip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs_dir", nargs="?", default="runs")
    p.add_argument("--out", default=None, help="also write the markdown to this file")
    args = p.parse_args(argv)
    table = aggregate(args.runs_dir)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
