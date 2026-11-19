"""Paired comparison of two run labels, matched by seed (e.g. direct fine-tuning vs. DAPT).

    python -m ner.compare runs/ --a bert --b bert+dapt [--out compare.md]

For every seed present in both groups: dev/test strict (IOB2) entity P/R/F1 of a and b and the
difference b - a; then mean ± sample std of the differences and how many seeds are positive.
The test strict F1 difference is also given for the fixed-last-epoch protocol (test_history).
The two groups must share the protocol (precision, selection metric, training hyper-parameters,
data limits and data hashes); otherwise the script exits with an error.
"""

import argparse
import json
import statistics
from pathlib import Path

# config keys that legitimately differ between the two groups
IGNORED_CONFIG = {"model", "label", "init_from", "seed", "output_dir", "data_dir", "save_model",
                  "keep_checkpoints", "save_total_limit", "logging_steps"}
METRICS = ("precision", "recall", "f1")


def load(runs_dir, label):
    runs = {}
    for path in sorted(Path(runs_dir).glob("**/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        if m["model"] != label:
            continue
        if m["seed"] in runs:
            raise SystemExit(f"{label}: two runs with seed {m['seed']} ({runs[m['seed']][1]} and {path})")
        runs[m["seed"]] = (m, path)
    if not runs:
        raise SystemExit(f"no metrics.json with model == {label!r} under {runs_dir}")
    return runs


def protocol(m):
    return {"precision": m.get("precision"), "select_metric": m["selection"]["metric"],
            "data_sha256": m["data"]["sha256"], "sizes_used": m["data"]["sizes_used"],
            "config": {k: v for k, v in m["config"].items() if k not in IGNORED_CONFIG},
            "base_checkpoint": [m["checkpoint"], m["revision"]]}


def check_protocols(a_runs, b_runs, a, b):
    ref = protocol(next(iter(a_runs.values()))[0])
    for label, runs in ((a, a_runs), (b, b_runs)):
        for seed, (m, path) in runs.items():
            proto = protocol(m)
            if proto != ref:
                diff = {k: (ref[k], proto[k]) for k in ref if ref[k] != proto[k]}
                if "config" in diff:
                    rc, pc = diff.pop("config")
                    diff |= {f"config.{k}": (rc.get(k), pc.get(k)) for k in rc.keys() | pc.keys()
                             if rc.get(k) != pc.get(k)}
                raise SystemExit(f"protocol mismatch ({label} seed {seed}, {path}) vs {a}: {diff}")


def pm(values):
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{100 * statistics.mean(values):+.2f} ± {100 * std:.2f}"


def compare(runs_dir, a, b):
    a_runs, b_runs = load(runs_dir, a), load(runs_dir, b)
    check_protocols(a_runs, b_runs, a, b)
    seeds = sorted(set(a_runs) & set(b_runs))
    if not seeds:
        raise SystemExit(f"no common seeds: {a} {sorted(a_runs)}, {b} {sorted(b_runs)}")
    unpaired = sorted(set(a_runs) ^ set(b_runs))
    p0 = protocol(a_runs[seeds[0]][0])
    lines = [f"Paired comparison `{b}` − `{a}` over seeds {', '.join(map(str, seeds))} "
             f"(precision {p0['precision']}, dev selection {p0['select_metric']}). "
             "Scores are percentages, seqeval strict IOB2 entity level; Δ = b − a.", ""]
    if unpaired:
        lines += [f"Seeds present in only one group (ignored): {unpaired}.", ""]
    for split, title in (("dev", "dev (selected epoch)"), ("test", "test, dev-selected epoch")):
        lines += [f"#### {title}", "",
                  "| seed | " + " | ".join(f"{g} {x}" for g in (a, b) for x in ("P", "R", "F1"))
                  + " | ΔP | ΔR | ΔF1 |", "|---:" + "|---:" * 9 + "|"]
        deltas = {k: [] for k in METRICS}
        for seed in seeds:
            sa, sb = a_runs[seed][0][split]["strict"], b_runs[seed][0][split]["strict"]
            cells = [f"{100 * s[k]:.2f}" for s in (sa, sb) for k in METRICS]
            for k in METRICS:
                deltas[k].append(sb[k] - sa[k])
            cells += [f"{100 * (sb[k] - sa[k]):+.2f}" for k in METRICS]
            lines.append(f"| {seed} | " + " | ".join(cells) + " |")
        lines.append("| mean ± std | " + " | " * 6 + " | ".join(pm(deltas[k]) for k in METRICS) + " |")
        lines.append("| seeds Δ > 0 | " + " | " * 6 + " | ".join(
            f"{sum(d > 0 for d in deltas[k])}/{len(seeds)}" for k in METRICS) + " |")
        lines.append("")

    lines += ["#### test, fixed last epoch (strict F1)", ""]
    paired = [s for s in seeds if a_runs[s][0].get("test_history") and b_runs[s][0].get("test_history")]
    if paired:
        lines += [f"| seed | {a} F1 | {b} F1 | ΔF1 |", "|---:|---:|---:|---:|"]
        deltas = []
        for seed in paired:
            fa = a_runs[seed][0]["test_history"][-1]["strict"]["f1"]
            fb = b_runs[seed][0]["test_history"][-1]["strict"]["f1"]
            deltas.append(fb - fa)
            lines.append(f"| {seed} | {100 * fa:.2f} | {100 * fb:.2f} | {100 * (fb - fa):+.2f} |")
        lines.append(f"| mean ± std | | | {pm(deltas)} ({sum(d > 0 for d in deltas)}/{len(paired)} > 0) |")
    else:
        lines.append("No paired runs with `test_history`.")
    missing = [s for s in seeds if s not in paired]
    if paired and missing:
        lines += ["", f"Seeds without `test_history` in one group (skipped): {missing}."]

    inits = {(b_runs[s][0].get("init_from") or {}).get("path") for s in seeds}
    lines += ["", f"`{b}` init_from: {', '.join(str(i) for i in sorted(inits, key=str))}"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs_dir")
    p.add_argument("--a", required=True, help="reference label, e.g. bert")
    p.add_argument("--b", required=True, help="compared label, e.g. bert+dapt")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    table = compare(args.runs_dir, args.a, args.b)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
