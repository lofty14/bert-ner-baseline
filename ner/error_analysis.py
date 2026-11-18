"""Entity-level error analysis of two run labels over the same seeds.

    python -m ner.error_analysis runs/v2 --a bert --b bert+dapt3ep [--split test] [--out FILE]

Reads runs/<root>/<label>/seed<k>/<split>_predictions.jsonl (one sentence per line: id, tokens, gold,
pred) for every seed present in both labels. Entities are extracted with strict IOB2 rules (the same
spans as seqeval mode='strict', scheme=IOB2): an entity starts at B-X and extends over the following
I-X tokens; an I- tag that does not continue an entity of the same type is not an entity. For every
seed the counts are checked against the strict precision / recall stored in that run's metrics.json.

Per label and seed: TP / FP / FN, FP split into boundary errors (overlaps a gold entity, different
span) and spurious predictions (no overlap with any gold entity), FN split into boundary errors
(overlaps a predicted entity) and complete misses. Then the entities that are stable across seeds:
false positives / true positives predicted by b in >= 2 seeds and by a in no seed, and vice versa.
Standard library only (no torch, no seqeval).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def entities(tags):
    """Strict IOB2 -> set of (start, end_exclusive, type)."""
    out, start, etype = set(), None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, typ = tag.partition("-")
        if start is not None and not (prefix == "I" and typ == etype):
            out.add((start, i, etype))
            start, etype = None, None
        if prefix == "B":
            start, etype = i, typ
    return out


def overlaps(e, others):
    return [o for o in others if o[0] < e[1] and e[0] < o[1] and o[2] == e[2]]


def nearest(e, others):
    """Overlapping entity with the largest overlap, else the closest one in token distance, else None."""
    ov = overlaps(e, others)
    if ov:
        return max(ov, key=lambda o: (min(e[1], o[1]) - max(e[0], o[0]), -o[0]))
    if not others:
        return None
    return min(others, key=lambda o: (max(o[0] - e[1], e[0] - o[1], 0), o[0]))


def load_run(root, label, seed, split):
    run = Path(root) / label / f"seed{seed}"
    rows = [json.loads(line) for line in (run / f"{split}_predictions.jsonl").open(encoding="utf-8")]
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return {r["id"]: r for r in rows}, metrics


def seeds_of(root, label):
    return {int(p.name[4:]) for p in (Path(root) / label).glob("seed*") if p.name[4:].isdigit()}


def analyse_seed(rows):
    """-> counts, {(sid, s, e, t)} of TP and FP keys."""
    c = Counter()
    tp_keys, fp_keys = set(), set()
    for sid, r in rows.items():
        g, p = entities(r["gold"]), entities(r["pred"])
        c["gold"] += len(g)
        c["pred"] += len(p)
        for e in p:
            if e in g:
                c["TP"] += 1
                tp_keys.add((sid, *e))
            else:
                c["FP"] += 1
                c["FP_boundary" if overlaps(e, g) else "FP_spurious"] += 1
                fp_keys.add((sid, *e))
        for e in g - p:
            c["FN"] += 1
            c["FN_boundary" if overlaps(e, p) else "FN_missed"] += 1
    return c, tp_keys, fp_keys


def check_against_metrics(label, seed, split, c, metrics):
    m = metrics[split]["strict"]
    prec = c["TP"] / c["pred"] if c["pred"] else 0.0
    rec = c["TP"] / c["gold"] if c["gold"] else 0.0
    if abs(prec - m["precision"]) > 1e-9 or abs(rec - m["recall"]) > 1e-9:
        raise SystemExit(f"{label} seed{seed} {split}: counts give P={prec:.6f} R={rec:.6f}, "
                         f"metrics.json has P={m['precision']:.6f} R={m['recall']:.6f}")


def span_text(tokens, e):
    return " ".join(tokens[e[0]:e[1]]) if e else "none"


COLS = ["TP", "FP", "FP_boundary", "FP_spurious", "FN", "FN_boundary", "FN_missed"]
HEAD = ("| label | seed | TP | FP | FP boundary | FP spurious | FN | FN boundary | FN missed "
        "| P | R | F1 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def row(label, seed, c):
    p = c["TP"] / (c["TP"] + c["FP"]) if c["TP"] + c["FP"] else 0.0
    r = c["TP"] / (c["TP"] + c["FN"]) if c["TP"] + c["FN"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return (f"| {label} | {seed} | " + " | ".join(str(c[k]) for k in COLS)
            + f" | {100 * p:.2f} | {100 * r:.2f} | {100 * f:.2f} |")


def stable_section(title, keys_from, keys_other, n_seeds, rows, min_seeds=2):
    """Entities in >= min_seeds seeds of keys_from (list of sets per seed) and in no seed of keys_other."""
    count = Counter(k for s in keys_from for k in s)
    other = set().union(*keys_other)
    picked = sorted((k for k, n in count.items() if n >= min_seeds and k not in other),
                    key=lambda k: (-count[k], k))
    lines = [f"#### {title}: {len(picked)}", ""]
    if not picked:
        return lines + ["(none)", ""]
    lines += ["| seeds | id | tokens | predicted span | nearest gold span | sentence |", "|---:|---:|---|---|---|---|"]
    for k in picked:
        sid, e = k[0], k[1:]
        r = rows[sid]
        gold = entities(r["gold"])
        near = None if e in gold else nearest(e, gold)
        near_txt = "(exact match)" if e in gold else span_text(r["tokens"], near)
        sent = " ".join(r["tokens"]).replace("|", "\\|")
        lines.append(f"| {count[k]}/{n_seeds} | {sid} | {e[0]}-{e[1]} | {span_text(r['tokens'], e).replace('|', chr(92) + '|')} "
                     f"| {near_txt.replace('|', chr(92) + '|')} | {sent} |")
    return lines + [""]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs_dir")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--out", default=None, help="write markdown here (default: stdout only)")
    args = p.parse_args(argv)

    seeds = sorted(seeds_of(args.runs_dir, args.a) & seeds_of(args.runs_dir, args.b))
    if not seeds:
        raise SystemExit("no common seeds")
    ref_rows = None
    res = {}  # label -> list of (seed, counts, tp, fp)
    for label in (args.a, args.b):
        res[label] = []
        for seed in seeds:
            rows, metrics = load_run(args.runs_dir, label, seed, args.split)
            if ref_rows is None:
                ref_rows = rows
            elif rows.keys() != ref_rows.keys() or any(
                    rows[i]["tokens"] != ref_rows[i]["tokens"] or rows[i]["gold"] != ref_rows[i]["gold"]
                    for i in rows):
                raise SystemExit(f"{label} seed{seed}: sentences / gold labels differ from the reference run")
            c, tp, fp = analyse_seed(rows)
            check_against_metrics(label, seed, args.split, c, metrics)
            res[label].append((seed, c, tp, fp))

    n = len(seeds)
    out = [f"Entity-level error analysis, `{args.b}` vs `{args.a}`, split `{args.split}`, seeds "
           f"{', '.join(map(str, seeds))}. Strict IOB2 entities (as seqeval mode='strict'); counts are "
           f"checked against each run's metrics.json. {len(ref_rows)} sentences, "
           f"{res[args.a][0][1]['gold']} gold entities.",
           "",
           "FP boundary = predicted entity overlaps a gold entity but the span differs; FP spurious = no "
           "overlap with any gold entity. FN boundary = gold entity overlaps a predicted entity; FN missed "
           "= no overlap. One predicted span can overlap two gold spans (and vice versa), so FP boundary "
           "and FN boundary need not be equal. P / R / F1 in the `all` rows are pooled over seeds (micro), so they "
           "can differ in the second decimal from the mean over seeds in results.md. `tokens` = start-end "
           "(end exclusive) token offsets.",
           "", "### Totals over seeds", "", HEAD]
    totals = {}
    for label in (args.a, args.b):
        t = Counter()
        for _, c, _, _ in res[label]:
            t.update(c)
        totals[label] = t
        out.append(row(label, "all", t))
    d = Counter({k: totals[args.b][k] - totals[args.a][k] for k in COLS})
    out.append(f"| Δ (b − a) | all | " + " | ".join(f"{d[k]:+d}" for k in COLS) + " | | | |")
    out += ["", "### Per seed", "", HEAD]
    for label in (args.a, args.b):
        for seed, c, _, _ in res[label]:
            out.append(row(label, seed, c))
    out += ["", "### Stable differences", "",
            f"Entities predicted by one label in at least 2 of {n} seeds and by the other label in none.",
            "Nearest gold span: the overlapping gold entity with the largest overlap, else the closest gold "
            "entity in the same sentence, else none.", ""]
    a_tp, a_fp = [s[2] for s in res[args.a]], [s[3] for s in res[args.a]]
    b_tp, b_fp = [s[2] for s in res[args.b]], [s[3] for s in res[args.b]]
    out += stable_section(f"false positives new in `{args.b}`", b_fp, a_fp, n, ref_rows)
    out += stable_section(f"false positives gone in `{args.b}` (only in `{args.a}`)", a_fp, b_fp, n, ref_rows)
    out += stable_section(f"true positives new in `{args.b}`", b_tp, a_tp, n, ref_rows)
    out += stable_section(f"true positives lost in `{args.b}` (only in `{args.a}`)", a_tp, b_tp, n, ref_rows)
    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
