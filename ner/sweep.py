"""Run ner.train for every (model, seed) pair, one after another, then aggregate.

    python -m ner.sweep --models bert biobert pubmedbert --seeds 1 2 3
    python -m ner.sweep --models bert --seeds 1 2 -- --epochs 5 --warmup_ratio 0.1   # extra args go to ner.train
    python -m ner.sweep --models bert --seeds 1 2 3 --init_from runs_dapt/bert_dapt/model   # -> runs/bert+dapt/

Each run is a separate subprocess (clean memory, independent seeding). Runs whose
metrics.json already exists are skipped unless --overwrite is given.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from .aggregate import aggregate
from .models import MODELS

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=sorted(MODELS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--runs_dir", default=str(REPO_ROOT / "runs"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--init_from", default=None, help="passed to ner.train (needs a single --models)")
    p.add_argument("--label", default=None, help="passed to ner.train (needs a single --models); "
                   "default: the model, or <model>+dapt with --init_from")
    args, train_args = p.parse_known_args(argv)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    if (args.init_from or args.label) and len(args.models) != 1:
        p.error("--init_from / --label need exactly one --models")
    extra = []
    if args.init_from:
        extra += ["--init_from", str(Path(args.init_from).resolve())]
    if args.label:
        extra += ["--label", args.label]

    failed = []
    for model in args.models:
        name = args.label or (f"{model}+dapt" if args.init_from else model)
        for seed in args.seeds:
            out_dir = Path(args.runs_dir) / name / f"seed{seed}"
            if (out_dir / "metrics.json").exists() and not args.overwrite:
                print(f"skip {name} seed{seed}: {out_dir / 'metrics.json'} exists")
                continue
            cmd = [sys.executable, "-m", "ner.train", "--model", model, "--seed", str(seed),
                   "--output_dir", str(out_dir), *extra, *train_args]
            print("\n$ " + " ".join(cmd), flush=True)
            if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
                failed.append(f"{name}/seed{seed}")
    print("\n" + aggregate(args.runs_dir))
    if failed:
        print("FAILED runs: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
