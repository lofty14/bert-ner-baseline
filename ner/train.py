"""Fine-tune one model with one seed on NCBI Disease and write runs/<model>/seed<k>/.

    python -m ner.train --model biobert --seed 1
    python -m ner.train --model bert --seed 1 --train_limit 200 --eval_limit 200 --epochs 1   # smoke test

Model selection uses the dev set (best dev entity F1 across epochs; strict IOB2 by default,
--select_metric default_f1 for seqeval default mode); the selected model is then evaluated once
on dev and test. The test set is also scored after every epoch (test_history in metrics.json),
for reporting a fixed last-epoch protocol only; it never influences selection.
Training runs in FP32 unless --precision bf16/fp16 is given.
"""

import argparse
import importlib.metadata
import json
import math
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
from transformers import (
    AutoModelForTokenClassification, DataCollatorForTokenClassification, EarlyStoppingCallback,
    Trainer, TrainerCallback, TrainingArguments, set_seed,
)

from .data import DEFAULT_DATA_DIR, ID2LABEL, LABEL2ID, LABELS, load_ncbi
from .models import MODELS, load_tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECT_METRICS = {"strict_f1": "strict_f1", "default_f1": "f1"}  # --select_metric -> compute_metrics key


# ----------------------------------------------------------------------------- data encoding
class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def encode(rows, tokenizer, max_length, split):
    """Tokenize words; label only the first sub-token of each word (others -100).

    Self-check: the kept labels must equal the original word labels for every sentence.
    """
    encoded = tokenizer(
        [tokens for tokens, _ in rows], truncation=True, max_length=max_length, is_split_into_words=True,
    )
    features, bad, truncated = [], [], 0
    for i, (tokens, tags) in enumerate(rows):
        word_ids = encoded.word_ids(batch_index=i)
        truncated += len(word_ids) >= max_length
        labels, previous = [], None
        for word_id in word_ids:
            if word_id is None or word_id == previous:
                labels.append(-100)
            else:
                labels.append(LABEL2ID[tags[word_id]])
            previous = word_id
        if [ID2LABEL[x] for x in labels if x != -100] != tags:
            bad.append(i)
        feature = {key: encoded[key][i] for key in encoded.keys()}
        feature["labels"] = labels
        features.append(feature)
    if bad:
        raise ValueError(
            f"Label alignment self-check failed on {split}: {len(bad)} sentences (first indices {bad[:5]}); "
            f"{truncated} sentences reach max_length={max_length} (truncation drops words; raise --max_length), "
            f"other failures mean a word produced no sub-tokens."
        )
    print(f"  alignment self-check OK on {split}: {len(rows)} sentences, max sub-tokens "
          f"{max(len(f['input_ids']) for f in features)}")
    return EncodedDataset(features)


# ----------------------------------------------------------------------------- metrics
def to_word_level(logits_or_ids, label_ids):
    """Collapse sub-token arrays to word-level label strings (gold, pred) using the -100 mask."""
    pred_ids = logits_or_ids if logits_or_ids.ndim == 2 else np.argmax(logits_or_ids, axis=-1)
    golds, preds = [], []
    for pred_row, gold_row in zip(pred_ids, label_ids):
        kept = gold_row != -100
        golds.append([LABELS[x] for x in gold_row[kept]])
        preds.append([LABELS[x] for x in pred_row[kept]])
    return golds, preds


def seqeval_scores(golds, preds):
    out = {"token_accuracy": accuracy_score(golds, preds)}
    for mode, kwargs in (("default", {}), ("strict", {"mode": "strict", "scheme": IOB2})):
        out[mode] = {
            "precision": precision_score(golds, preds, zero_division=0, **kwargs),
            "recall": recall_score(golds, preds, zero_division=0, **kwargs),
            "f1": f1_score(golds, preds, zero_division=0, **kwargs),
        }
    return out


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    s = seqeval_scores(*to_word_level(logits, labels))
    return {
        "precision": s["default"]["precision"], "recall": s["default"]["recall"], "f1": s["default"]["f1"],
        "strict_precision": s["strict"]["precision"], "strict_recall": s["strict"]["recall"],
        "strict_f1": s["strict"]["f1"], "accuracy": s["token_accuracy"],
    }


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)  # keep only predicted ids to save memory during eval


# ----------------------------------------------------------------------------- run
def device_info():
    if torch.cuda.is_available():
        return {"device": "cuda", "name": torch.cuda.get_device_name(0)}
    if torch.backends.mps.is_available():
        return {"device": "mps", "name": platform.machine()}
    return {"device": "cpu", "name": platform.processor() or platform.machine()}


def check_precision(precision, dev_info):
    """Fail fast if the requested mixed precision cannot run natively on this device."""
    if precision == "fp32":
        return
    device = dev_info["device"]
    if device == "cpu":
        raise SystemExit(f"--precision {precision} is not supported on CPU; use --precision fp32")
    if precision == "bf16":
        if device == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise SystemExit(f"--precision bf16 requested but {dev_info['name']} has no native bf16 "
                             "(e.g. Tesla T4 / pre-Ampere); use --precision fp32 or fp16")
        if device == "mps":
            reason = "matmul did not run in bf16"
            try:
                with torch.autocast("mps", dtype=torch.bfloat16):
                    x = torch.ones(2, 2, device="mps")
                    ok = (x @ x).dtype == torch.bfloat16
            except Exception as e:  # older macOS / torch without MPS bf16
                ok, reason = False, repr(e)
            if not ok:
                raise SystemExit(f"--precision bf16 requested but MPS bf16 autocast is unavailable "
                                 f"({reason}); use --precision fp32")


class RNGState:
    """Snapshot / restore python, numpy and torch (CPU + CUDA + MPS) RNG states."""

    def __init__(self):
        self.py, self.np, self.cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None

    def restore(self):
        random.setstate(self.py)
        np.random.set_state(self.np)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)
        if self.mps is not None:
            torch.mps.set_rng_state(self.mps)


class TestHistoryCallback(TrainerCallback):
    """After each dev evaluation, also score the test set with the current weights.

    Record-only: it uses Trainer.predict (fires on_predict, not on_evaluate, so no recursion and no
    entry in log_history), never touches selection, and restores all RNG states afterwards so the
    training trajectory is identical with or without it.
    """

    def __init__(self, dataset, rows):
        self.dataset, self.rows, self.trainer, self.history = dataset, rows, None, []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.trainer is None or not self.trainer.is_in_train:
            return
        rng = RNGState()
        was_training = self.trainer.model.training
        pred = self.trainer.predict(self.dataset, metric_key_prefix="test")
        if was_training:
            self.trainer.model.train()
        rng.restore()
        golds, preds = to_word_level(pred.predictions, pred.label_ids)
        assert golds == [tags for _, tags in self.rows], "test: gold labels drifted"
        entry = {"epoch": state.epoch, "global_step": state.global_step} | seqeval_scores(golds, preds)
        entry["loss"] = pred.metrics.get("test_loss")
        self.history.append(entry)


def versions():
    import accelerate

    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "accelerate": accelerate.__version__,
        "seqeval": importlib.metadata.version("seqeval"), "numpy": np.__version__,
    }


def write_predictions(path, rows, preds):
    with open(path, "w", encoding="utf-8") as f:
        for i, ((tokens, gold), pred) in enumerate(zip(rows, preds)):
            f.write(json.dumps({"id": i, "tokens": tokens, "gold": gold, "pred": pred}, ensure_ascii=False) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=None, help="default: --batch_size")
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--early_stopping_patience", type=int, default=0, help="0 = no early stopping")
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--keep_checkpoints", action="store_true", help="keep checkpoint-*/ dirs after the run")
    p.add_argument("--save_model", action="store_true", help="save the selected (best-dev) model to <output_dir>/model")
    p.add_argument("--train_limit", type=int, default=None, help="use only the first N training sentences")
    p.add_argument("--eval_limit", type=int, default=None, help="use only the first N dev and test sentences")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32",
                   help="training precision; bf16 needs native support (Ampere+ CUDA or MPS), fp16 uses a grad scaler")
    p.add_argument("--select_metric", choices=sorted(SELECT_METRICS), default="strict_f1",
                   help="dev metric for picking the best epoch: strict IOB2 entity F1 or seqeval default-mode F1")
    p.add_argument("--no_test_history", action="store_true",
                   help="skip the per-epoch test evaluation (test_history); for checking it does not affect training")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output_dir", default=None, help="default: runs/<model>/seed<seed>")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    spec = MODELS[args.model]
    out_dir = Path(args.output_dir or REPO_ROOT / "runs" / args.model / f"seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_info = device_info()
    check_precision(args.precision, dev_info)
    print(f"== {args.model} ({spec['checkpoint']}@{spec['revision'][:8]}) seed={args.seed} "
          f"device={dev_info} precision={args.precision} select={args.select_metric} -> {out_dir}")
    started = time.time()
    set_seed(args.seed)

    splits, audit = load_ncbi(args.data_dir)
    rows = {
        "train": splits["train"][: args.train_limit],
        "validation": splits["validation"][: args.eval_limit],
        "test": splits["test"][: args.eval_limit],
    }
    tokenizer = load_tokenizer(args.model)
    encoded = {split: encode(r, tokenizer, args.max_length, split) for split, r in rows.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        spec["checkpoint"], revision=spec["revision"], num_labels=len(LABELS),
        label2id=LABEL2ID, id2label=ID2LABEL,
    )

    # warmup_ratio was removed in transformers 5; compute integer warmup steps ourselves.
    steps_per_epoch = math.ceil(len(encoded["train"]) / args.batch_size)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = math.ceil(args.warmup_ratio * total_steps)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model=SELECT_METRICS[args.select_metric],
        greater_is_better=True,
        logging_steps=args.logging_steps,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
    )
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    test_cb = None if args.no_test_history else TestHistoryCallback(encoded["test"], rows["test"])
    if test_cb:
        callbacks.append(test_cb)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )
    if test_cb:
        test_cb.trainer = trainer
    train_started = time.time()
    train_output = trainer.train()
    train_seconds = time.time() - train_started

    sel_key = SELECT_METRICS[args.select_metric]
    dev_history = [
        {"epoch": h["epoch"], "f1": h["eval_f1"], "precision": h["eval_precision"], "recall": h["eval_recall"],
         "strict_f1": h["eval_strict_f1"], "strict_precision": h["eval_strict_precision"],
         "strict_recall": h["eval_strict_recall"], "accuracy": h["eval_accuracy"], "loss": h.get("eval_loss")}
        for h in trainer.state.log_history if "eval_f1" in h
    ]
    best = max(dev_history, key=lambda h: h[sel_key])  # first epoch wins ties, like Trainer

    # transformers 5.x saves BERT LayerNorm weights under legacy gamma/beta names, and
    # load_best_model_at_end restores the checkpoint with a plain load_state_dict that skips
    # them (LayerNorms would silently stay at the last epoch). Load the best checkpoint through
    # from_pretrained (which applies the key renaming) and copy its weights into the trainer's own
    # model object, so the final evaluation keeps the same accelerate wrapping / autocast as the
    # per-epoch dev evaluations. Verified below against the best dev scores seen during training.
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt:
        target = trainer.accelerator.unwrap_model(trainer.model)
        best_model = AutoModelForTokenClassification.from_pretrained(best_ckpt)
        source = best_model.state_dict()
        key_diff = set(target.state_dict()) ^ set(source)
        if key_diff:
            raise RuntimeError(f"best checkpoint keys differ from the trained model: {sorted(key_diff)[:10]}")
        target.load_state_dict(source, strict=True)
        del best_model, source
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final evaluation of the selected model on dev and test (same logits for metrics + predictions).
    results = {}
    for split, name in (("validation", "dev"), ("test", "test")):
        pred = trainer.predict(encoded[split], metric_key_prefix=name)
        golds, preds = to_word_level(pred.predictions, pred.label_ids)
        assert golds == [tags for _, tags in rows[split]], f"{split}: gold labels drifted"
        results[name] = seqeval_scores(golds, preds)
        results[name]["loss"] = pred.metrics.get(f"{name}_loss")
        write_predictions(out_dir / f"{name}_predictions.jsonl", rows[split], preds)

    # Fatal check: the reloaded model must reproduce the best dev scores seen during training.
    reload_gap = max(abs(results["dev"]["default"]["f1"] - best["f1"]),
                     abs(results["dev"]["strict"]["f1"] - best["strict_f1"]))
    if reload_gap > 1e-6:
        raise RuntimeError(
            f"dev F1 after reloading best checkpoint (default {results['dev']['default']['f1']:.6f}, "
            f"strict {results['dev']['strict']['f1']:.6f}) differs from the best epoch during training "
            f"(default {best['f1']:.6f}, strict {best['strict_f1']:.6f}); gap {reload_gap:.2e}")

    if args.save_model:
        trainer.save_model(str(out_dir / "model"))
    if not args.keep_checkpoints:
        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)

    metrics = {
        "model": args.model,
        "checkpoint": spec["checkpoint"],
        "revision": spec["revision"],
        "seed": args.seed,
        "precision": args.precision,
        "config": vars(args) | {"warmup_steps": warmup_steps, "steps_per_epoch": steps_per_epoch,
                                "total_steps": total_steps},
        "data": {
            "source": "spyysalo/ncbi-disease CoNLL (empty records dropped)",
            "sha256": {s: v["sha256"] for s, v in audit["splits"].items()},
            "sizes_used": {s: len(r) for s, r in rows.items()},
            "limited": bool(args.train_limit or args.eval_limit),
        },
        "device": dev_info,
        "training_args": json.loads(training_args.to_json_string()),
        "versions": versions(),
        "selection": {"metric": args.select_metric,
                      "metric_description": {"strict_f1": "dev entity F1, seqeval strict IOB2",
                                             "default_f1": "dev entity F1, seqeval default mode"}[args.select_metric],
                      "best_epoch": best["epoch"],
                      "best_dev_score_during_training": best[sel_key],
                      "best_dev_f1_during_training": best["f1"],
                      "best_dev_strict_f1_during_training": best["strict_f1"],
                      "best_checkpoint_reload_f1_gap": reload_gap,
                      "epochs_completed": trainer.state.epoch,
                      "early_stopping_patience": args.early_stopping_patience},
        "dev_history": dev_history,
        "test_history": test_cb.history if test_cb else None,
        "dev": results["dev"],
        "test": results["test"],
        "train_loss": train_output.training_loss,
        "train_loss_history": [{"step": h["step"], "epoch": h["epoch"], "loss": h["loss"]}
                               for h in trainer.state.log_history if "loss" in h],
        "seconds": {"train": round(train_seconds, 1), "total": round(time.time() - started, 1)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    t = results["test"]
    print(f"== done {args.model} seed={args.seed}: best epoch {best['epoch']} by {args.select_metric}, "
          f"dev F1 {results['dev']['default']['f1']:.4f}, test F1 {t['default']['f1']:.4f} "
          f"(strict {t['strict']['f1']:.4f}), {metrics['seconds']['total']}s -> {out_dir / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    main()
