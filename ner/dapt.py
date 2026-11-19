"""Domain-adaptive pre-training: continue masked-language-model training on the unlabelled corpus.

    python -m ner.dapt --model bert --max_steps 20000          # -> runs_dapt/bert_dapt/{model/,dapt.json}
    python -m ner.dapt --model bert --max_steps 0 --output_dir runs_dapt/bert_dapt0   # zero-step control
    python -m ner.dapt --model bert --limit_docs 600 --max_steps 30 --output_dir runs_dapt/_smoke/s30

The corpus comes from ``python -m ner.dapt_corpus build`` (data/dapt/corpus_{train,valid}.txt).
Each document is tokenized with the model's checked tokenizer (ner.models.load_tokenizer) and cut
into non-overlapping windows of max_length word pieces ([CLS] + max_length-2 pieces + [SEP]).
The validation windows are masked once with a fixed seed, so validation losses are comparable
across evaluations. ``--max_steps 0`` saves the original weights unchanged (control).

After saving, the model is reloaded with AutoModelForMaskedLM.from_pretrained and every tensor
must equal the in-memory model (fatal check; transformers 5.x writes BERT LayerNorm weights under
legacy gamma/beta names). dapt.json records the encoder tensor hash that ner.train --init_from
re-checks.
"""

import argparse
import json
import math
import time
from hashlib import sha256
from pathlib import Path

import torch
from transformers import (
    AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed,
)

from .models import MODELS, load_tokenizer
from .train import check_precision, device_info, versions

REPO_ROOT = Path(__file__).resolve().parent.parent


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def encoder_state(model):
    """CPU fp32 copies of the encoder tensors (``bert.*``), keyed by name."""
    return {k: v.detach().to("cpu") for k, v in model.state_dict().items() if k.startswith("bert.")}


def state_hash(state):
    h = sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].contiguous().numpy().tobytes())
    return h.hexdigest()


def compare_states(a, b):
    """-> (keys only in a, keys only in b, names of differing tensors, max abs difference)."""
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    diff, max_abs = [], 0.0
    for k in sorted(set(a) & set(b)):
        if a[k].shape != b[k].shape or not torch.equal(a[k], b[k]):
            diff.append(k)
            if a[k].shape == b[k].shape:
                max_abs = max(max_abs, (a[k].double() - b[k].double()).abs().max().item())
    return only_a, only_b, diff, max_abs


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


def make_windows(texts, tokenizer, max_length):
    """Non-overlapping [CLS] + pieces + [SEP] windows of at most max_length tokens."""
    size = max_length - 2
    features = []
    for start in range(0, len(texts), 1000):
        ids = tokenizer(texts[start:start + 1000], add_special_tokens=False, truncation=False,
                        verbose=False)["input_ids"]
        for doc in ids:
            for i in range(0, len(doc), size):
                piece = [tokenizer.cls_token_id] + doc[i:i + size] + [tokenizer.sep_token_id]
                features.append({"input_ids": piece,
                                 "special_tokens_mask": [1] + [0] * (len(piece) - 2) + [1]})
    return features


class MLMCollator:
    """Dynamic masking for training windows; windows that already carry labels are only padded."""

    def __init__(self, tokenizer, mlm_probability):
        self.mlm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        if "labels" not in features[0]:
            return self.mlm(features)
        length = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = length - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in batch.items()}


def premask(features, tokenizer, mlm_probability, seed):
    """Mask validation windows once (fixed seed) so every evaluation sees the same masks."""
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability, seed=seed)
    out = []
    for f in features:
        b = collator([f])
        out.append({"input_ids": b["input_ids"][0].tolist(), "labels": b["labels"][0].tolist()})
    return out


def read_docs(path, limit):
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[:limit] if limit else lines


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bert", choices=sorted(MODELS))
    p.add_argument("--corpus_dir", default=str(REPO_ROOT / "data" / "dapt"))
    p.add_argument("--output_dir", default=None, help="default: runs_dapt/<model>_dapt")
    p.add_argument("--max_steps", type=int, default=None, help="optimizer steps (0 = save original weights); "
                   "default: --epochs over the corpus")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--eval_batch_size", type=int, default=None, help="default: --batch_size")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--limit_docs", type=int, default=None, help="use only the first N documents of each corpus file")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    spec = MODELS[args.model]
    out_dir = Path(args.output_dir or REPO_ROOT / "runs_dapt" / f"{args.model}_dapt")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_info = device_info()
    check_precision(args.precision, dev_info)
    corpus = {name: Path(args.corpus_dir) / f"corpus_{name}.txt" for name in ("train", "valid")}
    print(f"== DAPT {args.model} ({spec['checkpoint']}@{spec['revision'][:8]}) device={dev_info} "
          f"precision={args.precision} -> {out_dir}")
    started = time.time()
    set_seed(args.seed)

    tokenizer = load_tokenizer(args.model)
    docs = {name: read_docs(path, args.limit_docs) for name, path in corpus.items()}
    train_windows = make_windows(docs["train"], tokenizer, args.max_length)
    valid_windows = premask(make_windows(docs["valid"], tokenizer, args.max_length), tokenizer,
                            args.mlm_probability, args.seed)
    n_pieces = sum(len(f["input_ids"]) for f in train_windows)
    print(f"  {len(docs['train'])} train docs -> {len(train_windows)} windows ({n_pieces} tokens), "
          f"{len(docs['valid'])} valid docs -> {len(valid_windows)} windows")

    model = AutoModelForMaskedLM.from_pretrained(spec["checkpoint"], revision=spec["revision"])

    steps_per_epoch = math.ceil(len(train_windows) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps is not None else math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = math.ceil(args.warmup_ratio * total_steps)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=max(total_steps, 1),  # max_steps 0 never calls train()
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        logging_steps=args.logging_steps,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=WindowDataset(train_windows),
        eval_dataset=WindowDataset(valid_windows), data_collator=MLMCollator(tokenizer, args.mlm_probability),
    )

    initial = trainer.evaluate()["eval_loss"]
    print(f"  initial valid loss {initial:.4f} (ppl {math.exp(initial):.2f})")
    train_seconds, train_loss = 0.0, None
    if total_steps > 0:
        t0 = time.time()
        train_output = trainer.train()
        train_seconds = time.time() - t0
        train_loss = train_output.training_loss
    final = trainer.evaluate()["eval_loss"] if total_steps > 0 else initial
    steps_done = trainer.state.global_step if total_steps > 0 else 0
    print(f"  final valid loss {final:.4f} (ppl {math.exp(final):.2f}) after {steps_done} steps")

    # Save and verify that a fresh from_pretrained reproduces every tensor of the in-memory model.
    model_dir = out_dir / "model"
    trained = trainer.accelerator.unwrap_model(trainer.model)
    trained.save_pretrained(str(model_dir))
    memory = {k: v.detach().to("cpu") for k, v in trained.state_dict().items()}
    reloaded, info = AutoModelForMaskedLM.from_pretrained(str(model_dir), output_loading_info=True)
    loading_problems = {k: v for k, v in info.items() if v}
    only_mem, only_disk, diff, max_abs = compare_states(memory, reloaded.state_dict())
    if loading_problems or only_mem or only_disk or diff:
        raise RuntimeError(f"saved model does not reload identically: loading_info={loading_problems}, "
                           f"keys only in memory {only_mem[:5]}, only on disk {only_disk[:5]}, "
                           f"differing tensors {diff[:5]} (max abs diff {max_abs:.3g})")
    encoder = encoder_state(reloaded)
    print(f"  save/reload self-check OK: {len(memory)} tensors identical")

    history = trainer.state.log_history
    report = {
        "base_model": args.model, "checkpoint": spec["checkpoint"], "revision": spec["revision"],
        "corpus": {"dir": str(args.corpus_dir),
                   "sha256": {f"corpus_{n}.txt": file_sha256(p) for n, p in corpus.items()},
                   "limit_docs": args.limit_docs,
                   "documents": {n: len(d) for n, d in docs.items()},
                   "windows": {"train": len(train_windows), "valid": len(valid_windows)},
                   "train_tokens": n_pieces},
        "config": vars(args) | {"warmup_steps": warmup_steps, "steps_per_epoch": steps_per_epoch,
                                "total_steps": total_steps},
        "training_args": json.loads(training_args.to_json_string()) if total_steps > 0 else None,
        "steps": steps_done,
        "train_loss": train_loss,
        "train_loss_history": [{"step": h["step"], "loss": h["loss"], "learning_rate": h.get("learning_rate")}
                               for h in history if "loss" in h],
        "valid_loss_history": [{"step": s, "loss": l} for s, l in sorted(
            ({0: initial} | {h["step"]: h["eval_loss"] for h in history if "eval_loss" in h and h["step"] > 0}).items())],
        "valid_loss": {"initial": initial, "final": final},
        "valid_perplexity": {"initial": math.exp(initial), "final": math.exp(final)},
        "encoder_sha256": state_hash(encoder),
        "encoder_tensors": len(encoder),
        "save_reload_check": {"tensors": len(memory), "identical": True},
        "device": dev_info,
        "versions": versions(),
        "seconds": {"train": round(train_seconds, 1), "total": round(time.time() - started, 1),
                    "per_step": round(train_seconds / steps_done, 3) if steps_done else None},
    }
    (out_dir / "dapt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"== done: {steps_done} steps, valid loss {initial:.4f} -> {final:.4f}, "
          f"{report['seconds']['per_step']} s/step, {report['seconds']['total']}s -> {model_dir}")
    return report


if __name__ == "__main__":
    main()
