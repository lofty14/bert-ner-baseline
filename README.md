# Disease NER on NCBI Disease (BERT / BioBERT / PubMedBERT)

COMP9444 group project: token-classification models for disease mention recognition on the
[NCBI Disease](https://huggingface.co/datasets/ncbi/ncbi_disease) corpus (labels `O`, `B-Disease`, `I-Disease`).

```
audit_ncbi_splits.py        data audit (CoNLL-only, or CoNLL vs. Hugging Face)
ner/                        unified experiment framework
  data.py                   download + SHA-256 check + parse CoNLL, audit summary
  models.py                 model registry (pinned revisions, tokenizer casing check)
  train.py                  one model x one seed  -> runs/<model>/seed<k>/
  sweep.py                  many models x seeds, then aggregate
  aggregate.py              mean ± std markdown tables from runs/**/metrics.json
baseline.ipynb              original BERT notebook (kept as is)
biobert.ipynb               original BioBERT v1.1 notebook (kept as is)
pubmedbert_baseline.ipynb   original PubMedBERT/BioBERT v1.2 notebook (kept as is)
```

## Data and audit

The single data source is the CoNLL release used by the Hugging Face loader,
[spyysalo/ncbi-disease `conll/`](https://github.com/spyysalo/ncbi-disease/tree/master/conll).
`ner` downloads it once to `data/ncbi_disease/` and refuses to run if a file's SHA-256 differs:

| split | file | sentences | tokens | B-Disease | I-Disease | orphan I | SHA-256 (prefix) |
|---|---|---:|---:|---:|---:|---:|---|
| train | train.tsv | 5432 | 136086 | 5145 | 6122 | 0 | `5f87b0de649e` |
| dev | devel.tsv | 923 | 23969 | 787 | 1090 | 0 | `b4e1e7efcc2a` |
| test | test.tsv | 940 | 24497 | 960 | 1087 | 0 | `8f088ce45dc4` |

Duplicate sentences (identical token sequences): 32 within train, 3 within test;
across splits train∩dev = 5, train∩test = 1, dev∩test = 1 (left as is, matching the official splits).

The Hugging Face `ncbi/ncbi_disease` loader yields one extra **empty** record at the end of each split
(5433/924/941). The unified framework drops empty records; the audit script ignores them when comparing.

```bash
python audit_ncbi_splits.py --no-hf --conll_dir data/ncbi_disease   # CoNLL only, no `datasets` needed
python audit_ncbi_splits.py                                          # also compare with HF (needs `datasets`)
```

The same CoNLL audit summary is printed at the start of every training run.

## Unified framework

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --use-pep517 -r requirements.txt     # seqeval 1.2.2 needs --use-pep517 with recent setuptools
```

`requirements.txt` pins the versions the framework was smoke-tested with (macOS arm64, Python 3.12,
Apple MPS, torch 2.13.0, transformers 5.22.0). The code only uses APIs that also exist in
transformers 4.57.x (`eval_strategy`, `processing_class`, integer `warmup_steps`), so it is expected to
run on Colab's preinstalled stack too, but only the pinned versions were actually tested.
(The tested environment was a `--system-site-packages` venv on top of an existing conda env that already
had torch/numpy/scikit-learn; only transformers, accelerate and seqeval were pip-installed. A completely
fresh `pip install -r requirements.txt` has not been exercised.)

### Models

| key | checkpoint | revision | tokenizer |
|---|---|---|---|
| `bert` | `google-bert/bert-base-cased` (= `bert-base-cased`) | `cd5ef92a` | cased |
| `biobert` | `dmis-lab/biobert-base-cased-v1.2` | `67c9c25b` | cased (forced: the repo has no tokenizer config, so the default would lowercase) |
| `pubmedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | `d673b883` | uncased |

Tokenizer casing is checked at load time and the run aborts if it is not as listed.

### Single run

```bash
python -m ner.train --model biobert --seed 1
# all options: --lr 2e-5 --epochs 3 --batch_size 8 --eval_batch_size --warmup_ratio 0 --weight_decay 0.01
#              --max_length 512 --early_stopping_patience 0 --save_total_limit 2 --keep_checkpoints
#              --save_model --train_limit N --eval_limit N --output_dir runs/<model>/seed<k>
#              --precision {fp32,bf16,fp16} (default fp32) --select_metric {strict_f1,default_f1} (default strict_f1)
#              --no_test_history
python -m ner.train --model bert --seed 1 --train_limit 200 --eval_limit 200 --epochs 1   # smoke test
```

What a run does:

- labels go on the **first sub-token** of each word (other sub-tokens / special tokens = `-100`);
  a self-check asserts that the kept labels reproduce the original word labels for every sentence;
- defaults match the old notebooks: lr 2e-5, batch 8, 3 epochs, no warmup, weight decay 0.01, linear decay, AdamW;
- `set_seed(seed)` plus `seed = data_seed = seed`;
- trains in **FP32** by default. `--precision bf16` needs native bf16 (Ampere or newer CUDA GPUs, or Apple
  MPS) and aborts otherwise, e.g. on a Tesla T4; `--precision fp16` uses AMP with a grad scaler. Both
  were only smoke-tested on Apple MPS, not on CUDA. Mixed precision is never switched on automatically;
- evaluates on dev after every epoch and **selects the epoch with the best dev entity F1**, by default the
  **strict IOB2** F1 that is also the headline metric (`--select_metric strict_f1`, i.e.
  `metric_for_best_model="strict_f1"`); `--select_metric default_f1` selects on seqeval default-mode F1
  (the v1 behaviour); optional early stopping;
- after every dev evaluation it also scores the **test** set with the current weights and stores it as
  `test_history` (default/strict P/R/F1 and token accuracy per epoch). This is record-only: it never takes part
  in selection, and RNG states are saved/restored around it so training is unchanged (`--no_test_history`
  turns it off). It allows reporting a "fixed last epoch" protocol next to the dev-selected one;
- evaluates the selected model on dev and test with seqeval entity-level P/R/F1 in **default** mode and
  **strict** mode (`mode="strict", scheme=IOB2`), plus word-level token accuracy;
- writes `runs/<model>/seed<k>/metrics.json` (full config, actual `precision`, the full `training_args`,
  model revision, data SHA-256, device, library versions, dev history, `test_history`, training-loss history,
  best epoch, dev/test metrics, timings) and `dev_predictions.jsonl` /
  `test_predictions.jsonl` (`{"id", "tokens", "gold", "pred"}` per sentence, word level) for error analysis;
- deletes checkpoints afterwards unless `--keep_checkpoints`; `--save_model` keeps the selected model.

Note: with transformers 5.x, `Trainer`'s own best-model reload skips BERT LayerNorm weights (they are
saved under legacy `gamma`/`beta` names; you will see an "unexpected keys" warning). `ner.train`
therefore loads the best checkpoint through `from_pretrained` and copies its weights (`load_state_dict`,
strict, identical key sets) into the trainer's own model object, so the final evaluation uses the same
accelerate wrapping / autocast precision as the per-epoch dev evaluations. The run **aborts** if the
reloaded model does not reproduce the best epoch's dev default and strict F1 within 1e-6
(`selection.best_checkpoint_reload_f1_gap` in `metrics.json`, expected 0).

### Multiple seeds and aggregation

```bash
python -m ner.sweep --models bert biobert pubmedbert --seeds 1 2 3                # runs sequentially, then aggregates
python -m ner.sweep --models biobert --seeds 1 2 3 -- --warmup_ratio 0.1 --epochs 5 \
       --early_stopping_patience 2                                                # args after -- go to ner.train
python -m ner.aggregate runs/ --out results.md                                    # mean ± sample std tables
```

`ner.sweep` skips runs whose `metrics.json` already exists (`--overwrite` to redo). Use a different
`--runs_dir` per hyper-parameter setting. `ner.aggregate` groups runs by protocol (precision + selection
metric; it prints a warning if one directory mixes protocols) and then by model (runs with
`--train_limit/--eval_limit` are shown separately as "limited/smoke"). For each protocol it reports dev and
test at the **dev-selected** epoch and test at the **fixed last epoch** (from `test_history`; older runs
without it are skipped and listed), each with strict and default scores.

### Colab

```python
!git clone <repo-url> group_project
%cd group_project
!pip install -q --use-pep517 "transformers==5.22.0" "accelerate==1.15.0" "seqeval==1.2.2"   # keep Colab's torch
!python -m ner.sweep --models bert biobert pubmedbert --seeds 1 2 3
# copy runs/ to Drive before the session ends, e.g.:
# from google.colab import drive; drive.mount('/content/drive'); !cp -r runs /content/drive/MyDrive/ncbi_runs
```

Training is FP32 unless `--precision` says otherwise (a Colab T4 has no native bf16, so `--precision bf16`
aborts there). If Colab's torch is too old for
transformers 5.22, either upgrade torch or try transformers 4.57.x (untested, see above).

## Results

### Previous single-seed notebook results (not comparable to the unified protocol)

These come from the original notebooks: one seed (42), HF dataset with trailing empty records,
test evaluated with the **last** epoch's weights (no dev-based selection), seqeval default mode.

| model | notebook | test P | test R | test F1 | token acc |
|---|---|---:|---:|---:|---:|
| BERT-base-cased | `baseline.ipynb` | 0.826 | 0.881 | 0.8528 | 0.984 |
| BioBERT v1.1 | `biobert.ipynb` | – | – | 0.8681 | – |

### Unified protocol v1 (dev-selected by default-mode F1, seeds 1/2/3)

Run on Colab (Tesla T4, transformers 5.22.0) with `colab_sweep.ipynb`, old-notebook hyper-parameters
(lr 2e-5, batch 8, 3 epochs, no warmup); for each seed the epoch with the best dev F1 in seqeval **default**
mode was evaluated on test. These are **v1 protocol** results produced before `--precision` existed, and
their precision was not FP32 as previously stated: v1 enabled bf16 whenever
`torch.cuda.is_bf16_supported()` returned true, which it does on a T4 through emulation, so training (and
the per-epoch dev evaluations used for selection) very likely ran in emulated bf16 autocast, while the final
dev/test evaluation used a freshly loaded, unwrapped model and therefore ran in FP32 (hence the small
non-zero `best_checkpoint_reload_f1_gap` of 0.001-0.002 in those runs). `metrics.json` of v1 did not store
the training arguments, so this cannot be confirmed from the artefacts. The current defaults (v2: FP32,
strict-F1 selection, consistent final evaluation) will replace these numbers once re-run.

**Summary (test, mean ± std over 3 seeds, %)**

| model | P (strict) | R (strict) | F1 (strict) | F1 (default) |
|---|---:|---:|---:|---:|
| bert | 83.40 ± 0.57 | 87.01 ± 0.84 | 85.17 ± 0.42 | 84.21 ± 0.42 |
| biobert (v1.2) | 86.16 ± 0.95 | 87.47 ± 2.69 | 86.79 ± 1.32 | 85.95 ± 1.35 |
| pubmedbert | 86.55 ± 0.49 | 89.10 ± 0.52 | **87.80 ± 0.25** | 87.28 ± 0.50 |

Full `ner.aggregate` output:

Scores are percentages, mean ± sample std over seeds. `default` = seqeval default mode, `strict` = mode='strict', scheme=IOB2.

#### dev

| model | n seeds | P (default) | R (default) | F1 (default) | P (strict) | R (strict) | F1 (strict) | token acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bert | 3 (1,2,3) | 81.66 ± 1.29 | 87.59 ± 0.65 | 84.51 ± 0.77 | 84.08 ± 0.68 | 87.42 ± 0.58 | 85.71 ± 0.23 | 98.63 ± 0.01 |
| biobert | 3 (1,2,3) | 83.29 ± 0.91 | 88.35 ± 0.90 | 85.74 ± 0.12 | 86.03 ± 1.47 | 87.63 ± 1.41 | 86.81 ± 0.12 | 98.74 ± 0.05 |
| pubmedbert | 3 (1,2,3) | 86.05 ± 1.51 | 88.69 ± 0.22 | 87.35 ± 0.87 | 87.36 ± 1.09 | 88.61 ± 0.26 | 87.98 ± 0.66 | 98.78 ± 0.05 |

#### test

| model | n seeds | P (default) | R (default) | F1 (default) | P (strict) | R (strict) | F1 (strict) | token acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bert | 3 (1,2,3) | 81.08 ± 0.79 | 87.60 ± 0.89 | 84.21 ± 0.42 | 83.40 ± 0.57 | 87.01 ± 0.84 | 85.17 ± 0.42 | 98.25 ± 0.03 |
| biobert | 3 (1,2,3) | 83.43 ± 0.88 | 88.65 ± 2.11 | 85.95 ± 1.35 | 86.16 ± 0.95 | 87.47 ± 2.69 | 86.79 ± 1.32 | 98.41 ± 0.18 |
| pubmedbert | 3 (1,2,3) | 85.19 ± 0.35 | 89.48 ± 0.79 | 87.28 ± 0.50 | 86.55 ± 0.49 | 89.10 ± 0.52 | 87.80 ± 0.25 | 98.56 ± 0.05 |

#### model selection

| model | best dev epoch per seed |
|---|---|
| bert | s1:3, s2:2, s3:2 |
| biobert | s1:2, s2:1, s3:2 |
| pubmedbert | s1:3, s2:3, s3:2 |
