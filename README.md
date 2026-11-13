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
python -m ner.train --model bert --seed 1 --train_limit 200 --eval_limit 200 --epochs 1   # smoke test
```

What a run does:

- labels go on the **first sub-token** of each word (other sub-tokens / special tokens = `-100`);
  a self-check asserts that the kept labels reproduce the original word labels for every sentence;
- defaults match the old notebooks: lr 2e-5, batch 8, 3 epochs, no warmup, weight decay 0.01, linear decay, AdamW;
- `set_seed(seed)` plus `seed = data_seed = seed`;
- evaluates on dev after every epoch and **selects the epoch with the best dev entity F1**
  (`load_best_model_at_end=True`, `metric_for_best_model="f1"`); optional early stopping;
- evaluates the selected model on dev and test with seqeval entity-level P/R/F1 in **default** mode and
  **strict** mode (`mode="strict", scheme=IOB2`), plus word-level token accuracy;
- writes `runs/<model>/seed<k>/metrics.json` (full config, model revision, data SHA-256, device, library
  versions, dev history, best epoch, dev/test metrics, timings) and `dev_predictions.jsonl` /
  `test_predictions.jsonl` (`{"id", "tokens", "gold", "pred"}` per sentence, word level) for error analysis;
- deletes checkpoints afterwards unless `--keep_checkpoints`; `--save_model` keeps the selected model.

Note: with transformers 5.x, `Trainer`'s own best-model reload skips BERT LayerNorm weights (they are
saved under legacy `gamma`/`beta` names; you will see an "unexpected keys" warning). `ner.train`
therefore reloads the best checkpoint through `from_pretrained` and checks that it reproduces the best
dev F1 (`selection.best_checkpoint_reload_f1_gap` in `metrics.json`, expected 0).

### Multiple seeds and aggregation

```bash
python -m ner.sweep --models bert biobert pubmedbert --seeds 1 2 3                # runs sequentially, then aggregates
python -m ner.sweep --models biobert --seeds 1 2 3 -- --warmup_ratio 0.1 --epochs 5 \
       --early_stopping_patience 2                                                # args after -- go to ner.train
python -m ner.aggregate runs/ --out results.md                                    # mean ± sample std tables
```

`ner.sweep` skips runs whose `metrics.json` already exists (`--overwrite` to redo). Use a different
`--runs_dir` per hyper-parameter setting; `ner.aggregate` groups by model only (runs with
`--train_limit/--eval_limit` are shown separately as "limited/smoke").

### Colab

```python
!git clone <repo-url> group_project
%cd group_project
!pip install -q --use-pep517 "transformers==5.22.0" "accelerate==1.15.0" "seqeval==1.2.2"   # keep Colab's torch
!python -m ner.sweep --models bert biobert pubmedbert --seeds 1 2 3
# copy runs/ to Drive before the session ends, e.g.:
# from google.colab import drive; drive.mount('/content/drive'); !cp -r runs /content/drive/MyDrive/ncbi_runs
```

On CUDA GPUs with bf16 support training uses bf16 automatically. If Colab's torch is too old for
transformers 5.22, either upgrade torch or try transformers 4.57.x (untested, see above).

## Results

### Previous single-seed notebook results (not comparable to the unified protocol)

These come from the original notebooks: one seed (42), HF dataset with trailing empty records,
test evaluated with the **last** epoch's weights (no dev-based selection), seqeval default mode.

| model | notebook | test P | test R | test F1 | token acc |
|---|---|---:|---:|---:|---:|
| BERT-base-cased | `baseline.ipynb` | 0.826 | 0.881 | 0.8528 | 0.984 |
| BioBERT v1.1 | `biobert.ipynb` | – | – | 0.8681 | – |

### Unified protocol (dev-selected, seeds 1/2/3) — to be filled

Paste the output of `python -m ner.aggregate runs/` here.

| model | dev F1 (default) | test P (default) | test R (default) | test F1 (default) | test F1 (strict) |
|---|---:|---:|---:|---:|---:|
| bert | TBD | TBD | TBD | TBD | TBD |
| biobert (v1.2) | TBD | TBD | TBD | TBD | TBD |
| pubmedbert | TBD | TBD | TBD | TBD | TBD |
