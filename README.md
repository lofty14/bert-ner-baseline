# BERT NER Baseline (NCBI Disease)

Fine-tunes `bert-base-cased` for disease named-entity recognition on the [NCBI Disease](https://huggingface.co/datasets/ncbi/ncbi_disease) dataset (labels: `O`, `B-Disease`, `I-Disease`).

## Setup

```bash
pip install torch datasets==3.6.0 transformers evaluate seqeval
```

Then run `baseline.ipynb`.

## Training config

- lr 2e-5, batch size 8, 3 epochs, weight decay 0.01

## Test results

| Precision | Recall | F1 | Accuracy |
|---|---|---|---|
| 0.826 | 0.881 | 0.853 | 0.984 |
