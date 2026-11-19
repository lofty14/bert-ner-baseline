# Slides 9–12 — Domain Adaptation

（给自己看的说明用中文；要贴进幻灯片和要念的部分是英文。数字出处：`runs/v2/results.md`、
`runs/v2/compare_dapt*.md`、`runs/v2/error_analysis_dapt*.md`、`data/dapt/audit.json`、`runs_dapt/*/dapt.json`。）

## 1. 幻灯片上的字（英文，直接贴；共 4 页，第 4 页可整页删掉）

### Slide 9 — Method(s)

**Title:** Method(s)

**Subtitle:** Domain adaptation: continue pre-training, then fine-tune

Diagram (two rows of boxes, same start and same end; the yellow box is the only difference):

```
A  Direct:  bert-base-cased ─────────────────────────────────► Fine-tune on NCBI Disease ► Disease NER
B  DAPT:    bert-base-cased ► Continue masked-LM on 29k PubMed ► Fine-tune on NCBI Disease ► Disease NER
                              abstracts (1 or 3 epochs)
```

- Domain-adaptive pre-training (DAPT), Gururangan et al., ACL 2020
- Start from general-domain BERT: BioBERT and PubMedBERT have already seen PubMed

| | |
|---|---|
| Changed | Encoder weights at the start of fine-tuning (nothing else) |
| Kept the same | Starting checkpoint and vocabulary · NER train/dev/test split · fine-tuning settings · seeds 1, 2, 3 · epoch chosen on dev · strict entity-level P/R/F1 |

（这一页不出现任何结果数字。）

### Slide 10 — Method(s)

**Title:** Method(s)

**Subtitle:** Domain adaptation: extra text and leak check

Left — table:

| | |
|---|---|
| Source | PubMed abstracts via NCBI E-utilities: diseases category (MeSH), English, with abstract |
| Sampling | 1990–2012, the same number per year, fixed random seed; PMID list is in the repository |
| Size | 29,257 abstracts, about 5.9M words (9.8M tokens) |
| Leak check 1 | All 792 NCBI Disease abstracts excluded by PMID: 2 found, both from the training set |
| Leak check 2 | 1,771 distinct dev/test sentences (all with ≥ 40 characters) searched in the corpus: 0 hits |

Small note under the table: 91 dev/test sentences are shorter than 40 characters and were not searched; the PMID check still covers them.

Right — figure `dapt/figs/mlm_valid_loss.png`, caption:
The language model did adapt. Masked-LM perplexity on held-out PubMed text: 6.86 (BERT) → 3.70 (DAPT 1 epoch) → 3.35 (DAPT 3 epochs)

### Slide 11 — Results

**Title:** Results

**Subtitle:** Domain adaptation: before vs after

Left — figure `dapt/figs/ner_f1_paired.png`. Right — table:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| BERT, direct fine-tuning | 84.26 ± 0.32 | 86.63 ± 0.87 | 85.43 ± 0.51 |
| BERT + DAPT 1 epoch | 83.58 ± 0.82 | 86.91 ± 0.64 | 85.21 ± 0.68 |
| BERT + DAPT 3 epochs | 83.22 ± 0.06 | 87.12 ± 1.30 | 85.12 ± 0.63 |

Under the table: Test set, strict entity-level, mean ± std over 3 seeds (%). Paired ΔF1 vs direct: −0.22 ± 0.49 (1 epoch), −0.31 ± 0.20 (3 epochs)

- No improvement: the differences are smaller than the spread between seeds (0.5–0.7) and than the change from re-running one configuration (about 1 point).
- Precision: suggestive of a small precision loss. Recall: slightly higher on test, not on dev.

**Footnote (small):** BERT baseline here is the mean of 3 seeds under our protocol (FP32, epoch selected on dev); it is not the single-seed number on the main results slide.

### Slide 12 — Discussion（可整页删除）

**Title:** Discussion

**Subtitle:** Domain adaptation: why it did not help

Left — **What changed in the predictions**

Test set, 3 seeds summed, direct → DAPT 3 epochs

| | | |
|---|---|---|
| True positives | 2495 → 2509 | +14 |
| False positives | 466 → 506 | +40 |
| False negatives | 385 → 371 | −14 |

- New false positive: the gene name APC tagged as a disease ("... amino acid changes in **APC** associated with both FAP and non-FAP colorectal cancer patients")
- Fixed boundary: "very-long-chain acyl-coenzyme A dehydrogenase deficiency" is now found as one full span

Right — **Likely reasons (hypotheses)**

- Scale: 9.8M tokens and at most 4,896 steps at batch 32. Gururangan et al. used 7.55B tokens and 12.5K steps at batch 2048; BioBERT used 4.5B words of PubMed abstracts.
- Vocabulary unchanged: DAPT keeps BERT's general-domain word pieces.
- The 5.4k labelled training sentences may already teach most of what a little unlabelled text adds.

Bottom: Limitations: one DAPT run (seed 42), 3 NER seeds, small corpus, run on Apple MPS only.

导师说过：轮数、学习率这类属于「实验设置」，不要放在方法页。所以上面没有写学习率和 batch；
被问到再说（见第 4 节）。

## 2. 讲稿（英文；每页一段，与 pptx 各页备注逐字一致）

按每分钟约 140 词估算：第 1 页（Slide 9）117 词，约 50 秒；第 2 页（Slide 10）72 词，约 31 秒；第 3 页（Slide 11）40 词，约 17 秒；第 4 页（Slide 12）105 词，约 45 秒。4 页合计 334 词，约 2 分 23 秒。

只讲 3 页时：删掉第 4 页（Slide 12），并在第 3 页备注末尾加一句：So for this task, a small amount of adaptation does not reproduce the benefit of a model pre-trained on all of PubMed.（22 词）。这样 3 页合计 251 词，约 1 分 48 秒。

### Slide 9

> Our fourth experiment asks a simple question: BioBERT and PubMedBERT are better than BERT because they were
> pre-trained on biomedical text. Can we get part of that benefit cheaply, by adapting plain BERT ourselves?
>
> The method is domain-adaptive pre-training, from Gururangan and colleagues, 2020. We take bert-base-cased and keep
> training it with the same masked-word objective it was pre-trained with, but on unlabelled PubMed abstracts. Then
> we fine-tune it for disease NER exactly as before.
>
> It is a controlled comparison with two arms. The only thing that differs is the encoder weights we start
> fine-tuning from. The starting checkpoint, the vocabulary, the data split, the fine-tuning settings, the three
> random seeds and the evaluation are all identical.

### Slide 10

> The extra text is about twenty-nine thousand PubMed abstracts on diseases. Before training we checked for leakage
> in two ways: we removed every abstract that is part of NCBI Disease by its PubMed ID, and we searched the corpus
> for the validation and test sentences themselves. Neither check found any validation or test text in the corpus.
>
> The adaptation itself worked: masked-word perplexity on held-out PubMed text dropped from 6.9 to 3.4.

### Slide 11

> But NER did not improve. Test F1 went from 85.4 to 85.2 and 85.1, which is inside the variation between seeds, and
> smaller than what we see when we simply re-run the same configuration. If anything, precision is slightly lower.

### Slide 12

> Why did it not help? Looking at the errors, the adapted model finds a few more true entities but adds more false
> positives; for example, it tags the gene name APC as a disease. We think the main reason is scale: our corpus is
> about ten million tokens, while the original paper used seven and a half billion, and BioBERT was trained on four
> and a half billion words. The vocabulary also stays general-domain. These are hypotheses, and we only ran one
> adaptation. So for this task, a small amount of adaptation does not reproduce the benefit of a model pre-trained on
> all of PubMed.

## 3. 结果（给结果页 / 讨论页的同学，或自己被问到时用）

Test set, strict entity-level, mean ± sample std over seeds 1/2/3 (%):

| Model | Precision | Recall | F1 | ΔF1 vs direct (paired) |
|---|---:|---:|---:|---:|
| BERT, direct fine-tuning | 84.26 ± 0.32 | 86.63 ± 0.87 | 85.43 ± 0.51 | — |
| BERT + DAPT 1 epoch | 83.58 ± 0.82 | 86.91 ± 0.64 | 85.21 ± 0.68 | −0.22 ± 0.49 (seeds: +0.10, −0.78, +0.02) |
| BERT + DAPT 3 epochs | 83.22 ± 0.06 | 87.12 ± 1.30 | 85.12 ± 0.63 | −0.31 ± 0.20 (seeds: −0.36, −0.09, −0.48) |

Dev set F1: 85.26 ± 0.56 → 84.26 ± 0.36 → 84.39 ± 0.38 (paired Δ −1.01 ± 0.83 and −0.87 ± 0.55).

What is measured:

- The language model adapted: validation MLM perplexity 6.86 → 3.70 → 3.35. Three epochs adapt more than one, and NER
  still does not follow.
- F1 differences on test are smaller than the spread between seeds (about 0.5–0.7) and smaller than the change
  from re-running one configuration (section 5).
- Precision tends to be lower: in all 12 paired comparisons at the dev-selected epoch (dev and test × 1 and 3 epochs
  × 3 seeds), by 0.14 to 2.73 points.
  - 3-epoch DAPT: also lower at matched fine-tuning epochs 2 and 3 on test in all 6 pairs (−0.74 to −1.46, mean about
    −1.0). Re-running one configuration changes test precision at a fixed epoch by only 0.1–0.3, so this is above
    that noise.
  - 1-epoch DAPT: mixed at matched epochs (13 of 18 pairs negative).
  - Wording to use: "suggestive of a small precision loss", not "DAPT reduces precision". One DAPT seed, three NER seeds.
- Recall is slightly higher on test (+0.28, +0.49) and flat or slightly lower on dev (−0.42, −0.21). So this is
  not a clean precision-for-recall trade.
- Error counts on test, three seeds summed, direct → DAPT 3 epochs: true positives 2495 → 2509 (+14),
  false positives 466 → 506 (+40; 25 of them spurious mentions with no overlapping gold entity, 15 boundary errors),
  false negatives 385 → 371 (−14).
- Run-to-run noise: see section 5 (zero-step control and repeat run); it is larger than the differences between arms.

Real examples (test set; DAPT 3 epochs in all 3 seeds vs direct in 0 seeds):

1. New false positive, a gene name tagged as a disease (sentence 920):
   "This study also provided evidence for the pathological nature of amino acid changes in **APC** associated with
   both FAP and non-FAP colorectal cancer patients."
2. Boundary fixed (sentence 623): "Reversal of severe hypertrophic cardiomyopathy and excellent neuropsychologic
   outcome in **very-long-chain acyl-coenzyme A dehydrogenase deficiency**." Direct fine-tuning cut this to
   "very -" in 2 of 3 seeds; the adapted model gets the full span.
3. Lost true positive (sentence 342): "In **SCA3**, gaze-evoked nystagmus was often present ..." Direct fine-tuning
   finds the abbreviation in all 3 seeds; the adapted model misses it in all 3.

Why it probably did not help (these are hypotheses, say so):

- Scale. We used 9.8M tokens and at most 4,896 steps at batch 32. Gururangan et al. used 7.55B tokens of biomedical
  papers and 12.5K steps at an effective batch of 2048. BioBERT was pre-trained on 4.5B words of PubMed abstracts
  (Lee et al., 2020). Our corpus is about 0.1% of that.
- Even at full scale the gain is modest. In our group's own v1 runs BioBERT is +1.6 F1 over BERT and PubMedBERT +2.6.
  In Gururangan et al. the biomedical gains were +2.3 (ChemProt) and +0.4 (RCT), both classification tasks, with RoBERTa.
- Vocabulary did not change. PubMedBERT's advantage partly comes from an in-domain vocabulary; DAPT keeps BERT's
  general-domain word pieces.
- The labelled set (5.4k training sentences) may already teach the encoder most of what a little unlabelled text adds.

Limitations to state: one DAPT seed (42); only run on Apple MPS (the Colab/CUDA path and mixed precision are
untested); small corpus; 3 NER seeds; 91 dev/test sentences shorter than 40 characters could not be fingerprinted
(the PMID check still covers them).

## 4. 可能被问到的问题

- **Why BERT and not BioBERT as the starting model?** BioBERT and PubMedBERT were already pre-trained on PubMed, so
  more PubMed text would mostly repeat what they have seen. Starting from general-domain BERT isolates the effect.
- **Experimental settings of DAPT?** Batch 32, sequences of 256 tokens, learning rate 5e-5 with 6% warm-up, 15%
  masking, 1 epoch = 1,632 steps and 3 epochs = 4,896 steps, FP32. Fine-tuning: learning rate 2e-5, batch 8, 3 epochs.
- **How do you know the adapted weights were really used?** The training script refuses to run unless every encoder
  tensor equals the saved DAPT checkpoint and differs from the original BERT (197 of 197 tensors differ).
- **Why is your BERT number (85.43) different from the results slide (86.77)?** Different protocol: ours is the mean
  of three seeds with the epoch chosen on the validation set; the other is a single run with seed 42.
- **Is a 0.3 drop real?** No. It is within seed variation, and re-running one identical configuration moved the
  reported test F1 by about 1 point because the selected epoch changed (section 5). We only claim "no improvement".
- **Where does the corpus come from?** PubMed via NCBI E-utilities, query "diseases category[MeSH Terms], has
  abstract, English", publication years 1990–2012, sampled evenly per year with a fixed seed; the PMID list is in
  the repository (`dapt/pmids.txt`).

## 5. 噪声量级（同配置重跑）

- Zero-step control (`--max_steps 0`, then fine-tune, seed 1): the saved encoder is identical to bert-base-cased
  (0 of 197 tensors differ), yet test strict F1 is 85.77 against 85.89 for the direct run with the same seed; 8 of 940
  test sentences get a different prediction. The cause is non-deterministic computation on Apple MPS (the training
  loss already differs by about 3e-7 at step 50).
- Repeat of the direct run, seed 1, nothing changed: test strict F1 per fine-tuning epoch is 83.14 / 84.80 / 85.96
  (original run: 83.00 / 84.94 / 85.89). At a fixed epoch the runs agree within about 0.2 F1.
- But the dev F1 of epochs 2 and 3 is almost tied (84.62 vs 84.51 in the repeat, 84.57 vs 84.63 in the original), so
  the repeat run selects epoch 2 and reports test F1 84.80 instead of 85.89; 91 of 940 test sentences differ.
- Takeaway: with dev-based epoch selection, one configuration and one seed can move the reported test F1 by about
  1 point on this hardware. Differences of 0.2–0.3 between arms cannot be interpreted. At the fixed last epoch the
  paired differences are −0.33 ± 0.54 (1 epoch) and −0.42 ± 0.36 (3 epochs), the same conclusion.
