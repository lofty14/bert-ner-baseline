Entity-level error analysis, `bert+dapt1ep` vs `bert`, split `test`, seeds 1, 2, 3. Strict IOB2 entities (as seqeval mode='strict'); counts are checked against each run's metrics.json. 940 sentences, 960 gold entities.

FP boundary = predicted entity overlaps a gold entity but the span differs; FP spurious = no overlap with any gold entity. FN boundary = gold entity overlaps a predicted entity; FN missed = no overlap. One predicted span can overlap two gold spans (and vice versa), so FP boundary and FN boundary need not be equal. P / R / F1 in the `all` rows are pooled over seeds (micro), so they can differ in the second decimal from the mean over seeds in results.md. `tokens` = start-end (end exclusive) token offsets.

### Totals over seeds

| label | seed | TP | FP | FP boundary | FP spurious | FN | FN boundary | FN missed | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert | all | 2495 | 466 | 234 | 232 | 385 | 212 | 173 | 84.26 | 86.63 | 85.43 |
| bert+dapt1ep | all | 2503 | 492 | 246 | 246 | 377 | 217 | 160 | 83.57 | 86.91 | 85.21 |
| Δ (b − a) | all | +8 | +26 | +12 | +14 | -8 | +5 | -13 | | | |

### Per seed

| label | seed | TP | FP | FP boundary | FP spurious | FN | FN boundary | FN missed | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert | 1 | 837 | 152 | 75 | 77 | 123 | 67 | 56 | 84.63 | 87.19 | 85.89 |
| bert | 2 | 836 | 159 | 80 | 79 | 124 | 73 | 51 | 84.02 | 87.08 | 85.52 |
| bert | 3 | 822 | 155 | 79 | 76 | 138 | 72 | 66 | 84.14 | 85.62 | 84.87 |
| bert+dapt1ep | 1 | 841 | 155 | 73 | 82 | 119 | 66 | 53 | 84.44 | 87.60 | 85.99 |
| bert+dapt1ep | 2 | 833 | 173 | 91 | 82 | 127 | 81 | 46 | 82.80 | 86.77 | 84.74 |
| bert+dapt1ep | 3 | 829 | 164 | 82 | 82 | 131 | 70 | 61 | 83.48 | 86.35 | 84.90 |

### Stable differences

Entities predicted by one label in at least 2 of 3 seeds and by the other label in none.
Nearest gold span: the overlapping gold entity with the largest overlap, else the closest gold entity in the same sentence, else none.

#### false positives new in `bert+dapt1ep`: 11

| seeds | id | tokens | predicted span | nearest gold span | sentence |
|---:|---:|---|---|---|---|
| 3/3 | 92 | 7-10 | primary prostate cancer | prostate cancer | Frequent inactivation of PTEN / MMAC1 in primary prostate cancer . |
| 2/3 | 28 | 10-11 | LOH | tumors | In 6 ( 38 % ) of 16 tumors without LOH , one mutation was detected , and in 9 ( 56 % ) of the tumors without LOH , both mutations were found . |
| 2/3 | 28 | 28-29 | LOH | tumors | In 6 ( 38 % ) of 16 tumors without LOH , one mutation was detected , and in 9 ( 56 % ) of the tumors without LOH , both mutations were found . |
| 2/3 | 512 | 53-54 | POMC | obesity | Recent studies in animal models elucidated a central role of alpha - MSH in the regulation of food intake by activation of the brain melanocortin - 4 - receptor ( MC4 - R ; refs 3 - 5 ) and the linkage of human obesity to chromosome 2 in close proximity to the POMC locus , led to the proposal of an association of POMC with human obesity . |
| 2/3 | 566 | 4-5 | severe | von Willebrand disease | A mouse model of severe von Willebrand disease : defects in hemostasis and thrombosis . |
| 2/3 | 696 | 9-11 | BRCA2 tumor | tumor | Stable interaction between the products of the BRCA1 and BRCA2 tumor suppressor genes in mitotic and meiotic cells . |
| 2/3 | 768 | 7-10 | deficiency in the | deficiency in the human complement regulatory protein Factor H | This study reports on six cases of deficiency in the human complement regulatory protein Factor H ( FH ) in the context of an acute renal disease . |
| 2/3 | 873 | 26-27 | C2 | deficiency of the second component of complement | The relationship of the genes coding for HLA to those coding for properdin Factor B allotypes and for deficiency of the second component of complement ( C2 ) was studied in families of patients with connective tissue disorders . |
| 2/3 | 903 | 17-19 | childhood cirrhosis | non - Indian childhood cirrhosis | The ATP7B gene has been excluded in the much rarer human copper overload disease non - Indian childhood cirrhosis , indicating genetic heterogeneity . |
| 2/3 | 907 | 2-3 | BAC | WD | However , BAC clones containing ATP7B and C04107 mapped to the canine chromosome regions CFA22q11 and CFA10q26 , respectively , demonstrating that WD cannot be homologous to CT . |
| 2/3 | 920 | 14-15 | APC | FAP | This study also provided evidence for the pathological nature of amino acid changes in APC associated with both FAP and non - FAP colorectal cancer patients . . |

#### false positives gone in `bert+dapt1ep` (only in `bert`): 2

| seeds | id | tokens | predicted span | nearest gold span | sentence |
|---:|---:|---|---|---|---|
| 2/3 | 425 | 18-25 | paternal uniparental disomy ( UPD ) 15 | uniparental disomy | Angelman syndrome ( AS ) is caused by chromosome 15q11 - q13 deletions of maternal origin , by paternal uniparental disomy ( UPD ) 15 , by imprinting defects , and by mutations in the UBE3A gene . |
| 2/3 | 623 | 10-12 | very - | very - long - chain acyl - coenzyme A dehydrogenase deficiency | Reversal of severe hypertrophic cardiomyopathy and excellent neuropsychologic outcome in very - long - chain acyl - coenzyme A dehydrogenase deficiency . |

#### true positives new in `bert+dapt1ep`: 4

| seeds | id | tokens | predicted span | nearest gold span | sentence |
|---:|---:|---|---|---|---|
| 3/3 | 299 | 10-11 | CD | (exact match) | The first was an association noted in the group of CD families with breast disease . |
| 2/3 | 10 | 39-43 | sporadic T - PLL | (exact match) | The evidence of a significant proportion of loss - of - function mutations and a complete absence of the normal copy of ATM in the majority of mutated tumours establishes somatic inactivation of this gene in the pathogenesis of sporadic T - PLL and suggests that ATM acts as a tumour suppressor . |
| 2/3 | 159 | 0-2 | Growth retardation | (exact match) | Growth retardation of the Smad4 - deficient embryos results from reduced cell proliferation rather than increased apoptosis . |
| 2/3 | 623 | 10-21 | very - long - chain acyl - coenzyme A dehydrogenase deficiency | (exact match) | Reversal of severe hypertrophic cardiomyopathy and excellent neuropsychologic outcome in very - long - chain acyl - coenzyme A dehydrogenase deficiency . |

#### true positives lost in `bert+dapt1ep` (only in `bert`): 3

| seeds | id | tokens | predicted span | nearest gold span | sentence |
|---:|---:|---|---|---|---|
| 2/3 | 0 | 12-17 | sporadic T - cell leukaemia | (exact match) | Clustering of missense mutations in the ataxia - telangiectasia gene in a sporadic T - cell leukaemia . |
| 2/3 | 155 | 6-7 | tumor | (exact match) | Mutations in the SMAD4 / DPC4 tumor suppressor gene , a key signal transducer in most TGFbeta - related pathways , are involved in 50 % of pancreatic cancers . |
| 2/3 | 423 | 20-22 | craniosynostotic condition | (exact match) | The overlap in clinical features and the presence , in the same genes , of mutations for more than one craniosynostotic condition - such as Saethre - Chotzen , Crouzon , and Pfeiffer syndromes - support the hypothesis that TWIST and FGFRs are components of the same molecular pathway involved in the modulation of craniofacial and limb development in humans . . |

