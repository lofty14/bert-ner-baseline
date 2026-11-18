Entity-level error analysis, `bert+dapt3ep` vs `bert`, split `test`, seeds 1, 2, 3. Strict IOB2 entities (as seqeval mode='strict'); counts are checked against each run's metrics.json. 940 sentences, 960 gold entities.

FP boundary = predicted entity overlaps a gold entity but the span differs; FP spurious = no overlap with any gold entity. FN boundary = gold entity overlaps a predicted entity; FN missed = no overlap. One predicted span can overlap two gold spans (and vice versa), so FP boundary and FN boundary need not be equal. P / R / F1 in the `all` rows are pooled over seeds (micro), so they can differ in the second decimal from the mean over seeds in results.md. `tokens` = start-end (end exclusive) token offsets.

### Totals over seeds

| label | seed | TP | FP | FP boundary | FP spurious | FN | FN boundary | FN missed | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert | all | 2495 | 466 | 234 | 232 | 385 | 212 | 173 | 84.26 | 86.63 | 85.43 |
| bert+dapt3ep | all | 2509 | 506 | 249 | 257 | 371 | 207 | 164 | 83.22 | 87.12 | 85.12 |
| Δ (b − a) | all | +14 | +40 | +15 | +25 | -14 | -5 | -9 | | | |

### Per seed

| label | seed | TP | FP | FP boundary | FP spurious | FN | FN boundary | FN missed | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert | 1 | 837 | 152 | 75 | 77 | 123 | 67 | 56 | 84.63 | 87.19 | 85.89 |
| bert | 2 | 836 | 159 | 80 | 79 | 124 | 73 | 51 | 84.02 | 87.08 | 85.52 |
| bert | 3 | 822 | 155 | 79 | 76 | 138 | 72 | 66 | 84.14 | 85.62 | 84.87 |
| bert+dapt3ep | 1 | 845 | 171 | 82 | 89 | 115 | 68 | 47 | 83.17 | 88.02 | 85.53 |
| bert+dapt3ep | 2 | 842 | 169 | 83 | 86 | 118 | 69 | 49 | 83.28 | 87.71 | 85.44 |
| bert+dapt3ep | 3 | 822 | 166 | 84 | 82 | 138 | 70 | 68 | 83.20 | 85.62 | 84.39 |

### Stable differences

Entities predicted by one label in at least 2 of 3 seeds and by the other label in none.
Nearest gold span: the overlapping gold entity with the largest overlap, else the closest gold entity in the same sentence, else none.

#### false positives new in `bert+dapt3ep`: 23

| seeds | id | tokens | predicted span | nearest gold span | sentence |
|---:|---:|---|---|---|---|
| 3/3 | 57 | 36-37 | DZ | none | RESULTS Six of 8 monozygotic ( MZ ) twin pairs were disease concordant , compared with 4 of 15 B27 - positive dizygotic ( DZ ) twin pairs ( 27 % ) and 4 of 32 DZ twin pairs overall ( 12 . 5 % ) . |
| 3/3 | 771 | 12-13 | deficiency | none | Southern blot and PCR analysis of DNA of one patient with homozygous deficiency ruled out the presence of a large deletion of the FH gene as the underlying defect for the deficiency . |
| 3/3 | 920 | 14-15 | APC | FAP | This study also provided evidence for the pathological nature of amino acid changes in APC associated with both FAP and non - FAP colorectal cancer patients . . |
| 2/3 | 0 | 12-13 | sporadic | sporadic T - cell leukaemia | Clustering of missense mutations in the ataxia - telangiectasia gene in a sporadic T - cell leukaemia . |
| 2/3 | 58 | 25-26 | MZ | none | Nonsignificant increases in similarity with regard to age at disease onset and all of the disease severity scores assessed were noted in disease - concordant MZ twins compared with concordant DZ twins . |
| 2/3 | 92 | 7-10 | primary prostate cancer | prostate cancer | Frequent inactivation of PTEN / MMAC1 in primary prostate cancer . |
| 2/3 | 136 | 26-27 | congenital | congenital DM | Decreased fertility of males with adult onset DM and contraction of the repeat upon male transmission contribute to the almost absent occurrence of paternal transmission of congenital DM . |
| 2/3 | 138 | 6-7 | congenital | congenital DM | We conclude that paternal transmission of congenital DM is rare and preferentially occurs with onset of DM past 30 years in the father . . |
| 2/3 | 145 | 11-13 | deficiency of | none | Low levels of beta hexosaminidase A in healthy individuals with apparent deficiency of this enzyme . |
| 2/3 | 160 | 14-15 | gastrulation | gastrulation defect | Aggregation of mutant Smad4 ES cells with wild - type tetraploid morulae rescues the gastrulation defect . |
| 2/3 | 214 | 21-22 | generalized | atrophic benign epidermolysis bullosa | Cycloheximide facilitates the identification of aberrant transcripts resulting from a novel splice - site mutation in COL17A1 in a patient with generalized atrophic benign epidermolysis bullosa . |
| 2/3 | 312 | 30-31 | isolated | isolated foveal hypoplasia | Mutations in PAX6 are responsible for human aniridia and have also been found in patients with Peters anomaly , with congenital cataracts , with autosomal dominant keratitis , and with isolated foveal hypoplasia . |
| 2/3 | 336 | 30-36 | autosomal dominant cerebellar ataxias type I | cerebellar ataxias type I | We compared horizontal eye movements ( visually guided saccades , antisaccades , and smooth pursuit ) in control subjects ( n = 14 ) and patients with three forms of autosomal dominant cerebellar ataxias type I spinocerebellar ataxias 1 and 2 ( SCA1 , n = 11 ; SCA2 , n = 10 ) and SCA3 / Machado - Joseph disease ( MJD ) ( n = 16 ) . |
| 2/3 | 512 | 53-54 | POMC | obesity | Recent studies in animal models elucidated a central role of alpha - MSH in the regulation of food intake by activation of the brain melanocortin - 4 - receptor ( MC4 - R ; refs 3 - 5 ) and the linkage of human obesity to chromosome 2 in close proximity to the POMC locus , led to the proposal of an association of POMC with human obesity . |
| 2/3 | 519 | 5-6 | mild | mild hyperphenylalaninemia | Phenylketonuria ( PKU ) and mild hyperphenylalaninemia ( MHP ) are allelic disorders caused by mutations in the gene encoding phenylalanine hydroxylase ( PAH ) . |
| 2/3 | 657 | 22-25 | multiple intestinal polyps | polyps | Mutations in APC are classically associated with familial adenomatous polyposis ( FAP ) , a highly penetrant autosomal dominant disorder characterized by multiple intestinal polyps and , without surgical intervention , the development of colorectal cancer ( CRC ) . |
| 2/3 | 836 | 3-7 | coproporphyrinogen oxidase gene defects | hereditary coproporphyria | Systematic analysis of coproporphyrinogen oxidase gene defects in hereditary coproporphyria and mutation update . |
| 2/3 | 868 | 31-32 | textbook | aniridia | This indicates that there is a heavy ascertainment bias in the selection of patients for PAX6 mutation analysis and that the missing PAX6 missense mutations frequently may underlie phenotypes distinct from textbook aniridia . |
| 2/3 | 903 | 17-19 | childhood cirrhosis | non - Indian childhood cirrhosis | The ATP7B gene has been excluded in the much rarer human copper overload disease non - Indian childhood cirrhosis , indicating genetic heterogeneity . |
| 2/3 | 904 | 23-26 | progressive liver disease | liver disease | By investigating the common autosomal recessive copper toxicosis ( CT ) in Bedlington terriers , we have identified a new locus involved in progressive liver disease . |
| 2/3 | 924 | 29-33 | . or colorectal cancer | colorectal cancer , | To assess the risk of this common APC allelic variant in colorectal carcinogenesis , we have analyzed a large cohort of unselected Ashkenazi Jewish subjects with adenomatous polyps and . or colorectal cancer , for the APC I1307K polymorphism . |
| 2/3 | 929 | 6-7 | APC | adenomas | Furthermore , compared with noncarriers , APC I1307K carriers had increased numbers of adenomas and colorectal cancers per patient ( P = . 03 ) , as well as a younger age at diagnosis . |
| 2/3 | 938 | 12-13 | benign | invasive lobular cancers | Immunohistochemical staining of human breast specimens also revealed BRCA1 nuclear foci in benign breast , invasive lobular cancers and low - grade ductal carcinomas . |

#### false positives gone in `bert+dapt3ep` (only in `bert`): 7

| seeds | id | tokens | predicted span | nearest gold span | sentence |
|---:|---:|---|---|---|---|
| 3/3 | 91 | 11-12 | neurodegeneration | dominantly inherited neurodegeneration | This mutation may be valuable for developing models of dominantly inherited neurodegeneration , as the early age of onset of symptoms suggests that this mutation may be particularly deleterious to the magnocellular neuron . . |
| 3/3 | 933 | 5-7 | BRCA1 tumour | tumour | Although the link between the BRCA1 tumour - suppressor gene and hereditary breast and ovarian cancer is established , the role , if any , of BRCA1 in non - familial cancers is unclear . |
| 2/3 | 425 | 18-25 | paternal uniparental disomy ( UPD ) 15 | uniparental disomy | Angelman syndrome ( AS ) is caused by chromosome 15q11 - q13 deletions of maternal origin , by paternal uniparental disomy ( UPD ) 15 , by imprinting defects , and by mutations in the UBE3A gene . |
| 2/3 | 471 | 3-6 | A - T | none | The ATM ( A - T , mutated ) gene on human chromosome 11q22 . |
| 2/3 | 623 | 10-12 | very - | very - long - chain acyl - coenzyme A dehydrogenase deficiency | Reversal of severe hypertrophic cardiomyopathy and excellent neuropsychologic outcome in very - long - chain acyl - coenzyme A dehydrogenase deficiency . |
| 2/3 | 697 | 8-9 | familial | breast and / or ovarian cancer | BRCA1 and BRCA2 account for most cases of familial , early onset breast and / or ovarian cancer and encode products that each interact with hRAD51 . |
| 2/3 | 768 | 7-9 | deficiency in | deficiency in the human complement regulatory protein Factor H | This study reports on six cases of deficiency in the human complement regulatory protein Factor H ( FH ) in the context of an acute renal disease . |

#### true positives new in `bert+dapt3ep`: 5

| seeds | id | tokens | predicted span | nearest gold span | sentence |
|---:|---:|---|---|---|---|
| 3/3 | 623 | 10-21 | very - long - chain acyl - coenzyme A dehydrogenase deficiency | (exact match) | Reversal of severe hypertrophic cardiomyopathy and excellent neuropsychologic outcome in very - long - chain acyl - coenzyme A dehydrogenase deficiency . |
| 2/3 | 3 | 21-23 | clonal malignancy | (exact match) | By analysing tumour DNA from patients with sporadic T - cell prolymphocytic leukaemia ( T - PLL ) , a rare clonal malignancy with similarities to a mature T - cell leukaemia seen in A - T , we demonstrate a high frequency of ATM mutations in T - PLL . |
| 2/3 | 10 | 39-43 | sporadic T - PLL | (exact match) | The evidence of a significant proportion of loss - of - function mutations and a complete absence of the normal copy of ATM in the majority of mutated tumours establishes somatic inactivation of this gene in the pathogenesis of sporadic T - PLL and suggests that ATM acts as a tumour suppressor . |
| 2/3 | 336 | 36-41 | spinocerebellar ataxias 1 and 2 | (exact match) | We compared horizontal eye movements ( visually guided saccades , antisaccades , and smooth pursuit ) in control subjects ( n = 14 ) and patients with three forms of autosomal dominant cerebellar ataxias type I spinocerebellar ataxias 1 and 2 ( SCA1 , n = 11 ; SCA2 , n = 10 ) and SCA3 / Machado - Joseph disease ( MJD ) ( n = 16 ) . |
| 2/3 | 933 | 6-7 | tumour | (exact match) | Although the link between the BRCA1 tumour - suppressor gene and hereditary breast and ovarian cancer is established , the role , if any , of BRCA1 in non - familial cancers is unclear . |

#### true positives lost in `bert+dapt3ep` (only in `bert`): 6

| seeds | id | tokens | predicted span | nearest gold span | sentence |
|---:|---:|---|---|---|---|
| 3/3 | 342 | 1-2 | SCA3 | (exact match) | In SCA3 , gaze - evoked nystagmus was often present as was saccade hypometria and smooth pursuit gain was markedly decreased . |
| 3/3 | 681 | 18-21 | renal cell carcinoma | (exact match) | To discover genes involved in von Hippel - Lindau ( VHL ) - mediated carcinogenesis , we used renal cell carcinoma cell lines stably transfected with wild - type VHL - expressing transgenes . |
| 2/3 | 0 | 12-17 | sporadic T - cell leukaemia | (exact match) | Clustering of missense mutations in the ataxia - telangiectasia gene in a sporadic T - cell leukaemia . |
| 2/3 | 53 | 12-14 | Rheumatic Diseases | (exact match) | METHODS Twins with AS were identified from the Royal National Hospital for Rheumatic Diseases database . |
| 2/3 | 210 | 0-1 | Piebaldism | (exact match) | Piebaldism with deafness : molecular evidence for an expanded syndrome . |
| 2/3 | 423 | 20-22 | craniosynostotic condition | (exact match) | The overlap in clinical features and the presence , in the same genes , of mutations for more than one craniosynostotic condition - such as Saethre - Chotzen , Crouzon , and Pfeiffer syndromes - support the hypothesis that TWIST and FGFRs are components of the same molecular pathway involved in the modulation of craniofacial and limb development in humans . . |

