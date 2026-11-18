Scores are percentages, mean ± sample std over seeds. `default` = seqeval default mode, `strict` = mode='strict', scheme=IOB2.

### precision: fp32; dev selection metric: strict_f1

#### dev (selected epoch)

| model | n seeds | P (default) | R (default) | F1 (default) | P (strict) | R (strict) | F1 (strict) | token acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bert | 3 (1,2,3) | 81.17 ± 0.37 | 87.25 ± 0.48 | 84.10 ± 0.40 | 83.60 ± 0.90 | 87.00 ± 0.48 | 85.26 ± 0.56 | 98.63 ± 0.05 |
| bert+dapt1ep | 3 (1,2,3) | 80.17 ± 0.47 | 86.79 ± 0.13 | 83.34 ± 0.21 | 82.06 ± 0.76 | 86.57 ± 0.07 | 84.26 ± 0.36 | 98.59 ± 0.04 |
| bert+dapt3ep | 3 (1,2,3) | 80.19 ± 0.62 | 86.91 ± 0.22 | 83.42 ± 0.26 | 82.13 ± 0.84 | 86.79 ± 0.22 | 84.39 ± 0.38 | 98.60 ± 0.04 |

#### test, dev-selected epoch

| model | n seeds | P (default) | R (default) | F1 (default) | P (strict) | R (strict) | F1 (strict) | token acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bert | 3 (1,2,3) | 81.57 ± 0.11 | 87.29 ± 0.68 | 84.33 ± 0.35 | 84.26 ± 0.32 | 86.63 ± 0.87 | 85.43 ± 0.51 | 98.30 ± 0.05 |
| bert+dapt1ep | 3 (1,2,3) | 81.18 ± 1.02 | 87.60 ± 0.55 | 84.27 ± 0.73 | 83.58 ± 0.82 | 86.91 ± 0.64 | 85.21 ± 0.68 | 98.33 ± 0.03 |
| bert+dapt3ep | 3 (1,2,3) | 81.25 ± 0.63 | 87.88 ± 1.05 | 84.44 ± 0.79 | 83.22 ± 0.06 | 87.12 ± 1.30 | 85.12 ± 0.63 | 98.35 ± 0.05 |

#### test, fixed last epoch

| model | n seeds | last epoch | P (default) | R (default) | F1 (default) | P (strict) | R (strict) | F1 (strict) | token acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert | 3 (1,2,3) | 3,3,3 | 81.57 ± 0.12 | 87.57 ± 0.30 | 84.46 ± 0.15 | 84.08 ± 0.53 | 87.08 ± 0.10 | 85.55 ± 0.32 | 98.32 ± 0.02 |
| bert+dapt1ep | 3 (1,2,3) | 3,3,3 | 81.39 ± 1.31 | 87.57 ± 0.91 | 84.36 ± 1.03 | 83.53 ± 0.91 | 86.98 ± 0.75 | 85.22 ± 0.81 | 98.34 ± 0.03 |
| bert+dapt3ep | 3 (1,2,3) | 3,3,3 | 81.16 ± 0.77 | 88.09 ± 0.69 | 84.48 ± 0.71 | 82.96 ± 0.46 | 87.43 ± 0.77 | 85.14 ± 0.60 | 98.34 ± 0.07 |

#### model selection

| model | best dev epoch per seed | epochs run |
|---|---|---|
| bert | s1:3, s2:3, s3:2 | s1:3, s2:3, s3:3 |
| bert+dapt1ep | s1:2, s2:3, s3:2 | s1:3, s2:3, s3:3 |
| bert+dapt3ep | s1:3, s2:3, s3:2 | s1:3, s2:3, s3:3 |
