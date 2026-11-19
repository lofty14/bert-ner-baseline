Paired comparison `bert+dapt3ep` − `bert` over seeds 1, 2, 3 (precision fp32, dev selection strict_f1). Scores are percentages, seqeval strict IOB2 entity level; Δ = b − a.

#### dev (selected epoch)

| seed | bert P | bert R | bert F1 | bert+dapt3ep P | bert+dapt3ep R | bert+dapt3ep F1 | ΔP | ΔR | ΔF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 82.59 | 86.79 | 84.63 | 81.45 | 87.04 | 84.15 | -1.14 | +0.25 | -0.48 |
| 2 | 83.92 | 87.55 | 85.70 | 81.87 | 86.66 | 84.20 | -2.05 | -0.89 | -1.50 |
| 3 | 84.30 | 86.66 | 85.46 | 83.07 | 86.66 | 84.83 | -1.23 | +0.00 | -0.64 |
| mean ± std |  |  |  |  |  |  | -1.47 ± 0.50 | -0.21 ± 0.60 | -0.87 ± 0.55 |
| seeds Δ > 0 |  |  |  |  |  |  | 0/3 | 1/3 | 0/3 |

#### test, dev-selected epoch

| seed | bert P | bert R | bert F1 | bert+dapt3ep P | bert+dapt3ep R | bert+dapt3ep F1 | ΔP | ΔR | ΔF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.63 | 87.19 | 85.89 | 83.17 | 88.02 | 85.53 | -1.46 | +0.83 | -0.36 |
| 2 | 84.02 | 87.08 | 85.52 | 83.28 | 87.71 | 85.44 | -0.74 | +0.62 | -0.09 |
| 3 | 84.14 | 85.62 | 84.87 | 83.20 | 85.62 | 84.39 | -0.94 | +0.00 | -0.48 |
| mean ± std |  |  |  |  |  |  | -1.04 ± 0.37 | +0.49 ± 0.43 | -0.31 ± 0.20 |
| seeds Δ > 0 |  |  |  |  |  |  | 0/3 | 2/3 | 0/3 |

#### test, fixed last epoch (strict F1)

| seed | bert F1 | bert+dapt3ep F1 | ΔF1 |
|---:|---:|---:|---:|
| 1 | 85.89 | 85.53 | -0.36 |
| 2 | 85.52 | 85.44 | -0.09 |
| 3 | 85.25 | 84.45 | -0.80 |
| mean ± std | | | -0.42 ± 0.36 (0/3 > 0) |

`bert+dapt3ep` init_from: runs_dapt/bert_dapt_3ep/model
