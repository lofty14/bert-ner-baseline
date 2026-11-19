Paired comparison `bert+dapt1ep` − `bert` over seeds 1, 2, 3 (precision fp32, dev selection strict_f1). Scores are percentages, seqeval strict IOB2 entity level; Δ = b − a.

#### dev (selected epoch)

| seed | bert P | bert R | bert F1 | bert+dapt1ep P | bert+dapt1ep R | bert+dapt1ep F1 | ΔP | ΔR | ΔF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 82.59 | 86.79 | 84.63 | 82.45 | 86.53 | 84.44 | -0.14 | -0.25 | -0.20 |
| 2 | 83.92 | 87.55 | 85.70 | 81.19 | 86.66 | 83.84 | -2.73 | -0.89 | -1.86 |
| 3 | 84.30 | 86.66 | 85.46 | 82.55 | 86.53 | 84.49 | -1.76 | -0.13 | -0.97 |
| mean ± std |  |  |  |  |  |  | -1.54 ± 1.31 | -0.42 ± 0.41 | -1.01 ± 0.83 |
| seeds Δ > 0 |  |  |  |  |  |  | 0/3 | 0/3 | 0/3 |

#### test, dev-selected epoch

| seed | bert P | bert R | bert F1 | bert+dapt1ep P | bert+dapt1ep R | bert+dapt1ep F1 | ΔP | ΔR | ΔF1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.63 | 87.19 | 85.89 | 84.44 | 87.60 | 85.99 | -0.19 | +0.42 | +0.10 |
| 2 | 84.02 | 87.08 | 85.52 | 82.80 | 86.77 | 84.74 | -1.22 | -0.31 | -0.78 |
| 3 | 84.14 | 85.62 | 84.87 | 83.48 | 86.35 | 84.90 | -0.65 | +0.73 | +0.02 |
| mean ± std |  |  |  |  |  |  | -0.69 ± 0.51 | +0.28 ± 0.53 | -0.22 ± 0.49 |
| seeds Δ > 0 |  |  |  |  |  |  | 0/3 | 2/3 | 2/3 |

#### test, fixed last epoch (strict F1)

| seed | bert F1 | bert+dapt1ep F1 | ΔF1 |
|---:|---:|---:|---:|
| 1 | 85.89 | 86.15 | +0.26 |
| 2 | 85.52 | 84.74 | -0.78 |
| 3 | 85.25 | 84.76 | -0.48 |
| mean ± std | | | -0.33 ± 0.54 (1/3 > 0) |

`bert+dapt1ep` init_from: runs_dapt/bert_dapt_1ep/model
