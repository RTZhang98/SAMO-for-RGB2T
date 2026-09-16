# BDD100K SAMO+ to FMB: per-class results

Checkpoint: `checkpoints/bdd_samo_plus_69.34.pth`

| Class | IoU (%) | Accuracy (%) |
|---|---:|---:|
| road | 88.15 | 94.51 |
| sidewalk | 38.68 | 67.23 |
| building | 82.44 | 92.16 |
| pole | 26.23 | 31.92 |
| traffic light | 30.45 | 35.10 |
| traffic sign | 53.00 | 67.60 |
| vegetation | 82.06 | 88.11 |
| sky | 93.86 | 97.66 |
| person | 70.43 | 77.91 |
| car | 83.91 | 90.49 |
| truck | 54.14 | 70.62 |
| bus | 82.77 | 87.12 |
| motorcycle | 36.55 | 49.89 |
| bicycle | 0.00 | NaN |

FMB mIoU reported by the project evaluator: **63.28%** over 280 images.
The evaluator skips zero/NaN classes when computing its mean; the fixed
14-class arithmetic mean is approximately 58.76%.
