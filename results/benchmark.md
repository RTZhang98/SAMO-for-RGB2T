# FMB / SCUT / SODA benchmark

Protocol: resize 1024x512, sliding crop 512x512, stride 341x341, no TTA.

| Source | FMB mIoU | SCUT mIoU | SODA mIoU | Average mIoU |
|---|---:|---:|---:|---:|
| Cityscapes | 58.83 | 75.65 | 70.01 | 68.1633 |
| BDD100K | 63.28 | 74.50 | 70.46 | 69.4133 |
| Mapillary | 63.88 | 75.36 | 72.82 | 70.6867 |

The custom evaluator prints the full 14-class IoU/accuracy table for each
thermal target. Its reported mIoU skips classes whose IoU is zero or NaN.
