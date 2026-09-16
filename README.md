# SAMO for RGB-to-Thermal Semantic Segmentation

This repository contains the clean training and evaluation release for the
SAMO+ models trained from Cityscapes, BDD100K, and Mapillary RGB sources and
evaluated on the FMB, SCUT, and SODA thermal domains.

## Released checkpoints

| RGB source | Checkpoint | FMB mIoU | SCUT mIoU | SODA mIoU | Average |
|---|---|---:|---:|---:|---:|
| Cityscapes | `checkpoints/citys_samo_plus_68.15.pth` | 58.83 | 75.65 | 70.01 | 68.16 |
| BDD100K | `checkpoints/bdd_samo_plus_69.34.pth` | 63.28 | 74.50 | 70.46 | 69.41 |
| Mapillary | `checkpoints/map_samo_plus_70.64.pth` | 63.88 | 75.36 | 72.82 | 70.69 |

The checkpoint files are distributed separately rather than stored in this
repository. Place the downloaded SAMO+ weights and converted DINOv2 ViT-L/14
backbone under `checkpoints/` using the filenames shown above.

## SAMO module version

This release contains only the SAMO+ v3 implementation. All three checkpoints
must be used with
`SpectrumDinoVisionTransformerV3`, `LoRASpectrumV3`, four adapter slots at
DINOv2 blocks `[7, 11, 15, 23]`, and semantic feature layout `BDN`.
The public entry points are the three configs under `configs/samo_plus_v3/`.
See [VERSIONING.md](VERSIONING.md) for the exact mapping.

Check all config/checkpoint pairs before evaluation:

```bash
python tools/check_checkpoint_compatibility.py
```

## Installation

The reference environment uses Python 3.10, PyTorch 2.0.1, CUDA 11.8,
MMCV 2.2.0, MMEngine 0.10.7, and MMDetection 3.3.0. A SAMO-customized copy of
MMSegmentation 1.2.2 is vendored in `mmseg/` so that the six-channel thermal
reference pipeline works without patching an installed MMSegmentation package.

```bash
conda create -n samo-rgb2t python=3.10 -y
conda activate samo-rgb2t
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 \
  -c pytorch -c nvidia -y
pip install -U openmim
mim install mmcv==2.2.0
pip install -r requirements.txt
```

Place the converted DINOv2 backbone here:

```text
checkpoints/dinov2_converted_512x512.pth
```

## Data layout

Set `RGB2T_DATA_ROOT` and `SAMO_STYLE_DIR` instead of editing the configs:

```bash
export RGB2T_DATA_ROOT=/path/to/RGB_to_Thermal
export SAMO_STYLE_DIR=/path/to/IR-StyleSet/50
```

Expected layout:

```text
RGB_to_Thermal/
├── Source_RGB/
│   ├── Citys/{image,label_remap}/
│   ├── BDD/{image,label_remap}/
│   └── Map/{image,label_remap}/
└── Target_Thermal/
    ├── FMB/{image,label_remap_nobike}/
    ├── SCUT/{image,label_remap}/
    └── SODA/{image,label_remap}/
```

The style directory is the unpaired thermal-reference pool used to construct
the six-channel `[grayscale RGB | thermal reference]` input.

## Evaluation

Evaluate all three checkpoints on FMB, SCUT, and SODA:

```bash
GPU_ID=0 bash scripts/test_all.sh
```

The standard protocol uses resize `1024x512`, sliding-window crop `512x512`,
stride `341x341`, and no TTA. Logs include per-class IoU and accuracy for every
target domain. To evaluate one checkpoint directly:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --config configs/samo_plus_v3/bdd_samo_plus_v3_dinov2_mask2former_512x512_bs1x4.py \
  --checkpoint checkpoints/bdd_samo_plus_69.34.pth \
  --backbone checkpoints/dinov2_converted_512x512.pth \
  --work-dir work_dirs/test/bdd \
  --resize-width 1024 --resize-height 512 \
  --crop-size 512 --stride 341
```

## Training

Train one source setting:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config configs/samo_plus_v3/bdd_samo_plus_v3_dinov2_mask2former_512x512_bs1x4.py \
  --work-dir work_dirs/train/bdd
```

Run the three source settings sequentially with `GPU_ID=0 bash
scripts/train_all.sh`. Add `RESUME=1` to resume existing runs.

## Model downloads

Model weights and datasets are hosted separately and are intentionally excluded
from this Git repository. After downloading the three SAMO+ checkpoints, verify
them with the hashes in `checkpoints/SHA256SUMS` before evaluation.

## Acknowledgements and license

The vendored `mmseg/` code is based on MMSegmentation 1.2.2 and retains its
Apache-2.0 notices; see `third_party_licenses/MMSegmentation-LICENSE`.
Add the appropriate license for the original SAMO code and released weights
before publishing the repository.
