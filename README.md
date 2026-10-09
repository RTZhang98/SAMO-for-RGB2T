# Spectrum-Adaptive Modulation for Generalizable RGB-to-Thermal Semantic Segmentation

**Runtong Zhang<sup>1,2</sup>, Fanman Meng<sup>1,*</sup>, Zihuan Qiu<sup>1</sup>, Mingzhou He<sup>1</sup>, Xiwei Zhang<sup>1</sup>, Qingbo Wu<sup>1</sup>, Linfeng Xu<sup>1</sup>, Hongliang Li<sup>1</sup>**

- <sup>1</sup> School of Information and Communication Engineering, University of Electronic Science and Technology of China, Chengdu, China 
- <sup>2</sup> School of Information and Control Engineering, Southwest University of Science and Technology, Mianyang, China 

## Overview

[![Overview of Spectrum-Adaptive Modulation (SAMO)](figures/overview-v10.png)](figures/overview-v10.pdf)

Official **SAMO+ (v3)** implementation for RGB-to-thermal semantic segmentation, using a DINOv2 backbone and a Mask2Former head.

## Datasets

**Dataset download (Google Drive): [RTSS Target Domains](https://drive.google.com/file/d/1qmVgyf2mYjJIM2wyETS-IP14RpLusE_V/view?usp=drive_link).**

The RTSS source domains—Cityscapes, BDD100K, and Mapillary—can be downloaded from their official websites: [Cityscapes](https://www.cityscapes-dataset.com/downloads/), [BDD100K](https://bdd-data.berkeley.edu/), and [Mapillary Vistas](https://www.mapillary.com/dataset/vistas/).

## Performance and Checkpoints

All scores are mIoU (%); the average is the arithmetic mean over the three thermal evaluation domains. The following **SAMO+** results are reported in Table III of the manuscript.

| RGB source | FMB | SCUT | SODA | Average | Checkpoint filename | Google Drive |
|---|---:|---:|---:|---:|---|---|
| Cityscapes | 58.85 | 75.61 | 69.98 | 68.15 | `citys_samo_plus_68.15.pth` | [Download](https://drive.google.com/file/d/1uualWhSpEJxQm1XjoR2TcJTqbaO3To2F/view?usp=drive_link) |
| BDD100K | 63.24 | 74.33 | 70.45 | 69.34 | `bdd_samo_plus_69.34.pth` | [Download](https://drive.google.com/file/d/1KBM6giHLUsLpaCpeySDdngRnpQisBG9ly/view?usp=drive_link)  |
| Mapillary | 63.69 | 75.45 | 72.77 | 70.64 | `map_samo_plus_70.64.pth` | [Download](https://drive.google.com/file/d/18sivFxdxI3dY5C4oxIBXN_7aMXS1zB80/view?usp=drive_link)  |

**Evaluation protocol:** resize to 1024 × 512, sliding-window crop of 512 × 512, stride of 341 × 341, seed 3407. The project evaluator excludes classes with zero or NaN IoU when computing each domain's mIoU; these values use that evaluation convention.

Place the downloaded weights under `checkpoints/`. Evaluation also requires the converted DINOv2 backbone at `checkpoints/dinov2_converted_512x512.pth`.

## Citation



```bibtex
@ARTICLE{11715954,
  author={Zhang, Runtong and Meng, Fanman and Qiu, Zihuan and He, Mingzhou and Zhang, Xiwei and Wu, Qingbo and Xu, Linfeng and Li, Hongliang},
  journal={IEEE Transactions on Image Processing}, 
  title={Spectrum-Adaptive Modulation for Generalizable RGB-to-Thermal Semantic Segmentation}, 
  year={2026},
  volume={35},
  number={},
  pages={10481-10496},
  doi={10.1109/TIP.2026.3736812}}
```

## Acknowledgements

We thank the authors and maintainers of the following open-source projects for sharing their code and models:

- [Rein](https://github.com/w1oves/Rein), for the foundation-model adaptation codebase.
- [Mask2Former](https://github.com/facebookresearch/Mask2Former), for the segmentation architecture.
- [DINOv2](https://github.com/facebookresearch/dinov2), for the pretrained visual backbone.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMEngine](https://github.com/open-mmlab/mmengine), and [MMCV](https://github.com/open-mmlab/mmcv), for the training and evaluation framework.

We also thank the creators of Cityscapes, BDD100K, Mapillary, FMB, SCUT, SODA, and LLVIP for making their datasets available to the research community.
