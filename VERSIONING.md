# SAMO+ v3 release mapping

This repository contains only the spatially correct SAMO+ v3 implementation.

- Backbone wrapper: `SpectrumDinoVisionTransformerV3`
- SAMO module: `LoRASpectrumV3`
- Implementation: `rein/models/backbones/spectrum_v3.py`
- Shared parameter layers: `rein/models/backbones/spectrum_core.py`
- Module placement: DINOv2 blocks `[7, 11, 15, 23]`
- Compact adapter slots: `[0, 1, 2, 3]`
- Semantic feature layout: `BDN`

## Released models

| Checkpoint | SAMO module | Head | Compatible config |
|---|---|---|---|
| `citys_samo_plus_68.15.pth` | `LoRASpectrumV3` | `SpectrumMask2FormerHeadPlus` | `configs/samo_plus_v3/citys_samo_plus_v3_dinov2_mask2former_512x512_bs1x4.py` |
| `bdd_samo_plus_69.34.pth` | `LoRASpectrumV3` | `SpectrumMask2FormerHeadPlus` | `configs/samo_plus_v3/bdd_samo_plus_v3_dinov2_mask2former_512x512_bs1x4.py` |
| `map_samo_plus_70.64.pth` | `LoRASpectrumV3` | `SpectrumMask2FormerHeadPlus` | `configs/samo_plus_v3/map_samo_plus_v3_dinov2_mask2former_512x512_bs1x4.py` |
