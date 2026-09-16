#!/usr/bin/env python3
"""Verify that every released SAMO+ checkpoint matches the v3 configuration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import rein  # noqa: E402,F401


RELEASES = {
    "citys": "citys_samo_plus_68.15.pth",
    "bdd": "bdd_samo_plus_69.34.pth",
    "map": "map_samo_plus_70.64.pth",
}
EXPECTED_LAYERS = [7, 11, 15, 23]
TOKEN_KEYS = (
    "backbone.spectrum.learnable_tokens_low_a",
    "backbone.spectrum.learnable_tokens_mid_a",
    "backbone.spectrum.learnable_tokens_high_a",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    return parser.parse_args()


def check_release(source: str, checkpoint_dir: Path) -> None:
    config_path = ROOT / "configs" / "samo_plus_v3" / (
        f"{source}_samo_plus_v3_dinov2_mask2former_512x512_bs1x4.py"
    )
    checkpoint_path = checkpoint_dir / RELEASES[source]
    cfg = Config.fromfile(config_path)

    errors = []
    if cfg.model.type != "SAMOPlusEncoderDecoder":
        errors.append(f"segmentor={cfg.model.type}")
    if cfg.model.backbone.type != "SpectrumDinoVisionTransformerV3":
        errors.append(f"backbone={cfg.model.backbone.type}")
    if cfg.model.backbone.Spectrum_config.type != "LoRASpectrumV3":
        errors.append(f"SAMO module={cfg.model.backbone.Spectrum_config.type}")
    if list(cfg.model.backbone.adapted_layers) != EXPECTED_LAYERS:
        errors.append(f"adapted_layers={cfg.model.backbone.adapted_layers}")
    if cfg.model.decode_head.type != "SpectrumMask2FormerHeadPlus":
        errors.append(f"head={cfg.model.decode_head.type}")
    if cfg.model.decode_head.samo_plus.semantic_feature_layout != "BDN":
        errors.append(
            "semantic_feature_layout="
            f"{cfg.model.decode_head.samo_plus.semantic_feature_layout}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    for key in TOKEN_KEYS:
        shape = tuple(state_dict[key].shape) if key in state_dict else None
        if shape != (4, 100, 16):
            errors.append(f"{key} shape={shape}, expected=(4, 100, 16)")

    prototype_levels = {
        key.split(".")[2]
        for key in state_dict
        if key.startswith("decode_head.semantic_losses.")
        and key.endswith(".semantic_prototypes")
    }
    if prototype_levels != {"0", "1", "2", "3"}:
        errors.append(f"SAMO+ prototype levels={sorted(prototype_levels)}")

    if errors:
        raise RuntimeError(f"{checkpoint_path.name}: " + "; ".join(errors))
    print(
        f"PASS {checkpoint_path.name}: SAMO+ v3, layers={EXPECTED_LAYERS}, "
        "adapter_slots=4, layout=BDN"
    )


def main() -> None:
    args = parse_args()
    for source in RELEASES:
        check_release(source, args.checkpoint_dir)


if __name__ == "__main__":
    main()
