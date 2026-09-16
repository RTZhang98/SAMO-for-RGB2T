#!/usr/bin/env python3
"""Evaluate one Spectrum v3 checkpoint with resize/stride/flip-TTA settings."""

from __future__ import annotations

import argparse
import copy
import json
import os
import os.path as osp
import sys
from typing import Any


PROJECT_ROOT = osp.abspath(osp.dirname(osp.dirname(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmseg.registry import HOOKS

import rein  # noqa: F401,E402  Register project modules.


@HOOKS.register_module(force=True)
class PrefixTTACheckpointHook(Hook):
    """Prefix flat checkpoint keys after the backbone-loading hook runs.

    ``SegTTAModel`` stores the actual segmentor below ``module``. The regular
    Spectrum checkpoint and the pretrained-backbone hook both use flat keys
    such as ``backbone.*`` and ``decode_head.*``. MMEngine recursively loads
    state dictionaries, so the prefix must be added to the checkpoint itself.
    """

    priority = "LOW"

    def after_load_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not state_dict or all(key.startswith("module.") for key in state_dict):
            return
        prefixed = {
            key if key.startswith("module.") else f"module.{key}": value
            for key, value in state_dict.items()
        }
        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = prefixed
        else:
            checkpoint.clear()
            checkpoint.update(prefixed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--backbone",
        default="checkpoints/dinov2_converted_512x512.pth",
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--resize-width", type=int, required=True)
    parser.add_argument("--resize-height", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Average original and horizontal-flip prediction probabilities.",
    )
    parser.add_argument("--launcher", default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    return args


def transform_type(transform: Any) -> str:
    if isinstance(transform, dict):
        return str(transform.get("type", ""))
    return str(getattr(transform, "type", ""))


def update_test_pipeline(
    dataset_cfg: Any,
    scale: tuple[int, int],
    use_tta: bool,
) -> None:
    """Update every leaf dataset below a possible ConcatDataset."""
    if "datasets" in dataset_cfg:
        for child in dataset_cfg["datasets"]:
            update_test_pipeline(child, scale, use_tta)
        return

    pipeline = copy.deepcopy(list(dataset_cfg["pipeline"]))
    for transform in pipeline:
        kind = transform_type(transform)
        if kind in {"Resize", "StyleResize"}:
            transform["scale"] = scale
            transform["keep_ratio"] = True

    if use_tta:
        pack_indices = [
            index
            for index, transform in enumerate(pipeline)
            if transform_type(transform) == "PackSegInputs"
        ]
        if len(pack_indices) != 1:
            raise RuntimeError(
                f"Expected one PackSegInputs in test pipeline, got {pack_indices}"
            )
        pack_transform = pipeline.pop(pack_indices[0])
        # LoadAnnotations and ConcatStyle intentionally remain outside TTA.
        # Therefore RandomFlip sees the concatenated six-channel input and GT,
        # making RGB/reference/annotation transforms spatially synchronous.
        pipeline.append(
            dict(
                type="TestTimeAug",
                transforms=[
                    [
                        dict(
                            type="RandomFlip",
                            prob=0.0,
                            direction="horizontal",
                        ),
                        dict(
                            type="RandomFlip",
                            prob=1.0,
                            direction="horizontal",
                        ),
                    ],
                    [pack_transform],
                ],
            )
        )
    dataset_cfg["pipeline"] = pipeline


def json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (int, float, str, bool)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = str(value)
    return result


def main() -> None:
    args = parse_args()
    config_path = osp.abspath(args.config)
    checkpoint_path = osp.abspath(args.checkpoint)
    backbone_path = osp.abspath(args.backbone)
    work_dir = osp.abspath(args.work_dir)

    for path, label in (
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
        (backbone_path, "backbone"),
    ):
        if not osp.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    cfg = Config.fromfile(config_path)
    cfg.work_dir = work_dir
    cfg.launcher = args.launcher
    cfg.load_from = checkpoint_path
    cfg.resume = False
    # A fixed seed keeps dataloader workers and random thermal-reference
    # sampling reproducible. Strict deterministic CUDA kernels cannot be used
    # here because Mask2Former invokes cumsum_cuda_kernel, for which PyTorch
    # 2.0 has no deterministic implementation.
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    if "env_cfg" in cfg:
        cfg.env_cfg["cudnn_benchmark"] = False

    crop = (args.crop_size, args.crop_size)
    stride = (args.stride, args.stride)
    cfg.model["test_cfg"] = dict(mode="slide", crop_size=crop, stride=stride)
    update_test_pipeline(
        cfg.test_dataloader.dataset,
        (args.resize_width, args.resize_height),
        args.tta,
    )

    custom_hooks = list(cfg.get("custom_hooks", []))
    custom_hooks.append(
        dict(type="LoadBackboneHook", checkpoint_path=backbone_path)
    )

    if args.tta:
        tta_model = copy.deepcopy(cfg.get("tta_model", dict(type="SegTTAModel")))
        tta_model["module"] = cfg.model
        cfg.model = tta_model
        # This LOW-priority hook runs after LoadBackboneHook, then prefixes both
        # adapter/head and injected pretrained-backbone keys with ``module.``.
        custom_hooks.append(dict(type="PrefixTTACheckpointHook"))

    cfg.custom_hooks = custom_hooks
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    payload = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "backbone": backbone_path,
        "resize": [args.resize_width, args.resize_height],
        "crop_size": list(crop),
        "stride": list(stride),
        "tta": {
            "enabled": args.tta,
            "transforms": (
                ["identity", "horizontal_flip"] if args.tta else ["identity"]
            ),
            "merge": (
                "mean predicted class probabilities"
                if args.tta
                else "none"
            ),
        },
        "seed": args.seed,
        "metrics": json_safe_metrics(metrics),
        "work_dir": work_dir,
    }
    print("HYPERPARAM_RESULT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
