#!/usr/bin/env python3
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
from mmengine.runner import Runner

import rein  # noqa: F401,E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAMO+.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="checkpoints/dinov2_converted_512x512.pth")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--resize-width", type=int, required=True)
    parser.add_argument("--resize-height", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--launcher", default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    return args

def update_test_pipeline(dataset_cfg: Any, scale: tuple[int, int]) -> None:
    if "datasets" in dataset_cfg:
        for child in dataset_cfg["datasets"]:
            update_test_pipeline(child, scale)
        return
    pipeline = copy.deepcopy(list(dataset_cfg["pipeline"]))
    for transform in pipeline:
        if transform.get("type") in {"Resize", "StyleResize"}:
            transform["scale"] = scale
            transform["keep_ratio"] = True
    dataset_cfg["pipeline"] = pipeline

def json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result = {}
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
    cfg.randomness = dict(seed=args.seed, deterministic=False)
    if "env_cfg" in cfg:
        cfg.env_cfg["cudnn_benchmark"] = False
    crop = (args.crop_size, args.crop_size)
    stride = (args.stride, args.stride)
    cfg.model["test_cfg"] = dict(mode="slide", crop_size=crop, stride=stride)
    update_test_pipeline(
        cfg.test_dataloader.dataset, (args.resize_width, args.resize_height)
    )
    cfg.custom_hooks = list(cfg.get("custom_hooks", [])) + [
        dict(type="LoadBackboneHook", checkpoint_path=backbone_path)
    ]
    metrics = Runner.from_cfg(cfg).test()
    payload = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "backbone": backbone_path,
        "resize": [args.resize_width, args.resize_height],
        "crop_size": list(crop),
        "stride": list(stride),
        "seed": args.seed,
        "metrics": json_safe_metrics(metrics),
        "work_dir": work_dir,
    }
    print("HYPERPARAM_RESULT_JSON=" + json.dumps(payload, sort_keys=True))

if __name__ == "__main__":
    main()
