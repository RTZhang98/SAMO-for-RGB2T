"""Shared source-training pipeline and optimizer for SAMO v3."""

import os


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="LoadStyleImageFromFile",
        style_folder=os.getenv("SAMO_STYLE_DIR", "data/IR-StyleSet/50"),
    ),
    dict(type="LoadAnnotations"),
    dict(type="RGB2Gray"),
    dict(
        type="RandomChoiceResize",
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type="StyleResize", scale=(512, 512)),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortionWithStyle"),
    dict(type="PackSegInputs"),
]

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW",
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999),
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "learnable_tokens": embed_multi,
            "reins.scale": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
