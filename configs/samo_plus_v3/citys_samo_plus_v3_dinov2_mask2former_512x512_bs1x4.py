"""Cityscapes-source SAMO(+) v3 with label-anchored semantic prototypes."""

_base_ = [
    "../spectrum_v3/citys_spectrum_v3_dinov2_mask2former_512x512_bs1x4.py"
]

custom_imports = dict(
    imports=[
        "rein.models.heads.spectrum_mask2former_plus",
        "rein.models.segmentors.samo_plus_encoder_decoder",
    ],
    allow_failed_imports=False,
)

model = dict(
    type="SAMOPlusEncoderDecoder",
    decode_head=dict(
        type="SpectrumMask2FormerHeadPlus",
        samo_plus=dict(
            enabled=True,
            num_classes=14,
            feature_dim=1024,
            prototype_momentum=0.99,
            semantic_temperature=0.1,
            samples_per_class=256,
            ignore_index=255,
            distributed_sync=True,
            semantic_feature_layout="BDN",
            num_semantic_levels=4,
            alpha=0.5,
            beta=2.0,
            gamma=1.0,
        ),
    ),
)

train_dataloader = dict(batch_size=4)

param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=80000, by_epoch=False)
]
train_cfg = dict(
    type="IterBasedTrainLoop",
    max_iters=80000,
    val_begin=22000,
    val_interval=2000,
)
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=2000,
        save_begin=22000,
        max_keep_ckpts=30,
    )
)
