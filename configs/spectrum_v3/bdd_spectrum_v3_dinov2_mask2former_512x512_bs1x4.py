"""BDD100K RGB source to FMB/SCUT/SODA with spatial Spectrum v3."""

_base_ = [
    "../_base_/datasets/dg_bdd2thermal_512x512.py",
    "../_base_/default_runtime.py",
    "../_base_/models/samo_v3_dinov2_mask2former.py",
    "../_base_/samo_v3_source_runtime.py",
]

model = dict(
    backbone=dict(
        type="SpectrumDinoVisionTransformerV3",
        adapted_layers=[7, 11, 15, 23],
        Spectrum_config=dict(type="LoRASpectrumV3", num_layers=4),
    )
)

train_dataloader = dict(
    batch_size=4,
    dataset=dict(pipeline={{_base_.train_pipeline}}),
)

param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)
]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=5000)
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=5,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
