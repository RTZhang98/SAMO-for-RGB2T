_base_ = [
    "./mapillary_512x512.py",
    "./FMB_512x512.py",
    "./SCUT_512x512.py",
    "./SODA_512x512.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_mapillary}},
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_FMB}},
            {{_base_.val_SCUT}},
            {{_base_.val_SODA}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetricThermal", iou_metrics=["mIoU"], dataset_keys=["FMB", "SCUT", "SODA"]
)
test_evaluator=val_evaluator
