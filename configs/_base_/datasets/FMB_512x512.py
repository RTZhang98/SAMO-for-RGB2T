import os

FMB_data_root = os.getenv("RGB2T_DATA_ROOT", "data/RGB_to_Thermal")
FMB_style_dir = os.getenv("SAMO_STYLE_DIR", "data/IR-StyleSet/50")
FMB_type = "RGB2ThermalDataset"
FMB_root = os.path.join(FMB_data_root, "Target_Thermal", "FMB")
FMB_crop_size = (512, 512)
FMB_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RGB2Gray"),
    dict(
        type="RandomChoiceResize",
        scales=[int(540 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size=FMB_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
FMB_test_pipeline = [
    # dict(type="LoadImageFromFile"),
    # dict(type="LoadAnnotations"),
    # dict(type="RGB2Gray"),
    # dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # dict(type="StyleResize", scale=(1024, 512), keep_ratio=True), # ours
    # # dict(type="PhotoMetricDistortion"),
    # dict(type="PhotoMetricDistortionWithStyle"), # ours
    # # dict(type="ConcatStyle"), # ours
    # dict(type="PackSegInputs"),

    dict(type="LoadImageFromFile"),
    dict(type="LoadStyleImageFromFile", style_folder=FMB_style_dir),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    dict(type="StyleResize", scale=(1024, 512), keep_ratio=True), # ours
    # dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    # dict(type="StyleResize", scale=(1280, 720), keep_ratio=True), # ours
    dict(type="ConcatStyle"), # ours
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_FMB = dict(
    type=FMB_type,
    data_root=os.path.join(FMB_data_root, "Train_Thermal", "FMB"),
    data_prefix=dict(
        img_path="image",
        seg_map_path="label_remap_nobike",
    ),
    pipeline=FMB_train_pipeline,
)
val_FMB = dict(
    type=FMB_type,
    data_root=FMB_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label_remap_nobike",
    ),
    pipeline=FMB_test_pipeline,
)
