import os

cityscapes_type = "RGB2ThermalDataset"
cityscapes_root = os.path.join(
    os.getenv("RGB2T_DATA_ROOT", "data/RGB_to_Thermal"), "Source_RGB", "Citys"
)
cityscapes_crop_size = (512, 512)
cityscapes_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=cityscapes_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
cityscapes_test_pipeline = [
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
    dict(type="LoadStyleImageFromFile", style_folder=os.getenv("SAMO_STYLE_DIR", "data/IR-StyleSet/50")),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    dict(type="StyleResize", scale=(1024, 512), keep_ratio=True), # ours
    # dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    # dict(type="StyleResize", scale=(1280, 720), keep_ratio=True), # ours
    dict(type="ConcatStyle"), # ours
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label_remap",
    ),
    pipeline=cityscapes_train_pipeline,
)
val_cityscapes = dict(
    type=cityscapes_type,
    data_root=cityscapes_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label_remap",
    ),
    pipeline=cityscapes_test_pipeline,
)
