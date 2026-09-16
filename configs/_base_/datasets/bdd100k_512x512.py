import os

bdd_type = "RGB2ThermalDataset"
bdd_root = os.path.join(
    os.getenv("RGB2T_DATA_ROOT", "data/RGB_to_Thermal"), "Source_RGB", "BDD"
)
bdd_crop_size = (512, 512)
bdd_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RGB2Gray"),
    dict(
        type="RandomChoiceResize",
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size=bdd_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
bdd_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_bdd = dict(
    type=bdd_type,
    data_root=bdd_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label_remap",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=bdd_train_pipeline,
)
val_bdd = dict(
    type=bdd_type,
    data_root=bdd_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label_remap",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=bdd_test_pipeline,
)
