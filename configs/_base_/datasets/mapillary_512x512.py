import os

mapillary_type = "RGB2ThermalDataset"
mapillary_root = os.path.join(
    os.getenv("RGB2T_DATA_ROOT", "data/RGB_to_Thermal"), "Source_RGB", "Map"
)
mapillary_crop_size = (512, 512)
mapillary_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1024, 512)),
    dict(type="RandomCrop", crop_size=mapillary_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
mapillary_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1024, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_mapillary = dict(
    type=mapillary_type,
    data_root=mapillary_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label_remap",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=mapillary_train_pipeline,
)
val_mapillary = dict(
    type=mapillary_type,
    data_root=mapillary_root,
    data_prefix=dict(
        img_path="image",
        seg_map_path="label_remap",
    ),
    img_suffix=".jpg",
    seg_map_suffix=".png",
    pipeline=mapillary_test_pipeline,
)
