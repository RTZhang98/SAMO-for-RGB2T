from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class RGB2ThermalDataset(BaseSegDataset):


    METAINFO = dict(
        classes=('road', 'sidewalk', 'building',
                 'pole','traffic light', 'traffic sign', 'vegetation',
                 'sky', 'person', 'car', 'truck',
                 'bus', 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70],
                 [153, 153, 153], [250, 170,30], [220, 220, 0],[107, 142, 35],
                 [70, 130, 180],[220, 20, 60], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 0, 230], [119, 11, 32]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
