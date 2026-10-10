from mmseg.registry import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class NightDrivingDataset(CityscapesDataset):


    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtCoarse_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
