import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class REFUGEDataset(BaseSegDataset):


    METAINFO = dict(
        classes=('background', ' Optic Cup', 'Optic Disc'),
        palette=[[120, 120, 120], [6, 230, 230], [56, 59, 120]])

    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)
