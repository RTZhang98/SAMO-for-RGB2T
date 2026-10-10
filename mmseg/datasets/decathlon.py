import copy
import os.path as osp
from typing import List

from mmengine.fileio import load

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DecathlonDataset(BaseSegDataset):


    def load_data_list(self) -> List[dict]:


        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        raw_data_list = annotations[
            'training'] if not self.test_mode else annotations['test']
        data_list = []
        for raw_data_info in raw_data_list:


            if isinstance(raw_data_info, dict):
                data_info = dict(
                    img_path=osp.join(self.data_root, raw_data_info['image']
                                      [2:]))
                data_info['seg_map_path'] = osp.join(
                    self.data_root, raw_data_info['label'][2:])
            else:
                data_info = dict(
                    img_path=osp.join(self.data_root, raw_data_info)[2:])
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        annotations.pop('training')
        annotations.pop('test')

        metainfo = copy.deepcopy(annotations)
        metainfo['classes'] = [*metainfo['labels'].values()]


        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        return data_list
