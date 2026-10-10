import os
from typing import Dict, List, Optional, Sequence, Union

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

try:
    from dsdl.dataset import DSDLDataset
except ImportError:
    DSDLDataset = None


@DATASETS.register_module()
class DSDLSegDataset(BaseSegDataset):


    METAINFO = {}

    def __init__(self,
                 specific_key_path: Dict = {},
                 pre_transform: Dict = {},
                 used_labels: Optional[Sequence] = None,
                 **kwargs) -> None:

        if DSDLDataset is None:
            raise RuntimeError(
                'Package dsdl is not installed. Please run "pip install dsdl".'
            )
        self.used_labels = used_labels

        loc_config = dict(type='LocalFileReader', working_dir='')
        if kwargs.get('data_root'):
            kwargs['ann_file'] = os.path.join(kwargs['data_root'],
                                              kwargs['ann_file'])
        required_fields = ['Image', 'LabelMap']

        self.dsdldataset = DSDLDataset(
            dsdl_yaml=kwargs['ann_file'],
            location_config=loc_config,
            required_fields=required_fields,
            specific_key_path=specific_key_path,
            transform=pre_transform,
        )
        BaseSegDataset.__init__(self, **kwargs)

    def load_data_list(self) -> List[Dict]:


        if self.used_labels:
            self._metainfo['classes'] = tuple(self.used_labels)
            self.label_map = self.get_label_map(self.used_labels)
        else:
            self._metainfo['classes'] = tuple(['background'] +
                                              self.dsdldataset.class_names)
        data_list = []

        for i, data in enumerate(self.dsdldataset):
            datainfo = dict(
                img_path=os.path.join(self.data_prefix['img_path'],
                                      data['Image'][0].location),
                seg_map_path=os.path.join(self.data_prefix['seg_map_path'],
                                          data['LabelMap'][0].location),
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[],
            )
            data_list.append(datainfo)

        return data_list

    def get_label_map(self,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:


        old_classes = ['background'] + self.dsdldataset.class_names
        if (new_classes is not None and old_classes is not None
                and list(new_classes) != list(old_classes)):

            label_map = {}
            if not set(new_classes).issubset(old_classes):
                raise ValueError(
                    f'new classes {new_classes} is not a '
                    f'subset of classes {old_classes} in class_dom.')
            for i, c in enumerate(old_classes):
                if c not in new_classes:
                    label_map[i] = 255
                else:
                    label_map[i] = new_classes.index(c)
            return label_map
        else:
            return None
