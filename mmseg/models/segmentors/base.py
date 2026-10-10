from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from mmengine.model import BaseModel
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.structures import SegDataSample
from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
from ..utils import resize


class BaseSegmentor(BaseModel, metaclass=ABCMeta):


    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_neck(self) -> bool:

        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self) -> bool:

        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self) -> bool:

        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, inputs: Tensor) -> bool:

        pass

    @abstractmethod
    def encode_decode(self, inputs: Tensor, batch_data_samples: SampleList):


        pass

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:


        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:

        pass

    @abstractmethod
    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:


        pass

    @abstractmethod
    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:


        pass

    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:


        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo

                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size

                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))


                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples
