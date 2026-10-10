from abc import ABCMeta, abstractmethod
from typing import List

from torch import Tensor

from mmseg.utils import ConfigType
from .decode_head import BaseDecodeHead


class BaseCascadeDecodeHead(BaseDecodeHead, metaclass=ABCMeta):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs, prev_output):

        pass

    def loss(self, inputs: List[Tensor], prev_output: Tensor,
             batch_data_samples: List[dict], train_cfg: ConfigType) -> Tensor:


        seg_logits = self.forward(inputs, prev_output)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)

        return losses

    def predict(self, inputs: List[Tensor], prev_output: Tensor,
                batch_img_metas: List[dict], tese_cfg: ConfigType):


        seg_logits = self.forward(inputs, prev_output)

        return self.predict_by_feat(seg_logits, batch_img_metas)
