from typing import List, Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class OhemCrossEntropy(nn.Module):


    def __init__(self,
                 ignore_label: int = 255,
                 thres: float = 0.7,
                 min_kept: int = 100000,
                 loss_weight: float = 1.0,
                 class_weight: Optional[Union[List[float], str]] = None,
                 loss_name: str = 'loss_ohem'):
        super().__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.class_weight = class_weight

    def forward(self, score: Tensor, target: Tensor) -> Tensor:


        pred = F.softmax(score, dim=1)
        if self.class_weight is not None:
            class_weight = score.new_tensor(self.class_weight)
        else:
            class_weight = None

        pixel_losses = F.cross_entropy(
            score,
            target,
            weight=class_weight,
            ignore_index=self.ignore_label,
            reduction='none').contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0

        pred = pred.gather(1, tmp_target.unsqueeze(1))

        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        if pred.numel() > 0:
            min_value = pred[min(self.min_kept, pred.numel() - 1)]
        else:
            return score.new_tensor(0.0)
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return self.loss_weight * pixel_losses.mean()

    @property
    def loss_name(self):
        return self.loss_name_
