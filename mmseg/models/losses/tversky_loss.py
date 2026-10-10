import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss


@weighted_loss
def tversky_loss(pred,
                 target,
                 valid_mask,
                 alpha=0.3,
                 beta=0.7,
                 smooth=1,
                 class_weight=None,
                 ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            tversky_loss = binary_tversky_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                alpha=alpha,
                beta=beta,
                smooth=smooth)
            if class_weight is not None:
                tversky_loss *= class_weight[i]
            total_loss += tversky_loss
    return total_loss / num_classes


@weighted_loss
def binary_tversky_loss(pred,
                        target,
                        valid_mask,
                        alpha=0.3,
                        beta=0.7,
                        smooth=1):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    TP = torch.sum(torch.mul(pred, target) * valid_mask, dim=1)
    FP = torch.sum(torch.mul(pred, 1 - target) * valid_mask, dim=1)
    FN = torch.sum(torch.mul(1 - pred, target) * valid_mask, dim=1)
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - tversky


@LOSSES.register_module()
class TverskyLoss(nn.Module):


    def __init__(self,
                 smooth=1,
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 alpha=0.3,
                 beta=0.7,
                 loss_name='loss_tversky'):
        super().__init__()
        self.smooth = smooth
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        assert (alpha + beta == 1.0), 'Sum of alpha and beta but be 1.0!'
        self.alpha = alpha
        self.beta = beta
        self._loss_name = loss_name

    def forward(self, pred, target, **kwargs):
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * tversky_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            alpha=self.alpha,
            beta=self.beta,
            smooth=self.smooth,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):


        return self._loss_name
