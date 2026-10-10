import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from torch import Tensor

from mmseg.registry import MODELS
from .utils import get_class_weight, weighted_loss


def compute_dtm(img_gt: Tensor, pred: Tensor) -> Tensor:


    fg_dtm = torch.zeros_like(pred)
    out_shape = pred.shape
    for b in range(out_shape[0]):
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].byte()
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = torch.from_numpy(posdis)

    return fg_dtm


@weighted_loss
def hd_loss(seg_soft: Tensor,
            gt: Tensor,
            seg_dtm: Tensor,
            gt_dtm: Tensor,
            class_weight=None,
            ignore_index=255) -> Tensor:


    assert seg_soft.shape[0] == gt.shape[0]
    total_loss = 0
    num_class = seg_soft.shape[1]
    if class_weight is not None:
        assert class_weight.ndim == num_class
    for i in range(1, num_class):
        if i != ignore_index:
            delta_s = (seg_soft[:, i, ...] - gt.float())**2
            s_dtm = seg_dtm[:, i, ...]**2
            g_dtm = gt_dtm[:, i, ...]**2
            dtm = s_dtm + g_dtm
            multiplied = torch.einsum('bxy, bxy->bxy', delta_s, dtm)
            hd_loss = multiplied.mean()
        if class_weight is not None:
            hd_loss *= class_weight[i]
        total_loss += hd_loss

    return total_loss / num_class


@MODELS.register_module()
class HuasdorffDisstanceLoss(nn.Module):


    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='loss_huasdorff_disstance',
                 **kwargs):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self._loss_name = loss_name
        self.ignore_index = ignore_index

    def forward(self,
                pred: Tensor,
                target: Tensor,
                avg_factor=None,
                reduction_override=None,
                **kwargs) -> Tensor:


        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred_soft = F.softmax(pred, dim=1)
        valid_mask = (target != self.ignore_index).long()
        target = target * valid_mask

        with torch.no_grad():
            gt_dtm = compute_dtm(target.cpu(), pred_soft)
            gt_dtm = gt_dtm.float()
            seg_dtm2 = compute_dtm(
                pred_soft.argmax(dim=1, keepdim=False).cpu(), pred_soft)
            seg_dtm2 = seg_dtm2.float()

        loss_hd = self.loss_weight * hd_loss(
            pred_soft,
            target,
            seg_dtm=seg_dtm2,
            gt_dtm=gt_dtm,
            reduction=reduction,
            avg_factor=avg_factor,
            class_weight=class_weight,
            ignore_index=self.ignore_index)
        return loss_hd

    @property
    def loss_name(self):
        return self._loss_name
