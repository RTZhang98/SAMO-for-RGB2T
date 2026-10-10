from abc import abstractmethod
from typing import Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import TASK_UTILS


class BaseMatchCost:


    def __init__(self, weight: Union[float, int] = 1.) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self, pred_instances: InstanceData,
                 gt_instances: InstanceData, **kwargs) -> Tensor:


        pass


@TASK_UTILS.register_module()
class ClassificationCost(BaseMatchCost):


    def __init__(self, weight: Union[float, int] = 1) -> None:
        super().__init__(weight=weight)

    def __call__(self, pred_instances: InstanceData,
                 gt_instances: InstanceData, **kwargs) -> Tensor:


        assert hasattr(pred_instances, 'scores'), \
            "pred_instances must contain 'scores'"
        assert hasattr(gt_instances, 'labels'), \
            "gt_instances must contain 'labels'"
        pred_scores = pred_instances.scores
        gt_labels = gt_instances.labels

        pred_scores = pred_scores.softmax(-1)
        cls_cost = -pred_scores[:, gt_labels]

        return cls_cost * self.weight


@TASK_UTILS.register_module()
class DiceCost(BaseMatchCost):


    def __init__(self,
                 pred_act: bool = False,
                 eps: float = 1e-3,
                 naive_dice: bool = True,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.pred_act = pred_act
        self.eps = eps
        self.naive_dice = naive_dice

    def _binary_mask_dice_loss(self, mask_preds: Tensor,
                               gt_masks: Tensor) -> Tensor:


        mask_preds = mask_preds.flatten(1)
        gt_masks = gt_masks.flatten(1).float()
        numerator = 2 * torch.einsum('nc,mc->nm', mask_preds, gt_masks)
        if self.naive_dice:
            denominator = mask_preds.sum(-1)[:, None] + \
                gt_masks.sum(-1)[None, :]
        else:
            denominator = mask_preds.pow(2).sum(1)[:, None] + \
                gt_masks.pow(2).sum(1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self, pred_instances: InstanceData,
                 gt_instances: InstanceData, **kwargs) -> Tensor:


        assert hasattr(pred_instances, 'masks'), \
            "pred_instances must contain 'masks'"
        assert hasattr(gt_instances, 'masks'), \
            "gt_instances must contain 'masks'"
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks

        if self.pred_act:
            pred_masks = pred_masks.sigmoid()
        dice_cost = self._binary_mask_dice_loss(pred_masks, gt_masks)
        return dice_cost * self.weight


@TASK_UTILS.register_module()
class CrossEntropyLossCost(BaseMatchCost):


    def __init__(self,
                 use_sigmoid: bool = True,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred: Tensor,
                              gt_labels: Tensor) -> Tensor:


        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
            torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost

    def __call__(self, pred_instances: InstanceData,
                 gt_instances: InstanceData, **kwargs) -> Tensor:


        assert hasattr(pred_instances, 'masks'), \
            "pred_instances must contain 'masks'"
        assert hasattr(gt_instances, 'masks'), \
            "gt_instances must contain 'masks'"
        pred_masks = pred_instances.masks
        gt_masks = gt_instances.masks
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(pred_masks, gt_masks)
        else:
            raise NotImplementedError

        return cls_cost * self.weight
