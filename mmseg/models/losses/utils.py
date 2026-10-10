import functools

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.fileio import load


def get_class_weight(class_weight):


    if isinstance(class_weight, str):

        if class_weight.endswith('.npy'):
            class_weight = np.load(class_weight)
        else:

            class_weight = load(class_weight)

    return class_weight


def reduce_loss(loss, reduction) -> torch.Tensor:


    reduction_enum = F._Reduction.get_enum(reduction)

    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss,
                       weight=None,
                       reduction='mean',
                       avg_factor=None) -> torch.Tensor:


    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight


    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:

        if reduction == 'mean':


            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)

        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):


    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):

        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper
