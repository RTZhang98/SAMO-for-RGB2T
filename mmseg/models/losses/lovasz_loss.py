import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.utils import is_list_of

from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss


def lovasz_grad(gt_sorted):


    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_binary_logits(logits, labels, ignore_index=None):


    logits = logits.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return logits, labels
    valid = (labels != ignore_index)
    vlogits = logits[valid]
    vlabels = labels[valid]
    return vlogits, vlabels


def flatten_probs(probs, labels, ignore_index=None):

    if probs.dim() == 3:

        B, H, W = probs.size()
        probs = probs.view(B, 1, H, W)
    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    valid = (labels != ignore_index)
    vprobs = probs[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobs, vlabels


def lovasz_hinge_flat(logits, labels):


    if len(labels) == 0:

        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits,
                 labels,
                 classes='present',
                 per_image=False,
                 class_weight=None,
                 reduction='mean',
                 avg_factor=None,
                 ignore_index=255):


    if per_image:
        loss = [
            lovasz_hinge_flat(*flatten_binary_logits(
                logit.unsqueeze(0), label.unsqueeze(0), ignore_index))
            for logit, label in zip(logits, labels)
        ]
        loss = weight_reduce_loss(
            torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_logits(logits, labels, ignore_index))
    return loss


def lovasz_softmax_flat(probs, labels, classes='present', class_weight=None):


    if probs.numel() == 0:

        return probs * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        if class_weight is not None:
            loss *= class_weight[c]
        losses.append(loss)
    return torch.stack(losses).mean()


def lovasz_softmax(probs,
                   labels,
                   classes='present',
                   per_image=False,
                   class_weight=None,
                   reduction='mean',
                   avg_factor=None,
                   ignore_index=255):


    if per_image:
        loss = [
            lovasz_softmax_flat(
                *flatten_probs(
                    prob.unsqueeze(0), label.unsqueeze(0), ignore_index),
                classes=classes,
                class_weight=class_weight)
            for prob, label in zip(probs, labels)
        ]
        loss = weight_reduce_loss(
            torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_softmax_flat(
            *flatten_probs(probs, labels, ignore_index),
            classes=classes,
            class_weight=class_weight)
    return loss


@MODELS.register_module()
class LovaszLoss(nn.Module):


    def __init__(self,
                 loss_type='multi_class',
                 classes='present',
                 per_image=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_lovasz'):
        super().__init__()
        assert loss_type in ('binary', 'multi_class'), "loss_type should be \
                                                    'binary' or 'multi_class'."

        if loss_type == 'binary':
            self.cls_criterion = lovasz_hinge
        else:
            self.cls_criterion = lovasz_softmax
        assert classes in ('all', 'present') or is_list_of(classes, int)
        if not per_image:
            assert reduction == 'none', "reduction should be 'none' when \
                                                        per_image is False."

        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self._loss_name = loss_name

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None


        if self.cls_criterion == lovasz_softmax:
            cls_score = F.softmax(cls_score, dim=1)

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            self.classes,
            self.per_image,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

    @property
    def loss_name(self):


        return self._loss_name
