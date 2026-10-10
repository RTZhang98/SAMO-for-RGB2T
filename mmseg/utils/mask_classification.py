from typing import List, Tuple

import torch
from mmcv.ops import point_sample
from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import TASK_UTILS
from mmseg.utils import ConfigType, SampleList


def seg_data_to_instance_data(ignore_index: int,
                              batch_data_samples: SampleList):


    batch_gt_instances = []

    for data_sample in batch_data_samples:
        gt_sem_seg = data_sample.gt_sem_seg.data
        classes = torch.unique(
            gt_sem_seg,
            sorted=False,
            return_inverse=False,
            return_counts=False)


        gt_labels = classes[classes != ignore_index]

        masks = []
        for class_id in gt_labels:
            masks.append(gt_sem_seg == class_id)

        if len(masks) == 0:
            gt_masks = torch.zeros(
                (0, gt_sem_seg.shape[-2],
                 gt_sem_seg.shape[-1])).to(gt_sem_seg).long()
        else:
            gt_masks = torch.stack(masks).squeeze(1).long()

        instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
        batch_gt_instances.append(instance_data)
    return batch_gt_instances


class MatchMasks:


    def __init__(self,
                 num_points: int,
                 num_queries: int,
                 num_classes: int,
                 assigner: ConfigType = None):
        assert assigner is not None, "\'assigner\' in decode_head.train_cfg" \
                                     'cannot be None'
        assert num_points > 0, 'num_points should be a positive integer.'
        self.num_points = num_points
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.assigner = TASK_UTILS.build(assigner)

    def get_targets(self, cls_scores: List[Tensor], mask_preds: List[Tensor],
                    batch_gt_instances: List[InstanceData]) -> Tuple:


        batch_size = cls_scores.shape[0]
        results = dict({
            'labels': [],
            'mask_targets': [],
            'mask_weights': [],
        })
        for i in range(batch_size):
            labels, mask_targets, mask_weights\
                = self._get_targets_single(cls_scores[i],
                                           mask_preds[i],
                                           batch_gt_instances[i])
            results['labels'].append(labels)
            results['mask_targets'].append(mask_targets)
            results['mask_weights'].append(mask_weights)


        labels = torch.stack(results['labels'], dim=0)

        mask_targets = torch.cat(results['mask_targets'], dim=0)

        mask_weights = torch.stack(results['mask_weights'], dim=0)

        avg_factor = sum(
            [len(gt_instances.labels) for gt_instances in batch_gt_instances])

        res = (labels, mask_targets, mask_weights, avg_factor)

        return res

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData) \
            -> Tuple[Tensor, Tensor, Tensor]:


        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks

        if len(gt_labels) == 0:
            labels = gt_labels.new_full((self.num_queries, ),
                                        self.num_classes,
                                        dtype=torch.long)
            mask_targets = gt_labels
            mask_weights = gt_labels.new_zeros((self.num_queries, ))
            return labels, mask_targets, mask_weights

        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)

        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)

        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)

        matched_quiery_inds, matched_label_inds = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances)
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[matched_quiery_inds] = gt_labels[matched_label_inds]

        mask_weights = gt_labels.new_zeros((self.num_queries, ))
        mask_weights[matched_quiery_inds] = 1
        mask_targets = gt_masks[matched_label_inds]

        return labels, mask_targets, mask_weights
