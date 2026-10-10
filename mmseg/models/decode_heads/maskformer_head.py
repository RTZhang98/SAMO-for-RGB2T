from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

try:
    from mmdet.models.dense_heads import MaskFormerHead as MMDET_MaskFormerHead
except ModuleNotFoundError:
    MMDET_MaskFormerHead = BaseModule

from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module()
class MaskFormerHead(MMDET_MaskFormerHead):


    def __init__(self,
                 num_classes: int = 150,
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.out_channels = kwargs['out_channels']
        self.align_corners = True
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index

        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)

    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):


        batch_img_metas = []
        batch_gt_instances = []
        for data_sample in batch_data_samples:


            metainfo = data_sample.metainfo
            metainfo['batch_input_shape'] = metainfo['img_shape']
            data_sample.set_metainfo(metainfo)
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)


            gt_labels = classes[classes != self.ignore_index]

            masks = []
            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros((0, gt_sem_seg.shape[-2],
                                        gt_sem_seg.shape[-1])).to(gt_sem_seg)
            else:
                gt_masks = torch.stack(masks).squeeze(1)

            instance_data = InstanceData(
                labels=gt_labels, masks=gt_masks.long())
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:


        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)


        all_cls_scores, all_mask_preds = self(x, batch_data_samples)


        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:


        batch_data_samples = []
        for metainfo in batch_img_metas:
            metainfo['batch_input_shape'] = metainfo['img_shape']
            batch_data_samples.append(SegDataSample(metainfo=metainfo))


        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]


        img_shape = batch_img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=img_shape,
            mode='bilinear',
            align_corners=False)


        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
        return seg_logits
