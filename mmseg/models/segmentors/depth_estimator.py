import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from ..utils import resize
from .encoder_decoder import EncoderDecoder


@MODELS.register_module()
class DepthEstimator(EncoderDecoder):


    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def extract_feat(self,
                     inputs: Tensor,
                     batch_img_metas: Optional[List[dict]] = None) -> Tensor:


        if getattr(self.backbone, 'class_embed_select', False) and \
                isinstance(batch_img_metas, list) and \
                'category_id' in batch_img_metas[0]:
            cat_ids = [meta['category_id'] for meta in batch_img_metas]
            cat_ids = torch.tensor(cat_ids).to(inputs.device)
            inputs = (inputs, cat_ids)

        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:


        x = self.extract_feat(inputs, batch_img_metas)
        depth = self.decode_head.predict(x, batch_img_metas, self.test_cfg)

        return depth

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:


        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:


        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:


        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        x = self.extract_feat(inputs, batch_img_metas)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:


        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        depth = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(depth, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:


        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_flip_inference(self, inputs: Tensor,
                             batch_img_metas: List[dict]) -> Tensor:


        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]


                crop_depth_map = self.encode_decode(crop_img, batch_img_metas)


                crop_depth_map_flip = self.encode_decode(
                    crop_img.flip(dims=(3, )), batch_img_metas)
                crop_depth_map_flip = crop_depth_map_flip.flip(dims=(3, ))
                crop_depth_map = (crop_depth_map + crop_depth_map_flip) / 2.0

                preds += F.pad(crop_depth_map,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        depth = preds / count_mat

        return depth

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:


        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole',
                                                      'slide_flip'], \
            f'Only "slide", "slide_flip" or "whole" test mode are ' \
            f'supported, but got {self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            depth_map = self.slide_inference(inputs, batch_img_metas)
        if self.test_cfg.mode == 'slide_flip':
            depth_map = self.slide_flip_inference(inputs, batch_img_metas)
        else:
            depth_map = self.whole_inference(inputs, batch_img_metas)

        return depth_map

    def postprocess_result(self,
                           depth: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:


        batch_size, C, H, W = depth.shape

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

                i_depth = depth[i:i + 1, :, padding_top:H - padding_bottom,
                                padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_depth = i_depth.flip(dims=(3, ))
                    else:
                        i_depth = i_depth.flip(dims=(2, ))


                i_depth = resize(
                    i_depth,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_depth = depth[i]

            data_samples[i].set_data(
                {'pred_depth_map': PixelData(**{'data': i_depth})})

        return data_samples
