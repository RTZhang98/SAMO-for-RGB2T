from typing import List, Optional

import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


@MODELS.register_module()
class MultimodalEncoderDecoder(BaseSegmentor):


    def __init__(self,
                 image_encoder: ConfigType,
                 text_encoder: ConfigType,
                 decode_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 asymetric_input: bool = True,
                 encoder_resolution: float = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            image_encoder.init_cfg = dict(
                type='Pretrained_Part', checkpoint=pretrained)
            text_encoder.init_cfg = dict(
                type='Pretrained_Part', checkpoint=pretrained)
            decode_head.init_cfg = dict(
                type='Pretrained_Part', checkpoint=pretrained)

        if asymetric_input:
            assert encoder_resolution is not None, \
                'if asymetric_input set True, ' \
                'clip_resolution must be a certain value'
        self.asymetric_input = asymetric_input
        self.encoder_resolution = encoder_resolution
        self.image_encoder = MODELS.build(image_encoder)
        self.text_encoder = MODELS.build(text_encoder)
        self._init_decode_head(decode_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:

        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:

        x = self.image_encoder(inputs)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:


        classifier_embeds = self.text_encoder()
        clip_inputs = inputs
        if self.asymetric_input:
            clip_inputs = F.interpolate(
                inputs, scale_factor=self.encoder_resolution, mode='bilinear')
        x = self.image_encoder(clip_inputs)
        seg_logits = self.decode_head.predict([inputs, x, classifier_embeds],
                                              batch_img_metas, self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:


        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:


        classifier_embeds = self.text_encoder()
        clip_inputs = inputs
        if self.asymetric_input:
            clip_inputs = F.interpolate(
                inputs, scale_factor=self.encoder_resolution, mode='bilinear')
        x = self.image_encoder(clip_inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            [inputs, x, classifier_embeds], data_samples)
        losses.update(loss_decode)

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

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:


        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
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


                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:


        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:


        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):


        assert rescale

        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)

        seg_pred = list(seg_pred)
        return seg_pred
