import math

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead


class SelfAttentionBlock(_SelfAttentionBlock):


    def __init__(self, in_channels, channels, conv_cfg, norm_cfg, act_cfg):
        super().__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=True,
            with_out=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.output_project = self.build_project(
            in_channels,
            in_channels,
            num_convs=1,
            use_conv_module=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):

        context = super().forward(x, x)
        return self.output_project(context)


@MODELS.register_module()
class ISAHead(BaseDecodeHead):


    def __init__(self, isa_channels, down_factor=(8, 8), **kwargs):
        super().__init__(**kwargs)
        self.down_factor = down_factor

        self.in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.global_relation = SelfAttentionBlock(
            self.channels,
            isa_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.local_relation = SelfAttentionBlock(
            self.channels,
            isa_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.out_conv = ConvModule(
            self.channels * 2,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):

        x_ = self._transform_inputs(inputs)
        x = self.in_conv(x_)
        residual = x

        n, c, h, w = x.size()
        loc_h, loc_w = self.down_factor
        glb_h, glb_w = math.ceil(h / loc_h), math.ceil(w / loc_w)
        pad_h, pad_w = glb_h * loc_h - h, glb_w * loc_w - w
        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                       pad_h - pad_h // 2)
            x = F.pad(x, padding)


        x = x.view(n, c, glb_h, loc_h, glb_w, loc_w)

        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(-1, c, glb_h, glb_w)

        x = self.global_relation(x)


        x = x.view(n, loc_h, loc_w, c, glb_h, glb_w)

        x = x.permute(0, 4, 5, 3, 1, 2)
        x = x.reshape(-1, c, loc_h, loc_w)

        x = self.local_relation(x)


        x = x.view(n, glb_h, glb_w, c, loc_h, loc_w)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(n, c, glb_h * loc_h, glb_w * loc_w)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w // 2 + w]

        x = self.out_conv(torch.cat([x, residual], dim=1))
        out = self.cls_seg(x)

        return out
