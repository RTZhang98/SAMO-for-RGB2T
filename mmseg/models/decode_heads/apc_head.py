import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead


class ACM(nn.Module):


    def __init__(self, pool_scale, fusion, in_channels, channels, conv_cfg,
                 norm_cfg, act_cfg):
        super().__init__()
        self.pool_scale = pool_scale
        self.fusion = fusion
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pooled_redu_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.input_redu_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.global_info = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.gla = nn.Conv2d(self.channels, self.pool_scale**2, 1, 1, 0)

        self.residual_conv = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.fusion:
            self.fusion_conv = ConvModule(
                self.channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, x):

        pooled_x = F.adaptive_avg_pool2d(x, self.pool_scale)

        x = self.input_redu_conv(x)

        pooled_x = self.pooled_redu_conv(pooled_x)
        batch_size = x.size(0)

        pooled_x = pooled_x.view(batch_size, self.channels,
                                 -1).permute(0, 2, 1).contiguous()

        affinity_matrix = self.gla(x + resize(
            self.global_info(F.adaptive_avg_pool2d(x, 1)), size=x.shape[2:])
                                   ).permute(0, 2, 3, 1).reshape(
                                       batch_size, -1, self.pool_scale**2)
        affinity_matrix = F.sigmoid(affinity_matrix)

        z_out = torch.matmul(affinity_matrix, pooled_x)

        z_out = z_out.permute(0, 2, 1).contiguous()

        z_out = z_out.view(batch_size, self.channels, x.size(2), x.size(3))
        z_out = self.residual_conv(z_out)
        z_out = F.relu(z_out + x)
        if self.fusion:
            z_out = self.fusion_conv(z_out)

        return z_out


@MODELS.register_module()
class APCHead(BaseDecodeHead):


    def __init__(self, pool_scales=(1, 2, 3, 6), fusion=True, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.fusion = fusion
        acm_modules = []
        for pool_scale in self.pool_scales:
            acm_modules.append(
                ACM(pool_scale,
                    self.fusion,
                    self.in_channels,
                    self.channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.acm_modules = nn.ModuleList(acm_modules)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):

        x = self._transform_inputs(inputs)
        acm_outs = [x]
        for acm_module in self.acm_modules:
            acm_outs.append(acm_module(x))
        acm_outs = torch.cat(acm_outs, dim=1)
        output = self.bottleneck(acm_outs)
        output = self.cls_seg(output)
        return output
