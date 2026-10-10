import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


class DCM(nn.Module):


    def __init__(self, filter_size, fusion, in_channels, channels, conv_cfg,
                 norm_cfg, act_cfg):
        super().__init__()
        self.filter_size = filter_size
        self.fusion = fusion
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.filter_gen_conv = nn.Conv2d(self.in_channels, self.channels, 1, 1,
                                         0)

        self.input_redu_conv = ConvModule(
            self.in_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.norm_cfg is not None:
            self.norm = build_norm_layer(self.norm_cfg, self.channels)[1]
        else:
            self.norm = None
        self.activate = build_activation_layer(self.act_cfg)

        if self.fusion:
            self.fusion_conv = ConvModule(
                self.channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, x):

        generated_filter = self.filter_gen_conv(
            F.adaptive_avg_pool2d(x, self.filter_size))
        x = self.input_redu_conv(x)
        b, c, h, w = x.shape

        x = x.view(1, b * c, h, w)

        generated_filter = generated_filter.view(b * c, 1, self.filter_size,
                                                 self.filter_size)
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = (pad, pad, pad, pad)
        else:
            p2d = (pad + 1, pad, pad + 1, pad)
        x = F.pad(input=x, pad=p2d, mode='constant', value=0)

        output = F.conv2d(input=x, weight=generated_filter, groups=b * c)

        output = output.view(b, c, h, w)
        if self.norm is not None:
            output = self.norm(output)
        output = self.activate(output)

        if self.fusion:
            output = self.fusion_conv(output)

        return output


@MODELS.register_module()
class DMHead(BaseDecodeHead):


    def __init__(self, filter_sizes=(1, 3, 5, 7), fusion=False, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(filter_sizes, (list, tuple))
        self.filter_sizes = filter_sizes
        self.fusion = fusion
        dcm_modules = []
        for filter_size in self.filter_sizes:
            dcm_modules.append(
                DCM(filter_size,
                    self.fusion,
                    self.in_channels,
                    self.channels,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.dcm_modules = nn.ModuleList(dcm_modules)
        self.bottleneck = ConvModule(
            self.in_channels + len(filter_sizes) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):

        x = self._transform_inputs(inputs)
        dcm_outs = [x]
        for dcm_module in self.dcm_modules:
            dcm_outs.append(dcm_module(x))
        dcm_outs = torch.cat(dcm_outs, dim=1)
        output = self.bottleneck(dcm_outs)
        output = self.cls_seg(output)
        return output
