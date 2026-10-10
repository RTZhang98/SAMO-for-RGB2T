import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class JPU(BaseModule):


    def __init__(self,
                 in_channels=(512, 1024, 2048),
                 mid_channels=512,
                 start_level=0,
                 end_level=-1,
                 dilations=(1, 2, 4, 8),
                 align_corners=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, tuple)
        assert isinstance(dilations, tuple)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.start_level = start_level
        self.num_ins = len(in_channels)
        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)

        self.dilations = dilations
        self.align_corners = align_corners

        self.conv_layers = nn.ModuleList()
        self.dilation_layers = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            conv_layer = nn.Sequential(
                ConvModule(
                    self.in_channels[i],
                    self.mid_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.conv_layers.append(conv_layer)
        for i in range(len(dilations)):
            dilation_layer = nn.Sequential(
                DepthwiseSeparableConvModule(
                    in_channels=(self.backbone_end_level - self.start_level) *
                    self.mid_channels,
                    out_channels=self.mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilations[i],
                    dilation=dilations[i],
                    dw_norm_cfg=norm_cfg,
                    dw_act_cfg=None,
                    pw_norm_cfg=norm_cfg,
                    pw_act_cfg=act_cfg))
            self.dilation_layers.append(dilation_layer)

    def forward(self, inputs):

        assert len(inputs) == len(self.in_channels), 'Length of inputs must \
                                           be the same with self.in_channels!'

        feats = [
            self.conv_layers[i - self.start_level](inputs[i])
            for i in range(self.start_level, self.backbone_end_level)
        ]

        h, w = feats[0].shape[2:]
        for i in range(1, len(feats)):
            feats[i] = resize(
                feats[i],
                size=(h, w),
                mode='bilinear',
                align_corners=self.align_corners)

        feat = torch.cat(feats, dim=1)
        concat_feat = torch.cat([
            self.dilation_layers[i](feat) for i in range(len(self.dilations))
        ],
                                dim=1)

        outs = []


        for i in range(self.start_level, self.backbone_end_level - 1):
            outs.append(inputs[i])
        outs.append(concat_feat)
        return tuple(outs)
