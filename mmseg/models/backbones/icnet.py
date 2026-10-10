import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..decode_heads.psp_head import PPM
from ..utils import resize


@MODELS.register_module()
class ICNet(BaseModule):


    def __init__(self,
                 backbone_cfg,
                 in_channels=3,
                 layer_channels=(512, 2048),
                 light_branch_middle_channels=32,
                 psp_out_channels=512,
                 out_channels=(64, 256, 256),
                 pool_scales=(1, 2, 3, 6),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 init_cfg=None):
        if backbone_cfg is None:
            raise TypeError('backbone_cfg must be passed from config file!')
        if init_cfg is None:
            init_cfg = [
                dict(type='Kaiming', mode='fan_out', layer='Conv2d'),
                dict(type='Constant', val=1, layer='_BatchNorm'),
                dict(type='Normal', mean=0.01, layer='Linear')
            ]
        super().__init__(init_cfg=init_cfg)
        self.align_corners = align_corners
        self.backbone = MODELS.build(backbone_cfg)


        self.backbone.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)

        self.psp_modules = PPM(
            pool_scales=pool_scales,
            in_channels=layer_channels[1],
            channels=psp_out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_corners=align_corners)

        self.psp_bottleneck = ConvModule(
            layer_channels[1] + len(pool_scales) * psp_out_channels,
            psp_out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv_sub1 = nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=light_branch_middle_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg),
            ConvModule(
                in_channels=light_branch_middle_channels,
                out_channels=light_branch_middle_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg),
            ConvModule(
                in_channels=light_branch_middle_channels,
                out_channels=out_channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))

        self.conv_sub2 = ConvModule(
            layer_channels[0],
            out_channels[1],
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        self.conv_sub4 = ConvModule(
            psp_out_channels,
            out_channels[2],
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        output = []


        output.append(self.conv_sub1(x))


        x = resize(
            x,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=self.align_corners)
        x = self.backbone.stem(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        output.append(self.conv_sub2(x))


        x = resize(
            x,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=self.align_corners)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        psp_outs = self.psp_modules(x) + [x]
        psp_outs = torch.cat(psp_outs, dim=1)
        x = self.psp_bottleneck(psp_outs)

        output.append(self.conv_sub4(x))

        return output
