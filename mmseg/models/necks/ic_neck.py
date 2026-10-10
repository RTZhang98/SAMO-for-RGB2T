import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize


class CascadeFeatureFusion(BaseModule):


    def __init__(self,
                 low_channels,
                 high_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.align_corners = align_corners
        self.conv_low = ConvModule(
            low_channels,
            out_channels,
            3,
            padding=2,
            dilation=2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv_high = ConvModule(
            high_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x_low, x_high):
        x_low = resize(
            x_low,
            size=x_high.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners)


        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        return x, x_low


@MODELS.register_module()
class ICNeck(BaseModule):


    def __init__(self,
                 in_channels=(64, 256, 256),
                 out_channels=128,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        assert len(in_channels) == 3, 'Length of input channels \
                                        must be 3!'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.cff_24 = CascadeFeatureFusion(
            self.in_channels[2],
            self.in_channels[1],
            self.out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

        self.cff_12 = CascadeFeatureFusion(
            self.out_channels,
            self.in_channels[0],
            self.out_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

    def forward(self, inputs):
        assert len(inputs) == 3, 'Length of input feature \
                                        maps must be 3!'

        x_sub1, x_sub2, x_sub4 = inputs
        x_cff_24, x_24 = self.cff_24(x_sub4, x_sub2)
        x_cff_12, x_12 = self.cff_12(x_cff_24, x_sub1)


        return x_24, x_12, x_cff_12
