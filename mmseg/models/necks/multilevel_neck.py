import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model.weight_init import xavier_init

from mmseg.registry import MODELS
from ..utils import resize


@MODELS.register_module()
class MultiLevelNeck(nn.Module):


    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=[0.5, 1, 2, 4],
                 norm_cfg=None,
                 act_cfg=None):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_outs = len(scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.lateral_convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        for _ in range(self.num_outs):
            self.convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(self.num_outs)]
        outs = []
        for i in range(self.num_outs):
            x_resize = resize(
                inputs[i], scale_factor=self.scales[i], mode='bilinear')
            outs.append(self.convs[i](x_resize))
        return tuple(outs)
