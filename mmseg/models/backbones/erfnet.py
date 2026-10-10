import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..utils import resize


class DownsamplerBlock(BaseModule):


    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-3),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.conv = build_conv_layer(
            self.conv_cfg,
            in_channels,
            out_channels - in_channels,
            kernel_size=3,
            stride=2,
            padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

    def forward(self, input):
        conv_out = self.conv(input)
        pool_out = self.pool(input)
        pool_out = resize(
            input=pool_out,
            size=conv_out.size()[2:],
            mode='bilinear',
            align_corners=False)
        output = torch.cat([conv_out, pool_out], 1)
        output = self.bn(output)
        output = self.act(output)
        return output


class NonBottleneck1d(BaseModule):


    def __init__(self,
                 channels,
                 drop_rate=0,
                 dilation=1,
                 num_conv_layer=2,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-3),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.act = build_activation_layer(self.act_cfg)

        self.convs_layers = nn.ModuleList()
        for conv_layer in range(num_conv_layer):
            first_conv_padding = (1, 0) if conv_layer == 0 else (dilation, 0)
            first_conv_dilation = 1 if conv_layer == 0 else (dilation, 1)
            second_conv_padding = (0, 1) if conv_layer == 0 else (0, dilation)
            second_conv_dilation = 1 if conv_layer == 0 else (1, dilation)

            self.convs_layers.append(
                build_conv_layer(
                    self.conv_cfg,
                    channels,
                    channels,
                    kernel_size=(3, 1),
                    stride=1,
                    padding=first_conv_padding,
                    bias=True,
                    dilation=first_conv_dilation))
            self.convs_layers.append(self.act)
            self.convs_layers.append(
                build_conv_layer(
                    self.conv_cfg,
                    channels,
                    channels,
                    kernel_size=(1, 3),
                    stride=1,
                    padding=second_conv_padding,
                    bias=True,
                    dilation=second_conv_dilation))
            self.convs_layers.append(
                build_norm_layer(self.norm_cfg, channels)[1])
            if conv_layer == 0:
                self.convs_layers.append(self.act)
            else:
                self.convs_layers.append(nn.Dropout(p=drop_rate))

    def forward(self, input):
        output = input
        for conv in self.convs_layers:
            output = conv(output)
        output = self.act(output + input)
        return output


class UpsamplerBlock(BaseModule):


    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=1e-3),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


@MODELS.register_module()
class ERFNet(BaseModule):


    def __init__(self,
                 in_channels=3,
                 enc_downsample_channels=(16, 64, 128),
                 enc_stage_non_bottlenecks=(5, 8),
                 enc_non_bottleneck_dilations=(2, 4, 8, 16),
                 enc_non_bottleneck_channels=(64, 128),
                 dec_upsample_channels=(64, 16),
                 dec_stages_non_bottleneck=(2, 2),
                 dec_non_bottleneck_channels=(64, 16),
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        assert len(enc_downsample_channels) \
               == len(dec_upsample_channels)+1, 'Number of downsample\
                     block of encoder does not \
                    match number of upsample block of decoder!'
        assert len(enc_downsample_channels) \
               == len(enc_stage_non_bottlenecks)+1, 'Number of \
                    downsample block of encoder does not match \
                    number of Non-bottleneck block of encoder!'
        assert len(enc_downsample_channels) \
               == len(enc_non_bottleneck_channels)+1, 'Number of \
                    downsample block of encoder does not match \
                    number of channels of Non-bottleneck block of encoder!'
        assert enc_stage_non_bottlenecks[-1] \
               % len(enc_non_bottleneck_dilations) == 0, 'Number of \
                    Non-bottleneck block of encoder does not match \
                    number of Non-bottleneck block of encoder!'
        assert len(dec_upsample_channels) \
               == len(dec_stages_non_bottleneck), 'Number of \
                upsample block of decoder does not match \
                number of Non-bottleneck block of decoder!'
        assert len(dec_stages_non_bottleneck) \
               == len(dec_non_bottleneck_channels), 'Number of \
                Non-bottleneck block of decoder does not match \
                number of channels of Non-bottleneck block of decoder!'

        self.in_channels = in_channels
        self.enc_downsample_channels = enc_downsample_channels
        self.enc_stage_non_bottlenecks = enc_stage_non_bottlenecks
        self.enc_non_bottleneck_dilations = enc_non_bottleneck_dilations
        self.enc_non_bottleneck_channels = enc_non_bottleneck_channels
        self.dec_upsample_channels = dec_upsample_channels
        self.dec_stages_non_bottleneck = dec_stages_non_bottleneck
        self.dec_non_bottleneck_channels = dec_non_bottleneck_channels
        self.dropout_ratio = dropout_ratio

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.encoder.append(
            DownsamplerBlock(self.in_channels, enc_downsample_channels[0]))

        for i in range(len(enc_downsample_channels) - 1):
            self.encoder.append(
                DownsamplerBlock(enc_downsample_channels[i],
                                 enc_downsample_channels[i + 1]))

            if i == len(enc_downsample_channels) - 2:
                iteration_times = int(enc_stage_non_bottlenecks[-1] /
                                      len(enc_non_bottleneck_dilations))
                for j in range(iteration_times):
                    for k in range(len(enc_non_bottleneck_dilations)):
                        self.encoder.append(
                            NonBottleneck1d(enc_downsample_channels[-1],
                                            self.dropout_ratio,
                                            enc_non_bottleneck_dilations[k]))
            else:
                for j in range(enc_stage_non_bottlenecks[i]):
                    self.encoder.append(
                        NonBottleneck1d(enc_downsample_channels[i + 1],
                                        self.dropout_ratio))

        for i in range(len(dec_upsample_channels)):
            if i == 0:
                self.decoder.append(
                    UpsamplerBlock(enc_downsample_channels[-1],
                                   dec_non_bottleneck_channels[i]))
            else:
                self.decoder.append(
                    UpsamplerBlock(dec_non_bottleneck_channels[i - 1],
                                   dec_non_bottleneck_channels[i]))
            for j in range(dec_stages_non_bottleneck[i]):
                self.decoder.append(
                    NonBottleneck1d(dec_non_bottleneck_channels[i]))

    def forward(self, x):
        for enc in self.encoder:
            x = enc(x)
        for dec in self.decoder:
            x = dec(x)
        return [x]
