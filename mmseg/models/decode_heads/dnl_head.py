import torch
from mmcv.cnn import NonLocal2d
from torch import nn

from mmseg.registry import MODELS
from .fcn_head import FCNHead


class DisentangledNonLocal2d(NonLocal2d):


    def __init__(self, *arg, temperature, **kwargs):
        super().__init__(*arg, **kwargs)
        self.temperature = temperature
        self.conv_mask = nn.Conv2d(self.in_channels, 1, kernel_size=1)

    def embedded_gaussian(self, theta_x, phi_x):


        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:

            pairwise_weight /= torch.tensor(
                theta_x.shape[-1],
                dtype=torch.float,
                device=pairwise_weight.device)**torch.tensor(
                    0.5, device=pairwise_weight.device)
        pairwise_weight /= torch.tensor(
            self.temperature, device=pairwise_weight.device)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def forward(self, x):

        n = x.size(0)


        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)


        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)


        theta_x -= theta_x.mean(dim=-2, keepdim=True)
        phi_x -= phi_x.mean(dim=-1, keepdim=True)

        pairwise_func = getattr(self, self.mode)

        pairwise_weight = pairwise_func(theta_x, phi_x)


        y = torch.matmul(pairwise_weight, g_x)

        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *x.size()[2:])


        unary_mask = self.conv_mask(x)
        unary_mask = unary_mask.view(n, 1, -1)
        unary_mask = unary_mask.softmax(dim=-1)

        unary_x = torch.matmul(unary_mask, g_x)

        unary_x = unary_x.permute(0, 2, 1).contiguous().reshape(
            n, self.inter_channels, 1, 1)

        output = x + self.conv_out(y + unary_x)

        return output


@MODELS.register_module()
class DNLHead(FCNHead):


    def __init__(self,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 temperature=0.05,
                 **kwargs):
        super().__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.temperature = temperature
        self.dnl_block = DisentangledNonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
            temperature=self.temperature)

    def forward(self, inputs):

        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.dnl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
