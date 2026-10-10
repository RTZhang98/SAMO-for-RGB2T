import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.registry import MODELS
from ..utils import PatchEmbed, resize


class TransformerEncoderLayer(BaseModule):


    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super().__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), identity=x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class VisionTransformer(BaseModule):


    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 patch_pad='corner',
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_origin=False,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 patch_bias=False,
                 pre_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 frozen_exclude=['all'],
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained
        self.out_origin = out_origin
        self.frozen_exclude = frozen_exclude

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding=patch_pad,
            bias=patch_bias,
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )

        num_patches = (img_size[0] // patch_size) * \
            (img_size[1] // patch_size)

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.pre_norm = pre_norm

        if self.pre_norm:
            self.pre_ln_name, pre_ln = build_norm_layer(
                norm_cfg, embed_dims, postfix='_pre')
            self.add_module(self.pre_ln_name, pre_ln)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    batch_first=True))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self._freeze()

    @property
    def pre_ln(self):
        return getattr(self, self.pre_ln_name)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') in ['Pretrained', 'Pretrained_Part']:
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            if self.init_cfg.get('type') == 'Pretrained':
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

            elif self.init_cfg.get('type') == 'Pretrained_Part':
                state_dict = checkpoint.copy()
                para_prefix = 'image_encoder'
                prefix_len = len(para_prefix) + 1
                for k, v in checkpoint.items():
                    state_dict.pop(k)
                    if para_prefix in k:
                        state_dict[k[prefix_len:]] = v

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    print_log(msg=f'Resize the pos_embed shape from '
                              f'{state_dict["pos_embed"].shape} to '
                              f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            load_state_dict(self, state_dict, strict=False, logger=None)
        elif self.init_cfg is not None:
            super().init_weights()
        else:


            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def _freeze(self):
        if 'all' in self.frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in self.frozen_exclude]):
                param.requires_grad = False

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):


        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):


        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)


        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:

            x = x[:, 1:]

        if self.pre_norm:
            x = self.pre_ln(x)

        outs = []
        if self.out_origin:
            if self.with_cls_token:

                out = x[:, 1:]
            else:
                out = x
            B, _, C = out.shape
            out = out.reshape(B, hw_shape[0], hw_shape[1],
                              C).permute(0, 3, 1, 2).contiguous()
            if self.output_cls_token:
                out = [out, x[:, 0]]
            outs.append(out)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:

                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
