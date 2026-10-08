from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.utils import SampleList

class _SAMOPlusMask2FormerBase(Mask2FormerHead):
    def __init__(
        self,
        replace_query_feat=False,
        semantic_loss_weight=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        feat_channels = kwargs["feat_channels"]
        del self.query_embed
        self.vpt_transforms = nn.ModuleList()
        self.replace_query_feat = replace_query_feat
        self.semantic_loss_weight = float(semantic_loss_weight)
        if self.semantic_loss_weight < 0:
            raise ValueError("semantic_loss_weight must be non-negative.")
        if replace_query_feat:
            del self.query_feat
            self.querys2feat = nn.Linear(feat_channels, feat_channels)

    def forward(
        self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList, mod="test"
    ) -> Tuple[List[Tensor]]:
        x, mid_output = x
        x, query_embed = x
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)
        if query_embed.ndim == 2:
            query_embed = query_embed.expand(batch_size, -1, -1)

        mask_features, multi_scale_memorys = self.pixel_decoder(x)

        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])

            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level

            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,

                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        if mod=='test':
            return cls_pred_list, mask_pred_list
        elif mod=='train':
            return cls_pred_list, mask_pred_list, mid_output

    def bound_loss(
        self,
        sem_rgb: torch.Tensor,
        dom_rgb: torch.Tensor,
        sem_ir: torch.Tensor,
        dom_ir: torch.Tensor,
        adapted: torch.Tensor,
        nu: float = 1.0,
        delta: float = 1.0,
        tau_sem: float = 0.1,
        tau_div: float = 0.1,
        eps: float = 1e-8
    ) -> torch.Tensor:

        def _flatten(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0], -1)

        def _cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a = F.normalize(a, dim=1, p=2, eps=eps)
            b = F.normalize(b, dim=1, p=2, eps=eps)
            return a @ b.t()

        s_rgb = _flatten(sem_rgb)
        s_th  = _flatten(sem_ir)
        d_th  = _flatten(dom_ir)
        s_pos = _flatten(adapted)

        B = s_rgb.shape[0]
        device = s_rgb.device
        dtype  = s_rgb.dtype

        if B < 2:

            L_sem = torch.zeros([], device=device, dtype=dtype)
            L_div = torch.zeros([], device=device, dtype=dtype)
            L_cross = torch.zeros([], device=device, dtype=dtype)
            return L_sem + L_div + L_cross

        neg_mask = torch.eye(B, device=device, dtype=torch.bool)

        if self.semantic_loss_weight == 0.0:
            L_sem = s_rgb.sum() * 0.0
        else:

            score_pos = (F.cosine_similarity(
                F.normalize(s_rgb, dim=1, eps=eps),
                F.normalize(s_pos, dim=1, eps=eps),
                dim=1
            ) / max(tau_sem, eps))

            score_neg_mat = _cosine_sim_matrix(s_rgb, s_th) / max(tau_sem, eps)
            score_neg_mat = score_neg_mat.masked_fill(neg_mask, float("-inf"))

            denom_sem = torch.logsumexp(
                torch.cat([score_pos.unsqueeze(1), score_neg_mat], dim=1),
                dim=1
            )
            L_sem = (
                (-(score_pos - denom_sem)).mean()
                * self.semantic_loss_weight
            )

        alpha = torch.full((B,), float(delta), device=device, dtype=dtype)
        pi_mat = torch.distributions.Dirichlet(alpha).sample((B,))
        dT_pos = pi_mat @ d_th

        if not hasattr(self, "cond_proj"):

            self.cond_proj = nn.LazyLinear(256).to(device=device)

        def phi(d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            x = torch.cat([d, s], dim=1)
            z = self.cond_proj(x)
            return F.normalize(z, dim=1, p=2, eps=eps)

        z_anchor = phi(d_th, s_rgb)
        z_pos    = phi(dT_pos, s_rgb)
        score_pos_div = (z_anchor * z_pos).sum(dim=1) / max(tau_div, eps)

        d_candidates = d_th.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
        s_conditions = s_rgb.unsqueeze(1).expand(B, B, -1).reshape(B * B, -1)
        z_candidates = phi(d_candidates, s_conditions).reshape(B, B, -1)

        z_anchor_exp = z_anchor.unsqueeze(1).expand(B, B, -1)
        score_neg_div = (z_anchor_exp * z_candidates).sum(dim=2) / max(tau_div, eps)

        score_neg_div = score_neg_div.masked_fill(neg_mask, float("-inf"))

        denom_div = torch.logsumexp(
            torch.cat([score_pos_div.unsqueeze(1), score_neg_div], dim=1),
            dim=1
        )
        L_div = (-(score_pos_div - denom_div)).mean()

        B = sem_rgb.shape[0]

        x_hsic = sem_rgb.view(B, 1, -1)
        y_hsic = dom_ir.view(B, 1, -1)

        L_cross = nu * self.hsic_loss(x_hsic, y_hsic)

        L_MI = L_sem + L_div + L_cross
        return L_MI

    def rbf_kernel(self, a: torch.Tensor, b: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:

        a_norm = torch.sum(a ** 2, dim=1).view(-1, 1)
        b_norm = torch.sum(b ** 2, dim=1).view(1, -1)
        dist_sq = a_norm + b_norm - 2 * torch.mm(a, b.t())
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    def hsic_loss(self, x: torch.Tensor, y: torch.Tensor, sigma_x: float = 1.0, sigma_y: float = 1.0) -> torch.Tensor:
        assert x.shape == y.shape, "x and y must have the same shape"
        B, C, L = x.shape
        if B < 2:
            return torch.tensor(0.0, device=x.device)

        x_flat = x.view(B, -1)
        y_flat = y.view(B, -1)

        K_x = self.rbf_kernel(x_flat, x_flat, sigma_x)
        K_y = self.rbf_kernel(y_flat, y_flat, sigma_y)

        H = torch.eye(B, device=x.device) - (1.0 / B) * torch.ones((B, B), device=x.device)

        KH_x = torch.mm(K_x, H)
        KH_y = torch.mm(K_y, H)
        hsic = torch.trace(torch.mm(KH_x, KH_y.t())) / ((B - 1) ** 2)
        
        return hsic
