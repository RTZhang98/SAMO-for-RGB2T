from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.utils import SampleList
from torch import Tensor
from typing import List, Tuple
import torch
import torch.nn as nn
from mmseg.models.builder import MODELS
from mmseg.utils import ConfigType, SampleList
from typing import Tuple, List, Dict
import torch.nn.functional as F


@MODELS.register_module()
class SpectrumMask2FormerHead(Mask2FormerHead):
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
        # use vpt_querys to replace query_embed
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        # query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
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

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
            train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        all_cls_scores, all_mask_preds, mid_output = self(x, batch_data_samples, mod="train")

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                batch_gt_instances, batch_img_metas)

        # ========== 新增：中间特征的解耦损失 ==========
        mid_losses = self._compute_mid_losses(mid_output)
        
        # 合并所有损失
        losses.update(mid_losses)

        return losses

    def _compute_mid_losses(self, mid_output: List[Dict[str, Tensor]]) -> dict:
        """
        计算中间特征的解耦损失
        
        Args:
            mid_output: List of dict, 每个dict包含:
                - 'sem_rgb': [B, C, L] - RGB语义特征
                - 'dom_rgb': [B, C, L] - RGB域特征
                - 'sem_ir': [B, C, L] - IR语义特征
                - 'dom_ir': [B, C, L] - IR域特征
                - 'fused_rgb': [B, C, L] - RGB重构特征
                - 'fused_ir': [B, C, L] - IR重构特征
                - 'adapted': [B, C, L] - 融合特征 (sem_rgb + dom_ir)
                - 'ori_rgb': [B, C, L] - 原始RGB特征
                - 'ori_ir': [B, C, L] - 原始IR特征
            train_cfg: 训练配置
        
        Returns:
            dict: 包含各种中间损失
        """
        # 获取损失权重（从配置中读取，或使用默认值）
        weight_mi_disentangle = 1.0
        weight_recon = 2
        weight_mi_bound = 0.5
        
        total_losses = {
            'loss_mi_disentangle': 0.0,  # 解耦的互信息最小化
            'loss_recon': 0.0,        # RGB重构损失
            'loss_mi_bound': 0.0,      # 语义互信息最大化
        }
        
        num_layers = len(mid_output)
        
        for layer_idx, layer_features in enumerate(mid_output):
            # 提取特征
            sem_rgb = layer_features['sem_rgb']      # [B, C, L]
            dom_rgb = layer_features['dom_rgb']      # [B, C, L]
            sem_ir = layer_features['sem_ir']        # [B, C, L]
            dom_ir = layer_features['dom_ir']        # [B, C, L]
            fused_rgb = layer_features['fused_rgb']  # [B, C, L]
            fused_ir = layer_features['fused_ir']    # [B, C, L]
            adapted = layer_features['adapted_low']      # [B, C, L]
            ori_rgb = layer_features['ori_rgb']      # [B, C, L]
            ori_ir = layer_features['ori_ir']        # [B, C, L]
            
            # ========== ① 解耦互信息最小化 ==========
            # RGB: semantic和domain应该独立
            mi_rgb = self.hsic_loss(sem_rgb, dom_rgb)
            # IR: semantic和domain应该独立
            mi_ir = self.hsic_loss(sem_ir, dom_ir)
            
            loss_mi_disentangle = (mi_rgb + mi_ir) / 2.0
            total_losses['loss_mi_disentangle'] += loss_mi_disentangle
            
            # ========== ② 重构损失 ==========
            # RGB重构应该接近原始RGB
            loss_recon_rgb = F.mse_loss(fused_rgb, ori_rgb)
            loss_recon_ir = F.mse_loss(fused_ir, ori_ir)
            total_losses['loss_recon'] += (loss_recon_ir + loss_recon_rgb)/2
            
            # ========== ③ 互信息下界最大化 ==========
            loss_mi_bound = self.bound_loss(sem_rgb, dom_rgb, sem_ir, dom_ir, adapted)
            total_losses['loss_mi_bound'] += loss_mi_bound
        
        # 平均所有层的损失
        for key in total_losses:
            total_losses[key] = total_losses[key] / num_layers
        
        # 加权损失
        weighted_losses = {
            'loss_mi_disentangle': total_losses['loss_mi_disentangle'] * weight_mi_disentangle,
            'loss_recon': total_losses['loss_recon'] * weight_recon,
            'loss_mi_bound': total_losses['loss_mi_bound'] * weight_mi_bound,
        }
        
        return weighted_losses

    def bound_loss(
        self,
        sem_rgb: torch.Tensor,   # s_rgb
        dom_rgb: torch.Tensor,   # unused (kept for interface compatibility)
        sem_ir: torch.Tensor,    # s_th  (thermal semantic negatives for L_sem)
        dom_ir: torch.Tensor,    # d_th  (reference thermal variation codes for L_div)
        adapted: torch.Tensor,   # s_T^+ proxy (from f_mod^{low})
        nu: float = 1.0,         # weighting for HSIC (L_cross)
        delta: float = 1.0,      # Dirichlet concentration δ for convex mix (Lambda)
        tau_sem: float = 0.1,    # temperature for L_sem
        tau_div: float = 0.1,    # temperature for L_div
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Paper-aligned MI-driven losses (final version in your method section):

        (i)  L_sem   : Semantic Sufficiency Loss (InfoNCE)
            pos = adapted  (s_T^{+})
            neg = sem_ir   (s_th) with i != j

        (ii) L_div   : Thermal-Variation Diversity Loss (conditional InfoNCE)
            pos = d_T^{+j} ~ Lambda, where Lambda is a Dirichlet convex combination of batch d_th
            neg = in-batch {d_th^i}_{i!=j}

            conditional similarity: C(d_a, d_b; s) = cos(phi(d_a;s), phi(d_b;s))

        (iii)L_cross : Cross-Component Decoupling Loss
            L_cross = nu * HSIC(s_rgb, d_th)

        Total: L_MI = L_sem + L_div + L_cross
        """

        # -------------------------
        # Helpers
        # -------------------------
        def _flatten(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0], -1)

        def _cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Cosine similarity matrix between two batches: (B,D) vs (B,D) -> (B,B)."""
            a = F.normalize(a, dim=1, p=2, eps=eps)
            b = F.normalize(b, dim=1, p=2, eps=eps)
            return a @ b.t()

        # -------------------------
        # Prepare flattened features
        # -------------------------
        s_rgb = _flatten(sem_rgb)      # (B, Ds)
        s_th  = _flatten(sem_ir)       # (B, Ds)
        d_th  = _flatten(dom_ir)       # (B, Dd)
        s_pos = _flatten(adapted)      # (B, Ds)  == s_T^{+}

        B = s_rgb.shape[0]
        device = s_rgb.device
        dtype  = s_rgb.dtype

        if B < 2:
            # Degenerate batch: return stable scalar loss
            # (HSIC etc. may be undefined for B=1)
            L_sem = torch.zeros([], device=device, dtype=dtype)
            L_div = torch.zeros([], device=device, dtype=dtype)
            L_cross = torch.zeros([], device=device, dtype=dtype)
            return L_sem + L_div + L_cross

        # =========================================================
        # (i) Semantic Sufficiency Loss  L_sem  (InfoNCE)
        # =========================================================
        neg_mask = torch.eye(B, device=device, dtype=torch.bool)
        # A zero weight is the controlled "w/o semantic term"
        # ablation: skip the term entirely so it contributes no supervision.
        if self.semantic_loss_weight == 0.0:
            L_sem = s_rgb.sum() * 0.0
        else:
            # score_pos[j] = cos(s_rgb^j, s_T^{+j}) / tau
            score_pos = (F.cosine_similarity(
                F.normalize(s_rgb, dim=1, eps=eps),
                F.normalize(s_pos, dim=1, eps=eps),
                dim=1
            ) / max(tau_sem, eps))  # (B,)

            # score_neg[j, i] = cos(s_rgb^j, s_th^i) / tau, with i != j
            score_neg_mat = _cosine_sim_matrix(s_rgb, s_th) / max(tau_sem, eps)  # (B,B)
            score_neg_mat = score_neg_mat.masked_fill(neg_mask, float("-inf"))

            # InfoNCE: -log exp(pos) / (exp(pos) + sum exp(neg))
            denom_sem = torch.logsumexp(
                torch.cat([score_pos.unsqueeze(1), score_neg_mat], dim=1),
                dim=1
            )  # (B,)
            L_sem = (
                (-(score_pos - denom_sem)).mean()
                * self.semantic_loss_weight
            )

        # =========================================================
        # (ii) Thermal-Variation Diversity Loss  L_div  (conditional InfoNCE)
        #      pos d_T^{+j} ~ Lambda (Dirichlet convex mix of batch d_th)
        #      neg = in-batch {d_th^i}_{i != j}
        # =========================================================

        # ---- Build d_T^{+j} for each anchor j via Dirichlet mixing ----
        # pi_mat: (B, B), each row is a simplex weight vector for one anchor j
        alpha = torch.full((B,), float(delta), device=device, dtype=dtype)
        pi_mat = torch.distributions.Dirichlet(alpha).sample((B,))  # (B,B)
        dT_pos = pi_mat @ d_th  # (B, Dd)  == d_T^{+j}

        # ---- Conditional cosine operator C(d_a, d_b; s) via lightweight conditional projection ----
        # We implement phi(d; s) = Norm( W [d; s] )
        # To keep this function self-contained, we lazily create a projector if missing.
        if not hasattr(self, "cond_proj"):
            # output dim can be set small; 256 works well; you can change if needed.
            self.cond_proj = nn.LazyLinear(256).to(device=device)

        def phi(d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            x = torch.cat([d, s], dim=1)               # (N, Dd+Ds)
            z = self.cond_proj(x)                      # (N, Dp)
            return F.normalize(z, dim=1, p=2, eps=eps) # (N, Dp)

        # ---- Compute C for positives: C(d_th^j, d_T^{+j}; s_rgb^j) ----
        z_anchor = phi(d_th, s_rgb)         # (B, Dp)
        z_pos    = phi(dT_pos, s_rgb)       # (B, Dp)
        score_pos_div = (z_anchor * z_pos).sum(dim=1) / max(tau_div, eps)  # (B,)

        # ---- Compute C for in-batch negatives: C(d_th^j, d_th^i; s_rgb^j) ----
        # For each anchor j, candidates i use the SAME condition s_rgb^j.
        # Vectorized construction:
        #   candidates: d_th repeated for each anchor j -> (B*B, Dd)
        #   conditions: s_rgb[j] repeated B times      -> (B*B, Ds)
        d_candidates = d_th.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)     # (B*B, Dd)
        s_conditions = s_rgb.unsqueeze(1).expand(B, B, -1).reshape(B * B, -1)   # (B*B, Ds)
        z_candidates = phi(d_candidates, s_conditions).reshape(B, B, -1)        # (B, B, Dp)

        # anchor embeddings expanded: (B,1,Dp) -> (B,B,Dp)
        z_anchor_exp = z_anchor.unsqueeze(1).expand(B, B, -1)
        score_neg_div = (z_anchor_exp * z_candidates).sum(dim=2) / max(tau_div, eps)  # (B,B)

        # exclude i == j
        score_neg_div = score_neg_div.masked_fill(neg_mask, float("-inf"))

        # InfoNCE for diversity
        denom_div = torch.logsumexp(
            torch.cat([score_pos_div.unsqueeze(1), score_neg_div], dim=1),
            dim=1
        )  # (B,)
        L_div = (-(score_pos_div - denom_div)).mean()

        # =========================================================
        # (iii) Cross-Component Decoupling Loss  L_cross = nu * HSIC(s_rgb, d_th)
        # HSIC expects x,y in shape (B, C, L) and requires x.shape == y.shape
        # =========================================================
        B = sem_rgb.shape[0]

        x_hsic = sem_rgb.view(B, 1, -1)   # (B,1,D)
        y_hsic = dom_ir.view(B, 1, -1)    # (B,1,D)  (use dom_ir directly, NOT flattened d_th)

        L_cross = nu * self.hsic_loss(x_hsic, y_hsic)

        # -------------------------
        # Total MI-driven loss
        # -------------------------
        L_MI = L_sem + L_div + L_cross
        return L_MI

    # def bound_loss(
    #     self,
    #     sem_rgb: torch.Tensor,   # s_rgb, shape: (B, D) or (B, ...)
    #     dom_rgb: torch.Tensor,   # not used in Eq.(12)(13)(14), keep for interface
    #     sem_ir: torch.Tensor,    # s_th,  shape: (B, D) or (B, ...)
    #     dom_ir: torch.Tensor,    # d_th,  shape: (B, C, L) or (B, ...)
    #     adapted: torch.Tensor,   # s_T^+ proxy, same batch as sem_rgb
    #     mu: float = 0.5,         # Eq.(12)
    #     nu: float = 1.0,         # Eq.(14)
    #     delta: float = 1.0,      # Dirichlet concentration δ in Eq.(11)
    #     tau_sem: float = 0.1,    # temperature for Eq.(13) (optional but common)
    #     tau_dom: float = 0.1,    # temperature for S in Eq.(12)
    #     eps: float = 1e-8
    # ) -> torch.Tensor:
    #     """
    #     Paper-aligned losses:
    #     (i)  L_dom   : Eq.(11)(12)  Dirichlet convex combination proxy d_T^+
    #     (ii) L_sem   : Eq.(13)      InfoNCE, positives=adapted, negatives=sem_ir (j!=i)
    #     (iii)L_cross : Eq.(14)      HSIC(s_rgb, d_th)
    #     Total: L_bound = L_dom + L_sem + L_cross (Eq.(15))
    #     """

    #     def _flatten(x: torch.Tensor) -> torch.Tensor:
    #         return x.reshape(x.shape[0], -1)

    #     def cosine_sim(a, b):
    #         a = F.normalize(_flatten(a), dim=1, p=2, eps=eps)
    #         b = F.normalize(_flatten(b), dim=1, p=2, eps=eps)
    #         return (a * b).sum(dim=1)

    #     def cosine_sim_matrix(a, b):
    #         a = F.normalize(_flatten(a), dim=1, p=2, eps=eps)
    #         b = F.normalize(_flatten(b), dim=1, p=2, eps=eps)
    #         return a @ b.t()

    #     # --------------------------
    #     # (i) Thermal Variation Alignment Loss (Eq.(11)(12))
    #     # --------------------------
    #     B = dom_ir.shape[0]
    #     if B < 2:
    #         # still define something stable
    #         L_dom = torch.zeros([], device=dom_ir.device, dtype=dom_ir.dtype)
    #     else:
    #         dth = _flatten(dom_ir)  # (B, Dd)
    #         # sample π ~ Dir(δ) and form d_T^+ = Σ_j π_j d_th^j
    #         alpha = torch.full((B,), float(delta), device=dom_ir.device, dtype=dom_ir.dtype)
    #         pi = torch.distributions.Dirichlet(alpha).sample()  # (B,)
    #         dT_pos = (pi.view(B, 1) * dth).sum(dim=0, keepdim=True)   # (1, D)

    #         # similarity S = exp(cos/tau_dom) -> -log S = -(cos/tau_dom)
    #         dT_pos_rep = dT_pos.repeat(B, 1)  # (B, Dd)
    #         cos = cosine_sim(dth, dT_pos_rep)  # (B,)
    #         S = (cos + 1.0) * 0.5              # -> [0,1]
    #         S = S.clamp(min=eps, max=1.0)      # avoid log(0)
    #         neg_log_S = -torch.log(S)          # >= 0
    #         # neg_log_S = -(cos / max(tau_dom, eps))  # (B,)

    #         l2 = (dth - dT_pos_rep).pow(2).mean(dim=1)  # ||·||_2^2  (B,)
    #         L_dom = (neg_log_S + mu * l2).mean()

    #     # --------------------------
    #     # (ii) Semantic Sufficiency Loss (Eq.(13))
    #     # positives: adapted as s_T^{+i}; negatives: sem_ir[j] for j!=i
    #     # --------------------------
    #     # sim_pos: (B,)
    #     s_pos = cosine_sim(sem_rgb, adapted) / max(tau_sem, eps)

    #     # sim_neg: (B,B) between sem_rgb[i] and sem_ir[j]
    #     sim_neg_mat = cosine_sim_matrix(sem_rgb, sem_ir) / max(tau_sem, eps)  # (B,B)

    #     # exclude j == i
    #     mask = torch.eye(B, device=sim_neg_mat.device, dtype=torch.bool)
    #     sim_neg_mat = sim_neg_mat.masked_fill(mask, float('-inf'))

    #     # Eq.(13): -log exp(pos) / (exp(pos) + Σ_{j!=i} exp(neg_ij))
    #     # use logsumexp for stability
    #     denom = torch.logsumexp(
    #         torch.cat([s_pos.unsqueeze(1), sim_neg_mat], dim=1), dim=1
    #     )  # (B,)
    #     L_sem = (-(s_pos - denom)).mean()

    #     # --------------------------
    #     # (iii) Cross-Component Decoupling Loss (Eq.(14))
    #     # --------------------------
    #     # HSIC expects (B, D) usually; flatten dom_ir
    #     L_cross = nu * self.hsic_loss(sem_rgb, dom_ir)

    #     # --------------------------
    #     # Total (Eq.(15))
    #     # --------------------------
    #     L_bound = L_dom + L_sem - L_cross
    #     return L_bound
    
    # def rbf_kernel(self, X: torch.Tensor, sigma: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    #     """
    #     RBF kernel Gram matrix K where K_ij = exp(-||Xi - Xj||^2 / (2*sigma^2)).
    #     X: (B, D)
    #     Returns: (B, B)
    #     """
    #     B = X.shape[0]
    #     # Pairwise squared distances: (B,B)
    #     dist2 = torch.cdist(X, X, p=2).pow(2)

    #     denom = 2.0 * (sigma ** 2) + eps
    #     K = torch.exp(-dist2 / denom)
    #     return K

    # def hsic_loss(
    #     self,
    #     x: torch.Tensor,
    #     y: torch.Tensor,
    #     sigma_x: float = 1.0,
    #     sigma_y: float = 1.0,
    #     eps: float = 1e-8,
    # ) -> torch.Tensor:
    #     """
    #     HSIC dependency loss to decouple x and y (minimize it).

    #     Supports any shape as long as batch is the first dim:
    #     x: (B, ...)
    #     y: (B, ...)
    #     Internally flattens to (B, D).

    #     HSIC = (1/(B-1)^2) * trace(Kx H Ky H)
    #     """
    #     assert x.shape[0] == y.shape[0], "x and y must have the same batch size"
    #     B = x.shape[0]
    #     if B < 2:
    #         return torch.zeros([], device=x.device, dtype=x.dtype)

    #     x_flat = x.reshape(B, -1)
    #     y_flat = y.reshape(B, -1)

    #     Kx = self.rbf_kernel(x_flat, sigma=sigma_x, eps=eps)  # (B,B)
    #     Ky = self.rbf_kernel(y_flat, sigma=sigma_y, eps=eps)  # (B,B)

    #     # Centering matrix H = I - (1/B)11^T
    #     H = torch.eye(B, device=x.device, dtype=x.dtype) - (1.0 / B) * torch.ones((B, B), device=x.device, dtype=x.dtype)

    #     # HSIC = tr(Kx H Ky H) / (B-1)^2
    #     KxH = Kx @ H
    #     KyH = Ky @ H
    #     hsic = torch.trace(KxH @ KyH.t()) / ((B - 1) ** 2 + eps)

    #     return hsic


    # def bound_loss(self, sem_rgb: torch.Tensor, dom_rgb: torch.Tensor, sem_ir: torch.Tensor, dom_ir: torch.Tensor, adapted: torch.Tensor) -> torch.Tensor:

    #     def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    #         """
    #         计算批次内两个张量的余弦相似度。
    #         Args:
    #             a, b: (B, D) 张量
    #         Returns:
    #             (B,) 张量，每个元素的相似度
    #         """
    #         a_flat = a.reshape(a.shape[0], -1)
    #         b_flat = b.reshape(b.shape[0], -1)
    #         a_norm = a_flat / (a_flat.norm(dim=1, keepdim=True) + 1e-8)
    #         b_norm = b_flat / (b_flat.norm(dim=1, keepdim=True) + 1e-8)
    #         return (a_norm * b_norm).sum(dim=1)  # (B,)

    #     def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    #         """
    #         计算批次内L2距离。
    #         Args:
    #             a, b: (B, C, L)
    #         Returns:
    #             (B,) 张量
    #         """
    #         a_flat = a.reshape(a.shape[0], -1)
    #         b_flat = b.reshape(b.shape[0], -1)
    #         return (a_flat - b_flat).norm(dim=1)  # (B,)

    #     def l_sem(sem_rgb: torch.Tensor, adapted: torch.Tensor, sem_ir: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    #         """
    #         语义匹配下界 \mathcal{L}_{sem} (InfoNCE-style)
    #         - 正样本: adapted (假设为 s_T^+)
    #         - 负样本: sem_ir (IR语义不一致) + 批次内其他
    #         """
    #         B = sem_rgb.shape[0]
            
    #         # 假设批次内所有为正/负混合；简单实现：正=adapted，负=sem_ir + shuffle
    #         sim_pos = cosine_similarity(sem_rgb, adapted)  # (B,)
    #         sim_neg1 = cosine_similarity(sem_rgb, sem_ir)  # (B,)
            
    #         # 额外负样本：shuffle sem_rgb 作为负
    #         shuffled = sem_rgb[torch.randperm(B)]
    #         sim_neg2 = cosine_similarity(sem_rgb, shuffled)  # (B,)
            
    #         # InfoNCE: 对于每个样本，numerator=exp(sim_pos/tau), denominator=sum exp(sim_neg/tau) + pos
    #         logits = torch.stack([sim_pos, sim_neg1, sim_neg2], dim=1) / tau  # (B, 3)
    #         labels = torch.zeros(B, dtype=torch.long, device=sem_rgb.device)  # 0 是正样本
    #         return F.cross_entropy(logits, labels)  # 最大化下界等价于最小化这个CE（但在总损失中用负号）

    #     def l_dom(dom_ir: torch.Tensor, mu: float = 0.5, num_combos: int = 5) -> torch.Tensor:
    #         """
    #         域覆盖下界 \mathcal{L}_{dom} - 更新版，使用批次内 dom_ir 的凸组合作为 d_T^+ 代理
    #         - 对于每个 dom_ir[i]，生成 num_combos 个凸组合（λ * dom_ir[i] + (1-λ) * dom_ir[j], j≠i）
    #         - 计算平均 sim 和 dist，然后 batch 平均
    #         - 线性形式：sim - mu * dist (最大化这个，鼓励覆盖凸组合风格)
    #         """
    #         B, C, L = dom_ir.shape
    #         if B < 2:
    #             return torch.tensor(0.0, device=dom_ir.device)  # 需要至少2个样本生成组合
            
    #         sim_total = 0.0
    #         dist_total = 0.0
            
    #         for i in range(B):
    #             # 随机选择 num_combos 个 j ≠ i
    #             other_indices = torch.tensor([j for j in range(B) if j != i], device=dom_ir.device)
    #             selected_js = other_indices[torch.randperm(len(other_indices))[:num_combos]]  # 随机选 num_combos 个
                
    #             # 生成凸组合
    #             lambdas = torch.rand(num_combos, device=dom_ir.device)  # λ ~ Uniform(0,1)
    #             combos = lambdas.view(-1, 1, 1, 1) * dom_ir[i].unsqueeze(0) + \
    #                     (1 - lambdas.view(-1, 1, 1, 1)) * dom_ir[selected_js]  # (num_combos, C, L)
                
    #             # 计算与 combos 的平均 sim 和 dist
    #             sim_i = cosine_similarity(dom_ir[i].unsqueeze(0).repeat(num_combos, 1, 1, 1), combos).mean()
    #             dist_i = l2_distance(dom_ir[i].unsqueeze(0).repeat(num_combos, 1, 1, 1), combos).mean()
                
    #             sim_total += sim_i
    #             dist_total += dist_i
            
    #         # batch 平均
    #         avg_sim = sim_total / B
    #         avg_dist = dist_total / B
            
    #         return avg_sim - mu * avg_dist  # 最大化这个下界

    #     def l_cross(sem_rgb: torch.Tensor, dom_ir: torch.Tensor, nu: float = 1.0) -> torch.Tensor:
    #         """
    #         交叉项惩罚 \mathcal{L}_{cross} (HSIC版本)
    #         - 最小化 sem_rgb 和 dom_ir 的依赖
    #         """
    #         return nu * self.hsic_loss(sem_rgb, dom_ir)  # 最小化这个

    #     loss_sem = l_sem(sem_rgb, adapted, sem_ir)
    #     loss_dom = l_dom(dom_ir)
    #     loss_cross = l_cross(sem_rgb, dom_ir)
    #     loss_bound = loss_sem + loss_dom - loss_cross
    #     return -loss_bound

    def rbf_kernel(self, a: torch.Tensor, b: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """
        计算RBF (Gaussian) 内核矩阵。
        Args:
            a (Tensor): (B, D) 
            b (Tensor): (B, D)
            sigma (float): 内核宽度（可调，越大越平滑）
        Returns:
            Tensor: (B, B) 内核矩阵
        """
        # 计算平方欧氏距离
        a_norm = torch.sum(a ** 2, dim=1).view(-1, 1)  # (B, 1)
        b_norm = torch.sum(b ** 2, dim=1).view(1, -1)  # (1, B)
        dist_sq = a_norm + b_norm - 2 * torch.mm(a, b.t())  # (B, B)
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    def hsic_loss(self, x: torch.Tensor, y: torch.Tensor, sigma_x: float = 1.0, sigma_y: float = 1.0) -> torch.Tensor:
        """
        计算HSIC作为x和y之间依赖的损失（最小化它以解耦）。
        公式：HSIC = (1/(B-1)^2) * trace(K_x H K_y H)，其中H是中心矩阵。
        
        Args:
            x (Tensor): Semantic representation, shape (B, C, L)
            y (Tensor): Domain representation, shape (B, C, L)
            sigma_x (float): x的RBF内核宽度
            sigma_y (float): y的RBF内核宽度
            
        Returns:
            Tensor: HSIC值（scalar），作为损失最小化（接近0表示独立）
        """
        assert x.shape == y.shape, "x and y must have the same shape"
        B, C, L = x.shape
        if B < 2:
            return torch.tensor(0.0, device=x.device)  # 需要至少2个样本
        
        # 展平为 (B, C*L) 以作为特征向量
        x_flat = x.view(B, -1)  # (B, D), D = C*L
        y_flat = y.view(B, -1)  # (B, D)
        
        # 计算内核矩阵
        K_x = self.rbf_kernel(x_flat, x_flat, sigma_x)  # (B, B)
        K_y = self.rbf_kernel(y_flat, y_flat, sigma_y)  # (B, B)
        
        # 中心矩阵 H = I - (1/B) * ones
        H = torch.eye(B, device=x.device) - (1.0 / B) * torch.ones((B, B), device=x.device)
        
        # HSIC公式：(1/(B-1)^2) * trace(K_x H K_y H)
        # 等价于：(1/(B-1)^2) * vec(K_x H)^T vec(K_y H)
        KH_x = torch.mm(K_x, H)
        KH_y = torch.mm(K_y, H)
        hsic = torch.trace(torch.mm(KH_x, KH_y.t())) / ((B - 1) ** 2)
        
        return hsic  # 正值，最小化到0
