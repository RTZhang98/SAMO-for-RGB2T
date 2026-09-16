"""SAMO(+) head with label-anchored semantic sufficiency."""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.models.builder import MODELS
from mmseg.utils import ConfigType, SampleList

from ..losses.label_anchored_semantic_loss import LabelAnchoredSemanticLoss
from .spectrum_mask2former import SpectrumMask2FormerHead


@MODELS.register_module()
class SpectrumMask2FormerHeadPlus(SpectrumMask2FormerHead):
    """A separate SAMO(+) version; the original head remains untouched.

    With ``samo_plus.enabled=False``, ``loss`` delegates directly to the
    original implementation.  With it enabled, only the original InfoNCE
    semantic term is replaced.  L_div, L_cross, L_rec, L_ind and segmentation
    losses retain their existing definitions.
    """

    def __init__(self, samo_plus: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        config = dict(samo_plus or {})
        self.samo_plus_enabled = bool(config.pop("enabled", False))
        self.semantic_feature_layout = config.pop(
            "semantic_feature_layout", "BDN"
        ).upper()
        self.num_semantic_levels = int(config.pop("num_semantic_levels", 4))
        self.alpha = float(config.pop("alpha", 0.5))
        self.beta = float(config.pop("beta", 2.0))
        self.gamma = float(config.pop("gamma", 1.0))

        if self.semantic_feature_layout != "BDN":
            raise ValueError(
                "SAMO v3 requires semantic_feature_layout='BDN', got "
                f"{self.semantic_feature_layout!r}."
            )
        if self.num_semantic_levels <= 0:
            raise ValueError("num_semantic_levels must be positive.")

        if self.samo_plus_enabled:
            loss_config = dict(
                num_classes=int(config.pop("num_classes", self.num_classes)),
                feature_dim=int(config.pop("feature_dim", 1024)),
                momentum=float(config.pop("prototype_momentum", 0.99)),
                temperature=float(config.pop("semantic_temperature", 0.1)),
                samples_per_class=int(config.pop("samples_per_class", 256)),
                ignore_index=int(config.pop("ignore_index", 255)),
                eps=float(config.pop("eps", 1e-6)),
                distributed_sync=bool(config.pop("distributed_sync", True)),
            )
            if config:
                raise ValueError(
                    f"Unknown SAMO(+) configuration fields: {sorted(config)}"
                )
            self.semantic_losses = nn.ModuleList(
                [
                    LabelAnchoredSemanticLoss(**loss_config)
                    for _ in range(self.num_semantic_levels)
                ]
            )

    @staticmethod
    def _source_labels(batch_data_samples: SampleList) -> Tensor:
        labels = []
        for sample in batch_data_samples:
            if not hasattr(sample, "gt_sem_seg"):
                raise ValueError(
                    "SAMO(+) requires source gt_sem_seg for every train sample."
                )
            label = sample.gt_sem_seg.data
            if label.ndim == 3 and label.shape[0] == 1:
                label = label[0]
            if label.ndim != 2:
                raise ValueError(
                    "Expected each gt_sem_seg to have shape [1,H,W] or [H,W], "
                    f"got {tuple(sample.gt_sem_seg.data.shape)}."
                )
            labels.append(label)
        return torch.stack(labels, dim=0)

    def _semantic_tokens_to_map(self, semantic_features: Tensor) -> Tensor:
        """Restore the ViT patch grid using the backbone's explicit layout."""
        if semantic_features.ndim == 4:
            return semantic_features
        if semantic_features.ndim != 3:
            raise ValueError(
                "Semantic features must be token [B,N,D]/[B,D,N] or map "
                f"[B,D,H,W], got {tuple(semantic_features.shape)}."
            )

        batch, feature_dim, num_patches = semantic_features.shape
        channel_first = semantic_features

        side = math.isqrt(num_patches)
        if side * side != num_patches:
            raise ValueError(
                "The current DINOv2/Spectrum token-to-map implementation "
                "requires a square patch grid, but received "
                f"N_patch={num_patches}."
            )
        return channel_first.reshape(batch, feature_dim, side, side).contiguous()

    def _original_semantic_term(
        self,
        sem_rgb: Tensor,
        sem_ir: Tensor,
        adapted: Tensor,
        tau_sem: float = 0.1,
        eps: float = 1e-8,
    ) -> Tensor:
        """The unmodified L_sem from ``SpectrumMask2FormerHead.bound_loss``."""
        s_rgb = sem_rgb.reshape(sem_rgb.shape[0], -1)
        s_th = sem_ir.reshape(sem_ir.shape[0], -1)
        s_pos = adapted.reshape(adapted.shape[0], -1)
        batch_size = s_rgb.shape[0]
        if batch_size < 2:
            return sem_rgb.sum() * 0.0

        score_pos = F.cosine_similarity(
            F.normalize(s_rgb, dim=1, eps=eps),
            F.normalize(s_pos, dim=1, eps=eps),
            dim=1,
        ) / max(tau_sem, eps)
        normalized_rgb = F.normalize(s_rgb, dim=1, p=2, eps=eps)
        normalized_thermal = F.normalize(s_th, dim=1, p=2, eps=eps)
        score_neg = (
            normalized_rgb @ normalized_thermal.transpose(0, 1)
        ) / max(tau_sem, eps)
        diagonal = torch.eye(
            batch_size, device=s_rgb.device, dtype=torch.bool
        )
        score_neg = score_neg.masked_fill(diagonal, float("-inf"))
        denominator = torch.logsumexp(
            torch.cat([score_pos.unsqueeze(1), score_neg], dim=1), dim=1
        )
        return (-(score_pos - denominator)).mean()

    def _compute_mid_losses_plus(
        self,
        mid_output: List[Dict[str, Tensor]],
        labels: Tensor,
    ) -> dict:
        if not mid_output:
            raise ValueError("SAMO(+) received no decoupling metadata.")
        if len(mid_output) > len(self.semantic_losses):
            raise ValueError(
                f"Received {len(mid_output)} semantic levels, but only "
                f"{len(self.semantic_losses)} prototype banks are configured."
            )

        loss_ind = labels.new_zeros((), dtype=torch.float32)
        loss_rec = labels.new_zeros((), dtype=torch.float32)
        loss_mi_plus = labels.new_zeros((), dtype=torch.float32)
        loss_sem_plus = labels.new_zeros((), dtype=torch.float32)
        initialized_counts = []
        present_counts = []

        for level_index, layer_features in enumerate(mid_output):
            sem_rgb = layer_features["sem_rgb"]
            dom_rgb = layer_features["dom_rgb"]
            sem_ir = layer_features["sem_ir"]
            dom_ir = layer_features["dom_ir"]
            fused_rgb = layer_features["fused_rgb"]
            fused_ir = layer_features["fused_ir"]
            adapted = layer_features["adapted_low"]
            ori_rgb = layer_features["ori_rgb"]
            ori_ir = layer_features["ori_ir"]

            # Existing L_ind and L_rec definitions are kept verbatim.
            loss_ind = loss_ind + (
                self.hsic_loss(sem_rgb, dom_rgb)
                + self.hsic_loss(sem_ir, dom_ir)
            ) / 2.0
            loss_rec = loss_rec + (
                F.mse_loss(fused_rgb, ori_rgb)
                + F.mse_loss(fused_ir, ori_ir)
            ) / 2.0

            # Existing bound_loss = L_sem + L_div + L_cross.  Subtracting its
            # original L_sem leaves the original L_div and L_cross untouched.
            original_mi = super().bound_loss(
                sem_rgb, dom_rgb, sem_ir, dom_ir, adapted
            )
            original_semantic = self._original_semantic_term(
                sem_rgb, sem_ir, adapted
            )
            semantic_map = self._semantic_tokens_to_map(sem_rgb)
            semantic_plus, diagnostics = self.semantic_losses[level_index](
                s_rgb=semantic_map,
                labels=labels,
                update_prototypes=self.training,
            )
            loss_sem_plus = loss_sem_plus + semantic_plus
            loss_mi_plus = (
                loss_mi_plus
                + original_mi
                - original_semantic
                + semantic_plus
            )
            initialized_counts.append(
                diagnostics["num_initialized_prototypes"]
            )
            present_counts.append(diagnostics["num_present_classes"])

        divisor = float(len(mid_output))
        loss_ind = loss_ind / divisor
        loss_rec = loss_rec / divisor
        loss_sem_plus = loss_sem_plus / divisor
        loss_mi_plus = loss_mi_plus / divisor
        loss_sd_plus = (
            self.alpha * loss_mi_plus
            + self.beta * loss_rec
            + self.gamma * loss_ind
        )

        # loss_sd_plus is the only aggregate below used for optimization.
        # The custom SAMO(+) segmentor logs the two detached component values
        # without adding them to MMEngine's automatic loss sum a second time.
        return {
            "loss_sd_plus": loss_sd_plus,
            "loss_sem_plus": loss_sem_plus.detach(),
            "loss_mi_plus": loss_mi_plus.detach(),
            "samo_plus_initialized_prototypes": torch.stack(
                initialized_counts
            ).mean(),
            "samo_plus_present_classes": torch.stack(present_counts).mean(),
        }

    def loss(
        self,
        x: tuple[Tensor],
        batch_data_samples: SampleList,
        train_cfg: ConfigType,
    ) -> dict:
        if not self.samo_plus_enabled:
            return super().loss(x, batch_data_samples, train_cfg)

        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples
        )
        all_cls_scores, all_mask_preds, mid_output = self(
            x, batch_data_samples, mod="train"
        )
        losses = self.loss_by_feat(
            all_cls_scores,
            all_mask_preds,
            batch_gt_instances,
            batch_img_metas,
        )
        labels = self._source_labels(batch_data_samples)
        losses.update(self._compute_mid_losses_plus(mid_output, labels))
        return losses
