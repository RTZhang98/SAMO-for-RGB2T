"""Label-anchored semantic-sufficiency loss for SAMO(+)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelAnchoredSemanticLoss(nn.Module):
    """Class-balanced prototype posterior supervised by source labels.

    EMA prototypes are FP32 buffers rather than trainable parameters.  At each
    training iteration the detached class statistics update the prototypes
    first; the posterior is then evaluated against that updated snapshot.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        momentum: float = 0.99,
        temperature: float = 0.1,
        samples_per_class: int = 256,
        ignore_index: int = 255,
        eps: float = 1e-6,
        distributed_sync: bool = True,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.samples_per_class = int(samples_per_class)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
        self.distributed_sync = bool(distributed_sync)

        self.register_buffer(
            "semantic_prototypes",
            torch.zeros(self.num_classes, self.feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(self.num_classes, dtype=torch.bool),
        )

    def _align_and_flatten(
        self, s_rgb: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if s_rgb.ndim != 4:
            raise ValueError(
                "s_rgb must have shape [B,D,H,W], "
                f"but received {tuple(s_rgb.shape)}."
            )
        if s_rgb.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, "
                f"but received D={s_rgb.shape[1]}."
            )
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 3:
            raise ValueError(
                "labels must have shape [B,H,W] or [B,1,H,W], "
                f"but received {tuple(labels.shape)}."
            )
        if labels.shape[0] != s_rgb.shape[0]:
            raise ValueError(
                f"Feature/label batch sizes differ: {s_rgb.shape[0]} and "
                f"{labels.shape[0]}."
            )

        spatial_size = s_rgb.shape[-2:]
        if labels.shape[-2:] != spatial_size:
            labels = F.interpolate(
                labels.unsqueeze(1).float(),
                size=spatial_size,
                mode="nearest",
            ).squeeze(1)
        labels = labels.long()

        flat_features = s_rgb.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
        flat_labels = labels.reshape(-1)
        valid = (
            (flat_labels != self.ignore_index)
            & (flat_labels >= 0)
            & (flat_labels < self.num_classes)
        )
        return flat_features[valid], flat_labels[valid]

    def _class_balanced_sample(
        self, features: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Build the paper's class-balanced feature-location set P."""
        if labels.numel() == 0:
            empty_classes = labels.new_empty((0,))
            return features, labels, empty_classes

        present_classes = torch.unique(labels, sorted=True)
        if self.samples_per_class <= 0:
            return features, labels, present_classes

        sampled_indices = []
        for class_id in present_classes:
            class_indices = torch.nonzero(labels == class_id, as_tuple=False).squeeze(1)
            count = class_indices.numel()
            if count >= self.samples_per_class:
                selection = torch.randperm(count, device=labels.device)[
                    : self.samples_per_class
                ]
            else:
                selection = torch.randint(
                    count,
                    (self.samples_per_class,),
                    device=labels.device,
                )
            sampled_indices.append(class_indices[selection])

        indices = torch.cat(sampled_indices, dim=0)
        return features[indices], labels[indices], present_classes

    def _distributed_enabled(self) -> bool:
        return (
            self.distributed_sync
            and dist.is_available()
            and dist.is_initialized()
        )

    @torch.no_grad()
    def _update_prototypes(self, features: Tensor, labels: Tensor) -> None:
        class_sums = torch.zeros(
            self.num_classes,
            self.feature_dim,
            device=self.semantic_prototypes.device,
            dtype=torch.float32,
        )
        class_counts = torch.zeros(
            self.num_classes,
            device=self.semantic_prototypes.device,
            dtype=torch.float32,
        )
        if labels.numel() > 0:
            detached_features = features.detach().float()
            class_sums.index_add_(0, labels, detached_features)
            class_counts.index_add_(
                0,
                labels,
                torch.ones(labels.shape[0], device=labels.device, dtype=torch.float32),
            )

        # Every rank executes the same two collectives, including ranks with no
        # local valid labels, so disjoint class sets cannot deadlock DDP.
        if self._distributed_enabled():
            dist.all_reduce(class_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_counts, op=dist.ReduceOp.SUM)

        globally_present = torch.nonzero(class_counts > 0, as_tuple=False).squeeze(1)
        for class_id in globally_present:
            class_index = int(class_id.item())
            batch_mean = class_sums[class_index] / class_counts[class_index]
            batch_mean = F.normalize(batch_mean, dim=0, eps=self.eps)
            if not self.prototype_initialized[class_index]:
                updated = batch_mean
                self.prototype_initialized[class_index] = True
            else:
                updated = (
                    self.momentum * self.semantic_prototypes[class_index]
                    + (1.0 - self.momentum) * batch_mean
                )
                updated = F.normalize(updated, dim=0, eps=self.eps)
            self.semantic_prototypes[class_index].copy_(updated)

    def _zero_diagnostics(self, s_rgb: Tensor) -> Dict[str, Tensor]:
        zero = s_rgb.detach().new_zeros((), dtype=torch.float32)
        return {
            "num_valid_locations": zero,
            "num_sampled_locations": zero.clone(),
            "num_present_classes": zero.clone(),
            "num_initialized_prototypes": self.prototype_initialized.sum()
            .detach()
            .float(),
            "mean_correct_similarity": zero.clone(),
        }

    def forward(
        self,
        s_rgb: Tensor,
        labels: Tensor,
        update_prototypes: bool = True,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        valid_features, valid_labels = self._align_and_flatten(s_rgb, labels)
        num_valid = valid_labels.numel()
        if num_valid == 0:
            # DDP ranks must still join prototype-statistic collectives.
            if update_prototypes and self._distributed_enabled():
                self._update_prototypes(valid_features, valid_labels)
            return s_rgb.sum() * 0.0, self._zero_diagnostics(s_rgb)

        sampled_features, sampled_labels, present_classes = (
            self._class_balanced_sample(valid_features, valid_labels)
        )

        if update_prototypes:
            self._update_prototypes(sampled_features, sampled_labels)

        initialized_targets = self.prototype_initialized[sampled_labels]
        posterior_features = sampled_features[initialized_targets]
        posterior_labels = sampled_labels[initialized_targets]
        if posterior_labels.numel() == 0:
            loss = s_rgb.sum() * 0.0
            mean_correct_similarity = s_rgb.detach().new_zeros(
                (), dtype=torch.float32
            )
        else:
            normalized_features = F.normalize(
                posterior_features.float(), dim=1, eps=self.eps
            )
            prototype_snapshot = self.semantic_prototypes.detach().float()
            logits = (
                normalized_features @ prototype_snapshot.transpose(0, 1)
            ) / self.temperature
            logits = logits.masked_fill(
                ~self.prototype_initialized.unsqueeze(0), -1e4
            )

            per_location_loss = F.cross_entropy(
                logits, posterior_labels, reduction="none"
            )
            if self.samples_per_class <= 0:
                class_losses = [
                    per_location_loss[posterior_labels == class_id].mean()
                    for class_id in torch.unique(posterior_labels, sorted=True)
                ]
                loss = torch.stack(class_losses).mean()
            else:
                loss = per_location_loss.mean()

            correct_prototypes = prototype_snapshot[posterior_labels]
            mean_correct_similarity = (
                normalized_features * correct_prototypes
            ).sum(dim=1).mean()

        diagnostics = {
            "num_valid_locations": torch.as_tensor(
                num_valid, device=s_rgb.device, dtype=torch.float32
            ).detach(),
            "num_sampled_locations": torch.as_tensor(
                sampled_labels.numel(), device=s_rgb.device, dtype=torch.float32
            ).detach(),
            "num_present_classes": torch.as_tensor(
                present_classes.numel(), device=s_rgb.device, dtype=torch.float32
            ).detach(),
            "num_initialized_prototypes": self.prototype_initialized.sum()
            .detach()
            .float(),
            "mean_correct_similarity": mean_correct_similarity.detach().float(),
        }
        return loss, diagnostics

