"""Spatially correct SAMO v3 implementation for ViT patch tokens."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from mmseg.models.builder import MODELS

from .spectrum_core import LoRASpectrumCore


class SpatialSpectrumV3Mixin:
    """Correct spatial FFT and frequency-delta fusion for B,N,D tokens."""

    @staticmethod
    def _tokens_to_spatial(tokens: Tensor) -> tuple[Tensor, int, int]:
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected patch tokens in [B,N_patch,D], got {tuple(tokens.shape)}"
            )
        batch, num_patches, embedding_dim = tokens.shape
        height = math.isqrt(num_patches)
        width = height
        if height * width != num_patches:
            raise ValueError(
                "SpectrumV3 currently requires a square patch grid, but "
                f"N_patch={num_patches} is not a perfect square."
            )
        spatial = (
            tokens.transpose(1, 2)
            .reshape(batch, embedding_dim, height, width)
            .contiguous()
            .float()
        )
        return spatial, height, width

    @staticmethod
    def _spatial_to_channel_sequence(spatial: Tensor) -> Tensor:
        """Convert B,D,H,W to the B,D,N layout expected by LF/MF/HF modules."""
        return spatial.flatten(2).contiguous()

    @staticmethod
    def _channel_sequence_to_tokens(sequence: Tensor) -> Tensor:
        """Convert processor output B,D,N back to ViT token layout B,N,D."""
        return sequence.transpose(1, 2).contiguous()

    def forward(
        self,
        feats: Tensor,
        ref: Tensor,
        layer: int,
        batch_first: bool = False,
        has_cls_token: bool = True,
    ) -> tuple[Tensor, dict]:
        if not batch_first:
            raise ValueError(
                "SpectrumV3 expects batch-first ViT tokens in [B,N_patch,D]."
            )
        if feats.shape != ref.shape:
            raise ValueError(
                f"Source/reference token shapes differ: {feats.shape} vs {ref.shape}"
            )

        if has_cls_token:
            cls_token, patch_tokens = torch.tensor_split(feats, [1], dim=1)
            _, reference_tokens = torch.tensor_split(ref, [1], dim=1)
        else:
            cls_token = None
            patch_tokens = feats
            reference_tokens = ref

        if patch_tokens.shape[-1] != self.embed_dims:
            raise ValueError(
                f"Configured embed_dims={self.embed_dims}, but received "
                f"D={patch_tokens.shape[-1]}."
            )

        low_tokens = self.get_tokens(layer, component="low")
        mid_tokens = self.get_tokens(layer, component="mid")
        high_tokens = self.get_tokens(layer, component="high")

        source_spatial, height, width = self._tokens_to_spatial(patch_tokens)
        reference_spatial, ref_height, ref_width = self._tokens_to_spatial(
            reference_tokens
        )
        if (height, width) != (ref_height, ref_width):
            raise ValueError("Source/reference patch grids differ.")

        source_frequency = torch.fft.fftshift(
            torch.fft.fft2(source_spatial, dim=(-2, -1)), dim=(-2, -1)
        )
        reference_frequency = torch.fft.fftshift(
            torch.fft.fft2(reference_spatial, dim=(-2, -1)), dim=(-2, -1)
        )
        masks = self.create_frequency_masks(height, width, patch_tokens.device)

        source_bands = []
        reference_bands = []
        for mask in masks:
            expanded_mask = mask[None, None]
            source_band = torch.fft.ifft2(
                torch.fft.ifftshift(
                    source_frequency * expanded_mask, dim=(-2, -1)
                ),
                dim=(-2, -1),
            ).real
            reference_band = torch.fft.ifft2(
                torch.fft.ifftshift(
                    reference_frequency * expanded_mask, dim=(-2, -1)
                ),
                dim=(-2, -1),
            ).real
            source_bands.append(self._spatial_to_channel_sequence(source_band))
            reference_bands.append(
                self._spatial_to_channel_sequence(reference_band)
            )

        # These are the exact pre-modulation tensors in B,D,N layout.
        source_low, mid_output = self.low_component_func(
            source_bands[0], reference_bands[0]
        )
        source_mid = self.mid_component_func(
            source_bands[1], reference_bands[1]
        )
        source_high = self.high_component_func(
            source_bands[2], reference_bands[2]
        )

        # forward_delta_feat consumes N,B,D.  Make both token and embedding
        # axes explicit rather than relying on N_patch == D.
        source_tokens_nbd = patch_tokens.permute(1, 0, 2).contiguous()
        low_nbd = self._channel_sequence_to_tokens(source_low).permute(
            1, 0, 2
        ).contiguous()
        mid_nbd = self._channel_sequence_to_tokens(source_mid).permute(
            1, 0, 2
        ).contiguous()
        high_nbd = self._channel_sequence_to_tokens(source_high).permute(
            1, 0, 2
        ).contiguous()

        delta_low = self.forward_delta_feat(
            low_nbd,
            low_tokens,
            layer,
            self.mlp_token2feat_low,
            self.mlp_delta_f_low,
        )
        delta_mid = self.forward_delta_feat(
            mid_nbd,
            mid_tokens,
            layer,
            self.mlp_token2feat_mid,
            self.mlp_delta_f_mid,
        )
        delta_high = self.forward_delta_feat(
            high_nbd,
            high_tokens,
            layer,
            self.mlp_token2feat_high,
            self.mlp_delta_f_high,
        )

        delta = self.fuse_delta_features_weighted(
            delta_low, delta_mid, delta_high
        ) * self.scale
        output_tokens = (source_tokens_nbd + delta).permute(1, 0, 2).contiguous()
        if cls_token is not None:
            output_tokens = torch.cat((cls_token, output_tokens), dim=1)
        return output_tokens, mid_output

    def fuse_delta_features_weighted(
        self,
        delta_feat_low: Tensor,
        delta_feat_mid: Tensor,
        delta_feat_high: Tensor,
    ) -> Tensor:
        """Fuse N,B,D deltas along D and return N,B,D."""
        concatenated = torch.cat(
            (delta_feat_low, delta_feat_mid, delta_feat_high), dim=2
        )  # N,B,3D
        concatenated = concatenated.permute(1, 2, 0).contiguous()  # B,3D,N
        fused = self.fusion_linear(concatenated)  # B,D,N
        return F.relu(fused).permute(2, 0, 1).contiguous()  # N,B,D


@MODELS.register_module()
class LoRASpectrumV3(SpatialSpectrumV3Mixin, LoRASpectrumCore):
    """LoRA Spectrum with spatially correct patch-grid frequency bands."""
