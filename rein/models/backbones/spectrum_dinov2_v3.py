from copy import deepcopy

import torch

from mmseg.models.builder import BACKBONES, MODELS

from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train

@BACKBONES.register_module()
class SpectrumDinoVisionTransformerV3(DinoVisionTransformer):

    def __init__(
        self,
        Spectrum_config=None,
        adapted_layers=(7, 11, 15, 23),
        **kwargs,
    ):
        if Spectrum_config is None:
            raise ValueError("Spectrum_config must be provided for Spectrum v3.")

        adapted_layers = tuple(int(layer) for layer in adapted_layers)
        if not adapted_layers:
            raise ValueError("adapted_layers must contain at least one block index.")
        if len(set(adapted_layers)) != len(adapted_layers):
            raise ValueError(f"adapted_layers contains duplicates: {adapted_layers}")
        if tuple(sorted(adapted_layers)) != adapted_layers:
            raise ValueError("adapted_layers must be in increasing block order.")

        spectrum_config = deepcopy(Spectrum_config)
        spectrum_config["num_layers"] = len(adapted_layers)
        super().__init__(**kwargs)
        self.spectrum = MODELS.build(spectrum_config)

        invalid = [layer for layer in adapted_layers if not 0 <= layer < len(self.blocks)]
        if invalid:
            raise ValueError(
                f"Invalid adapted layers {invalid} for a {len(self.blocks)}-block backbone."
            )
        missing_outputs = sorted(set(self.out_indices) - set(adapted_layers))
        if missing_outputs:
            raise ValueError(
                "Every out_indices block must currently be adapted because the "
                f"decode head consumes its Spectrum metadata; missing {missing_outputs}."
            )

        self.adapted_layers = adapted_layers
        self._block_to_adapter = {
            block_index: adapter_index
            for adapter_index, block_index in enumerate(adapted_layers)
        }

    def forward_features(self, x, masks=None):
        _, channels, height, width = x.shape
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size

        if channels != 3:

            tokens = self.prepare_tokens_with_masks(
                x.reshape(-1, 3, height, width), masks
            )
            source_tokens = tokens[::2]
            reference_tokens = tokens[1::2]
        else:

            source_tokens = self.prepare_tokens_with_masks(x, masks)
            reference_tokens = source_tokens.detach().clone()

        source_batch_size = source_tokens.shape[0]
        outputs = []
        spectrum_metadata = []
        for block_index, block in enumerate(self.blocks):
            with torch.no_grad():
                reference_tokens = block(reference_tokens).detach()
            source_tokens = block(source_tokens)

            adapter_index = self._block_to_adapter.get(block_index)
            mid_output = None
            if adapter_index is not None:
                source_tokens, mid_output = self.spectrum(
                    source_tokens,
                    reference_tokens,
                    adapter_index,
                    batch_first=True,
                    has_cls_token=True,
                )

            if block_index in self.out_indices:
                outputs.append(
                    source_tokens[:, 1:, :]
                    .permute(0, 2, 1)
                    .reshape(source_batch_size, -1, patch_height, patch_width)
                    .contiguous()
                )
                spectrum_metadata.append(mid_output)

        return self.spectrum.return_auto(outputs), spectrum_metadata

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["spectrum"])
        set_train(self, ["spectrum"])
        return self

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [key for key in state if "spectrum" not in key]
        for key in keys:
            state.pop(key)
            if destination is not None:
                destination.pop(key, None)
        return state
