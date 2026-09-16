"""Encoder-decoder loss parsing for SAMO(+) component logging."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import torch
from torch import Tensor

from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS


@MODELS.register_module()
class SAMOPlusEncoderDecoder(EncoderDecoder):
    """Log SAMO(+) components without double-counting aggregate loss.

    MMEngine normally optimizes every dictionary item whose key contains
    ``loss``.  ``loss_sem_plus`` and ``loss_mi_plus`` are requested diagnostic
    decompositions of ``loss_sd_plus``, so they are excluded only from the
    optimizer sum while remaining in the normal log dictionary.
    """

    _metric_only_suffixes = (
        "loss_sem_plus",
        "loss_mi_plus",
    )

    def parse_losses(
        self, losses: Dict[str, Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        metric_only = {
            key: value
            for key, value in losses.items()
            if key.endswith(self._metric_only_suffixes)
        }
        optimized = {
            key: value for key, value in losses.items() if key not in metric_only
        }
        loss, log_vars = super().parse_losses(optimized)
        for key, value in metric_only.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{key} must be a tensor, got {type(value)}.")
            log_vars[key] = value.mean()
        return loss, OrderedDict(log_vars)

