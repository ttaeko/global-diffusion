"""Transparent boundary between the epoch-500 macro model and stock base model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from terrain_diffusion.models.mp_layers import mp_concat


STOCK_CONDITIONING_MEAN = torch.tensor(
    [14.99, 11.65, 15.87, 619.26, 833.12, 69.40, 0.66],
    dtype=torch.float32,
)
STOCK_CONDITIONING_STD = torch.tensor(
    [21.72, 21.78, 10.40, 452.29, 738.09, 34.59, 0.47],
    dtype=torch.float32,
)
STOCK_HISTOGRAM_RAW = torch.zeros(1, 5, dtype=torch.float32)

MACRO_CHANNEL_NAMES = (
    "mean_elevation_signed_sqrt_m",
    "p5_elevation_signed_sqrt_m",
    "annual_mean_temperature_c",
    "temperature_seasonality",
    "annual_precipitation_mm",
    "precipitation_seasonality",
)


@dataclass(frozen=True)
class AdaptedConditioning:
    """All observable products of the non-learned interface adapter."""

    physical_6x4x4: torch.Tensor
    physical_with_mask_7x4x4: torch.Tensor
    normalized_7x4x4: torch.Tensor
    vector_58: torch.Tensor


def adapt_macro_patch(macro_physical: torch.Tensor) -> AdaptedConditioning:
    """Construct the stock 58-value base conditioning from one 4x4 macro patch.

    The macro checkpoint already emits the stock coarse-stage semantics. This
    function only validates/copies those values, appends the all-valid mask,
    applies the stock normalization, and lays out the exact stock vector:
    16 mean + 16 p5 + 4 climate means + 16 mask + 5 histogram + 1 noise.
    """
    patch = torch.as_tensor(macro_physical, dtype=torch.float32)
    if patch.shape != (6, 4, 4):
        raise ValueError(
            f"Expected an aligned macro patch shaped (6, 4, 4), got {tuple(patch.shape)}"
        )
    if not torch.isfinite(patch).all():
        raise ValueError("Macro conditioning contains non-finite values")

    mask = torch.ones(1, 4, 4, dtype=patch.dtype, device=patch.device)
    physical_with_mask = torch.cat([patch, mask], dim=0)
    means = STOCK_CONDITIONING_MEAN.to(patch.device).view(7, 1, 1)
    stds = STOCK_CONDITIONING_STD.to(patch.device).view(7, 1, 1)
    normalized = (physical_with_mask - means) / stds

    climate_means = normalized[2:6, 1:3, 1:3].mean(dim=(1, 2))
    noise_level_norm = torch.tensor(
        [(0.0 - 0.5) * np.sqrt(12.0)], dtype=patch.dtype, device=patch.device
    )
    vector = mp_concat(
        [
            normalized[0].reshape(-1),
            normalized[1].reshape(-1),
            climate_means,
            normalized[6].reshape(-1),
            STOCK_HISTOGRAM_RAW.to(patch.device).reshape(-1),
            noise_level_norm,
        ],
        dim=0,
    ).float()
    if vector.shape != (58,):
        raise RuntimeError(f"Stock conditioning vector must have 58 values, got {vector.shape}")
    return AdaptedConditioning(patch, physical_with_mask, normalized, vector)

