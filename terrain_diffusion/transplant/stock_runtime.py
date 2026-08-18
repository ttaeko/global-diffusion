"""Literal one-tile stock base, decoder, and Laplacian inference operations."""

from __future__ import annotations

import numpy as np
import torch

from terrain_diffusion.data.laplacian_encoder import laplacian_decode, laplacian_denoise
from terrain_diffusion.inference.portable_rng import fill_standard_normal
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler


SIGMA_DATA = 0.5
LOWFREQ_MEAN = -31.4
LOWFREQ_STD = 38.6


def _tile_seed(base_seed: int, ty: int, tx: int) -> int:
    h = (int(base_seed) & 0xFFFFFFFFFFFFFFFF) * 0x9E3779B9
    h = (h + (int(ty) & 0xFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    return (h * 0x9E3779B9 + (int(tx) & 0xFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF


def gaussian_noise_patch(
    base_seed: int,
    y0: int,
    x0: int,
    height: int,
    width: int,
    channels: int,
    tile_height: int,
    tile_width: int,
) -> np.ndarray:
    """Stock coordinate-stable Gaussian field, retained for inference parity."""
    out = np.empty((channels, height, width), dtype=np.float32)
    ty0, ty1 = y0 // tile_height, (y0 + height - 1) // tile_height
    tx0, tx1 = x0 // tile_width, (x0 + width - 1) // tile_width
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            tile_y0, tile_x0 = ty * tile_height, tx * tile_width
            oy0, oy1 = max(y0, tile_y0), min(y0 + height, tile_y0 + tile_height)
            ox0, ox1 = max(x0, tile_x0), min(x0 + width, tile_x0 + tile_width)
            tile = np.empty((channels, tile_height, tile_width), dtype=np.float32)
            fill_standard_normal(_tile_seed(base_seed, ty, tx), tile.ravel())
            out[:, oy0-y0:oy1-y0, ox0-x0:ox1-x0] = tile[
                :, oy0-tile_y0:oy1-tile_y0, ox0-tile_x0:ox1-tile_x0
            ]
    return out


@torch.no_grad()
def generate_hybrid_stage(base_model, conditioning_vector: torch.Tensor, seed: int, device: str, *, stage: int, tile_row: int, tile_col: int, previous: torch.Tensor | None = None):
    """Run one literal WorldPipeline latent stage at its overlapping tile coordinate."""
    scheduler = EDMDPMSolverMultistepScheduler(
        sigma_min=0.002, sigma_max=80, sigma_data=SIGMA_DATA
    )
    dtype = next(base_model.parameters()).dtype
    cond = conditioning_vector.view(1, 58).to(device=device, dtype=dtype)
    if stage not in (0, 1):
        raise ValueError(f"Expected latent stage 0 or 1, got {stage}")
    sample = torch.zeros((1, 5, 64, 64), device=device, dtype=dtype) if previous is None else previous.view(1, 5, 64, 64).to(device=device, dtype=dtype) * SIGMA_DATA
    t = (torch.atan(scheduler.sigmas[0] / SIGMA_DATA) if stage == 0 else torch.arctan(torch.tensor(0.35) / SIGMA_DATA)).to(device=device, dtype=dtype)
    noise = torch.from_numpy(gaussian_noise_patch(seed + 5819 + stage, tile_row * 32, tile_col * 32, 64, 64, 5, 64, 64)).unsqueeze(0).to(device=device, dtype=dtype)
    t_view = t.view(1, 1, 1, 1)
    x_t = torch.cos(t_view) * sample + torch.sin(t_view) * noise * SIGMA_DATA
    pred = -base_model(x_t / SIGMA_DATA, noise_labels=t.view(1), conditional_inputs=[cond])
    sample = torch.cos(t_view) * x_t - torch.sin(t_view) * SIGMA_DATA * pred
    hybrid = sample[0].cpu().float() / SIGMA_DATA
    if hybrid.shape != (5, 64, 64) or not torch.isfinite(hybrid).all():
        raise RuntimeError(f"Invalid stock hybrid output: {tuple(hybrid.shape)}")
    return hybrid


@torch.no_grad()
def generate_hybrid(base_model, conditioning_vector: torch.Tensor, seed: int, device: str):
    """Single-tile compatibility wrapper retaining the prior output."""
    initial = generate_hybrid_stage(base_model, conditioning_vector, seed, device, stage=0, tile_row=0, tile_col=0)
    return generate_hybrid_stage(base_model, conditioning_vector, seed, device, stage=1, tile_row=0, tile_col=0, previous=initial)


@torch.no_grad()
def generate_residual(decoder_model, hybrid: torch.Tensor, seed: int, device: str, *, tile_row: int = 0, tile_col: int = 0, tile_stride: int = 512):
    """Run the stock one-evaluation 240 m-to-30 m decoder for one 512px tile."""
    scheduler = EDMDPMSolverMultistepScheduler(
        sigma_min=0.002, sigma_max=80, sigma_data=SIGMA_DATA
    )
    dtype = next(decoder_model.parameters()).dtype
    latents = hybrid[:4].view(1, 4, 64, 64)
    latents_up = torch.nn.functional.interpolate(
        latents, size=(512, 512), mode="nearest"
    ).to(device=device, dtype=dtype)
    sample = torch.zeros((1, 1, 512, 512), device=device, dtype=dtype)
    t = torch.atan(scheduler.sigmas[0] / SIGMA_DATA).to(device=device, dtype=dtype)
    noise = torch.from_numpy(
        gaussian_noise_patch(seed + 5819, tile_row * tile_stride, tile_col * tile_stride, 512, 512, 1, 512, 512)
    ).unsqueeze(0).to(device=device, dtype=dtype)
    z = noise * SIGMA_DATA
    t_view = t.view(1, 1, 1, 1)
    x_t = torch.cos(t_view) * sample + torch.sin(t_view) * z
    model_in = torch.cat([x_t / SIGMA_DATA, latents_up], dim=1)
    pred = -decoder_model(model_in, noise_labels=t.view(1), conditional_inputs=[])
    sample = torch.cos(t_view) * x_t - torch.sin(t_view) * SIGMA_DATA * pred
    residual_normalized = sample[0, 0].cpu().float() / SIGMA_DATA
    if residual_normalized.shape != (512, 512) or not torch.isfinite(residual_normalized).all():
        raise RuntimeError(f"Invalid decoder residual: {tuple(residual_normalized.shape)}")
    return residual_normalized


def reconstruct_elevation(
    residual_normalized: torch.Tensor,
    hybrid: torch.Tensor,
    residual_mean: float = 0.0,
    residual_std: float = 0.7,
):
    """Apply the stock signed-sqrt Laplacian reconstruction and invert the transform."""
    residual_sqrt = residual_normalized * residual_std + residual_mean
    lowfreq_sqrt = hybrid[4] * LOWFREQ_STD + LOWFREQ_MEAN
    residual_sqrt, lowfreq_sqrt = laplacian_denoise(
        residual_sqrt, lowfreq_sqrt, sigma=5
    )
    elevation_sqrt = laplacian_decode(residual_sqrt, lowfreq_sqrt)
    elevation_m = torch.sign(elevation_sqrt) * torch.square(elevation_sqrt)
    expected_shape = tuple(residual_normalized.shape)
    if elevation_m.shape != expected_shape or not torch.isfinite(elevation_m).all():
        raise RuntimeError(f"Invalid reconstructed DEM: {tuple(elevation_m.shape)}")
    return residual_sqrt, lowfreq_sqrt, elevation_sqrt, elevation_m
