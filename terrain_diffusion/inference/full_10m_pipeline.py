"""End-to-end 7.68 km -> 240 m -> 30 m -> 10 m inference adapter.

The upstream stack is deliberately delegated to ``smoke_transplant``.  It is
the promoted, coordinate-stable stock mosaic implementation, so this module
only adds the deterministic 30 m -> 10 m pass and its exact-parent closure.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from terrain_diffusion.hydrology.conditioning import build_hydrology_conditioning
from terrain_diffusion.hydrology.decoder_contract import build_fine_hydrology_geometry
from terrain_diffusion.hydrology.profile_contract import DEFAULT_HYDROLOGY_PROFILE
from terrain_diffusion.hydrology.training_profile import (
    build_hydrology_training_profile,
    enforce_profile_on_refined_terrain,
)
from terrain_diffusion.models.one_pass_conditioning import build_10m_terrain_conditioning
from terrain_diffusion.models.one_pass_residual import (
    block_mean_3x3,
    project_zero_block_mean_3x,
    smooth_exact_upsample_3x,
)
from terrain_diffusion.models.one_pass_unet import OnePassUNet
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.training.datasets.one_pass_dataset import (
    HYDRO_CHANNEL_INDICES,
    RESIDUAL_STD_M,
    normalize_10m_conditioning,
)
from terrain_diffusion.transplant.smoke import (
    DEFAULT_MACRO_MODEL,
    DEFAULT_MACRO_STATS,
    DEFAULT_STOCK_MODEL,
    DEFAULT_STOCK_REVISION,
    _choose_mountain_crop,
    _get_lowfreq_as_elevation,
    _load_stock_submodel,
    _release_model,
    _select_device,
    _sample_macro,
    smoke_transplant,
)
from terrain_diffusion.transplant.adapter import adapt_macro_patch
from terrain_diffusion.transplant.stock_runtime import (
    LOWFREQ_MEAN,
    LOWFREQ_STD,
    generate_hybrid,
    generate_residual,
    reconstruct_elevation,
)
from terrain_diffusion.hydrology.mosaic import (
    reconcile_stock_mosaic_hydrology_30m,
    run_stock_mosaic_hydrology,
)


TERRAIN_HALO_30M = 4  # Largest Candidate-A descriptor is a 9x9 parent window.


def _origins(length: int, tile: int, stride: int) -> tuple[int, ...]:
    if tile > length:
        raise ValueError("10 m tile must not exceed the requested 30 m window")
    result = list(range(0, length - tile + 1, stride))
    if result[-1] != length - tile:
        result.append(length - tile)
    return tuple(result)


def _blend_window(size: int, device: torch.device) -> torch.Tensor:
    # Positive weights avoid uncovered edge pixels and match the upstream blend
    # policy's important property without coupling this stage to stock details.
    x = torch.arange(size, device=device, dtype=torch.float32)
    ramp = torch.sin(torch.pi * (x + 0.5) / size).square()
    return (1e-3 + 0.999 * ramp[:, None] * ramp[None, :])[None, None]


def _signed_sqrt(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.sqrt(np.abs(values))


def _load_one_pass(checkpoint: Path, device: str, base_channels: int) -> OnePassUNet:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    model = OnePassUNet(in_channels=20, out_channels=1, base_channels=base_channels)
    model.load_state_dict(state, strict=True)
    residual_std = float(payload.get("residual_std_m", RESIDUAL_STD_M))
    if not np.isclose(residual_std, RESIDUAL_STD_M, rtol=0.0, atol=1e-8):
        raise click.ClickException(
            f"Checkpoint residual_std_m={residual_std} does not match Candidate-A "
            f"contract {RESIDUAL_STD_M}"
        )
    return model.eval().to(device)


def _hydrology_values(output: Path) -> np.ndarray:
    return build_hydrology_conditioning(
        np.load(output / "reconciled_30m_flow_direction_d8.npy"),
        np.load(output / "reconciled_30m_accumulation_area_m2.npy"),
        np.load(output / "reconciled_30m_catchment_id.npy"),
        np.load(output / "reconciled_30m_channel_mask.npy").astype(bool),
        np.load(output / "reconciled_30m_stream_order.npy"),
        resolution_m=30.0,
        lake_id=np.load(output / "reconciled_30m_lake_id.npy"),
        mean_discharge_m3s=np.load(output / "reconciled_30m_mean_discharge_m3s.npy"),
        distance_scale_m=DEFAULT_HYDROLOGY_PROFILE.conditioning_distance_scale_m,
    ).values


def _condition_tile(parent: torch.Tensor, hydrology: torch.Tensor, y: int, x: int, size: int) -> torch.Tensor:
    """Build the training-identical 20-channel condition with real mosaic halo."""
    halo = TERRAIN_HALO_30M
    parent_pad = F.pad(parent, (halo, halo, halo, halo), mode="replicate")
    context = parent_pad[:, :, y:y + size + 2 * halo, x:x + size + 2 * halo]
    terrain = build_10m_terrain_conditioning(context)[:, :, halo * 3:-halo * 3, halo * 3:-halo * 3]
    hydro_crop = hydrology[:, HYDRO_CHANNEL_INDICES, y:y + size, x:x + size]
    continuous = F.interpolate(hydro_crop[:, :5], scale_factor=3, mode="bilinear", align_corners=False)
    discrete = F.interpolate(hydro_crop[:, 5:], scale_factor=3, mode="nearest")
    return normalize_10m_conditioning(torch.cat((terrain, continuous, discrete), dim=1))


def _save_hillshade(path: Path, elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    gy, gx = np.gradient(elevation, resolution_m, resolution_m)
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    shade = np.sin(np.deg2rad(45)) * np.cos(slope) + np.cos(np.deg2rad(45)) * np.sin(slope) * np.cos(np.deg2rad(315) - aspect)
    shade = np.clip(shade, 0.0, 1.0)
    plt.imsave(path, shade, cmap="gray")
    return shade


def _save_overlay(path: Path, hillshade: np.ndarray, channels: np.ndarray, lakes: np.ndarray) -> None:
    image = np.repeat(hillshade[..., None], 3, axis=2)
    image[lakes] = (0.0, 0.65, 1.0)
    image[channels] = (0.05, 0.30, 1.0)
    plt.imsave(path, image)


def _float32_parent_tolerance(
    parent: torch.Tensor,
    ulps: int = 4,
) -> float:
    """Tolerance for A3 evaluated through float32 arithmetic.

    Parent preservation is algebraically exact, but construction,
    projection, reconstruction, and the final 3x3 reduction each
    introduce float32 rounding.

    Four parent ULPs is still a sub-millimetre-scale tolerance over
    ordinary terrain elevations while allowing expected numerical noise.
    """
    parent = parent.float()

    next_up = torch.nextafter(
        parent,
        torch.full_like(parent, float("inf")),
    )

    next_down = torch.nextafter(
        parent,
        torch.full_like(parent, float("-inf")),
    )

    ulp = torch.maximum(
        (next_up - parent).abs(),
        (parent - next_down).abs(),
    )

    return float(
        (ulps * ulp).max().cpu()
    )


def _sample_single_stock_crop(*, output: Path, macro_model: Path, macro_stats: Path, stock_model: str, stock_revision: str | None, allow_download: bool, seed: int, land_fraction: float, macro_steps: int, coarse_row: int, coarse_col: int, use_mountain_crop: bool, device: str) -> None:
    """Run exactly one stock 7.68 km -> 240 m -> 30 m crop for a fast smoke.

    The 4x4 macro patch has the same stock-conditioning contract as one tile
    in the promoted mosaic; its central 2x2 macro cells exactly parent the
    64x64 low-frequency and 512x512 30 m outputs.
    """
    macro = EDMUnet2D.from_pretrained(macro_model).eval().to(device)
    normalized, physical, _, _ = _sample_macro(macro, seed, land_fraction, macro_steps, macro_stats, device)
    macro_elevation = torch.sign(physical[0]) * torch.square(physical[0])
    np.save(output / "macro_normalized_6x256x256.npy", normalized.numpy().astype(np.float32))
    np.save(output / "macro_physical_6x256x256.npy", physical.numpy().astype(np.float32))
    np.save(output / "macro_elevation_m_256x256.npy", macro_elevation.numpy().astype(np.float32))
    if use_mountain_crop:
        coarse_row, coarse_col, _ = _choose_mountain_crop(physical)
    patch = adapt_macro_patch(physical[:, coarse_row:coarse_row + 4, coarse_col:coarse_col + 4])
    _release_model(macro, device)
    del macro

    base = _load_stock_submodel(stock_model, "base_model", stock_revision, allow_download).eval().to(device)
    hybrid = generate_hybrid(base, patch.vector_58, seed, device)
    _release_model(base, device)
    del base
    np.save(output / "hybrid_0_z1_normalized.npy", hybrid[0].numpy())
    np.save(output / "hybrid_1_z2_normalized.npy", hybrid[1].numpy())
    np.save(output / "hybrid_2_z3_normalized.npy", hybrid[2].numpy())
    np.save(output / "hybrid_3_z4_normalized.npy", hybrid[3].numpy())
    np.save(output / "hybrid_4_lowfreq_normalized.npy", hybrid[4].numpy())
    lowfreq_sqrt = hybrid[4] * LOWFREQ_STD + LOWFREQ_MEAN
    lowfreq_m = _get_lowfreq_as_elevation(lowfreq_sqrt).numpy()
    np.save(output / "hybrid_4_lowfreq_signed_sqrt_physical.npy", lowfreq_sqrt.numpy())
    macro_slice = slice(coarse_row + 1, coarse_row + 3)
    macro_col_slice = slice(coarse_col + 1, coarse_col + 3)
    run_stock_mosaic_hydrology(
        output,
        macro_elevation_m=macro_elevation[macro_slice, macro_col_slice].numpy(),
        macro_precipitation_mm=physical[4, macro_slice, macro_col_slice].numpy(),
        lowfreq_m=lowfreq_m,
    )

    decoder = _load_stock_submodel(stock_model, "decoder_model", stock_revision, allow_download).eval().to(device)
    residual_normalized = generate_residual(decoder, hybrid, seed, device)
    _, _, elevation_sqrt, elevation_m = reconstruct_elevation(residual_normalized, hybrid)
    _release_model(decoder, device)
    del decoder
    np.save(output / "decoder_residual_normalized_512x512.npy", residual_normalized.numpy())
    np.save(output / "final_elevation_signed_sqrt_512x512.npy", elevation_sqrt.numpy())
    np.save(output / "final_dem_m_3584x3584.npy", elevation_m.numpy())
    reconcile_stock_mosaic_hydrology_30m(output, elevation_m.numpy())


@click.command("sample-full-10m")
@click.option("--one-pass-checkpoint", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Trained Candidate-A 30 m -> 10 m checkpoint (.pt).")
@click.option("--one-pass-base-channels", default=48, show_default=True, type=click.IntRange(min=1))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--macro-model", default=DEFAULT_MACRO_MODEL, type=click.Path(exists=True, file_okay=False, path_type=Path), show_default=True)
@click.option("--macro-stats", default=DEFAULT_MACRO_STATS, type=click.Path(exists=True, dir_okay=False, path_type=Path), show_default=True)
@click.option("--stock-model", default=DEFAULT_STOCK_MODEL, show_default=True)
@click.option("--stock-revision", default=DEFAULT_STOCK_REVISION, show_default=True)
@click.option("--allow-download", is_flag=True, help="Allow stock-model download; default is cache-only.")
@click.option("--seed", default=74, show_default=True, type=int)
@click.option("--land-fraction", default=0.60, show_default=True, type=click.FloatRange(0.0, 1.0))
@click.option("--macro-steps", default=30, show_default=True, type=click.IntRange(min=1))
@click.option("--coarse-row", default=120, show_default=True, type=click.IntRange(0, 240))
@click.option("--coarse-col", default=120, show_default=True, type=click.IntRange(0, 240))
@click.option("--use-mountain-crop", "use_mountain_crop", is_flag=True, help="Select the upstream sampler's strongest-relief macro crop.")
@click.option("--upstream-mode", type=click.Choice(("full", "single")), default="full", show_default=True, help="full uses the promoted 107.52 km mosaic; single evaluates exactly one stock base and decoder crop for smoke testing.")
@click.option("--window-row-30m", default=None, type=click.IntRange(min=0), help="Top row of the delivered 30 m window; default centers it.")
@click.option("--window-col-30m", default=None, type=click.IntRange(min=0), help="Left column of the delivered 30 m window; default centers it.")
@click.option("--window-size-30m", default=1120, show_default=True, type=click.IntRange(min=128), help="Square final window; 1120 cells = 33.6 km.")
@click.option("--tile-size-30m", default=128, show_default=True, type=click.IntRange(min=8), help="One-pass U-Net tile edge in parent cells.")
@click.option("--tile-overlap-30m", default=32, show_default=True, type=click.IntRange(min=0), help="Parent-cell overlap between 10 m U-Net tiles.")
@click.option("--device", default=None, help="cuda, mps, or cpu; auto-select when omitted.")
def sample_full_10m(one_pass_checkpoint, one_pass_base_channels, output, macro_model, macro_stats, stock_model, stock_revision, allow_download, seed, land_fraction, macro_steps, coarse_row, coarse_col, use_mountain_crop, upstream_mode, window_row_30m, window_col_30m, window_size_30m, tile_size_30m, tile_overlap_30m, device):
    """Run the promoted upstream mosaic then deterministic tiled 30 m -> 10 m."""
    if tile_size_30m % 8:
        raise click.UsageError("--tile-size-30m must be divisible by 8 for OnePassUNet")
    if not 0 <= tile_overlap_30m < tile_size_30m:
        raise click.UsageError("--tile-overlap-30m must be in [0, tile-size-30m)")
    if output.exists() and any(output.iterdir()):
        raise click.ClickException(f"Output directory already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = _select_device(device)

    started = time.perf_counter()
    if upstream_mode == "full":
        # This is intentionally the existing command callback, not a copied
        # stock sampler. It writes all 7.68 km, 240 m, and 30 m artifacts.
        smoke_transplant.callback(macro_model, macro_stats, stock_model, stock_revision, allow_download, output, seed, land_fraction, macro_steps, coarse_row, coarse_col, use_mountain_crop, True, True, device)
    else:
        _sample_single_stock_crop(
            output=output, macro_model=macro_model, macro_stats=macro_stats,
            stock_model=stock_model, stock_revision=stock_revision,
            allow_download=allow_download, seed=seed, land_fraction=land_fraction,
            macro_steps=macro_steps, coarse_row=coarse_row, coarse_col=coarse_col,
            use_mountain_crop=use_mountain_crop, device=device,
        )
    upstream_seconds = time.perf_counter() - started

    parent_np = np.load(output / "reconciled_30m_hydrology_repaired_dem_m.npy").astype(np.float32)
    hydro_np = _hydrology_values(output)
    height, width = parent_np.shape
    size = int(window_size_30m)
    if size > min(height, width):
        raise click.ClickException(f"Requested {size}x{size} parent window exceeds upstream {height}x{width} mosaic")
    y0 = (height - size) // 2 if window_row_30m is None else int(window_row_30m)
    x0 = (width - size) // 2 if window_col_30m is None else int(window_col_30m)
    if y0 < 0 or x0 < 0 or y0 + size > height or x0 + size > width:
        raise click.ClickException("Requested 30 m output window lies outside the upstream mosaic")

    model = _load_one_pass(one_pass_checkpoint, device, one_pass_base_channels)
    parent_full = torch.from_numpy(parent_np)[None, None].to(device)
    hydro_full = torch.from_numpy(hydro_np)[None].to(device)
    tile10, stride = tile_size_30m * 3, tile_size_30m - tile_overlap_30m
    ys, xs = _origins(size, tile_size_30m, stride), _origins(size, tile_size_30m, stride)
    residual_sum = torch.zeros((1, 1, size * 3, size * 3), device=device)
    weight_sum = torch.zeros_like(residual_sum)
    weight = _blend_window(tile10, torch.device(device))
    ten_started = time.perf_counter()
    with torch.no_grad():
        for local_y in ys:
            for local_x in xs:
                condition = _condition_tile(parent_full, hydro_full, y0 + local_y, x0 + local_x, tile_size_30m)
                raw = model(condition)
                ay, ax = local_y * 3, local_x * 3
                residual_sum[:, :, ay:ay + tile10, ax:ax + tile10] += raw * weight
                weight_sum[:, :, ay:ay + tile10, ax:ax + tile10] += weight
    raw_normalized = residual_sum / weight_sum
    # Blending mixes parent footprints. Close the final, full mosaic once in
    # residual space before P3 reconstruction, which restores A3 exactly.
    residual_m = project_zero_block_mean_3x(raw_normalized) * RESIDUAL_STD_M
    parent_window = parent_full[:, :, y0:y0 + size, x0:x0 + size]
    base10 = smooth_exact_upsample_3x(parent_window)
    generated10 = base10 + residual_m
    ten_seconds = time.perf_counter() - ten_started

    parent_error = block_mean_3x3(generated10) - parent_window
    max_parent_error = float(parent_error.abs().max().cpu())
    parent_tolerance = _float32_parent_tolerance(parent_window)
    if max_parent_error > parent_tolerance:
        raise RuntimeError(f"Final 10 m parent preservation failed: {max_parent_error:.6g} m")

    # Reuse the existing profile and fine-geometry builders.  The profile
    # reassertion is saved as an auditable hydrology candidate; a final Q3
    # closure is mandatory because arbitrary vertical profile corrections do
    # not themselves preserve each authoritative 30 m mean.
    hydro_window = hydro_np[:, y0:y0 + size, x0:x0 + size]
    profile = build_hydrology_training_profile(
        _signed_sqrt(parent_np[y0:y0 + size, x0:x0 + size]), hydro_window,
        **DEFAULT_HYDROLOGY_PROFILE.profile_kwargs(resolution_m=30.0),
        sea_level_elevation_m=0.0,
        lake_water_surface_elevation_m=np.load(output / "reconciled_30m_water_surface_elevation_m.npy")[y0:y0 + size, x0:x0 + size],
        strict_outlet_floor=False,
    )
    geometry = build_fine_hydrology_geometry(hydro_window, profile, terrain_elevation_30m_m=parent_np[y0:y0 + size, x0:x0 + size])
    profile_candidate, profile_correction = enforce_profile_on_refined_terrain(generated10[0, 0].cpu().numpy(), profile, refinement=3)
    final10 = base10 + project_zero_block_mean_3x(torch.from_numpy(profile_candidate)[None, None].to(device) - base10)
    final_error = float((block_mean_3x3(final10) - parent_window).abs().max().cpu())
    if final_error > parent_tolerance:
        raise RuntimeError(f"Final hydrology-closed parent preservation failed: {final_error:.6g} m")

    raw_np, residual_np = raw_normalized[0, 0].cpu().numpy(), residual_m[0, 0].cpu().numpy()
    final_np = final10[0, 0].cpu().numpy().astype(np.float32)
    np.save(output / "one_pass_10m_raw_residual_normalized.npy", raw_np.astype(np.float32))
    np.save(output / "one_pass_10m_projected_residual_m.npy", residual_np.astype(np.float32))
    np.save(output / "one_pass_10m_before_hydrology_m.npy", generated10[0, 0].cpu().numpy().astype(np.float32))
    np.save(output / "one_pass_10m_profile_reasserted_m.npy", profile_candidate.astype(np.float32))
    np.save(output / "one_pass_10m_profile_reassertion_correction_m.npy", profile_correction.astype(np.float32))
    np.save(output / "final_10m_dem_m.npy", final_np)
    for name in ("conditioning", "profile_conditioning", "channel_centerline_mask", "channel_coverage", "target_bed_elevation_m", "lake_mask", "lake_coverage", "water_surface_elevation_m", "river_width_m"):
        np.save(output / f"final_10m_{name}.npy", np.asarray(getattr(geometry, name)))
    np.save(output / "final_30m_parent_dem_m.npy", parent_window[0, 0].cpu().numpy().astype(np.float32))
    shade30 = _save_hillshade(output / "final_30m_hillshade.png", parent_window[0, 0].cpu().numpy(), 30.0)
    shade10 = _save_hillshade(output / "final_10m_hillshade.png", final_np, 10.0)
    _save_overlay(output / "final_10m_hydrology_overlay.png", shade10, geometry.channel_centerline_mask, geometry.lake_mask)
    _save_overlay(output / "final_30m_hydrology_overlay.png", shade30, profile.channel_mask, hydro_window[7] >= 0.5)

    report = {
        "upstream_seconds": upstream_seconds, "one_pass_10m_seconds": ten_seconds,
        "dimensions": {"macro_m": 7680, "lowfreq_m": 240, "parent_m": 30, "final_m": 10, "parent_shape": [size, size], "final_shape": [size * 3, size * 3]},
        "one_pass_checkpoint": str(one_pass_checkpoint.resolve()), "residual_std_m": RESIDUAL_STD_M,
        "upstream_mode": upstream_mode, "tile_size_30m": tile_size_30m, "tile_overlap_30m": tile_overlap_30m, "tile_count": len(ys) * len(xs),
        "window_origin_30m": [y0, x0], "maximum_parent_error_before_hydrology_m": max_parent_error, "maximum_parent_error_final_m": final_error, "parent_preservation_tolerance_m": parent_tolerance,
        "final_10m_statistics_m": {"min": float(final_np.min()), "max": float(final_np.max()), "mean": float(final_np.mean())},
        "hydrology": {"channel_centerline_cells": int(geometry.channel_centerline_mask.sum()), "lake_cells": int(geometry.lake_mask.sum()), "profile_reassertion_max_abs_m": float(np.abs(profile_correction).max())},
        "hydrology_closure": "Profile reassertion is retained as an auditable candidate; final Q3 residual closure is applied afterwards so A3(final_10m)==authoritative_30m.",
    }
    (output / "full_10m_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    click.echo(json.dumps(report, indent=2))
