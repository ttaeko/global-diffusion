"""Preview FOEN-calibrated precipitation, discharge, profiles, and basins."""

from __future__ import annotations

from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np


def preview_foen_conditioning(
    plan_file: str | Path,
    conditioning_file: str | Path,
    output_file: str | Path,
    *,
    downsample: int = 8,
) -> Path:
    with h5py.File(plan_file, "r") as plan:
        elevation = plan["elevation_breached_m"][...]
        land = plan["land_mask"][...] == 1
    with h5py.File(conditioning_file, "r") as conditioning:
        precipitation = conditioning["annual_precipitation_calibrated_mm"][...]
        discharge = conditioning["mean_discharge_calibrated_m3s"][...]
        channels = conditioning["channel_mask"][...] == 1
        incision = conditioning["profile_incision_m"][...]
        level40 = conditioning["subbasin_level_40"][...]
        level150 = conditioning["subbasin_level_150"][...]
        level1000 = conditioning["subbasin_level_1000"][...]
        resolution_m = float(conditioning.attrs["resolution_m"])
    extent = [0, elevation.shape[1] * resolution_m / 1000,
              elevation.shape[0] * resolution_m / 1000, 0]
    land_small = _block_any(land, downsample)
    precipitation_small = _block_mean(precipitation, downsample)
    precipitation_small[~land_small] = np.nan
    channel_small = _block_any(channels, downsample)
    discharge_small = _block_max(np.where(channels, discharge, 0), downsample)
    incision_small = _block_max(incision, downsample)
    incision_small[~channel_small] = np.nan
    terrain_small = _block_mean(elevation, downsample)
    terrain_small[~land_small] = np.nan
    divide40 = _block_any(_boundaries(level40, land), downsample)
    divide150 = _block_any(_boundaries(level150, land), downsample)
    divide1000 = _block_any(_boundaries(level1000, land), downsample)

    figure, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    axes[0, 0].set_facecolor("#173b57")
    image = axes[0, 0].imshow(precipitation_small, cmap="Blues", extent=extent)
    axes[0, 0].set_title("Swiss-baseline annual precipitation")
    figure.colorbar(image, ax=axes[0, 0], shrink=0.75, label="mm/year")

    log_q = np.log10(np.maximum(discharge_small, 1e-3))
    log_q[~channel_small] = np.nan
    image = axes[0, 1].imshow(log_q, cmap="viridis", extent=extent, vmin=-2, vmax=3)
    axes[0, 1].set_title("Calibrated mean river discharge")
    figure.colorbar(image, ax=axes[0, 1], shrink=0.75, label="log10 m³/s")

    axes[1, 0].set_facecolor("#173b57")
    axes[1, 0].imshow(terrain_small, cmap="terrain", extent=extent)
    image = axes[1, 0].imshow(incision_small, cmap="magma", extent=extent, vmin=0, vmax=120)
    axes[1, 0].set_title("10 m longitudinal-profile incision target")
    figure.colorbar(image, ax=axes[1, 0], shrink=0.75, label="source-scale metres")

    axes[1, 1].set_facecolor("#173b57")
    axes[1, 1].imshow(terrain_small, cmap="terrain", extent=extent)
    overlay = np.zeros((*land_small.shape, 4), dtype=np.float32)
    overlay[divide40] = (0.1, 0.9, 1.0, 0.35)
    overlay[divide150] = (1.0, 0.85, 0.05, 0.65)
    overlay[divide1000] = (0.95, 0.05, 0.1, 0.95)
    axes[1, 1].imshow(overlay, extent=extent)
    axes[1, 1].set_title("Nested basins: level 40 cyan · 150 yellow · 1000 red")
    for axis in axes.flat:
        axis.set_xlabel("west–east distance (km)")
        axis.set_ylabel("north–south distance (km)")
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def _reshape(values: np.ndarray, factor: int) -> np.ndarray:
    rows, cols = values.shape
    if rows % factor or cols % factor:
        raise ValueError("Raster dimensions must be divisible by downsample")
    return values.reshape(rows // factor, factor, cols // factor, factor)


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    return np.nanmean(_reshape(values, factor), axis=(1, 3))


def _block_max(values: np.ndarray, factor: int) -> np.ndarray:
    return np.nanmax(_reshape(values, factor), axis=(1, 3))


def _block_any(values: np.ndarray, factor: int) -> np.ndarray:
    return np.any(_reshape(values, factor), axis=(1, 3))


def _boundaries(labels: np.ndarray, land: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    edge = land[1:] & land[:-1] & (labels[1:] != labels[:-1])
    boundary[1:] |= edge
    boundary[:-1] |= edge
    edge = land[:, 1:] & land[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    boundary[:, 1:] |= edge
    boundary[:, :-1] |= edge
    return boundary


@click.command("preview-foen-conditioning")
@click.argument("plan_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("conditioning_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option("--downsample", default=8, show_default=True, type=click.IntRange(min=1))
def preview_foen_conditioning_cli(**kwargs):
    """Render calibrated climate, rivers, profiles, and basin hierarchy."""

    output = preview_foen_conditioning(**kwargs)
    click.echo(f"Saved {output}")
