"""Cartographic preview of a persistent hierarchical hydrology plan."""

from __future__ import annotations

from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np


def preview_hierarchical_hydrology_plan(
    plan_file: str | Path,
    output_file: str | Path,
    *,
    downsample: int = 8,
) -> Path:
    """Render terrain, drainage, lakes, and reconciled continental divides."""

    with h5py.File(plan_file, "r") as artifact:
        elevation = artifact["elevation_breached_m"][...]
        land = artifact["land_mask"][...] == 1
        lake = artifact["lake_id"][...] != np.iinfo(np.uint32).max
        accumulation = artifact["accumulation_area_m2"][...] / 1_000_000.0
        divide = artifact["continental_divide_mask"][...] == 1
        resolution_m = float(artifact.attrs["resolution_m"])
    if elevation.shape[0] % downsample or elevation.shape[1] % downsample:
        raise ValueError("Plan dimensions must be divisible by downsample")
    terrain = _block_mean(elevation, downsample)
    land_small = _block_any(land, downsample)
    lake_small = _block_any(lake, downsample)
    divide_small = _block_any(divide, downsample)
    accumulation_small = _block_max(accumulation, downsample)
    terrain[~land_small] = np.nan

    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    extent_km = [
        0,
        elevation.shape[1] * resolution_m / 1000,
        elevation.shape[0] * resolution_m / 1000,
        0,
    ]
    axes[0].set_facecolor("#173b57")
    image = axes[0].imshow(terrain, cmap="terrain", extent=extent_km)
    river = accumulation_small >= 11.3
    river_rgba = np.zeros((*river.shape, 4), dtype=np.float32)
    river_rgba[river] = (0.05, 0.45, 0.95, 0.9)
    river_rgba[lake_small] = (0.02, 0.65, 1.0, 1.0)
    axes[0].imshow(river_rgba, extent=extent_km)
    divide_rgba = np.zeros((*divide_small.shape, 4), dtype=np.float32)
    divide_rgba[divide_small] = (0.95, 0.1, 0.1, 0.75)
    axes[0].imshow(divide_rgba, extent=extent_km)
    axes[0].set_title("Corrected terrain · rivers/lakes (blue) · continental divides (red)")
    figure.colorbar(image, ax=axes[0], shrink=0.72, label="elevation (m)")

    log_area = np.log10(np.maximum(accumulation_small, 0.01))
    log_area[~land_small] = np.nan
    drainage = axes[1].imshow(log_area, cmap="Blues", extent=extent_km, vmin=-1, vmax=5.5)
    axes[1].imshow(divide_rgba, extent=extent_km)
    axes[1].set_title("Drainage accumulation and reconciled continental divides")
    figure.colorbar(drainage, ax=axes[1], shrink=0.72, label="log10 upstream area (km²)")
    for axis in axes:
        axis.set_xlabel("west–east distance (km)")
        axis.set_ylabel("north–south distance (km)")
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def _reshape(values: np.ndarray, factor: int) -> np.ndarray:
    rows, cols = values.shape
    return values.reshape(rows // factor, factor, cols // factor, factor)


def _block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    return np.nanmean(_reshape(values, factor), axis=(1, 3))


def _block_max(values: np.ndarray, factor: int) -> np.ndarray:
    return np.nanmax(_reshape(values, factor), axis=(1, 3))


def _block_any(values: np.ndarray, factor: int) -> np.ndarray:
    return np.any(_reshape(values, factor), axis=(1, 3))


@click.command("preview-hierarchical-hydrology-plan")
@click.argument("plan_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option("--downsample", default=8, show_default=True, type=click.IntRange(min=1))
def preview_hierarchical_hydrology_plan_cli(**kwargs):
    """Render a cartographic PNG from a hierarchical plan."""

    output = preview_hierarchical_hydrology_plan(**kwargs)
    click.echo(f"Saved {output}")
