"""Visual diagnostic for selective depression breaching."""

from __future__ import annotations

from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np

from .hybrid_conditioning import hybrid_fill_breach_route


def preview_hybrid_conditioning(
    surface_file: str | Path,
    output_file: str | Path,
    *,
    row_start: int = 0,
    col_start: int = 0,
    size: int = 2048,
) -> Path:
    with h5py.File(surface_file, "r") as surface:
        elevation = surface["elevation_m"][
            row_start:row_start + size, col_start:col_start + size
        ]
    if elevation.shape != (size, size):
        raise ValueError("Requested preview crop lies outside the surface")
    land = elevation > 0
    result = hybrid_fill_breach_route(
        elevation, resolution_m=240.0, land_mask=land, passes=2
    )
    incision = elevation - result.elevation_breached_m
    accumulation = np.log10(
        np.maximum(result.routing.accumulation_area_m2 / 1e6, 240.0**2 / 1e6)
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    axes[0, 0].imshow(elevation, cmap="terrain", vmin=-500, vmax=3500)
    axes[0, 0].set_title("Raw elevation")
    axes[0, 1].imshow(incision, cmap="magma", vmin=0, vmax=500)
    axes[0, 1].set_title("Selective breach incision (m)")
    axes[1, 0].imshow(
        result.routing.elevation_correction_m, cmap="magma", vmin=0, vmax=100
    )
    axes[1, 0].set_title("Residual fill correction (m)")
    axes[1, 1].imshow(accumulation, cmap="viridis", vmin=0, vmax=5)
    axes[1, 1].set_title("Log10 accumulation (km2)")
    for axis in axes.ravel():
        axis.axis("off")
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


@click.command("preview-hybrid-conditioning")
@click.argument("surface_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option("--row-start", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--col-start", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--size", default=2048, show_default=True, type=click.IntRange(min=128))
def preview_hybrid_conditioning_cli(**kwargs):
    """Render a real-surface selective-breaching diagnostic crop."""

    output = preview_hybrid_conditioning(**kwargs)
    click.echo(f"Saved hybrid conditioning preview to {output}")
