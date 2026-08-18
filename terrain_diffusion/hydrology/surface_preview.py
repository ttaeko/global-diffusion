"""Cartographic overview of a completed integrated planner surface."""

from __future__ import annotations

from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm
import numpy as np


def render_integrated_surface_preview(
    surface_file: str | Path,
    output_file: str | Path,
    *,
    overview_size: int = 1024,
) -> Path:
    path = Path(surface_file)
    output = Path(output_file)
    with h5py.File(path, "r") as surface:
        if not bool(surface.attrs.get("complete", False)):
            raise ValueError("Surface export is incomplete")
        dataset = surface["elevation_m"]
        if dataset.shape[0] != dataset.shape[1] or dataset.shape[0] % overview_size:
            raise ValueError("overview_size must evenly divide the square surface")
        elevation = dataset[...]
    factor = elevation.shape[0] // overview_size
    overview = elevation.reshape(
        overview_size, factor, overview_size, factor
    ).mean(axis=(1, 3)).astype(np.float32)
    del elevation
    light = LightSource(azdeg=315, altdeg=35)
    relief = light.shade(
        np.clip(overview, -1000, 4500),
        cmap=plt.get_cmap("terrain"),
        vert_exag=0.8,
        blend_mode="overlay",
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    axes[0].imshow(
        overview,
        cmap="terrain",
        norm=TwoSlopeNorm(vmin=-3000, vcenter=0, vmax=4500),
    )
    axes[0].contour(overview, [0], colors="black", linewidths=0.25)
    axes[0].set_title("Elevation and coastline")
    axes[1].imshow(relief)
    axes[1].set_title("Shaded relief")
    for axis in axes:
        axis.axis("off")
    figure.suptitle("Integrated macro + 240 m base terrain — 1966 km field")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


@click.command("preview-integrated-surface")
@click.argument("surface_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option("--overview-size", default=1024, show_default=True, type=click.IntRange(min=128))
def preview_integrated_surface_cli(surface_file, output_file, overview_size):
    """Render a cartographic overview of a completed 240 m export."""

    output = render_integrated_surface_preview(
        surface_file, output_file, overview_size=overview_size
    )
    click.echo(f"Saved integrated surface preview to {output}")
