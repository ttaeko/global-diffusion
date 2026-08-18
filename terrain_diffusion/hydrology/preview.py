"""World-scale diagnostic rendering for persistent hydrology plans."""

from __future__ import annotations

from pathlib import Path

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np

from .world_plan import WorldPlanStore


def render_world_plan_preview(
    world_plan_directory: str | Path,
    output: str | Path,
    *,
    size: int = 1024,
) -> None:
    store = WorldPlanStore(world_plan_directory)
    with h5py.File(store.rasters_path, "r") as rasters:
        group = rasters["levels/global_240m"]
        height, width = group["elevation_final_m"].shape
        if height != width or height % size:
            raise ValueError("Preview size must divide the square global grid")
        factor = height // size
        elevation = group["elevation_final_m"][::factor, ::factor]
        accumulation = group["accumulation_area_m2"][::factor, ::factor]
        channels = _block_max(group["channel_mask"][:], factor) == 1
        lakes = _block_lake(group["lake_id"][:], factor)

    land = elevation > 0
    terrain = np.clip((elevation + 500.0) / 4500.0, 0, 1)
    image = plt.get_cmap("terrain")(terrain)[..., :3]
    image[~land] = np.asarray([0.05, 0.16, 0.32])
    image[channels] = np.asarray([0.05, 0.38, 0.95])
    image[lakes] = np.asarray([0.02, 0.55, 0.95])

    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Elevation, selected rivers, and lakes")
    log_accumulation = np.log10(np.maximum(accumulation / 1_000_000.0, 0.01))
    axes[1].imshow(log_accumulation, cmap="magma", vmin=-1, vmax=5)
    axes[1].set_title("Log10 contributing area (km²)")
    for axis in axes:
        axis.axis("off")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _block_max(values: np.ndarray, factor: int) -> np.ndarray:
    height, width = values.shape
    return values.reshape(
        height // factor, factor, width // factor, factor
    ).max(axis=(1, 3))


def _block_lake(values: np.ndarray, factor: int) -> np.ndarray:
    present = values != np.iinfo(np.uint32).max
    return _block_max(present.astype(np.uint8), factor).astype(bool)


@click.command("preview-hydrology-world-plan")
@click.argument("world_plan_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.option("--size", default=1024, show_default=True, type=click.IntRange(min=128))
def preview_hydrology_world_plan(world_plan_directory, output, size):
    """Render world-scale elevation, rivers, lakes, and accumulation."""

    render_world_plan_preview(world_plan_directory, output, size=size)
    click.echo(f"Saved hydrology preview to {output}")
