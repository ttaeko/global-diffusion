"""Export the integrated pipeline's 240 m surface for global hydrology.

This is deliberately separate from the hydrology world-plan store.  Diffusion
inference can take many hours and is resumable tile by tile; routing is a fast,
deterministic second operation over the completed frozen surface.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import click
import h5py
import numpy as np
import scipy.ndimage
import torch

from terrain_diffusion.inference.world_pipeline import WorldPipeline, normalize_tensor
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.training.datasets.h5_macro_terrain_dataset import load_macro_stats
from terrain_diffusion.hydrology.profile_contract import DEFAULT_HYDROLOGY_PROFILE


MACRO_RESOLUTION_M = 7680.0
PLANNER_RESOLUTION_M = 240.0
MACRO_TO_PLANNER = 32
ELEVATION_LATENT_CHANNEL = 4
MACRO_ELEVATION_CHANNEL = 0
MACRO_PRECIPITATION_CHANNEL = 4
SIGNED_SQRT_MEAN = -31.4
SIGNED_SQRT_STD = 38.6
SURFACE_SCHEMA_VERSION = 1


def signed_sqrt_latent_to_metres(values: np.ndarray) -> np.ndarray:
    """Decode the stock base model's normalized elevation latent."""

    signed_sqrt = np.asarray(values, dtype=np.float32) * SIGNED_SQRT_STD + SIGNED_SQRT_MEAN
    return (np.sign(signed_sqrt) * np.square(signed_sqrt)).astype(np.float32)


def _sha256_path(path: str | Path) -> str:
    """Stable digest of a model file or directory, including relative names."""

    root = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for item in paths:
        digest.update(str(item.relative_to(root) if root.is_dir() else item.name).encode())
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _model_identity(model: str | Path) -> tuple[str, str]:
    """Resolve a local path or a locally cached Hugging Face model revision."""

    path = Path(model).expanduser()
    if path.exists():
        resolved = path.resolve()
    else:
        try:
            from huggingface_hub import snapshot_download

            resolved = Path(snapshot_download(str(model), local_files_only=True)).resolve()
        except Exception as error:
            raise FileNotFoundError(
                f"Cannot resolve {model!s} locally; download the base pipeline before export"
            ) from error
    return str(resolved), _sha256_path(resolved)


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def _surface_provenance(
    *,
    base_pipeline: str,
    macro_model: str,
    stats_file: str,
    seed: int,
    macro_cells: int,
    macro_steps: int,
    land_fraction: float,
    device_backend: str,
    batch_size: int,
) -> dict:
    base_resolved, base_digest = _model_identity(base_pipeline)
    macro_resolved, macro_digest = _model_identity(macro_model)
    return {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "seed": int(seed),
        "macro_cells": int(macro_cells),
        "planner_cells": int(macro_cells) * MACRO_TO_PLANNER,
        "macro_resolution_m": MACRO_RESOLUTION_M,
        "planner_resolution_m": PLANNER_RESOLUTION_M,
        "macro_steps": int(macro_steps),
        "macro_land_fraction": float(land_fraction),
        "macro_rng_schema": "world-pipeline-gaussian-patch-v1",
        "macro_rng_seed_offset": int(0x4D414352),
        "macro_tile_size": 256,
        "macro_tile_stride": 192,
        "coordinate_origin_30m": [0, 0],
        "device_backend": str(device_backend),
        "latent_batch_size": int(batch_size),
        "base_pipeline": str(base_pipeline),
        "base_pipeline_resolved": base_resolved,
        "base_pipeline_sha256": base_digest,
        "macro_model": str(macro_model),
        "macro_model_resolved": macro_resolved,
        "macro_model_sha256": macro_digest,
        "stats_file": str(Path(stats_file).resolve()),
        "stats_file_sha256": _sha256_path(stats_file),
        "elevation_source": "integrated WorldPipeline base latent channel 4",
        "precipitation_source": "integrated learned macro channel 4, bilinear at 240 m",
    }


def _open_surface(
    output_path: Path,
    provenance: dict,
    *,
    tile_size: int,
) -> h5py.File:
    planner_cells = int(provenance["planner_cells"])
    tile_rows = (planner_cells + tile_size - 1) // tile_size
    existed = output_path.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    surface = h5py.File(output_path, "a")
    expected = json.dumps(provenance, sort_keys=True)
    if existed:
        actual = surface.attrs.get("provenance_json")
        if actual != expected:
            surface.close()
            raise ValueError(
                "Output provenance differs from this export. Use the original settings "
                "to resume or choose a new output file."
            )
        if int(surface.attrs["tile_size"]) != tile_size:
            surface.close()
            raise ValueError("tile_size must match the interrupted export")
        return surface

    surface.attrs["provenance_json"] = expected
    surface.attrs["tile_size"] = int(tile_size)
    surface.attrs["complete"] = False
    surface.attrs["macro_complete"] = False
    chunks = (min(tile_size, planner_cells), min(tile_size, planner_cells))
    surface.create_dataset(
        "elevation_m", (planner_cells, planner_cells), dtype="f4",
        chunks=chunks, compression="lzf", fillvalue=np.nan,
    )
    surface.create_dataset(
        "annual_precipitation_mm", (planner_cells, planner_cells), dtype="f4",
        chunks=chunks, compression="lzf", fillvalue=np.nan,
    )
    macro = surface.create_group("macro")
    macro.create_dataset(
        "elevation_m", (provenance["macro_cells"], provenance["macro_cells"]), dtype="f4"
    )
    macro.create_dataset(
        "annual_precipitation_mm",
        (provenance["macro_cells"], provenance["macro_cells"]), dtype="f4",
    )
    surface.create_dataset("completed_tiles", (tile_rows, tile_rows), dtype="?", fillvalue=False)
    surface.flush()
    return surface


def _configure_world(
    base_pipeline: str,
    macro_model: str,
    stats_file: str,
    *,
    seed: int,
    batch_size: int,
    cache_size_bytes: int,
    macro_steps: int,
    land_fraction: float,
) -> WorldPipeline:
    means, stds = load_macro_stats(stats_file)
    world = WorldPipeline.from_pretrained(
        base_pipeline,
        seed=seed,
        latents_batch_size=batch_size,
        torch_compile=False,
        caching_strategy="indirect",
        cache_limit=cache_size_bytes,
        log_mode="info",
    )
    world.macro_model = EDMUnet2D.from_pretrained(macro_model)
    world.macro_enabled = True
    world.macro_steps = int(macro_steps)
    world.macro_land_fraction = float(land_fraction)
    world.macro_means = list(means)
    world.macro_stds = list(stds)
    world.kwargs.update(
        macro_enabled=True,
        macro_steps=int(macro_steps),
        macro_land_fraction=float(land_fraction),
        macro_means=list(means),
        macro_stds=list(stds),
    )
    return world


def export_global_planner_surface(
    base_pipeline: str,
    macro_model: str,
    stats_file: str,
    output_file: str | Path,
    *,
    pipeline_cache: str | Path,
    macro_cells: int = 256,
    tile_size: int = 256,
    device: str | None = None,
    batch_size: int = 8,
    cache_size_bytes: int = 2 * 1024**3,
    macro_steps: int = 30,
    seed: int = 74,
    land_fraction: float = 0.60,
    max_tiles: int | None = None,
) -> dict:
    """Freeze the learned macro fields and actual 240 m base elevation.

    ``max_tiles`` intentionally permits a bounded smoke run. Re-running with the
    same arguments resumes from ``completed_tiles`` and retains the pipeline's
    expensive internal tile cache.
    """

    if macro_cells <= 0 or tile_size <= 0 or tile_size % 32:
        raise ValueError("macro_cells must be positive and tile_size a positive multiple of 32")
    if max_tiles is not None and max_tiles <= 0:
        raise ValueError("max_tiles must be positive")
    output_path = Path(output_file)
    chosen_device = _resolve_device(device)
    provenance = _surface_provenance(
        base_pipeline=base_pipeline,
        macro_model=macro_model,
        stats_file=stats_file,
        seed=seed,
        macro_cells=macro_cells,
        macro_steps=macro_steps,
        land_fraction=land_fraction,
        device_backend=chosen_device,
        batch_size=batch_size,
    )
    surface = _open_surface(output_path, provenance, tile_size=tile_size)
    world = _configure_world(
        base_pipeline, macro_model, stats_file, seed=seed, batch_size=batch_size,
        cache_size_bytes=cache_size_bytes, macro_steps=macro_steps,
        land_fraction=land_fraction,
    )
    click.echo(f"Generating integrated surface on {chosen_device}")
    world.to(chosen_device).bind(str(pipeline_cache), compression="lzf", compression_opts=None)
    started = time.monotonic()
    generated = 0
    try:
        if not bool(surface.attrs.get("macro_complete", False)):
            # Querying this from the bound hierarchy guarantees climate and base
            # elevation belong to the same infinite, blended macro realization.
            macro_weighted = world.coarse[:, 0:macro_cells, 0:macro_cells]
            macro_fields = normalize_tensor(macro_weighted, dim=0).detach().cpu().numpy()
            macro_elevation = np.sign(macro_fields[MACRO_ELEVATION_CHANNEL]) * np.square(
                macro_fields[MACRO_ELEVATION_CHANNEL]
            )
            macro_precip = np.maximum(macro_fields[MACRO_PRECIPITATION_CHANNEL], 0.0)
            surface["macro/elevation_m"][...] = macro_elevation.astype(np.float32)
            surface["macro/annual_precipitation_mm"][...] = macro_precip.astype(np.float32)
            planner_precip = scipy.ndimage.zoom(
                macro_precip,
                MACRO_TO_PLANNER,
                order=1,
                mode="nearest",
                prefilter=False,
                grid_mode=True,
            ).astype(np.float32)
            surface["annual_precipitation_mm"][...] = planner_precip
            surface.attrs["macro_complete"] = True
            surface.flush()

        size = int(provenance["planner_cells"])
        completed = surface["completed_tiles"]
        for tile_i in range(completed.shape[0]):
            for tile_j in range(completed.shape[1]):
                if completed[tile_i, tile_j]:
                    continue
                if max_tiles is not None and generated >= max_tiles:
                    break
                i0, j0 = tile_i * tile_size, tile_j * tile_size
                i1, j1 = min(i0 + tile_size, size), min(j0 + tile_size, size)
                weighted = world.latents[:, i0:i1, j0:j1]
                latent = normalize_tensor(weighted, dim=0)[ELEVATION_LATENT_CHANNEL]
                elevation = signed_sqrt_latent_to_metres(
                    latent.detach().cpu().numpy()
                )
                surface["elevation_m"][i0:i1, j0:j1] = elevation
                completed[tile_i, tile_j] = True
                generated += 1
                surface.flush()
                elapsed = time.monotonic() - started
                total_done = int(np.count_nonzero(completed[...]))
                total_tiles = int(completed.size)
                click.echo(
                    f"planner tile {tile_i},{tile_j}: {total_done}/{total_tiles} "
                    f"({elapsed / generated:.1f} s/new tile)"
                )
            if max_tiles is not None and generated >= max_tiles:
                break

        is_complete = bool(np.all(completed[...]))
        surface.attrs["complete"] = is_complete
        surface.attrs["completed_tile_count"] = int(np.count_nonzero(completed[...]))
        surface.flush()
        return {
            "output_file": str(output_path),
            "pipeline_cache": str(pipeline_cache),
            "generated_tiles": generated,
            "completed_tiles": int(surface.attrs["completed_tile_count"]),
            "total_tiles": int(completed.size),
            "complete": is_complete,
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        world.close()
        surface.close()


def build_world_plan_from_planner_surface(
    surface_file: str | Path,
    output_directory: str | Path,
    *,
    channel_minimum_area_km2: float = DEFAULT_HYDROLOGY_PROFILE.channel_minimum_area_km2,
    runoff_ratio: float = DEFAULT_HYDROLOGY_PROFILE.runoff_ratio,
):
    """Build hydrology only after an integrated surface export is complete."""

    from terrain_diffusion.hydrology.macro_world import (
        _sha256,
        persist_world_plan_from_surfaces,
    )

    path = Path(surface_file)
    with h5py.File(path, "r") as surface:
        if not bool(surface.attrs.get("complete", False)):
            done = int(np.count_nonzero(surface["completed_tiles"][...]))
            total = int(surface["completed_tiles"].size)
            raise ValueError(
                f"Integrated surface is incomplete ({done}/{total} tiles); resume export first"
            )
        export_provenance = json.loads(surface.attrs["provenance_json"])
        elevation = surface["elevation_m"][...]
        precipitation = surface["annual_precipitation_mm"][...]
        macro_elevation = surface["macro/elevation_m"][...]
        macro_precipitation = surface["macro/annual_precipitation_mm"][...]
    provenance = {
        "world_plan_role": "integrated-base-routing-plan",
        "integrated_surface": str(path.resolve()),
        "integrated_surface_sha256": _sha256(path),
        "integrated_export": export_provenance,
        "hydrology_channel_minimum_area_km2": float(channel_minimum_area_km2),
        "hydrology_runoff_ratio": float(runoff_ratio),
    }
    return persist_world_plan_from_surfaces(
        output_directory,
        world_id=path.stem,
        seed=int(export_provenance["seed"]),
        planner_elevation_m=elevation,
        planner_precipitation_mm=precipitation,
        macro_elevation_m=macro_elevation,
        macro_precipitation_mm=macro_precipitation,
        provenance=provenance,
        channel_minimum_area_km2=channel_minimum_area_km2,
        runoff_ratio=runoff_ratio,
    )


@click.command("export-global-planner-surface")
@click.argument("base_pipeline")
@click.argument("macro_model", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option("--stats-file", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--pipeline-cache", required=True, type=click.Path(dir_okay=False))
@click.option("--macro-cells", default=256, show_default=True, type=click.IntRange(min=1))
@click.option("--tile-size", default=256, show_default=True, type=click.IntRange(min=32))
@click.option("--device", default=None)
@click.option("--batch-size", default=8, show_default=True, type=click.IntRange(min=1))
@click.option("--cache-size-bytes", default=2 * 1024**3, show_default=True, type=int)
@click.option("--macro-steps", default=30, show_default=True, type=click.IntRange(min=1))
@click.option("--seed", default=74, show_default=True, type=int)
@click.option("--land-fraction", default=0.60, show_default=True, type=click.FloatRange(0, 1))
@click.option("--max-tiles", default=None, type=click.IntRange(min=1))
def export_global_planner_surface_cli(**kwargs):
    """Export exact 240 m base terrain from the integrated learned pipeline."""

    result = export_global_planner_surface(**kwargs)
    click.echo(json.dumps(result, indent=2))


@click.command("build-hydrology-world-plan-from-surface")
@click.argument("surface_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_directory", type=click.Path(file_okay=False))
@click.option(
    "--channel-minimum-area-km2",
    default=DEFAULT_HYDROLOGY_PROFILE.channel_minimum_area_km2,
    show_default=True,
)
@click.option(
    "--runoff-ratio",
    default=DEFAULT_HYDROLOGY_PROFILE.runoff_ratio,
    show_default=True,
)
def build_world_plan_from_planner_surface_cli(**kwargs):
    """Route a completed integrated 240 m surface into a persistent plan."""

    store = build_world_plan_from_planner_surface(**kwargs)
    click.echo(f"Created integrated hydrology world plan at {store.root}")
