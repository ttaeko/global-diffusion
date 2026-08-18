"""Sample exact learned macro fields for expandable atlas regions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import click
import h5py
import numpy as np

from terrain_diffusion.hydrology.atlas import HydrologyAtlas, RegionKey
from terrain_diffusion.hydrology.base_surface import (
    MACRO_ELEVATION_CHANNEL,
    MACRO_PRECIPITATION_CHANNEL,
    _configure_world,
    _model_identity,
    _resolve_device,
)
from terrain_diffusion.inference.world_pipeline import normalize_tensor
from terrain_diffusion.hydrology.macro_topology import (
    analyze_macro_basin_closure,
    analyze_macro_landmass_closure,
    route_closed_continental_basins,
)


def export_macro_atlas_ring(
    atlas_directory: str | Path,
    base_pipeline: str,
    macro_model: str,
    stats_file: str,
    *,
    pipeline_cache: str | Path,
    centre_row: int = 0,
    centre_col: int = 0,
    radius: int = 1,
    device: str | None = None,
    batch_size: int = 8,
    cache_size_bytes: int = 2 * 1024**3,
    macro_steps: int = 30,
    seed: int = 74,
    land_fraction: float = 0.60,
) -> tuple[Path, ...]:
    """Generate a square macro-region ring in one cache-sharing session."""

    if radius < 0:
        raise ValueError("radius cannot be negative")
    atlas = HydrologyAtlas(atlas_directory)
    manifest = atlas.read_manifest()
    if seed != manifest.world_seed:
        raise ValueError("Requested seed differs from atlas seed")
    chosen_device = _resolve_device(device)
    base_resolved, base_hash = _model_identity(base_pipeline)
    macro_resolved, macro_hash = _model_identity(macro_model)
    provenance = {
        "schema_version": 1,
        "world_seed": int(seed),
        "macro_steps": int(macro_steps),
        "macro_land_fraction": float(land_fraction),
        "device_backend": chosen_device,
        "base_pipeline": str(base_pipeline),
        "base_pipeline_resolved": base_resolved,
        "base_pipeline_sha256": base_hash,
        "macro_model": str(macro_model),
        "macro_model_resolved": macro_resolved,
        "macro_model_sha256": macro_hash,
        "stats_file": str(Path(stats_file).resolve()),
        "stats_file_sha256": _sha256(Path(stats_file)),
    }
    region_min_row = int(centre_row) - radius
    region_max_row = int(centre_row) + radius
    region_min_col = int(centre_col) - radius
    region_max_col = int(centre_col) + radius
    macro_row0 = region_min_row * 256
    macro_row1 = (region_max_row + 1) * 256
    macro_col0 = region_min_col * 256
    macro_col1 = (region_max_col + 1) * 256

    world = _configure_world(
        base_pipeline,
        macro_model,
        stats_file,
        seed=seed,
        batch_size=batch_size,
        cache_size_bytes=cache_size_bytes,
        macro_steps=macro_steps,
        land_fraction=land_fraction,
    )
    click.echo(
        f"Generating macro mosaic rows {region_min_row}:{region_max_row}, "
        f"cols {region_min_col}:{region_max_col} on {chosen_device}"
    )
    world.to(chosen_device).bind(
        str(pipeline_cache), compression="lzf", compression_opts=None
    )
    try:
        weighted = world.coarse[:, macro_row0:macro_row1, macro_col0:macro_col1]
        fields = normalize_tensor(weighted, dim=0).detach().cpu().numpy().astype(np.float32)
    finally:
        world.close()

    output_directory = atlas.root / "macro_regions"
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for region_row in range(region_min_row, region_max_row + 1):
        for region_col in range(region_min_col, region_max_col + 1):
            key = RegionKey(region_row, region_col)
            local_row = (region_row - region_min_row) * 256
            local_col = (region_col - region_min_col) * 256
            tile = fields[:, local_row:local_row + 256, local_col:local_col + 256]
            path = output_directory / _region_filename(key)
            tile_provenance = {
                **provenance,
                "region_row": region_row,
                "region_col": region_col,
                "macro_origin_row": key.macro_origin[0],
                "macro_origin_col": key.macro_origin[1],
            }
            expected = json.dumps(tile_provenance, sort_keys=True)
            if path.exists():
                with h5py.File(path, "r") as existing:
                    if existing.attrs.get("provenance_json") != expected:
                        raise ValueError(f"Existing macro region has different provenance: {path}")
                digest = _sha256(path)
                atlas.register_macro_region(key, path, digest)
                outputs.append(path)
                continue
            temporary = path.with_suffix(".partial.h5")
            with h5py.File(temporary, "w") as output:
                output.attrs["provenance_json"] = expected
                output.create_dataset("coarse_fields", data=tile, compression="lzf")
                macro_elevation = np.sign(tile[MACRO_ELEVATION_CHANNEL]) * np.square(
                    tile[MACRO_ELEVATION_CHANNEL]
                )
                output.create_dataset(
                    "elevation_m", data=macro_elevation.astype(np.float32), compression="lzf"
                )
                output.create_dataset(
                    "annual_precipitation_mm",
                    data=np.maximum(tile[MACRO_PRECIPITATION_CHANNEL], 0).astype(np.float32),
                    compression="lzf",
                )
            temporary.replace(path)
            digest = _sha256(path)
            atlas.register_macro_region(key, path, digest)
            outputs.append(path)
            click.echo(f"Stored macro region ({region_row}, {region_col}): {path}")
    return tuple(outputs)


def _region_filename(key: RegionKey) -> str:
    return f"r{key.row:+06d}_c{key.col:+06d}.h5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_macro_atlas_closure(
    atlas_directory: str | Path,
    *,
    centre_row: int = 0,
    centre_col: int = 0,
    radius: int = 1,
    freeze_closed: bool = True,
) -> dict:
    """Stitch a rectangular macro mosaic and close central landmasses."""

    if radius < 0:
        raise ValueError("radius cannot be negative")
    atlas = HydrologyAtlas(atlas_directory)
    manifest = atlas.read_manifest()
    minimum_row, maximum_row = centre_row - radius, centre_row + radius
    minimum_col, maximum_col = centre_col - radius, centre_col + radius
    records: dict[tuple[int, int], str] = {}
    with atlas.open_catalog() as connection:
        for row, col, path in connection.execute(
            """SELECT region_row, region_col, artifact_path FROM macro_regions
               WHERE region_row BETWEEN ? AND ? AND region_col BETWEEN ? AND ?""",
            (minimum_row, maximum_row, minimum_col, maximum_col),
        ):
            records[(int(row), int(col))] = path
    expected = {
        (row, col)
        for row in range(minimum_row, maximum_row + 1)
        for col in range(minimum_col, maximum_col + 1)
    }
    missing = sorted(expected - records.keys())
    if missing:
        raise ValueError(f"Macro mosaic is incomplete; missing regions: {missing}")
    side = (radius * 2 + 1) * 256
    elevation = np.empty((side, side), dtype=np.float32)
    for (region_row, region_col), path in records.items():
        local_row = (region_row - minimum_row) * 256
        local_col = (region_col - minimum_col) * 256
        with h5py.File(path, "r") as artifact:
            elevation[local_row:local_row + 256, local_col:local_col + 256] = artifact[
                "elevation_m"
            ][...]
    focus = np.zeros(elevation.shape, dtype=bool)
    focus_row = (centre_row - minimum_row) * 256
    focus_col = (centre_col - minimum_col) * 256
    focus[focus_row:focus_row + 256, focus_col:focus_col + 256] = True
    macro_origin_row = minimum_row * 256
    macro_origin_col = minimum_col * 256
    analysis = analyze_macro_landmass_closure(
        elevation,
        world_seed=manifest.world_seed,
        macro_origin_row=macro_origin_row,
        macro_origin_col=macro_origin_col,
        focus_mask=focus,
    )
    frozen: list[str] = []
    if freeze_closed:
        with atlas.open_catalog() as connection:
            already_frozen = {
                row[0] for row in connection.execute("SELECT landmass_id FROM macro_landmasses")
            }
        for landmass in analysis.landmasses:
            if not landmass.closed or landmass.landmass_id in already_frozen:
                continue
            mask = analysis.labels == landmass.provisional_label
            drainage = route_closed_continental_basins(
                elevation,
                mask,
                world_seed=manifest.world_seed,
                macro_origin_row=macro_origin_row,
                macro_origin_col=macro_origin_col,
            )
            atlas.freeze_continental_drainage(
                drainage,
                macro_origin_row=macro_origin_row,
                macro_origin_col=macro_origin_col,
            )
            frozen.append(drainage.landmass_id)
    report = {
        "centre_region": [int(centre_row), int(centre_col)],
        "radius": int(radius),
        "mosaic_macro_shape": list(elevation.shape),
        "focus_landmass_count": len(analysis.landmasses),
        "closed_landmass_count": sum(item.closed for item in analysis.landmasses),
        "open_landmass_count": sum(not item.closed for item in analysis.landmasses),
        "newly_frozen_landmass_ids": frozen,
        "required_regions": [[key.row, key.col] for key in analysis.required_regions],
        "landmasses": [
            {
                "landmass_id": item.landmass_id,
                "cell_count": item.cell_count,
                "closed": item.closed,
                "bounds_macro": [
                    item.min_macro_row, item.min_macro_col,
                    item.max_macro_row, item.max_macro_col,
                ],
                "required_regions": [[key.row, key.col] for key in item.required_regions],
            }
            for item in sorted(
                analysis.landmasses, key=lambda item: item.cell_count, reverse=True
            )
        ],
    }
    report_path = atlas.root / (
        f"macro_closure_r{centre_row:+05d}_c{centre_col:+05d}_radius{radius}.json"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def analyze_macro_atlas_basins(
    atlas_directory: str | Path,
    *,
    centre_row: int = 0,
    centre_col: int = 0,
    radius: int = 1,
    freeze_closed: bool = True,
) -> dict:
    """Freeze complete drainage basins intersecting a requested region."""

    atlas = HydrologyAtlas(atlas_directory)
    manifest = atlas.read_manifest()
    if radius < 0:
        raise ValueError("radius cannot be negative")
    minimum_row, maximum_row = centre_row - radius, centre_row + radius
    minimum_col, maximum_col = centre_col - radius, centre_col + radius
    records: dict[tuple[int, int], str] = {}
    with atlas.open_catalog() as connection:
        for row, col, path in connection.execute(
            """SELECT region_row, region_col, artifact_path FROM macro_regions
               WHERE region_row BETWEEN ? AND ? AND region_col BETWEEN ? AND ?""",
            (minimum_row, maximum_row, minimum_col, maximum_col),
        ):
            records[(int(row), int(col))] = path
    expected = {
        (row, col)
        for row in range(minimum_row, maximum_row + 1)
        for col in range(minimum_col, maximum_col + 1)
    }
    missing = sorted(expected - records.keys())
    if missing:
        raise ValueError(f"Macro mosaic is incomplete; missing regions: {missing}")
    side = (radius * 2 + 1) * 256
    elevation = np.empty((side, side), dtype=np.float32)
    for (region_row, region_col), path in records.items():
        local_row = (region_row - minimum_row) * 256
        local_col = (region_col - minimum_col) * 256
        with h5py.File(path, "r") as artifact:
            elevation[local_row:local_row + 256, local_col:local_col + 256] = artifact[
                "elevation_m"
            ][...]
    focus = np.zeros(elevation.shape, dtype=bool)
    focus_row = (centre_row - minimum_row) * 256
    focus_col = (centre_col - minimum_col) * 256
    focus[focus_row:focus_row + 256, focus_col:focus_col + 256] = True
    macro_origin_row = minimum_row * 256
    macro_origin_col = minimum_col * 256
    analysis = analyze_macro_basin_closure(
        elevation,
        world_seed=manifest.world_seed,
        macro_origin_row=macro_origin_row,
        macro_origin_col=macro_origin_col,
        focus_mask=focus,
    )
    newly_frozen: list[str] = []
    if freeze_closed:
        for basin in analysis.basins:
            if not basin.closed:
                continue
            atlas.freeze_macro_drainage_basin(
                basin,
                analysis.routing.catchment_id,
                analysis.land_mask,
                macro_origin_row=macro_origin_row,
                macro_origin_col=macro_origin_col,
            )
            newly_frozen.append(basin.basin_id)
    closed = [basin for basin in analysis.basins if basin.closed]
    opened = [basin for basin in analysis.basins if not basin.closed]
    report = {
        "centre_region": [int(centre_row), int(centre_col)],
        "radius": int(radius),
        "mosaic_macro_shape": list(elevation.shape),
        "focus_basin_count": len(analysis.basins),
        "closed_basin_count": len(closed),
        "open_basin_count": len(opened),
        "closed_focus_area_km2": float(sum(basin.area_km2 for basin in closed)),
        "open_focus_area_km2": float(sum(basin.area_km2 for basin in opened)),
        "newly_frozen_basin_count": len(newly_frozen),
        "required_regions": [[key.row, key.col] for key in analysis.required_regions],
        "largest_open_basins": [
            {
                "basin_id": basin.basin_id,
                "area_km2": basin.area_km2,
                "maximum_accumulation_km2": basin.maximum_accumulation_km2,
                "outlet_kind": basin.outlet_kind,
                "required_regions": [
                    [key.row, key.col] for key in basin.required_regions
                ],
            }
            for basin in sorted(opened, key=lambda item: item.area_km2, reverse=True)[:20]
        ],
    }
    report_path = atlas.root / (
        f"macro_basins_r{centre_row:+05d}_c{centre_col:+05d}_radius{radius}.json"
    )
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


@click.command("export-macro-atlas-ring")
@click.argument("atlas_directory", type=click.Path(exists=True, file_okay=False))
@click.argument("base_pipeline")
@click.argument("macro_model", type=click.Path(exists=True))
@click.option("--stats-file", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--pipeline-cache", required=True, type=click.Path(dir_okay=False))
@click.option("--centre-row", default=0, show_default=True, type=int)
@click.option("--centre-col", default=0, show_default=True, type=int)
@click.option("--radius", default=1, show_default=True, type=click.IntRange(min=0))
@click.option("--device", default=None)
@click.option("--batch-size", default=8, show_default=True, type=click.IntRange(min=1))
@click.option("--cache-size-bytes", default=2 * 1024**3, show_default=True, type=int)
@click.option("--macro-steps", default=30, show_default=True, type=click.IntRange(min=1))
@click.option("--seed", default=74, show_default=True, type=int)
@click.option("--land-fraction", default=0.60, show_default=True, type=click.FloatRange(0, 1))
def export_macro_atlas_ring_cli(**kwargs):
    """Generate and register a square ring of learned macro regions."""

    outputs = export_macro_atlas_ring(**kwargs)
    click.echo(f"Registered {len(outputs)} macro regions")


@click.command("analyze-macro-atlas-closure")
@click.argument("atlas_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--centre-row", default=0, show_default=True, type=int)
@click.option("--centre-col", default=0, show_default=True, type=int)
@click.option("--radius", default=1, show_default=True, type=click.IntRange(min=0))
@click.option("--freeze-closed/--no-freeze-closed", default=True, show_default=True)
def analyze_macro_atlas_closure_cli(**kwargs):
    """Analyze and freeze closed continents in a registered macro mosaic."""

    report = analyze_macro_atlas_closure(**kwargs)
    click.echo(json.dumps(report, indent=2))


@click.command("analyze-macro-atlas-basins")
@click.argument("atlas_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--centre-row", default=0, show_default=True, type=int)
@click.option("--centre-col", default=0, show_default=True, type=int)
@click.option("--radius", default=1, show_default=True, type=click.IntRange(min=0))
@click.option("--freeze-closed/--no-freeze-closed", default=True, show_default=True)
def analyze_macro_atlas_basins_cli(**kwargs):
    """Freeze complete ocean basins without requiring a finite continent."""

    report = analyze_macro_atlas_basins(**kwargs)
    click.echo(json.dumps(report, indent=2))
