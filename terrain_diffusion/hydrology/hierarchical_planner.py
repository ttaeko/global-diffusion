"""Hierarchical 240 m hydrology over a learned regional terrain surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import click
import h5py
import numpy as np

from .atlas import HydrologyAtlas, RegionKey
from .compiled_routing import strahler_order_d8
from .conditioning import build_hydrology_conditioning
from .hybrid_conditioning import hybrid_fill_breach_route
from .lakes import identify_depression_lakes
from .macro_constraints import (
    build_hierarchical_routing_zones,
    materialize_macro_boundary_conditions,
)
from .routing import select_channels
from .runoff import mean_discharge_from_runoff
from .profile_contract import DEFAULT_HYDROLOGY_PROFILE
from .world_plan import D8_DIRECTION_OFFSETS


@dataclass(frozen=True)
class DivideCrossingMetrics:
    crossing_cells: int
    major_crossing_cells: int
    major_crossings_above_100_km2: int
    major_crossings_above_1000_km2: int
    major_crossings_above_10000_km2: int
    maximum_major_crossing_area_km2: float


@dataclass(frozen=True)
class ReconciledBasinProjection:
    basin_code: np.ndarray
    continental_divide_mask: np.ndarray
    inflow_contract_conflicts: int


def diagnose_divide_crossings(
    flow_direction: np.ndarray,
    accumulation_area_m2: np.ndarray,
    macro_basin_code: np.ndarray,
    major_basin_codes: set[int] | frozenset[int],
) -> DivideCrossingMetrics:
    """Measure detailed rivers crossing the snapped macro catchment prior."""

    flow = np.asarray(flow_direction, dtype=np.uint8)
    accumulation = np.asarray(accumulation_area_m2, dtype=np.float64)
    basins = np.asarray(macro_basin_code, dtype=np.uint32)
    if not (flow.shape == accumulation.shape == basins.shape):
        raise ValueError("Divide diagnostic rasters must align")
    crossing = np.zeros(flow.shape, dtype=bool)
    major_crossing = np.zeros(flow.shape, dtype=bool)
    major_codes = np.asarray(sorted(major_basin_codes), dtype=np.uint32)
    for code, (delta_row, delta_col) in D8_DIRECTION_OFFSETS.items():
        rows, cols = np.nonzero(flow == code)
        target_rows = rows + delta_row
        target_cols = cols + delta_col
        changed = basins[rows, cols] != basins[target_rows, target_cols]
        crossing[rows[changed], cols[changed]] = True
        if major_codes.size:
            is_major = np.isin(basins[rows, cols], major_codes) | np.isin(
                basins[target_rows, target_cols], major_codes
            )
            selected = changed & is_major
            major_crossing[rows[selected], cols[selected]] = True
    area_km2 = accumulation / 1_000_000.0
    maximum = float(np.max(area_km2[major_crossing], initial=0.0))
    return DivideCrossingMetrics(
        crossing_cells=int(np.count_nonzero(crossing)),
        major_crossing_cells=int(np.count_nonzero(major_crossing)),
        major_crossings_above_100_km2=int(np.count_nonzero(major_crossing & (area_km2 >= 100))),
        major_crossings_above_1000_km2=int(np.count_nonzero(major_crossing & (area_km2 >= 1000))),
        major_crossings_above_10000_km2=int(np.count_nonzero(major_crossing & (area_km2 >= 10000))),
        maximum_major_crossing_area_km2=maximum,
    )


def reconcile_basin_projection(
    flow_direction: np.ndarray,
    catchment_id: np.ndarray,
    macro_basin_code: np.ndarray,
    initial_accumulation_area_m2: np.ndarray,
    major_basin_codes: set[int] | frozenset[int],
) -> ReconciledBasinProjection:
    """Move macro identities onto exact 240 m catchment boundaries.

    An outlet supplies the default identity of each detailed catchment. A
    cross-region inflow overrides it when necessary, with the largest inherited
    river winning if multiple macro contracts unexpectedly merge.
    """

    flow = np.asarray(flow_direction, dtype=np.uint8)
    catchments = np.asarray(catchment_id, dtype=np.uint32)
    prior = np.asarray(macro_basin_code, dtype=np.uint32)
    inherited = np.asarray(initial_accumulation_area_m2, dtype=np.float64)
    if not (flow.shape == catchments.shape == prior.shape == inherited.shape):
        raise ValueError("Basin projection rasters must align")
    nodata = np.iinfo(np.uint32).max
    valid = catchments != nodata
    if not np.any(valid):
        raise ValueError("Basin projection contains no routed catchments")
    count = int(catchments[valid].max()) + 1
    lookup = np.full(count, nodata, dtype=np.uint32)
    weights = np.full(count, -1.0, dtype=np.float64)
    outlet = valid & (flow == 0)
    lookup[catchments[outlet]] = prior[outlet]
    weights[catchments[outlet]] = 0.0
    inflow_rows, inflow_cols = np.nonzero(inherited > 0)
    conflicts = 0
    order = np.argsort(inherited[inflow_rows, inflow_cols], kind="stable")
    for position in order:
        row, col = int(inflow_rows[position]), int(inflow_cols[position])
        catchment = int(catchments[row, col])
        if catchment == nodata:
            continue
        code = prior[row, col]
        if lookup[catchment] != nodata and lookup[catchment] != code:
            conflicts += 1
        weight = inherited[row, col]
        if weight >= weights[catchment]:
            lookup[catchment] = code
            weights[catchment] = weight
    if np.any(lookup[catchments[valid]] == nodata):
        raise RuntimeError("A routed catchment has no basin identity anchor")
    projected = np.full(flow.shape, nodata, dtype=np.uint32)
    projected[valid] = lookup[catchments[valid]]
    boundary = np.zeros(flow.shape, dtype=bool)
    major = np.asarray(sorted(major_basin_codes), dtype=np.uint32)
    if major.size:
        different = projected[1:, :] != projected[:-1, :]
        relevant = np.isin(projected[1:, :], major) | np.isin(projected[:-1, :], major)
        edge = different & relevant
        boundary[1:, :] |= edge
        boundary[:-1, :] |= edge
        different = projected[:, 1:] != projected[:, :-1]
        relevant = np.isin(projected[:, 1:], major) | np.isin(projected[:, :-1], major)
        edge = different & relevant
        boundary[:, 1:] |= edge
        boundary[:, :-1] |= edge
    return ReconciledBasinProjection(projected, boundary, conflicts)


def build_hierarchical_hydrology_plan(
    atlas_directory: str | Path,
    surface_file: str | Path,
    constraint_file: str | Path,
    output_file: str | Path,
    *,
    region_row: int = 0,
    region_col: int = 0,
    minimum_major_basin_area_km2: float = 100_000.0,
    outlet_corridor_cells: int = 16,
    channel_minimum_area_km2: float = DEFAULT_HYDROLOGY_PROFILE.channel_minimum_area_km2,
    runoff_ratio: float = DEFAULT_HYDROLOGY_PROFILE.runoff_ratio,
    fill_tolerance_m: float = 10.0,
    breach_minimum_area_km2: float = 10.0,
    breach_minimum_depth_m: float = 50.0,
    maximum_breach_incision_m: float = 800.0,
    breach_passes: int = 2,
) -> dict:
    """Build and persist the accepted macro-prior/terrain-authoritative plan."""

    started = time.monotonic()
    atlas = HydrologyAtlas(atlas_directory)
    manifest = atlas.read_manifest()
    region = RegionKey(region_row, region_col)
    output = Path(output_file)
    if output.exists():
        raise FileExistsError(f"Hydrology plan already exists: {output}")
    with h5py.File(surface_file, "r") as surface:
        provenance = json.loads(surface.attrs["provenance_json"])
        if int(provenance["seed"]) != manifest.world_seed:
            raise ValueError("Surface seed differs from atlas seed")
        elevation = surface["elevation_m"][...].astype(np.float32, copy=False)
        precipitation = surface["annual_precipitation_mm"][...].astype(np.float32, copy=False)
    with h5py.File(constraint_file, "r") as artifact:
        macro_basin_code = artifact["basin_code_240m"][...]
        divide_relaxation = artifact["divide_relaxation_mask_240m"][...] == 1
    if not (elevation.shape == precipitation.shape == macro_basin_code.shape):
        raise ValueError("Surface and constraint rasters do not align")
    land = np.isfinite(elevation) & (elevation > 0)
    hierarchy = build_hierarchical_routing_zones(
        atlas_directory,
        macro_basin_code,
        minimum_major_basin_area_km2=minimum_major_basin_area_km2,
    )
    boundary = materialize_macro_boundary_conditions(
        constraint_file,
        region=region,
        major_basin_codes=hierarchy.major_basin_codes,
        outlet_corridor_cells=outlet_corridor_cells,
    )

    # The macro basin raster is deliberately not passed as routing_zones. It is
    # an authoritative topology prior at 7.68 km, while the 240 m terrain is
    # authoritative for the exact location of divides and minor catchments.
    from .compiled_routing import priority_flood_route_compiled

    initial = priority_flood_route_compiled(
        elevation,
        resolution_m=manifest.planner_resolution_m,
        land_mask=land,
        terminal_mask=boundary.terminal_mask,
        open_boundary=False,
        initial_accumulation_area_m2=boundary.initial_accumulation_area_m2,
    )
    lakes = identify_depression_lakes(
        elevation,
        initial.elevation_conditioned_m,
        resolution_m=manifest.planner_resolution_m,
        land_mask=land,
        maximum_total_lake_fraction=0.005,
    )
    hybrid = hybrid_fill_breach_route(
        elevation,
        resolution_m=manifest.planner_resolution_m,
        land_mask=land,
        fill_tolerance_m=fill_tolerance_m,
        breach_minimum_area_km2=breach_minimum_area_km2,
        breach_minimum_depth_m=breach_minimum_depth_m,
        preserve_mask=lakes.lake_mask,
        maximum_breach_incision_m=maximum_breach_incision_m,
        terminal_mask=boundary.terminal_mask,
        initial_accumulation_area_m2=boundary.initial_accumulation_area_m2,
        passes=breach_passes,
    )
    routing = hybrid.routing
    precipitation = precipitation.copy()
    precipitation[~land | ~np.isfinite(precipitation)] = 0
    discharge = mean_discharge_from_runoff(
        routing.flow_direction,
        routing.processing_order,
        precipitation,
        resolution_m=manifest.planner_resolution_m,
        runoff_ratio=runoff_ratio,
        initial_discharge_m3s=boundary.initial_discharge_m3s,
    )
    channels = select_channels(
        routing.accumulation_area_m2,
        minimum_area_km2=channel_minimum_area_km2,
        land_mask=land,
    )
    stream_order = strahler_order_d8(
        routing.flow_direction, routing.processing_order, channels
    )
    divide_metrics = diagnose_divide_crossings(
        routing.flow_direction,
        routing.accumulation_area_m2,
        macro_basin_code,
        hierarchy.major_basin_codes,
    )
    reconciled = reconcile_basin_projection(
        routing.flow_direction,
        routing.catchment_id,
        macro_basin_code,
        boundary.initial_accumulation_area_m2,
        hierarchy.major_basin_codes,
    )
    correction = routing.elevation_correction_m[land & ~lakes.lake_mask]
    report = {
        "elapsed_seconds": float(time.monotonic() - started),
        "major_basin_area_threshold_km2": float(minimum_major_basin_area_km2),
        "major_basin_count": len(hierarchy.major_basin_codes),
        "enabled_portal_count": boundary.portal_count,
        "inflow_portal_count": boundary.inflow_count,
        "outlet_portal_count": boundary.outlet_count,
        "lake_count": len(lakes.records),
        "lake_fraction_of_land": float(lakes.lake_mask.sum() / land.sum()),
        "breach_fraction_of_land": float(hybrid.breach_mask.sum() / land.sum()),
        "fill_fraction_above_10m": float(np.mean(correction > 10)),
        "fill_fraction_above_50m": float(np.mean(correction > 50)),
        "fill_fraction_above_100m": float(np.mean(correction > 100)),
        "maximum_total_incision_m": float(
            np.max(elevation - hybrid.elevation_breached_m, initial=0.0)
        ),
        "input_drainage_area_km2": float(
            (land.sum() * manifest.planner_resolution_m**2
             + boundary.initial_accumulation_area_m2.sum()) / 1_000_000.0
        ),
        "terminal_drainage_area_km2": float(
            routing.accumulation_area_m2[routing.flow_direction == 0].sum()
            / 1_000_000.0
        ),
        "divide_crossings": asdict(divide_metrics),
        "inflow_contract_basin_conflicts": reconciled.inflow_contract_conflicts,
        "breach_passes": [asdict(item) for item in hybrid.metrics],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with h5py.File(temporary, "w") as artifact:
        artifact.attrs["schema_version"] = 1
        artifact.attrs["complete"] = True
        artifact.attrs["world_seed"] = manifest.world_seed
        artifact.attrs["region_row"] = region.row
        artifact.attrs["region_col"] = region.col
        artifact.attrs["resolution_m"] = manifest.planner_resolution_m
        artifact.attrs["surface_sha256"] = _sha256(Path(surface_file))
        artifact.attrs["constraints_sha256"] = _sha256(Path(constraint_file))
        artifact.attrs["report_json"] = json.dumps(report, sort_keys=True)
        artifact.attrs["major_basin_codes_json"] = json.dumps(
            sorted(hierarchy.major_basin_codes)
        )
        _write(artifact, "elevation_generated_m", elevation)
        _write(artifact, "elevation_breached_m", hybrid.elevation_breached_m)
        _write(artifact, "elevation_conditioned_m", routing.elevation_conditioned_m)
        _write(artifact, "water_surface_elevation_m", lakes.water_surface_elevation_m)
        _write(artifact, "annual_precipitation_mm", precipitation)
        _write(artifact, "land_mask", land.astype(np.uint8))
        _write(artifact, "lake_id", lakes.lake_id)
        _write(artifact, "breach_mask", hybrid.breach_mask.astype(np.uint8))
        _write(artifact, "flow_direction", routing.flow_direction)
        _write(artifact, "accumulation_area_m2", routing.accumulation_area_m2)
        _write(artifact, "mean_discharge_m3s", discharge)
        _write(artifact, "catchment_id", routing.catchment_id)
        _write(artifact, "stream_order", stream_order)
        _write(artifact, "channel_mask", channels.astype(np.uint8))
        _write(artifact, "macro_basin_code", macro_basin_code)
        _write(artifact, "basin_code", reconciled.basin_code)
        _write(artifact, "macro_divide_relaxation_mask", divide_relaxation.astype(np.uint8))
        _write(artifact, "continental_divide_prior", hierarchy.protected_divide_mask.astype(np.uint8))
        _write(artifact, "continental_divide_mask", reconciled.continental_divide_mask.astype(np.uint8))
        _write(artifact, "boundary_terminal_mask", boundary.terminal_mask.astype(np.uint8))
        lake_group = artifact.create_group("lakes")
        for name in ("lake_id", "cell_count", "area_m2", "maximum_depth_m", "mean_depth_m", "volume_m3", "surface_elevation_m"):
            lake_group.create_dataset(name, data=np.asarray([getattr(item, name) for item in lakes.records]))
    temporary.replace(output)
    return report


def _write(artifact: h5py.File, name: str, values: np.ndarray) -> None:
    artifact.create_dataset(name, data=values, chunks=(256, 256), compression="lzf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@click.command("build-hierarchical-hydrology-plan")
@click.argument("atlas_directory", type=click.Path(exists=True, file_okay=False))
@click.argument("surface_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("constraint_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option("--region-row", default=0, show_default=True, type=int)
@click.option("--region-col", default=0, show_default=True, type=int)
@click.option("--minimum-major-basin-area-km2", default=100000.0, show_default=True)
@click.option("--outlet-corridor-cells", default=16, show_default=True, type=click.IntRange(min=0))
@click.option("--breach-passes", default=2, show_default=True, type=click.IntRange(min=1))
def build_hierarchical_hydrology_plan_cli(**kwargs):
    """Build corrected 240 m terrain and persistent regional drainage maps."""

    try:
        report = build_hierarchical_hydrology_plan(**kwargs)
    except (FileExistsError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(report, indent=2, sort_keys=True))
