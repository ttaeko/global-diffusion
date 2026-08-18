"""Build FOEN-calibrated climate, river-profile, and basin conditioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import click
import h5py
from numba import njit
import numpy as np

from .atlas import RegionKey
from .calibration import _distribution
from .compiled_routing import (
    accumulate_values_d8,
    processing_order_from_d8,
)
from .macro_constraints import (
    build_hierarchical_routing_zones,
    materialize_macro_boundary_conditions,
)
from .runoff import SECONDS_PER_YEAR
from .profile_contract import DEFAULT_HYDROLOGY_PROFILE


_NODATA_U32 = np.iinfo(np.uint32).max
_DROW = np.asarray((0, 0, 1, 1, 1, 0, -1, -1, -1), dtype=np.int8)
_DCOL = np.asarray((0, 1, 1, 0, -1, -1, -1, 0, 1), dtype=np.int8)


@dataclass(frozen=True)
class PrecipitationTransform:
    scale: float
    offset_mm: float
    minimum_mm: float
    maximum_mm: float


def fit_simple_swiss_precipitation_transform(
    generated_distribution: dict,
    swiss_distribution: dict,
) -> PrecipitationTransform:
    """Fit one robust affine transform from p10/p90 and clip to Swiss p01/p99."""

    generated = generated_distribution["quantiles"]
    swiss = swiss_distribution["quantiles"]
    denominator = float(generated["p90"] - generated["p10"])
    if denominator <= 0:
        raise ValueError("Generated precipitation p10/p90 range must be positive")
    scale = float((swiss["p90"] - swiss["p10"]) / denominator)
    offset = float(swiss["p10"] - scale * generated["p10"])
    return PrecipitationTransform(
        scale=scale,
        offset_mm=offset,
        minimum_mm=float(swiss["p01"]),
        maximum_mm=float(swiss["p99"]),
    )


def apply_precipitation_transform(
    precipitation_mm: np.ndarray,
    transform: PrecipitationTransform,
    land_mask: np.ndarray,
) -> np.ndarray:
    precipitation = np.asarray(precipitation_mm, dtype=np.float32)
    land = np.asarray(land_mask, dtype=bool)
    if precipitation.shape != land.shape:
        raise ValueError("Precipitation and land mask must align")
    result = precipitation * transform.scale + transform.offset_mm
    result = np.clip(result, transform.minimum_mm, transform.maximum_mm)
    result[~land] = 0
    return result.astype(np.float32)


def build_foen_conditioning_package(
    plan_file: str | Path,
    atlas_directory: str | Path,
    constraint_file: str | Path,
    foen_targets_file: str | Path,
    swiss_calibration_file: str | Path,
    plan_calibration_file: str | Path,
    output_file: str | Path,
    *,
    maximum_profile_incision_m: float = (
        DEFAULT_HYDROLOGY_PROFILE.maximum_profile_incision_m
    ),
) -> dict:
    """Persist calibrated fields consumed later by the 10 m refinement stage."""

    output = Path(output_file)
    if output.exists():
        raise FileExistsError(f"FOEN conditioning package already exists: {output}")
    with Path(foen_targets_file).open("r", encoding="utf-8") as handle:
        targets = json.load(handle)
    with Path(swiss_calibration_file).open("r", encoding="utf-8") as handle:
        swiss = json.load(handle)
    with Path(plan_calibration_file).open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    with h5py.File(plan_file, "r") as plan:
        resolution_m = float(plan.attrs["resolution_m"])
        world_seed = int(plan.attrs["world_seed"])
        region = RegionKey(int(plan.attrs["region_row"]), int(plan.attrs["region_col"]))
        flow = plan["flow_direction"][...]
        accumulation = plan["accumulation_area_m2"][...]
        elevation = plan["elevation_breached_m"][...]
        precipitation = plan["annual_precipitation_mm"][...]
        land = plan["land_mask"][...] == 1
        lakes = plan["lake_id"][...] != _NODATA_U32
        macro_basin_code = plan["macro_basin_code"][...]

    precipitation_transform = fit_simple_swiss_precipitation_transform(
        calibration["generated_climate"]["annual_precipitation_mm"],
        swiss["annual_precipitation_mm"],
    )
    expected_transform = (
        DEFAULT_HYDROLOGY_PROFILE.generated_precipitation_scale,
        DEFAULT_HYDROLOGY_PROFILE.generated_precipitation_offset_mm,
        DEFAULT_HYDROLOGY_PROFILE.generated_precipitation_minimum_mm,
        DEFAULT_HYDROLOGY_PROFILE.generated_precipitation_maximum_mm,
    )
    actual_transform = (
        precipitation_transform.scale,
        precipitation_transform.offset_mm,
        precipitation_transform.minimum_mm,
        precipitation_transform.maximum_mm,
    )
    if not np.allclose(actual_transform, expected_transform, rtol=0.0, atol=1e-6):
        raise ValueError(
            "FOEN precipitation calibration does not match the V4 profile contract"
        )
    calibrated_precipitation = apply_precipitation_transform(
        precipitation, precipitation_transform, land
    )
    processing_order = processing_order_from_d8(flow, land)
    threshold_km2 = float(
        calibration["recommendations"]["channel_minimum_area_km2"]
    )
    runoff_ratio = float(calibration["recommendations"]["runoff_ratio"])
    channels = (
        land & (flow >= 1) & (flow <= 8)
        & (accumulation >= threshold_km2 * 1_000_000.0)
    )

    hierarchy = build_hierarchical_routing_zones(
        atlas_directory, macro_basin_code, minimum_major_basin_area_km2=100_000.0
    )
    boundary = materialize_macro_boundary_conditions(
        constraint_file,
        region=region,
        major_basin_codes=hierarchy.major_basin_codes,
        outlet_corridor_cells=16,
    )
    with h5py.File(constraint_file, "r") as constraints:
        boundary_runoff_ratio = float(
            constraints.attrs.get(
                "runoff_ratio", DEFAULT_HYDROLOGY_PROFILE.runoff_ratio
            )
        )
    upstream_unit_precipitation_volume = (
        precipitation_transform.scale
        * boundary.initial_discharge_m3s
        / boundary_runoff_ratio
        + precipitation_transform.offset_mm
        / 1000.0
        * boundary.initial_accumulation_area_m2
        / SECONDS_PER_YEAR
    )
    inherited_discharge = runoff_ratio * upstream_unit_precipitation_volume
    local_discharge = (
        calibrated_precipitation.astype(np.float64)
        / 1000.0
        * runoff_ratio
        * resolution_m**2
        / SECONDS_PER_YEAR
    )
    local_discharge += inherited_discharge
    input_discharge_m3s = float(local_discharge.sum())
    calibrated_discharge = accumulate_values_d8(
        flow, processing_order[::-1].copy(), local_discharge
    ).astype(np.float32)
    terminal_discharge_m3s = float(
        calibrated_discharge[flow == 0].astype(np.float64).sum()
    )

    desired_grade, target_bed, profile_incision, valley_half_width = (
        build_longitudinal_profile_targets(
            flow,
            processing_order,
            accumulation,
            elevation,
            channels,
            lakes,
            calibration,
            resolution_m=resolution_m,
            channel_threshold_km2=threshold_km2,
            maximum_incision_m=maximum_profile_incision_m,
        )
    )

    hierarchy_targets = {
        name: float(
            targets["catchments"][f"level_{name}km_grouped_area_km2"]
            ["quantiles"]["p50"]
        )
        for name in ("40", "150", "1000")
    }
    level40, pours40 = build_flow_partition(
        flow, processing_order, accumulation, land,
        target_area_km2=hierarchy_targets["40"],
        resolution_m=resolution_m,
    )
    raw150, pours150 = build_flow_partition(
        flow, processing_order, accumulation, land,
        target_area_km2=hierarchy_targets["150"],
        resolution_m=resolution_m,
    )
    raw1000, pours1000 = build_flow_partition(
        flow, processing_order, accumulation, land,
        target_area_km2=hierarchy_targets["1000"],
        resolution_m=resolution_m,
    )
    level150 = nest_partition(level40, pours40, raw150, land)
    level1000, nested150_ids, nested150_anchors = nest_partition_from_labels(
        level150, raw1000, land
    )

    partition_report = {
        "level_40": _partition_report(
            level40, land, resolution_m, hierarchy_targets["40"]
        ),
        "level_150": _partition_report(
            level150, land, resolution_m, hierarchy_targets["150"]
        ),
        "level_1000": _partition_report(
            level1000, land, resolution_m, hierarchy_targets["1000"]
        ),
    }
    report = {
        "schema_version": 1,
        "world_seed": world_seed,
        "region": [region.row, region.col],
        "precipitation_transform": asdict(precipitation_transform),
        "generated_precipitation_before_mm": _distribution(precipitation[land]),
        "generated_precipitation_after_mm": _distribution(calibrated_precipitation[land]),
        "runoff_ratio": runoff_ratio,
        "boundary_runoff_ratio": boundary_runoff_ratio,
        "input_discharge_m3s": input_discharge_m3s,
        "terminal_discharge_m3s": terminal_discharge_m3s,
        "discharge_conservation_relative_error": float(
            abs(terminal_discharge_m3s - input_discharge_m3s)
            / max(input_discharge_m3s, 1e-12)
        ),
        "channel_minimum_area_km2": threshold_km2,
        "channel_count": int(channels.sum()),
        "maximum_profile_incision_m": float(np.max(profile_incision)),
        "profile_incision_cap_fraction": float(
            np.mean(profile_incision[channels] >= maximum_profile_incision_m - 1e-3)
        ),
        "partitions": partition_report,
        "partition_containment": {
            "level_40_within_level_150": True,
            "level_150_within_level_1000": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with h5py.File(temporary, "w") as artifact:
        artifact.attrs["schema_version"] = 1
        artifact.attrs["hydrology_profile_schema"] = DEFAULT_HYDROLOGY_PROFILE.schema
        artifact.attrs["hydrology_profile_version"] = DEFAULT_HYDROLOGY_PROFILE.version
        artifact.attrs["hydrology_profile_sha256"] = DEFAULT_HYDROLOGY_PROFILE.fingerprint
        artifact.attrs["complete"] = True
        artifact.attrs["world_seed"] = world_seed
        artifact.attrs["region_row"] = region.row
        artifact.attrs["region_col"] = region.col
        artifact.attrs["resolution_m"] = resolution_m
        artifact.attrs["report_json"] = json.dumps(report, sort_keys=True)
        _write(artifact, "annual_precipitation_calibrated_mm", calibrated_precipitation)
        _write(artifact, "mean_discharge_calibrated_m3s", calibrated_discharge)
        _write(artifact, "channel_mask", channels.astype(np.uint8))
        _write(artifact, "target_channel_grade", desired_grade)
        _write(artifact, "target_bed_elevation_m", target_bed)
        _write(artifact, "profile_incision_m", profile_incision)
        _write(artifact, "valley_corridor_half_width_m", valley_half_width)
        _write(artifact, "subbasin_level_40", level40)
        _write(artifact, "subbasin_level_150", level150)
        _write(artifact, "subbasin_level_1000", level1000)
        pours = artifact.create_group("pour_points")
        pours.create_dataset("level_40_flat_index", data=pours40)
        pours.create_dataset("level_150_flat_index", data=pours150)
        pours.create_dataset("level_1000_flat_index", data=pours1000)
        pours.create_dataset("nested_level_150_id", data=nested150_ids)
        pours.create_dataset(
            "nested_level_150_anchor_flat_index", data=nested150_anchors
        )
    temporary.replace(output)
    return report


def build_longitudinal_profile_targets(
    flow: np.ndarray,
    processing_order: np.ndarray,
    accumulation: np.ndarray,
    elevation_m: np.ndarray,
    channel_mask: np.ndarray,
    lake_mask: np.ndarray,
    calibration: dict,
    *,
    resolution_m: float,
    channel_threshold_km2: float,
    maximum_incision_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct conservative Swiss-grade river targets without changing terrain."""

    if maximum_incision_m <= 0:
        raise ValueError("maximum_incision_m must be positive")
    generated = calibration["generated_channel_network"]["slope_by_source_elevation_m"]
    swiss = calibration["swiss_channel_morphology"]["slope_by_source_elevation_m"]
    band_edges = np.asarray((0.0, 500.0, 1000.0, 2000.0, np.inf))
    factors = np.ones(4, dtype=np.float32)
    floors = np.zeros(4, dtype=np.float32)
    ceilings = np.ones(4, dtype=np.float32)
    for index, name in enumerate(("0_to_500", "500_to_1000", "1000_to_2000", "2000_to_inf")):
        generated_median = max(float(generated[name]["quantiles"]["p50"]), 1e-4)
        swiss_quantiles = swiss[name]["quantiles"]
        factors[index] = min(3.0, float(swiss_quantiles["p50"]) / generated_median)
        floors[index] = float(swiss_quantiles["p25"]) * 0.25
        ceilings[index] = float(swiss_quantiles["p90"])
    desired = _desired_channel_grades(
        flow,
        elevation_m,
        channel_mask,
        band_edges,
        factors,
        floors,
        ceilings,
        resolution_m,
    )
    target_bed = _propagate_target_bed(
        flow,
        np.ascontiguousarray(processing_order, dtype=np.uint32),
        np.asarray(elevation_m, dtype=np.float32),
        np.asarray(channel_mask, dtype=np.bool_),
        np.asarray(lake_mask, dtype=np.bool_),
        desired,
        float(resolution_m),
        float(maximum_incision_m),
    )
    incision = np.zeros(elevation_m.shape, dtype=np.float32)
    incision[channel_mask] = np.maximum(
        0.0, elevation_m[channel_mask] - target_bed[channel_mask]
    )
    target_bed[~channel_mask] = np.nan
    desired[~channel_mask] = np.nan
    area_km2 = accumulation / 1_000_000.0
    width = np.full(elevation_m.shape, np.nan, dtype=np.float32)
    width[channel_mask] = np.clip(
        60.0 + 35.0 * np.log10(np.maximum(area_km2[channel_mask] / channel_threshold_km2, 1.0)),
        60.0,
        600.0,
    )
    return desired, target_bed, incision, width


def build_flow_partition(
    flow: np.ndarray,
    processing_order: np.ndarray,
    accumulation_area_m2: np.ndarray,
    land_mask: np.ndarray,
    *,
    target_area_km2: float,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition a D8 graph into deterministic downstream accumulation bands."""

    if target_area_km2 <= 0 or resolution_m <= 0:
        raise ValueError("target_area_km2 and resolution_m must be positive")
    flow = np.ascontiguousarray(flow, dtype=np.uint8)
    order = np.ascontiguousarray(processing_order, dtype=np.uint32)
    accumulation = np.ascontiguousarray(accumulation_area_m2, dtype=np.float64)
    land = np.ascontiguousarray(land_mask, dtype=np.bool_)
    cuts = _partition_cuts(
        flow,
        order,
        land,
        float(resolution_m**2),
        target_area_km2 * 1_000_000.0,
    )
    pour_points = np.flatnonzero(cuts).astype(np.uint32)
    labels = np.full(flow.shape, _NODATA_U32, dtype=np.uint32)
    labels.flat[pour_points] = np.arange(pour_points.size, dtype=np.uint32)
    _propagate_partition_labels(flow, order, land, labels)
    if np.any(land & (labels == _NODATA_U32)):
        raise RuntimeError("Some land cells did not reach a sub-basin pour point")
    return labels, pour_points


def nest_partition(
    child_labels: np.ndarray,
    child_pour_points: np.ndarray,
    raw_parent_labels: np.ndarray,
    land_mask: np.ndarray,
) -> np.ndarray:
    """Snap a coarser partition to whole child units, guaranteeing containment."""

    child = np.asarray(child_labels, dtype=np.uint32)
    parent = np.asarray(raw_parent_labels, dtype=np.uint32)
    land = np.asarray(land_mask, dtype=bool)
    mapping = parent.ravel()[np.asarray(child_pour_points, dtype=np.uint32)]
    if np.any(mapping == _NODATA_U32):
        raise RuntimeError("A child pour point has no parent partition")
    nested = np.full(child.shape, _NODATA_U32, dtype=np.uint32)
    nested[land] = mapping[child[land]]
    return nested


def nest_partition_from_labels(
    child_labels: np.ndarray,
    raw_parent_labels: np.ndarray,
    land_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nest already-snapped child units using an extant cell as each anchor."""

    child = np.asarray(child_labels, dtype=np.uint32)
    parent = np.asarray(raw_parent_labels, dtype=np.uint32)
    land = np.asarray(land_mask, dtype=bool)
    land_indices = np.flatnonzero(land)
    child_values = child.ravel()[land_indices]
    identifiers, first = np.unique(child_values, return_index=True)
    anchors = land_indices[first].astype(np.uint32)
    mapping = np.full(int(identifiers.max()) + 1, _NODATA_U32, dtype=np.uint32)
    mapping[identifiers] = parent.ravel()[anchors]
    if np.any(mapping[identifiers] == _NODATA_U32):
        raise RuntimeError("A snapped child unit has no parent partition")
    nested = np.full(child.shape, _NODATA_U32, dtype=np.uint32)
    nested[land] = mapping[child[land]]
    return nested, identifiers.astype(np.uint32), anchors


def _partition_report(
    labels: np.ndarray,
    land: np.ndarray,
    resolution_m: float,
    target_area_km2: float,
) -> dict:
    values = labels[land]
    counts = np.bincount(values)
    counts = counts[counts > 0]
    areas = counts.astype(np.float64) * resolution_m**2 / 1_000_000.0
    mature = areas >= target_area_km2 * 0.25
    return {
        "unit_count": int(counts.size),
        "local_area_km2": _distribution(areas),
        "mature_minimum_fraction_of_target": 0.25,
        "mature_unit_count": int(np.count_nonzero(mature)),
        "mature_land_coverage_fraction": float(areas[mature].sum() / areas.sum()),
        "mature_local_area_km2": _distribution(areas[mature]),
    }


@njit(cache=True)
def _partition_cuts(flow, order, land, cell_area_units, target_area_m2):
    rows, cols = flow.shape
    cuts = np.zeros(flow.shape, dtype=np.bool_)
    residual = np.zeros(flow.shape, dtype=np.float64)
    # `cell_area_units` is supplied separately to keep this kernel usable for
    # weighted partitions later. It is square metres in the current caller.
    for position in range(order.size):
        index = int(order[position])
        row = index // cols
        col = index - row * cols
        residual[row, col] += cell_area_units
        code = int(flow[row, col])
        closes = residual[row, col] >= target_area_m2
        if code < 1 or code > 8:
            cuts[row, col] = True
            continue
        next_row = row + _DROW[code]
        next_col = col + _DCOL[code]
        if not land[next_row, next_col]:
            cuts[row, col] = True
            continue
        if closes:
            cuts[row, col] = True
        else:
            residual[next_row, next_col] += residual[row, col]
    return cuts


@njit(cache=True)
def _propagate_partition_labels(flow, order, land, labels):
    cols = flow.shape[1]
    for position in range(order.size - 1, -1, -1):
        index = int(order[position])
        row = index // cols
        col = index - row * cols
        if labels[row, col] != _NODATA_U32:
            continue
        code = int(flow[row, col])
        if code < 1 or code > 8:
            continue
        next_row = row + _DROW[code]
        next_col = col + _DCOL[code]
        if land[next_row, next_col]:
            labels[row, col] = labels[next_row, next_col]


@njit(cache=True)
def _desired_channel_grades(
    flow, elevation, channels, band_edges, factors, floors, ceilings, resolution_m
):
    rows, cols = flow.shape
    desired = np.zeros(flow.shape, dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            if not channels[row, col]:
                continue
            code = int(flow[row, col])
            if code < 1 or code > 8:
                continue
            next_row = row + _DROW[code]
            next_col = col + _DCOL[code]
            distance = resolution_m * (1.41421356237 if code in (2, 4, 6, 8) else 1.0)
            current = max(0.0, (elevation[row, col] - elevation[next_row, next_col]) / distance)
            band = 0
            while band + 1 < band_edges.size - 1 and elevation[row, col] >= band_edges[band + 1]:
                band += 1
            desired[row, col] = min(
                ceilings[band], max(floors[band], current * factors[band])
            )
    return desired


@njit(cache=True)
def _propagate_target_bed(
    flow, order, elevation, channels, lakes, desired, resolution_m, maximum_incision_m
):
    bed = elevation.copy()
    cols = flow.shape[1]
    for position in range(order.size):
        index = int(order[position])
        row = index // cols
        col = index - row * cols
        if not channels[row, col] or lakes[row, col]:
            continue
        code = int(flow[row, col])
        if code < 1 or code > 8:
            continue
        next_row = row + _DROW[code]
        next_col = col + _DCOL[code]
        if not channels[next_row, next_col] or lakes[next_row, next_col]:
            continue
        distance = resolution_m * (1.41421356237 if code in (2, 4, 6, 8) else 1.0)
        proposed = bed[row, col] - desired[row, col] * distance
        minimum = elevation[next_row, next_col] - maximum_incision_m
        bed[next_row, next_col] = min(bed[next_row, next_col], max(minimum, proposed))
    return bed


def _write(artifact: h5py.File, name: str, values: np.ndarray) -> None:
    artifact.create_dataset(name, data=values, chunks=(256, 256), compression="lzf")


@click.command("build-foen-conditioning")
@click.argument("plan_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("atlas_directory", type=click.Path(exists=True, file_okay=False))
@click.argument("constraint_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option(
    "--foen-targets-file", default="samples/hydrology/foen_calibration_targets.json",
    show_default=True, type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--swiss-calibration-file",
    default="samples/hydrology/swiss_dem_240m_calibration_v2.json",
    show_default=True, type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--plan-calibration-file",
    default="samples/hydrology/foen_hierarchical_plan_calibration.json",
    show_default=True, type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--maximum-profile-incision-m",
    default=DEFAULT_HYDROLOGY_PROFILE.maximum_profile_incision_m,
    show_default=True,
)
def build_foen_conditioning_cli(**kwargs):
    """Build calibrated climate, river-profile, and sub-basin fields."""

    try:
        report = build_foen_conditioning_package(**kwargs)
    except (FileExistsError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(report, indent=2, sort_keys=True))
