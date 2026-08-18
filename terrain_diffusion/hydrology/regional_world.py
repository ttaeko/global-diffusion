"""Authoritative regional hydrology under a persistent global world plan."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3

import click
import numpy as np

from .compiled_routing import CompiledRoutingResult, processing_order_from_d8
from .conditioning import (
    HYDROLOGY_CONDITIONING_CHANNELS,
    HydrologyConditioning,
)
from .decoder_contract import (
    DEFAULT_HYDROLOGY_DECODER,
    FineHydrologyGeometry,
    HydrologyDecoderContract,
    build_fine_hydrology_geometry,
    decoder_provenance,
    require_matching_decoder,
)
from .lakes import LakePlan
from .multires import build_regional_boundary_conditions
from .network import append_river_graph, extract_river_graph
from .planner import HydrologyPlannerConfig, PlannedHydrology, plan_hydrology
from .profile_contract import (
    DEFAULT_HYDROLOGY_PROFILE,
    HydrologyProfileContract,
    profile_provenance,
    require_matching_profile,
)
from .training_profile import (
    HydrologyTrainingProfile,
    apply_hydrology_terrain_transform,
    build_hydrology_training_profile,
)
from .world_plan import WorldPlanStore


_NODATA_U32 = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class AuthoritativeHydrologyWindow:
    """Frozen 30 m plan plus the derived 10 m model-conditioning contract."""

    planned: PlannedHydrology
    profile: HydrologyTrainingProfile
    terrain_base_30m_m: np.ndarray
    geometry_10m: FineHydrologyGeometry
    conditioning_10m: np.ndarray
    provenance: dict
    regional_group_paths: tuple[str, ...]
    fine_group_paths: tuple[str, ...]
    cache_hit: bool = False


def plan_and_persist_regional_window(
    world_plan_directory: str | Path,
    elevation_m: np.ndarray,
    climate: np.ndarray,
    native_bounds: tuple[int, int, int, int],
    *,
    world_seed: int,
    source_reference: str,
    source_sha256: str | None = None,
    precipitation_is_calibrated: bool = False,
    contract: HydrologyProfileContract = DEFAULT_HYDROLOGY_PROFILE,
    decoder_contract: HydrologyDecoderContract = DEFAULT_HYDROLOGY_DECODER,
) -> AuthoritativeHydrologyWindow:
    """Route, profile, persist, and derive fine conditioning for one window.

    `native_bounds` are regional 30 m coordinates in row/column order. The
    solve inherits global catchment ownership, upstream area, and upstream
    discharge. Annual precipitation is taken from climate channel 2 and is
    mandatory; no reference-precipitation fallback is used here.
    """

    contract.validate()
    decoder_contract.validate()
    store = WorldPlanStore(world_plan_directory)
    manifest = store.read_manifest()
    require_matching_profile(manifest.provenance, contract)
    if int(world_seed) != manifest.world_seed:
        raise ValueError(
            f"Regional seed {world_seed} does not match world seed "
            f"{manifest.world_seed}"
        )
    levels = {level.name: level for level in manifest.levels}
    if not {"global_240m", "regional_30m", "fine_10m"} <= levels.keys():
        raise ValueError(
            "World plan lacks the global_240m/regional_30m/fine_10m hierarchy"
        )

    elevation = np.asarray(elevation_m, dtype=np.float32)
    climate_values = np.asarray(climate, dtype=np.float32)
    if elevation.ndim != 2 or climate_values.ndim != 3:
        raise ValueError("Elevation must be 2D and climate must be channel-first 3D")
    if climate_values.shape[0] < 3 or climate_values.shape[1:] != elevation.shape:
        raise ValueError("Climate must align with elevation and include precipitation")
    generated_precipitation = climate_values[2].copy()
    if (
        not np.isfinite(generated_precipitation).any()
        or np.nanmax(generated_precipitation) <= 0
    ):
        raise ValueError("Authoritative regional hydrology requires actual precipitation")
    precipitation = (
        generated_precipitation.copy()
        if precipitation_is_calibrated
        else contract.calibrate_generated_precipitation(generated_precipitation)
    )

    i1, j1, i2, j2 = map(int, native_bounds)
    if elevation.shape != (i2 - i1, j2 - j1):
        raise ValueError("native_bounds do not match elevation shape")
    if any(value % 8 for value in (i1, j1, i2, j2)):
        raise ValueError("Regional bounds must align to the 240 m global grid")
    if min(i1, j1) < 0:
        raise ValueError("Regional bounds must be non-negative")

    content_sha = source_sha256 or _hash_window_inputs(
        elevation, climate_values, native_bounds, world_seed
    )
    window_id = f"regional_30m:{i1}:{j1}:{i2}:{j2}"
    existing = _existing_window(store, window_id)
    if existing is not None:
        if (
            existing["source_sha256"] == content_sha
            and existing["profile_sha256"] == contract.fingerprint
        ):
            require_matching_decoder(
                json.loads(existing["provenance_json"]), decoder_contract
            )
            return _load_authoritative_window(
                store,
                native_bounds,
                contract,
                decoder_contract,
                existing,
                cache_hit=True,
            )
        raise ValueError(
            f"Frozen window {window_id} exists with different source/profile data"
        )
    _reject_conflicting_overlap(store, native_bounds)

    global_row0, global_col0 = i1 // 8, j1 // 8
    global_height, global_width = elevation.shape[0] // 8, elevation.shape[1] // 8
    with store.open_rasters() as rasters:
        global_group = rasters["levels/global_240m"]
        global_flow = global_group["flow_direction"][:]
        global_accumulation = global_group["accumulation_area_m2"][:]
        global_catchments = global_group["catchment_id"][:]
        global_land = global_group["land_mask"][:] == 1
        global_discharge = global_group["mean_discharge_m3s"][:]
    boundary = build_regional_boundary_conditions(
        global_flow,
        global_accumulation,
        global_catchments,
        global_land,
        row_start=global_row0,
        col_start=global_col0,
        height=global_height,
        width=global_width,
        refinement=8,
    )
    inherited_discharge = np.zeros(elevation.shape, dtype=np.float64)
    for portal in boundary.portals:
        if portal.kind == "inflow":
            value = global_discharge[portal.global_row, portal.global_col]
            if np.isfinite(value):
                inherited_discharge[portal.regional_row, portal.regional_col] += value

    valid_zone = boundary.routing_zones != _NODATA_U32
    land = (elevation > manifest.sea_level_elevation_m) & valid_zone
    land |= boundary.terminal_mask
    precipitation[~np.isfinite(precipitation) | ~land] = 0
    planned = plan_hydrology(
        elevation,
        resolution_m=30.0,
        land_mask=land,
        precipitation_mm_year=precipitation,
        terminal_mask=boundary.terminal_mask,
        open_boundary=False,
        routing_zones=boundary.routing_zones,
        initial_accumulation_area_m2=boundary.initial_accumulation_area_m2,
        initial_discharge_m3s=inherited_discharge,
        config=HydrologyPlannerConfig(
            channel_minimum_area_km2=contract.channel_minimum_area_km2,
            reference_precipitation_mm_year=(
                contract.reference_precipitation_mm_year
            ),
            runoff_ratio=contract.runoff_ratio,
            conditioning_distance_scale_m=contract.conditioning_distance_scale_m,
        ),
    )
    if planned.conditioning is None:
        raise RuntimeError("Hydrology planner did not return conditioning")
    profile = build_hydrology_training_profile(
        _signed_sqrt(elevation),
        planned.conditioning.values,
        **contract.profile_kwargs(resolution_m=30.0),
        sea_level_elevation_m=manifest.sea_level_elevation_m,
        lake_water_surface_elevation_m=(
            planned.lakes.water_surface_elevation_m
        ),
        strict_outlet_floor=True,
    )
    terrain_base = apply_hydrology_terrain_transform(elevation, profile)

    with store.open_network() as connection:
        maximum_lake_id = connection.execute(
            "SELECT COALESCE(MAX(lake_id), -1) FROM lakes"
        ).fetchone()[0]
    lake_id_start = int(maximum_lake_id) + 1
    regional_lake_ids = planned.lakes.lake_id.copy()
    has_lake = regional_lake_ids != _NODATA_U32
    regional_lake_ids[has_lake] += lake_id_start

    provenance = {
        **profile_provenance(contract),
        **decoder_provenance(decoder_contract),
        "world_id": manifest.world_id,
        "world_seed": int(world_seed),
        "native_30m_bounds": [i1, j1, i2, j2],
        "source_reference": source_reference,
        "source_sha256": content_sha,
        "precipitation_source": (
            "frozen_global_240m_calibrated_bilinear_v1"
            if precipitation_is_calibrated
            else "generated_climate_channel_2_foen_affine_v3"
        ),
        "precipitation_input_calibrated": bool(precipitation_is_calibrated),
        "precipitation_generated_mean_mm_year": float(
            np.mean(generated_precipitation[land])
        ),
        "precipitation_mean_mm_year": float(np.mean(precipitation[land])),
        "inherited_inflow_portals": int(
            sum(portal.kind == "inflow" for portal in boundary.portals)
        ),
        "inherited_discharge_m3s": float(inherited_discharge.sum()),
        "terrain_transform": "minimal_uphill_bed_repair_v4",
    }
    regional_fields = _regional_fields(
        elevation,
        precipitation,
        land,
        planned,
        profile,
        terrain_base,
        regional_lake_ids,
    )
    regional_paths = store.write_level_window(
        "regional_30m", i1, j1, regional_fields, provenance=provenance
    )

    fine_geometry = build_fine_hydrology_geometry(
        planned.conditioning.values,
        profile,
        lake_water_surface_elevation_m=planned.lakes.water_surface_elevation_m,
        terrain_elevation_30m_m=terrain_base,
        contract=decoder_contract,
    )
    fine_fields = _fine_fields(regional_fields, fine_geometry)
    fine_paths = store.write_level_window(
        "fine_10m", i1 * 3, j1 * 3, fine_fields, provenance=provenance
    )
    store.record_window_provenance(
        window_id, provenance, (*regional_paths, *fine_paths)
    )

    with store.open_network() as connection:
        with connection:
            connection.executemany(
                """INSERT INTO lakes
                   (lake_id, outlet_node_id, kind, surface_elevation_m,
                    area_m2, volume_m3)
                   VALUES (?, NULL, 'natural', ?, ?, ?)""",
                [
                    (
                        lake_id_start + record.lake_id,
                        record.surface_elevation_m,
                        record.area_m2,
                        record.volume_m3,
                    )
                    for record in planned.lakes.records
                ],
            )
            graph = extract_river_graph(
                planned.routing.flow_direction,
                planned.channel_mask,
                terrain_base,
                planned.routing.accumulation_area_m2,
                planned.routing.catchment_id,
                planned.stream_order,
                resolution_m=30.0,
                origin_x_m=manifest.origin_x_m + j1 * 30.0,
                origin_z_m=manifest.origin_z_m + i1 * 30.0,
                mean_discharge_m3s=planned.mean_discharge_m3s,
                lake_id=regional_lake_ids,
            )
            append_river_graph(
                connection,
                graph,
                level_name="regional_30m",
                elevation_m=terrain_base,
                resolution_m=30.0,
                row_offset=i1,
                col_offset=j1,
                origin_x_m=manifest.origin_x_m,
                origin_z_m=manifest.origin_z_m,
                manage_transaction=False,
            )
            connection.execute(
                """INSERT INTO regional_windows
                   (window_id, level_name, row_start, col_start, row_stop, col_stop,
                    source_path, source_sha256, lake_id_start, lake_count,
                    profile_schema, profile_version, profile_sha256,
                    provenance_json, storage_paths_json)
                   VALUES (?, 'regional_30m', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    window_id,
                    i1,
                    j1,
                    i2,
                    j2,
                    source_reference,
                    content_sha,
                    lake_id_start if planned.lakes.records else None,
                    len(planned.lakes.records),
                    contract.schema,
                    contract.version,
                    contract.fingerprint,
                    json.dumps(provenance, sort_keys=True),
                    json.dumps([*regional_paths, *fine_paths]),
                ),
            )
            connection.executemany(
                """INSERT INTO regional_portals
                   (window_id, sequence_index, kind, global_row, global_col,
                    regional_row, regional_col, catchment_id,
                    upstream_area_m2, upstream_discharge_m3s)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        window_id,
                        index,
                        portal.kind,
                        portal.global_row,
                        portal.global_col,
                        i1 + portal.regional_row,
                        j1 + portal.regional_col,
                        portal.catchment_id,
                        portal.upstream_area_m2,
                        (
                            float(
                                global_discharge[
                                    portal.global_row, portal.global_col
                                ]
                            )
                            if portal.kind == "inflow"
                            and np.isfinite(
                                global_discharge[
                                    portal.global_row, portal.global_col
                                ]
                            )
                            else 0.0
                        ),
                    )
                    for index, portal in enumerate(boundary.portals)
                ],
            )
    return AuthoritativeHydrologyWindow(
        planned=planned,
        profile=profile,
        terrain_base_30m_m=terrain_base,
        geometry_10m=fine_geometry,
        conditioning_10m=fine_geometry.conditioning,
        provenance=provenance,
        regional_group_paths=regional_paths,
        fine_group_paths=fine_paths,
    )


def ingest_v4_hydrology_window(
    world_plan_directory: str | Path,
    v4_sample: str | Path,
    *,
    contract: HydrologyProfileContract = DEFAULT_HYDROLOGY_PROFILE,
) -> AuthoritativeHydrologyWindow:
    """Ingest one V4 NPZ sample under the authoritative V4 contract."""

    sample_path = Path(v4_sample)
    with np.load(sample_path) as sample:
        elevation = np.asarray(sample["elevation_m"], dtype=np.float32)
        climate = np.asarray(sample["climate"], dtype=np.float32)
        bounds = tuple(map(int, np.asarray(sample["native_bounds"]).tolist()))
        seed = int(sample["seed"])
    return plan_and_persist_regional_window(
        world_plan_directory,
        elevation,
        climate,
        bounds,
        world_seed=seed,
        source_reference=str(sample_path.resolve()),
        source_sha256=_sha256(sample_path),
        contract=contract,
    )


def _regional_fields(
    elevation: np.ndarray,
    precipitation: np.ndarray,
    land: np.ndarray,
    planned: PlannedHydrology,
    profile: HydrologyTrainingProfile,
    terrain_base: np.ndarray,
    lake_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    conditioning = planned.conditioning
    assert conditioning is not None
    profile_incision = np.zeros(elevation.shape, dtype=np.float32)
    has_target_bed = np.isfinite(profile.target_bed_elevation_m)
    profile_incision[has_target_bed] = np.maximum(
        0.0,
        elevation[has_target_bed] - profile.target_bed_elevation_m[has_target_bed],
    )
    fields: dict[str, np.ndarray] = {
        "elevation_raw_m": elevation,
        "elevation_conditioned_m": planned.routing.elevation_conditioned_m,
        "elevation_correction_m": planned.routing.elevation_correction_m,
        "elevation_final_m": terrain_base,
        "water_surface_elevation_m": planned.lakes.water_surface_elevation_m,
        "annual_precipitation_mm": precipitation,
        "land_mask": land.astype(np.uint8),
        "lake_id": lake_ids,
        "flow_direction": planned.routing.flow_direction,
        "accumulation_area_m2": planned.routing.accumulation_area_m2,
        "mean_discharge_m3s": planned.mean_discharge_m3s,
        "catchment_id": planned.routing.catchment_id,
        "stream_order": planned.stream_order,
        "channel_mask": planned.channel_mask.astype(np.uint8),
        "profile_channel_mask": profile.channel_mask.astype(np.uint8),
        "profile_incision_m": profile_incision,
        "target_channel_grade": profile.target_downstream_drop_m / 30.0,
        "valley_corridor": profile.conditioning[3],
        "target_bed_elevation_m": profile.target_bed_elevation_m,
        "terrain_profile_correction_m": profile.terrain_correction_m,
    }
    fields.update(
        {
            f"hydro_{name}": conditioning.values[index]
            for index, name in enumerate(HYDROLOGY_CONDITIONING_CHANNELS)
        }
    )
    return fields


def _fine_fields(
    regional_fields: dict[str, np.ndarray], geometry: FineHydrologyGeometry
) -> dict[str, np.ndarray]:
    nearest_names = (
        "elevation_raw_m",
        "elevation_conditioned_m",
        "elevation_correction_m",
        "elevation_final_m",
        "water_surface_elevation_m",
        "annual_precipitation_mm",
        "land_mask",
        "lake_id",
        "flow_direction",
        "accumulation_area_m2",
        "mean_discharge_m3s",
        "catchment_id",
        "stream_order",
        "channel_mask",
        "profile_channel_mask",
        "profile_incision_m",
        "target_channel_grade",
        "valley_corridor",
        "target_bed_elevation_m",
        "terrain_profile_correction_m",
    )
    fields = {
        name: np.repeat(np.repeat(regional_fields[name], 3, axis=0), 3, axis=1)
        for name in nearest_names
    }
    fine_lake_id = fields["lake_id"].copy()
    fine_lake_id[~geometry.lake_mask] = _NODATA_U32
    fields.update(
        {
            "channel_mask": geometry.channel_centerline_mask.astype(np.uint8),
            "profile_channel_mask": geometry.channel_centerline_mask.astype(np.uint8),
            "target_bed_elevation_m": geometry.target_bed_elevation_m,
            "water_surface_elevation_m": geometry.water_surface_elevation_m,
            "lake_id": fine_lake_id,
            "channel_coverage": geometry.channel_coverage,
            "lake_coverage": geometry.lake_coverage,
            "decoder_freedom_mask": geometry.freedom_mask,
            "river_width_m": geometry.river_width_m,
            "decoder_profile_channel": geometry.profile_conditioning[0],
            "decoder_profile_incision": geometry.profile_conditioning[1],
            "decoder_profile_grade": geometry.profile_conditioning[2],
            "decoder_valley_corridor": geometry.profile_conditioning[3],
            "decoder_bed_relative_elevation": (
                geometry.bed_relative_conditioning
            ),
            "valley_corridor": geometry.profile_conditioning[3],
        }
    )
    fields.update(
        {
            f"hydro_{name}": geometry.conditioning[index]
            for index, name in enumerate(HYDROLOGY_CONDITIONING_CHANNELS)
        }
    )
    return fields


def _load_authoritative_window(
    store: WorldPlanStore,
    bounds: tuple[int, int, int, int],
    contract: HydrologyProfileContract,
    decoder_contract: HydrologyDecoderContract,
    existing: dict,
    *,
    cache_hit: bool,
) -> AuthoritativeHydrologyWindow:
    i1, j1, i2, j2 = map(int, bounds)
    height, width = i2 - i1, j2 - j1
    names = (
        "elevation_raw_m",
        "elevation_conditioned_m",
        "elevation_correction_m",
        "elevation_final_m",
        "water_surface_elevation_m",
        "land_mask",
        "lake_id",
        "flow_direction",
        "accumulation_area_m2",
        "mean_discharge_m3s",
        "catchment_id",
        "stream_order",
        "channel_mask",
        *(f"hydro_{name}" for name in HYDROLOGY_CONDITIONING_CHANNELS),
    )
    fields = store.read_level_window(
        "regional_30m", i1, j1, height, width, tuple(names)
    )
    land = fields["land_mask"] == 1
    flow = fields["flow_direction"]
    order = processing_order_from_d8(
        np.ascontiguousarray(flow), np.ascontiguousarray(land)
    )
    routing = CompiledRoutingResult(
        elevation_conditioned_m=fields["elevation_conditioned_m"],
        elevation_correction_m=fields["elevation_correction_m"],
        flow_direction=flow,
        accumulation_area_m2=fields["accumulation_area_m2"],
        catchment_id=fields["catchment_id"],
        processing_order=order,
        outlet_count=int(np.count_nonzero(land & (flow == 0))),
    )
    lake_id = fields["lake_id"]
    lake_mask = lake_id != _NODATA_U32
    lakes = LakePlan(
        lake_id=lake_id,
        lake_mask=lake_mask,
        water_surface_elevation_m=fields["water_surface_elevation_m"],
        terrain_elevation_m=fields["elevation_raw_m"],
        records=(),
    )
    conditioning = HydrologyConditioning(
        values=np.stack(
            [fields[f"hydro_{name}"] for name in HYDROLOGY_CONDITIONING_CHANNELS]
        ).astype(np.float32)
    )
    planned = PlannedHydrology(
        routing=routing,
        lakes=lakes,
        channel_mask=fields["channel_mask"] == 1,
        stream_order=fields["stream_order"],
        mean_discharge_m3s=fields["mean_discharge_m3s"],
        conditioning=conditioning,
    )
    profile = build_hydrology_training_profile(
        _signed_sqrt(fields["elevation_raw_m"]),
        conditioning.values,
        **contract.profile_kwargs(resolution_m=30.0),
        sea_level_elevation_m=store.read_manifest().sea_level_elevation_m,
        lake_water_surface_elevation_m=fields[
            "water_surface_elevation_m"
        ],
        strict_outlet_floor=True,
    )
    fine_names = [
        *(f"hydro_{name}" for name in HYDROLOGY_CONDITIONING_CHANNELS),
        "channel_mask",
        "channel_coverage",
        "target_bed_elevation_m",
        "lake_coverage",
        "water_surface_elevation_m",
        "decoder_freedom_mask",
        "river_width_m",
        "decoder_profile_channel",
        "decoder_profile_incision",
        "decoder_profile_grade",
        "decoder_valley_corridor",
    ]
    if decoder_contract.version == 2:
        fine_names.append("decoder_bed_relative_elevation")
    fine_fields = store.read_level_window(
        "fine_10m",
        i1 * 3,
        j1 * 3,
        height * 3,
        width * 3,
        tuple(fine_names),
    )
    fine_conditioning = np.stack(
        [fine_fields[f"hydro_{name}"] for name in HYDROLOGY_CONDITIONING_CHANNELS]
    ).astype(np.float32)
    fine_geometry = FineHydrologyGeometry(
        conditioning=fine_conditioning,
        profile_conditioning=np.stack(
            (
                fine_fields["decoder_profile_channel"],
                fine_fields["decoder_profile_incision"],
                fine_fields["decoder_profile_grade"],
                fine_fields["decoder_valley_corridor"],
            )
        ).astype(np.float32),
        bed_relative_conditioning=(
            fine_fields["decoder_bed_relative_elevation"].astype(np.float32)
            if decoder_contract.version == 2
            else np.zeros((height * 3, width * 3), dtype=np.float32)
        ),
        channel_centerline_mask=fine_fields["channel_mask"] == 1,
        channel_coverage=fine_fields["channel_coverage"].astype(np.float32),
        target_bed_elevation_m=fine_fields["target_bed_elevation_m"].astype(
            np.float32
        ),
        lake_mask=fine_fields["lake_coverage"] >= 0.5,
        lake_coverage=fine_fields["lake_coverage"].astype(np.float32),
        water_surface_elevation_m=fine_fields[
            "water_surface_elevation_m"
        ].astype(np.float32),
        freedom_mask=fine_fields["decoder_freedom_mask"].astype(np.float32),
        river_width_m=fine_fields["river_width_m"].astype(np.float32),
    )
    fine_geometry.validate()
    paths = tuple(json.loads(existing.get("storage_paths_json") or "[]"))
    regional_paths = tuple(path for path in paths if "/regional_30m/" in path)
    fine_paths = tuple(path for path in paths if "/fine_10m/" in path)
    return AuthoritativeHydrologyWindow(
        planned=planned,
        profile=profile,
        terrain_base_30m_m=fields["elevation_final_m"],
        geometry_10m=fine_geometry,
        conditioning_10m=fine_conditioning,
        provenance=json.loads(existing["provenance_json"]),
        regional_group_paths=regional_paths,
        fine_group_paths=fine_paths,
        cache_hit=cache_hit,
    )


def _existing_window(store: WorldPlanStore, window_id: str) -> dict | None:
    with store.open_network() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM regional_windows WHERE window_id = ?", (window_id,)
        ).fetchone()
    return None if row is None else dict(row)


def _reject_conflicting_overlap(
    store: WorldPlanStore, bounds: tuple[int, int, int, int]
) -> None:
    i1, j1, i2, j2 = map(int, bounds)
    with store.open_network() as connection:
        row = connection.execute(
            """SELECT window_id FROM regional_windows
               WHERE NOT (row_stop <= ? OR row_start >= ?
                           OR col_stop <= ? OR col_start >= ?)
               LIMIT 1""",
            (i1, i2, j1, j2),
        ).fetchone()
    if row is not None:
        raise ValueError(
            "Authoritative regional windows may not overlap; conflicting window: "
            f"{row[0]}"
        )


def _hash_window_inputs(
    elevation: np.ndarray,
    climate: np.ndarray,
    bounds: tuple[int, int, int, int],
    seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(bounds, dtype="<i8").tobytes())
    digest.update(np.asarray([seed], dtype="<i8").tobytes())
    for values in (elevation, climate):
        array = np.ascontiguousarray(values, dtype="<f4")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _signed_sqrt(elevation_m: np.ndarray) -> np.ndarray:
    values = np.asarray(elevation_m, dtype=np.float32)
    return np.sign(values) * np.sqrt(np.abs(values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@click.command("ingest-v4-hydrology")
@click.argument("world_plan_directory", type=click.Path(exists=True, file_okay=False))
@click.argument("v4_sample", type=click.Path(exists=True, dir_okay=False))
def ingest_v4_hydrology(world_plan_directory, v4_sample):
    """Freeze a V4 sample and its derived 10 m conditioning under Hydrology V4."""

    try:
        result = ingest_v4_hydrology_window(world_plan_directory, v4_sample)
    except (FileExistsError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        json.dumps(
            {
                "regional_groups": result.regional_group_paths,
                "fine_groups": result.fine_group_paths,
                "profile_sha256": result.provenance["hydrology_profile_sha256"],
                "cache_hit": result.cache_hit,
            },
            indent=2,
            sort_keys=True,
        )
    )
