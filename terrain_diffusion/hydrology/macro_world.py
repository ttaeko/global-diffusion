"""Build a persistent hydrology world plan from a frozen macro sample."""

from __future__ import annotations

import hashlib
from pathlib import Path

import click
import numpy as np
import scipy.ndimage

from .network import extract_river_graph, write_river_graph
from .planner import HydrologyPlannerConfig, plan_hydrology
from .decoder_contract import DEFAULT_HYDROLOGY_DECODER, decoder_provenance
from .profile_contract import DEFAULT_HYDROLOGY_PROFILE, profile_provenance
from .world_plan import WorldPlanStore, default_world_manifest


def macro_sample_to_planner_surface(
    macro_elevation_m: np.ndarray,
    macro_precipitation_mm: np.ndarray,
    *,
    refinement: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate macro fields to the 240 m global routing grid."""

    elevation = np.asarray(macro_elevation_m, dtype=np.float32)
    precipitation = np.asarray(macro_precipitation_mm, dtype=np.float32)
    if elevation.ndim != 2 or precipitation.shape != elevation.shape:
        raise ValueError("Macro elevation and precipitation must be matching 2D arrays")
    if refinement <= 1:
        raise ValueError("refinement must exceed one")
    # Elevation was learned in signed-sqrt space. Interpolating in that space
    # preserves coast transitions and avoids metre-space overshoot at peaks.
    signed_sqrt = np.sign(elevation) * np.sqrt(np.abs(elevation))
    planner_sqrt = scipy.ndimage.zoom(
        signed_sqrt,
        refinement,
        order=3,
        mode="nearest",
        prefilter=True,
        grid_mode=True,
    )
    planner_elevation = np.sign(planner_sqrt) * np.square(planner_sqrt)
    planner_precipitation = scipy.ndimage.zoom(
        precipitation,
        refinement,
        order=1,
        mode="nearest",
        prefilter=False,
        grid_mode=True,
    )
    return (
        planner_elevation.astype(np.float32),
        np.maximum(planner_precipitation, 0).astype(np.float32),
    )


def persist_world_plan_from_surfaces(
    output_directory: str | Path,
    *,
    world_id: str,
    seed: int,
    planner_elevation_m: np.ndarray,
    planner_precipitation_mm: np.ndarray,
    macro_elevation_m: np.ndarray,
    macro_precipitation_mm: np.ndarray,
    provenance: dict,
    channel_minimum_area_km2: float = DEFAULT_HYDROLOGY_PROFILE.channel_minimum_area_km2,
    runoff_ratio: float = DEFAULT_HYDROLOGY_PROFILE.runoff_ratio,
) -> WorldPlanStore:
    """Route aligned frozen surfaces and persist the complete global plan."""

    planner_elevation = np.asarray(planner_elevation_m, dtype=np.float32)
    planner_precipitation = np.asarray(planner_precipitation_mm, dtype=np.float32)
    macro_elevation = np.asarray(macro_elevation_m, dtype=np.float32)
    macro_precipitation = np.asarray(macro_precipitation_mm, dtype=np.float32)
    if planner_elevation.ndim != 2 or planner_elevation.shape[0] != planner_elevation.shape[1]:
        raise ValueError("Planner elevation must be square")
    if planner_precipitation.shape != planner_elevation.shape:
        raise ValueError("Planner precipitation must align with planner elevation")
    if macro_elevation.ndim != 2 or macro_elevation.shape[0] != macro_elevation.shape[1]:
        raise ValueError("Macro elevation must be square")
    if macro_precipitation.shape != macro_elevation.shape:
        raise ValueError("Macro precipitation must align with macro elevation")
    if planner_elevation.shape[0] != macro_elevation.shape[0] * 32:
        raise ValueError("Planner surface must contain exactly 32 cells per macro cell")
    if not np.all(np.isfinite(planner_elevation)) or not np.all(np.isfinite(planner_precipitation)):
        raise ValueError("Planner surfaces contain non-finite cells")

    land = planner_elevation > 0
    raw_planner_precipitation_mean = float(np.mean(planner_precipitation[land]))
    planner_precipitation = (
        DEFAULT_HYDROLOGY_PROFILE.calibrate_generated_precipitation(
            planner_precipitation
        )
    )
    planner_precipitation[~land] = 0.0
    macro_land = macro_elevation > 0
    macro_precipitation = DEFAULT_HYDROLOGY_PROFILE.calibrate_generated_precipitation(
        macro_precipitation
    )
    macro_precipitation[~macro_land] = 0.0
    provenance = {
        **provenance,
        **profile_provenance(DEFAULT_HYDROLOGY_PROFILE),
        **decoder_provenance(DEFAULT_HYDROLOGY_DECODER),
        "precipitation_source": "generated_climate_foen_affine_v3",
        "generated_precipitation_mean_before_mm_year": (
            raw_planner_precipitation_mean
        ),
        "generated_precipitation_mean_after_mm_year": float(
            np.mean(planner_precipitation[land])
        ),
    }
    manifest = default_world_manifest(
        world_id,
        seed,
        macro_cells=macro_elevation.shape[0],
        provenance=provenance,
    )
    store = WorldPlanStore.create(output_directory, manifest)
    planned = plan_hydrology(
        planner_elevation,
        resolution_m=240.0,
        land_mask=land,
        precipitation_mm_year=planner_precipitation,
        config=HydrologyPlannerConfig(
            channel_minimum_area_km2=channel_minimum_area_km2,
            runoff_ratio=runoff_ratio,
        ),
        build_conditioning=False,
    )
    store.write_routing_result(
        "/levels/global_240m",
        elevation_raw_m=planner_elevation,
        land_mask=land,
        result=planned.routing,
        channel_mask=planned.channel_mask,
        stream_order=planned.stream_order,
        mean_discharge_m3s=planned.mean_discharge_m3s,
        lake_id=planned.lakes.lake_id,
        water_surface_elevation_m=planned.lakes.water_surface_elevation_m,
        elevation_final_m=planned.lakes.terrain_elevation_m,
        annual_precipitation_mm=planner_precipitation,
    )
    with store.open_rasters("r+") as rasters:
        macro = rasters["levels/macro_7680m"]
        macro["elevation_raw_m"][...] = macro_elevation
        macro["elevation_final_m"][...] = macro_elevation
        macro["annual_precipitation_mm"][...] = macro_precipitation
        macro["land_mask"][...] = macro_elevation > 0

    graph = extract_river_graph(
        planned.routing.flow_direction,
        planned.channel_mask,
        planned.lakes.terrain_elevation_m,
        planned.routing.accumulation_area_m2,
        planned.routing.catchment_id,
        planned.stream_order,
        resolution_m=240.0,
        mean_discharge_m3s=planned.mean_discharge_m3s,
        lake_id=planned.lakes.lake_id,
    )
    with store.open_network() as connection:
        write_river_graph(
            connection,
            graph,
            level_name="global_240m",
            elevation_m=planned.lakes.terrain_elevation_m,
            resolution_m=240.0,
        )
        lake_outlets = {
            node.lake_id: node.node_id
            for node in graph.nodes
            if node.kind == "lake_outlet" and node.lake_id is not None
        }
        with connection:
            connection.executemany(
                """INSERT INTO lakes
                   (lake_id, outlet_node_id, kind, surface_elevation_m,
                    area_m2, volume_m3)
                   VALUES (?, ?, 'natural', ?, ?, ?)""",
                [
                    (
                        record.lake_id,
                        lake_outlets.get(record.lake_id),
                        record.surface_elevation_m,
                        record.area_m2,
                        record.volume_m3,
                    )
                    for record in planned.lakes.records
                ],
            )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('river_node_count', ?)",
            (str(len(graph.nodes)),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('river_edge_count', ?)",
            (str(len(graph.edges)),),
        )
        connection.commit()
    return store


def build_world_plan_from_macro_sample(
    macro_sample: str | Path,
    output_directory: str | Path,
    *,
    macro_cells: int | None = None,
    channel_minimum_area_km2: float = DEFAULT_HYDROLOGY_PROFILE.channel_minimum_area_km2,
    runoff_ratio: float = DEFAULT_HYDROLOGY_PROFILE.runoff_ratio,
) -> WorldPlanStore:
    """Route one frozen square macro sample and persist its complete global plan."""

    sample_path = Path(macro_sample)
    with np.load(sample_path) as sample:
        elevation = np.asarray(sample["elevation_m"], dtype=np.float32)
        coarse = np.asarray(sample["coarse"], dtype=np.float32)
        seed = int(sample["seed"])
    if elevation.ndim != 2 or elevation.shape[0] != elevation.shape[1]:
        raise ValueError("Macro sample elevation must be square")
    if coarse.shape[0] < 5 or coarse.shape[1:] != elevation.shape:
        raise ValueError("Macro sample must include at least five aligned coarse channels")
    size = elevation.shape[0] if macro_cells is None else int(macro_cells)
    if size <= 0 or size > elevation.shape[0]:
        raise ValueError("macro_cells lies outside the supplied sample")
    elevation = elevation[:size, :size]
    precipitation = coarse[4, :size, :size]
    planner_elevation, planner_precipitation = macro_sample_to_planner_surface(
        elevation, precipitation
    )
    provenance = {
        "world_plan_role": "macro-routing-prototype",
        "macro_sample": str(sample_path),
        "macro_sample_sha256": _sha256(sample_path),
        "global_surface": "signed-sqrt cubic macro interpolation",
        "hydrology_channel_minimum_area_km2": channel_minimum_area_km2,
        "hydrology_runoff_ratio": runoff_ratio,
    }
    return persist_world_plan_from_surfaces(
        output_directory,
        world_id=sample_path.stem,
        seed=seed,
        planner_elevation_m=planner_elevation,
        planner_precipitation_mm=planner_precipitation,
        macro_elevation_m=elevation,
        macro_precipitation_mm=precipitation,
        provenance=provenance,
        channel_minimum_area_km2=channel_minimum_area_km2,
        runoff_ratio=runoff_ratio,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@click.command("build-hydrology-world-plan")
@click.argument("macro_sample", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_directory", type=click.Path(file_okay=False))
@click.option("--macro-cells", type=click.IntRange(min=1), default=None)
@click.option(
    "--channel-minimum-area-km2",
    default=DEFAULT_HYDROLOGY_PROFILE.channel_minimum_area_km2,
    show_default=True,
)
@click.option(
    "--runoff-ratio", default=DEFAULT_HYDROLOGY_PROFILE.runoff_ratio,
    show_default=True,
)
def build_hydrology_world_plan(
    macro_sample, output_directory, macro_cells,
    channel_minimum_area_km2, runoff_ratio,
):
    """Build a persistent global hydrology plan from MACRO_SAMPLE."""

    store = build_world_plan_from_macro_sample(
        macro_sample,
        output_directory,
        macro_cells=macro_cells,
        channel_minimum_area_km2=channel_minimum_area_km2,
        runoff_ratio=runoff_ratio,
    )
    click.echo(f"Created hydrology world plan at {store.root}")
