"""Project frozen macro basins and river crossings into a 240 m region."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import click
import h5py
import numpy as np
import scipy.ndimage
from numba import njit

from .atlas import HydrologyAtlas, RegionKey
from .macro_topology import MacroBasinClosureAnalysis, analyze_macro_basin_closure
from .runoff import mean_discharge_from_runoff
from .profile_contract import DEFAULT_HYDROLOGY_PROFILE
from .world_plan import D8_DIRECTION_OFFSETS


_NODATA_U32 = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class MacroPortal:
    contract_id: str
    kind: str
    planner_row: int
    planner_col: int
    basin_id: str
    basin_code: int
    upstream_area_m2: float
    mean_discharge_m3s: float


@dataclass(frozen=True)
class MacroRegionConstraints:
    basin_code: np.ndarray
    divide_relaxation_mask: np.ndarray
    portals: tuple[MacroPortal, ...]


@dataclass(frozen=True)
class MacroBoundaryConditions:
    """Materialized immutable contracts for one regional routing solve."""

    terminal_mask: np.ndarray
    initial_accumulation_area_m2: np.ndarray
    initial_discharge_m3s: np.ndarray
    portal_count: int
    inflow_count: int
    outlet_count: int


@dataclass(frozen=True)
class HierarchicalRoutingZones:
    """Hard continental sectors plus a shared, reorganizable minor network."""

    routing_zones: np.ndarray
    protected_divide_mask: np.ndarray
    major_basin_codes: frozenset[int]
    minor_zone_code: int


def build_macro_region_constraints(
    atlas_directory: str | Path,
    surface_file: str | Path,
    *,
    centre_row: int = 0,
    centre_col: int = 0,
    radius: int = 1,
    divide_relaxation_cells: int = 16,
    runoff_ratio: float = DEFAULT_HYDROLOGY_PROFILE.runoff_ratio,
) -> tuple[Path, MacroRegionConstraints]:
    """Persist snapped basin zones and immutable cross-region flow portals."""

    atlas = HydrologyAtlas(atlas_directory)
    manifest = atlas.read_manifest()
    elevation_macro, precipitation_macro = _load_macro_mosaic(
        atlas, centre_row=centre_row, centre_col=centre_col, radius=radius
    )
    side = elevation_macro.shape[0]
    focus = np.zeros((side, side), dtype=bool)
    focus_offset = radius * 256
    focus[focus_offset:focus_offset + 256, focus_offset:focus_offset + 256] = True
    macro_origin_row = (centre_row - radius) * 256
    macro_origin_col = (centre_col - radius) * 256
    analysis = analyze_macro_basin_closure(
        elevation_macro,
        world_seed=manifest.world_seed,
        macro_origin_row=macro_origin_row,
        macro_origin_col=macro_origin_col,
        focus_mask=focus,
    )
    if any(not basin.closed for basin in analysis.basins):
        raise ValueError("Cannot build constraints while a focus basin remains open")
    basin_by_local = {basin.local_catchment_id: basin for basin in analysis.basins}
    code_by_local = {
        local_id: atlas.basin_code(basin.basin_id)
        for local_id, basin in basin_by_local.items()
    }
    centre_slice = np.s_[
        focus_offset:focus_offset + 256,
        focus_offset:focus_offset + 256,
    ]
    centre_catchments = analysis.routing.catchment_id[centre_slice]
    centre_land = analysis.land_mask[centre_slice]
    macro_codes = np.full((256, 256), _NODATA_U32, dtype=np.uint32)
    for local_id, code in code_by_local.items():
        macro_codes[centre_land & (centre_catchments == local_id)] = code
    macro_codes = _fill_unassigned_codes(macro_codes)

    with h5py.File(surface_file, "r") as surface:
        provenance = json.loads(surface.attrs["provenance_json"])
        if int(provenance["seed"]) != manifest.world_seed:
            raise ValueError("Surface seed differs from atlas seed")
        elevation_240m = surface["elevation_m"][...]
    expected_size = 256 * 32
    if elevation_240m.shape != (expected_size, expected_size):
        raise ValueError("Surface is not one canonical atlas region")
    discharge = mean_discharge_from_runoff(
        analysis.routing.flow_direction,
        analysis.routing.processing_order,
        precipitation_macro,
        resolution_m=7680.0,
        runoff_ratio=runoff_ratio,
    )
    portals = _extract_portals(
        atlas,
        analysis,
        discharge,
        basin_by_local,
        code_by_local,
        centre=RegionKey(centre_row, centre_col),
        local_row0=focus_offset,
        local_col0=focus_offset,
    )
    row_origin, col_origin = centre_row * 8192, centre_col * 8192
    anchors = [
        (
            portal.planner_row - row_origin,
            portal.planner_col - col_origin,
            portal.basin_code,
        )
        for portal in portals
    ]
    snapped_codes, relaxation = snap_projected_basin_zones(
        elevation_240m,
        macro_codes,
        refinement=32,
        relaxation_cells=divide_relaxation_cells,
        anchors=anchors,
    )
    output_directory = atlas.root / "constraints"
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / f"r{centre_row:+06d}_c{centre_col:+06d}_v3.h5"
    if output.exists():
        raise FileExistsError(f"Constraint artifact already exists: {output}")
    with h5py.File(output, "w") as artifact:
        artifact.attrs["schema_version"] = 3
        artifact.attrs["world_seed"] = manifest.world_seed
        artifact.attrs["region_row"] = centre_row
        artifact.attrs["region_col"] = centre_col
        artifact.attrs["macro_radius"] = radius
        artifact.attrs["divide_relaxation_cells"] = divide_relaxation_cells
        artifact.attrs["runoff_ratio"] = runoff_ratio
        artifact.create_dataset("macro_basin_code", data=macro_codes, compression="lzf")
        artifact.create_dataset(
            "basin_code_240m", data=snapped_codes, chunks=(256, 256), compression="lzf"
        )
        artifact.create_dataset(
            "divide_relaxation_mask_240m",
            data=relaxation.astype(np.uint8),
            chunks=(256, 256),
            compression="lzf",
        )
        portal_group = artifact.create_group("portals")
        portal_group.create_dataset(
            "planner_row", data=np.asarray([p.planner_row for p in portals], dtype=np.int64)
        )
        portal_group.create_dataset(
            "planner_col", data=np.asarray([p.planner_col for p in portals], dtype=np.int64)
        )
        portal_group.create_dataset(
            "basin_code", data=np.asarray([p.basin_code for p in portals], dtype=np.uint32)
        )
        portal_group.create_dataset(
            "upstream_area_m2",
            data=np.asarray([p.upstream_area_m2 for p in portals], dtype=np.float64),
        )
        portal_group.create_dataset(
            "mean_discharge_m3s",
            data=np.asarray([p.mean_discharge_m3s for p in portals], dtype=np.float32),
        )
        string_dtype = h5py.string_dtype("utf-8")
        portal_group.create_dataset(
            "kind", data=np.asarray([p.kind for p in portals], dtype=object), dtype=string_dtype
        )
        portal_group.create_dataset(
            "contract_id",
            data=np.asarray([p.contract_id for p in portals], dtype=object),
            dtype=string_dtype,
        )
        portal_group.create_dataset(
            "basin_id", data=np.asarray([p.basin_id for p in portals], dtype=object),
            dtype=string_dtype,
        )
    return output, MacroRegionConstraints(snapped_codes, relaxation, portals)


def snap_projected_basin_zones(
    elevation_240m: np.ndarray,
    macro_basin_code: np.ndarray,
    *,
    refinement: int = 32,
    relaxation_cells: int = 16,
    anchors: list[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Move blocky macro divides toward high-resolution ridgelines."""

    elevation = np.asarray(elevation_240m, dtype=np.float32)
    macro = np.asarray(macro_basin_code, dtype=np.uint32)
    if elevation.shape != (macro.shape[0] * refinement, macro.shape[1] * refinement):
        raise ValueError("Macro basin codes do not nest exactly into elevation")
    if relaxation_cells < 0:
        raise ValueError("relaxation_cells cannot be negative")
    projected = np.repeat(np.repeat(macro, refinement, axis=0), refinement, axis=1)
    if anchors:
        for row, col, code in anchors:
            if not (0 <= row < projected.shape[0] and 0 <= col < projected.shape[1]):
                raise ValueError("Basin anchor lies outside projected geometry")
            projected[row, col] = np.uint32(code)
    boundary = np.zeros(projected.shape, dtype=bool)
    boundary[1:, :] |= projected[1:, :] != projected[:-1, :]
    boundary[:-1, :] |= projected[:-1, :] != projected[1:, :]
    boundary[:, 1:] |= projected[:, 1:] != projected[:, :-1]
    boundary[:, :-1] |= projected[:, :-1] != projected[:, 1:]
    relaxation = scipy.ndimage.binary_dilation(
        boundary, iterations=relaxation_cells
    ) if relaxation_cells else boundary
    unique_codes, inverse = np.unique(projected, return_inverse=True)
    markers = (inverse.reshape(projected.shape) + 1).astype(np.int32)
    markers[relaxation] = 0
    if anchors:
        code_to_marker = {
            int(code): index + 1 for index, code in enumerate(unique_codes)
        }
        for row, col, code in anchors:
            markers[row, col] = code_to_marker[int(code)]
    # Preserve at least one seed for narrow macro basins erased by dilation.
    for marker in range(1, unique_codes.size + 1):
        if np.any(markers == marker):
            continue
        candidates = np.argwhere(projected == unique_codes[marker - 1])
        if candidates.size:
            values = elevation[candidates[:, 0], candidates[:, 1]]
            row, col = candidates[int(np.nanargmin(values))]
            markers[row, col] = marker
    finite = elevation[np.isfinite(elevation)]
    low, high = np.percentile(finite, [1, 99])
    scaled = np.clip((elevation - low) / max(high - low, 1e-6), 0, 1)
    cost = np.asarray(np.round(scaled * 255), dtype=np.uint8)
    watershed = scipy.ndimage.watershed_ift(cost, markers)
    snapped = unique_codes[np.clip(watershed - 1, 0, unique_codes.size - 1)]
    return snapped.astype(np.uint32), relaxation


def _fill_unassigned_codes(codes: np.ndarray) -> np.ndarray:
    missing = codes == _NODATA_U32
    if not np.any(~missing):
        raise ValueError("Region contains no assigned macro basin")
    if not np.any(missing):
        return codes
    indices = scipy.ndimage.distance_transform_edt(
        missing, return_distances=False, return_indices=True
    )
    return codes[tuple(indices)].astype(np.uint32)


def _load_macro_mosaic(
    atlas: HydrologyAtlas,
    *,
    centre_row: int,
    centre_col: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    minimum_row, maximum_row = centre_row - radius, centre_row + radius
    minimum_col, maximum_col = centre_col - radius, centre_col + radius
    with atlas.open_catalog() as connection:
        records = {
            (int(row), int(col)): path
            for row, col, path in connection.execute(
                """SELECT region_row, region_col, artifact_path FROM macro_regions
                   WHERE region_row BETWEEN ? AND ? AND region_col BETWEEN ? AND ?""",
                (minimum_row, maximum_row, minimum_col, maximum_col),
            )
        }
    expected = {
        (row, col)
        for row in range(minimum_row, maximum_row + 1)
        for col in range(minimum_col, maximum_col + 1)
    }
    if expected - records.keys():
        raise ValueError(f"Macro mosaic is incomplete: {sorted(expected - records.keys())}")
    side = (radius * 2 + 1) * 256
    elevation = np.empty((side, side), dtype=np.float32)
    precipitation = np.empty_like(elevation)
    for (row, col), path in records.items():
        i = (row - minimum_row) * 256
        j = (col - minimum_col) * 256
        with h5py.File(path, "r") as artifact:
            elevation[i:i + 256, j:j + 256] = artifact["elevation_m"][...]
            precipitation[i:i + 256, j:j + 256] = artifact[
                "annual_precipitation_mm"
            ][...]
    return elevation, precipitation


def _extract_portals(
    atlas: HydrologyAtlas,
    analysis: MacroBasinClosureAnalysis,
    discharge: np.ndarray,
    basin_by_local: dict,
    code_by_local: dict[int, int],
    *,
    centre: RegionKey,
    local_row0: int,
    local_col0: int,
) -> tuple[MacroPortal, ...]:
    flow = analysis.routing.flow_direction
    accumulation = analysis.routing.accumulation_area_m2
    catchments = analysis.routing.catchment_id
    row1, col1 = local_row0 + 256, local_col0 + 256
    portals: list[MacroPortal] = []
    for row in range(local_row0 - 1, row1 + 1):
        for col in range(local_col0 - 1, col1 + 1):
            inside_source = local_row0 <= row < row1 and local_col0 <= col < col1
            code = int(flow[row, col])
            if not 1 <= code <= 8:
                continue
            delta_row, delta_col = D8_DIRECTION_OFFSETS[code]
            next_row, next_col = row + delta_row, col + delta_col
            inside_target = (
                local_row0 <= next_row < row1 and local_col0 <= next_col < col1
            )
            if inside_source == inside_target:
                continue
            catchment = int(catchments[row, col])
            basin = basin_by_local.get(catchment)
            if basin is None:
                continue
            if inside_source:
                source_region = centre
                global_macro_row = analysis.macro_origin_row + next_row
                global_macro_col = analysis.macro_origin_col + next_col
                destination_region = RegionKey(global_macro_row // 256, global_macro_col // 256)
                local_inside_row, local_inside_col = row - local_row0, col - local_col0
                edge_delta_row, edge_delta_col = delta_row, delta_col
                kind = "outlet"
            else:
                global_macro_row = analysis.macro_origin_row + row
                global_macro_col = analysis.macro_origin_col + col
                source_region = RegionKey(global_macro_row // 256, global_macro_col // 256)
                destination_region = centre
                local_inside_row = next_row - local_row0
                local_inside_col = next_col - local_col0
                edge_delta_row, edge_delta_col = -delta_row, -delta_col
                kind = "inflow"
            planner_row, planner_col = _project_macro_edge(
                local_inside_row, local_inside_col,
                edge_delta_row, edge_delta_col, refinement=32,
            )
            planner_row += centre.row * 8192
            planner_col += centre.col * 8192
            contract_id = atlas.add_boundary_contract(
                source_region=source_region,
                destination_region=destination_region,
                global_planner_row=planner_row,
                global_planner_col=planner_col,
                basin_id=basin.basin_id,
                upstream_area_m2=float(accumulation[row, col]),
                mean_discharge_m3s=float(discharge[row, col]),
            )
            portals.append(
                MacroPortal(
                    contract_id=contract_id,
                    kind=kind,
                    planner_row=planner_row,
                    planner_col=planner_col,
                    basin_id=basin.basin_id,
                    basin_code=code_by_local[catchment],
                    upstream_area_m2=float(accumulation[row, col]),
                    mean_discharge_m3s=float(discharge[row, col]),
                )
            )
    # Basins whose macro outlet is inside this region terminate at known ocean
    # rather than at a cross-region contract.
    for local_id, basin in basin_by_local.items():
        outlet_local_row = basin.outlet_macro_row - analysis.macro_origin_row
        outlet_local_col = basin.outlet_macro_col - analysis.macro_origin_col
        if not (
            local_row0 <= outlet_local_row < row1
            and local_col0 <= outlet_local_col < col1
        ):
            continue
        planner_row = (
            (outlet_local_row - local_row0) * 32 + 16 + centre.row * 8192
        )
        planner_col = (
            (outlet_local_col - local_col0) * 32 + 16 + centre.col * 8192
        )
        portals.append(
            MacroPortal(
                contract_id="",
                kind="ocean_outlet",
                planner_row=planner_row,
                planner_col=planner_col,
                basin_id=basin.basin_id,
                basin_code=code_by_local[local_id],
                upstream_area_m2=float(accumulation[outlet_local_row, outlet_local_col]),
                mean_discharge_m3s=float(discharge[outlet_local_row, outlet_local_col]),
            )
        )
    return tuple(portals)


def load_macro_region_constraints(
    constraint_file: str | Path,
    *,
    region: RegionKey,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load zones and materialize terminal/inflow arrays for one 240 m solve."""

    with h5py.File(constraint_file, "r") as artifact:
        if (
            int(artifact.attrs["region_row"]) != region.row
            or int(artifact.attrs["region_col"]) != region.col
        ):
            raise ValueError("Constraint artifact belongs to another region")
        zones = artifact["basin_code_240m"][...]
        relaxation = artifact["divide_relaxation_mask_240m"][...] == 1
        portal_rows = artifact["portals/planner_row"][...]
        portal_cols = artifact["portals/planner_col"][...]
        kinds = artifact["portals/kind"].asstr()[...]
        areas = artifact["portals/upstream_area_m2"][...]
        discharges = artifact["portals/mean_discharge_m3s"][...]
    terminals = np.zeros(zones.shape, dtype=bool)
    initial_area = np.zeros(zones.shape, dtype=np.float64)
    initial_discharge = np.zeros(zones.shape, dtype=np.float64)
    row_origin, col_origin = region.row * 8192, region.col * 8192
    for row, col, kind, area, discharge in zip(
        portal_rows, portal_cols, kinds, areas, discharges
    ):
        local_row, local_col = int(row) - row_origin, int(col) - col_origin
        if not (0 <= local_row < zones.shape[0] and 0 <= local_col < zones.shape[1]):
            raise ValueError("Portal lies outside its constraint region")
        if kind in {"outlet", "ocean_outlet"}:
            terminals[local_row, local_col] = True
        elif kind == "inflow":
            initial_area[local_row, local_col] += float(area)
            initial_discharge[local_row, local_col] += float(discharge)
        else:
            raise ValueError(f"Unknown macro portal kind: {kind}")
    return zones, terminals, initial_area, initial_discharge


def materialize_macro_boundary_conditions(
    constraint_file: str | Path,
    *,
    region: RegionKey,
    major_basin_codes: set[int] | frozenset[int] | None = None,
    outlet_corridor_cells: int = 0,
) -> MacroBoundaryConditions:
    """Load exact cross-region contracts without making every basin a wall.

    Cross-region inflows and outflows are always authoritative. Internal ocean
    outlets are only forced for hard, continental-scale basins; smaller basins
    are allowed to find a more accurate outlet on the 240 m coastline.
    """

    if outlet_corridor_cells < 0:
        raise ValueError("outlet_corridor_cells cannot be negative")
    with h5py.File(constraint_file, "r") as artifact:
        if (
            int(artifact.attrs["region_row"]) != region.row
            or int(artifact.attrs["region_col"]) != region.col
        ):
            raise ValueError("Constraint artifact belongs to another region")
        shape = artifact["basin_code_240m"].shape
        portal_rows = artifact["portals/planner_row"][...]
        portal_cols = artifact["portals/planner_col"][...]
        portal_codes = artifact["portals/basin_code"][...]
        kinds = artifact["portals/kind"].asstr()[...]
        areas = artifact["portals/upstream_area_m2"][...]
        discharges = artifact["portals/mean_discharge_m3s"][...]
    terminals = np.zeros(shape, dtype=bool)
    initial_area = np.zeros(shape, dtype=np.float64)
    initial_discharge = np.zeros(shape, dtype=np.float64)
    major = None if major_basin_codes is None else set(major_basin_codes)
    row_origin, col_origin = region.planner_origin
    enabled = inflows = outlets = 0
    for row, col, code, kind, area, discharge in zip(
        portal_rows, portal_cols, portal_codes, kinds, areas, discharges
    ):
        local_row, local_col = int(row) - row_origin, int(col) - col_origin
        if not (0 <= local_row < shape[0] and 0 <= local_col < shape[1]):
            raise ValueError("Portal lies outside its constraint region")
        if kind == "inflow":
            initial_area[local_row, local_col] += float(area)
            initial_discharge[local_row, local_col] += float(discharge)
            inflows += 1
            enabled += 1
        elif kind == "outlet":
            _mark_boundary_corridor(
                terminals, local_row, local_col, outlet_corridor_cells
            )
            outlets += 1
            enabled += 1
        elif kind == "ocean_outlet":
            if major is None or int(code) in major:
                terminals[local_row, local_col] = True
                outlets += 1
                enabled += 1
        else:
            raise ValueError(f"Unknown macro portal kind: {kind}")
    return MacroBoundaryConditions(
        terminal_mask=terminals,
        initial_accumulation_area_m2=initial_area,
        initial_discharge_m3s=initial_discharge,
        portal_count=enabled,
        inflow_count=inflows,
        outlet_count=outlets,
    )


def _mark_boundary_corridor(
    mask: np.ndarray, row: int, col: int, radius: int
) -> None:
    """Mark the fine candidates represented by one coarse edge crossing."""

    rows, cols = mask.shape
    if row not in (0, rows - 1) and col not in (0, cols - 1):
        # Internal ocean outlets are point constraints, never corridors.
        mask[row, col] = True
        return
    if row in (0, rows - 1):
        mask[row, max(0, col - radius):min(cols, col + radius + 1)] = True
    if col in (0, cols - 1):
        mask[max(0, row - radius):min(rows, row + radius + 1), col] = True


def build_hierarchical_routing_zones(
    atlas_directory: str | Path,
    basin_code_240m: np.ndarray,
    *,
    minimum_major_basin_area_km2: float = 25_000.0,
) -> HierarchicalRoutingZones:
    """Collapse small macro basins while preserving continental-scale divides."""

    if minimum_major_basin_area_km2 <= 0:
        raise ValueError("minimum_major_basin_area_km2 must be positive")
    atlas = HydrologyAtlas(atlas_directory)
    with atlas.open_catalog() as connection:
        rows = connection.execute(
            """SELECT c.basin_code, b.area_km2
               FROM basin_codes AS c
               JOIN macro_drainage_basins AS b ON b.basin_id = c.basin_id
               WHERE b.area_km2 >= ?""",
            (float(minimum_major_basin_area_km2),),
        ).fetchall()
        used_codes = {
            int(row[0]) for row in connection.execute("SELECT basin_code FROM basin_codes")
        }
    major_codes = frozenset(int(code) for code, _ in rows)
    minor_zone_code = 0
    while minor_zone_code in used_codes or minor_zone_code == _NODATA_U32:
        minor_zone_code += 1
    source = np.asarray(basin_code_240m, dtype=np.uint32)
    zones = np.full(source.shape, np.uint32(minor_zone_code), dtype=np.uint32)
    for code in major_codes:
        zones[source == code] = np.uint32(code)
    protected = _zone_boundary_mask(zones)
    return HierarchicalRoutingZones(
        routing_zones=zones,
        protected_divide_mask=protected,
        major_basin_codes=major_codes,
        minor_zone_code=minor_zone_code,
    )


def _zone_boundary_mask(zones: np.ndarray) -> np.ndarray:
    boundary = np.zeros(zones.shape, dtype=bool)
    different = zones[1:, :] != zones[:-1, :]
    boundary[1:, :] |= different
    boundary[:-1, :] |= different
    different = zones[:, 1:] != zones[:, :-1]
    boundary[:, 1:] |= different
    boundary[:, :-1] |= different
    return boundary


def ensure_routing_zone_terminals(
    elevation_m: np.ndarray,
    land_mask: np.ndarray,
    routing_zones: np.ndarray,
    terminal_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Give every disconnected land/zone component a valid drainage terminal."""

    elevation = np.ascontiguousarray(elevation_m, dtype=np.float32)
    land = np.ascontiguousarray(land_mask, dtype=np.bool_)
    zones = np.asarray(routing_zones, dtype=np.uint32).copy(order="C")
    terminals = np.asarray(terminal_mask, dtype=bool).copy()
    if not (elevation.shape == land.shape == zones.shape == terminals.shape):
        raise ValueError("Terminal cleanup rasters must align")
    coastline = land & scipy.ndimage.binary_dilation(~land, structure=np.ones((3, 3)))
    perimeter_land = np.zeros(land.shape, dtype=bool)
    perimeter_land[0, :] = land[0, :]
    perimeter_land[-1, :] = land[-1, :]
    perimeter_land[:, 0] = land[:, 0]
    perimeter_land[:, -1] = land[:, -1]
    coastline |= perimeter_land
    coastal_elevation = np.where(coastline, elevation, np.inf)
    coastal_added = 0
    boundary_added = 0
    reassigned_interior_components = 0
    component_count = 0
    for _ in range(8):
        labels, component_count = _label_land_zone_components(land, zones)
        present = np.unique(labels[terminals & land])
        present = present[present != 0]
        all_components = np.arange(1, component_count + 1, dtype=np.int32)
        missing = np.setdiff1d(all_components, present, assume_unique=True)
        if missing.size == 0:
            return terminals, zones, {
                "component_count": int(component_count),
                "added_coastal_terminals": coastal_added,
                "added_boundary_terminals": boundary_added,
                "reassigned_interior_components": reassigned_interior_components,
            }
        coastal_positions = scipy.ndimage.minimum_position(
            coastal_elevation, labels, index=missing
        )
        objects = scipy.ndimage.find_objects(labels)
        interior: list[int] = []
        for component_id, coast_position in zip(missing, coastal_positions):
            if np.isfinite(coastal_elevation[coast_position]):
                terminals[coast_position] = True
                coastal_added += 1
                if perimeter_land[coast_position]:
                    boundary_added += 1
            else:
                interior.append(int(component_id))
        if not interior:
            continue
        for component_id in interior:
            component_slice = objects[component_id - 1]
            if component_slice is None:
                continue
            expanded = tuple(
                slice(max(0, item.start - 1), min(size, item.stop + 1))
                for item, size in zip(component_slice, land.shape)
            )
            component = labels[expanded] == component_id
            neighbor = (
                scipy.ndimage.binary_dilation(component, structure=np.ones((3, 3)))
                & ~component
                & land[expanded]
            )
            candidates = zones[expanded][neighbor]
            current = zones[expanded][component][0]
            candidates = candidates[candidates != current]
            if candidates.size == 0:
                # An earlier reassignment in this same pass can already have
                # merged this stale component into an equal-zone neighbor.
                # Relabel on the next iteration instead of inventing a sink.
                continue
            values, counts = np.unique(candidates, return_counts=True)
            replacement = values[np.argmax(counts)]
            zone_window = zones[expanded]
            zone_window[component] = replacement
            reassigned_interior_components += 1
    raise RuntimeError("Basin-zone connectivity cleanup did not converge")


@njit(cache=True)
def _label_land_zone_components(
    land: np.ndarray, zones: np.ndarray
) -> tuple[np.ndarray, int]:
    rows, cols = land.shape
    labels = np.zeros(land.shape, dtype=np.int32)
    stack = np.empty(land.size, dtype=np.uint32)
    component = 0
    for start_row in range(rows):
        for start_col in range(cols):
            if not land[start_row, start_col] or labels[start_row, start_col] != 0:
                continue
            component += 1
            stack_size = 1
            stack[0] = np.uint32(start_row * cols + start_col)
            labels[start_row, start_col] = component
            zone = zones[start_row, start_col]
            while stack_size:
                stack_size -= 1
                index = int(stack[stack_size])
                row = index // cols
                col = index - row * cols
                for delta_row in (-1, 0, 1):
                    for delta_col in (-1, 0, 1):
                        if delta_row == 0 and delta_col == 0:
                            continue
                        next_row = row + delta_row
                        next_col = col + delta_col
                        if (
                            next_row < 0 or next_row >= rows
                            or next_col < 0 or next_col >= cols
                            or not land[next_row, next_col]
                            or zones[next_row, next_col] != zone
                            or labels[next_row, next_col] != 0
                        ):
                            continue
                        labels[next_row, next_col] = component
                        stack[stack_size] = np.uint32(next_row * cols + next_col)
                        stack_size += 1
    return labels, component


def _project_macro_edge(
    row: int, col: int, delta_row: int, delta_col: int, *, refinement: int
) -> tuple[int, int]:
    planner_row = row * refinement + refinement // 2
    planner_col = col * refinement + refinement // 2
    if delta_row < 0:
        planner_row = row * refinement
    elif delta_row > 0:
        planner_row = (row + 1) * refinement - 1
    if delta_col < 0:
        planner_col = col * refinement
    elif delta_col > 0:
        planner_col = (col + 1) * refinement - 1
    return planner_row, planner_col


@click.command("build-macro-region-constraints")
@click.argument("atlas_directory", type=click.Path(exists=True, file_okay=False))
@click.argument("surface_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--centre-row", default=0, show_default=True, type=int)
@click.option("--centre-col", default=0, show_default=True, type=int)
@click.option("--radius", default=1, show_default=True, type=click.IntRange(min=0))
@click.option("--divide-relaxation-cells", default=16, show_default=True, type=click.IntRange(min=0))
@click.option(
    "--runoff-ratio", default=DEFAULT_HYDROLOGY_PROFILE.runoff_ratio,
    show_default=True,
)
def build_macro_region_constraints_cli(**kwargs):
    """Build 240 m basin zones and cross-region river contracts."""

    output, constraints = build_macro_region_constraints(**kwargs)
    click.echo(
        f"Saved {len(constraints.portals)} portals and snapped basin zones to {output}"
    )
