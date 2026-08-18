"""Detect when macro landmasses are complete enough to freeze drainage.

An elevation mosaic may be expanded indefinitely. A connected land component
is safe to freeze only when no cell touches the currently sampled perimeter;
otherwise an unseen continuation could change its continental drainage graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from .atlas import REGION_MACRO_CELLS, RegionKey, stable_world_id
from .compiled_routing import CompiledRoutingResult, priority_flood_route_compiled


@dataclass(frozen=True)
class MacroLandmass:
    provisional_label: int
    landmass_id: str | None
    cell_count: int
    min_macro_row: int
    min_macro_col: int
    max_macro_row: int
    max_macro_col: int
    touches_north: bool
    touches_east: bool
    touches_south: bool
    touches_west: bool
    required_regions: tuple[RegionKey, ...]

    @property
    def closed(self) -> bool:
        return not (
            self.touches_north
            or self.touches_east
            or self.touches_south
            or self.touches_west
        )


@dataclass(frozen=True)
class MacroClosureAnalysis:
    labels: np.ndarray
    landmasses: tuple[MacroLandmass, ...]
    macro_origin_row: int
    macro_origin_col: int

    @property
    def required_regions(self) -> tuple[RegionKey, ...]:
        return tuple(sorted({key for item in self.landmasses for key in item.required_regions}))


@dataclass(frozen=True)
class ContinentalBasin:
    local_catchment_id: int
    basin_id: str
    outlet_macro_row: int
    outlet_macro_col: int
    area_km2: float
    maximum_accumulation_km2: float


@dataclass(frozen=True)
class ContinentalDrainage:
    routing: CompiledRoutingResult
    landmass_id: str
    basins: tuple[ContinentalBasin, ...]
    divide_mask: np.ndarray
    landmass_mask: np.ndarray


@dataclass(frozen=True)
class MacroDrainageBasin:
    local_catchment_id: int
    basin_id: str
    outlet_macro_row: int
    outlet_macro_col: int
    outlet_kind: str
    cell_count: int
    area_km2: float
    maximum_accumulation_km2: float
    min_macro_row: int
    min_macro_col: int
    max_macro_row: int
    max_macro_col: int
    closed: bool
    required_regions: tuple[RegionKey, ...]


@dataclass(frozen=True)
class MacroBasinClosureAnalysis:
    routing: CompiledRoutingResult
    basins: tuple[MacroDrainageBasin, ...]
    divide_mask: np.ndarray
    land_mask: np.ndarray
    macro_origin_row: int
    macro_origin_col: int

    @property
    def required_regions(self) -> tuple[RegionKey, ...]:
        return tuple(sorted({key for basin in self.basins for key in basin.required_regions}))


def analyze_macro_landmass_closure(
    elevation_m: np.ndarray,
    *,
    world_seed: int,
    macro_origin_row: int,
    macro_origin_col: int,
    focus_mask: np.ndarray | None = None,
) -> MacroClosureAnalysis:
    """Classify sampled landmasses and identify regions needed to close them.

    Connectivity is eight-neighbour because a diagonal ridge/land bridge must
    not be split merely by raster orientation. If ``focus_mask`` is supplied,
    only components intersecting it are returned; this lets an expansion close
    the continents relevant to a requested playable region without requiring
    every distant component in the mosaic to be complete.
    """

    elevation = np.asarray(elevation_m, dtype=np.float32)
    if elevation.ndim != 2 or min(elevation.shape) == 0:
        raise ValueError("elevation_m must be a non-empty 2D array")
    land = np.isfinite(elevation) & (elevation > 0)
    if focus_mask is not None:
        focus = np.asarray(focus_mask, dtype=bool)
        if focus.shape != elevation.shape:
            raise ValueError("focus_mask must align with elevation_m")
    else:
        focus = None
    labels, count = scipy.ndimage.label(land, structure=np.ones((3, 3), dtype=np.uint8))
    records: list[MacroLandmass] = []
    rows, cols = elevation.shape
    for label in range(1, count + 1):
        mask = labels == label
        if focus is not None and not np.any(mask & focus):
            continue
        local_rows, local_cols = np.nonzero(mask)
        row_min, row_max = int(local_rows.min()), int(local_rows.max())
        col_min, col_max = int(local_cols.min()), int(local_cols.max())
        north = bool(np.any(mask[0, :]))
        south = bool(np.any(mask[-1, :]))
        west = bool(np.any(mask[:, 0]))
        east = bool(np.any(mask[:, -1]))
        required = _required_neighbor_regions(
            mask,
            macro_origin_row=macro_origin_row,
            macro_origin_col=macro_origin_col,
        )
        closed = not (north or south or west or east)
        global_row_min = macro_origin_row + row_min
        global_col_min = macro_origin_col + col_min
        identity = None
        if closed:
            identity = stable_world_id(
                world_seed, "landmass", global_row_min, global_col_min
            )
        records.append(
            MacroLandmass(
                provisional_label=label,
                landmass_id=identity,
                cell_count=int(mask.sum()),
                min_macro_row=global_row_min,
                min_macro_col=global_col_min,
                max_macro_row=macro_origin_row + row_max,
                max_macro_col=macro_origin_col + col_max,
                touches_north=north,
                touches_east=east,
                touches_south=south,
                touches_west=west,
                required_regions=required,
            )
        )
    return MacroClosureAnalysis(
        labels=labels.astype(np.uint32),
        landmasses=tuple(records),
        macro_origin_row=int(macro_origin_row),
        macro_origin_col=int(macro_origin_col),
    )


def route_closed_continental_basins(
    elevation_m: np.ndarray,
    landmass_mask: np.ndarray,
    *,
    world_seed: int,
    macro_origin_row: int,
    macro_origin_col: int,
) -> ContinentalDrainage:
    """Route a fully enclosed macro landmass and freeze ocean-basin identities."""

    elevation = np.asarray(elevation_m, dtype=np.float32)
    land = np.asarray(landmass_mask, dtype=bool)
    if elevation.ndim != 2 or land.shape != elevation.shape:
        raise ValueError("landmass_mask must align with a 2D elevation raster")
    if not np.any(land):
        raise ValueError("landmass_mask is empty")
    if np.any(land[0]) or np.any(land[-1]) or np.any(land[:, 0]) or np.any(land[:, -1]):
        raise ValueError("Continental drainage cannot freeze a landmass touching coverage")
    component_labels, component_count = scipy.ndimage.label(
        land, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if component_count != 1:
        raise ValueError("landmass_mask must contain exactly one connected component")
    local_rows, local_cols = np.nonzero(land)
    anchor_row = macro_origin_row + int(local_rows.min())
    anchor_col = macro_origin_col + int(local_cols.min())
    landmass_id = stable_world_id(
        world_seed, "landmass", anchor_row, anchor_col
    )
    routing = priority_flood_route_compiled(
        elevation,
        resolution_m=7680.0,
        land_mask=land,
        open_boundary=False,
    )
    divide = _catchment_divides(routing.catchment_id, land)
    cell_area_m2 = 7680.0**2
    basins: list[ContinentalBasin] = []
    for catchment in np.unique(routing.catchment_id[land]):
        local_id = int(catchment)
        basin_land = land & (routing.catchment_id == catchment)
        terminals = np.argwhere(
            (routing.catchment_id == catchment) & (routing.flow_direction == 0)
        )
        if terminals.size == 0:
            raise RuntimeError(f"Macro catchment {local_id} has no terminal")
        outlet_row, outlet_col = map(int, terminals[0])
        global_outlet_row = macro_origin_row + outlet_row
        global_outlet_col = macro_origin_col + outlet_col
        basins.append(
            ContinentalBasin(
                local_catchment_id=local_id,
                basin_id=stable_world_id(
                    world_seed,
                    "continental_basin",
                    anchor_row,
                    anchor_col,
                    global_outlet_row,
                    global_outlet_col,
                ),
                outlet_macro_row=global_outlet_row,
                outlet_macro_col=global_outlet_col,
                area_km2=float(np.count_nonzero(basin_land) * cell_area_m2 / 1e6),
                maximum_accumulation_km2=float(
                    np.max(routing.accumulation_area_m2[basin_land]) / 1e6
                ),
            )
        )
    return ContinentalDrainage(
        routing=routing,
        landmass_id=landmass_id,
        basins=tuple(sorted(basins, key=lambda basin: basin.basin_id)),
        divide_mask=divide,
        landmass_mask=land.copy(),
    )


def analyze_macro_basin_closure(
    elevation_m: np.ndarray,
    *,
    world_seed: int,
    macro_origin_row: int,
    macro_origin_col: int,
    focus_mask: np.ndarray,
) -> MacroBasinClosureAnalysis:
    """Freeze ocean-draining catchments without requiring a finite continent."""

    elevation = np.asarray(elevation_m, dtype=np.float32)
    focus = np.asarray(focus_mask, dtype=bool)
    if elevation.ndim != 2 or focus.shape != elevation.shape:
        raise ValueError("focus_mask must align with a 2D elevation raster")
    land = np.isfinite(elevation) & (elevation > 0)
    if not np.any(focus & land):
        raise ValueError("focus_mask contains no land")
    routing = priority_flood_route_compiled(
        elevation,
        resolution_m=7680.0,
        land_mask=land,
        # Land on the sampled perimeter must remain routable inward. Treating
        # it as an artificial outlet would hide possible upstream continuation
        # and incorrectly mark truncated basins as closed.
        open_boundary=False,
    )
    focus_ids = np.unique(routing.catchment_id[focus & land])
    cell_area_km2 = 7680.0**2 / 1e6
    basins: list[MacroDrainageBasin] = []
    for catchment in focus_ids:
        local_id = int(catchment)
        basin_land = land & (routing.catchment_id == catchment)
        local_rows, local_cols = np.nonzero(basin_land)
        terminals = np.argwhere(
            (routing.catchment_id == catchment) & (routing.flow_direction == 0)
        )
        if terminals.size == 0:
            raise RuntimeError(f"Macro catchment {local_id} has no terminal")
        # The catchment label is born at one terminal. Keep lexicographic
        # selection defensive for numerical shelf cases.
        outlet_row, outlet_col = map(int, terminals[0])
        outlet_kind = "ocean" if not land[outlet_row, outlet_col] else "boundary"
        touches_perimeter = bool(
            np.any(basin_land[0, :])
            or np.any(basin_land[-1, :])
            or np.any(basin_land[:, 0])
            or np.any(basin_land[:, -1])
        )
        closed = outlet_kind == "ocean" and not touches_perimeter
        required = () if closed else _required_neighbor_regions(
            basin_land,
            macro_origin_row=macro_origin_row,
            macro_origin_col=macro_origin_col,
        )
        global_outlet_row = macro_origin_row + outlet_row
        global_outlet_col = macro_origin_col + outlet_col
        # Outlet plus the lexicographically first upstream cell separates the
        # rare case of two independently routed basins sharing one ocean cell.
        anchor_row = macro_origin_row + int(local_rows.min())
        anchor_candidates = local_cols[local_rows == local_rows.min()]
        anchor_col = macro_origin_col + int(anchor_candidates.min())
        basin_id = stable_world_id(
            world_seed,
            "macro_drainage_basin",
            global_outlet_row,
            global_outlet_col,
            anchor_row,
            anchor_col,
        )
        basins.append(
            MacroDrainageBasin(
                local_catchment_id=local_id,
                basin_id=basin_id,
                outlet_macro_row=global_outlet_row,
                outlet_macro_col=global_outlet_col,
                outlet_kind=outlet_kind,
                cell_count=int(local_rows.size),
                area_km2=float(local_rows.size * cell_area_km2),
                maximum_accumulation_km2=float(
                    np.max(routing.accumulation_area_m2[basin_land]) / 1e6
                ),
                min_macro_row=macro_origin_row + int(local_rows.min()),
                min_macro_col=macro_origin_col + int(local_cols.min()),
                max_macro_row=macro_origin_row + int(local_rows.max()),
                max_macro_col=macro_origin_col + int(local_cols.max()),
                closed=closed,
                required_regions=required,
            )
        )
    return MacroBasinClosureAnalysis(
        routing=routing,
        basins=tuple(sorted(basins, key=lambda basin: basin.basin_id)),
        divide_mask=_catchment_divides(routing.catchment_id, land),
        land_mask=land,
        macro_origin_row=int(macro_origin_row),
        macro_origin_col=int(macro_origin_col),
    )


def _catchment_divides(catchments: np.ndarray, land: np.ndarray) -> np.ndarray:
    divide = np.zeros(land.shape, dtype=bool)
    rows, cols = land.shape
    for delta_row, delta_col in ((0, 1), (1, 0), (1, 1), (1, -1)):
        source_rows = slice(max(0, -delta_row), min(rows, rows - delta_row))
        source_cols = slice(max(0, -delta_col), min(cols, cols - delta_col))
        target_rows = slice(max(0, delta_row), min(rows, rows + delta_row))
        target_cols = slice(max(0, delta_col), min(cols, cols + delta_col))
        different = (
            land[source_rows, source_cols]
            & land[target_rows, target_cols]
            & (catchments[source_rows, source_cols] != catchments[target_rows, target_cols])
        )
        divide[source_rows, source_cols] |= different
        divide[target_rows, target_cols] |= different
    return divide


def _required_neighbor_regions(
    component: np.ndarray,
    *,
    macro_origin_row: int,
    macro_origin_col: int,
) -> tuple[RegionKey, ...]:
    rows, cols = component.shape
    required: set[RegionKey] = set()

    def add(global_row: int, global_col: int) -> None:
        required.add(
            RegionKey(
                global_row // REGION_MACRO_CELLS,
                global_col // REGION_MACRO_CELLS,
            )
        )

    for col in np.flatnonzero(component[0]):
        add(macro_origin_row - 1, macro_origin_col + int(col))
    for col in np.flatnonzero(component[-1]):
        add(macro_origin_row + rows, macro_origin_col + int(col))
    for row in np.flatnonzero(component[:, 0]):
        add(macro_origin_row + int(row), macro_origin_col - 1)
    for row in np.flatnonzero(component[:, -1]):
        add(macro_origin_row + int(row), macro_origin_col + cols)

    # Eight-neighbour connectivity can continue through a diagonal corner even
    # when the orthogonal neighbor's touching cell is water.
    if component[0, 0]:
        add(macro_origin_row - 1, macro_origin_col - 1)
    if component[0, -1]:
        add(macro_origin_row - 1, macro_origin_col + cols)
    if component[-1, 0]:
        add(macro_origin_row + rows, macro_origin_col - 1)
    if component[-1, -1]:
        add(macro_origin_row + rows, macro_origin_col + cols)
    return tuple(sorted(required))
