"""Boundary reconciliation between the 240 m plan and 30 m routing tiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compiled_routing import CompiledRoutingResult, priority_flood_route_compiled
from .world_plan import D8_DIRECTION_OFFSETS


_NODATA_U32 = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class FlowPortal:
    kind: str
    global_row: int
    global_col: int
    regional_row: int
    regional_col: int
    catchment_id: int
    upstream_area_m2: float = 0.0


@dataclass(frozen=True)
class RegionalBoundaryConditions:
    terminal_mask: np.ndarray
    initial_accumulation_area_m2: np.ndarray
    routing_zones: np.ndarray
    portals: tuple[FlowPortal, ...]


def build_regional_boundary_conditions(
    global_flow_direction: np.ndarray,
    global_accumulation_area_m2: np.ndarray,
    global_catchment_id: np.ndarray,
    global_land_mask: np.ndarray,
    *,
    row_start: int,
    col_start: int,
    height: int,
    width: int,
    refinement: int = 8,
) -> RegionalBoundaryConditions:
    """Project global outlets, inflows, and basin ownership into a fine tile.

    A 30 m cell inherits its 240 m parent's catchment. This is deliberately a
    hard topological constraint: the regional solve can move rivers within a
    coarse basin but cannot steal area from an adjacent global basin.
    """

    flow = np.asarray(global_flow_direction, dtype=np.uint8)
    accumulation = np.asarray(global_accumulation_area_m2, dtype=np.float64)
    catchments = np.asarray(global_catchment_id, dtype=np.uint32)
    land = np.asarray(global_land_mask, dtype=bool)
    if not (flow.shape == accumulation.shape == catchments.shape == land.shape):
        raise ValueError("All global routing arrays must have the same shape")
    if refinement <= 1 or min(height, width) <= 0:
        raise ValueError("refinement must exceed one and window dimensions must be positive")
    if row_start < 0 or col_start < 0:
        raise ValueError("window origin must be non-negative")
    if row_start + height > flow.shape[0] or col_start + width > flow.shape[1]:
        raise ValueError("global window lies outside the routing grid")

    coarse_catchments = catchments[
        row_start : row_start + height,
        col_start : col_start + width,
    ]
    zones = np.repeat(np.repeat(coarse_catchments, refinement, axis=0), refinement, axis=1)
    regional_shape = (height * refinement, width * refinement)
    terminals = np.zeros(regional_shape, dtype=bool)
    injected = np.zeros(regional_shape, dtype=np.float64)
    portals: list[FlowPortal] = []

    # Every global land flow that exits the window becomes a pinned regional
    # terminal. A true global land terminal (for example a lake outlet modelled
    # as a sink) is projected to the centre of its parent cell.
    for local_row in range(height):
        for local_col in range(width):
            global_row = row_start + local_row
            global_col = col_start + local_col
            if not land[global_row, global_col]:
                continue
            code = int(flow[global_row, global_col])
            catchment_id = int(catchments[global_row, global_col])
            if code == 0:
                regional_row = local_row * refinement + refinement // 2
                regional_col = local_col * refinement + refinement // 2
            elif 1 <= code <= 8:
                delta_row, delta_col = D8_DIRECTION_OFFSETS[code]
                receiver_row = global_row + delta_row
                receiver_col = global_col + delta_col
                if (
                    row_start <= receiver_row < row_start + height
                    and col_start <= receiver_col < col_start + width
                ):
                    continue
                regional_row, regional_col = _project_edge_cell(
                    local_row, local_col, delta_row, delta_col, refinement
                )
            else:
                continue
            terminals[regional_row, regional_col] = True
            portals.append(
                FlowPortal(
                    "outlet",
                    global_row,
                    global_col,
                    regional_row,
                    regional_col,
                    catchment_id,
                )
            )

    # Inspect the one-cell ring outside the window. Any global cell whose D8
    # receiver enters the window contributes its already accumulated upstream
    # area at the matching regional boundary portal.
    for global_row, global_col in _outside_ring(
        row_start, col_start, height, width, flow.shape
    ):
        if not land[global_row, global_col]:
            continue
        code = int(flow[global_row, global_col])
        if not 1 <= code <= 8:
            continue
        delta_row, delta_col = D8_DIRECTION_OFFSETS[code]
        receiver_row = global_row + delta_row
        receiver_col = global_col + delta_col
        if not (
            row_start <= receiver_row < row_start + height
            and col_start <= receiver_col < col_start + width
        ):
            continue
        local_row = receiver_row - row_start
        local_col = receiver_col - col_start
        # Project from the receiving cell toward the source just outside.
        regional_row, regional_col = _project_edge_cell(
            local_row, local_col, -delta_row, -delta_col, refinement
        )
        upstream_area = float(accumulation[global_row, global_col])
        injected[regional_row, regional_col] += upstream_area
        portals.append(
            FlowPortal(
                "inflow",
                global_row,
                global_col,
                regional_row,
                regional_col,
                int(catchments[global_row, global_col]),
                upstream_area,
            )
        )

    return RegionalBoundaryConditions(
        terminal_mask=terminals,
        initial_accumulation_area_m2=injected,
        routing_zones=zones,
        portals=tuple(portals),
    )


def route_regional_tile(
    elevation_m: np.ndarray,
    land_mask: np.ndarray,
    boundary: RegionalBoundaryConditions,
    *,
    resolution_m: float = 30.0,
) -> CompiledRoutingResult:
    """Route one regional tile without violating the global basin graph."""

    elevation = np.asarray(elevation_m)
    land = np.asarray(land_mask, dtype=bool)
    if not (
        elevation.shape
        == land.shape
        == boundary.terminal_mask.shape
        == boundary.initial_accumulation_area_m2.shape
        == boundary.routing_zones.shape
    ):
        raise ValueError("Regional terrain and boundary arrays must have the same shape")
    # A projected terminal must be routable even when the generated shoreline
    # differs by a few 30 m cells from its 240 m parent.
    routed_land = (
        land & (boundary.routing_zones != _NODATA_U32)
    ) | boundary.terminal_mask
    return priority_flood_route_compiled(
        elevation,
        resolution_m=resolution_m,
        land_mask=routed_land,
        terminal_mask=boundary.terminal_mask,
        open_boundary=False,
        initial_accumulation_area_m2=boundary.initial_accumulation_area_m2,
        routing_zones=boundary.routing_zones,
    )


def _project_edge_cell(
    local_row: int,
    local_col: int,
    delta_row: int,
    delta_col: int,
    refinement: int,
) -> tuple[int, int]:
    base_row = local_row * refinement
    base_col = local_col * refinement
    if delta_row < 0:
        row = base_row
    elif delta_row > 0:
        row = base_row + refinement - 1
    else:
        row = base_row + refinement // 2
    if delta_col < 0:
        col = base_col
    elif delta_col > 0:
        col = base_col + refinement - 1
    else:
        col = base_col + refinement // 2
    return row, col


def _outside_ring(
    row_start: int,
    col_start: int,
    height: int,
    width: int,
    shape: tuple[int, int],
):
    row_min = max(0, row_start - 1)
    row_max = min(shape[0], row_start + height + 1)
    col_min = max(0, col_start - 1)
    col_max = min(shape[1], col_start + width + 1)
    for row in range(row_min, row_max):
        for col in range(col_min, col_max):
            if (
                row_start <= row < row_start + height
                and col_start <= col < col_start + width
            ):
                continue
            yield row, col
