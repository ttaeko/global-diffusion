"""Deterministic reference flow routing used to validate the world planner.

This module prioritizes explicit invariants over continent-scale speed.  It is
appropriate for unit tests, DEM calibration windows, and individual regional
tiles.  The same results provide the contract for the later tiled/compiled
global implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Iterator

import numpy as np

from .world_plan import D8_DIRECTION_OFFSETS


_OFFSET_TO_D8 = {offset: code for code, offset in D8_DIRECTION_OFFSETS.items()}


@dataclass(frozen=True)
class RoutingResult:
    elevation_conditioned_m: np.ndarray
    elevation_correction_m: np.ndarray
    flow_direction: np.ndarray
    receiver: np.ndarray
    accumulation_area_m2: np.ndarray
    catchment_id: np.ndarray
    processing_order: np.ndarray


def priority_flood_route(
    elevation_m: np.ndarray,
    *,
    resolution_m: float,
    land_mask: np.ndarray | None = None,
    terminal_mask: np.ndarray | None = None,
    open_boundary: bool = True,
) -> RoutingResult:
    """Condition depressions and construct an acyclic D8 receiver graph.

    Ocean cells, explicitly supplied terminals, and optionally valid boundary
    cells are roots.  A land cell receives the lowest spill elevation required
    to connect it to one of those roots.  Its receiver is the already-processed
    cell through which that spill was reached, which guarantees termination
    even across a perfectly flat filled depression.
    """

    elevation = np.asarray(elevation_m, dtype=np.float64)
    if elevation.ndim != 2 or min(elevation.shape) < 1:
        raise ValueError("elevation_m must be a non-empty 2D array")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    valid = np.isfinite(elevation)
    if not np.any(valid):
        raise ValueError("elevation_m has no finite cells")

    if land_mask is None:
        land = valid.copy()
    else:
        land = np.asarray(land_mask, dtype=bool)
        if land.shape != elevation.shape:
            raise ValueError("land_mask shape does not match elevation_m")
        land &= valid

    terminals = valid & ~land
    if terminal_mask is not None:
        requested = np.asarray(terminal_mask, dtype=bool)
        if requested.shape != elevation.shape:
            raise ValueError("terminal_mask shape does not match elevation_m")
        terminals |= requested & valid
    if open_boundary:
        terminals[0, :] |= valid[0, :]
        terminals[-1, :] |= valid[-1, :]
        terminals[:, 0] |= valid[:, 0]
        terminals[:, -1] |= valid[:, -1]
    if not np.any(terminals):
        raise ValueError("Routing requires an ocean, explicit terminal, or open boundary")

    rows, cols = elevation.shape
    conditioned = elevation.copy()
    receiver = np.full(elevation.size, -2, dtype=np.int64)
    visited = np.zeros(elevation.shape, dtype=bool)
    heap: list[tuple[float, int]] = []

    for flat_index in np.flatnonzero(terminals):
        row, col = divmod(int(flat_index), cols)
        visited[row, col] = True
        receiver[flat_index] = -1
        heapq.heappush(heap, (float(conditioned[row, col]), int(flat_index)))

    order: list[int] = []
    while heap:
        spill_elevation, flat_index = heapq.heappop(heap)
        order.append(flat_index)
        row, col = divmod(flat_index, cols)
        for next_row, next_col in _neighbors(row, col, rows, cols):
            if visited[next_row, next_col] or not valid[next_row, next_col]:
                continue
            visited[next_row, next_col] = True
            next_index = next_row * cols + next_col
            next_elevation = max(float(elevation[next_row, next_col]), spill_elevation)
            conditioned[next_row, next_col] = next_elevation
            receiver[next_index] = flat_index
            heapq.heappush(heap, (next_elevation, next_index))

    if np.any(valid & ~visited):
        raise RuntimeError("Some valid cells are disconnected from every routing terminal")

    flow_direction = np.full(elevation.shape, 255, dtype=np.uint8)
    for flat_index in np.flatnonzero(valid):
        downstream = int(receiver[flat_index])
        if downstream == -1:
            flow_direction.flat[flat_index] = 0
            continue
        row, col = divmod(int(flat_index), cols)
        next_row, next_col = divmod(downstream, cols)
        flow_direction[row, col] = _OFFSET_TO_D8[(next_row - row, next_col - col)]

    processing_order = np.asarray(order, dtype=np.int64)
    accumulation = _accumulate_area(
        receiver, processing_order, land.ravel(), resolution_m * resolution_m
    )
    catchments = _label_catchments(receiver, processing_order, valid.ravel())
    correction = conditioned - elevation
    correction[~valid] = np.nan
    return RoutingResult(
        elevation_conditioned_m=conditioned.astype(np.float32),
        elevation_correction_m=correction.astype(np.float32),
        flow_direction=flow_direction,
        receiver=receiver.reshape(elevation.shape),
        accumulation_area_m2=accumulation.reshape(elevation.shape),
        catchment_id=catchments.reshape(elevation.shape),
        processing_order=processing_order,
    )


def strahler_order(
    receiver: np.ndarray,
    processing_order: np.ndarray,
    channel_mask: np.ndarray,
) -> np.ndarray:
    """Compute Strahler order over an acyclic receiver graph."""

    receivers = np.asarray(receiver, dtype=np.int64).ravel()
    channels = np.asarray(channel_mask, dtype=bool)
    if channels.size != receivers.size:
        raise ValueError("channel_mask shape does not match receiver")
    order = np.zeros(receivers.size, dtype=np.uint8)
    maximum_upstream = np.zeros(receivers.size, dtype=np.uint8)
    maximum_count = np.zeros(receivers.size, dtype=np.uint8)

    for cell in np.asarray(processing_order, dtype=np.int64)[::-1]:
        if not channels.ravel()[cell]:
            continue
        upstream_order = int(maximum_upstream[cell])
        cell_order = 1 if upstream_order == 0 else upstream_order
        if upstream_order > 0 and maximum_count[cell] >= 2:
            cell_order += 1
        cell_order = min(cell_order, np.iinfo(np.uint8).max)
        order[cell] = cell_order
        downstream = int(receivers[cell])
        if downstream < 0 or not channels.ravel()[downstream]:
            continue
        if cell_order > maximum_upstream[downstream]:
            maximum_upstream[downstream] = cell_order
            maximum_count[downstream] = 1
        elif cell_order == maximum_upstream[downstream]:
            maximum_count[downstream] += 1
    return order.reshape(channels.shape)


def select_channels(
    accumulation_area_m2: np.ndarray,
    *,
    minimum_area_km2: float,
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Select candidate channel cells by contributing area."""

    if minimum_area_km2 < 0:
        raise ValueError("minimum_area_km2 must be non-negative")
    accumulation = np.asarray(accumulation_area_m2)
    selected = accumulation >= minimum_area_km2 * 1_000_000.0
    if land_mask is not None:
        land = np.asarray(land_mask, dtype=bool)
        if land.shape != accumulation.shape:
            raise ValueError("land_mask shape does not match accumulation")
        selected &= land
    return selected


def trace_receiver_path(receiver: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Return a cell-to-terminal path, raising if the graph contains a cycle."""

    receivers = np.asarray(receiver, dtype=np.int64)
    rows, cols = receivers.shape
    row, col = start
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError("start lies outside receiver grid")
    path: list[tuple[int, int]] = []
    seen: set[int] = set()
    cell = row * cols + col
    while cell >= 0:
        if cell in seen:
            raise RuntimeError("Receiver graph contains a cycle")
        seen.add(cell)
        path.append(divmod(cell, cols))
        cell = int(receivers.ravel()[cell])
    return path


def _neighbors(row: int, col: int, rows: int, cols: int) -> Iterator[tuple[int, int]]:
    for delta_row, delta_col in D8_DIRECTION_OFFSETS.values():
        next_row = row + delta_row
        next_col = col + delta_col
        if 0 <= next_row < rows and 0 <= next_col < cols:
            yield next_row, next_col


def _accumulate_area(
    receiver: np.ndarray,
    processing_order: np.ndarray,
    land: np.ndarray,
    cell_area_m2: float,
) -> np.ndarray:
    accumulation = land.astype(np.float64) * cell_area_m2
    for cell in processing_order[::-1]:
        downstream = int(receiver[cell])
        if downstream >= 0:
            accumulation[downstream] += accumulation[cell]
    return accumulation


def _label_catchments(
    receiver: np.ndarray,
    processing_order: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    nodata = np.iinfo(np.uint32).max
    catchments = np.full(receiver.size, nodata, dtype=np.uint32)
    next_id = 0
    for cell in processing_order:
        downstream = int(receiver[cell])
        if downstream < 0:
            catchments[cell] = next_id
            next_id += 1
        else:
            catchments[cell] = catchments[downstream]
    catchments[~valid] = nodata
    return catchments
