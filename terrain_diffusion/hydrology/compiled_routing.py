"""Memory-bounded compiled priority-flood routing for large planner grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


_DROW = np.asarray((0, 0, 1, 1, 1, 0, -1, -1, -1), dtype=np.int8)
_DCOL = np.asarray((0, 1, 1, 0, -1, -1, -1, 0, 1), dtype=np.int8)
_NODATA_U32 = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class CompiledRoutingResult:
    """Routing products without a memory-heavy per-cell int64 receiver."""

    elevation_conditioned_m: np.ndarray
    elevation_correction_m: np.ndarray
    flow_direction: np.ndarray
    accumulation_area_m2: np.ndarray
    catchment_id: np.ndarray
    processing_order: np.ndarray
    outlet_count: int


@dataclass(frozen=True)
class RoutingMemoryEstimate:
    cells: int
    construction_peak_bytes: int
    products_bytes: int

    @property
    def construction_peak_gib(self) -> float:
        return self.construction_peak_bytes / 1024**3

    @property
    def products_gib(self) -> float:
        return self.products_bytes / 1024**3


def estimate_routing_memory(shape: tuple[int, int]) -> RoutingMemoryEstimate:
    """Estimate the principal array memory for a compiled global solve."""

    cells = int(shape[0]) * int(shape[1])
    if cells <= 0:
        raise ValueError("shape must be positive")
    # Input f32, conditioned f32, active/land/visited bool, flow u8,
    # heap u32, and processing order u32. Temporary masks can add a little.
    construction = cells * (4 + 4 + 1 + 1 + 1 + 1 + 4 + 4)
    # Final conditioned/correction f32, flow u8, accumulation f64,
    # catchment u32, and order u32.
    products = cells * (4 + 4 + 1 + 8 + 4 + 4)
    return RoutingMemoryEstimate(cells, construction, products)


def priority_flood_route_compiled(
    elevation_m: np.ndarray,
    *,
    resolution_m: float,
    land_mask: np.ndarray | None = None,
    terminal_mask: np.ndarray | None = None,
    open_boundary: bool = True,
    initial_accumulation_area_m2: np.ndarray | None = None,
    routing_zones: np.ndarray | None = None,
) -> CompiledRoutingResult:
    """Route a large grid using compact D8 state and Numba kernels.

    Ocean is marked terminal but only coastline ocean cells enter the priority
    queue. This is equivalent for land routing and avoids filling the heap with
    the interior of a large sea. `initial_accumulation_area_m2` can inject
    upstream area at regional tile entrances during hierarchical routing.
    """

    elevation = np.ascontiguousarray(elevation_m, dtype=np.float32)
    if elevation.ndim != 2 or min(elevation.shape) < 1:
        raise ValueError("elevation_m must be a non-empty 2D array")
    if elevation.size > np.iinfo(np.uint32).max:
        raise ValueError("A routing grid cannot exceed uint32 address space")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")

    valid = np.isfinite(elevation)
    if land_mask is None:
        land = valid.copy()
    else:
        land = np.ascontiguousarray(land_mask, dtype=np.bool_)
        if land.shape != elevation.shape:
            raise ValueError("land_mask shape does not match elevation_m")
        land &= valid
    if not np.any(land):
        raise ValueError("land_mask contains no routable cells")

    active_terminals = np.zeros(elevation.shape, dtype=np.bool_)
    ocean = valid & ~land
    use_zones = routing_zones is not None
    # Coastline ocean remains a valid terminal in zoned solves. Its zone code
    # decides which adjacent land sector it can seed; NODATA ocean used by
    # nested regional solves therefore remains isolated automatically.
    active_terminals |= _coastline_ocean(ocean, land)
    if terminal_mask is not None:
        requested = np.asarray(terminal_mask, dtype=np.bool_)
        if requested.shape != elevation.shape:
            raise ValueError("terminal_mask shape does not match elevation_m")
        active_terminals |= requested & valid
    if open_boundary:
        active_terminals[0, :] |= valid[0, :]
        active_terminals[-1, :] |= valid[-1, :]
        active_terminals[:, 0] |= valid[:, 0]
        active_terminals[:, -1] |= valid[:, -1]
    if not np.any(active_terminals):
        raise ValueError("Routing requires a coastline, explicit terminal, or open boundary")

    if routing_zones is None:
        zones = np.zeros((1, 1), dtype=np.uint32)
    else:
        zones = np.ascontiguousarray(routing_zones, dtype=np.uint32)
        if zones.shape != elevation.shape:
            raise ValueError("routing_zones shape does not match elevation_m")
        if np.any(land & (zones == _NODATA_U32)):
            raise ValueError("Every land cell requires a valid routing zone")

    conditioned = elevation.copy()
    flow_direction = np.full(elevation.shape, 255, dtype=np.uint8)
    flow_direction[ocean] = 0
    visited = np.ascontiguousarray(~land)
    heap = np.empty(elevation.size, dtype=np.uint32)
    processing_order = np.empty(elevation.size, dtype=np.uint32)

    order_length, visited_land = _priority_flood_kernel(
        conditioned,
        land,
        np.ascontiguousarray(active_terminals),
        visited,
        flow_direction,
        heap,
        processing_order,
        zones,
        use_zones,
    )
    land_count = int(np.count_nonzero(land))
    if visited_land != land_count:
        raise RuntimeError(
            f"Only {visited_land} of {land_count} land cells reached a routing terminal"
        )
    processing_order = processing_order[:order_length].copy()
    del heap, visited, active_terminals, ocean

    if initial_accumulation_area_m2 is None:
        initial = np.zeros(elevation.shape, dtype=np.float64)
    else:
        initial = np.array(
            initial_accumulation_area_m2,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if initial.shape != elevation.shape:
            raise ValueError("initial_accumulation_area_m2 shape does not match elevation_m")
        if np.any(initial < 0) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_accumulation_area_m2 must be finite and non-negative")

    accumulation = initial
    accumulation[land] += float(resolution_m) ** 2
    _accumulate_d8_kernel(flow_direction, processing_order, accumulation)
    if use_zones:
        catchments = np.full(elevation.shape, _NODATA_U32, dtype=np.uint32)
        catchments[land] = zones[land]
        outlet_count = np.unique(zones[land]).size
    else:
        catchments, outlet_count = _label_catchments_kernel(
            flow_direction, processing_order, valid
        )
    correction = conditioned - elevation
    correction[~valid] = np.nan
    return CompiledRoutingResult(
        elevation_conditioned_m=conditioned,
        elevation_correction_m=correction,
        flow_direction=flow_direction,
        accumulation_area_m2=accumulation,
        catchment_id=catchments,
        processing_order=processing_order,
        outlet_count=int(outlet_count),
    )


def receiver_indices_from_d8(flow_direction: np.ndarray) -> np.ndarray:
    """Expand compact D8 directions to flat int64 receivers for diagnostics."""

    flow = np.asarray(flow_direction, dtype=np.uint8)
    if flow.ndim != 2:
        raise ValueError("flow_direction must be 2D")
    receiver = np.full(flow.shape, -2, dtype=np.int64)
    rows, cols = flow.shape
    for code in range(1, 9):
        source_row, source_col = np.nonzero(flow == code)
        target_row = source_row + int(_DROW[code])
        target_col = source_col + int(_DCOL[code])
        if (
            np.any(target_row < 0)
            or np.any(target_row >= rows)
            or np.any(target_col < 0)
            or np.any(target_col >= cols)
        ):
            raise ValueError(f"D8 code {code} points outside the grid")
        receiver[source_row, source_col] = target_row * cols + target_col
    receiver[flow == 0] = -1
    return receiver


def strahler_order_d8(
    flow_direction: np.ndarray,
    processing_order: np.ndarray,
    channel_mask: np.ndarray,
) -> np.ndarray:
    """Compute Strahler order directly from compact D8 directions."""

    flow = np.ascontiguousarray(flow_direction, dtype=np.uint8)
    channels = np.ascontiguousarray(channel_mask, dtype=np.bool_)
    order = np.ascontiguousarray(processing_order, dtype=np.uint32)
    if flow.shape != channels.shape:
        raise ValueError("channel_mask shape does not match flow_direction")
    return _strahler_d8_kernel(flow, order, channels)


def accumulate_values_d8(
    flow_direction: np.ndarray,
    processing_order: np.ndarray,
    local_values: np.ndarray,
) -> np.ndarray:
    """Accumulate any non-negative extensive value down an existing D8 graph."""

    flow = np.ascontiguousarray(flow_direction, dtype=np.uint8)
    order = np.ascontiguousarray(processing_order, dtype=np.uint32)
    values = np.ascontiguousarray(local_values, dtype=np.float64)
    if flow.shape != values.shape:
        raise ValueError("local_values shape does not match flow_direction")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("local_values must be finite and non-negative")
    _accumulate_d8_kernel(flow, order, values)
    return values


def processing_order_from_d8(
    flow_direction: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Reconstruct a deterministic upstream-to-downstream topological order."""

    flow = np.ascontiguousarray(flow_direction, dtype=np.uint8)
    active = np.ascontiguousarray(active_mask, dtype=np.bool_)
    if flow.ndim != 2 or active.shape != flow.shape:
        raise ValueError("flow_direction and active_mask must be aligned 2D arrays")
    order, length = _topological_order_kernel(flow, active)
    active_count = int(np.count_nonzero(active))
    if length != active_count:
        raise ValueError("Active D8 graph contains a cycle or invalid receiver")
    return order[:length].copy()


def _coastline_ocean(ocean: np.ndarray, land: np.ndarray) -> np.ndarray:
    adjacent_land = np.zeros(ocean.shape, dtype=np.bool_)
    rows, cols = ocean.shape
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue
            source_rows = slice(max(0, -delta_row), min(rows, rows - delta_row))
            source_cols = slice(max(0, -delta_col), min(cols, cols - delta_col))
            target_rows = slice(max(0, delta_row), min(rows, rows + delta_row))
            target_cols = slice(max(0, delta_col), min(cols, cols + delta_col))
            adjacent_land[target_rows, target_cols] |= land[source_rows, source_cols]
    return ocean & adjacent_land


@njit(cache=True)
def _topological_order_kernel(
    flow: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, int]:
    rows, cols = flow.shape
    indegree = np.zeros(flow.shape, dtype=np.uint8)
    active_count = 0
    for row in range(rows):
        for col in range(cols):
            if not active[row, col]:
                continue
            active_count += 1
            code = int(flow[row, col])
            if code < 1 or code > 8:
                continue
            next_row = row + _DROW[code]
            next_col = col + _DCOL[code]
            if (
                0 <= next_row < rows and 0 <= next_col < cols
                and active[next_row, next_col]
            ):
                indegree[next_row, next_col] += 1
    queue = np.empty(active_count, dtype=np.uint32)
    tail = 0
    for row in range(rows):
        for col in range(cols):
            if active[row, col] and indegree[row, col] == 0:
                queue[tail] = np.uint32(row * cols + col)
                tail += 1
    head = 0
    while head < tail:
        index = int(queue[head])
        head += 1
        row = index // cols
        col = index - row * cols
        code = int(flow[row, col])
        if code < 1 or code > 8:
            continue
        next_row = row + _DROW[code]
        next_col = col + _DCOL[code]
        if not (
            0 <= next_row < rows and 0 <= next_col < cols
            and active[next_row, next_col]
        ):
            continue
        indegree[next_row, next_col] -= 1
        if indegree[next_row, next_col] == 0:
            queue[tail] = np.uint32(next_row * cols + next_col)
            tail += 1
    return queue, tail


@njit(cache=True)
def _priority_flood_kernel(
    conditioned: np.ndarray,
    land: np.ndarray,
    terminals: np.ndarray,
    visited: np.ndarray,
    flow: np.ndarray,
    heap: np.ndarray,
    order: np.ndarray,
    zones: np.ndarray,
    use_zones: bool,
) -> tuple[int, int]:
    rows, cols = conditioned.shape
    heap_size = 0
    visited_land = 0

    for row in range(rows):
        for col in range(cols):
            if not terminals[row, col]:
                continue
            index = row * cols + col
            if land[row, col] and not visited[row, col]:
                visited_land += 1
            visited[row, col] = True
            flow[row, col] = 0
            heap_size = _heap_push(heap, heap_size, index, conditioned, cols)

    order_size = 0
    while heap_size:
        index, heap_size = _heap_pop(heap, heap_size, conditioned, cols)
        order[order_size] = index
        order_size += 1
        row = index // cols
        col = index - row * cols
        spill = conditioned[row, col]

        for code in range(1, 9):
            next_row = row + _DROW[code]
            next_col = col + _DCOL[code]
            if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                continue
            if visited[next_row, next_col] or not land[next_row, next_col]:
                continue
            if use_zones and zones[next_row, next_col] != zones[row, col]:
                continue
            visited[next_row, next_col] = True
            visited_land += 1
            next_index = next_row * cols + next_col
            if conditioned[next_row, next_col] < spill:
                conditioned[next_row, next_col] = spill
            # The neighbor drains back to the current cell.
            flow[next_row, next_col] = ((code + 3) % 8) + 1
            heap_size = _heap_push(
                heap, heap_size, next_index, conditioned, cols
            )
    return order_size, visited_land


@njit(cache=True)
def _heap_less(
    first: int,
    second: int,
    conditioned: np.ndarray,
    cols: int,
) -> bool:
    first_value = conditioned[first // cols, first % cols]
    second_value = conditioned[second // cols, second % cols]
    return first_value < second_value or (
        first_value == second_value and first < second
    )


@njit(cache=True)
def _heap_push(
    heap: np.ndarray,
    heap_size: int,
    index: int,
    conditioned: np.ndarray,
    cols: int,
) -> int:
    position = heap_size
    heap_size += 1
    while position > 0:
        parent = (position - 1) // 2
        parent_index = int(heap[parent])
        if not _heap_less(index, parent_index, conditioned, cols):
            break
        heap[position] = parent_index
        position = parent
    heap[position] = index
    return heap_size


@njit(cache=True)
def _heap_pop(
    heap: np.ndarray,
    heap_size: int,
    conditioned: np.ndarray,
    cols: int,
) -> tuple[int, int]:
    result = int(heap[0])
    heap_size -= 1
    if heap_size == 0:
        return result, 0
    replacement = int(heap[heap_size])
    position = 0
    while True:
        left = position * 2 + 1
        if left >= heap_size:
            break
        right = left + 1
        child = left
        if right < heap_size and _heap_less(
            int(heap[right]), int(heap[left]), conditioned, cols
        ):
            child = right
        child_index = int(heap[child])
        if not _heap_less(child_index, replacement, conditioned, cols):
            break
        heap[position] = child_index
        position = child
    heap[position] = replacement
    return result, heap_size


@njit(cache=True)
def _accumulate_d8_kernel(
    flow: np.ndarray,
    processing_order: np.ndarray,
    accumulation: np.ndarray,
) -> None:
    rows, cols = flow.shape
    for position in range(processing_order.size - 1, -1, -1):
        index = int(processing_order[position])
        row = index // cols
        col = index - row * cols
        code = int(flow[row, col])
        if code < 1 or code > 8:
            continue
        next_row = row + _DROW[code]
        next_col = col + _DCOL[code]
        accumulation[next_row, next_col] += accumulation[row, col]


@njit(cache=True)
def _label_catchments_kernel(
    flow: np.ndarray,
    processing_order: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, int]:
    rows, cols = flow.shape
    labels = np.full(flow.shape, _NODATA_U32, dtype=np.uint32)
    next_id = 0
    for position in range(processing_order.size):
        index = int(processing_order[position])
        row = index // cols
        col = index - row * cols
        code = int(flow[row, col])
        if code == 0:
            labels[row, col] = next_id
            next_id += 1
        elif code <= 8:
            next_row = row + _DROW[code]
            next_col = col + _DCOL[code]
            labels[row, col] = labels[next_row, next_col]
    for row in range(rows):
        for col in range(cols):
            if not valid[row, col]:
                labels[row, col] = _NODATA_U32
    return labels, next_id


@njit(cache=True)
def _strahler_d8_kernel(
    flow: np.ndarray,
    processing_order: np.ndarray,
    channels: np.ndarray,
) -> np.ndarray:
    rows, cols = flow.shape
    result = np.zeros(flow.shape, dtype=np.uint8)
    maximum_upstream = np.zeros(flow.shape, dtype=np.uint8)
    maximum_count = np.zeros(flow.shape, dtype=np.uint8)
    for position in range(processing_order.size - 1, -1, -1):
        index = int(processing_order[position])
        row = index // cols
        col = index - row * cols
        if not channels[row, col]:
            continue
        upstream_order = int(maximum_upstream[row, col])
        cell_order = 1 if upstream_order == 0 else upstream_order
        if upstream_order > 0 and maximum_count[row, col] >= 2:
            cell_order += 1
        if cell_order > 255:
            cell_order = 255
        result[row, col] = cell_order
        code = int(flow[row, col])
        if code < 1 or code > 8:
            continue
        next_row = row + _DROW[code]
        next_col = col + _DCOL[code]
        if not channels[next_row, next_col]:
            continue
        if cell_order > maximum_upstream[next_row, next_col]:
            maximum_upstream[next_row, next_col] = cell_order
            maximum_count[next_row, next_col] = 1
        elif cell_order == maximum_upstream[next_row, next_col]:
            maximum_count[next_row, next_col] += 1
    return result
