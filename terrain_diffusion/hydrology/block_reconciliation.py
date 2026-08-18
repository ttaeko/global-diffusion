"""Planner-constrained D8 reconciliation on final integer block heights."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

from terrain_diffusion.hydrology.planner import plan_hydrology
from terrain_diffusion.hydrology.world_plan import D8_DIRECTION_OFFSETS
from terrain_diffusion.inference.relief_map import get_relief_map


_OFFSET_TO_D8 = {offset: code for code, offset in D8_DIRECTION_OFFSETS.items()}
_NEIGHBORS = tuple(
    (delta_row, delta_column, np.hypot(delta_row, delta_column))
    for delta_row in (-1, 0, 1)
    for delta_column in (-1, 0, 1)
    if delta_row or delta_column
)


@dataclass(frozen=True)
class BlockFlowContract:
    """A deterministic bed-height constraint passed to a downstream tile."""

    world_row: int
    world_column: int
    maximum_bed_height: int
    accumulation_area_m2: float = 0.0
    stream_order: int = 0
    entry_world_row: int | None = None
    entry_world_column: int | None = None
    entry_maximum_bed_height: int | None = None
    mean_discharge_m3s: float = 0.0


@dataclass(frozen=True)
class BlockReconciliationDiagnostics:
    route_cells: int
    route_edges: int
    corrected_cells: int
    corrected_fraction: float
    total_incision_blocks: int
    maximum_incision_blocks: int
    mean_incision_blocks: float
    nonascending_fraction_before: float
    nonascending_fraction_after: float
    unresolved_edges: int
    used_fallback_segments: int


@dataclass(frozen=True)
class BlockReconciliationResult:
    block_height: np.ndarray
    correction_blocks: np.ndarray
    channel_mask: np.ndarray
    flow_direction: np.ndarray
    accumulation_area_m2: np.ndarray
    mean_discharge_m3s: np.ndarray
    downstream_contracts: tuple[BlockFlowContract, ...]
    diagnostics: BlockReconciliationDiagnostics


@dataclass(frozen=True)
class BlockRiverResult:
    """Terrain and water columns for a materialized block-space river."""

    block_height: np.ndarray
    water_mask: np.ndarray
    water_surface_height: np.ndarray
    estimated_width_m: np.ndarray
    width_blocks: np.ndarray
    lateral_carving_blocks: np.ndarray


@dataclass(frozen=True)
class BlockLakeResult:
    """Level lake surfaces and minimum-depth terrain columns in block space."""

    block_height: np.ndarray
    water_mask: np.ndarray
    water_surface_height: np.ndarray
    terrain_correction_blocks: np.ndarray


def _point_segment_distance_squared(row, column, start, end):
    vector_row = end[0] - start[0]
    vector_column = end[1] - start[1]
    length_squared = vector_row * vector_row + vector_column * vector_column
    if length_squared == 0:
        return float((row - start[0]) ** 2 + (column - start[1]) ** 2)
    position = (
        (row - start[0]) * vector_row
        + (column - start[1]) * vector_column
    ) / length_squared
    position = min(1.0, max(0.0, position))
    projected_row = start[0] + position * vector_row
    projected_column = start[1] + position * vector_column
    return float((row - projected_row) ** 2 + (column - projected_column) ** 2)


def _straight_d8_path(start, end):
    row, column = start
    target_row, target_column = end
    path = [(row, column)]
    while (row, column) != (target_row, target_column):
        row += int(np.sign(target_row - row))
        column += int(np.sign(target_column - column))
        path.append((row, column))
    return path


def _constrained_d8_path(
    height,
    start,
    end,
    *,
    corridor_radius,
    deviation_weight,
    uphill_weight,
    elevation_weight,
):
    """Find an exact-height D8 path close to one planner segment."""
    if start == end:
        return [start], False
    rows, columns = height.shape
    row_min = max(0, min(start[0], end[0]) - corridor_radius)
    row_max = min(rows - 1, max(start[0], end[0]) + corridor_radius)
    column_min = max(0, min(start[1], end[1]) - corridor_radius)
    column_max = min(columns - 1, max(start[1], end[1]) + corridor_radius)
    allowed = {}
    minimum_height = np.inf
    radius_squared = float(corridor_radius**2)
    for row in range(row_min, row_max + 1):
        for column in range(column_min, column_max + 1):
            distance_squared = _point_segment_distance_squared(
                row, column, start, end
            )
            if distance_squared <= radius_squared:
                allowed[(row, column)] = distance_squared
                minimum_height = min(minimum_height, float(height[row, column]))
    if start not in allowed or end not in allowed:
        return _straight_d8_path(start, end), True

    queue = [(0.0, start)]
    costs = {start: 0.0}
    parent = {}
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != costs.get(current):
            continue
        if current == end:
            break
        for delta_row, delta_column, distance in _NEIGHBORS:
            neighbor = (current[0] + delta_row, current[1] + delta_column)
            if neighbor not in allowed:
                continue
            uphill = max(
                0.0,
                float(height[neighbor]) - float(height[current]),
            )
            lowland = max(0.0, float(height[neighbor]) - minimum_height)
            next_cost = cost + distance
            next_cost += deviation_weight * allowed[neighbor]
            next_cost += uphill_weight * uphill
            next_cost += elevation_weight * lowland
            if next_cost < costs.get(neighbor, np.inf):
                costs[neighbor] = next_cost
                parent[neighbor] = current
                heapq.heappush(queue, (next_cost, neighbor))
    if end not in costs:
        return _straight_d8_path(start, end), True
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path, False


def _constrained_polyline_path(
    height,
    start,
    end,
    polyline,
    *,
    corridor_radius,
    deviation_weight,
    uphill_weight,
    elevation_weight,
    turn_weight,
):
    """Find one exact-height path through a complete planner-reach corridor."""
    if start == end:
        return [start], False
    guide = np.zeros(height.shape, dtype=bool)
    for source, target in zip(polyline[:-1], polyline[1:]):
        for position in _straight_d8_path(source, target):
            guide[position] = True
    guide[start] = True
    guide[end] = True
    distance_to_guide = scipy.ndimage.distance_transform_edt(~guide)
    allowed_mask = distance_to_guide <= corridor_radius
    if not np.any(allowed_mask):
        return _straight_d8_path(start, end), True
    smoothed_height = scipy.ndimage.gaussian_filter(
        height.astype(np.float32),
        sigma=max(1.0, corridor_radius / 2.0),
    )
    relative_height = height - smoothed_height
    minimum_relative_height = float(relative_height[allowed_mask].min())

    start_state = (start, -1)
    queue = [(0.0, start, -1)]
    costs = {start_state: 0.0}
    parent = {}
    end_state = None
    while queue:
        cost, current, previous_direction = heapq.heappop(queue)
        state = (current, previous_direction)
        if cost != costs.get(state):
            continue
        if current == end:
            end_state = state
            break
        for direction, (delta_row, delta_column, distance) in enumerate(_NEIGHBORS):
            neighbor = (current[0] + delta_row, current[1] + delta_column)
            if not (
                0 <= neighbor[0] < height.shape[0]
                and 0 <= neighbor[1] < height.shape[1]
                and allowed_mask[neighbor]
            ):
                continue
            uphill = max(0.0, float(height[neighbor]) - float(height[current]))
            lowland = max(
                0.0,
                float(relative_height[neighbor]) - minimum_relative_height,
            )
            next_cost = cost + distance
            next_cost += deviation_weight * distance_to_guide[neighbor] ** 2
            next_cost += uphill_weight * uphill
            next_cost += elevation_weight * lowland
            if previous_direction >= 0 and direction != previous_direction:
                next_cost += turn_weight
            next_state = (neighbor, direction)
            if next_cost < costs.get(next_state, np.inf):
                costs[next_state] = next_cost
                parent[next_state] = state
                heapq.heappush(queue, (next_cost, neighbor, direction))
    if end_state is None:
        return _straight_d8_path(start, end), True
    path = [end_state[0]]
    state = end_state
    while state != start_state:
        state = parent[state]
        path.append(state[0])
    path.reverse()
    return path, False


def _planner_reaches(order, downstream, outside_targets):
    incoming = {node: 0 for node in order}
    for target in downstream.values():
        incoming[target] += 1
    reaches = []
    for start in order:
        if incoming[start] == 1:
            continue
        if start not in downstream and start not in outside_targets:
            reaches.append(([start], None))
            continue
        cells = [start]
        current = start
        while current in downstream:
            target = downstream[current]
            cells.append(target)
            current = target
            if incoming[current] != 1:
                break
        reaches.append((cells, outside_targets.get(current)))
    return reaches


def _unclipped_block_center(cell, scale):
    return cell[0] * scale + scale // 2, cell[1] * scale + scale // 2


def _ray_boundary_intersection(start, outside, shape):
    """Intersect an inside-to-outside planner ray with the raster boundary."""
    candidates = []
    for axis in (0, 1):
        delta = outside[axis] - start[axis]
        if delta == 0:
            continue
        boundary = shape[axis] - 1 if delta > 0 else 0
        position = (boundary - start[axis]) / delta
        if 0.0 <= position <= 1.0:
            other_axis = 1 - axis
            other = start[other_axis] + position * (
                outside[other_axis] - start[other_axis]
            )
            if 0.0 <= other <= shape[other_axis] - 1:
                candidates.append(position)
    if not candidates:
        return (
            min(shape[0] - 1, max(0, outside[0])),
            min(shape[1] - 1, max(0, outside[1])),
        )
    position = min(candidates)
    return (
        int(round(start[0] + position * (outside[0] - start[0]))),
        int(round(start[1] + position * (outside[1] - start[1]))),
    )


def _planner_graph(channel_mask, flow_direction):
    channels = np.asarray(channel_mask, dtype=bool)
    flow = np.asarray(flow_direction)
    nodes = tuple(map(tuple, np.argwhere(channels)))
    node_set = set(nodes)
    downstream = {}
    incoming = {node: 0 for node in nodes}
    outside_targets = {}
    for node in nodes:
        code = int(flow[node])
        if code not in D8_DIRECTION_OFFSETS:
            continue
        delta_row, delta_column = D8_DIRECTION_OFFSETS[code]
        target = (node[0] + delta_row, node[1] + delta_column)
        if target in node_set:
            downstream[node] = target
            incoming[target] += 1
        elif not (
            0 <= target[0] < channels.shape[0]
            and 0 <= target[1] < channels.shape[1]
        ):
            outside_targets[node] = target
    queue = [node for node in nodes if incoming[node] == 0]
    heapq.heapify(queue)
    order = []
    while queue:
        node = heapq.heappop(queue)
        order.append(node)
        if node in downstream:
            target = downstream[node]
            incoming[target] -= 1
            if incoming[target] == 0:
                heapq.heappush(queue, target)
    if len(order) != len(nodes):
        raise ValueError("planner channel graph contains a directed cycle")
    return order, downstream, outside_targets


def _block_anchor(
    cell,
    scale,
    shape,
    *,
    height=None,
    search_radius=0,
    deviation_weight=0.03,
):
    center = (
        min(shape[0] - 1, max(0, cell[0] * scale + scale // 2)),
        min(shape[1] - 1, max(0, cell[1] * scale + scale // 2)),
    )
    if (
        height is None
        or search_radius <= 0
        or not (0 <= cell[0] * scale < shape[0])
        or not (0 <= cell[1] * scale < shape[1])
    ):
        return center
    row_start = max(cell[0] * scale, center[0] - search_radius)
    row_stop = min((cell[0] + 1) * scale, center[0] + search_radius + 1)
    column_start = max(cell[1] * scale, center[1] - search_radius)
    column_stop = min(
        (cell[1] + 1) * scale,
        center[1] + search_radius + 1,
    )
    candidates = []
    for row in range(row_start, row_stop):
        for column in range(column_start, column_stop):
            distance_squared = (
                (row - center[0]) ** 2 + (column - center[1]) ** 2
            )
            score = float(height[row, column]) + deviation_weight * distance_squared
            candidates.append((score, distance_squared, row, column))
    _, _, row, column = min(candidates)
    return row, column


def reconcile_planned_block_channels(
    block_height: np.ndarray,
    planner_channel_mask: np.ndarray,
    planner_flow_direction: np.ndarray,
    *,
    planner_accumulation_area_m2: np.ndarray | None = None,
    planner_mean_discharge_m3s: np.ndarray | None = None,
    planner_stream_order: np.ndarray | None = None,
    planner_to_block_scale: int = 15,
    world_block_origin: tuple[int, int] = (0, 0),
    upstream_contracts: tuple[BlockFlowContract, ...] = (),
    corridor_radius_blocks: int = 12,
    minimum_drop_blocks: int = 0,
    maximum_incision_blocks: int = 8,
    strict: bool = True,
    anchor_search_radius_blocks: int = 4,
    anchor_deviation_weight: float = 0.03,
    deviation_weight: float = 0.003,
    uphill_weight: float = 5.0,
    elevation_weight: float = 0.7,
    turn_weight: float = 0.0,
) -> BlockReconciliationResult:
    """Carve a minimal, planner-constrained D8 bed into final block heights."""
    original = np.asarray(block_height)
    if original.ndim != 2 or not np.issubdtype(original.dtype, np.integer):
        raise ValueError("block_height must be a two-dimensional integer array")
    channels = np.asarray(planner_channel_mask, dtype=bool)
    flow = np.asarray(planner_flow_direction)
    expected_shape = (
        channels.shape[0] * planner_to_block_scale,
        channels.shape[1] * planner_to_block_scale,
    )
    if original.shape != expected_shape:
        raise ValueError(
            f"block height shape {original.shape} does not match planner grid "
            f"and scale {expected_shape}"
        )
    if flow.shape != channels.shape:
        raise ValueError("planner flow direction does not match channel mask")
    if (
        corridor_radius_blocks < 0
        or anchor_search_radius_blocks < 0
        or minimum_drop_blocks < 0
    ):
        raise ValueError("radii and minimum drop must be non-negative")
    if maximum_incision_blocks < 0:
        raise ValueError("maximum incision must be non-negative")

    order, downstream, outside_targets = _planner_graph(channels, flow)
    accumulation = (
        np.asarray(planner_accumulation_area_m2)
        if planner_accumulation_area_m2 is not None
        else np.zeros(channels.shape, dtype=np.float64)
    )
    discharge = (
        np.asarray(planner_mean_discharge_m3s)
        if planner_mean_discharge_m3s is not None
        else np.zeros(channels.shape, dtype=np.float64)
    )
    if accumulation.shape != channels.shape or discharge.shape != channels.shape:
        raise ValueError("planner accumulation or discharge shape does not match")
    detailed_edges = set()
    detailed_nodes = set()
    detailed_accumulation = {}
    detailed_discharge = {}
    outlet_paths = {}
    fallback_segments = 0
    anchors = {
        node: _block_anchor(
            node,
            planner_to_block_scale,
            original.shape,
            height=original,
            search_radius=anchor_search_radius_blocks,
            deviation_weight=anchor_deviation_weight,
        )
        for node in order
    }
    for cells, outside_target in _planner_reaches(
        order, downstream, outside_targets
    ):
        if len(cells) == 1 and outside_target is None:
            node = cells[0]
            position = anchors[node]
            detailed_nodes.add(position)
            detailed_accumulation[position] = float(accumulation[node])
            detailed_discharge[position] = float(discharge[node])
            continue
        start = anchors[cells[0]]
        guide = [
            _block_anchor(cell, planner_to_block_scale, original.shape)
            for cell in cells
        ]
        if outside_target is None:
            target = anchors[cells[-1]]
            route_target = target
        else:
            inside_center = _unclipped_block_center(
                cells[-1], planner_to_block_scale
            )
            outside_center = _unclipped_block_center(
                outside_target, planner_to_block_scale
            )
            target = _ray_boundary_intersection(
                inside_center, outside_center, original.shape
            )
            outgoing = (
                int(np.sign(outside_center[0] - inside_center[0])),
                int(np.sign(outside_center[1] - inside_center[1])),
            )
            approach_steps = min(
                corridor_radius_blocks,
                planner_to_block_scale // 2,
            )
            route_target = (
                target[0] - outgoing[0] * approach_steps,
                target[1] - outgoing[1] * approach_steps,
            )
            guide.append(target)
        path, fallback = _constrained_polyline_path(
            original,
            start,
            route_target,
            guide,
            corridor_radius=corridor_radius_blocks,
            deviation_weight=deviation_weight,
            uphill_weight=uphill_weight,
            elevation_weight=elevation_weight,
            turn_weight=turn_weight,
        )
        if outside_target is not None:
            path.extend(_straight_d8_path(route_target, target)[1:])
        fallback_segments += int(fallback)
        detailed_nodes.update(path)
        detailed_edges.update(zip(path[:-1], path[1:]))
        coarse_centers = np.asarray(
            [
                _unclipped_block_center(cell, planner_to_block_scale)
                for cell in cells
            ],
            dtype=np.float64,
        )
        for position in path:
            distances = np.sum(
                np.square(coarse_centers - np.asarray(position)), axis=1
            )
            coarse_node = cells[int(np.argmin(distances))]
            detailed_accumulation[position] = max(
                detailed_accumulation.get(position, 0.0),
                float(accumulation[coarse_node]),
            )
            detailed_discharge[position] = max(
                detailed_discharge.get(position, 0.0),
                float(discharge[coarse_node]),
            )
        if outside_target is not None:
            outlet_paths[cells[-1]] = path

    origin_row, origin_column = world_block_origin
    contract_constraints = {}
    for contract in upstream_contracts:
        target = (
            contract.world_row - origin_row,
            contract.world_column - origin_column,
        )
        if not (
            0 <= target[0] < original.shape[0]
            and 0 <= target[1] < original.shape[1]
        ):
            continue
        detailed_nodes.add(target)
        contract_constraints[target] = min(
            contract_constraints.get(target, np.inf),
            int(contract.maximum_bed_height),
        )
        if (
            contract.entry_world_row is None
            or contract.entry_world_column is None
        ):
            continue
        entry = (
            min(
                original.shape[0] - 1,
                max(0, contract.entry_world_row - origin_row),
            ),
            min(
                original.shape[1] - 1,
                max(0, contract.entry_world_column - origin_column),
            ),
        )
        path, fallback = _constrained_d8_path(
            original,
            entry,
            target,
            corridor_radius=corridor_radius_blocks,
            deviation_weight=deviation_weight,
            uphill_weight=uphill_weight,
            elevation_weight=elevation_weight,
        )
        fallback_segments += int(fallback)
        detailed_nodes.update(path)
        detailed_edges.update(zip(path[:-1], path[1:]))
        for position in path:
            detailed_accumulation[position] = max(
                detailed_accumulation.get(position, 0.0),
                float(contract.accumulation_area_m2),
            )
            detailed_discharge[position] = max(
                detailed_discharge.get(position, 0.0),
                float(contract.mean_discharge_m3s),
            )
        if contract.entry_maximum_bed_height is not None:
            contract_constraints[entry] = min(
                contract_constraints.get(entry, np.inf),
                int(contract.entry_maximum_bed_height),
            )

    adjacency = {node: set() for node in detailed_nodes}
    indegree = {node: 0 for node in detailed_nodes}
    for source, target in detailed_edges:
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
    queue = [node for node, count in indegree.items() if count == 0]
    heapq.heapify(queue)
    detailed_order = []
    while queue:
        node = heapq.heappop(queue)
        detailed_order.append(node)
        for target in sorted(adjacency[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(queue, target)
    if len(detailed_order) != len(detailed_nodes):
        raise ValueError("exact-height detailed paths created a directed cycle")

    corrected = original.astype(np.int32, copy=True)
    for position, maximum_height in contract_constraints.items():
        corrected[position] = min(corrected[position], int(maximum_height))

    parents = {node: [] for node in detailed_nodes}
    for source, target in detailed_edges:
        parents[target].append(source)
    before_nonascending = []
    unresolved = 0
    for node in detailed_order:
        if not parents[node]:
            continue
        allowed_height = min(
            corrected[parent] - minimum_drop_blocks for parent in parents[node]
        )
        before_nonascending.extend(
            original[node] <= original[parent] - minimum_drop_blocks
            for parent in parents[node]
        )
        required_height = min(int(corrected[node]), int(allowed_height))
        minimum_height = int(original[node]) - maximum_incision_blocks
        if required_height < minimum_height:
            unresolved += sum(
                required_height < minimum_height for _ in parents[node]
            )
            corrected[node] = minimum_height
        else:
            corrected[node] = required_height

    after_nonascending = [
        corrected[target] <= corrected[source] - minimum_drop_blocks
        for source, target in detailed_edges
    ]
    if strict and unresolved:
        raise ValueError(
            f"block hydrology reconciliation requires more than "
            f"{maximum_incision_blocks} blocks of incision on {unresolved} edges"
        )

    correction = corrected - original.astype(np.int32)
    channel = np.zeros(original.shape, dtype=bool)
    flow_d8 = np.full(original.shape, -1, dtype=np.int16)
    for node in detailed_nodes:
        channel[node] = True
    for source, targets in adjacency.items():
        if not targets:
            continue
        target = min(targets, key=lambda item: (corrected[item], item))
        offset = (target[0] - source[0], target[1] - source[1])
        flow_d8[source] = int(_OFFSET_TO_D8[offset])

    stream_order = (
        np.asarray(planner_stream_order)
        if planner_stream_order is not None
        else np.zeros(channels.shape, dtype=np.uint8)
    )
    contracts = []
    for coarse_node, path in outlet_paths.items():
        outside = outside_targets[coarse_node]
        target_world = (
            origin_row + outside[0] * planner_to_block_scale + planner_to_block_scale // 2,
            origin_column + outside[1] * planner_to_block_scale + planner_to_block_scale // 2,
        )
        contracts.append(
            BlockFlowContract(
                world_row=int(target_world[0]),
                world_column=int(target_world[1]),
                maximum_bed_height=int(corrected[path[-1]] - minimum_drop_blocks),
                accumulation_area_m2=float(accumulation[coarse_node]),
                stream_order=int(stream_order[coarse_node]),
                entry_world_row=int(origin_row + path[-1][0]),
                entry_world_column=int(origin_column + path[-1][1]),
                entry_maximum_bed_height=int(corrected[path[-1]]),
                mean_discharge_m3s=float(discharge[coarse_node]),
            )
        )

    detailed_accumulation_raster = np.zeros(original.shape, dtype=np.float32)
    detailed_discharge_raster = np.zeros(original.shape, dtype=np.float32)
    for position in detailed_nodes:
        detailed_accumulation_raster[position] = detailed_accumulation.get(
            position, 0.0
        )
        detailed_discharge_raster[position] = detailed_discharge.get(position, 0.0)

    incision = -correction[channel]
    diagnostics = BlockReconciliationDiagnostics(
        route_cells=int(np.count_nonzero(channel)),
        route_edges=len(detailed_edges),
        corrected_cells=int(np.count_nonzero(correction)),
        corrected_fraction=float(np.count_nonzero(correction) / correction.size),
        total_incision_blocks=int(incision.sum()),
        maximum_incision_blocks=int(incision.max(initial=0)),
        mean_incision_blocks=float(incision.mean()) if incision.size else 0.0,
        nonascending_fraction_before=(
            float(np.mean(before_nonascending)) if before_nonascending else 1.0
        ),
        nonascending_fraction_after=(
            float(np.mean(after_nonascending)) if after_nonascending else 1.0
        ),
        unresolved_edges=int(unresolved),
        used_fallback_segments=int(fallback_segments),
    )
    return BlockReconciliationResult(
        block_height=corrected,
        correction_blocks=correction,
        channel_mask=channel,
        flow_direction=flow_d8,
        accumulation_area_m2=detailed_accumulation_raster,
        mean_discharge_m3s=detailed_discharge_raster,
        downstream_contracts=tuple(contracts),
        diagnostics=diagnostics,
    )


def materialize_block_river(
    reconciliation: BlockReconciliationResult,
    *,
    metres_per_block: float = 2.0,
    width_coefficient: float = 2.5,
    width_exponent: float = 0.5,
    minimum_width_blocks: int = 1,
) -> BlockRiverResult:
    """Create water-bearing block columns from a reconciled channel route.

    The hydraulic-width estimate is ``coefficient * discharge ** exponent``.
    Rivers narrower than one horizontal block remain one block wide.
    """
    if metres_per_block <= 0 or width_coefficient < 0 or width_exponent < 0:
        raise ValueError("river width parameters must be non-negative")
    if minimum_width_blocks < 1:
        raise ValueError("minimum river width must be at least one block")
    channel = np.asarray(reconciliation.channel_mask, dtype=bool)
    if not np.any(channel):
        empty_height = np.full(channel.shape, -1, dtype=np.int32)
        return BlockRiverResult(
            block_height=reconciliation.block_height.copy(),
            water_mask=channel.copy(),
            water_surface_height=empty_height,
            estimated_width_m=np.zeros(channel.shape, dtype=np.float32),
            width_blocks=np.zeros(channel.shape, dtype=np.uint16),
            lateral_carving_blocks=np.zeros(channel.shape, dtype=np.int32),
        )

    discharge = np.maximum(reconciliation.mean_discharge_m3s, 0.0)
    center_width_m = width_coefficient * np.power(discharge, width_exponent)
    center_width_blocks = np.maximum(
        minimum_width_blocks,
        np.ceil(center_width_m / metres_per_block).astype(np.int32),
    )
    base_water = channel.copy()
    base_surface = np.full(channel.shape, -1, dtype=np.int32)
    base_surface[channel] = reconciliation.block_height[channel] + 1
    base_width_m = center_width_m.copy()
    base_width_blocks = center_width_blocks.copy()
    for source_row, source_column in zip(*np.nonzero(reconciliation.flow_direction >= 0)):
        code = int(reconciliation.flow_direction[source_row, source_column])
        delta_row, delta_column = D8_DIRECTION_OFFSETS[code]
        if delta_row == 0 or delta_column == 0:
            continue
        target = source_row + delta_row, source_column + delta_column
        candidates = (
            (source_row, target[1]),
            (target[0], source_column),
        )
        bridge = min(
            candidates,
            key=lambda position: (
                reconciliation.block_height[position],
                position,
            ),
        )
        base_water[bridge] = True
        target_surface = int(reconciliation.block_height[target]) + 1
        if base_surface[bridge] < 0:
            base_surface[bridge] = target_surface
        else:
            base_surface[bridge] = min(base_surface[bridge], target_surface)
        base_width_m[bridge] = max(
            center_width_m[source_row, source_column],
            center_width_m[target],
        )
        base_width_blocks[bridge] = max(
            center_width_blocks[source_row, source_column],
            center_width_blocks[target],
        )
    distances, nearest = scipy.ndimage.distance_transform_edt(
        ~base_water,
        return_indices=True,
    )
    nearest_width_blocks = base_width_blocks[nearest[0], nearest[1]]
    radius = np.where(
        nearest_width_blocks <= 1,
        0.0,
        nearest_width_blocks / 2.0,
    )
    water_mask = distances <= radius
    nearest_surface = base_surface[nearest[0], nearest[1]]
    nearest_bed = nearest_surface - 1
    riverbed = reconciliation.block_height.astype(np.int32, copy=True)
    riverbed[water_mask] = np.minimum(riverbed[water_mask], nearest_bed[water_mask])
    water_surface = np.full(channel.shape, -1, dtype=np.int32)
    water_surface[water_mask] = nearest_surface[water_mask]
    width_m = np.zeros(channel.shape, dtype=np.float32)
    width_m[water_mask] = base_width_m[nearest[0], nearest[1]][water_mask]
    width_blocks = np.zeros(channel.shape, dtype=np.uint16)
    width_blocks[water_mask] = nearest_width_blocks[water_mask]
    return BlockRiverResult(
        block_height=riverbed,
        water_mask=water_mask,
        water_surface_height=water_surface,
        estimated_width_m=width_m,
        width_blocks=width_blocks,
        lateral_carving_blocks=riverbed - reconciliation.block_height,
    )


def materialize_block_lakes(
    block_height: np.ndarray,
    planner_lake_mask: np.ndarray,
    planner_water_surface_elevation_m: np.ndarray,
    *,
    planner_to_block_scale: int = 15,
    metres_per_block: float = 2.0,
    minimum_depth_blocks: int = 1,
) -> BlockLakeResult:
    """Materialize planned lake polygons as level water-bearing block columns."""

    height = np.asarray(block_height, dtype=np.int32)
    lakes = np.asarray(planner_lake_mask, dtype=bool)
    surfaces = np.asarray(planner_water_surface_elevation_m, dtype=np.float32)
    if lakes.shape != surfaces.shape or lakes.ndim != 2:
        raise ValueError("Planner lake mask and water surface must align")
    if planner_to_block_scale <= 0 or metres_per_block <= 0:
        raise ValueError("Lake materialization scales must be positive")
    if minimum_depth_blocks < 1:
        raise ValueError("Lake minimum depth must be at least one block")
    expected = (
        lakes.shape[0] * planner_to_block_scale,
        lakes.shape[1] * planner_to_block_scale,
    )
    if height.shape != expected:
        raise ValueError("Block terrain does not match planner lake scale")
    if np.any(lakes & ~np.isfinite(surfaces)):
        raise ValueError("Every planned lake cell requires a water surface")

    water_mask = np.repeat(
        np.repeat(lakes, planner_to_block_scale, axis=0),
        planner_to_block_scale,
        axis=1,
    )
    surface_m = np.repeat(
        np.repeat(surfaces, planner_to_block_scale, axis=0),
        planner_to_block_scale,
        axis=1,
    )
    water_surface = np.full(height.shape, -1, dtype=np.int32)
    water_surface[water_mask] = np.rint(
        surface_m[water_mask] / metres_per_block
    ).astype(np.int32)
    corrected = height.copy()
    maximum_bed = water_surface - minimum_depth_blocks
    corrected[water_mask] = np.minimum(
        corrected[water_mask], maximum_bed[water_mask]
    )
    return BlockLakeResult(
        block_height=corrected,
        water_mask=water_mask,
        water_surface_height=water_surface,
        terrain_correction_blocks=corrected - height,
    )


def _overlay_channel(image, channel_mask):
    result = np.asarray(image, dtype=np.float32).copy()
    visible = scipy.ndimage.binary_dilation(channel_mask, iterations=2)
    color = np.asarray([0.05, 0.35, 1.0], dtype=np.float32)
    result[visible] = result[visible] * 0.2 + color * 0.8
    return np.clip(result, 0.0, 1.0)


def render_block_river(image, water_mask):
    result = np.asarray(image, dtype=np.float32).copy()
    bank = scipy.ndimage.binary_dilation(water_mask, iterations=1) & ~water_mask
    result[bank] *= 0.72
    water_color = np.asarray([0.02, 0.30, 0.78], dtype=np.float32)
    result[water_mask] = result[water_mask] * 0.12 + water_color * 0.88
    return np.clip(result, 0.0, 1.0)


@click.command("reconcile-block-hydrology")
@click.argument(
    "section_directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--output", "-o", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--corridor-radius-blocks",
    default=12,
    show_default=True,
    type=click.IntRange(min=0),
)
@click.option(
    "--minimum-drop-blocks",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
)
@click.option(
    "--anchor-search-radius-blocks",
    default=4,
    show_default=True,
    type=click.IntRange(min=0),
)
@click.option(
    "--maximum-incision-blocks",
    default=8,
    show_default=True,
    type=click.IntRange(min=0),
)
@click.option("--strict/--allow-unresolved", default=True, show_default=True)
def reconcile_block_hydrology_cli(
    section_directory,
    output,
    corridor_radius_blocks,
    minimum_drop_blocks,
    anchor_search_radius_blocks,
    maximum_incision_blocks,
    strict,
):
    """Apply exact-height channel reconciliation to a full-pipeline sample."""
    output = output or section_directory / "block_hydrology_reconciliation"
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(
        (section_directory / "report.json").read_text(encoding="utf-8")
    )
    with np.load(section_directory / "section.npz", allow_pickle=False) as arrays:
        context30 = arrays["context_30m_elevation_m"]
        elevation2 = arrays["elevation_2m"]
        block_height = arrays["block_height"]
    planned = plan_hydrology(context30, resolution_m=30.0)
    size = int(report["section_size_30m"])
    center_row, center_column = report["section_center_context_30m"]
    start_row = int(center_row) - size // 2
    start_column = int(center_column) - size // 2
    section = np.s_[start_row:start_row + size, start_column:start_column + size]
    planner_channels = planned.channel_mask[section]
    planner_flow = planned.routing.flow_direction[section]
    planner_accumulation = planned.routing.accumulation_area_m2[section]
    planner_order = planned.stream_order[section]
    world_origin = (
        (int(report["native_30m_bounds"][0]) + start_row) * 15,
        (int(report["native_30m_bounds"][1]) + start_column) * 15,
    )
    try:
        result = reconcile_planned_block_channels(
            block_height,
            planner_channels,
            planner_flow,
            planner_accumulation_area_m2=planner_accumulation,
            planner_mean_discharge_m3s=planned.mean_discharge_m3s[section],
            planner_stream_order=planner_order,
            world_block_origin=world_origin,
            corridor_radius_blocks=corridor_radius_blocks,
            anchor_search_radius_blocks=anchor_search_radius_blocks,
            minimum_drop_blocks=minimum_drop_blocks,
            maximum_incision_blocks=maximum_incision_blocks,
            strict=strict,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    river = materialize_block_river(result)
    reconciled_elevation = result.block_height.astype(np.float32) * 2.0
    before = get_relief_map(
        block_height * 2.0, None, None, None, resolution=2, vmin=0, vmax=3000
    )
    after = get_relief_map(
        reconciled_elevation, None, None, None, resolution=2, vmin=0, vmax=3000
    )
    before_overlay = _overlay_channel(before, result.channel_mask)
    after_overlay = _overlay_channel(after, result.channel_mask)
    plt.imsave(output / "before.png", before_overlay)
    plt.imsave(output / "after.png", after_overlay)
    plt.imsave(output / "before_terrain.png", before)
    plt.imsave(output / "after_terrain.png", after)
    river_relief = get_relief_map(
        river.block_height * 2.0,
        None,
        None,
        None,
        resolution=2,
        vmin=0,
        vmax=3000,
    )
    river_render = render_block_river(river_relief, river.water_mask)
    plt.imsave(output / "world_with_river.png", river_render)
    water_rows, water_columns = np.nonzero(river.water_mask)
    padding = 48
    row_start = max(0, int(water_rows.min()) - padding)
    row_stop = min(river.water_mask.shape[0], int(water_rows.max()) + padding + 1)
    column_start = max(0, int(water_columns.min()) - padding)
    column_stop = min(
        river.water_mask.shape[1],
        int(water_columns.max()) + padding + 1,
    )
    figure, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
    axis.imshow(
        river_render[row_start:row_stop, column_start:column_stop],
        interpolation="nearest",
    )
    axis.set_title("Materialized river (one block minimum width)")
    axis.axis("off")
    figure.savefig(output / "river_zoom.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    axes[0].imshow(before_overlay)
    axes[0].set_title("Before: exact generated block heights")
    axes[1].imshow(after_overlay)
    axes[1].set_title("After: reconciled nonascending D8 bed")
    correction_image = axes[2].imshow(
        result.correction_blocks,
        cmap="magma_r",
        vmin=-maximum_incision_blocks,
        vmax=0,
    )
    axes[2].set_title("Channel correction (blocks)")
    figure.colorbar(correction_image, ax=axes[2])
    for axis in axes:
        axis.axis("off")
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)
    np.savez_compressed(
        output / "reconciled.npz",
        original_elevation_2m=elevation2,
        original_block_height=block_height,
        reconciled_block_height=result.block_height,
        correction_blocks=result.correction_blocks,
        channel_mask=result.channel_mask,
        flow_direction=result.flow_direction,
        accumulation_area_m2=result.accumulation_area_m2,
        mean_discharge_m3s=result.mean_discharge_m3s,
        riverbed_block_height=river.block_height,
        water_mask=river.water_mask,
        water_surface_height=river.water_surface_height,
        estimated_river_width_m=river.estimated_width_m,
        river_width_blocks=river.width_blocks,
        lateral_carving_blocks=river.lateral_carving_blocks,
    )
    channel_width_m = river.estimated_width_m[result.channel_mask]
    water_width_blocks = river.width_blocks[river.water_mask]
    reconciliation_report = {
        "schema_version": 1,
        "source": str(section_directory),
        "world_block_origin": list(world_origin),
        "settings": {
            "corridor_radius_blocks": int(corridor_radius_blocks),
            "anchor_search_radius_blocks": int(anchor_search_radius_blocks),
            "minimum_drop_blocks": int(minimum_drop_blocks),
            "maximum_incision_blocks": int(maximum_incision_blocks),
            "strict": bool(strict),
        },
        "diagnostics": asdict(result.diagnostics),
        "river": {
            "water_columns": int(np.count_nonzero(river.water_mask)),
            "minimum_width_blocks": int(water_width_blocks.min()),
            "maximum_width_blocks": int(water_width_blocks.max()),
            "estimated_width_m": {
                "minimum": float(channel_width_m.min()),
                "median": float(np.median(channel_width_m)),
                "maximum": float(channel_width_m.max()),
            },
            "lateral_carved_columns": int(
                np.count_nonzero(river.lateral_carving_blocks)
            ),
        },
        "downstream_contracts": [
            asdict(contract) for contract in result.downstream_contracts
        ],
    }
    (output / "report.json").write_text(
        json.dumps(reconciliation_report, indent=2) + "\n",
        encoding="utf-8",
    )
    click.echo(f"Saved block hydrology reconciliation to {output}")


if __name__ == "__main__":
    reconcile_block_hydrology_cli()
