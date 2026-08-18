"""Extract and persist a stable vector river graph from routed channel cells."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import sqlite3

import numpy as np
from numba import njit


_DROW = np.asarray((0, 0, 1, 1, 1, 0, -1, -1, -1), dtype=np.int8)
_DCOL = np.asarray((0, 1, 1, 0, -1, -1, -1, 0, 1), dtype=np.int8)
_NODATA_U32 = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class RiverNode:
    node_id: int
    kind: str
    row: int
    col: int
    x_m: float
    z_m: float
    elevation_m: float
    catchment_id: int | None
    lake_id: int | None


@dataclass(frozen=True)
class RiverEdge:
    edge_id: int
    from_node_id: int
    to_node_id: int
    cells: tuple[tuple[int, int], ...]
    length_m: float
    upstream_area_m2: float
    mean_discharge_m3s: float | None
    stream_order: int


@dataclass(frozen=True)
class RiverGraph:
    nodes: tuple[RiverNode, ...]
    edges: tuple[RiverEdge, ...]


def extract_river_graph(
    flow_direction: np.ndarray,
    channel_mask: np.ndarray,
    elevation_m: np.ndarray,
    accumulation_area_m2: np.ndarray,
    catchment_id: np.ndarray,
    stream_order: np.ndarray,
    *,
    resolution_m: float,
    origin_x_m: float = 0.0,
    origin_z_m: float = 0.0,
    mean_discharge_m3s: np.ndarray | None = None,
    lake_id: np.ndarray | None = None,
) -> RiverGraph:
    """Collapse channel-cell chains into deterministic nodes and polylines."""

    flow = np.ascontiguousarray(flow_direction, dtype=np.uint8)
    channels = np.ascontiguousarray(channel_mask, dtype=np.bool_)
    elevation = np.asarray(elevation_m, dtype=np.float32)
    accumulation = np.asarray(accumulation_area_m2, dtype=np.float64)
    catchments = np.asarray(catchment_id, dtype=np.uint32)
    orders = np.asarray(stream_order, dtype=np.uint8)
    arrays = (channels, elevation, accumulation, catchments, orders)
    if flow.ndim != 2 or any(array.shape != flow.shape for array in arrays):
        raise ValueError("All river-graph rasters must have the same 2D shape")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    discharge = None if mean_discharge_m3s is None else np.asarray(mean_discharge_m3s)
    lakes = None if lake_id is None else np.asarray(lake_id, dtype=np.uint32)
    if discharge is not None and discharge.shape != flow.shape:
        raise ValueError("mean_discharge_m3s shape does not match flow_direction")
    if lakes is not None and lakes.shape != flow.shape:
        raise ValueError("lake_id shape does not match flow_direction")

    upstream_count = _channel_upstream_count(flow, channels)
    downstream_is_channel = _downstream_channel_mask(flow, channels)
    if lakes is None:
        lake_inlet = np.zeros(flow.shape, dtype=bool)
        lake_outlet = np.zeros(flow.shape, dtype=bool)
    else:
        lake_inlet, lake_outlet = _lake_transition_nodes(flow, channels, lakes)
    node_mask = channels & (
        (upstream_count != 1) | ~downstream_is_channel | lake_inlet | lake_outlet
    )
    rows, cols = flow.shape
    node_cells = np.flatnonzero(node_mask)
    node_by_cell = {int(cell): node_id for node_id, cell in enumerate(node_cells)}
    nodes: list[RiverNode] = []
    for node_id, cell in enumerate(node_cells):
        row, col = divmod(int(cell), cols)
        if lake_outlet[row, col]:
            kind = "lake_outlet"
        elif lake_inlet[row, col]:
            kind = "lake_inlet"
        elif not downstream_is_channel[row, col]:
            kind = "outlet"
        elif upstream_count[row, col] == 0:
            kind = "source"
        else:
            kind = "junction"
        catchment_value = int(catchments[row, col])
        lake_value = _NODATA_U32 if lakes is None else int(lakes[row, col])
        nodes.append(
            RiverNode(
                node_id=node_id,
                kind=kind,
                row=row,
                col=col,
                x_m=origin_x_m + (col + 0.5) * resolution_m,
                z_m=origin_z_m + (row + 0.5) * resolution_m,
                elevation_m=float(elevation[row, col]),
                catchment_id=None if catchment_value == _NODATA_U32 else catchment_value,
                lake_id=None if lake_value == _NODATA_U32 else lake_value,
            )
        )

    edges: list[RiverEdge] = []
    for from_cell in node_cells:
        row, col = divmod(int(from_cell), cols)
        if not downstream_is_channel[row, col]:
            continue
        cells = [(row, col)]
        length_m = 0.0
        current = int(from_cell)
        while True:
            current_row, current_col = divmod(current, cols)
            code = int(flow[current_row, current_col])
            next_row = current_row + int(_DROW[code])
            next_col = current_col + int(_DCOL[code])
            length_m += resolution_m * (np.sqrt(2.0) if code in (2, 4, 6, 8) else 1.0)
            current = next_row * cols + next_col
            cells.append((next_row, next_col))
            if current in node_by_cell:
                break
        from_node = node_by_cell[int(from_cell)]
        to_node = node_by_cell[current]
        edge_id = len(edges)
        q = None if discharge is None else float(discharge[cells[-1]])
        edges.append(
            RiverEdge(
                edge_id=edge_id,
                from_node_id=from_node,
                to_node_id=to_node,
                cells=tuple(cells),
                length_m=float(length_m),
                upstream_area_m2=float(accumulation[cells[-1]]),
                mean_discharge_m3s=q,
                stream_order=int(np.max([orders[cell] for cell in cells])),
            )
        )
    return RiverGraph(tuple(nodes), tuple(edges))


def write_river_graph(
    connection: sqlite3.Connection,
    graph: RiverGraph,
    *,
    level_name: str,
    elevation_m: np.ndarray,
    resolution_m: float,
    origin_x_m: float = 0.0,
    origin_z_m: float = 0.0,
) -> None:
    """Write a graph into an empty schema-v1 network database."""

    existing = connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    if existing:
        raise ValueError("Network database already contains nodes")
    elevation = np.asarray(elevation_m)
    with connection:
        connection.executemany(
            """INSERT INTO nodes
               (node_id, kind, level_name, row_index, col_index, x_m, z_m,
                elevation_m, catchment_id, lake_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    node.node_id, node.kind, level_name, node.row, node.col,
                    node.x_m, node.z_m, node.elevation_m,
                    node.catchment_id, node.lake_id,
                )
                for node in graph.nodes
            ],
        )
        connection.executemany(
            """INSERT INTO edges
               (edge_id, from_node_id, to_node_id, kind, length_m,
                upstream_area_m2, mean_discharge_m3s, stream_order, width_m, depth_m)
               VALUES (?, ?, ?, 'river', ?, ?, ?, ?, NULL, NULL)""",
            [
                (
                    edge.edge_id, edge.from_node_id, edge.to_node_id,
                    edge.length_m, edge.upstream_area_m2,
                    edge.mean_discharge_m3s, edge.stream_order,
                )
                for edge in graph.edges
            ],
        )
        point_rows = []
        for edge in graph.edges:
            for sequence, (row, col) in enumerate(edge.cells):
                point_rows.append(
                    (
                        edge.edge_id, sequence, row, col,
                        origin_x_m + (col + 0.5) * resolution_m,
                        origin_z_m + (row + 0.5) * resolution_m,
                        float(elevation[row, col]),
                    )
                )
        connection.executemany(
            """INSERT INTO edge_points
               (edge_id, sequence_index, row_index, col_index, x_m, z_m, elevation_m)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            point_rows,
        )


def append_river_graph(
    connection: sqlite3.Connection,
    graph: RiverGraph,
    *,
    level_name: str,
    elevation_m: np.ndarray,
    resolution_m: float,
    row_offset: int,
    col_offset: int,
    origin_x_m: float = 0.0,
    origin_z_m: float = 0.0,
    manage_transaction: bool = True,
) -> tuple[int, int]:
    """Append a regional graph with globally unique IDs and grid indices."""

    if resolution_m <= 0 or min(row_offset, col_offset) < 0:
        raise ValueError("Regional graph geometry is invalid")
    elevation = np.asarray(elevation_m)
    node_start = int(
        connection.execute("SELECT COALESCE(MAX(node_id), -1) + 1 FROM nodes").fetchone()[0]
    )
    edge_start = int(
        connection.execute("SELECT COALESCE(MAX(edge_id), -1) + 1 FROM edges").fetchone()[0]
    )
    transaction = connection if manage_transaction else nullcontext()
    with transaction:
        connection.executemany(
            """INSERT INTO nodes
               (node_id, kind, level_name, row_index, col_index, x_m, z_m,
                elevation_m, catchment_id, lake_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    node_start + node.node_id,
                    node.kind,
                    level_name,
                    row_offset + node.row,
                    col_offset + node.col,
                    origin_x_m + (col_offset + node.col + 0.5) * resolution_m,
                    origin_z_m + (row_offset + node.row + 0.5) * resolution_m,
                    node.elevation_m,
                    node.catchment_id,
                    node.lake_id,
                )
                for node in graph.nodes
            ],
        )
        connection.executemany(
            """INSERT INTO edges
               (edge_id, from_node_id, to_node_id, kind, length_m,
                upstream_area_m2, mean_discharge_m3s, stream_order, width_m, depth_m)
               VALUES (?, ?, ?, 'river', ?, ?, ?, ?, NULL, NULL)""",
            [
                (
                    edge_start + edge.edge_id,
                    node_start + edge.from_node_id,
                    node_start + edge.to_node_id,
                    edge.length_m,
                    edge.upstream_area_m2,
                    edge.mean_discharge_m3s,
                    edge.stream_order,
                )
                for edge in graph.edges
            ],
        )
        point_rows = []
        for edge in graph.edges:
            for sequence, (row, col) in enumerate(edge.cells):
                point_rows.append(
                    (
                        edge_start + edge.edge_id,
                        sequence,
                        row_offset + row,
                        col_offset + col,
                        origin_x_m + (col_offset + col + 0.5) * resolution_m,
                        origin_z_m + (row_offset + row + 0.5) * resolution_m,
                        float(elevation[row, col]),
                    )
                )
        connection.executemany(
            """INSERT INTO edge_points
               (edge_id, sequence_index, row_index, col_index, x_m, z_m, elevation_m)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            point_rows,
        )
        connection.execute(
            """UPDATE lakes
               SET outlet_node_id = (
                   SELECT MIN(nodes.node_id) FROM nodes
                   WHERE nodes.kind = 'lake_outlet'
                     AND nodes.lake_id = lakes.lake_id
               )
               WHERE outlet_node_id IS NULL
                 AND EXISTS (
                   SELECT 1 FROM nodes
                   WHERE nodes.kind = 'lake_outlet'
                     AND nodes.lake_id = lakes.lake_id
               )"""
        )
    return node_start, edge_start


@njit(cache=True)
def _channel_upstream_count(flow, channels):
    rows, cols = flow.shape
    result = np.zeros(flow.shape, dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            if not channels[row, col]:
                continue
            code = int(flow[row, col])
            if code < 1 or code > 8:
                continue
            next_row = row + _DROW[code]
            next_col = col + _DCOL[code]
            if channels[next_row, next_col] and result[next_row, next_col] < 255:
                result[next_row, next_col] += 1
    return result


@njit(cache=True)
def _downstream_channel_mask(flow, channels):
    rows, cols = flow.shape
    result = np.zeros(flow.shape, dtype=np.bool_)
    for row in range(rows):
        for col in range(cols):
            if not channels[row, col]:
                continue
            code = int(flow[row, col])
            if code < 1 or code > 8:
                continue
            result[row, col] = channels[
                row + _DROW[code], col + _DCOL[code]
            ]
    return result


@njit(cache=True)
def _lake_transition_nodes(flow, channels, lakes):
    rows, cols = flow.shape
    inlet = np.zeros(flow.shape, dtype=np.bool_)
    outlet = np.zeros(flow.shape, dtype=np.bool_)
    for row in range(rows):
        for col in range(cols):
            if not channels[row, col]:
                continue
            lake = lakes[row, col]
            code = int(flow[row, col])
            if lake != _NODATA_U32 and 1 <= code <= 8:
                next_row = row + _DROW[code]
                next_col = col + _DCOL[code]
                if lakes[next_row, next_col] != lake:
                    outlet[row, col] = True
            if lake == _NODATA_U32:
                continue
            for upstream_row in range(max(0, row - 1), min(rows, row + 2)):
                for upstream_col in range(max(0, col - 1), min(cols, col + 2)):
                    if not channels[upstream_row, upstream_col]:
                        continue
                    upstream_code = int(flow[upstream_row, upstream_col])
                    if upstream_code < 1 or upstream_code > 8:
                        continue
                    if (
                        upstream_row + _DROW[upstream_code] == row
                        and upstream_col + _DCOL[upstream_code] == col
                        and lakes[upstream_row, upstream_col] != lake
                    ):
                        inlet[row, col] = True
    return inlet, outlet
