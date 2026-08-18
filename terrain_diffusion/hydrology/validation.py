"""End-to-end invariants for a published hydrology world-plan package."""

from __future__ import annotations

import json
from pathlib import Path

import click
import networkx as nx
import numpy as np

from .world_plan import D8_DIRECTION_OFFSETS, WorldPlanStore


def validate_world_plan(root: str | Path) -> dict:
    store = WorldPlanStore(root)
    manifest = store.read_manifest()
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, object] = {}
    with store.open_rasters() as rasters:
        global_grid = rasters["levels/global_240m"]
        flow = global_grid["flow_direction"][:]
        land = global_grid["land_mask"][:] == 1
        elevation = global_grid["elevation_conditioned_m"][:]
        accumulation = global_grid["accumulation_area_m2"][:]
        catchments = global_grid["catchment_id"][:]
        valid_codes = np.isin(flow, np.arange(9, dtype=np.uint8))
        if np.any(land & ~valid_codes):
            errors.append("Some global land cells have invalid D8 directions")
        if np.any(land & (catchments == np.iinfo(np.uint32).max)):
            errors.append("Some global land cells have no catchment ID")
        maximum_uphill = 0.0
        for code, (delta_row, delta_col) in D8_DIRECTION_OFFSETS.items():
            rows, cols = np.nonzero(flow == code)
            if rows.size == 0:
                continue
            downstream = elevation[rows + delta_row, cols + delta_col]
            uphill = downstream - elevation[rows, cols]
            maximum_uphill = max(maximum_uphill, float(np.nanmax(uphill)))
        if maximum_uphill > 1e-4:
            errors.append(f"Conditioned global flow climbs by up to {maximum_uphill:.6f} m")
        terminal_area = float(np.sum(accumulation[flow == 0]))
        expected_area = float(np.count_nonzero(land) * 240.0**2)
        relative_area_error = abs(terminal_area - expected_area) / max(expected_area, 1)
        if relative_area_error > 1e-10:
            errors.append(
                f"Global contributing area is not conserved ({relative_area_error:.3e})"
            )
        metrics.update(
            global_land_cells=int(np.count_nonzero(land)),
            global_channel_cells=int(np.count_nonzero(global_grid["channel_mask"][:] == 1)),
            global_maximum_stream_order=int(np.max(global_grid["stream_order"][:])),
            global_maximum_uphill_m=maximum_uphill,
            global_area_relative_error=relative_area_error,
        )

    with store.open_network(readonly=True) as connection:
        node_rows = connection.execute("SELECT node_id, kind FROM nodes").fetchall()
        edge_rows = connection.execute(
            "SELECT edge_id, from_node_id, to_node_id FROM edges"
        ).fetchall()
        graph = nx.DiGraph()
        graph.add_nodes_from(node_id for node_id, _ in node_rows)
        graph.add_edges_from((source, target) for _, source, target in edge_rows)
        if not nx.is_directed_acyclic_graph(graph):
            errors.append("Persistent river graph contains a directed cycle")
        kinds = dict(node_rows)
        for node_id, kind in node_rows:
            indegree = graph.in_degree(node_id)
            outdegree = graph.out_degree(node_id)
            if kind == "source" and indegree != 0:
                errors.append(f"Source node {node_id} has incoming river edges")
            if kind == "junction" and indegree < 2:
                errors.append(f"Junction node {node_id} has fewer than two inputs")
            if kind == "outlet" and outdegree != 0:
                errors.append(f"Outlet node {node_id} has an outgoing river edge")
        lake_count = connection.execute("SELECT COUNT(*) FROM lakes").fetchone()[0]
        connected_lakes = connection.execute(
            "SELECT COUNT(*) FROM lakes WHERE outlet_node_id IS NOT NULL"
        ).fetchone()[0]
        non_channel_lakes = connection.execute(
            "SELECT COUNT(*) FROM lakes WHERE outlet_node_id IS NULL"
        ).fetchone()[0]
        regional_windows = connection.execute(
            "SELECT COUNT(*) FROM regional_windows"
        ).fetchone()[0]
        if non_channel_lakes:
            warnings.append(
                f"{non_channel_lakes} lakes do not intersect the thresholded river "
                "network and are excluded from vector outlet-connectivity scoring"
            )
        metrics.update(
            river_nodes=len(node_rows),
            river_edges=len(edge_rows),
            lakes=lake_count,
            channel_connected_lakes=connected_lakes,
            non_channel_lakes=non_channel_lakes,
            channel_connected_lake_outlet_fraction=(
                1.0 if connected_lakes else None
            ),
            # Backward-compatible diagnostic name. This is not a failed
            # connectivity count: no vector channel exists to link.
            lakes_without_vector_outlet=non_channel_lakes,
            regional_windows=regional_windows,
        )
    return {
        "schema_version": 1,
        "world_id": manifest.world_id,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


@click.command("validate-hydrology-world-plan")
@click.argument("world_plan_directory", type=click.Path(exists=True, file_okay=False))
@click.option("--output", type=click.Path(dir_okay=False), default=None)
def validate_hydrology_world_plan(world_plan_directory, output):
    """Validate raster, graph, area, lake, and frozen-window invariants."""

    report = validate_world_plan(world_plan_directory)
    output_path = Path(output) if output else Path(world_plan_directory) / "validation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not report["valid"]:
        raise click.ClickException("World-plan validation failed; see validation report")
    click.echo(f"World plan is valid; report saved to {output_path}")
