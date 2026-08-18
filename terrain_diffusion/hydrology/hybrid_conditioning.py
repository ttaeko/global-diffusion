"""Selective fill-and-breach conditioning for generated terrain.

Generated elevation contains many harmless sub-grid pits and a much smaller
number of large artificial basins. Filling every basin creates broad flats.
This module fills small pits but cuts narrow outlet beds through large/deep
components, then reroutes iteratively.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from .compiled_routing import CompiledRoutingResult, priority_flood_route_compiled
from .world_plan import D8_DIRECTION_OFFSETS


@dataclass(frozen=True)
class BreachPassMetrics:
    pass_index: int
    candidate_components: int
    breached_paths: int
    breached_cells: int
    maximum_incision_m: float
    fill_fraction_above_tolerance_before: float
    skipped_excessive_incision_paths: int


@dataclass(frozen=True)
class HybridConditioningResult:
    elevation_breached_m: np.ndarray
    routing: CompiledRoutingResult
    breach_mask: np.ndarray
    metrics: tuple[BreachPassMetrics, ...]


def hybrid_fill_breach_route(
    elevation_m: np.ndarray,
    *,
    resolution_m: float,
    land_mask: np.ndarray | None = None,
    fill_tolerance_m: float = 10.0,
    breach_minimum_area_km2: float = 10.0,
    breach_minimum_depth_m: float = 50.0,
    minimum_bed_slope: float = 1e-5,
    preserve_mask: np.ndarray | None = None,
    maximum_breach_incision_m: float | None = 800.0,
    terminal_mask: np.ndarray | None = None,
    initial_accumulation_area_m2: np.ndarray | None = None,
    routing_zones: np.ndarray | None = None,
    passes: int = 2,
) -> HybridConditioningResult:
    """Condition terrain with narrow breaches instead of blanket large fills."""

    terrain = np.asarray(elevation_m, dtype=np.float32).copy()
    generated_terrain = terrain.copy()
    if terrain.ndim != 2 or min(terrain.shape) == 0:
        raise ValueError("elevation_m must be a non-empty 2D array")
    if land_mask is None:
        land = np.isfinite(terrain) & (terrain > 0)
    else:
        land = np.asarray(land_mask, dtype=bool)
        if land.shape != terrain.shape:
            raise ValueError("land_mask must align with elevation_m")
    if fill_tolerance_m <= 0 or breach_minimum_area_km2 <= 0:
        raise ValueError("Fill tolerance and breach area must be positive")
    if breach_minimum_depth_m <= fill_tolerance_m or minimum_bed_slope < 0:
        raise ValueError("Breach depth must exceed fill tolerance and slope cannot be negative")
    if passes <= 0:
        raise ValueError("passes must be positive")
    if preserve_mask is None:
        preserve = np.zeros(terrain.shape, dtype=bool)
    else:
        preserve = np.asarray(preserve_mask, dtype=bool)
        if preserve.shape != terrain.shape:
            raise ValueError("preserve_mask must align with elevation_m")
    if maximum_breach_incision_m is not None and maximum_breach_incision_m <= 0:
        raise ValueError("maximum_breach_incision_m must be positive when supplied")

    breach_mask = np.zeros(terrain.shape, dtype=bool)
    history: list[BreachPassMetrics] = []
    routing = priority_flood_route_compiled(
        terrain,
        resolution_m=resolution_m,
        land_mask=land,
        terminal_mask=terminal_mask,
        initial_accumulation_area_m2=initial_accumulation_area_m2,
        routing_zones=routing_zones,
    )
    cell_area_km2 = resolution_m**2 / 1e6
    for pass_index in range(passes):
        correction = routing.elevation_correction_m
        problem = land & ~preserve & (correction > fill_tolerance_m)
        labels, component_count = scipy.ndimage.label(
            problem, structure=np.ones((3, 3), dtype=np.uint8)
        )
        if component_count == 0:
            break
        sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
        component_ids = np.arange(1, component_count + 1, dtype=np.int32)
        depths = scipy.ndimage.maximum(correction, labels, index=component_ids)
        selected = component_ids[
            (sizes[1:] * cell_area_km2 >= breach_minimum_area_km2)
            | (depths >= breach_minimum_depth_m)
        ]
        if selected.size == 0:
            break
        pits = scipy.ndimage.maximum_position(correction, labels, index=selected)
        breached_paths = 0
        cells_before = int(breach_mask.sum())
        maximum_incision = 0.0
        skipped_excessive = 0
        for component_id, pit in zip(selected, pits):
            path = _downstream_escape_path(
                routing.flow_direction,
                terrain,
                land,
                labels,
                int(component_id),
                (int(pit[0]), int(pit[1])),
                preserve,
            )
            if path and preserve[path[-1]]:
                path = path[:-1]
            if len(path) < 2:
                continue
            start_elevation = float(terrain[path[0]])
            end_elevation = min(
                float(terrain[path[-1]]),
                start_elevation - minimum_bed_slope * resolution_m * (len(path) - 1),
            )
            bed = np.linspace(
                start_elevation, end_elevation, len(path), dtype=np.float32
            )
            rows = np.fromiter((cell[0] for cell in path), dtype=np.intp)
            cols = np.fromiter((cell[1] for cell in path), dtype=np.intp)
            original = terrain[rows, cols].copy()
            proposed = np.minimum(original, bed)
            incision = original - proposed
            if (
                maximum_breach_incision_m is not None
                and float(
                    np.max(generated_terrain[rows, cols] - proposed)
                ) > maximum_breach_incision_m
            ):
                skipped_excessive += 1
                continue
            terrain[rows, cols] = proposed
            changed = incision > 1e-4
            if np.any(changed):
                breach_mask[rows[changed], cols[changed]] = True
                maximum_incision = max(maximum_incision, float(incision.max()))
                breached_paths += 1
        history.append(
            BreachPassMetrics(
                pass_index=pass_index,
                candidate_components=int(selected.size),
                breached_paths=breached_paths,
                breached_cells=int(breach_mask.sum()) - cells_before,
                maximum_incision_m=maximum_incision,
                fill_fraction_above_tolerance_before=float(problem.sum() / land.sum()),
                skipped_excessive_incision_paths=skipped_excessive,
            )
        )
        if breached_paths == 0:
            break
        routing = priority_flood_route_compiled(
            terrain,
            resolution_m=resolution_m,
            land_mask=land,
            terminal_mask=terminal_mask,
            initial_accumulation_area_m2=initial_accumulation_area_m2,
            routing_zones=routing_zones,
        )
    return HybridConditioningResult(
        elevation_breached_m=terrain,
        routing=routing,
        breach_mask=breach_mask,
        metrics=tuple(history),
    )


def _downstream_escape_path(
    flow_direction: np.ndarray,
    elevation_m: np.ndarray,
    land_mask: np.ndarray,
    component_labels: np.ndarray,
    component_id: int,
    start: tuple[int, int],
    preserve_mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Follow filled routing past the saddle to terrain lower than the pit."""

    rows, cols = elevation_m.shape
    path = [start]
    row, col = start
    start_elevation = float(elevation_m[row, col])
    left_component = False
    maximum_steps = rows * cols
    for _ in range(maximum_steps):
        code = int(flow_direction[row, col])
        if code < 1 or code > 8:
            break
        delta_row, delta_col = D8_DIRECTION_OFFSETS[code]
        row += delta_row
        col += delta_col
        if row < 0 or row >= rows or col < 0 or col >= cols:
            break
        path.append((row, col))
        if preserve_mask[row, col]:
            break
        if int(component_labels[row, col]) != component_id:
            left_component = True
        if left_component and (
            not land_mask[row, col]
            or float(elevation_m[row, col]) < start_elevation
        ):
            break
    return path
