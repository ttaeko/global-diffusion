"""Depression inventory and explicit lake surfaces for routed terrain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage


_NODATA_U32 = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class LakeRecord:
    lake_id: int
    cell_count: int
    area_m2: float
    maximum_depth_m: float
    mean_depth_m: float
    volume_m3: float
    surface_elevation_m: float


@dataclass(frozen=True)
class LakePlan:
    lake_id: np.ndarray
    lake_mask: np.ndarray
    water_surface_elevation_m: np.ndarray
    terrain_elevation_m: np.ndarray
    records: tuple[LakeRecord, ...]


def identify_depression_lakes(
    elevation_raw_m: np.ndarray,
    elevation_conditioned_m: np.ndarray,
    *,
    resolution_m: float,
    minimum_area_km2: float = 0.25,
    minimum_maximum_depth_m: float = 2.0,
    depth_tolerance_m: float = 0.05,
    land_mask: np.ndarray | None = None,
    maximum_lake_area_km2: float = 200.0,
    maximum_total_lake_fraction: float = 0.005,
) -> LakePlan:
    """Retain substantial filled depressions as lakes instead of flat terrain.

    Small/shallow components remain part of the conditioned drainage surface.
    For retained lakes the physical terrain is the original basin floor and a
    separate, level water surface is stored above it.
    """

    raw = np.asarray(elevation_raw_m, dtype=np.float32)
    conditioned = np.asarray(elevation_conditioned_m, dtype=np.float32)
    if raw.shape != conditioned.shape or raw.ndim != 2:
        raise ValueError("raw and conditioned elevation must be matching 2D arrays")
    if resolution_m <= 0 or minimum_area_km2 < 0:
        raise ValueError("resolution must be positive and minimum area non-negative")
    if minimum_maximum_depth_m < 0 or depth_tolerance_m < 0:
        raise ValueError("depth thresholds must be non-negative")
    if maximum_lake_area_km2 <= 0:
        raise ValueError("maximum_lake_area_km2 must be positive")
    if not 0 <= maximum_total_lake_fraction <= 1:
        raise ValueError("maximum_total_lake_fraction must lie between zero and one")
    if land_mask is None:
        land = np.isfinite(raw)
    else:
        land = np.asarray(land_mask, dtype=bool)
        if land.shape != raw.shape:
            raise ValueError("land_mask shape does not match elevation")

    depth = conditioned - raw
    depression = np.isfinite(depth) & (depth > depth_tolerance_m)
    components, component_count = scipy.ndimage.label(
        depression,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    cell_area_m2 = float(resolution_m) ** 2
    minimum_cells = max(
        1, int(np.ceil(minimum_area_km2 * 1_000_000.0 / cell_area_m2))
    )

    component_ids = np.arange(1, component_count + 1, dtype=np.int32)
    counts = np.bincount(components.ravel(), minlength=component_count + 1)[1:]
    maximum_depths = np.asarray(
        scipy.ndimage.maximum(depth, components, index=component_ids), dtype=np.float64
    )
    depth_sums = np.asarray(
        scipy.ndimage.sum(depth, components, index=component_ids), dtype=np.float64
    )
    # A simple filled depression has a constant conditioned spill surface. The
    # component mean remains stable for the rare nested numerical shelf.
    surfaces = np.asarray(
        scipy.ndimage.mean(conditioned, components, index=component_ids),
        dtype=np.float64,
    )
    eligible = (counts >= minimum_cells) & (
        maximum_depths >= minimum_maximum_depth_m
    ) & (counts * cell_area_m2 <= maximum_lake_area_km2 * 1_000_000.0)
    # Spend most of the budget on small components, then sample the remainder
    # in stable hash order. This preserves many ordinary lakes plus a limited
    # long tail, instead of filling the world budget with a few giant basins.
    selected = np.zeros(component_count, dtype=bool)
    budget_cells = int(np.floor(np.count_nonzero(land) * maximum_total_lake_fraction))
    used_cells = 0
    eligible_indices = np.flatnonzero(eligible)
    component_keys = (
        (eligible_indices.astype(np.uint64) + 1) * np.uint64(2654435761)
    ) & np.uint64(0xFFFFFFFF)
    small_first = eligible_indices[np.lexsort((component_keys, counts[eligible_indices]))]
    small_budget = int(budget_cells * 0.70)
    for index in small_first:
        cell_count = int(counts[index])
        if used_cells + cell_count > small_budget:
            continue
        selected[index] = True
        used_cells += cell_count
    remaining = eligible_indices[~selected[eligible_indices]]
    remaining_keys = (
        (remaining.astype(np.uint64) + 1) * np.uint64(2654435761)
    ) & np.uint64(0xFFFFFFFF)
    for index in remaining[np.argsort(remaining_keys)]:
        cell_count = int(counts[index])
        if used_cells + cell_count > budget_cells:
            continue
        selected[index] = True
        used_cells += cell_count
    selected_components = component_ids[selected]
    selected_counts = counts[selected]
    selected_maximum_depths = maximum_depths[selected]
    selected_depth_sums = depth_sums[selected]
    selected_surfaces = surfaces[selected]

    id_lookup = np.full(component_count + 1, _NODATA_U32, dtype=np.uint32)
    id_lookup[selected_components] = np.arange(
        selected_components.size, dtype=np.uint32
    )
    lake_ids = id_lookup[components]
    surface_lookup = np.full(component_count + 1, np.nan, dtype=np.float32)
    surface_lookup[selected_components] = selected_surfaces.astype(np.float32)
    water_surface = surface_lookup[components]
    records = [
        LakeRecord(
            lake_id=lake_id,
            cell_count=int(cell_count),
            area_m2=float(cell_count * cell_area_m2),
            maximum_depth_m=float(maximum_depth),
            mean_depth_m=float(depth_sum / cell_count),
            volume_m3=float(depth_sum * cell_area_m2),
            surface_elevation_m=float(surface),
        )
        for lake_id, (cell_count, maximum_depth, depth_sum, surface) in enumerate(
            zip(
                selected_counts,
                selected_maximum_depths,
                selected_depth_sums,
                selected_surfaces,
            )
        )
    ]

    lake_mask = lake_ids != _NODATA_U32
    # The filled surface is a routing construct. Physical terrain remains the
    # generated surface; only water level is added for retained lakes. River
    # incision is performed by the finer hydrology-conditioned stage.
    terrain = raw.copy()
    return LakePlan(
        lake_id=lake_ids,
        lake_mask=lake_mask,
        water_surface_elevation_m=water_surface,
        terrain_elevation_m=terrain,
        records=tuple(records),
    )
