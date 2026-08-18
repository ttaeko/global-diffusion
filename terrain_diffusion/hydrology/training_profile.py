"""Local river-profile targets for hydrology-aware decoder training.

The persistent hydrology contract describes topology and this module turns that
topology into an authoritative terrain base. Version 4 performs minimal local
repairs only where a planned channel would otherwise rise downstream. Valley,
bank, shoreline, and floodplain morphology remain learned-model responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from terrain_diffusion.hydrology.compiled_routing import processing_order_from_d8
from terrain_diffusion.hydrology.profile_contract import (
    DEFAULT_HYDROLOGY_PROFILE,
)
from terrain_diffusion.hydrology.world_plan import D8_DIRECTION_OFFSETS


HYDROLOGY_PROFILE_CHANNELS = (
    "channel_mask",
    "profile_incision",
    "target_channel_grade",
    "valley_corridor",
)

# Lake shorelines and 30 m channel centre-lines are rasterized independently.
# Preserve sub-block overlaps instead of raising or incising terrain at their
# shared edge; larger conflicts still indicate an invalid routing assignment.
_OUTLET_LEVEL_RASTER_TOLERANCE_M = 0.5


@dataclass(frozen=True)
class HydrologyTrainingProfile:
    conditioning: np.ndarray
    channel_mask: np.ndarray
    target_bed_elevation_m: np.ndarray
    target_downstream_drop_m: np.ndarray
    terrain_correction_m: np.ndarray
    feasible_edge_mask: np.ndarray


def flow_direction_from_vectors(
    flow_east: np.ndarray,
    flow_south: np.ndarray,
) -> np.ndarray:
    """Recover repository D8 codes from normalized east/south vectors."""

    east = np.asarray(flow_east, dtype=np.float32)
    south = np.asarray(flow_south, dtype=np.float32)
    if east.shape != south.shape:
        raise ValueError("Flow-vector components must align")
    delta_col = np.sign(east).astype(np.int8)
    delta_row = np.sign(south).astype(np.int8)
    flow = np.zeros(east.shape, dtype=np.uint8)
    for code, (row_offset, column_offset) in D8_DIRECTION_OFFSETS.items():
        flow[(delta_row == row_offset) & (delta_col == column_offset)] = code
    flow[(np.abs(east) + np.abs(south)) < 0.25] = 0
    return flow


def build_hydrology_training_profile(
    lowres_signed_sqrt: np.ndarray,
    hydrology: np.ndarray,
    *,
    resolution_m: float = 30.0,
    minimum_grade: float = 0.0,
    maximum_grade: float = 0.25,
    maximum_incision_m: float = (
        DEFAULT_HYDROLOGY_PROFILE.maximum_profile_incision_m
    ),
    corridor_half_width_m: float = 60.0,
    corridor_truncation: float = 2.5,
    sea_level_elevation_m: float = 0.0,
    lake_water_surface_elevation_m: np.ndarray | None = None,
    strict_outlet_floor: bool = False,
) -> HydrologyTrainingProfile:
    """Create a minimally repaired, incision-only channel-bed target.

    Natural downstream drops and flats are retained. Only rising edges are
    repaired, bounded relative to the source DEM and to the terminal sea/lake
    water level inherited by that channel. Edges that cannot be made
    nonascending within the local incision cap are excluded from the strict
    grade loss and expose a routing/planning defect instead of propagating a
    deep trench through the terrain.
    """

    values = np.asarray(hydrology, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 8:
        raise ValueError("Hydrology conditioning must have at least eight channels")
    signed_sqrt = np.asarray(lowres_signed_sqrt, dtype=np.float32)
    if signed_sqrt.shape != values.shape[1:]:
        raise ValueError("Terrain and hydrology crops must align")
    if resolution_m <= 0 or maximum_incision_m <= 0:
        raise ValueError("Resolution and maximum incision must be positive")
    if not 0 <= minimum_grade <= maximum_grade:
        raise ValueError("Hydrology grade bounds are invalid")
    if corridor_half_width_m <= 0 or corridor_truncation <= 0:
        raise ValueError("Corridor dimensions must be positive")
    if not np.isfinite(sea_level_elevation_m):
        raise ValueError("Sea level must be finite")

    elevation = np.sign(signed_sqrt) * np.square(signed_sqrt)
    channel = (values[5] > 0) & (values[7] < 0.5) & np.isfinite(elevation)
    flow = flow_direction_from_vectors(values[3], values[4])
    height, width = elevation.shape

    # Crops cut some world-graph edges.  Convert those exits to local terminals
    # so processing_order_from_d8 sees a closed, valid graph.
    rows, columns = np.indices(flow.shape)
    row_offsets = np.zeros(flow.shape, dtype=np.int8)
    column_offsets = np.zeros(flow.shape, dtype=np.int8)
    for code, (row_offset, column_offset) in D8_DIRECTION_OFFSETS.items():
        mask = flow == code
        row_offsets[mask] = row_offset
        column_offsets[mask] = column_offset
    target_rows = rows + row_offsets
    target_columns = columns + column_offsets
    exits = (
        (target_rows < 0)
        | (target_rows >= height)
        | (target_columns < 0)
        | (target_columns >= width)
    )
    flow[exits] = 0
    valid = np.isfinite(elevation)
    order = processing_order_from_d8(
        np.ascontiguousarray(flow), np.ascontiguousarray(valid)
    )

    lakes = values[7] >= 0.5
    if lake_water_surface_elevation_m is None:
        lake_surfaces = np.full(elevation.shape, np.nan, dtype=np.float32)
    else:
        lake_surfaces = np.asarray(
            lake_water_surface_elevation_m, dtype=np.float32
        )
        if lake_surfaces.shape != elevation.shape:
            raise ValueError("Lake water surfaces must align with terrain")
        lake_surfaces = np.where(
            lakes & np.isfinite(lake_surfaces), lake_surfaces, np.nan
        ).astype(np.float32)

    # Propagate the terminal sea/lake water level upstream over each channel
    # component. A channel may never be deterministically cut below its outlet.
    outlet_floor = np.full(
        elevation.shape, float(sea_level_elevation_m), dtype=np.float32
    )
    for flat_index in order[::-1]:
        row, column = divmod(int(flat_index), width)
        if not channel[row, column]:
            continue
        code = int(flow[row, column])
        if code not in D8_DIRECTION_OFFSETS:
            continue
        row_offset, column_offset = D8_DIRECTION_OFFSETS[code]
        next_row = row + row_offset
        next_column = column + column_offset
        if channel[next_row, next_column]:
            outlet_floor[row, column] = outlet_floor[next_row, next_column]
        elif lakes[next_row, next_column]:
            surface = lake_surfaces[next_row, next_column]
            if np.isfinite(surface):
                outlet_floor[row, column] = surface

    materially_below_outlet = channel & (
        elevation
        < outlet_floor - _OUTLET_LEVEL_RASTER_TOLERANCE_M
    )
    if strict_outlet_floor and np.any(materially_below_outlet):
        deficit = outlet_floor[materially_below_outlet] - elevation[
            materially_below_outlet
        ]
        first_row, first_column = np.argwhere(materially_below_outlet)[0]
        raise ValueError(
            "Channel source terrain lies below its terminal sea/lake level; "
            f"cells={int(materially_below_outlet.sum())}, "
            f"maximum_deficit_m={float(np.max(deficit)):.6f}, "
            f"first_cell=({int(first_row)}, {int(first_column)}), "
            f"terrain_m={float(elevation[first_row, first_column]):.6f}, "
            f"outlet_level_m={float(outlet_floor[first_row, first_column]):.6f}"
        )
    # Training DEMs can encode a river bed below an observed water surface,
    # while rasterized generated shorelines can differ by a few centimetres.
    # An incision-only transform cannot raise either case. Preserve the source
    # height exactly and prohibit deterministic repair from deepening it.
    source_below_outlet = channel & (elevation < outlet_floor)
    outlet_floor[source_below_outlet] = elevation[source_below_outlet]

    bed = elevation.copy()
    target_drop = np.zeros(elevation.shape, dtype=np.float32)
    feasible = np.zeros(elevation.shape, dtype=bool)
    for flat_index in order:
        row, column = divmod(int(flat_index), width)
        if not channel[row, column]:
            continue
        code = int(flow[row, column])
        if code not in D8_DIRECTION_OFFSETS:
            continue
        row_offset, column_offset = D8_DIRECTION_OFFSETS[code]
        next_row = row + row_offset
        next_column = column + column_offset
        if not channel[next_row, next_column]:
            continue
        distance = resolution_m * (
            np.sqrt(2.0) if row_offset and column_offset else 1.0
        )
        requested_drop = minimum_grade * distance
        maximum_downstream_bed = bed[row, column] - requested_drop
        if bed[next_row, next_column] > maximum_downstream_bed:
            proposed = maximum_downstream_bed
            lower_bound = max(
                float(elevation[next_row, next_column] - maximum_incision_m),
                float(outlet_floor[next_row, next_column]),
            )
            bed[next_row, next_column] = min(
                bed[next_row, next_column], max(lower_bound, proposed)
            )
        achieved_drop = bed[row, column] - bed[next_row, next_column]
        target_drop[row, column] = max(0.0, achieved_drop)
        feasible[row, column] = achieved_drop >= requested_drop - 1e-4

    incision = np.zeros(elevation.shape, dtype=np.float32)
    incision[channel] = np.maximum(0.0, elevation[channel] - bed[channel])
    if np.any(channel):
        distance_pixels, nearest = scipy.ndimage.distance_transform_edt(
            ~channel,
            return_indices=True,
        )
        distance_m = distance_pixels * resolution_m
        corridor = np.exp(
            -0.5 * np.square(distance_m / float(corridor_half_width_m))
        ).astype(np.float32)
        corridor[
            distance_m > corridor_half_width_m * corridor_truncation
        ] = 0.0
        # The corridor is conditioning only. Deterministic terrain repair is
        # confined to the prescribed bed; the learned 10 m model owns valley
        # and bank morphology.
        correction = incision.copy()
    else:
        corridor = np.zeros(elevation.shape, dtype=np.float32)
        correction = np.zeros(elevation.shape, dtype=np.float32)

    grade = target_drop / resolution_m
    grade_reference = 0.001
    grade_scale = np.log1p(maximum_grade / grade_reference)
    grade_normalized = np.log1p(grade / grade_reference) / grade_scale
    conditioning = np.stack(
        (
            channel.astype(np.float32),
            np.clip(incision / maximum_incision_m, 0.0, 1.0),
            np.clip(grade_normalized, 0.0, 1.0),
            corridor,
        ),
        axis=0,
    ).astype(np.float32)
    target_bed = np.full(elevation.shape, np.nan, dtype=np.float32)
    target_bed[channel] = bed[channel]
    return HydrologyTrainingProfile(
        conditioning=conditioning,
        channel_mask=channel,
        target_bed_elevation_m=target_bed,
        target_downstream_drop_m=target_drop,
        terrain_correction_m=correction.astype(np.float32),
        feasible_edge_mask=feasible,
    )


def apply_hydrology_terrain_transform(
    elevation_m: np.ndarray,
    profile: HydrologyTrainingProfile,
) -> np.ndarray:
    """Apply the V4 deterministic minimal bed-repair transform.

    Channel cells are set exactly to the prescribed bed. The surrounding
    corridor is conditioning only, so the transform cannot raise terrain or
    pre-empt the learned model's valley and bank morphology.
    """

    elevation = np.asarray(elevation_m, dtype=np.float32)
    if elevation.shape != profile.terrain_correction_m.shape:
        raise ValueError("Elevation and hydrology profile must align")
    correction = np.asarray(profile.terrain_correction_m, dtype=np.float32)
    if np.any(correction[np.isfinite(correction)] < 0):
        raise RuntimeError("Hydrology terrain correction must be incision-only")
    transformed = elevation - correction
    target = np.asarray(profile.target_bed_elevation_m, dtype=np.float32)
    prescribed = profile.channel_mask & np.isfinite(target)
    transformed[prescribed] = target[prescribed]
    # The profile is built from signed-sqrt conditioning. Reconstructing an
    # otherwise unchanged float32 elevation through sqrt -> square can land one
    # ULP above the source value. Accept only that representational error, then
    # clamp it away so the returned terrain still satisfies the exact
    # incision-only invariant. Materially raised profile targets remain errors.
    float32_ulp = np.abs(np.spacing(elevation))
    numerical_tolerance = np.maximum(1e-5, 2.0 * float32_ulp)
    if np.any(transformed > elevation + numerical_tolerance):
        raise RuntimeError("Hydrology terrain transform attempted to raise terrain")
    transformed = np.minimum(transformed, elevation)
    if not np.isfinite(transformed[np.isfinite(elevation)]).all():
        raise RuntimeError("Hydrology terrain transform produced non-finite values")
    return transformed.astype(np.float32, copy=False)


def enforce_profile_on_refined_terrain(
    elevation_m: np.ndarray,
    profile: HydrologyTrainingProfile,
    *,
    refinement: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-assert the absolute bed after stochastic fine-scale refinement.

    A single signed vertical correction is applied to each fine block,
    preserving the decoder's within-block bank morphology. Unlike the
    pre-refinement terrain transform, this post-model reassertion may lift a
    stochastic undercut back to the already incision-only authoritative bed.
    The returned second array is the signed high-resolution correction in
    metres; positive values incise and negative values lift.
    """

    elevation = np.asarray(elevation_m, dtype=np.float32)
    if refinement <= 1:
        raise ValueError("refinement must exceed one")
    low_shape = profile.channel_mask.shape
    if elevation.shape != (low_shape[0] * refinement, low_shape[1] * refinement):
        raise ValueError("Refined terrain does not match the hydrology profile")
    low = np.median(
        elevation.reshape(
            low_shape[0], refinement, low_shape[1], refinement
        ),
        axis=(1, 3),
    )
    target = profile.target_bed_elevation_m
    prescribed = profile.channel_mask & np.isfinite(target)
    required = np.zeros(low_shape, dtype=np.float32)
    required[prescribed] = low[prescribed] - target[prescribed]
    if np.any(prescribed):
        _, nearest = scipy.ndimage.distance_transform_edt(
            ~prescribed, return_indices=True
        )
        correction_low = required[nearest[0], nearest[1]] * profile.conditioning[3]
    else:
        correction_low = required
    correction = np.repeat(
        np.repeat(correction_low, refinement, axis=0), refinement, axis=1
    ).astype(np.float32)
    transformed = elevation - correction
    reduced = np.median(
        transformed.reshape(
            low_shape[0], refinement, low_shape[1], refinement
        ),
        axis=(1, 3),
    )
    if np.any(
        np.abs(reduced[prescribed] - target[prescribed])
        > np.maximum(1e-4, 2.0 * np.abs(np.spacing(target[prescribed])))
    ):
        raise RuntimeError("Post-refinement hydrology profile reassertion failed")
    return transformed.astype(np.float32), correction
