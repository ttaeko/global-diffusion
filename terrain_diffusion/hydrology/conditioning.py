"""Hydrology-derived conditioning rasters for the future terrain refiner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage

from .world_plan import D8_DIRECTION_OFFSETS


_NODATA_U32 = np.iinfo(np.uint32).max

HYDROLOGY_CONDITIONING_CHANNELS = (
    "log_accumulation",
    "log_discharge",
    "channel_proximity",
    "flow_east",
    "flow_south",
    "stream_order",
    "catchment_boundary_proximity",
    "lake_mask",
)


@dataclass(frozen=True)
class HydrologyConditioning:
    values: np.ndarray
    channel_names: tuple[str, ...] = HYDROLOGY_CONDITIONING_CHANNELS


def build_hydrology_conditioning(
    flow_direction: np.ndarray,
    accumulation_area_m2: np.ndarray,
    catchment_id: np.ndarray,
    channel_mask: np.ndarray,
    stream_order: np.ndarray,
    *,
    resolution_m: float,
    lake_id: np.ndarray | None = None,
    mean_discharge_m3s: np.ndarray | None = None,
    distance_scale_m: float = 2000.0,
    accumulation_scale_km2: float = 1000.0,
) -> HydrologyConditioning:
    """Build stable, bounded channels without feeding arbitrary basin IDs."""

    flow = np.asarray(flow_direction, dtype=np.uint8)
    accumulation = np.asarray(accumulation_area_m2, dtype=np.float64)
    catchments = np.asarray(catchment_id, dtype=np.uint32)
    channels = np.asarray(channel_mask, dtype=bool)
    order = np.asarray(stream_order, dtype=np.uint8)
    if not (
        flow.shape == accumulation.shape == catchments.shape == channels.shape == order.shape
    ):
        raise ValueError("All hydrology inputs must have the same shape")
    if resolution_m <= 0 or distance_scale_m <= 0 or accumulation_scale_km2 <= 0:
        raise ValueError("Physical scales must be positive")

    valid = flow != 255
    log_accumulation = np.log1p(np.maximum(accumulation, 0) / 1_000_000.0)
    log_accumulation /= np.log1p(accumulation_scale_km2)
    log_accumulation = np.clip(log_accumulation, 0.0, 1.0)
    if mean_discharge_m3s is None:
        log_discharge = np.zeros(flow.shape, dtype=np.float64)
    else:
        discharge = np.asarray(mean_discharge_m3s, dtype=np.float64)
        if discharge.shape != flow.shape:
            raise ValueError("mean_discharge_m3s shape does not match flow_direction")
        log_discharge = np.clip(np.log1p(np.maximum(discharge, 0)) / np.log1p(1000.0), 0, 1)

    channel_distance = scipy.ndimage.distance_transform_edt(~channels) * resolution_m
    channel_proximity = np.exp(-channel_distance / distance_scale_m)

    flow_east = np.zeros(flow.shape, dtype=np.float32)
    flow_south = np.zeros(flow.shape, dtype=np.float32)
    for code, (delta_row, delta_col) in D8_DIRECTION_OFFSETS.items():
        mask = flow == code
        norm = np.sqrt(float(delta_row**2 + delta_col**2))
        flow_east[mask] = delta_col / norm
        flow_south[mask] = delta_row / norm

    maximum_order = max(1, int(np.max(order)))
    normalized_order = order.astype(np.float32) / maximum_order
    boundaries = _catchment_boundaries(catchments, valid)
    boundary_distance = scipy.ndimage.distance_transform_edt(~boundaries) * resolution_m
    boundary_proximity = np.exp(-boundary_distance / distance_scale_m)

    if lake_id is None:
        lakes = np.zeros(flow.shape, dtype=np.float32)
    else:
        ids = np.asarray(lake_id, dtype=np.uint32)
        if ids.shape != flow.shape:
            raise ValueError("lake_id shape does not match flow_direction")
        lakes = (ids != _NODATA_U32).astype(np.float32)

    values = np.stack(
        (
            log_accumulation,
            log_discharge,
            channel_proximity,
            flow_east,
            flow_south,
            normalized_order,
            boundary_proximity,
            lakes,
        ),
        axis=0,
    ).astype(np.float32)
    values[:, ~valid] = 0
    return HydrologyConditioning(values=values)


def _catchment_boundaries(catchments: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(catchments.shape, dtype=bool)
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left = tuple(left)
        right = tuple(right)
        different = (
            valid[left]
            & valid[right]
            & (catchments[left] != catchments[right])
        )
        boundaries[left] |= different
        boundaries[right] |= different
    return boundaries
