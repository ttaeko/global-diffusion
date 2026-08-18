"""End-to-end deterministic hydrology planning over one terrain raster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compiled_routing import (
    CompiledRoutingResult,
    priority_flood_route_compiled,
    strahler_order_d8,
)
from .conditioning import HydrologyConditioning, build_hydrology_conditioning
from .lakes import LakePlan, identify_depression_lakes
from .profile_contract import DEFAULT_HYDROLOGY_PROFILE
from .routing import select_channels
from .runoff import mean_discharge_from_runoff


@dataclass(frozen=True)
class HydrologyPlannerConfig:
    channel_minimum_area_km2: float = (
        DEFAULT_HYDROLOGY_PROFILE.channel_minimum_area_km2
    )
    lake_minimum_area_km2: float = 0.25
    lake_minimum_depth_m: float = 2.0
    lake_maximum_area_km2: float = 200.0
    maximum_total_lake_fraction: float = 0.005
    reference_precipitation_mm_year: float = (
        DEFAULT_HYDROLOGY_PROFILE.reference_precipitation_mm_year
    )
    runoff_ratio: float = DEFAULT_HYDROLOGY_PROFILE.runoff_ratio
    conditioning_distance_scale_m: float = (
        DEFAULT_HYDROLOGY_PROFILE.conditioning_distance_scale_m
    )


@dataclass(frozen=True)
class PlannedHydrology:
    routing: CompiledRoutingResult
    lakes: LakePlan
    channel_mask: np.ndarray
    stream_order: np.ndarray
    mean_discharge_m3s: np.ndarray
    conditioning: HydrologyConditioning | None


def plan_hydrology(
    elevation_m: np.ndarray,
    *,
    resolution_m: float,
    land_mask: np.ndarray | None = None,
    precipitation_mm_year: np.ndarray | None = None,
    terminal_mask: np.ndarray | None = None,
    open_boundary: bool = True,
    routing_zones: np.ndarray | None = None,
    initial_accumulation_area_m2: np.ndarray | None = None,
    initial_discharge_m3s: np.ndarray | None = None,
    config: HydrologyPlannerConfig | None = None,
    build_conditioning: bool = True,
) -> PlannedHydrology:
    """Produce terrain corrections and all hydrology conditioning channels."""

    settings = config or HydrologyPlannerConfig()
    elevation = np.asarray(elevation_m, dtype=np.float32)
    if land_mask is None:
        land = np.isfinite(elevation) & (elevation > 0)
    else:
        land = np.asarray(land_mask, dtype=bool)
    routing = priority_flood_route_compiled(
        elevation,
        resolution_m=resolution_m,
        land_mask=land,
        terminal_mask=terminal_mask,
        open_boundary=open_boundary,
        initial_accumulation_area_m2=initial_accumulation_area_m2,
        routing_zones=routing_zones,
    )
    lakes = identify_depression_lakes(
        elevation,
        routing.elevation_conditioned_m,
        resolution_m=resolution_m,
        minimum_area_km2=settings.lake_minimum_area_km2,
        minimum_maximum_depth_m=settings.lake_minimum_depth_m,
        land_mask=land,
        maximum_lake_area_km2=settings.lake_maximum_area_km2,
        maximum_total_lake_fraction=settings.maximum_total_lake_fraction,
    )
    channels = select_channels(
        routing.accumulation_area_m2,
        minimum_area_km2=settings.channel_minimum_area_km2,
        land_mask=land,
    )
    order = strahler_order_d8(
        routing.flow_direction,
        routing.processing_order,
        channels,
    )
    if precipitation_mm_year is None:
        precipitation = np.full(
            elevation.shape,
            settings.reference_precipitation_mm_year,
            dtype=np.float32,
        )
    else:
        precipitation = np.asarray(precipitation_mm_year, dtype=np.float32)
    precipitation = precipitation.copy()
    precipitation[~np.isfinite(precipitation) | ~land] = 0
    discharge = mean_discharge_from_runoff(
        routing.flow_direction,
        routing.processing_order,
        precipitation,
        resolution_m=resolution_m,
        runoff_ratio=settings.runoff_ratio,
        initial_discharge_m3s=initial_discharge_m3s,
    )
    conditioning = None
    if build_conditioning:
        conditioning = build_hydrology_conditioning(
            routing.flow_direction,
            routing.accumulation_area_m2,
            routing.catchment_id,
            channels,
            order,
            resolution_m=resolution_m,
            lake_id=lakes.lake_id,
            mean_discharge_m3s=discharge,
            distance_scale_m=settings.conditioning_distance_scale_m,
        )
    return PlannedHydrology(
        routing=routing,
        lakes=lakes,
        channel_mask=channels,
        stream_order=order,
        mean_discharge_m3s=discharge,
        conditioning=conditioning,
    )
