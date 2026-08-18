"""Convert climate into physically accumulated mean river discharge."""

from __future__ import annotations

import numpy as np

from .compiled_routing import accumulate_values_d8


SECONDS_PER_YEAR = 31_557_600.0


def mean_discharge_from_runoff(
    flow_direction: np.ndarray,
    processing_order: np.ndarray,
    precipitation_mm_year: np.ndarray,
    *,
    resolution_m: float,
    runoff_ratio: float | np.ndarray = 0.65,
    initial_discharge_m3s: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate long-term runoff and return mean discharge in m3/s."""

    precipitation = np.asarray(precipitation_mm_year, dtype=np.float64)
    ratio = np.asarray(runoff_ratio, dtype=np.float64)
    if precipitation.shape != np.asarray(flow_direction).shape:
        raise ValueError("precipitation shape does not match flow_direction")
    if ratio.ndim and ratio.shape != precipitation.shape:
        raise ValueError("spatial runoff_ratio must match precipitation shape")
    if np.any(~np.isfinite(precipitation)) or np.any(precipitation < 0):
        raise ValueError("precipitation must be finite and non-negative")
    if np.any(~np.isfinite(ratio)) or np.any((ratio < 0) | (ratio > 1)):
        raise ValueError("runoff_ratio must lie between zero and one")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")

    local_discharge = (
        precipitation
        / 1000.0
        * ratio
        * float(resolution_m) ** 2
        / SECONDS_PER_YEAR
    )
    if initial_discharge_m3s is not None:
        initial = np.asarray(initial_discharge_m3s, dtype=np.float64)
        if initial.shape != precipitation.shape:
            raise ValueError("initial_discharge_m3s must match precipitation shape")
        if np.any(~np.isfinite(initial)) or np.any(initial < 0):
            raise ValueError("initial_discharge_m3s must be finite and non-negative")
        local_discharge = local_discharge + initial
    return accumulate_values_d8(
        flow_direction,
        processing_order,
        local_discharge,
    ).astype(np.float32)
