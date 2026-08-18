"""Compare a generated hierarchical plan with reproducible FOEN targets."""

from __future__ import annotations

import json
from pathlib import Path

import click
import h5py
import numpy as np

from .calibration import (
    _distribution,
    channel_longitudinal_slopes,
    channel_slope_by_elevation_band,
    fit_channel_area_threshold_bisection,
    save_calibration_targets,
)
from .world_plan import D8_DIRECTION_OFFSETS


def calibrate_hierarchical_plan_against_foen(
    plan_file: str | Path,
    target_file: str | Path,
    swiss_calibration_file: str | Path,
) -> dict:
    """Fit generated channel density and compare climate/discharge diagnostics."""

    with Path(target_file).open("r", encoding="utf-8") as handle:
        targets = json.load(handle)
    with Path(swiss_calibration_file).open("r", encoding="utf-8") as handle:
        swiss = json.load(handle)
    with h5py.File(plan_file, "r") as plan:
        resolution_m = float(plan.attrs["resolution_m"])
        flow = plan["flow_direction"][...]
        accumulation = plan["accumulation_area_m2"][...]
        land = plan["land_mask"][...] == 1
        elevation = plan["elevation_breached_m"][...]
        precipitation = plan["annual_precipitation_mm"][...]
        discharge = plan["mean_discharge_m3s"][...]
        stream_order = plan["stream_order"][...]
        lake = plan["lake_id"][...] != np.iinfo(np.uint32).max
        source_report = json.loads(plan.attrs["report_json"])
    target_density = float(
        targets["river_network"]["mapped_drainage_density_km_per_km2"]
    )
    threshold_fit = fit_channel_area_threshold_bisection(
        flow,
        accumulation,
        land,
        resolution_m=resolution_m,
        target_density_km_per_km2=target_density,
        lower_km2=4.0,
        upper_km2=32.0,
    )
    threshold = float(threshold_fit["best"]["minimum_area_km2"])
    channels = (
        land & (flow >= 1) & (flow <= 8)
        & (accumulation >= threshold * 1_000_000.0)
    )
    network = _channel_statistics(
        flow,
        accumulation,
        elevation,
        discharge,
        stream_order,
        channels,
        land,
        resolution_m,
    )
    generated_precipitation = precipitation[land & np.isfinite(precipitation)]
    swiss_precipitation = swiss.get("annual_precipitation_mm")
    swiss_precipitation_median = (
        None
        if not swiss_precipitation
        else swiss_precipitation["quantiles"]["p50"]
    )
    generated_precipitation_median = float(np.median(generated_precipitation))
    precipitation_factor = (
        None
        if swiss_precipitation_median is None
        else float(swiss_precipitation_median / generated_precipitation_median)
    )
    fitted_runoff_ratio = float(
        swiss["precipitation_aware_runoff_fit"]["fitted_runoff_ratio"]
    )
    comparable_maximum_area = float(
        swiss["discharge_area_fit"]["drainage_area_km2"]["maximum"]
    )
    comparable = channels & (
        accumulation <= comparable_maximum_area * 1_000_000.0
    )
    return {
        "schema_version": 1,
        "sources": {
            "plan": str(plan_file),
            "foen_targets": str(target_file),
            "swiss_calibration": str(swiss_calibration_file),
        },
        "channel_initiation_fit": threshold_fit,
        "generated_channel_network": network,
        "generated_climate": {
            "annual_precipitation_mm": _distribution(generated_precipitation),
            "median_ratio_swiss_to_generated": precipitation_factor,
        },
        "comparable_generated_channel_discharge_m3s": _distribution(
            discharge[comparable]
        ),
        "foen_segment_discharge_m3s": targets["river_network"][
            "mean_annual_discharge_m3s"
        ],
        "swiss_precipitation_aware_runoff_fit": swiss[
            "precipitation_aware_runoff_fit"
        ],
        "swiss_channel_morphology": swiss["channel_morphology"],
        "generated_lake_fraction_of_land": float(lake.sum() / land.sum()),
        "planner_conditioning": {
            key: source_report[key]
            for key in (
                "breach_fraction_of_land",
                "fill_fraction_above_10m",
                "fill_fraction_above_50m",
                "fill_fraction_above_100m",
                "maximum_total_incision_m",
            )
        },
        "recommendations": {
            "channel_minimum_area_km2": threshold,
            "runoff_ratio": fitted_runoff_ratio,
            "precipitation_rescaling_is_not_automatic": True,
            "reason": (
                "Runoff ratio is physically calibrated independently of the learned "
                "climate distribution; climate bias must be corrected at the macro "
                "conditioning stage rather than hidden in runoff."
            ),
        },
    }


def _channel_statistics(
    flow: np.ndarray,
    accumulation: np.ndarray,
    elevation: np.ndarray,
    discharge: np.ndarray,
    stream_order: np.ndarray,
    channels: np.ndarray,
    land: np.ndarray,
    resolution_m: float,
) -> dict:
    lengths: list[np.ndarray] = []
    orders: list[np.ndarray] = []
    for code, (delta_row, delta_col) in D8_DIRECTION_OFFSETS.items():
        rows, cols = np.nonzero(channels & (flow == code))
        target_rows, target_cols = rows + delta_row, cols + delta_col
        continues = channels[target_rows, target_cols]
        rows, cols = rows[continues], cols[continues]
        target_rows, target_cols = target_rows[continues], target_cols[continues]
        distance = resolution_m * (np.sqrt(2.0) if code in (2, 4, 6, 8) else 1.0)
        lengths.append(np.full(rows.size, distance / 1000.0, dtype=np.float32))
        orders.append(stream_order[rows, cols])
    length = np.concatenate(lengths) if lengths else np.empty(0)
    slope = channel_longitudinal_slopes(
        flow, elevation, channels, resolution_m=resolution_m
    )
    order = np.concatenate(orders) if orders else np.empty(0, dtype=np.uint8)
    order_length = {
        str(value): float(length[order == value].sum())
        for value in np.unique(order)
    }
    area_km2 = accumulation[channels] / 1_000_000.0
    specific_runoff = discharge[channels] / area_km2 * 31_557.6
    return {
        "channel_cell_count": int(channels.sum()),
        "total_length_km": float(length.sum()),
        "drainage_density_km_per_km2": float(
            length.sum()
            / (np.count_nonzero(land) * resolution_m**2 / 1_000_000.0)
        ),
        "longitudinal_slope": _distribution(slope),
        "slope_by_source_elevation_m": channel_slope_by_elevation_band(
            flow, elevation, channels, resolution_m=resolution_m
        ),
        "upstream_area_km2": _distribution(area_km2),
        "mean_discharge_m3s": _distribution(discharge[channels]),
        "specific_runoff_mm_year": _distribution(specific_runoff),
        "stream_order_length_km": order_length,
    }


@click.command("calibrate-hierarchical-plan-foen")
@click.argument("plan_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--targets", default="samples/hydrology/foen_calibration_targets.json",
    show_default=True, type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--swiss-calibration",
    default="samples/hydrology/swiss_dem_240m_calibration_v2.json",
    show_default=True, type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output", default="samples/hydrology/foen_hierarchical_plan_calibration.json",
    show_default=True, type=click.Path(dir_okay=False),
)
def calibrate_hierarchical_plan_foen(plan_file, targets, swiss_calibration, output):
    """Calibrate one generated hierarchical plan against FOEN observations."""

    report = calibrate_hierarchical_plan_against_foen(
        plan_file, targets, swiss_calibration
    )
    save_calibration_targets(report, output)
    click.echo(f"Saved generated-plan FOEN calibration to {output}")
