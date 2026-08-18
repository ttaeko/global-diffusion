"""Extract reproducible hydrology targets from the supplied FOEN datasets."""

from __future__ import annotations

from collections import Counter
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

import click
import numpy as np
import shapefile

from .compiled_routing import accumulate_values_d8, priority_flood_route_compiled
from .runoff import SECONDS_PER_YEAR
from .world_plan import D8_DIRECTION_OFFSETS


_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def extract_foen_calibration_targets(
    catchment_archive: str | Path,
    mq_shapefile: str | Path,
) -> dict:
    """Summarize basin hierarchy, mapped rivers, discharge, and seasonality."""

    archive = Path(catchment_archive)
    mq_path = Path(mq_shapefile)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not mq_path.is_file():
        raise FileNotFoundError(mq_path)
    if shutil.which("ogr2ogr") is None:
        raise RuntimeError("ogr2ogr (GDAL) is required to read the Deflate64 FOEN archive")

    detailed_rows = _read_ogr_csv(
        archive,
        "EZG_Gewaesser.gpkg",
        """
        SELECT TEILEZGNR, TEZGNR40, TEZGNR150, TEZGNR1000,
               Nebenarm, InterneSenke, SEE, FLUSSGB,
               ST_Area(Shape) / 1000000.0 AS area_km2
        FROM Teileinzugsgebiet
        """,
    )
    aggregate_rows = _read_ogr_csv(
        archive,
        "EZG_ebene_40km.gpkg",
        "SELECT tezgnr40, teilezgflaeche AS area_km2 FROM ebene_40km",
    )

    detailed_areas = _positive_floats(row["area_km2"] for row in detailed_rows)
    aggregate_areas = _positive_floats(row["area_km2"] for row in aggregate_rows)
    lake_types = Counter(row["SEE"] or "unknown" for row in detailed_rows)
    major_basins = Counter(row["FLUSSGB"] or "unknown" for row in detailed_rows)
    side_arms = sum(_as_int(row["Nebenarm"]) != 0 for row in detailed_rows)
    internal_sinks = sum(_as_float(row["InterneSenke"]) != 0 for row in detailed_rows)

    mq = _read_mq_statistics(mq_path)
    reference_area_km2 = float(np.sum(aggregate_areas))
    mq["mapped_drainage_density_km_per_km2"] = (
        mq["total_segment_length_km"] / reference_area_km2
    )

    return {
        "schema_version": 2,
        "sources": {
            "catchment_archive": str(archive),
            "mq_shapefile": str(mq_path),
        },
        "catchments": {
            "detailed_count": len(detailed_rows),
            "detailed_area_km2": _distribution(detailed_areas),
            "level_40km_count": len(aggregate_rows),
            "level_40km_area_km2": _distribution(aggregate_areas),
            "reference_partition_area_km2": reference_area_km2,
            "level_40km_unique_ids": len(
                {row["TEZGNR40"] for row in detailed_rows if row["TEZGNR40"]}
            ),
            "level_150km_unique_ids": len(
                {row["TEZGNR150"] for row in detailed_rows if row["TEZGNR150"]}
            ),
            "level_1000km_unique_ids": len(
                {row["TEZGNR1000"] for row in detailed_rows if row["TEZGNR1000"]}
            ),
            "level_40km_grouped_area_km2": _grouped_area_distribution(
                detailed_rows, "TEZGNR40"
            ),
            "level_150km_grouped_area_km2": _grouped_area_distribution(
                detailed_rows, "TEZGNR150"
            ),
            "level_1000km_grouped_area_km2": _grouped_area_distribution(
                detailed_rows, "TEZGNR1000"
            ),
            "side_arm_count": side_arms,
            "internal_sink_count": internal_sinks,
            "lake_type_counts": dict(sorted(lake_types.items())),
            "major_basin_counts": dict(sorted(major_basins.items())),
        },
        "river_network": mq,
    }


def save_calibration_targets(report: dict, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def calibrate_swiss_dem(
    dem_path: str | Path,
    catchment_archive: str | Path,
    target_report: dict,
    *,
    resolution_m: float = 240.0,
    precipitation_mean_path: str | Path | None = None,
    precipitation_to_annual_multiplier: float = 12.0,
) -> dict:
    """Fit channel initiation area on Swiss terrain to the FOEN map density."""

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import rasterize

    dem_path = Path(dem_path)
    archive = Path(catchment_archive)
    with rasterio.open(dem_path) as source:
        width = int(np.ceil(source.width * source.res[0] / resolution_m))
        height = int(np.ceil(source.height * abs(source.res[1]) / resolution_m))
        elevation = source.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.average,
        ).astype(np.float32)
        transform = source.transform * source.transform.scale(
            source.width / width, source.height / height
        )
        nodata = source.nodata
    valid = np.isfinite(elevation)
    if nodata is not None:
        valid &= elevation < float(nodata) * 0.5
    elevation[~valid] = np.nan

    routing = priority_flood_route_compiled(
        elevation,
        resolution_m=resolution_m,
        land_mask=valid,
    )
    swiss_geometries = _read_simplified_foen_geometries(archive, resolution_m / 2)
    swiss_mask = rasterize(
        ((geometry, 1) for geometry in swiss_geometries),
        out_shape=elevation.shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)
    swiss_mask &= valid

    target_density = float(
        target_report["river_network"]["mapped_drainage_density_km_per_km2"]
    )
    fitted = fit_channel_area_threshold(
        routing.flow_direction,
        routing.accumulation_area_m2,
        swiss_mask,
        resolution_m=resolution_m,
        target_density_km_per_km2=target_density,
    )
    channel_threshold = float(fitted["best"]["minimum_area_km2"])
    reference_channels = (
        swiss_mask & (routing.flow_direction >= 1) & (routing.flow_direction <= 8)
        & (routing.accumulation_area_m2 >= channel_threshold * 1_000_000.0)
    )
    correction = routing.elevation_correction_m[swiss_mask]
    mq_path = Path(target_report["sources"]["mq_shapefile"])
    discharge_fit = _fit_mq_discharge_to_accumulation(
        mq_path,
        routing.accumulation_area_m2,
        transform,
        resolution_m=resolution_m,
    )
    runoff_fit = None
    if precipitation_mean_path is not None:
        annual_precipitation = _read_precipitation_on_grid(
            precipitation_mean_path,
            elevation.shape,
            transform,
            destination_crs="EPSG:2056",
            annual_multiplier=precipitation_to_annual_multiplier,
        )
        # The Swiss climate raster covers the national extent, while several
        # routed headwaters begin just outside it. Nearest extrapolation avoids
        # treating those upstream cells as zero-rainfall desert.
        annual_precipitation = _fill_nearest_finite(annual_precipitation)
        annual_precipitation[~valid] = 0
        unit_runoff = (
            annual_precipitation.astype(np.float64)
            / 1000.0
            * resolution_m**2
            / SECONDS_PER_YEAR
        )
        potential_discharge = accumulate_values_d8(
            routing.flow_direction,
            routing.processing_order,
            unit_runoff,
        )
        runoff_fit = _fit_mq_runoff_ratio(
            mq_path,
            routing.accumulation_area_m2,
            potential_discharge,
            transform,
            resolution_m=resolution_m,
        )
    return {
        "schema_version": 1,
        "dem": str(dem_path),
        "resolution_m": float(resolution_m),
        "raster_shape": list(elevation.shape),
        "swiss_analysis_cells": int(np.count_nonzero(swiss_mask)),
        "swiss_analysis_area_km2": float(
            np.count_nonzero(swiss_mask) * resolution_m**2 / 1_000_000.0
        ),
        "target_drainage_density_km_per_km2": target_density,
        "annual_precipitation_mm": (
            None
            if precipitation_mean_path is None
            else _distribution(annual_precipitation[swiss_mask])
        ),
        "channel_initiation_fit": fitted,
        "channel_morphology": {
            "channel_cell_count": int(np.count_nonzero(reference_channels)),
            "longitudinal_slope": _distribution(
                channel_longitudinal_slopes(
                    routing.flow_direction,
                    elevation,
                    reference_channels,
                    resolution_m=resolution_m,
                )
            ),
            "slope_by_source_elevation_m": channel_slope_by_elevation_band(
                routing.flow_direction,
                elevation,
                reference_channels,
                resolution_m=resolution_m,
            ),
        },
        "depression_conditioning": {
            "corrected_cell_fraction": float(np.mean(correction > 0)),
            "correction_m": _distribution(correction),
        },
        "discharge_area_fit": discharge_fit,
        "precipitation_aware_runoff_fit": runoff_fit,
    }


def channel_longitudinal_slopes(
    flow_direction: np.ndarray,
    elevation_m: np.ndarray,
    channel_mask: np.ndarray,
    *,
    resolution_m: float,
) -> np.ndarray:
    """Return non-negative physical bed gradients for connected channel links."""

    flow = np.asarray(flow_direction, dtype=np.uint8)
    elevation = np.asarray(elevation_m, dtype=np.float32)
    channels = np.asarray(channel_mask, dtype=bool)
    if not (flow.shape == elevation.shape == channels.shape):
        raise ValueError("Channel slope rasters must align")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    results: list[np.ndarray] = []
    for code, (delta_row, delta_col) in D8_DIRECTION_OFFSETS.items():
        rows, cols = np.nonzero(channels & (flow == code))
        target_rows, target_cols = rows + delta_row, cols + delta_col
        continues = channels[target_rows, target_cols]
        rows, cols = rows[continues], cols[continues]
        target_rows, target_cols = target_rows[continues], target_cols[continues]
        distance = resolution_m * (
            np.sqrt(2.0) if code in (2, 4, 6, 8) else 1.0
        )
        results.append(
            np.maximum(
                0.0,
                (elevation[rows, cols] - elevation[target_rows, target_cols])
                / distance,
            )
        )
    return np.concatenate(results) if results else np.empty(0, dtype=np.float32)


def channel_slope_by_elevation_band(
    flow_direction: np.ndarray,
    elevation_m: np.ndarray,
    channel_mask: np.ndarray,
    *,
    resolution_m: float,
    band_edges_m: tuple[float, ...] = (0.0, 500.0, 1000.0, 2000.0, np.inf),
) -> dict:
    """Summarize channel grades within comparable source-elevation bands."""

    flow = np.asarray(flow_direction, dtype=np.uint8)
    elevation = np.asarray(elevation_m, dtype=np.float32)
    channels = np.asarray(channel_mask, dtype=bool)
    if len(band_edges_m) < 2 or any(
        right <= left for left, right in zip(band_edges_m, band_edges_m[1:])
    ):
        raise ValueError("Elevation band edges must be strictly increasing")
    slopes: list[np.ndarray] = []
    source_elevations: list[np.ndarray] = []
    for code, (delta_row, delta_col) in D8_DIRECTION_OFFSETS.items():
        rows, cols = np.nonzero(channels & (flow == code))
        target_rows, target_cols = rows + delta_row, cols + delta_col
        continues = channels[target_rows, target_cols]
        rows, cols = rows[continues], cols[continues]
        target_rows, target_cols = target_rows[continues], target_cols[continues]
        distance = resolution_m * (
            np.sqrt(2.0) if code in (2, 4, 6, 8) else 1.0
        )
        slopes.append(
            np.maximum(
                0.0,
                (elevation[rows, cols] - elevation[target_rows, target_cols])
                / distance,
            )
        )
        source_elevations.append(elevation[rows, cols])
    slope = np.concatenate(slopes) if slopes else np.empty(0, dtype=np.float32)
    source = (
        np.concatenate(source_elevations)
        if source_elevations else np.empty(0, dtype=np.float32)
    )
    result = {}
    for lower, upper in zip(band_edges_m, band_edges_m[1:]):
        selected = (source >= lower) & (source < upper)
        label = f"{int(lower)}_to_{'inf' if np.isinf(upper) else int(upper)}"
        result[label] = _distribution(slope[selected])
    return result


def _read_precipitation_on_grid(
    path: str | Path,
    shape: tuple[int, int],
    transform,
    *,
    destination_crs,
    annual_multiplier: float,
) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if annual_multiplier <= 0:
        raise ValueError("annual precipitation multiplier must be positive")
    destination = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    destination *= float(annual_multiplier)
    destination[(destination <= 0) | ~np.isfinite(destination)] = np.nan
    return destination


def _fill_nearest_finite(values: np.ndarray) -> np.ndarray:
    import scipy.ndimage

    source = np.asarray(values, dtype=np.float32)
    missing = ~np.isfinite(source)
    if np.all(missing):
        raise ValueError("Precipitation raster contains no finite values")
    if not np.any(missing):
        return source.copy()
    indices = scipy.ndimage.distance_transform_edt(
        missing, return_distances=False, return_indices=True
    )
    return source[tuple(indices)].astype(np.float32)


def channel_density_km_per_km2(
    flow_direction: np.ndarray,
    accumulation_area_m2: np.ndarray,
    land_mask: np.ndarray,
    *,
    resolution_m: float,
    minimum_area_km2: float,
) -> float:
    """Measure raster-channel length per land area for threshold fitting."""

    flow = np.asarray(flow_direction, dtype=np.uint8)
    accumulation = np.asarray(accumulation_area_m2, dtype=np.float64)
    land = np.asarray(land_mask, dtype=bool)
    if not (flow.shape == accumulation.shape == land.shape):
        raise ValueError("Routing arrays must have the same shape")
    if resolution_m <= 0 or minimum_area_km2 < 0:
        raise ValueError("resolution_m must be positive and threshold non-negative")
    selected = land & (flow >= 1) & (flow <= 8)
    selected &= accumulation >= minimum_area_km2 * 1_000_000.0
    diagonal = np.isin(flow, (2, 4, 6, 8))
    channel_length_km = float(
        np.sum(np.where(diagonal[selected], np.sqrt(2.0), 1.0))
        * resolution_m
        / 1000.0
    )
    land_area_km2 = float(np.count_nonzero(land) * resolution_m**2 / 1_000_000.0)
    return channel_length_km / land_area_km2 if land_area_km2 else 0.0


def fit_channel_area_threshold(
    flow_direction: np.ndarray,
    accumulation_area_m2: np.ndarray,
    land_mask: np.ndarray,
    *,
    resolution_m: float,
    target_density_km_per_km2: float,
    candidates_km2: Iterable[float] | None = None,
) -> dict:
    """Choose the contributing-area threshold closest to observed density."""

    if target_density_km_per_km2 < 0:
        raise ValueError("target drainage density must be non-negative")
    if candidates_km2 is None:
        candidates = np.geomspace(0.25, 1024.0, 49)
    else:
        candidates = np.asarray(tuple(candidates_km2), dtype=np.float64)
    if candidates.size == 0 or np.any(candidates < 0):
        raise ValueError("candidate thresholds must be non-empty and non-negative")
    trials = []
    for threshold in candidates:
        density = channel_density_km_per_km2(
            flow_direction,
            accumulation_area_m2,
            land_mask,
            resolution_m=resolution_m,
            minimum_area_km2=float(threshold),
        )
        trials.append(
            {
                "minimum_area_km2": float(threshold),
                "density_km_per_km2": density,
                "absolute_error": abs(density - target_density_km_per_km2),
            }
        )
    best = min(trials, key=lambda trial: trial["absolute_error"])
    return {"best": best, "trials": trials}


def fit_channel_area_threshold_bisection(
    flow_direction: np.ndarray,
    accumulation_area_m2: np.ndarray,
    land_mask: np.ndarray,
    *,
    resolution_m: float,
    target_density_km_per_km2: float,
    lower_km2: float = 0.25,
    upper_km2: float = 1024.0,
    iterations: int = 24,
) -> dict:
    """Refine a monotonic channel-density threshold without a dense sweep."""

    if lower_km2 < 0 or upper_km2 <= lower_km2 or iterations <= 0:
        raise ValueError("Invalid bisection threshold range or iteration count")
    trials: list[dict] = []
    lower, upper = float(lower_km2), float(upper_km2)
    for _ in range(iterations):
        threshold = (lower + upper) / 2.0
        density = channel_density_km_per_km2(
            flow_direction,
            accumulation_area_m2,
            land_mask,
            resolution_m=resolution_m,
            minimum_area_km2=threshold,
        )
        trials.append(
            {
                "minimum_area_km2": threshold,
                "density_km_per_km2": density,
                "absolute_error": abs(density - target_density_km_per_km2),
            }
        )
        if density > target_density_km_per_km2:
            lower = threshold
        else:
            upper = threshold
    for threshold in (lower, upper):
        density = channel_density_km_per_km2(
            flow_direction,
            accumulation_area_m2,
            land_mask,
            resolution_m=resolution_m,
            minimum_area_km2=threshold,
        )
        trials.append(
            {
                "minimum_area_km2": threshold,
                "density_km_per_km2": density,
                "absolute_error": abs(density - target_density_km_per_km2),
            }
        )
    return {
        "best": min(trials, key=lambda trial: trial["absolute_error"]),
        "final_bracket_km2": [lower, upper],
        "trials": trials,
    }


def _read_mq_statistics(path: Path) -> dict:
    reader = shapefile.Reader(str(path))
    field_names = [field[0] for field in reader.fields[1:]]
    records = [dict(zip(field_names, record)) for record in reader.iterRecords()]
    annual = _positive_floats(record["MQN_JAHR"] for record in records)
    lengths_km = _positive_floats(record["Shape_Leng"] for record in records) / 1000.0
    regimes = Counter(
        str(record["REGIMETYP"]).strip() or "unknown" for record in records
    )
    month_fields = (
        "MQN_JAN", "MQN_FEB", "MQN_MAR", "MQN_APR", "MQN_MAI", "MQN_JUN",
        "MQN_JUL", "MQN_AUG", "MQN_SEP", "MQN_OKT", "MQN_NOV", "MQN_DEZ",
    )
    monthly = np.asarray(
        [[_as_float(record[field]) for field in month_fields] for record in records],
        dtype=np.float64,
    )
    monthly[monthly <= 0] = np.nan
    monthly_median = np.nanmedian(monthly, axis=0)
    reader.close()
    return {
        "segment_count": len(records),
        "positive_annual_discharge_count": int(annual.size),
        "mean_annual_discharge_m3s": _distribution(annual),
        "segment_length_km": _distribution(lengths_km),
        "total_segment_length_km": float(np.sum(lengths_km)),
        "regime_type_counts": dict(sorted(regimes.items())),
        "monthly_segment_median_discharge_m3s": {
            month: float(value)
            for month, value in zip(
                ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
                monthly_median,
            )
        },
    }


def _fit_mq_discharge_to_accumulation(
    path: Path,
    accumulation_area_m2: np.ndarray,
    transform,
    *,
    resolution_m: float,
    search_radius_m: float = 720.0,
) -> dict:
    """Match MQ segments to nearby DEM flow and fit Q = coefficient * area^p."""

    reader = shapefile.Reader(str(path))
    field_names = [field[0] for field in reader.fields[1:]]
    annual_index = field_names.index("MQN_JAHR")
    inverse = ~transform
    radius = max(0, int(np.ceil(search_radius_m / resolution_m)))
    rows, cols = accumulation_area_m2.shape
    matched_area: list[float] = []
    matched_discharge: list[float] = []
    for shape_record in reader.iterShapeRecords():
        discharge = _as_float(shape_record.record[annual_index])
        points = shape_record.shape.points
        if discharge <= 0 or not points:
            continue
        candidates = (points[0], points[len(points) // 2], points[-1])
        maximum_area = 0.0
        for x_lv03, y_lv03 in candidates:
            col_float, row_float = inverse * (
                float(x_lv03) + 2_000_000.0,
                float(y_lv03) + 1_000_000.0,
            )
            row = int(np.floor(row_float))
            col = int(np.floor(col_float))
            row0, row1 = max(0, row - radius), min(rows, row + radius + 1)
            col0, col1 = max(0, col - radius), min(cols, col + radius + 1)
            if row0 >= row1 or col0 >= col1:
                continue
            local = float(np.nanmax(accumulation_area_m2[row0:row1, col0:col1]))
            maximum_area = max(maximum_area, local)
        if maximum_area > 0:
            matched_area.append(maximum_area / 1_000_000.0)
            matched_discharge.append(discharge)
    reader.close()

    area = np.asarray(matched_area, dtype=np.float64)
    discharge = np.asarray(matched_discharge, dtype=np.float64)
    usable = (area > 0) & (discharge > 0)
    area, discharge = area[usable], discharge[usable]
    if area.size < 2:
        raise RuntimeError("Too few MQ segments matched the routed DEM")
    log_area = np.log(area)
    log_discharge = np.log(discharge)
    exponent, log_coefficient = np.polyfit(log_area, log_discharge, 1)
    prediction = log_coefficient + exponent * log_area
    residual = log_discharge - prediction
    total = np.sum((log_discharge - np.mean(log_discharge)) ** 2)
    r_squared = 1.0 - float(np.sum(residual**2) / total) if total > 0 else 0.0
    specific_runoff_mm_year = discharge / area * 31_557.6
    return {
        "matched_segment_count": int(area.size),
        "search_radius_m": float(search_radius_m),
        "drainage_area_km2": _distribution(area),
        "specific_runoff_mm_year": _distribution(specific_runoff_mm_year),
        "power_law": {
            "formula": "mean_discharge_m3s = coefficient * area_km2 ** exponent",
            "coefficient": float(np.exp(log_coefficient)),
            "exponent": float(exponent),
            "log_space_r_squared": r_squared,
            "log_space_rmse": float(np.sqrt(np.mean(residual**2))),
        },
    }


def _fit_mq_runoff_ratio(
    path: Path,
    accumulation_area_m2: np.ndarray,
    potential_discharge_m3s: np.ndarray,
    transform,
    *,
    resolution_m: float,
    search_radius_m: float = 720.0,
) -> dict:
    """Fit Q = runoff_ratio * accumulated precipitation volume."""

    reader = shapefile.Reader(str(path))
    field_names = [field[0] for field in reader.fields[1:]]
    annual_index = field_names.index("MQN_JAHR")
    inverse = ~transform
    radius = max(0, int(np.ceil(search_radius_m / resolution_m)))
    rows, cols = accumulation_area_m2.shape
    observed: list[float] = []
    potential: list[float] = []
    for shape_record in reader.iterShapeRecords():
        discharge = _as_float(shape_record.record[annual_index])
        points = shape_record.shape.points
        if discharge <= 0 or not points:
            continue
        best_area = 0.0
        best_potential = 0.0
        for x_lv03, y_lv03 in (points[0], points[len(points) // 2], points[-1]):
            col_float, row_float = inverse * (
                float(x_lv03) + 2_000_000.0,
                float(y_lv03) + 1_000_000.0,
            )
            row, col = int(np.floor(row_float)), int(np.floor(col_float))
            row0, row1 = max(0, row - radius), min(rows, row + radius + 1)
            col0, col1 = max(0, col - radius), min(cols, col + radius + 1)
            if row0 >= row1 or col0 >= col1:
                continue
            local_area = accumulation_area_m2[row0:row1, col0:col1]
            flat = int(np.nanargmax(local_area))
            local_maximum = float(local_area.flat[flat])
            if local_maximum > best_area:
                best_area = local_maximum
                best_potential = float(
                    potential_discharge_m3s[row0:row1, col0:col1].flat[flat]
                )
        if best_potential > 0:
            observed.append(discharge)
            potential.append(best_potential)
    reader.close()
    observed_array = np.asarray(observed, dtype=np.float64)
    potential_array = np.asarray(potential, dtype=np.float64)
    ratios = observed_array / potential_array
    usable = np.isfinite(ratios) & (ratios >= 0.02) & (ratios <= 1.5)
    if np.count_nonzero(usable) < 2:
        raise RuntimeError("Too few MQ segments yielded plausible runoff ratios")
    log_ratios = np.log(ratios[usable])
    fitted = float(np.exp(np.median(log_ratios)))
    prediction = potential_array[usable] * fitted
    residual = np.log(observed_array[usable]) - np.log(prediction)
    return {
        "matched_segment_count": int(ratios.size),
        "plausible_segment_count": int(np.count_nonzero(usable)),
        "plausible_ratio_range": [0.02, 1.5],
        "fitted_runoff_ratio": fitted,
        "segment_runoff_ratio": _distribution(ratios[usable]),
        "log_space_rmse": float(np.sqrt(np.mean(residual**2))),
        "precipitation_is_accumulated_over_routed_upstream_area": True,
    }


def _read_ogr_csv(archive: Path, member: str, sql: str) -> list[dict[str, str]]:
    data_source = f"/vsizip/{archive.resolve()}/{member}"
    command = (
        "ogr2ogr", "-f", "CSV", "/vsistdout/", data_source,
        "-dialect", "sqlite", "-sql", " ".join(sql.split()),
    )
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return list(csv.DictReader(io.StringIO(completed.stdout)))


def _read_simplified_foen_geometries(
    archive: Path,
    tolerance_m: float,
) -> list[dict]:
    data_source = f"/vsizip/{archive.resolve()}/EZG_ebene_40km.gpkg"
    completed = subprocess.run(
        (
            "ogr2ogr", "-f", "GeoJSON", "/vsistdout/", data_source,
            "ebene_40km", "-simplify", str(float(tolerance_m)),
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    collection = json.loads(completed.stdout)
    geometries = [
        feature["geometry"]
        for feature in collection.get("features", [])
        if feature.get("geometry") is not None
    ]
    if not geometries:
        raise RuntimeError("FOEN aggregate catchment layer contained no geometries")
    return geometries


def _positive_floats(values: Iterable[object]) -> np.ndarray:
    result = np.asarray([_as_float(value) for value in values], dtype=np.float64)
    return result[np.isfinite(result) & (result > 0)]


def _grouped_area_distribution(rows: list[dict[str, str]], field: str) -> dict:
    totals: dict[str, float] = {}
    for row in rows:
        identifier = row.get(field, "")
        area = _as_float(row.get("area_km2"))
        if identifier and area > 0:
            totals[identifier] = totals.get(identifier, 0.0) + area
    return _distribution(np.asarray(list(totals.values()), dtype=np.float64))


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    quantiles = np.quantile(values, _QUANTILES)
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
        "quantiles": {
            f"p{int(probability * 100):02d}": float(value)
            for probability, value in zip(_QUANTILES, quantiles)
        },
    }


@click.command("extract-hydrology-targets")
@click.option(
    "--catchment-archive",
    default="data/alps/hydrology/wasser-einzugsgebietsgliederung_2056.gpkg.zip",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--mq-shapefile",
    default="data/alps/hydrology/MQ-GWN-CH/Datensatz/MittlererAbfluss_Regimetyp.shp",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output",
    default="samples/hydrology/foen_calibration_targets.json",
    show_default=True,
    type=click.Path(dir_okay=False),
)
def extract_hydrology_targets(catchment_archive, mq_shapefile, output):
    """Extract reusable calibration targets from the local FOEN packages."""

    report = extract_foen_calibration_targets(catchment_archive, mq_shapefile)
    save_calibration_targets(report, output)
    click.echo(f"Saved FOEN calibration targets to {output}")


@click.command("calibrate-swiss-dem-hydrology")
@click.option(
    "--dem",
    default="data/alps/alti3d.tif",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--catchment-archive",
    default="data/alps/hydrology/wasser-einzugsgebietsgliederung_2056.gpkg.zip",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--targets",
    default="samples/hydrology/foen_calibration_targets.json",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--resolution-m", default=240.0, show_default=True, type=float)
@click.option(
    "--precipitation-mean",
    default="data/alps/climate/precip_mean_lv95.tif",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--precipitation-to-annual-multiplier", default=12.0, show_default=True, type=float
)
@click.option(
    "--output",
    default="samples/hydrology/swiss_dem_240m_calibration_v2.json",
    show_default=True,
    type=click.Path(dir_okay=False),
)
def calibrate_swiss_dem_hydrology(
    dem, catchment_archive, targets, resolution_m, precipitation_mean,
    precipitation_to_annual_multiplier, output
):
    """Fit raster channel initiation to the supplied FOEN observations."""

    with Path(targets).open("r", encoding="utf-8") as handle:
        target_report = json.load(handle)
    report = calibrate_swiss_dem(
        dem,
        catchment_archive,
        target_report,
        resolution_m=resolution_m,
        precipitation_mean_path=precipitation_mean,
        precipitation_to_annual_multiplier=precipitation_to_annual_multiplier,
    )
    save_calibration_targets(report, output)
    click.echo(f"Saved Swiss DEM hydrology calibration to {output}")
