"""Rasterization contract for observed river centerlines and lake polygons."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import click
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import bounds as geometry_bounds
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_geom
from rasterio.windows import from_bounds as window_from_bounds


OBSERVED_HYDROLOGY_SCHEMA = "terrain-diffusion-observed-hydrology"
OBSERVED_HYDROLOGY_VERSION = 1
OBSERVED_HYDROLOGY_CHANNELS = (
    "river_centerline_mask",
    "river_mean_discharge_m3s",
    "river_width_m",
    "lake_mask",
    "lake_surface_elevation_m",
)


@dataclass(frozen=True)
class ObservedHydrologyRaster:
    values: np.ndarray
    valid: np.ndarray
    channel_names: tuple[str, ...] = OBSERVED_HYDROLOGY_CHANNELS

    def validate(self) -> None:
        if self.values.ndim != 3 or self.values.shape[0] != len(self.channel_names):
            raise ValueError("Observed hydrology must be a five-channel raster")
        if self.valid.shape != self.values.shape[1:]:
            raise ValueError("Observed hydrology validity mask does not align")


def rasterize_observed_hydrology(
    reference_dem: str | Path,
    output_file: str | Path,
    *,
    river_vector: str | Path | None = None,
    river_layer: str | None = None,
    river_discharge_field: str | None = None,
    river_width_field: str | None = None,
    lake_vector: str | Path | None = None,
    lake_layer: str | None = None,
    lake_level_field: str | None = None,
    coverage_vector: str | Path | None = None,
    coverage_layer: str | None = None,
    coverage_field: str | None = None,
    coverage_values: tuple[str, ...] = (),
    resolution_m: float = 30.0,
    crop_to_observations: bool = True,
) -> dict[str, Any]:
    """Create an aligned five-band supervision raster from observed vectors.

    GeoJSON and ESRI Shapefile sources work with the base dependencies. GPKG
    and virtual `/vsizip/` paths use Fiona when installed. Missing discharge,
    width, or lake-level attributes remain NaN rather than becoming false zero
    observations. Lake levels can be inferred from the median reference DEM.
    """

    if river_vector is None and lake_vector is None:
        raise ValueError("At least one observed river or lake source is required")
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    output = Path(output_file)
    if output.exists():
        raise FileExistsError(f"Observed hydrology raster exists: {output}")

    source_records: list[dict[str, Any]] = []
    river_features: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    lake_features: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    coverage_features: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

    with rasterio.open(reference_dem) as dem:
        if dem.crs is None or not dem.crs.is_projected:
            raise ValueError("Reference DEM must use a projected CRS")
        destination_crs = dem.crs
        if river_vector is not None:
            features, source_crs = _read_vector_features(river_vector, river_layer)
            river_features = [
                (transform_geom(source_crs, destination_crs, geometry), properties)
                for geometry, properties in features
            ]
            source_records.append(_source_record("rivers", river_vector, river_layer))
        if lake_vector is not None:
            features, source_crs = _read_vector_features(lake_vector, lake_layer)
            lake_features = [
                (transform_geom(source_crs, destination_crs, geometry), properties)
                for geometry, properties in features
            ]
            lake_features = _lake_polygon_features(lake_features)
            source_records.append(_source_record("lakes", lake_vector, lake_layer))
        if coverage_vector is not None:
            features, source_crs = _read_vector_features(
                coverage_vector, coverage_layer
            )
            if coverage_values and not coverage_field:
                raise ValueError("coverage_values require coverage_field")
            selected = [
                (geometry, properties)
                for geometry, properties in features
                if not coverage_values
                or str(properties.get(coverage_field)) in coverage_values
            ]
            if not selected:
                raise ValueError("Coverage filter selected no vector features")
            coverage_features = [
                (transform_geom(source_crs, destination_crs, geometry), properties)
                for geometry, properties in selected
            ]
            source_records.append(
                _source_record("coverage", coverage_vector, coverage_layer)
            )

        bounds = dem.bounds
        if crop_to_observations:
            bounds = _observation_bounds(
                [geometry for geometry, _ in river_features + lake_features],
                reference_bounds=dem.bounds,
                resolution_m=resolution_m,
            )
        width = max(1, int(np.ceil((bounds.right - bounds.left) / resolution_m)))
        height = max(1, int(np.ceil((bounds.top - bounds.bottom) / resolution_m)))
        transform = from_origin(bounds.left, bounds.top, resolution_m, resolution_m)
        dem_window = window_from_bounds(*bounds, transform=dem.transform)
        elevation = dem.read(
            1,
            window=dem_window,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.average,
        ).filled(np.nan).astype(np.float32)

    valid = np.isfinite(elevation)
    if coverage_features:
        coverage_mask = rasterize(
            ((geometry, 1) for geometry, _ in coverage_features),
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        valid &= coverage_mask > 0
        del coverage_mask
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(OBSERVED_HYDROLOGY_CHANNELS),
        "dtype": "float32",
        "crs": destination_crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": np.nan,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as target:
        missing = np.full((height, width), np.nan, dtype=np.float32)
        target.write(missing, 2)
        target.write(missing, 3)
        target.write(missing, 5)

        river_mask = (
            rasterize(
                ((geometry, 1.0) for geometry, _ in river_features),
                out_shape=(height, width),
                transform=transform,
                fill=0.0,
                all_touched=True,
                dtype="float32",
            )
            if river_features
            else np.zeros((height, width), dtype=np.float32)
        )
        river_cells = int(np.count_nonzero(river_mask > 0.5))
        target.write(river_mask, 1)
        del river_mask
        if river_discharge_field:
            target.write(
                _rasterize_numeric_property(
                    river_features,
                    river_discharge_field,
                    (height, width),
                    transform,
                ),
                2,
            )
        if river_width_field:
            target.write(
                _rasterize_numeric_property(
                    river_features,
                    river_width_field,
                    (height, width),
                    transform,
                ),
                3,
            )

        lake_labels = (
            rasterize(
                (
                    (geometry, index + 1)
                    for index, (geometry, _) in enumerate(lake_features)
                ),
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype="uint32",
            )
            if lake_features
            else np.zeros((height, width), dtype=np.uint32)
        )
        lake_cells = int(np.count_nonzero(lake_labels))
        target.write((lake_labels > 0).astype(np.float32), 4)
        # swissTLM3D includes complete cross-border lake polygons. Retain those
        # water observations even where the administrative coverage ends.
        valid |= lake_labels > 0
        if lake_features:
            target.write(
                _lake_surface_levels(
                    lake_labels,
                    elevation,
                    lake_features,
                    lake_level_field,
                ),
                5,
            )
        target.write_mask(valid.astype(np.uint8) * 255)
        target.descriptions = OBSERVED_HYDROLOGY_CHANNELS
        provenance = {
            "schema": OBSERVED_HYDROLOGY_SCHEMA,
            "version": OBSERVED_HYDROLOGY_VERSION,
            "resolution_m": float(resolution_m),
            "channel_names": list(OBSERVED_HYDROLOGY_CHANNELS),
            "sources": source_records,
            "reference_dem": str(Path(reference_dem).resolve()),
            "reference_dem_sha256": _sha256(Path(reference_dem)),
            "crop_to_observations": bool(crop_to_observations),
            "coverage_field": coverage_field,
            "coverage_values": list(coverage_values),
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "shape": [height, width],
            "river_cells": river_cells,
            "lake_cells": lake_cells,
        }
        target.update_tags(
            observed_hydrology_schema=OBSERVED_HYDROLOGY_SCHEMA,
            observed_hydrology_version=str(OBSERVED_HYDROLOGY_VERSION),
            provenance_json=json.dumps(provenance, sort_keys=True),
        )
    return provenance


def _observation_bounds(
    geometries: Iterable[Mapping[str, Any]],
    *,
    reference_bounds: Any,
    resolution_m: float,
) -> rasterio.coords.BoundingBox:
    bounds = [geometry_bounds(geometry) for geometry in geometries]
    if not bounds:
        raise ValueError("Observed sources contain no usable geometries")
    left = max(
        reference_bounds.left,
        np.floor(min(b[0] for b in bounds) / resolution_m) * resolution_m,
    )
    bottom = max(
        reference_bounds.bottom,
        np.floor(min(b[1] for b in bounds) / resolution_m) * resolution_m,
    )
    right = min(
        reference_bounds.right,
        np.ceil(max(b[2] for b in bounds) / resolution_m) * resolution_m,
    )
    top = min(
        reference_bounds.top,
        np.ceil(max(b[3] for b in bounds) / resolution_m) * resolution_m,
    )
    if right <= left or top <= bottom:
        raise ValueError("Observed sources do not overlap the reference DEM")
    return rasterio.coords.BoundingBox(left, bottom, right, top)


def _lake_surface_levels(
    labels: np.ndarray,
    elevation: np.ndarray,
    features: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    level_field: str | None,
) -> np.ndarray:
    """Resolve one level per lake without repeatedly scanning the full raster."""

    from scipy import ndimage

    feature_count = len(features)
    indices = np.arange(1, feature_count + 1, dtype=np.uint32)
    usable = (labels > 0) & np.isfinite(elevation)
    inferred = np.full(feature_count, np.nan, dtype=np.float64)
    if np.any(usable):
        inferred[:] = ndimage.median(
            elevation[usable],
            labels=labels[usable],
            index=indices,
        )
    resolved = inferred.astype(np.float32)
    if level_field:
        for index, (_, properties) in enumerate(features):
            observed = _numeric(properties.get(level_field))
            if observed is not None:
                resolved[index] = observed
    levels = np.full(labels.shape, np.nan, dtype=np.float32)
    water = labels > 0
    levels[water] = resolved[labels[water] - 1]
    return levels


def read_observed_hydrology_window(
    source: rasterio.io.DatasetReader,
    *,
    bounds: tuple[float, float, float, float],
    destination_crs: Any,
    shape: tuple[int, int],
) -> ObservedHydrologyRaster:
    """Reproject a supervision raster into one dataset tile."""

    if source.count != len(OBSERVED_HYDROLOGY_CHANNELS):
        raise ValueError("Observed hydrology raster must have five bands")
    if tuple(source.descriptions) != OBSERVED_HYDROLOGY_CHANNELS:
        raise ValueError("Observed hydrology band contract is incompatible")
    height, width = shape
    destination_transform = rasterio.transform.from_bounds(*bounds, width, height)
    values = np.full((source.count, height, width), np.nan, dtype=np.float32)
    for index in range(source.count):
        resampling = Resampling.nearest if index in (0, 3, 4) else Resampling.max
        reproject(
            source=rasterio.band(source, index + 1),
            destination=values[index],
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=destination_transform,
            dst_crs=destination_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    valid = np.isfinite(values[0]) | np.isfinite(values[3])
    values[0] = np.nan_to_num(values[0], nan=0.0)
    values[3] = np.nan_to_num(values[3], nan=0.0)
    result = ObservedHydrologyRaster(values=values, valid=valid)
    result.validate()
    return result


def _read_vector_features(
    path: str | Path, layer: str | None
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], Any]:
    text_path = str(path)
    suffix = Path(text_path).suffix.lower()
    if suffix in {".json", ".geojson"}:
        with Path(text_path).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        crs = document.get("crs", {}).get("properties", {}).get("name")
        if not crs:
            crs = "EPSG:4326"
        return [
            (feature["geometry"], feature.get("properties") or {})
            for feature in document["features"]
            if feature.get("geometry") is not None
        ], crs
    if suffix == ".shp":
        import shapefile

        reader = shapefile.Reader(text_path)
        fields = [field[0] for field in reader.fields[1:]]
        features = [
            (record.shape.__geo_interface__, dict(zip(fields, record.record)))
            for record in reader.iterShapeRecords()
            if record.shape.shapeType != shapefile.NULL
        ]
        projection = Path(text_path).with_suffix(".prj")
        if not projection.exists():
            raise ValueError(f"Shapefile projection is missing: {projection}")
        return features, projection.read_text(encoding="utf-8")
    try:
        import fiona
    except ImportError as error:
        raise RuntimeError(
            "Reading GPKG or virtual vector sources requires the optional "
            "Fiona dependency"
        ) from error
    fiona_path = text_path
    if suffix == ".zip" and Path(text_path).exists():
        fiona_path = f"zip://{Path(text_path).resolve()}"
    with fiona.open(fiona_path, layer=layer) as source:
        crs = source.crs_wkt or source.crs
        features = [
            (feature["geometry"], dict(feature.get("properties") or {}))
            for feature in source
            if feature.get("geometry") is not None
        ]
    return features, crs


def _rasterize_numeric_property(
    features: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    field: str,
    shape: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    pairs = []
    for geometry, properties in features:
        value = _numeric(properties.get(field))
        if value is not None:
            pairs.append((geometry, value))
    if not pairs:
        return np.full(shape, np.nan, dtype=np.float32)
    values = rasterize(
        pairs,
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        all_touched=True,
        merge_alg=rasterio.enums.MergeAlg.replace,
        dtype="float32",
    )
    return values.astype(np.float32)


def _lake_polygon_features(
    features: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Accept lake polygons directly or polygonize swissTLM3D shorelines."""

    polygon_features = [
        (geometry, properties)
        for geometry, properties in features
        if geometry.get("type") in {"Polygon", "MultiPolygon"}
    ]
    line_features = [
        (geometry, properties)
        for geometry, properties in features
        if geometry.get("type") in {"LineString", "MultiLineString"}
    ]
    if line_features:
        from shapely.geometry import mapping, shape
        from shapely.ops import polygonize

        unresolved = []
        for geometry, properties in line_features:
            shoreline = shape(geometry)
            polygons = list(polygonize([shoreline]))
            if polygons:
                polygon_features.extend(
                    (mapping(polygon), properties) for polygon in polygons
                )
            else:
                unresolved.append(shoreline)
        polygon_features.extend(
            (mapping(polygon), {}) for polygon in polygonize(unresolved)
        )
    if not polygon_features:
        raise ValueError("Lake source contains no polygonizable polygons/shorelines")
    return polygon_features


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _source_record(kind: str, path: str | Path, layer: str | None) -> dict[str, Any]:
    source_path = Path(str(path))
    record = {"kind": kind, "path": str(path), "layer": layer}
    if source_path.exists() and source_path.is_file():
        record["sha256"] = _sha256(source_path)
        if source_path.suffix.lower() == ".shp":
            record["sidecars"] = {
                sidecar.suffix.lower(): _sha256(sidecar)
                for sidecar in sorted(source_path.parent.glob(f"{source_path.stem}.*"))
                if sidecar.is_file()
            }
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@click.command("rasterize-observed-hydrology")
@click.argument("reference_dem", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option("--river-vector", type=str)
@click.option("--river-layer")
@click.option("--river-discharge-field")
@click.option("--river-width-field")
@click.option("--lake-vector", type=str)
@click.option("--lake-layer")
@click.option("--lake-level-field")
@click.option("--coverage-vector", type=str)
@click.option("--coverage-layer")
@click.option("--coverage-field")
@click.option("--coverage-value", "coverage_values", multiple=True)
@click.option("--resolution-m", default=30.0, show_default=True, type=float)
@click.option(
    "--crop-to-observations/--full-reference-extent",
    default=True,
    show_default=True,
)
def rasterize_observed_hydrology_cli(**kwargs):
    """Rasterize observed rivers/lakes against a reference DEM."""

    try:
        report = rasterize_observed_hydrology(**kwargs)
    except (FileExistsError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(report, indent=2, sort_keys=True))
