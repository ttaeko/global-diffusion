"""Paired macro-context / stock-base-noise attribution at physical 240 m.

This is a diagnostic entry point only.  It uses the transplant's exact macro
sampler, 4x4-to-58 adapter, and stock base sampler; it neither trains nor
modifies any model or hydrology product.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch

from terrain_diffusion.hydrology.compiled_routing import priority_flood_route_compiled
from terrain_diffusion.hydrology.hybrid_conditioning import hybrid_fill_breach_route
from terrain_diffusion.hydrology.lakes import identify_depression_lakes
from terrain_diffusion.transplant.adapter import adapt_macro_patch
from terrain_diffusion.transplant.smoke import (
    DEFAULT_MACRO_MODEL,
    DEFAULT_MACRO_STATS,
    DEFAULT_STOCK_MODEL,
    DEFAULT_STOCK_REVISION,
    _choose_mountain_crop,
    _load_stock_submodel,
    _release_model,
    _sample_macro,
    _select_device,
)
from terrain_diffusion.transplant.stock_runtime import LOWFREQ_MEAN, LOWFREQ_STD, generate_hybrid


RESOLUTION_M = 240.0
DEPRESSION_TOLERANCE_M = 0.05


def _parse_seeds(value: str | None, count: int, start: int, option: str) -> list[int]:
    if value:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
        if not seeds:
            raise click.BadParameter("must contain at least one integer", param_hint=option)
        if len(set(seeds)) != len(seeds):
            raise click.BadParameter("must not contain duplicates", param_hint=option)
        return seeds
    return list(range(start, start + count))


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hillshade(dem: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(dem, RESOLUTION_M, RESOLUTION_M)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth, altitude = np.deg2rad(315.0), np.deg2rad(45.0)
    return np.clip(
        np.sin(altitude) * np.cos(slope)
        + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect), 0.0, 1.0
    )


def _save_preview(path: Path, elevation_m: np.ndarray, closed_mask: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.imshow(_hillshade(elevation_m), cmap="gray", origin="upper")
    overlay = np.ma.masked_where(~closed_mask, closed_mask)
    axis.imshow(overlay, cmap="autumn", alpha=0.58, origin="upper")
    axis.set(title="240 m hillshade; priority-flood depression overlay", xticks=[], yticks=[])
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _terrain_statistics(elevation_m: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(elevation_m)
    if not np.any(finite):
        return {key: float("nan") for key in (
            "elevation_mean_m", "elevation_std_m", "elevation_min_m", "elevation_max_m",
            "relief_m", "slope_mean_m_per_m", "slope_std_m_per_m", "slope_rms_m_per_m",
            "laplacian_rms_m_per_m2",
        )}
    dem = elevation_m[finite]
    gy, gx = np.gradient(elevation_m, RESOLUTION_M, RESOLUTION_M)
    slope = np.hypot(gx, gy)[finite]
    laplacian = np.gradient(gx, RESOLUTION_M, axis=1) + np.gradient(gy, RESOLUTION_M, axis=0)
    return {
        "elevation_mean_m": float(np.mean(dem)), "elevation_std_m": float(np.std(dem)),
        "elevation_min_m": float(np.min(dem)), "elevation_max_m": float(np.max(dem)),
        "relief_m": float(np.ptp(dem)), "slope_mean_m_per_m": float(np.mean(slope)),
        "slope_std_m_per_m": float(np.std(slope)), "slope_rms_m_per_m": float(np.sqrt(np.mean(slope**2))),
        "laplacian_rms_m_per_m2": float(np.sqrt(np.mean(laplacian[finite] ** 2))),
    }


def _basin_metrics(elevation_m: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    """Inventory raw depressions through the existing priority-flood/lake stack."""
    land = np.isfinite(elevation_m) & (elevation_m > 0.0)
    empty = np.zeros(elevation_m.shape, dtype=bool)
    if not np.any(land):
        return {
            "land_fraction": 0.0, "sink_count": 0, "closed_basin_count": 0,
            "closed_basin_area_km2": 0.0, "closed_basin_area_fraction": 0.0,
            "largest_closed_basin_area_km2": 0.0, "median_closed_basin_area_km2": 0.0,
            "lake_candidate_count": 0, "lake_candidate_area_fraction": 0.0,
            "breach_candidate_components": 0, "breach_path_count": 0,
            "breach_cell_count": 0, "breach_maximum_incision_m": 0.0,
        }, empty
    routing = priority_flood_route_compiled(elevation_m, resolution_m=RESOLUTION_M, land_mask=land)
    # A connected component requiring positive priority-flood correction is an
    # internal depression with this 4x4 window's open edge/ocean as its outlet.
    closed = land & (routing.elevation_correction_m > DEPRESSION_TOLERANCE_M)
    labels, count = scipy.ndimage.label(closed, structure=np.ones((3, 3), dtype=np.uint8))
    areas_km2 = np.bincount(labels.ravel(), minlength=count + 1)[1:] * (RESOLUTION_M ** 2 / 1e6)
    lakes = identify_depression_lakes(
        elevation_m, routing.elevation_conditioned_m, resolution_m=RESOLUTION_M, land_mask=land
    )
    hybrid = hybrid_fill_breach_route(
        elevation_m, resolution_m=RESOLUTION_M, land_mask=land, preserve_mask=lakes.lake_mask
    )
    return {
        "land_fraction": float(land.mean()), "sink_count": int(count),
        "closed_basin_count": int(count), "closed_basin_area_km2": float(areas_km2.sum()),
        "closed_basin_area_fraction": float(closed.sum() / land.sum()),
        "largest_closed_basin_area_km2": float(areas_km2.max(initial=0.0)),
        "median_closed_basin_area_km2": float(np.median(areas_km2)) if count else 0.0,
        "lake_candidate_count": len(lakes.records),
        "lake_candidate_area_fraction": float(lakes.lake_mask.sum() / land.sum()),
        "breach_candidate_components": int(sum(item.candidate_components for item in hybrid.metrics)),
        "breach_path_count": int(sum(item.breached_paths for item in hybrid.metrics)),
        "breach_cell_count": int(hybrid.breach_mask.sum()),
        "breach_maximum_incision_m": float(max((item.maximum_incision_m for item in hybrid.metrics), default=0.0)),
    }, closed


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict], context_ids: list[str], base_seeds: list[int]) -> tuple[list[dict], dict]:
    summary_metrics = (
        "sink_count", "closed_basin_count", "closed_basin_area_km2", "closed_basin_area_fraction",
        "largest_closed_basin_area_km2", "median_closed_basin_area_km2", "lake_candidate_count",
        "lake_candidate_area_fraction", "breach_candidate_components", "breach_path_count",
        "breach_cell_count", "breach_maximum_incision_m", "elevation_mean_m", "elevation_std_m",
        "relief_m", "slope_mean_m_per_m", "slope_rms_m_per_m", "laplacian_rms_m_per_m2",
    )
    attribution_metrics = ("sink_count", "closed_basin_count", "closed_basin_area_fraction")
    grouped: list[dict] = []
    variance: dict[str, dict] = {}
    for metric in summary_metrics:
        for label, key, values in (
            ("macro_context", "macro_context_id", context_ids),
            ("base_seed", "base_seed", base_seeds),
        ):
            for value in values:
                series = np.asarray([row[metric] for row in rows if row[key] == value], dtype=float)
                grouped.append({"group_by": label, "group": value, "metric": metric,
                                "mean": float(np.mean(series)), "std": float(np.std(series)),
                                "minimum": float(np.min(series)), "maximum": float(np.max(series)),
                                "range": float(np.ptp(series)), "n": int(series.size)})
        if metric not in attribution_metrics:
            continue
        matrix = np.asarray([[next(row[metric] for row in rows if row["macro_context_id"] == context and row["base_seed"] == seed) for seed in base_seeds] for context in context_ids], dtype=float)
        grand = float(matrix.mean())
        context_means, seed_means = matrix.mean(axis=1), matrix.mean(axis=0)
        total = float(np.mean((matrix - grand) ** 2))
        macro = float(np.mean((context_means - grand) ** 2))
        noise = float(np.mean((seed_means - grand) ** 2))
        interaction = float(np.mean((matrix - context_means[:, None] - seed_means[None, :] + grand) ** 2))
        variance[metric] = {"total_variance": total, "macro_context_variance": macro,
                            "base_seed_variance": noise, "interaction_variance": interaction,
                            "macro_context_fraction": macro / total if total else 0.0,
                            "base_seed_fraction": noise / total if total else 0.0,
                            "interaction_fraction": interaction / total if total else 0.0,
                            "range": float(matrix.max() - matrix.min())}
    return grouped, variance


@click.command("audit-240m-basin-attribution")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--macro-model", type=click.Path(path_type=Path, exists=True, file_okay=False), default=DEFAULT_MACRO_MODEL, show_default=True)
@click.option("--macro-stats", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=DEFAULT_MACRO_STATS, show_default=True)
@click.option("--stock-model", default=DEFAULT_STOCK_MODEL, show_default=True)
@click.option("--stock-revision", default=DEFAULT_STOCK_REVISION, show_default=True)
@click.option("--allow-download", is_flag=True, help="Allow stock-model download; default is cache-only.")
@click.option("--macro-seeds", help="Comma-separated macro seeds; overrides --num-contexts.")
@click.option("--num-contexts", type=click.IntRange(1), default=5, show_default=True)
@click.option("--macro-seed-start", type=int, default=74, show_default=True)
@click.option("--base-seeds", help="Comma-separated stock-base seeds; overrides --num-base-seeds.")
@click.option("--num-base-seeds", type=click.IntRange(1), default=5, show_default=True)
@click.option("--base-seed-start", type=int, default=1000, show_default=True)
@click.option("--land-fraction", type=click.FloatRange(0.0, 1.0), default=0.60, show_default=True)
@click.option("--macro-steps", type=click.IntRange(1), default=30, show_default=True)
@click.option("--coarse-row", type=click.IntRange(0, 252), default=120, show_default=True)
@click.option("--coarse-col", type=click.IntRange(0, 252), default=120, show_default=True)
@click.option("--use-mountain-crop", "choose_mountain_crop", is_flag=True, help="Use transplant.smoke's deterministic mountain-crop selector for each macro sample.")
@click.option("--diagnostics-only-from-saved-contexts", type=click.Path(path_type=Path, exists=True, file_okay=False), help="Reuse CONTEXTS_DIR/context_*/macro_patch_physical_6x4x4.npy; do not sample/load the macro model.")
@click.option("--device", default=None, help="cuda, mps, or cpu; auto-selected when omitted.")
def audit_240m_basin_attribution(**kwargs):
    """Run the paired 240 m attribution matrix without decoder or reconciliation."""
    output: Path = kwargs["output"]
    if (output / "matrix.csv").exists():
        raise click.ClickException(f"Completed output exists at {output}; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    macro_seeds = _parse_seeds(kwargs["macro_seeds"], kwargs["num_contexts"], kwargs["macro_seed_start"], "--macro-seeds")
    base_seeds = _parse_seeds(kwargs["base_seeds"], kwargs["num_base_seeds"], kwargs["base_seed_start"], "--base-seeds")
    device = _select_device(kwargs["device"])
    contexts_dir = output / "contexts"; contexts_dir.mkdir()
    contexts: list[dict] = []
    if kwargs["diagnostics_only_from_saved_contexts"]:
        source_contexts = kwargs["diagnostics_only_from_saved_contexts"]
        saved = sorted(source_contexts.glob("context_*/metadata.json"))
        if not saved:
            raise click.ClickException("No saved contexts found under the supplied directory")
        for metadata_path in saved:
            meta = json.loads(metadata_path.read_text())
            patch = np.load(metadata_path.parent / "macro_patch_physical_6x4x4.npy")
            destination = contexts_dir / meta["macro_context_id"]
            destination.mkdir()
            np.save(destination / "macro_patch_physical_6x4x4.npy", patch)
            _save_json(destination / "metadata.json", meta)
            contexts.append({**meta, "patch": patch})
        macro_seeds = [int(item["macro_seed"]) for item in contexts]
    else:
        # Macro has its own checkpoint class; importing it lazily keeps saved-context mode model-free.
        from terrain_diffusion.models.edm_unet import EDMUnet2D
        macro = EDMUnet2D.from_pretrained(kwargs["macro_model"]).eval().to(device)
        for index, seed in enumerate(macro_seeds):
            _, physical, _, _ = _sample_macro(macro, seed, kwargs["land_fraction"], kwargs["macro_steps"], kwargs["macro_stats"], device)
            row, col = kwargs["coarse_row"], kwargs["coarse_col"]
            crop = None
            if kwargs["choose_mountain_crop"]:
                row, col, crop = _choose_mountain_crop(physical)
            patch = physical[:, row:row + 4, col:col + 4].detach().cpu().numpy().astype(np.float32)
            context_id = f"context_{index:02d}"
            directory = contexts_dir / context_id; directory.mkdir()
            np.save(directory / "macro_patch_physical_6x4x4.npy", patch)
            meta = {"macro_context_id": context_id, "macro_seed": seed, "coarse_row": row, "coarse_col": col,
                    "land_fraction": kwargs["land_fraction"], "macro_steps": kwargs["macro_steps"],
                    "use_mountain_crop": bool(kwargs["choose_mountain_crop"]), "mountain_crop": crop,
                    "patch_sha256": _sha256_array(patch)}
            _save_json(directory / "metadata.json", meta)
            contexts.append({**meta, "patch": patch})
        _release_model(macro, device)
    rows: list[dict] = []
    base = _load_stock_submodel(kwargs["stock_model"], "base_model", kwargs["stock_revision"], kwargs["allow_download"]).eval().to(device)
    if (base.config.in_channels, base.config.out_channels) != (5, 5) or len(base.config.conditional_inputs) != 1 or base.config.conditional_inputs[0][1] != 58:
        raise click.ClickException("Stock base model does not match the production 5-channel / 58-value interface")
    for context in contexts:
        adapted = adapt_macro_patch(torch.from_numpy(context["patch"]))
        for base_seed in base_seeds:
            sample_dir = output / context["macro_context_id"] / f"seed_{base_seed:04d}"; sample_dir.mkdir(parents=True)
            hybrid = generate_hybrid(base, adapted.vector_58, base_seed, device)
            lowfreq_sqrt = hybrid[4].numpy() * LOWFREQ_STD + LOWFREQ_MEAN
            elevation_m = np.sign(lowfreq_sqrt) * np.square(lowfreq_sqrt)
            metrics, closed = _basin_metrics(elevation_m)
            np.save(sample_dir / "lowfreq_elevation_m_240m.npy", elevation_m.astype(np.float32))
            _save_preview(sample_dir / "preview_closed_basins.png", elevation_m, closed)
            row = {key: value for key, value in context.items() if key not in {"patch", "mountain_crop"}}
            row.update({"base_seed": base_seed, "resolution_m": RESOLUTION_M, **_terrain_statistics(elevation_m), **metrics})
            _save_json(sample_dir / "metrics.json", row); rows.append(row)
    _release_model(base, device)
    _write_csv(output / "matrix.csv", rows)
    context_ids = [item["macro_context_id"] for item in contexts]
    grouped, variance = _summaries(rows, context_ids, base_seeds)
    _write_csv(output / "summary_by_factor.csv", grouped)
    _save_json(output / "attribution_summary.json", {"variance_decomposition": variance, "contexts": [{key: value for key, value in item.items() if key != "patch"} for item in contexts], "base_seeds": base_seeds})
    click.echo(f"Saved {len(rows)} samples, matrix.csv, summary_by_factor.csv, and attribution_summary.json to {output}")


if __name__ == "__main__":
    audit_240m_basin_attribution()
