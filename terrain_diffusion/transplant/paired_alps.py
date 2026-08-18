"""Paired real-Alps diagnostic for the unchanged transplanted stock stack."""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from terrain_diffusion.data.laplacian_encoder import laplacian_encode
from terrain_diffusion.transplant.adapter import MACRO_CHANNEL_NAMES, adapt_macro_patch
from terrain_diffusion.transplant.smoke import (
    DEFAULT_STOCK_MODEL,
    DEFAULT_STOCK_REVISION,
    HYBRID_NAMES,
    SOURCE_ROOT,
    _load_stock_submodel,
    _release_model,
    _save_array,
    _save_scalar_preview,
    _select_device,
    _sha256,
)
from terrain_diffusion.transplant.stock_runtime import (
    LOWFREQ_MEAN,
    LOWFREQ_STD,
    generate_hybrid,
    generate_residual,
    reconstruct_elevation,
)


MACRO_OUTPUT_SLICE = slice(1, 3)
RESOLUTION_30M = 30.0
RESOLUTION_7680M = 7680.0
TARGET_SIZE_30M = 512
CONTEXT_SIZE_30M = 1024

DEFAULT_SOURCE_DEMS = (
    SOURCE_ROOT / "data/alps/alti3d.tif",
    SOURCE_ROOT / "data/alps/sources/austria/austria.tif",
)
DEFAULT_CLIMATE = (
    SOURCE_ROOT / "data/global/wc2.1_10m_bio_1.tif",
    SOURCE_ROOT / "data/global/wc2.1_10m_bio_4.tif",
    SOURCE_ROOT / "data/global/wc2.1_10m_bio_12.tif",
    SOURCE_ROOT / "data/global/wc2.1_10m_bio_15.tif",
)


def _signed_sqrt(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.sign(values) * np.sqrt(np.abs(values))


def _signed_square(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.sign(values) * np.square(values)


def _block_mean(values: np.ndarray, scale: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    height, width = values.shape
    if height % scale or width % scale:
        raise ValueError(f"Shape {values.shape} is not divisible by block scale {scale}")
    return values.reshape(
        height // scale, scale, width // scale, scale
    ).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def _block_percentile(values: np.ndarray, scale: int, percentile: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    height, width = values.shape
    if height % scale or width % scale:
        raise ValueError(f"Shape {values.shape} is not divisible by block scale {scale}")
    blocks = values.reshape(height // scale, scale, width // scale, scale)
    return np.percentile(blocks, percentile, axis=(1, 3)).astype(np.float32)


def _source_id(path: Path) -> str:
    resolved = path.resolve()
    if resolved == DEFAULT_SOURCE_DEMS[0].resolve():
        return "switzerland"
    if resolved == DEFAULT_SOURCE_DEMS[1].resolve():
        return "austria"
    return path.stem


def _random_source_contexts(source_dems: tuple[Path, ...], seed: int) -> list[dict]:
    """Propose random 30.72 km contexts from each 10 m Alpine source root."""
    rng = np.random.default_rng(seed)
    source_order = rng.permutation(len(source_dems))
    contexts = []
    attempts_per_source = 64
    for source_order_rank, source_index in enumerate(source_order):
        path = source_dems[int(source_index)]
        with rasterio.open(path) as source:
            if source.crs is None:
                raise click.ClickException(f"Target DEM has no CRS: {path}")
            if not np.isclose(source.res[0], 10.0) or not np.isclose(source.res[1], 10.0):
                raise click.ClickException(f"Expected a 10 m source DEM: {path}")
            context_source_pixels = int(CONTEXT_SIZE_30M * RESOLUTION_30M / source.res[0])
            if source.height < context_source_pixels or source.width < context_source_pixels:
                raise click.ClickException(f"Source is smaller than one stock context: {path}")
            max_row = source.height - context_source_pixels
            max_col = source.width - context_source_pixels
            for attempt in range(attempts_per_source):
                row = int(rng.integers(max_row + 1))
                col = int(rng.integers(max_col + 1))
                # Align the 30 m diagnostic grid to the native 10 m source grid.
                context_transform_30m = (
                    source.transform
                    * Affine.translation(col, row)
                    * Affine.scale(RESOLUTION_30M / source.res[0])
                )
                context_transform_7680m = context_transform_30m * Affine.scale(256)
                output_transform_30m = context_transform_30m * Affine.translation(256, 256)
                output_left, output_top = output_transform_30m * (0, 0)
                output_right, output_bottom = output_transform_30m * (
                    TARGET_SIZE_30M, TARGET_SIZE_30M
                )
                contexts.append({
                    "case_id": "random_alpine_source_context",
                    "source_id": _source_id(path),
                    "source_order_rank": int(source_order_rank),
                    "selection_seed": int(seed),
                    "selection_attempt": int(attempt),
                    "attempts_per_source": attempts_per_source,
                    "crs_wkt": source.crs.to_wkt(),
                    "source_path": str(path.resolve()),
                    "source_shape": list(source.shape),
                    "source_resolution_m": list(source.res),
                    "source_window_row_col_size_10m": [row, col, context_source_pixels],
                    "context_transform_30m": list(context_transform_30m)[:6],
                    "context_transform_7680m": list(context_transform_7680m)[:6],
                    "output_transform_30m": list(output_transform_30m)[:6],
                    "output_bounds_projected": [output_left, output_bottom, output_right, output_top],
                })
    return contexts


def _read_target_context(
    target_dem: Path, destination_crs: str, destination_transform: Affine
) -> tuple[np.ndarray, dict]:
    destination = np.full(
        (CONTEXT_SIZE_30M, CONTEXT_SIZE_30M), np.nan, dtype=np.float32
    )
    with rasterio.open(target_dem) as source:
        if source.crs is None:
            raise click.ClickException(f"Target DEM has no CRS: {target_dem}")
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=destination_transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        source_info = {
            "path": str(target_dem.resolve()),
            "crs": source.crs.to_string(),
            "shape": list(source.shape),
            "resolution": list(source.res),
            "nodata": source.nodata,
            "bytes": target_dem.stat().st_size,
        }
    valid = np.isfinite(destination) & (np.abs(destination) < 1e10)
    if not valid.all():
        raise click.ClickException(
            "The selected random stock window is not fully covered by the 30 m target "
            f"({valid.mean():.6f} valid); refusing to invent or fill target data"
        )
    return destination, source_info


def _read_climate_context(
    paths: tuple[Path, ...], destination_crs: str, destination_transform: Affine
) -> tuple[np.ndarray, list[dict]]:
    channels = []
    metadata = []
    for path in paths:
        destination = np.full((4, 4), np.nan, dtype=np.float32)
        with rasterio.open(path) as source:
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=destination_transform,
                dst_crs=destination_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            metadata.append(
                {
                    "path": str(path.resolve()),
                    "crs": source.crs.to_string(),
                    "shape": list(source.shape),
                    "nodata": source.nodata,
                }
            )
        if not np.isfinite(destination).all():
            raise click.ClickException(
                f"Climate source does not fully cover the exact Alps context: {path}"
            )
        channels.append(destination)
    return np.stack(channels).astype(np.float32), metadata


def _save_geotiff(path: Path, values: np.ndarray, transform: Affine, crs: str) -> None:
    array = np.asarray(values, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[-2],
        width=array.shape[-1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as destination:
        destination.write(array, 1)


def _save_conditioning_figure(path: Path, values: np.ndarray) -> None:
    cmaps = ("terrain", "terrain", "coolwarm", "magma", "Blues", "viridis")
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for index, (axis, name, cmap) in enumerate(zip(axes.flat, MACRO_CHANNEL_NAMES, cmaps)):
        image = axis.imshow(values[index], cmap=cmap)
        axis.set_title(name.replace("_", " "))
        axis.set_xticks(range(4))
        axis.set_yticks(range(4))
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _pearson(reference: np.ndarray, generated: np.ndarray) -> float | None:
    reference = np.asarray(reference, dtype=np.float64).ravel()
    generated = np.asarray(generated, dtype=np.float64).ravel()
    if reference.size < 2 or reference.std() == 0.0 or generated.std() == 0.0:
        return None
    return float(np.corrcoef(reference, generated)[0, 1])


@click.command("paired-alps-diagnostic")
@click.option(
    "--source-dem",
    "source_dems",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    multiple=True,
    default=DEFAULT_SOURCE_DEMS,
    show_default=True,
    help="10 m DEM source root(s); one source root is chosen uniformly per seed.",
)
@click.option("--stock-model", default=DEFAULT_STOCK_MODEL, show_default=True)
@click.option("--stock-revision", default=DEFAULT_STOCK_REVISION, show_default=True)
@click.option(
    "--allow-download",
    is_flag=True,
    help="Allow Hugging Face downloads; default is the pinned local snapshot only.",
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--seed", type=int, default=74, show_default=True)
@click.option("--device", default=None, help="cuda, mps, or cpu; auto-selected when omitted.")
def paired_alps_diagnostic(
    source_dems: tuple[Path, ...],
    stock_model: str,
    stock_revision: str | None,
    allow_download: bool,
    output: Path,
    seed: int,
    device: str | None,
):
    """Run the unchanged stock downstream stack on one random Alpine source tile."""
    output = output / ("_seed_" + str(seed))
    if (output / "metadata.json").exists():
        raise click.ClickException(
            f"Completed output already exists at {output}; choose a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    device = _select_device(device)
    possible_cases = _random_source_contexts(source_dems, seed)
    target_context_m = None
    target_source = None
    case = None
    coverage_errors = []
    for possible_case in possible_cases:
        try:
            values, source = _read_target_context(
                Path(possible_case["source_path"]),
                possible_case["crs_wkt"],
                Affine(*possible_case["context_transform_30m"]),
            )
        except click.ClickException as error:
            coverage_errors.append(str(error))
            continue
        case = possible_case
        target_context_m = values
        target_source = source
        break
    if case is None or target_context_m is None or target_source is None:
        detail = coverage_errors[-1] if coverage_errors else "no candidate was tested"
        raise click.ClickException(
            f"No random Alpine source context had complete paired 30 m coverage: {detail}"
        )
    crs = case["crs_wkt"]
    context_transform_30m = Affine(*case["context_transform_30m"])
    context_transform_7680m = Affine(*case["context_transform_7680m"])
    output_transform_30m = Affine(*case["output_transform_30m"])

    click.echo(
        f"Random {case['source_id']} context: "
        f"attempt={case['selection_attempt']}/{case['attempts_per_source']}, "
        f"source-order={case['source_order_rank']}"
    )
    click.echo(f"Device: {device}")
    click.echo(f"Reading paired target DEM: {case['source_path']}")
    target_dem_m = target_context_m[256:768, 256:768]
    target_sqrt_context = _signed_sqrt(target_context_m)
    mean_sqrt = _block_mean(target_sqrt_context, 256)
    p5_sqrt = _block_percentile(target_sqrt_context, 256, 5.0)
    climate, climate_sources = _read_climate_context(
        DEFAULT_CLIMATE, crs, context_transform_7680m
    )
    conditioning_physical = np.concatenate(
        [mean_sqrt[None], p5_sqrt[None], climate], axis=0
    ).astype(np.float32)
    adapted = adapt_macro_patch(torch.from_numpy(conditioning_physical))

    click.echo("Real paired conditioning: [6, 4, 4] + adapter all-valid mask")
    _save_array(output / "target_dem_m_512x512.npy", target_dem_m)
    _save_geotiff(output / "target_dem_m_512x512.tif", target_dem_m, output_transform_30m, crs)
    _save_scalar_preview(output / "target_dem_m.png", target_dem_m)
    _save_array(output / "conditioning_physical_6x4x4.npy", conditioning_physical)
    _save_array(output / "conditioning_relief_signed_sqrt_4x4.npy", mean_sqrt - p5_sqrt)
    _save_array(output / "conditioning_with_mask_7x4x4.npy", adapted.physical_with_mask_7x4x4)
    _save_array(output / "conditioning_normalized_7x4x4.npy", adapted.normalized_7x4x4)
    _save_array(output / "conditioning_vector_58.npy", adapted.vector_58)
    _save_conditioning_figure(output / "conditioning_physical_6x4x4.png", conditioning_physical)

    target_sqrt = _signed_sqrt(target_dem_m)
    _, target_lowfreq = laplacian_encode(target_sqrt, 64, sigma=5)
    target_lowfreq = np.asarray(target_lowfreq, dtype=np.float32)
    target_lowfreq_normalized = (target_lowfreq - LOWFREQ_MEAN) / LOWFREQ_STD
    _save_array(output / "target_lowfreq_signed_sqrt_64x64.npy", target_lowfreq)
    _save_array(output / "target_lowfreq_normalized_64x64.npy", target_lowfreq_normalized)
    _save_scalar_preview(output / "target_lowfreq_signed_sqrt.png", target_lowfreq)

    click.echo(f"Loading unchanged stock base_model from {stock_model} at {stock_revision}")
    base = _load_stock_submodel(
        stock_model, "base_model", stock_revision, allow_download
    ).eval().to(device)
    if (base.config.in_channels, base.config.out_channels) != (5, 5):
        raise click.ClickException("Stock base model must be the transplanted 5-channel model")
    if len(base.config.conditional_inputs) != 1 or base.config.conditional_inputs[0][1] != 58:
        raise click.ClickException("Stock base model must accept one 58-value condition")
    hybrid = generate_hybrid(base, adapted.vector_58, seed, device)
    for index, name in enumerate(HYBRID_NAMES):
        _save_array(output / f"hybrid_{index}_{name}_normalized.npy", hybrid[index])
        _save_scalar_preview(
            output / f"hybrid_{index}_{name}_normalized.png",
            hybrid[index],
            cmap="coolwarm",
        )
    generated_lowfreq = hybrid[4] * LOWFREQ_STD + LOWFREQ_MEAN
    _save_array(output / "generated_lowfreq_signed_sqrt_64x64.npy", generated_lowfreq)
    _save_scalar_preview(output / "generated_lowfreq_signed_sqrt.png", generated_lowfreq)
    _release_model(base, device)
    del base

    click.echo(f"Loading unchanged stock decoder_model from {stock_model} at {stock_revision}")
    decoder = _load_stock_submodel(
        stock_model, "decoder_model", stock_revision, allow_download
    ).eval().to(device)
    if (decoder.config.in_channels, decoder.config.out_channels) != (5, 1):
        raise click.ClickException("Stock decoder must be the transplanted 5-to-1 model")
    residual_normalized = generate_residual(decoder, hybrid, seed, device)
    residual_sqrt, reconstruction_lowfreq, elevation_sqrt, final_dem_m = reconstruct_elevation(
        residual_normalized, hybrid, residual_mean=0.0, residual_std=0.7
    )
    _release_model(decoder, device)
    del decoder

    final_numpy = final_dem_m.numpy().astype(np.float32)
    final_240m = _block_mean(final_numpy, 8)
    final_7680m_physical_mean = _block_mean(final_numpy, 256)
    generated_macro_sqrt = _block_mean(_signed_sqrt(final_numpy), 256)
    generated_macro_m = _signed_square(generated_macro_sqrt)
    supplied_macro_sqrt = conditioning_physical[0, MACRO_OUTPUT_SLICE, MACRO_OUTPUT_SLICE]
    supplied_macro_m = _signed_square(supplied_macro_sqrt)
    macro_difference_m = generated_macro_m - supplied_macro_m
    target_240m = _block_mean(target_dem_m, 8)
    target_7680m_physical_mean = _block_mean(target_dem_m, 256)
    final_minus_target_30m = final_numpy - target_dem_m
    final_minus_target_240m = final_240m - target_240m

    _save_array(output / "generated_residual_normalized_512x512.npy", residual_normalized)
    _save_array(output / "generated_residual_signed_sqrt_512x512.npy", residual_sqrt)
    _save_scalar_preview(output / "generated_residual_signed_sqrt.png", residual_sqrt, cmap="coolwarm")
    _save_array(output / "reconstruction_lowfreq_signed_sqrt_64x64.npy", reconstruction_lowfreq)
    _save_array(output / "generated_final_dem_m_512x512.npy", final_dem_m)
    _save_geotiff(output / "generated_final_dem_m_512x512.tif", final_numpy, output_transform_30m, crs)
    _save_scalar_preview(output / "generated_final_dem_m.png", final_dem_m)
    _save_array(output / "generated_minus_target_dem_m_512x512.npy", final_minus_target_30m)
    _save_scalar_preview(
        output / "generated_minus_target_dem_m.png", final_minus_target_30m, cmap="coolwarm"
    )
    _save_array(output / "generated_final_dem_m_64x64_at_240m.npy", final_240m)
    _save_array(output / "target_dem_m_64x64_at_240m.npy", target_240m)
    _save_array(output / "generated_minus_target_dem_m_64x64_at_240m.npy", final_minus_target_240m)
    _save_scalar_preview(output / "generated_final_dem_m_at_240m.png", final_240m)
    transform_240m = output_transform_30m * Affine.scale(8)
    _save_geotiff(
        output / "generated_final_dem_m_64x64_at_240m.tif", final_240m, transform_240m, crs
    )
    _save_array(output / "generated_final_dem_physical_mean_m_2x2_at_7680m.npy", final_7680m_physical_mean)
    _save_array(output / "target_dem_physical_mean_m_2x2_at_7680m.npy", target_7680m_physical_mean)
    _save_array(output / "supplied_macro_elevation_m_2x2.npy", supplied_macro_m)
    _save_array(output / "generated_macro_elevation_m_2x2.npy", generated_macro_m)
    _save_array(output / "generated_minus_supplied_macro_elevation_m_2x2.npy", macro_difference_m)
    _save_scalar_preview(output / "generated_minus_supplied_macro_elevation_m.png", macro_difference_m, cmap="coolwarm")
    transform_7680m = output_transform_30m * Affine.scale(256)
    _save_geotiff(
        output / "generated_final_dem_physical_mean_m_2x2_at_7680m.tif",
        final_7680m_physical_mean,
        transform_7680m,
        crs,
    )

    difference = macro_difference_m.astype(np.float64)
    metrics = {
        "comparison": "generated and supplied elevation reduced in the macro signed-sqrt-mean representation, then inverted to metres",
        "sample_count": int(difference.size),
        "interpretation_note": "correlation is descriptive only because one stock tile contains four 7.68 km comparison cells",
        "bias_m": float(difference.mean()),
        "mae_m": float(np.abs(difference).mean()),
        "rmse_m": float(np.sqrt(np.square(difference).mean())),
        "pearson_correlation": _pearson(supplied_macro_m, generated_macro_m),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    output_bounds = case["output_bounds_projected"]
    output_bounds_wgs84 = transform_bounds(crs, "EPSG:4326", *output_bounds, densify_pts=21)
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--xandergos--terrain-diffusion-30m/snapshots"
        / str(stock_revision)
    )
    metadata = {
        "purpose": "diagnostic only; no training, fine-tuning, or model changes",
        "selected_crop": case,
        "window_selection": "deterministic-random source-root choice, then a random 30.72 km source-aligned context; central 2x2 7.68 km cells are the stock output footprint",
        "output_bounds_wgs84": list(output_bounds_wgs84),
        "target_source": target_source,
        "climate_sources": climate_sources,
        "conditioning_construction": {
            "elevation": "arithmetic mean of signed-sqrt SwissALTI3D elevation over each aligned 256x256 30 m block",
            "p5": "5th percentile of signed-sqrt SwissALTI3D elevation over the same block; adapter receives absolute p5",
            "relief_diagnostic": "signed-sqrt mean minus signed-sqrt p5",
            "climate": "WorldClim BIO 1, 4, 12, 15 bilinearly reprojected to the aligned 4x4 7.68 km context",
            "mask": "all ones, after requiring 100% finite target coverage; appended by the unchanged adapter",
        },
        "stock_model": stock_model,
        "stock_revision": stock_revision,
        "macro_checkpoint_used": False,
        "macro_checkpoint_note": "real paired conditioning is supplied directly to the unchanged adapter",
        "stock_base_weights": str((snapshot / "base_model/diffusion_pytorch_model.safetensors").resolve()),
        "stock_decoder_weights": str((snapshot / "decoder_model/diffusion_pytorch_model.safetensors").resolve()),
        "stock_base_sha256": _sha256(snapshot / "base_model/diffusion_pytorch_model.safetensors"),
        "stock_decoder_sha256": _sha256(snapshot / "decoder_model/diffusion_pytorch_model.safetensors"),
        "seed": int(seed),
        "device": device,
        "resolutions_m": {"conditioning": 7680, "hybrid": 240, "target_and_output": 30},
        "shapes": {
            "target_dem": list(target_dem_m.shape),
            "conditioning_physical": list(conditioning_physical.shape),
            "conditioning_with_mask": list(adapted.physical_with_mask_7x4x4.shape),
            "conditioning_vector": list(adapted.vector_58.shape),
            "hybrid": list(hybrid.shape),
            "generated_residual": list(residual_normalized.shape),
            "generated_final_dem": list(final_numpy.shape),
            "generated_240m": list(final_240m.shape),
            "generated_7680m": list(final_7680m_physical_mean.shape),
        },
        "aggregation_definitions": {
            "30m_to_240m": "aligned non-overlapping arithmetic mean of physical elevation over 8x8 pixels",
            "30m_to_7680m_physical": "aligned non-overlapping arithmetic mean of physical elevation over 256x256 pixels",
            "macro_fidelity": "aligned mean of signed-sqrt elevation over 256x256 pixels, followed by signed-square inversion to metres; compared with conditioning channel 0",
            "target_lowfreq": "stock laplacian_encode(signed_sqrt(target_dem), downsample_size=64, sigma=5); no autoencoder",
        },
        "metrics": metrics,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    click.echo(json.dumps(metrics, indent=2))
    click.echo(f"Saved paired Alps diagnostic to {output.resolve()}")
