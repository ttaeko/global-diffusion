"""One-sample end-to-end smoke test for the clean architecture transplant."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from terrain_diffusion.inference.portable_rng import standard_normal
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.datasets.h5_macro_terrain_dataset import load_macro_stats
from terrain_diffusion.transplant.adapter import (
    MACRO_CHANNEL_NAMES,
    STOCK_CONDITIONING_MEAN,
    STOCK_CONDITIONING_STD,
    adapt_macro_patch,
)
from terrain_diffusion.transplant.stock_runtime import (
    LOWFREQ_MEAN,
    LOWFREQ_STD,
    generate_hybrid,
    generate_hybrid_stage,
    generate_residual,
    reconstruct_elevation,
)
from terrain_diffusion.hydrology.mosaic import (
    reconcile_stock_mosaic_hydrology_30m,
    render_river_overlay_on_30m_hillshade,
    run_stock_mosaic_hydrology,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT.parent / "terrain-diffusion"
DEFAULT_MACRO_MODEL = (
    SOURCE_ROOT / "remote_backups/cuda-epoch500/latest_checkpoint/saved_model"
)
DEFAULT_MACRO_STATS = SOURCE_ROOT / "data/macro_terrain.h5"
DEFAULT_STOCK_MODEL = "xandergos/terrain-diffusion-30m"
DEFAULT_STOCK_REVISION = "9ef8030cb805b433b98ec25c5dddefbac07a9e26"
HYBRID_NAMES = ("z1", "z2", "z3", "z4", "lowfreq")
MOSAIC_TILES_PER_SIDE = 7
LATENT_TILE_SIZE = 64
LATENT_TILE_STRIDE = 32
LATENT_WINDOWS_PER_SIDE = 13
DECODER_TILE_SIZE = 512
DECODER_TILE_STRIDE = 384
DECODER_WINDOWS_PER_SIDE = 9
MACRO_CELLS_PER_STOCK_TILE = 2
MACRO_CONTEXT_CELLS = MOSAIC_TILES_PER_SIDE * MACRO_CELLS_PER_STOCK_TILE + 2
MACRO_FOOTPRINT_CELLS = MOSAIC_TILES_PER_SIDE * MACRO_CELLS_PER_STOCK_TILE
STOCK_TILE_SIZE_PX = 512
NATIVE_RESOLUTION_M = 30

def _get_lowfreq_as_elevation(lowfreq_sqrt):
    lowfreq_m = np.sign(lowfreq_sqrt) * np.square(lowfreq_sqrt)
    return lowfreq_m

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


@torch.no_grad()
def _sample_macro(model, seed: int, land_fraction: float, steps: int, stats_file: Path, device: str):
    scheduler = EDMDPMSolverMultistepScheduler(
        sigma_min=0.002, sigma_max=80, sigma_data=0.5
    )
    scheduler.set_timesteps(steps)
    noise = standard_normal(seed, (1, 6, 256, 256), dtype=np.float32)
    sample = torch.from_numpy(noise).to(device) * scheduler.sigmas[0]
    conditional_count = len(model.conditional_layers)
    if conditional_count != 1:
        raise RuntimeError(
            f"Epoch-500 macro model must have one conditioning input, found {conditional_count}"
        )
    conditional_inputs = [
        torch.tensor([land_fraction * 2.0 - 1.0], device=device, dtype=sample.dtype)
    ]
    for timestep, sigma in zip(scheduler.timesteps, scheduler.sigmas):
        timestep, sigma = timestep.to(device), sigma.to(device)
        scaled = scheduler.precondition_inputs(sample, sigma)
        label = scheduler.trigflow_precondition_noise(sigma.view(-1)).to(device)
        prediction = model(
            scaled, noise_labels=label, conditional_inputs=conditional_inputs
        )
        sample = scheduler.step(prediction, timestep, sample).prev_sample
    normalized = sample[0].cpu().float() / scheduler.config.sigma_data
    means_values, stds_values = load_macro_stats(str(stats_file))
    means = torch.tensor(means_values).view(6, 1, 1)
    stds = torch.tensor(stds_values).view(6, 1, 1)
    physical = normalized * stds + means
    # Training channel 1 is mean-minus-p5; the stock coarse/base interface uses
    # absolute p5 in the same signed-square-root representation.
    physical[1] = physical[0] - physical[1]
    if not torch.isfinite(physical).all():
        raise RuntimeError("Macro sampling produced non-finite values")
    return normalized, physical, means_values, stds_values


def _save_array(path: Path, array: torch.Tensor | np.ndarray) -> None:
    values = array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else array
    np.save(path, np.asarray(values, dtype=np.float32))


def _save_scalar_preview(path: Path, values, cmap: str = "terrain") -> None:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
    plt.imsave(path, np.asarray(array), cmap=cmap)


def _stock_linear_weight_window(size: int) -> torch.Tensor:
    """Exact WorldPipeline.linear_weight_window, kept local to this transplant."""
    mid = (size - 1) / 2
    axis = torch.arange(size, dtype=torch.float32)
    weight = 1 - (1 - 1e-3) * torch.clamp(torch.abs(axis - mid) / mid, 0, 1)
    return weight[:, None] * weight[None, :]


def _blend_stock_tiles(tiles: list[torch.Tensor], starts: list[tuple[int, int]], size: int, output_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum stock-weighted windows, then normalize by the summed weight channel."""
    weight = _stock_linear_weight_window(size)
    values = torch.zeros((tiles[0].shape[0], output_size, output_size), dtype=torch.float32)
    weight_sum = torch.zeros((output_size, output_size), dtype=torch.float32)
    for tile, (top, left) in zip(tiles, starts):
        values[:, top:top + size, left:left + size] += tile.float() * weight
        weight_sum[top:top + size, left:left + size] += weight
    if torch.any(weight_sum <= 0):
        raise RuntimeError("Stock blending left uncovered output pixels")
    return values / weight_sum, weight_sum


def _signed_square(values: np.ndarray) -> np.ndarray:
    """Convert the model's signed-square-root elevation representation to metres."""
    return np.sign(values) * np.square(values)


def _choose_mountain_crop(macro_physical: torch.Tensor) -> tuple[int, int, dict[str, float]]:
    """Choose the highest-relief valid mosaic context from its central footprint.

    A seven-by-seven stock mosaic decodes the central 14x14 macro cells. The
    one-cell border is conditioning context and does not affect this score.
    """
    macro = macro_physical.detach().cpu().numpy()
    mean_elevation_m = _signed_square(macro[0])
    p5_elevation_m = _signed_square(macro[1])
    best: tuple[float, int, int, dict[str, float]] | None = None
    context_limit = 256 - MACRO_CONTEXT_CELLS + 1
    for row in range(context_limit):
        for col in range(context_limit):
            central_mean = mean_elevation_m[row + 1 : row + 1 + MACRO_FOOTPRINT_CELLS, col + 1 : col + 1 + MACRO_FOOTPRINT_CELLS]
            central_p5 = p5_elevation_m[row + 1 : row + 1 + MACRO_FOOTPRINT_CELLS, col + 1 : col + 1 + MACRO_FOOTPRINT_CELLS]
            relief = central_mean - central_p5
            mean_relief_m = float(np.mean(relief))
            elevation_range_m = float(np.ptp(central_mean))
            # Adjacent central-cell elevation differences are a cheap, coarse
            # slope-like signal.  At 7.68 km spacing its numeric value is metres.
            edge_differences = np.concatenate(
                (np.diff(central_mean, axis=0).ravel(), np.diff(central_mean, axis=1).ravel())
            )
            gradient_rms_m = float(np.sqrt(np.mean(np.square(edge_differences))))
            score = mean_relief_m + 0.5 * elevation_range_m + 0.5 * gradient_rms_m
            components = {
                "score": float(score),
                "central_mean_relief_m": mean_relief_m,
                "central_elevation_range_m": elevation_range_m,
                "central_adjacent_gradient_rms_m": gradient_rms_m,
            }
            candidate = (score, row, col, components)
            # Strict comparison keeps ties deterministic: first in row-major
            # order wins, rather than depending on platform-specific sorting.
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    return best[1], best[2], best[3]


def _save_macro_crop_preview(path: Path, elevation_m: torch.Tensor, row: int, col: int) -> None:
    """Save the full macro elevation map with context and decoded footprint."""
    figure, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = axis.imshow(elevation_m.detach().cpu().numpy(), cmap="terrain", origin="upper")
    axis.add_patch(Rectangle((col, row), MACRO_CONTEXT_CELLS, MACRO_CONTEXT_CELLS, fill=False, edgecolor="cyan", linewidth=2, label="16x16 conditioning context"))
    axis.add_patch(Rectangle((col + 1, row + 1), MACRO_FOOTPRINT_CELLS, MACRO_FOOTPRINT_CELLS, fill=False, edgecolor="magenta", linewidth=2, label="central 14x14 decoded footprint"))
    axis.set(title="7.68 km macro mean elevation", xlabel="coarse column", ylabel="coarse row")
    axis.legend(loc="upper right")
    figure.colorbar(image, ax=axis, label="elevation (m)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _final_dem_diagnostics(elevation_m: torch.Tensor, resolution_m: float = 30.0) -> dict[str, float]:
    """Small deterministic terrain-detail summary; it does not affect generation."""
    dem = elevation_m.detach().cpu().numpy().astype(np.float64)
    gradient_y, gradient_x = np.gradient(dem, resolution_m, resolution_m)
    diagnostics = {"gradient_rms_m_per_m": float(np.sqrt(np.mean(gradient_x**2 + gradient_y**2)))}
    centered = dem - np.mean(dem)
    spectrum = np.fft.fft2(centered)
    frequencies = np.fft.fftfreq(dem.shape[0], d=resolution_m)
    frequency_y, frequency_x = np.meshgrid(frequencies, frequencies, indexing="ij")
    radial_frequency = np.hypot(frequency_x, frequency_y)
    for lower_m, upper_m in ((60, 120), (120, 240), (240, 480), (480, 960)):
        mask = (radial_frequency >= 1.0 / upper_m) & (radial_frequency < 1.0 / lower_m)
        band = np.fft.ifft2(spectrum * mask).real
        diagnostics[f"band_rms_{lower_m}_{upper_m}m"] = float(np.sqrt(np.mean(band**2)))
    return diagnostics


def _save_hillshade(path: Path, elevation_m: torch.Tensor, resolution_m: float = 30.0) -> None:
    dem = elevation_m.detach().cpu().numpy()
    gradient_y, gradient_x = np.gradient(dem, resolution_m, resolution_m)
    slope = np.arctan(np.hypot(gradient_x, gradient_y))
    aspect = np.arctan2(-gradient_x, gradient_y)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    hillshade = np.sin(altitude) * np.cos(slope) + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    _save_scalar_preview(path, np.clip(hillshade, 0.0, 1.0), cmap="gray")


def _write_tiff_if_available(path: Path, values: torch.Tensor) -> bool:
    try:
        import tifffile
    except ImportError:
        return False
    tifffile.imwrite(path, values.detach().cpu().numpy().astype(np.float32))
    return True


def _load_stock_submodel(
    stock_model: str,
    subfolder: str,
    revision: str | None,
    allow_download: bool,
):
    if not allow_download and revision and not Path(stock_model).exists():
        repo_cache_name = "models--" + stock_model.replace("/", "--")
        pinned_snapshot = (
            Path.home()
            / ".cache/huggingface/hub"
            / repo_cache_name
            / "snapshots"
            / revision
        )
        local_submodel = pinned_snapshot / subfolder
        if local_submodel.exists():
            # Use the concrete snapshot directory. This remains reliable when
            # XDG_CACHE_HOME points Matplotlib/font caches at a temporary path.
            return EDMUnet2D.from_pretrained(
                local_submodel,
                local_files_only=True,
            )
    kwargs = {
        "subfolder": subfolder,
        "local_files_only": not allow_download,
    }
    if revision and not Path(stock_model).exists():
        kwargs["revision"] = revision
    return EDMUnet2D.from_pretrained(stock_model, **kwargs)


def _release_model(model, device: str) -> None:
    """Release a completed stage so the next literal stock stage fits on-device."""
    model.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


@click.command("smoke-transplant")
@click.option(
    "--macro-model",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=DEFAULT_MACRO_MODEL,
    show_default=True,
)
@click.option(
    "--macro-stats",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_MACRO_STATS,
    show_default=True,
)
@click.option("--stock-model", default=DEFAULT_STOCK_MODEL, show_default=True)
@click.option("--stock-revision", default=DEFAULT_STOCK_REVISION, show_default=True)
@click.option(
    "--allow-download",
    is_flag=True,
    help="Allow Hugging Face downloads. The default is cache-only and makes no network requests.",
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--seed", type=int, default=74, show_default=True)
@click.option("--land-fraction", type=click.FloatRange(0.0, 1.0), default=0.60)
@click.option("--macro-steps", type=click.IntRange(min=1), default=30, show_default=True)
@click.option("--coarse-row", type=click.IntRange(0, 240), default=120, show_default=True)
@click.option("--coarse-col", type=click.IntRange(0, 240), default=120, show_default=True)
@click.option(
    "--choose-mountain-crop",
    is_flag=True,
    help="Choose the valid 16x16 context whose central 14x14 footprint has the strongest coarse relief.",
)
@click.option("--run-hydrology", is_flag=True, help="Run the transplanted 7.68 km -> 240 m hydrology hierarchy on the stock lowfreq DEM.")
@click.option("--reconcile-hydrology-30m", is_flag=True, help="Run the existing 240 m -> 30 m regional hydrology reconciliation after --run-hydrology.")
@click.option("--device", default=None, help="cuda, mps, or cpu; auto-selected when omitted.")
def smoke_transplant(
    macro_model,
    macro_stats,
    stock_model,
    stock_revision,
    allow_download,
    output,
    seed,
    land_fraction,
    macro_steps,
    coarse_row,
    coarse_col,
    choose_mountain_crop,
    run_hydrology,
    reconcile_hydrology_30m,
    device,
):
    """Run one honest epoch-500 macro -> stock base -> stock decoder sample."""
    if reconcile_hydrology_30m and not run_hydrology:
        raise click.UsageError("--reconcile-hydrology-30m requires --run-hydrology")
    device = _select_device(device)
    if (output / "metadata.json").exists():
        raise click.ClickException(
            f"Completed output already exists at {output}; choose a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    click.echo(f"Device: {device}")
    click.echo(f"Loading promoted macro checkpoint: {macro_model}")
    macro = EDMUnet2D.from_pretrained(macro_model).eval().to(device)

    click.echo(f"Sampling 7.68 km macro output: [6, 256, 256], steps={macro_steps}")
    macro_normalized, macro_physical, macro_means, macro_stds = _sample_macro(
        macro, seed, land_fraction, macro_steps, macro_stats, device
    )
    macro_elevation_m = torch.sign(macro_physical[0]) * torch.square(macro_physical[0])
    _save_array(output / "macro_normalized_6x256x256.npy", macro_normalized)
    _save_array(output / "macro_physical_6x256x256.npy", macro_physical)
    _save_array(output / "macro_elevation_m_256x256.npy", macro_elevation_m)
    _save_scalar_preview(output / "macro_elevation.png", macro_elevation_m)

    mountain_crop = None
    if choose_mountain_crop:
        coarse_row, coarse_col, mountain_crop = _choose_mountain_crop(macro_physical)
        click.echo(
            "Selected mountain crop: "
            f"row={coarse_row}, col={coarse_col}, score={mountain_crop['score']:.2f} "
            f"(mean relief={mountain_crop['central_mean_relief_m']:.2f} m, "
            f"range={mountain_crop['central_elevation_range_m']:.2f} m, "
            f"gradient RMS={mountain_crop['central_adjacent_gradient_rms_m']:.2f} m)"
        )
        _save_macro_crop_preview(
            output / "macro_elevation_selected_crop.png", macro_elevation_m, coarse_row, coarse_col
        )

    adapted_tiles = []
    for tile_row in range(LATENT_WINDOWS_PER_SIDE):
        for tile_col in range(LATENT_WINDOWS_PER_SIDE):
            patch_row = coarse_row + tile_row
            patch_col = coarse_col + tile_col
            adapted_tiles.append(adapt_macro_patch(macro_physical[:, patch_row:patch_row + 4, patch_col:patch_col + 4]))
    adapted = adapted_tiles[0]
    _save_array(output / "conditioning_physical_169x6x4x4.npy", np.stack([item.physical_6x4x4.numpy() for item in adapted_tiles]))
    _save_array(output / "conditioning_with_mask_169x7x4x4.npy", np.stack([item.physical_with_mask_7x4x4.numpy() for item in adapted_tiles]))
    _save_array(output / "conditioning_normalized_169x7x4x4.npy", np.stack([item.normalized_7x4x4.numpy() for item in adapted_tiles]))
    _save_array(output / "conditioning_vector_169x58.npy", np.stack([item.vector_58.numpy() for item in adapted_tiles]))
    click.echo(f"7.68 output: {list(macro_physical.shape)}")
    click.echo(f"stock conditioning patches: 169 x {list(adapted.physical_with_mask_7x4x4.shape)}")
    click.echo(f"stock conditioning vectors: 169 x {list(adapted.vector_58.shape)}")

    _release_model(macro, device)
    del macro
    click.echo(f"Loading stock base_model from {stock_model} at {stock_revision}")
    base = _load_stock_submodel(
        stock_model, "base_model", stock_revision, allow_download
    ).eval().to(device)
    if (base.config.in_channels, base.config.out_channels) != (5, 5):
        raise click.ClickException("Stock base model must be a 5-channel diffusion model")
    if len(base.config.conditional_inputs) != 1 or base.config.conditional_inputs[0][1] != 58:
        raise click.ClickException("Stock base model must accept one 58-value conditioning tensor")
    latent_starts = [(row * LATENT_TILE_STRIDE, col * LATENT_TILE_STRIDE) for row in range(LATENT_WINDOWS_PER_SIDE) for col in range(LATENT_WINDOWS_PER_SIDE)]
    first_stage = [generate_hybrid_stage(base, item.vector_58, seed, device, stage=0, tile_row=row, tile_col=col) for (row, col), item in zip([(r, c) for r in range(LATENT_WINDOWS_PER_SIDE) for c in range(LATENT_WINDOWS_PER_SIDE)], adapted_tiles)]
    latent_extent = MOSAIC_TILES_PER_SIDE * 64
    first_blended, latent_weight_sum = _blend_stock_tiles(first_stage, latent_starts, LATENT_TILE_SIZE, latent_extent)
    second_stage = [generate_hybrid_stage(base, item.vector_58, seed, device, stage=1, tile_row=row, tile_col=col, previous=first_blended[:, top:top + 64, left:left + 64]) for (row, col), (top, left), item in zip([(r, c) for r in range(LATENT_WINDOWS_PER_SIDE) for c in range(LATENT_WINDOWS_PER_SIDE)], latent_starts, adapted_tiles)]
    hybrid_mosaic, latent_weight_sum = _blend_stock_tiles(second_stage, latent_starts, LATENT_TILE_SIZE, latent_extent)
    hybrid = torch.stack(second_stage)
    click.echo(f"latent diffusion mosaic: {list(hybrid_mosaic.shape)}")
    for index, name in enumerate(HYBRID_NAMES):
        _save_array(output / f"hybrid_{index}_{name}_normalized.npy", hybrid_mosaic[index])
        _save_scalar_preview(
            output / f"hybrid_{index}_{name}_normalized.png", hybrid_mosaic[index], cmap="coolwarm"
        )
    lowfreq_sqrt_raw = hybrid_mosaic[4] * LOWFREQ_STD + LOWFREQ_MEAN
    _save_array(output / "hybrid_4_lowfreq_signed_sqrt_physical.npy", lowfreq_sqrt_raw)
    # This is the generated physical 240 m stock low-frequency terrain, not
    # an interpolation of the 7.68 km macro DEM.
    lowfreq_m = _get_lowfreq_as_elevation(lowfreq_sqrt_raw)
    if run_hydrology:
        macro_hydrology_crop = macro_elevation_m[
            coarse_row + 1:coarse_row + 1 + MACRO_FOOTPRINT_CELLS,
            coarse_col + 1:coarse_col + 1 + MACRO_FOOTPRINT_CELLS,
        ]
        hydrology_report = run_stock_mosaic_hydrology(
            output,
            macro_elevation_m=macro_hydrology_crop.detach().cpu().numpy(),
            macro_precipitation_mm=macro_physical[4,
                coarse_row + 1:coarse_row + 1 + MACRO_FOOTPRINT_CELLS,
                coarse_col + 1:coarse_col + 1 + MACRO_FOOTPRINT_CELLS,
            ].detach().cpu().numpy(),
            lowfreq_m=lowfreq_m.detach().cpu().numpy(),
        )
        click.echo(f"Hydrology completed on generated 240 m lowfreq terrain: {hydrology_report['planner_shape']}")
    click.echo(f"latent channels to decoder mosaic: {list(hybrid_mosaic[:4].shape)}")

    _release_model(base, device)
    del base
    click.echo(f"Loading stock decoder_model from {stock_model} at {stock_revision}")
    decoder = _load_stock_submodel(
        stock_model, "decoder_model", stock_revision, allow_download
    ).eval().to(device)
    if (decoder.config.in_channels, decoder.config.out_channels) != (5, 1):
        raise click.ClickException("Stock decoder must map residual+4 latents (5) to 1 channel")
    decoder_starts = [(row * DECODER_TILE_STRIDE, col * DECODER_TILE_STRIDE) for row in range(DECODER_WINDOWS_PER_SIDE) for col in range(DECODER_WINDOWS_PER_SIDE)]
    decoder_tiles = [generate_residual(decoder, hybrid_mosaic[:, top // 8:top // 8 + 64, left // 8:left // 8 + 64], seed, device, tile_row=row, tile_col=col, tile_stride=DECODER_TILE_STRIDE) for (row, col), (top, left) in zip([(r, c) for r in range(DECODER_WINDOWS_PER_SIDE) for c in range(DECODER_WINDOWS_PER_SIDE)], decoder_starts)]
    residual_normalized, decoder_weight_sum = _blend_stock_tiles([tile.unsqueeze(0) for tile in decoder_tiles], decoder_starts, DECODER_TILE_SIZE, 3584)
    residual_normalized = residual_normalized[0]
    residual_sqrt = residual_normalized * 0.7
    lowfreq_sqrt = hybrid_mosaic[4] * LOWFREQ_STD + LOWFREQ_MEAN
    elevation_sqrt = None
    elevation_m = None
    # Reconstruction remains stock Laplacian reconstruction, now performed on
    # the normalized blended residual and blended low-frequency latent field.
    residual_sqrt, lowfreq_sqrt, elevation_sqrt, elevation_m = reconstruct_elevation(residual_normalized, hybrid_mosaic, residual_mean=0.0, residual_std=0.7)
    click.echo(f"decoder residual mosaic: {[1, *residual_normalized.shape]}")
    click.echo(f"final DEM: {list(elevation_m.shape)}")
    _save_array(output / "decoder_residual_normalized_3584x3584.npy", residual_normalized)
    _save_array(output / "decoder_residual_signed_sqrt_3584x3584.npy", residual_sqrt)
    _save_array(output / "reconstruction_lowfreq_signed_sqrt_448x448.npy", lowfreq_sqrt)
    _save_array(output / "final_elevation_signed_sqrt_3584x3584.npy", elevation_sqrt)
    _save_array(output / "final_dem_m_3584x3584.npy", elevation_m)
    _save_array(output / "latent_weight_sum_448x448.npy", latent_weight_sum)
    _save_array(output / "decoder_weight_sum_3584x3584.npy", decoder_weight_sum)
    _save_scalar_preview(output / "decoder_weight_sum.png", decoder_weight_sum, cmap="viridis")
    _save_scalar_preview(output / "decoder_residual_signed_sqrt.png", residual_sqrt, cmap="coolwarm")
    _save_scalar_preview(output / "final_dem_m.png", elevation_m)
    _save_hillshade(output / "final_dem_hillshade.png", elevation_m)
    if run_hydrology:
        render_river_overlay_on_30m_hillshade(
            output, elevation_m.detach().cpu().numpy()
        )
    if reconcile_hydrology_30m:
        reconciliation_report = reconcile_stock_mosaic_hydrology_30m(
            output, elevation_m.detach().cpu().numpy()
        )
        click.echo(
            "30 m hydrology reconciliation completed: "
            f"{reconciliation_report['portal_count']} inherited portals"
        )
    _save_hillshade(output / "lowfreq_hillshade.png", _get_lowfreq_as_elevation(lowfreq_sqrt))
    final_dem_diagnostics = _final_dem_diagnostics(elevation_m)
    (output / "final_dem_diagnostics.json").write_text(
        json.dumps(final_dem_diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    click.echo(
        "Final DEM diagnostics: "
        + ", ".join(f"{name}={value:.4g}" for name, value in final_dem_diagnostics.items())
    )
    wrote_tiff = _write_tiff_if_available(output / "final_dem_m_3584x3584.tif", elevation_m)
    _release_model(decoder, device)
    del decoder

    metadata = {
        "purpose": "mechanical architecture transplant smoke test; not a quality evaluation",
        "seed": int(seed),
        "requested_land_fraction": float(land_fraction),
        "macro_steps": int(macro_steps),
        "device": device,
        "macro_checkpoint": str(macro_model.resolve()),
        "macro_checkpoint_sha256": _sha256(macro_model / "diffusion_pytorch_model.safetensors"),
        "macro_stats_file": str(macro_stats.resolve()),
        "macro_means": macro_means,
        "macro_stds": macro_stds,
        "macro_channel_names": MACRO_CHANNEL_NAMES,
        "coarse_patch_top_left": [int(coarse_row), int(coarse_col)],
        "choose_mountain_crop": bool(choose_mountain_crop),
        "run_hydrology": bool(run_hydrology),
        "reconcile_hydrology_30m": bool(reconcile_hydrology_30m),
        "mountain_crop_score": mountain_crop,
        "stock_model": stock_model,
        "stock_revision": stock_revision,
        "stock_conditioning_mean": STOCK_CONDITIONING_MEAN.tolist(),
        "stock_conditioning_std": STOCK_CONDITIONING_STD.tolist(),
        "stock_histogram_raw": [0.0] * 5,
        "stock_conditioning_noise_level": 0.0,
        "hybrid_channel_names": HYBRID_NAMES,
        "lowfreq_mean": LOWFREQ_MEAN,
        "lowfreq_std": LOWFREQ_STD,
        "residual_mean": 0.0,
        "residual_std": 0.7,
        "mosaic_tiles_per_side": MOSAIC_TILES_PER_SIDE,
        "latent_tile_size": LATENT_TILE_SIZE,
        "latent_tile_stride": LATENT_TILE_STRIDE,
        "latent_overlap": LATENT_TILE_SIZE - LATENT_TILE_STRIDE,
        "latent_windows_per_side": LATENT_WINDOWS_PER_SIDE,
        "decoder_tile_size": DECODER_TILE_SIZE,
        "decoder_tile_stride": DECODER_TILE_STRIDE,
        "decoder_overlap": DECODER_TILE_SIZE - DECODER_TILE_STRIDE,
        "decoder_windows_per_side": DECODER_WINDOWS_PER_SIDE,
        "weighting": "WorldPipeline linear_weight_window: separable linear ramps with eps=1e-3; sum weighted tiles then divide by summed weights",
        "macro_context_cells": MACRO_CONTEXT_CELLS,
        "macro_output_footprint_cells": MACRO_FOOTPRINT_CELLS,
        "output_extent_km": MOSAIC_TILES_PER_SIDE * STOCK_TILE_SIZE_PX * NATIVE_RESOLUTION_M / 1000,
        "latent_compression": 8,
        "native_resolution_m": NATIVE_RESOLUTION_M,
        "shapes": {
            "macro": list(macro_physical.shape),
            "conditioning_patches": [len(adapted_tiles), *list(adapted.physical_with_mask_7x4x4.shape)],
            "conditioning_vectors": [len(adapted_tiles), *list(adapted.vector_58.shape)],
            "hybrid_tiles": list(hybrid.shape),
            "hybrid_mosaic": list(hybrid_mosaic.shape),
            "decoder_latent_mosaic": list(hybrid_mosaic[:4].shape),
            "decoder_residual": [1, *residual_normalized.shape],
            "final_dem": list(elevation_m.shape),
        },
        "wrote_float32_tiff": wrote_tiff,
        "final_dem_diagnostics": final_dem_diagnostics,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    click.echo(f"Saved smoke-test artifacts to {output.resolve()}")
