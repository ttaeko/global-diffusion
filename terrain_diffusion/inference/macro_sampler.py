"""Standalone sampler for inspecting a trained macro-geography model."""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from terrain_diffusion.inference.portable_rng import standard_normal
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.training.datasets.h5_macro_terrain_dataset import (
    DEFAULT_MACRO_MEANS,
    DEFAULT_MACRO_STDS,
    load_macro_stats,
)


@click.command("sample-macro")
@click.argument("model_path")
@click.option("--output", "-o", default="macro_sample", show_default=True)
@click.option("--seed", default=74, show_default=True, type=int)
@click.option("--size", default=256, show_default=True, type=int)
@click.option("--steps", default=30, show_default=True, type=int)
@click.option("--device", default=None, help="Defaults to CUDA when available, otherwise CPU.")
@click.option(
    "--land-fraction", default=0.6, show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Approximate requested land coverage for a conditioned macro model.",
)
@click.option(
    "--stats-file",
    type=click.Path(exists=True, dir_okay=False),
    help="macro_terrain.h5 used for training. Legacy defaults are used when omitted.",
)
def sample_macro(model_path, output, seed, size, steps, device, land_fraction, stats_file):
    """Sample MODEL_PATH and save its six coarse channels plus a relief preview."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"
    click.echo(f"Sampling on device: {device}")
    model = EDMUnet2D.from_pretrained(model_path).eval().to(device)
    scheduler = EDMDPMSolverMultistepScheduler(
        sigma_min=0.002, sigma_max=80, sigma_data=0.5
    )
    scheduler.set_timesteps(steps)

    noise = standard_normal(seed, (1, 6, size, size), dtype=np.float32)
    sample = torch.from_numpy(noise).to(device) * scheduler.sigmas[0]
    conditional_count = len(model.conditional_layers)
    if conditional_count == 1:
        conditional_inputs = [
            torch.tensor([land_fraction * 2.0 - 1.0], device=device, dtype=sample.dtype)
        ]
    elif conditional_count == 0:
        conditional_inputs = []
    else:
        raise click.ClickException(
            f"Macro sampler supports zero or one conditional input, model has {conditional_count}"
        )
    with torch.no_grad():
        for timestep, sigma in zip(scheduler.timesteps, scheduler.sigmas):
            timestep = timestep.to(device)
            sigma = sigma.to(device)
            scaled = scheduler.precondition_inputs(sample, sigma)
            label = scheduler.trigflow_precondition_noise(sigma.view(-1)).to(device)
            output_tensor = model(
                scaled, noise_labels=label, conditional_inputs=conditional_inputs
            )
            sample = scheduler.step(output_tensor, timestep, sample).prev_sample

    normalized = sample[0].cpu().float() / scheduler.config.sigma_data
    if not torch.isfinite(normalized).all():
        raise click.ClickException("Macro sampling produced non-finite values")
    if stats_file:
        macro_means, macro_stds = load_macro_stats(stats_file)
    else:
        macro_means, macro_stds = DEFAULT_MACRO_MEANS, DEFAULT_MACRO_STDS
    means = torch.tensor(macro_means).view(6, 1, 1)
    stds = torch.tensor(macro_stds).view(6, 1, 1)
    channels = normalized * stds + means
    channels[1] = channels[0] - channels[1]

    elev_sqrt = channels[0].numpy()
    elevation_m = np.sign(elev_sqrt) * np.square(elev_sqrt)
    generated_land_fraction = float(np.mean(elevation_m > 0.0))
    lag1_correlation = float(
        np.corrcoef(elevation_m[:, :-1].ravel(), elevation_m[:, 1:].ravel())[0, 1]
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        coarse=channels.numpy(),
        elevation_m=elevation_m,
        seed=np.int64(seed),
        requested_land_fraction=np.float32(land_fraction),
        generated_land_fraction=np.float32(generated_land_fraction),
        lag1_correlation=np.float32(lag1_correlation),
    )
    plt.imsave(output_path.with_suffix(".png"), elevation_m, cmap="terrain", vmin=-1000, vmax=5000)
    click.echo(
        f"Generated land fraction: {generated_land_fraction:.3f}; "
        f"elevation lag-1 correlation: {lag1_correlation:.3f}"
    )
    click.echo(f"Saved {output_path.with_suffix('.npz')} and {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    sample_macro()
