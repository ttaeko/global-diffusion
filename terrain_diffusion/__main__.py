"""CLI for the clean global-diffusion transplant."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from safetensors import safe_open

from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.transplant.paired_alps import paired_alps_diagnostic
from terrain_diffusion.transplant.smoke import (
    DEFAULT_MACRO_MODEL,
    DEFAULT_MACRO_STATS,
    DEFAULT_STOCK_MODEL,
    DEFAULT_STOCK_REVISION,
    smoke_transplant,
)
from terrain_diffusion.inference.full_10m_pipeline import sample_full_10m


@click.group()
def main():
    """Mechanical 7.68 km -> stock 240 m -> stock 30 m proof of concept."""


@main.command("verify-transplant")
def verify_transplant():
    """Perform static path/config/import checks without loading weights or sampling."""
    failures = []
    for label, path in (
        ("macro checkpoint", DEFAULT_MACRO_MODEL),
        ("macro stats", DEFAULT_MACRO_STATS),
    ):
        if not path.exists():
            failures.append(f"Missing {label}: {path}")

    cache_snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--xandergos--terrain-diffusion-30m/snapshots"
        / DEFAULT_STOCK_REVISION
    )
    expected = (
        cache_snapshot / "base_model/config.json",
        cache_snapshot / "base_model/diffusion_pytorch_model.safetensors",
        cache_snapshot / "decoder_model/config.json",
        cache_snapshot / "decoder_model/diffusion_pytorch_model.safetensors",
    )
    missing_cache = [str(path) for path in expected if not path.exists()]
    topology_checks = {}
    model_directories = {
        "macro": DEFAULT_MACRO_MODEL,
        "stock_base": cache_snapshot / "base_model",
        "stock_decoder": cache_snapshot / "decoder_model",
    }
    if not failures and not missing_cache:
        for name, directory in model_directories.items():
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            config = {key: value for key, value in config.items() if not key.startswith("_")}
            with torch.device("meta"):
                model = EDMUnet2D(**config)
            configured_keys = set(model.state_dict())
            with safe_open(
                directory / "diffusion_pytorch_model.safetensors",
                framework="pt",
                device="cpu",
            ) as weights:
                checkpoint_keys = set(weights.keys())
            if configured_keys != checkpoint_keys:
                failures.append(f"{name} config/weight tensor keys do not match")
            topology_checks[name] = {
                "tensor_count": len(checkpoint_keys),
                "config_matches_weights": configured_keys == checkpoint_keys,
            }
    report = {
        "macro_checkpoint": str(DEFAULT_MACRO_MODEL),
        "macro_stats": str(DEFAULT_MACRO_STATS),
        "stock_model": DEFAULT_STOCK_MODEL,
        "stock_revision": DEFAULT_STOCK_REVISION,
        "stock_cache_snapshot": str(cache_snapshot),
        "stock_cache_complete_for_smoke": not missing_cache,
        "missing_stock_cache_files": missing_cache,
        "topology_checks": topology_checks,
        "imports": "ok",
    }
    click.echo(json.dumps(report, indent=2))
    if failures:
        raise click.ClickException("; ".join(failures))


main.add_command(smoke_transplant)
main.add_command(sample_full_10m)
main.add_command(paired_alps_diagnostic)


if __name__ == "__main__":
    main()
