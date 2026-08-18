from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LightSource

from terrain_diffusion.models.one_pass_residual import (
    block_mean_3x3,
    smooth_exact_upsample_3x,
)


def hillshade(elevation_m):
    ls = LightSource(
        azdeg=315,
        altdeg=45,
    )

    return ls.hillshade(
        elevation_m,
        vert_exag=1.0,
        dx=10.0,
        dy=10.0,
    )


@torch.no_grad()
def render_validation_sample(
    model,
    sample,
    device,
    residual_std_m,
    output_path,
    title=None,
):
    model.eval()

    conditioning = (
        sample["conditioning"]
        .unsqueeze(0)
        .to(device)
    )

    parent_30m = (
        sample["parent_30m"]
        .unsqueeze(0)
        .to(device)
    )

    target_10m = (
        sample["target_10m"]
        .unsqueeze(0)
        .to(device)
    )

    # ---------------------------------------------------------
    # Predict normalized residual
    # ---------------------------------------------------------

    prediction_norm = model(conditioning)

    # Hard parent-preservation constraint
    from terrain_diffusion.models.one_pass_residual import (
        project_zero_block_mean_3x,
    )

    prediction_norm = project_zero_block_mean_3x(
        prediction_norm
    )

    prediction_m = (
        prediction_norm * residual_std_m
    )

    # ---------------------------------------------------------
    # Reconstruct terrain
    # ---------------------------------------------------------

    base_10m = smooth_exact_upsample_3x(
        parent_30m
    )

    prediction_10m = (
        base_10m + prediction_m
    )

    # ---------------------------------------------------------
    # Verify physical contract
    # ---------------------------------------------------------

    recovered_parent = block_mean_3x3(
        prediction_10m
    )

    parent_error = (
        recovered_parent - parent_30m
    ).abs().max().item()

    # ---------------------------------------------------------
    # Move to numpy
    # ---------------------------------------------------------

    base_np = (
        base_10m[0, 0]
        .cpu()
        .numpy()
    )

    pred_np = (
        prediction_10m[0, 0]
        .cpu()
        .numpy()
    )

    target_np = (
        target_10m[0, 0]
        .cpu()
        .numpy()
    )

    error_np = pred_np - target_np

    mae = np.mean(np.abs(error_np))
    rmse = np.sqrt(np.mean(error_np ** 2))

    # Same illumination for all terrain panels
    base_hs = hillshade(base_np)
    pred_hs = hillshade(pred_np)
    target_hs = hillshade(target_np)

    error_limit = max(
        np.percentile(np.abs(error_np), 99),
        0.1,
    )

    # ---------------------------------------------------------
    # Render
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 12),
    )

    axes[0, 0].imshow(
        base_hs,
        cmap="gray",
    )
    axes[0, 0].set_title(
        "30 m parent synthesis"
    )

    axes[0, 1].imshow(
        target_hs,
        cmap="gray",
    )
    axes[0, 1].set_title(
        "True 10 m terrain"
    )

    axes[1, 0].imshow(
        pred_hs,
        cmap="gray",
    )
    axes[1, 0].set_title(
        f"Predicted 10 m terrain\n"
        f"MAE={mae:.3f} m  RMSE={rmse:.3f} m"
    )

    im = axes[1, 1].imshow(
        error_np,
        cmap="RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )
    axes[1, 1].set_title(
        "Prediction error [m]\n"
        f"99% scale ±{error_limit:.2f} m"
    )

    fig.colorbar(
        im,
        ax=axes[1, 1],
        shrink=0.8,
        label="metres",
    )

    for ax in axes.flat:
        ax.axis("off")

    if title is not None:
        fig.suptitle(
            title,
            fontsize=14,
        )

    fig.text(
        0.5,
        0.01,
        (
            f"Parent preservation max error: "
            f"{parent_error:.6f} m"
        ),
        ha="center",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout(
        rect=(0, 0.025, 1, 0.97)
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)