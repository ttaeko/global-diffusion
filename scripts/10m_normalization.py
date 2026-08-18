import torch

from terrain_diffusion.training.datasets.one_pass_dataset import (
    Terrain10mDataset,
)


CHANNEL_NAMES = [
    "smooth_parent",
    "repeated_parent",
    "grad_x",
    "grad_y",
    "tpi_90m",
    "tpi_270m",
    "relief_90m",
    "relief_270m",
    "roughness_90m",
    "roughness_270m",
    "phase_x",
    "phase_y",
    "curvature",
    "log_accumulation",
    "log_discharge",
    "channel_proximity",
    "flow_east",
    "flow_south",
    "stream_order",
    "lake_mask",
]


def main():
    torch.manual_seed(0)

    dataset = Terrain10mDataset(
        "data/alps_curriculum_10m_hydro_v4.h5",
        split="train",
        crop_size_30m=128,
    )

    # float64 accumulators for accurate statistics
    channel_sum = torch.zeros(20, dtype=torch.float64)
    channel_sumsq = torch.zeros(20, dtype=torch.float64)
    channel_min = torch.full(
        (20,),
        float("inf"),
        dtype=torch.float64,
    )
    channel_max = torch.full(
        (20,),
        float("-inf"),
        dtype=torch.float64,
    )

    target_sum = 0.0
    target_sumsq = 0.0
    target_min = float("inf")
    target_max = float("-inf")

    channel_count = 0
    target_count = 0

    for i in range(len(dataset)):
        sample = dataset[i]

        x = sample["conditioning"].double()
        y = sample["residual_target"].double()

        # x: [20, H, W]
        channel_sum += x.sum(dim=(1, 2))
        channel_sumsq += (x * x).sum(dim=(1, 2))

        channel_min = torch.minimum(
            channel_min,
            x.amin(dim=(1, 2)),
        )
        channel_max = torch.maximum(
            channel_max,
            x.amax(dim=(1, 2)),
        )

        channel_count += x.shape[1] * x.shape[2]

        # y: [1, H, W]
        target_sum += y.sum().item()
        target_sumsq += (y * y).sum().item()
        target_min = min(target_min, y.min().item())
        target_max = max(target_max, y.max().item())
        target_count += y.numel()

        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(dataset)}")

    # ---------------------------------------------------------
    # Conditioning statistics
    # ---------------------------------------------------------

    means = channel_sum / channel_count

    variances = (
        channel_sumsq / channel_count
        - means.square()
    )

    stds = torch.sqrt(
        torch.clamp(variances, min=0.0)
    )

    print("\nCONDITIONING STATISTICS\n")

    for i, name in enumerate(CHANNEL_NAMES):
        print(
            f"{i:2d} {name:22s} "
            f"mean={means[i]:12.6f} "
            f"std={stds[i]:12.6f} "
            f"min={channel_min[i]:12.6f} "
            f"max={channel_max[i]:12.6f}"
        )

    # ---------------------------------------------------------
    # Residual target statistics
    # ---------------------------------------------------------

    target_mean = target_sum / target_count

    target_var = (
        target_sumsq / target_count
        - target_mean**2
    )

    target_std = target_var**0.5

    print("\nRESIDUAL TARGET\n")
    print("mean:", target_mean)
    print("std: ", target_std)
    print("min: ", target_min)
    print("max: ", target_max)


if __name__ == "__main__":
    main()