import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from terrain_diffusion.models.one_pass_residual import (
    block_mean_3x3,
    project_zero_block_mean_3x,
    smooth_exact_upsample_3x,
)
from terrain_diffusion.models.one_pass_unet import (
    OnePassUNet,
)
from terrain_diffusion.training.datasets.one_pass_dataset import (
    RESIDUAL_STD_M,
    Terrain10mDataset,
)
from terrain_diffusion.eval.render_10m import (
    render_validation_sample,
)

from terrain_diffusion.training.losses import (
    gradient_l1_loss,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="runs/one_pass_10m_candidate_a",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--crop-size-30m",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--render-count",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--gradient-weight",
        type=float,
        default=0.0,
        help="Weight of residual gradient L1 loss.",
    )

    return parser.parse_args()


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    best_val_loss,
    args,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "args": vars(args),
        "residual_std_m": RESIDUAL_STD_M,
    }

    # Write atomically so an interrupted save does
    # not destroy the previous checkpoint.
    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        state,
        temp_path,
    )

    os.replace(
        temp_path,
        path,
    )


def validate(
    model,
    loader,
    device,
    criterion,
    max_batches=None,
):
    model.eval()

    loss_sum = 0.0
    mae_m_sum = 0.0
    rmse_m_sum = 0.0
    parent_error_max = 0.0

    sample_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            conditioning = (
                batch["conditioning"]
                .to(device)
            )

            target_norm = (
                batch["residual_target"]
                .to(device)
            )

            target_10m = (
                batch["target_10m"]
                .to(device)
            )

            parent_30m = (
                batch["parent_30m"]
                .to(device)
            )

            prediction_norm = model(
                conditioning
            )

            prediction_norm = (
                project_zero_block_mean_3x(
                    prediction_norm
                )
            )

            loss = criterion(
                prediction_norm,
                target_norm,
            )

            prediction_m = (
                prediction_norm
                * RESIDUAL_STD_M
            )

            base_10m = (
                smooth_exact_upsample_3x(
                    parent_30m
                )
            )

            prediction_10m = (
                base_10m
                + prediction_m
            )

            error_m = (
                prediction_10m
                - target_10m
            )

            batch_size = conditioning.shape[0]

            mae_m = (
                error_m.abs()
                .mean(dim=(1, 2, 3))
            )

            rmse_m = torch.sqrt(
                (error_m ** 2)
                .mean(dim=(1, 2, 3))
            )

            recovered_parent = (
                block_mean_3x3(
                    prediction_10m
                )
            )

            parent_error = (
                recovered_parent
                - parent_30m
            ).abs().max().item()

            parent_error_max = max(
                parent_error_max,
                parent_error,
            )

            loss_sum += (
                loss.item()
                * batch_size
            )

            mae_m_sum += (
                mae_m.sum().item()
            )

            rmse_m_sum += (
                rmse_m.sum().item()
            )

            sample_count += batch_size

    return {
        "loss": loss_sum / sample_count,
        "mae_m": mae_m_sum / sample_count,
        "rmse_m": rmse_m_sum / sample_count,
        "parent_error_max_m": parent_error_max,
    }


def main():
    args = parse_args()

    set_seed(args.seed)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=2,
        )

    device = choose_device()

    print("Device:", device)

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    train_dataset = Terrain10mDataset(
        args.data,
        split="train",
        crop_size_30m=args.crop_size_30m,
    )

    val_dataset = Terrain10mDataset(
        args.data,
        split="val",
        crop_size_30m=args.crop_size_30m,
    )

    print(
        "Train samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = OnePassUNet(
        in_channels=20,
        out_channels=1,
        base_channels=48,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    criterion = nn.L1Loss()

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        global_step = checkpoint[
            "global_step"
        ]

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf"),
        )

        print(
            "Resumed from:",
            args.resume,
        )

        print(
            "Starting epoch:",
            start_epoch,
        )

    # ---------------------------------------------------------
    # Fixed validation samples for renders
    # ---------------------------------------------------------

    render_indices = []

    if len(val_dataset) > 0:
        if args.render_count == 1:
            render_indices = [0]

        else:
            render_indices = np.linspace(
                0,
                len(val_dataset) - 1,
                min(
                    args.render_count,
                    len(val_dataset),
                ),
                dtype=int,
            ).tolist()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(
        start_epoch,
        args.epochs,
    ):
        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        train_l1_sum = 0.0
        train_gradient_sum = 0.0

        for batch_index, batch in enumerate(
            train_loader
        ):
            if (
                args.max_train_batches is not None
                and batch_index
                >= args.max_train_batches
            ):
                break

            conditioning = (
                batch["conditioning"]
                .to(device)
            )

            target = (
                batch["residual_target"]
                .to(device)
            )

            optimizer.zero_grad()

            prediction = model(
                conditioning
            )

            # Physical constraint first.
            prediction = project_zero_block_mean_3x(
                prediction
            )

            # Ordinary residual accuracy
            loss_l1 = criterion(
                prediction,
                target,
            )

            # Local slope / sharpness accuracy
            loss_gradient = gradient_l1_loss(
                prediction,
                target,
            )

            # Candidate A:
            #   gradient_weight = 0
            #
            # Candidate B:
            #   gradient_weight > 0
            loss = (
                loss_l1
                + args.gradient_weight * loss_gradient
            )

            loss.backward()
            optimizer.step()

            batch_size = conditioning.shape[0]

            train_loss_sum += (
                loss.item() * batch_size
            )

            train_l1_sum += (
                loss_l1.item() * batch_size
            )

            train_gradient_sum += (
                loss_gradient.item() * batch_size
            )

            train_samples += batch_size
            global_step += 1

            weighted_gradient = (
                args.gradient_weight
                * loss_gradient.item()
            )

            if batch_index % 5 == 0:
                print(
                    f"epoch={epoch:03d} "
                    f"batch={batch_index:04d} "
                    f"step={global_step:07d} "
                    f"loss={loss.item():.6f} "
                    f"l1={loss_l1.item():.6f} "
                    f"grad={loss_gradient.item():.6f}"
                    f"weighted_grad={weighted_gradient:.6f}"
                )

            train_loss = (
                train_loss_sum / train_samples
            )

            train_l1 = (
                train_l1_sum / train_samples
            )

            train_gradient = (
                train_gradient_sum / train_samples
            )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        metrics = validate(
            model,
            val_loader,
            device,
            criterion,
            max_batches=args.max_val_batches,
        )

        print()
        print(
            f"EPOCH {epoch:03d}"
        )
        print(
            f"  train loss:       "
            f"{train_loss:.6f}"
        )
        print(
            f"  val loss:         "
            f"{metrics['loss']:.6f}"
        )
        print(
            f"  val terrain MAE:  "
            f"{metrics['mae_m']:.4f} m"
        )
        print(
            f"  val terrain RMSE: "
            f"{metrics['rmse_m']:.4f} m"
        )
        print(
            f"  parent max error: "
            f"{metrics['parent_error_max_m']:.6f} m"
        )
        print()

        # -----------------------------------------------------
        # Metric log
        # -----------------------------------------------------

        record = {
            "epoch": epoch,
            "global_step": global_step,

            "train_loss": train_loss,
            "train_l1": train_l1,
            "train_gradient": train_gradient,

            "val_loss": metrics["loss"],
            "val_mae_m": metrics["mae_m"],
            "val_rmse_m": metrics["rmse_m"],
            "parent_error_max_m": (
                metrics["parent_error_max_m"]
            ),
        }

        with open(
            output_dir / "metrics.jsonl",
            "a",
        ) as f:
            f.write(
                json.dumps(record)
                + "\n"
            )

        # -----------------------------------------------------
        # Always save latest
        # -----------------------------------------------------

        save_checkpoint(
            output_dir
            / "checkpoints"
            / "latest.pt",
            model,
            optimizer,
            epoch,
            global_step,
            best_val_loss,
            args,
        )

        # -----------------------------------------------------
        # Periodic checkpoint
        # -----------------------------------------------------

        if (
            (epoch + 1)
            % args.save_every
            == 0
        ):
            save_checkpoint(
                output_dir
                / "checkpoints"
                / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_val_loss,
                args,
            )

        # -----------------------------------------------------
        # Best checkpoint
        # -----------------------------------------------------

        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]

            save_checkpoint(
                output_dir
                / "checkpoints"
                / "best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_val_loss,
                args,
            )

            print(
                "New best checkpoint."
            )

        # -----------------------------------------------------
        # Fixed validation renders
        # -----------------------------------------------------

        if (
            epoch % args.render_every
            == 0
        ):
            for render_number, index in enumerate(
                render_indices
            ):
                sample = val_dataset[index]

                render_validation_sample(
                    model=model,
                    sample=sample,
                    device=device,
                    residual_std_m=RESIDUAL_STD_M,
                    output_path=(
                        output_dir
                        / "renders"
                        / f"epoch_{epoch:04d}"
                        / (
                            f"sample_"
                            f"{render_number:02d}.png"
                        )
                    ),
                    title=(
                        f"Candidate A — "
                        f"epoch {epoch}"
                    ),
                )


if __name__ == "__main__":
    main()