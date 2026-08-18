import torch
import torch.nn.functional as F


def gradient_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    L1 loss on first spatial derivatives.

    prediction, target:
        [B, C, H, W]
    """

    # Horizontal finite differences
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]

    # Vertical finite differences
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]

    target_dy = target[..., 1:, :] - target[..., :-1, :]

    loss_x = F.l1_loss(pred_dx, target_dx)
    loss_y = F.l1_loss(pred_dy, target_dy)

    return 0.5 * (loss_x + loss_y)