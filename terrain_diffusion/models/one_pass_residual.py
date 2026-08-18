import torch
import torch.nn.functional as F

def block_mean_3x3(x):
    b, c, h, w = x.shape
    x = x.reshape(
        b, c,
        h // 3, 3,
        w // 3, 3,
    )
    return x.mean(dim=(3, 5))

def smooth_exact_upsample_3x(parent):
    # 1. Smoothly interpolate 30 m -> 10 m
    smooth = F.interpolate(
        parent,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )

    # 2. Find what each 3x3 block currently averages to
    smooth_mean = block_mean_3x3(smooth)

    # 3. Find the error relative to the authoritative 30 m parent
    correction = parent - smooth_mean

    # 4. Apply the same correction to all 9 children of each parent cell
    correction_up = correction.repeat_interleave(3, dim=-2)
    correction_up = correction_up.repeat_interleave(3, dim=-1)

    # 5. Smooth surface, but with exact parent-cell means
    return smooth + correction_up

def project_zero_block_mean_3x(residual):
    mean = block_mean_3x3(residual)

    mean_up = mean.repeat_interleave(3, dim=-2)
    mean_up = mean_up.repeat_interleave(3, dim=-1)

    return residual - mean_up

def reconstruct_10m(parent_30m, raw_residual):
    return 0

parent = torch.randn(2, 1, 256, 256)
residual = torch.randn(2, 1, 768, 768)
