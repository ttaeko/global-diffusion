import torch
import torch.nn.functional as F

from terrain_diffusion.models.one_pass_residual import smooth_exact_upsample_3x


def repeat_upsample_3x(x):
    x = x.repeat_interleave(3, dim=-2)
    x = x.repeat_interleave(3, dim=-1)
    return x

def gradient_xy_30m(parent):
    """
    Compute centered finite-difference gradients on the 30 m parent DEM.

    Input:
        parent: [B, 1, H, W]

    Returns:
        grad_x: [B, 1, H, W]
        grad_y: [B, 1, H, W]
    """

    # Replicate the edge pixels so the output remains H x W
    padded = F.pad(
        parent,
        pad=(1, 1, 1, 1),
        mode="replicate",
    )

    # Central difference:
    # dz/dx = (right - left) / (2 * 30 m)
    grad_x = (
        padded[:, :, 1:-1, 2:]
        - padded[:, :, 1:-1, :-2]
    ) / 60.0

    # dz/dy = (bottom - top) / (2 * 30 m)
    grad_y = (
        padded[:, :, 2:, 1:-1]
        - padded[:, :, :-2, 1:-1]
    ) / 60.0

    return grad_x, grad_y

def local_mean(x, kernel_size):
    """
    Local mean with replicate padding.

    Input:
        x: [B, C, H, W]

    Output:
        [B, C, H, W]
    """

    padding = kernel_size // 2

    padded = F.pad(
        x,
        pad=(padding, padding, padding, padding),
        mode="replicate",
    )

    return F.avg_pool2d(
        padded,
        kernel_size=kernel_size,
        stride=1,
    )


def tpi(parent, kernel_size):
    """
    Topographic Position Index:

        TPI = elevation - local mean elevation
    """

    return parent - local_mean(parent, kernel_size)

def local_max(x, kernel_size):
    padding = kernel_size // 2

    padded = F.pad(
        x,
        pad=(padding, padding, padding, padding),
        mode="replicate",
    )

    return F.max_pool2d(
        padded,
        kernel_size=kernel_size,
        stride=1,
    )


def local_min(x, kernel_size):
    padding = kernel_size // 2

    padded = F.pad(
        x,
        pad=(padding, padding, padding, padding),
        mode="replicate",
    )

    # PyTorch has max pooling but no ordinary min pooling.
    # min(x) = -max(-x)
    return -F.max_pool2d(
        -padded,
        kernel_size=kernel_size,
        stride=1,
    )


def local_relief(x, kernel_size):
    """
    Local elevation range:

        relief = max(z) - min(z)
    """

    maximum = local_max(x, kernel_size)
    minimum = local_min(x, kernel_size)

    return maximum - minimum

def local_roughness(x, kernel_size):
    """
    Local standard deviation of elevation.

        std = sqrt(E[x^2] - E[x]^2)
    """

    mean = local_mean(x, kernel_size)
    mean_squared = local_mean(x * x, kernel_size)

    variance = mean_squared - mean * mean

    # Floating point rounding can occasionally produce tiny
    # negative values such as -1e-7.
    variance = torch.clamp(variance, min=0.0)

    return torch.sqrt(variance)

def child_phase_3x(parent_30m):
    """
    Build x/y position channels for each 10 m child inside
    its corresponding 3x3 parent footprint.

    Input:
        parent_30m: [B, 1, H, W]

    Returns:
        phase_x: [B, 1, 3H, 3W]
        phase_y: [B, 1, 3H, 3W]
    """

    b, _, h, w = parent_30m.shape

    h10 = h * 3
    w10 = w * 3

    # Repeating sequence:
    # 0, 1, 2, 0, 1, 2, ...
    # shifted to:
    # -1, 0, 1, -1, 0, 1, ...
    x = (
        torch.arange(
            w10,
            device=parent_30m.device,
            dtype=parent_30m.dtype,
        ) % 3
    ) - 1

    y = (
        torch.arange(
            h10,
            device=parent_30m.device,
            dtype=parent_30m.dtype,
        ) % 3
    ) - 1

    # [W] -> [1, 1, 1, W] -> [B, 1, H, W]
    phase_x = x.view(1, 1, 1, w10).expand(
        b, 1, h10, w10
    )

    # [H] -> [1, 1, H, 1] -> [B, 1, H, W]
    phase_y = y.view(1, 1, h10, 1).expand(
        b, 1, h10, w10
    )

    return phase_x, phase_y

def laplacian_curvature_30m(parent):
    """
    Discrete Laplacian of the 30 m parent DEM.

    Input:
        parent: [B, 1, H, W], elevation in metres

    Returns:
        curvature: [B, 1, H, W]

    Units are approximately 1 / metre.
    """

    padded = F.pad(
        parent,
        pad=(1, 1, 1, 1),
        mode="replicate",
    )

    center = padded[:, :, 1:-1, 1:-1]

    left = padded[:, :, 1:-1, :-2]
    right = padded[:, :, 1:-1, 2:]

    top = padded[:, :, :-2, 1:-1]
    bottom = padded[:, :, 2:, 1:-1]

    dx = 30.0

    curvature = (
        left
        + right
        + top
        + bottom
        - 4.0 * center
    ) / (dx * dx)

    return curvature


def build_10m_terrain_conditioning(parent_30m):
    # 1: smooth exact 10 m parent
    smooth_parent_10m = smooth_exact_upsample_3x(parent_30m)

    # 2: repeated authoritative parent
    repeated_parent_10m = repeat_upsample_3x(parent_30m)

    # 3-4: gradients
    grad_x_30m, grad_y_30m = gradient_xy_30m(parent_30m)

    grad_x_10m = F.interpolate(
        grad_x_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )

    grad_y_10m = F.interpolate(
        grad_y_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )

    # 5-6: multiscale TPI
    tpi_90m_30m = tpi(parent_30m, kernel_size=3)
    tpi_270m_30m = tpi(parent_30m, kernel_size=9)

    tpi_90m_10m = F.interpolate(
        tpi_90m_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )

    tpi_270m_10m = F.interpolate(
        tpi_270m_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )

    # 7-8: local relief
    relief_90m_30m = local_relief(
        parent_30m,
        kernel_size=3,
    )

    relief_270m_30m = local_relief(
        parent_30m,
        kernel_size=9,
    )

    relief_90m_10m = F.interpolate(
        relief_90m_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )
    
    relief_270m_10m = F.interpolate(
        relief_270m_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
     )

    # 9-10: local roughness
    roughness_90m_30m = local_roughness(
        parent_30m,
        kernel_size=3,
    )

    roughness_270m_30m = local_roughness(
        parent_30m,
        kernel_size=9,
    )

    roughness_90m_10m = F.interpolate(
        roughness_90m_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )

    roughness_270m_10m = F.interpolate(
        roughness_270m_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )    

    # 11-12: position inside each 3x3 parent footprint
    phase_x_10m, phase_y_10m = child_phase_3x(parent_30m)

    # 13: Laplacian curvature
    curvature_30m = laplacian_curvature_30m(parent_30m)

    curvature_10m = F.interpolate(
        curvature_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )


    condition = torch.cat(
    [
        smooth_parent_10m,       # 1
        repeated_parent_10m,     # 2

        grad_x_10m,              # 3
        grad_y_10m,              # 4

        tpi_90m_10m,             # 5
        tpi_270m_10m,            # 6

        relief_90m_10m,          # 7
        relief_270m_10m,         # 8

        roughness_90m_10m,       # 9
        roughness_270m_10m,      # 10

        phase_x_10m,             # 11
        phase_y_10m,             # 12

        curvature_10m,           # 13
    ],
    dim=1,
    )

    return condition


def build_10m_hydrology_conditioning(hydrology_30m):

    """
    Convert canonical 30 m hydrology fields into 10 m
    conditioning channels.

    Returns:
        [B, 7, 3H, 3W]
    """

    channel_mask_30m = hydrology_30m["channel_mask"]
    accumulation_30m = hydrology_30m["flow_accumulation"]
    discharge_30m = hydrology_30m["discharge"]
    flow_x_30m = hydrology_30m["flow_dir_x"]
    flow_y_30m = hydrology_30m["flow_dir_y"]
    stream_order_30m = hydrology_30m["stream_order"]
    lake_mask_30m = hydrology_30m["lake_mask"]

    # utils
    log_accumulation_30m = torch.log1p(
        torch.clamp(accumulation_30m, min=0.0)
    )

    log_discharge_30m = torch.log1p(
        torch.clamp(discharge_30m, min=0.0)
    )

    # masks
    channel_mask_10m = F.interpolate(
        channel_mask_30m.float(),
        scale_factor=3,
        mode="nearest",
    )

    lake_mask_10m = F.interpolate(
        lake_mask_30m.float(),
        scale_factor=3,
        mode="nearest",
    )

    # continous fields
    accumulation_10m = F.interpolate(
        log_accumulation_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )

    discharge_10m = F.interpolate(
        log_discharge_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )

    # flow direction
    flow_x_10m = F.interpolate(
        flow_x_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )

    flow_y_10m = F.interpolate(
        flow_y_30m,
        scale_factor=3,
        mode="bilinear",
        align_corners=False,
    )

    # stream order
    stream_order_10m = F.interpolate(
        stream_order_30m.float(),
        scale_factor=3,
        mode="nearest",
    )

    condition = torch.cat(
        [
            channel_mask_10m,    # 1
            accumulation_10m,    # 2
            discharge_10m,       # 3
            flow_x_10m,          # 4
            flow_y_10m,          # 5
            stream_order_10m,    # 6
            lake_mask_10m,       # 7
        ],
        dim=1,
    )

    return condition


# final conditioning builder
def build_10m_conditioning(
    parent_30m,
    hydrology_30m,
):
    terrain = build_10m_terrain_conditioning(
        parent_30m
    )

    hydrology = build_10m_hydrology_conditioning(
        hydrology_30m
    )

    return torch.cat(
        [terrain, hydrology],
        dim=1,
    )