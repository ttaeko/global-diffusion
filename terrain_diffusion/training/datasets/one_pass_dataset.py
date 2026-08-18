import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from terrain_diffusion.models.one_pass_residual import (
    block_mean_3x3,
    smooth_exact_upsample_3x,
)

from terrain_diffusion.models.one_pass_conditioning import (
    build_10m_terrain_conditioning,
)

TERRAIN_MEAN = torch.tensor([
    1121.844864,  # smooth_parent
    1121.844864,  # repeated_parent
    -0.002050,    # grad_x
    0.000508,     # grad_y
    0.000922,     # tpi_90m
    0.010885,     # tpi_270m
    27.897096,    # relief_90m
    102.059178,   # relief_270m
    8.949297,     # roughness_90m
    25.801343,    # roughness_270m
], dtype=torch.float32)

TERRAIN_STD = torch.tensor([
    773.885308,
    773.874489,
    0.339069,
    0.352345,
    2.523979,
    11.028648,
    25.709181,
    86.976716,
    8.242136,
    22.274991,
], dtype=torch.float32)

CURVATURE_MEAN = -0.000003
CURVATURE_STD = 0.009397

RESIDUAL_STD_M = 1.6912292128

ELEVATION_MEAN = 1121.844864
ELEVATION_STD = 773.88

HYDRO_CHANNEL_INDICES = [
    0,  # log_accumulation
    1,  # log_discharge
    2,  # channel_proximity
    3,  # flow_east
    4,  # flow_south
    5,  # stream_order
    7,  # lake_mask
]


def normalize_10m_conditioning(conditioning: torch.Tensor) -> torch.Tensor:
    """Apply the Candidate-A feature normalization used during training.

    ``conditioning`` must contain the unnormalised 13 terrain channels followed
    by the seven already-canonical hydrology channels.  Keeping this operation
    here makes inference consume exactly the same numerical contract as the
    dataset, including intentionally unnormalised phase and hydrology fields.
    """
    if conditioning.ndim != 4 or conditioning.shape[1] != 20:
        raise ValueError("Expected [B, 20, H, W] one-pass conditioning")
    result = conditioning.clone()
    mean = TERRAIN_MEAN.to(result).view(1, 10, 1, 1)
    std = TERRAIN_STD.to(result).view(1, 10, 1, 1)
    result[:, 0:10] = (result[:, 0:10] - mean) / std
    result[:, 12:13] = (result[:, 12:13] - CURVATURE_MEAN) / CURVATURE_STD
    return result

def inverse_signed_sqrt(x):
    return torch.sign(x) * x.square()

class Terrain10mDataset(Dataset):
    def __init__(
        self,
        h5_path,
        split="train",
        crop_size_30m=128,
    ):
        self.h5_path = h5_path
        self.split = split
        self.crop_size_30m = crop_size_30m
        self.crop_size_10m = crop_size_30m * 3

        self.samples = []

        with h5py.File(self.h5_path, "r") as f:
            stage = f["10"]

            for region_name in stage:
                region = stage[region_name]

                for chunk_name in region:
                    chunk = region[chunk_name]

                    sample_split = chunk["residual"].attrs["split"]

                    if sample_split == self.split:
                        self.samples.append(
                            (region_name, chunk_name)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        region_name, chunk_name = self.samples[index]

        with h5py.File(self.h5_path, "r") as f:
            g = f["10"][region_name][chunk_name]

            lowfreq = torch.from_numpy(
                g["lowfreq"][...]
            ).float()

            residual_old = torch.from_numpy(
                g["residual"][...]
            ).float()

            hydrology = torch.from_numpy(
                g["hydrology"][...]
            ).float()

        # ---------------------------------------------------------
        # 1. Reconstruct original 10 m DEM in signed-sqrt space
        # ---------------------------------------------------------

        lowfreq_up = TF.resize(
            lowfreq.unsqueeze(0),
            size=residual_old.shape,
            interpolation=InterpolationMode.BILINEAR,
        ).squeeze(0)

        target_10m_sqrt = residual_old + lowfreq_up

        # ---------------------------------------------------------
        # 2. Convert original terrain back to physical metres
        # ---------------------------------------------------------

        target_10m_m = inverse_signed_sqrt(
            target_10m_sqrt
        )

        # Add B and C dimensions temporarily:
        # [1536,1536] -> [1,1,1536,1536]
        target_4d = target_10m_m.unsqueeze(0).unsqueeze(0)

        # ---------------------------------------------------------
        # 3. Construct our NEW authoritative 30 m parent
        #
        # Exact aligned 3x3 block mean in physical metres.
        # ---------------------------------------------------------

        parent_30m = block_mean_3x3(
            target_4d
        )

        # [1,1,512,512] -> [512,512]
        parent_30m = parent_30m.squeeze(0).squeeze(0)

        # ---------------------------------------------------------
        # 4. Construct smooth-exact 10 m base over the FULL tile
        #
        # Doing this before cropping avoids artificial interpolation
        # behavior at random crop boundaries.
        # ---------------------------------------------------------

        base_10m = smooth_exact_upsample_3x(
            parent_30m.unsqueeze(0).unsqueeze(0)
        )

        base_10m = base_10m.squeeze(0).squeeze(0)

        # ---------------------------------------------------------
        # 5. New exact residual target
        # ---------------------------------------------------------

        residual_target = target_10m_m - base_10m

        # ---------------------------------------------------------
        # 6. Choose aligned crop in 30 m coordinates
        # ---------------------------------------------------------

        crop30 = self.crop_size_30m
        crop10 = self.crop_size_10m

        # Largest terrain descriptor uses a 9x9 window,
        # so we need 4 parent cells of context on every side.
        halo = 4
        halo10 = halo * 3

        h30, w30 = parent_30m.shape

        if self.split == "train":
            y30 = torch.randint(
                halo,
                h30 - crop30 - halo + 1,
                (1,),
            ).item()

            x30 = torch.randint(
                halo,
                w30 - crop30 - halo + 1,
                (1,),
            ).item()

        else:
            y30 = (h30 - crop30) // 2
            x30 = (w30 - crop30) // 2


        # Corresponding 10 m coordinates
        y10 = y30 * 3
        x10 = x30 * 3

        # ---------------------------------------------------------
        # 7. Parent context for terrain conditioning
        # ---------------------------------------------------------

        parent_context = parent_30m[
            y30 - halo:y30 + crop30 + halo,
            x30 - halo:x30 + crop30 + halo,
        ]

        # [136,136] -> [1,1,136,136]
        parent_context = parent_context.unsqueeze(0).unsqueeze(0)

        # ---------------------------------------------------------
        # 8. Build 13 terrain conditioning channels
        # ---------------------------------------------------------

        terrain_conditioning = build_10m_terrain_conditioning(
            parent_context
        )

        # [1,13,408,408]
        #
        # Remove the 12-pixel 10 m halo so we're left with
        # the actual 384x384 training region.
        terrain_conditioning = terrain_conditioning[
            :,
            :,
            halo10:-halo10,
            halo10:-halo10,
        ]

        # ---------------------------------------------------------
        # 9. Crop training targets
        # ---------------------------------------------------------

        parent_crop = parent_30m[
            y30:y30 + crop30,
            x30:x30 + crop30,
        ]

        target_crop = target_10m_m[
            y10:y10 + crop10,
            x10:x10 + crop10,
        ]

        residual_crop = residual_target[
            y10:y10 + crop10,
            x10:x10 + crop10,
        ]

        # ---------------------------------------------------------
        # 10. Hydrology conditioning
        # ---------------------------------------------------------

        hydro_crop = hydrology[
            HYDRO_CHANNEL_INDICES,
            y30:y30 + crop30,
            x30:x30 + crop30,
        ]
        
        # [7,128,128] -> [1,7,128,128]
        hydro_crop = hydro_crop.unsqueeze(0)

        hydro_continuous = hydro_crop[:, 0:5]

        hydro_continuous_10m = F.interpolate(
            hydro_continuous,
            scale_factor=3,
            mode="bilinear",
            align_corners=False,
        )

        hydro_discrete = hydro_crop[:, 5:7]

        hydro_discrete_10m = F.interpolate(
            hydro_discrete,
            scale_factor=3,
            mode="nearest",
        )

        hydro_10m = torch.cat(
            [
                hydro_continuous_10m,
                hydro_discrete_10m,
            ],
            dim=1,
        )

        # ---------------------------------------------------------
        # 11. Final 20-channel conditioning tensor
        # ---------------------------------------------------------

        conditioning = torch.cat(
            [
                terrain_conditioning,  # 13 channels
                hydro_10m,             # 7 channels
            ],
            dim=1,
        )

        conditioning = normalize_10m_conditioning(conditioning)

        residual_crop_normalized = (
            residual_crop / RESIDUAL_STD_M
        )        

        # Remove temporary batch dimension
        conditioning = conditioning.squeeze(0)        

        return {
            "conditioning": conditioning.squeeze(0),

            # What the network trains against
            "residual_target": residual_crop_normalized.unsqueeze(0),

            # Useful for reconstruction/evaluation
            "residual_target_m": residual_crop.unsqueeze(0),
            "parent_30m": parent_crop.unsqueeze(0),
            "target_10m": target_crop.unsqueeze(0),
        }  

    def inverse_signed_sqrt(x):
        """
        signed-sqrt -> physical elevation in metres
        """
        return torch.sign(x) * x.square()

if __name__ == "__main__":
    dataset = Terrain10mDataset(
        "data/alps_curriculum_10m_hydro_v4.h5",
        split="train",
        crop_size_30m=128,
    )

    sample = dataset[0]

    for key, value in sample.items():
        print(key, value.shape)
