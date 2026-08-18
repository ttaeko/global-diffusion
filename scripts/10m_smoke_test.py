import torch
from terrain_diffusion.models.one_pass_unet import OnePassUNet
from terrain_diffusion.training.datasets.one_pass_dataset import Terrain10mDataset

DATASET_PATH = "data/alps_curriculum_10m_hydro_v4.h5"

dataset = Terrain10mDataset(
    DATASET_PATH,
    split="train",
    crop_size_30m=128,
)

sample = dataset[0]

x = sample["conditioning"].unsqueeze(0)

model = OnePassUNet(
    in_channels=20,
    out_channels=1,
    base_channels=48,
)

with torch.no_grad():
    y = model(x)

print("Input: ", x.shape)
print("Output:", y.shape)