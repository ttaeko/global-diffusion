"""Training dataset for the continent-scale macro terrain model."""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# These are the statistics used by the existing 30 m coarse/base interface.
# Keeping macro outputs in this representation lets a trained macro model replace
# the old coarse stage without retraining the base or decoder models.
DEFAULT_MACRO_MEANS = [
    -37.67916460232751,
    2.22578822145657,
    18.030293275011356,
    333.8442390481231,
    1350.1259248456176,
    52.444339366764396,
]
DEFAULT_MACRO_STDS = [
    39.68515115440358,
    3.0981253981231522,
    8.940333096712806,
    322.25238547630295,
    856.3430083394657,
    30.982620765341043,
]


def load_macro_stats(h5_file: str) -> tuple[list[float], list[float]]:
    """Load the normalization paired with a built macro dataset."""
    with h5py.File(h5_file, "r") as f:
        if "data_means" not in f.attrs or "data_stds" not in f.attrs:
            raise ValueError(f"{h5_file} does not contain macro normalization attributes")
        means = np.asarray(f.attrs["data_means"], dtype=np.float64)
        stds = np.asarray(f.attrs["data_stds"], dtype=np.float64)
    if means.shape != (6,) or stds.shape != (6,):
        raise ValueError(f"{h5_file} macro normalization must contain six channels")
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError(f"{h5_file} contains invalid macro normalization")
    return means.tolist(), stds.tolist()


class H5MacroTerrainDataset(Dataset):
    """Draw large square crops from equal-distance latitude bands.

    The HDF5 file is produced by ``build-macro-dataset`` and contains one or
    more ``bands/<id>`` arrays with shape ``(6, H, W)``.  Channels match the
    numerically stable representation used to train the legacy coarse model:

    ``[mean elevation sqrt, mean-minus-p5, temperature, temperature std,
    precipitation, precipitation variability]``. At inference, channel 1 is
    converted back to absolute p5 elevation before the base stage consumes it.

    ``split`` reserves a longitudinal holdout within every latitude band. This
    avoids drawing overlapping train and validation crops while preserving the
    full range of climates in both splits.
    """

    def __init__(
        self,
        h5_file: str,
        crop_size: int = 256,
        sigma_data: float = 0.5,
        split: str = "train",
        validation_fraction: float = 0.2,
        samples_per_epoch: int = 100_000,
        augment: bool = True,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        seed: int = 0,
        land_fraction_bins: list[float] | None = None,
        land_fraction_probabilities: list[float] | None = None,
        land_catalog_stride: int = 16,
        condition_on_land_fraction: bool = False,
    ):
        self.h5_file = h5_file
        self.crop_size = int(crop_size)
        self.sigma_data = float(sigma_data)
        self.split = split
        self.validation_fraction = float(validation_fraction)
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        self.samples_per_epoch = int(samples_per_epoch)
        self.augment = bool(augment) and split == "train"
        self.base_seed = int(seed)
        self.rng = torch.Generator().manual_seed(self.base_seed)
        self.condition_on_land_fraction = bool(condition_on_land_fraction)
        self.land_fraction_bins = None
        self.land_fraction_probabilities = None
        self.land_catalog_stride = int(land_catalog_stride)
        if self.land_catalog_stride < 1:
            raise ValueError("land_catalog_stride must be positive")
        if land_fraction_bins is not None:
            bins = np.asarray(land_fraction_bins, dtype=np.float64)
            probabilities = np.asarray(land_fraction_probabilities, dtype=np.float64)
            if (
                bins.ndim != 1
                or bins.size < 2
                or bins[0] != 0.0
                or bins[-1] < 1.0
                or np.any(np.diff(bins) <= 0)
            ):
                raise ValueError("land_fraction_bins must increase from 0 through at least 1")
            if probabilities.shape != (bins.size - 1,) or np.any(probabilities < 0):
                raise ValueError("land_fraction_probabilities must have one nonnegative value per bin")
            if probabilities.sum() <= 0:
                raise ValueError("land_fraction_probabilities must have positive total weight")
            self.land_fraction_bins = bins
            self.land_fraction_probabilities = torch.tensor(
                probabilities / probabilities.sum(), dtype=torch.float64
            )

        with h5py.File(self.h5_file, "r") as f:
            if "bands" not in f:
                raise ValueError(f"{h5_file} has no 'bands' group")
            all_keys = sorted(f["bands"].keys(), key=lambda value: int(value))
            usable = [
                key for key in all_keys
                if min(f[f"bands/{key}"].shape[-2:]) >= self.crop_size
            ]
            file_means = f.attrs.get("data_means")
            file_stds = f.attrs.get("data_stds")

        if not usable:
            raise ValueError(
                f"No macro bands in {h5_file} are large enough for a "
                f"{self.crop_size}x{self.crop_size} crop"
            )

        if split not in ("train", "val", "validation"):
            raise ValueError("split must be 'train' or 'val'")
        self.keys = usable

        norm_means = means if means is not None else file_means
        norm_stds = stds if stds is not None else file_stds
        if norm_means is None or norm_stds is None:
            # Backward compatibility for early macro datasets which did not
            # persist their own normalization metadata.
            norm_means = DEFAULT_MACRO_MEANS
            norm_stds = DEFAULT_MACRO_STDS
        if len(norm_means) != 6 or len(norm_stds) != 6:
            raise ValueError("macro means and stds must contain six values")
        if not np.isfinite(norm_means).all() or not np.isfinite(norm_stds).all():
            raise ValueError("macro means and stds must be finite")
        if np.any(np.asarray(norm_stds) <= 0):
            raise ValueError("macro standard deviations must be positive")
        self.means = torch.tensor(norm_means, dtype=torch.float32).view(6, 1, 1)
        self.stds = torch.tensor(norm_stds, dtype=torch.float32).view(6, 1, 1)
        self.land_catalog = self._build_land_catalog() if self.land_fraction_bins is not None else None

    def __len__(self):
        return self.samples_per_epoch

    def set_seed(self, seed):
        self.rng = torch.Generator().manual_seed(int(seed))

    def _random_int(self, high_inclusive: int) -> int:
        if high_inclusive <= 0:
            return 0
        return int(torch.randint(high_inclusive + 1, (1,), generator=self.rng).item())

    def _longitude_range(self, width: int) -> tuple[int, int]:
        holdout_start = int(round(width * (1.0 - self.validation_fraction)))
        if self.split == "train":
            j_min, j_max_exclusive = 0, holdout_start
        else:
            j_min, j_max_exclusive = holdout_start, width
        if j_max_exclusive - j_min < self.crop_size:
            return 0, width
        return j_min, j_max_exclusive

    def _build_land_catalog(self):
        """Index crop positions by land coverage without loading full 6-channel crops.

        A small fixed-stride catalogue makes the requested sampling distribution
        exact and avoids rejection sampling repeatedly reading megabytes from HDF5.
        """
        catalog = [[] for _ in range(len(self.land_fraction_bins) - 1)]
        with h5py.File(self.h5_file, "r") as f:
            for key in self.keys:
                dataset = f[f"bands/{key}"]
                _, height, width = dataset.shape
                j_min, j_max_exclusive = self._longitude_range(width)
                max_i = height - self.crop_size
                max_j = j_max_exclusive - self.crop_size
                rows = list(range(0, max_i + 1, self.land_catalog_stride))
                cols = list(range(j_min, max_j + 1, self.land_catalog_stride))
                if rows[-1] != max_i:
                    rows.append(max_i)
                if cols[-1] != max_j:
                    cols.append(max_j)

                land = dataset[0].astype(np.float32) > 0.0
                integral = np.pad(
                    np.cumsum(np.cumsum(land, axis=0, dtype=np.uint32), axis=1, dtype=np.uint32),
                    ((1, 0), (1, 0)),
                )
                for i in rows:
                    jj = np.asarray(cols, dtype=np.int64)
                    sums = (
                        integral[i + self.crop_size, jj + self.crop_size]
                        - integral[i, jj + self.crop_size]
                        - integral[i + self.crop_size, jj]
                        + integral[i, jj]
                    )
                    fractions = sums.astype(np.float64) / float(self.crop_size ** 2)
                    bin_ids = np.searchsorted(
                        self.land_fraction_bins[1:-1], fractions, side="right"
                    )
                    for j, fraction, bin_id in zip(cols, fractions, bin_ids):
                        catalog[int(bin_id)].append((key, i, j, float(fraction)))

        missing = [
            index for index, (entries, probability) in enumerate(
                zip(catalog, self.land_fraction_probabilities.tolist())
            )
            if probability > 0 and not entries
        ]
        if missing:
            raise ValueError(f"No macro crops exist for requested land-fraction bins {missing}")
        return catalog

    def __getitem__(self, index):
        selected_fraction = None
        if self.land_catalog is not None and self.split == "train":
            bin_id = int(torch.multinomial(
                self.land_fraction_probabilities, 1, generator=self.rng
            ).item())
            choices = self.land_catalog[bin_id]
            key, i, j, selected_fraction = choices[self._random_int(len(choices) - 1)]
        elif self.split == "train":
            key = self.keys[self._random_int(len(self.keys) - 1)]
        else:
            key = self.keys[index % len(self.keys)]

        with h5py.File(self.h5_file, "r") as f:
            dataset = f[f"bands/{key}"]
            _, height, width = dataset.shape
            j_min, j_max_exclusive = self._longitude_range(width)
            if self.split == "train" and self.land_catalog is None:
                i = self._random_int(height - self.crop_size)
                j = j_min + self._random_int(j_max_exclusive - j_min - self.crop_size)
            elif self.split != "train":
                # A deterministic low-discrepancy traversal for validation.
                i = (index * 104729) % (height - self.crop_size + 1)
                j = j_min + (index * 130363) % (j_max_exclusive - j_min - self.crop_size + 1)
            crop = torch.from_numpy(
                dataset[:, i:i + self.crop_size, j:j + self.crop_size].astype(np.float32)
            )

        if self.augment:
            transform = self._random_int(7)
            if transform >= 4:
                crop = torch.flip(crop, dims=(-1,))
            crop = torch.rot90(crop, k=transform % 4, dims=(-2, -1))

        image = ((crop - self.means) / self.stds) * self.sigma_data
        result = {"image": image}
        if self.condition_on_land_fraction:
            if selected_fraction is None:
                selected_fraction = float((crop[0] > 0.0).float().mean().item())
            # Centre the scalar so the conditioning embedding sees a symmetric
            # range while preserving a direct, interpretable 0..1 API.
            result["cond_inputs"] = [torch.tensor(selected_fraction * 2.0 - 1.0)]
            # Explicit float32 is required on Metal: DataLoader otherwise
            # collates a Python float into float64, which MPS cannot transfer.
            result["land_fraction"] = torch.tensor(
                selected_fraction, dtype=torch.float32
            )
        return result
