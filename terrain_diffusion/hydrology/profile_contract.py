"""Versioned contract shared by every hydrology-producing pipeline stage.

The contract intentionally contains only physical and algorithmic parameters.
Model checkpoint paths and generated-world coordinates belong in per-window
provenance, while these values must remain identical between dataset building,
regional planning, benchmarking, and block generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


HYDROLOGY_PROFILE_SCHEMA = "terrain-diffusion-hydrology-profile"
HYDROLOGY_PROFILE_VERSION = 4


@dataclass(frozen=True)
class HydrologyProfileContract:
    """Immutable physical contract for authoritative generated hydrology."""

    schema: str = HYDROLOGY_PROFILE_SCHEMA
    version: int = HYDROLOGY_PROFILE_VERSION
    channel_minimum_area_km2: float = 11.875
    runoff_ratio: float = 0.628448430696258
    reference_precipitation_mm_year: float = 1350.0
    generated_precipitation_scale: float = 0.9248982412666115
    generated_precipitation_offset_mm: float = 726.9146354115265
    generated_precipitation_minimum_mm: float = 764.0210266113281
    generated_precipitation_maximum_mm: float = 2232.251220703125
    # V4 repairs only actual downstream rises. Natural flats are valid and do
    # not receive an artificial longitudinal grade.
    minimum_channel_grade: float = 0.0
    maximum_channel_grade: float = 0.25
    # A route requiring more than this local repair is a routing/planning
    # failure, not permission to excavate a new valley.
    maximum_profile_incision_m: float = 12.0
    corridor_half_width_m: float = 60.0
    corridor_truncation: float = 2.5
    conditioning_distance_scale_m: float = 2000.0
    deterministic_terrain_transform: bool = True

    def validate(self) -> None:
        if self.schema != HYDROLOGY_PROFILE_SCHEMA:
            raise ValueError(f"Unsupported hydrology profile schema: {self.schema!r}")
        if self.version != HYDROLOGY_PROFILE_VERSION:
            raise ValueError(
                f"Unsupported hydrology profile version {self.version}; "
                f"expected {HYDROLOGY_PROFILE_VERSION}"
            )
        if self.channel_minimum_area_km2 <= 0:
            raise ValueError("channel_minimum_area_km2 must be positive")
        if not 0.0 <= self.runoff_ratio <= 1.0:
            raise ValueError("runoff_ratio must lie between zero and one")
        if self.reference_precipitation_mm_year <= 0:
            raise ValueError("reference precipitation must be positive")
        if self.generated_precipitation_scale <= 0:
            raise ValueError("generated precipitation scale must be positive")
        if (
            self.generated_precipitation_minimum_mm <= 0
            or self.generated_precipitation_maximum_mm
            <= self.generated_precipitation_minimum_mm
        ):
            raise ValueError("generated precipitation clipping bounds are invalid")
        if not 0 <= self.minimum_channel_grade <= self.maximum_channel_grade:
            raise ValueError("channel grade bounds are invalid")
        if self.maximum_profile_incision_m <= 0:
            raise ValueError("maximum_profile_incision_m must be positive")
        if self.corridor_half_width_m <= 0 or self.corridor_truncation <= 0:
            raise ValueError("corridor dimensions must be positive")
        if self.conditioning_distance_scale_m <= 0:
            raise ValueError("conditioning distance scale must be positive")
        if not self.deterministic_terrain_transform:
            raise ValueError("Version 4 requires the deterministic terrain transform")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def profile_kwargs(self, *, resolution_m: float) -> dict[str, float]:
        return {
            "resolution_m": float(resolution_m),
            "minimum_grade": self.minimum_channel_grade,
            "maximum_grade": self.maximum_channel_grade,
            "maximum_incision_m": self.maximum_profile_incision_m,
            "corridor_half_width_m": self.corridor_half_width_m,
            "corridor_truncation": self.corridor_truncation,
        }

    def calibrate_generated_precipitation(self, values):
        """Apply the frozen FOEN/Swiss affine calibration to generated climate."""

        import numpy as np

        precipitation = np.asarray(values, dtype=np.float32)
        calibrated = (
            precipitation * self.generated_precipitation_scale
            + self.generated_precipitation_offset_mm
        )
        return np.clip(
            calibrated,
            self.generated_precipitation_minimum_mm,
            self.generated_precipitation_maximum_mm,
        ).astype(np.float32)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "HydrologyProfileContract":
        contract = cls(**dict(values))
        contract.validate()
        return contract


DEFAULT_HYDROLOGY_PROFILE = HydrologyProfileContract()
DEFAULT_HYDROLOGY_PROFILE.validate()


def load_hydrology_profile_contract(
    path: str | Path | None = None,
) -> HydrologyProfileContract:
    """Load an explicit contract, or return the repository V4 contract."""

    if path is None:
        return DEFAULT_HYDROLOGY_PROFILE
    with Path(path).open("r", encoding="utf-8") as handle:
        return HydrologyProfileContract.from_dict(json.load(handle))


def profile_provenance(contract: HydrologyProfileContract) -> dict[str, Any]:
    contract.validate()
    return {
        "hydrology_profile_schema": contract.schema,
        "hydrology_profile_version": contract.version,
        "hydrology_profile_sha256": contract.fingerprint,
        "hydrology_profile": contract.to_dict(),
    }


def require_matching_profile(
    values: Mapping[str, Any],
    contract: HydrologyProfileContract = DEFAULT_HYDROLOGY_PROFILE,
) -> None:
    """Reject artifacts or checkpoints built under a different profile."""

    schema = values.get("hydrology_profile_schema")
    version = values.get("hydrology_profile_version")
    fingerprint = values.get("hydrology_profile_sha256")
    if schema != contract.schema or int(version or -1) != contract.version:
        raise ValueError(
            "Hydrology profile schema/version mismatch: "
            f"artifact={schema!r}/v{version!r}, "
            f"required={contract.schema!r}/v{contract.version}"
        )
    if fingerprint != contract.fingerprint:
        raise ValueError("Hydrology profile parameters do not match the V4 contract")
