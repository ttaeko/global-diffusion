"""Versioned 30 m -> 10 m hydrology geometry and decoder composition contract.

The regional planner owns topology, discharge, bed elevation, lake surfaces,
and boundary continuity at 30 m.  This module converts that plan into subcell
10 m geometry and defines which terrain degrees of freedom the stochastic
decoder is allowed to change.  Exact river-bed anchors and lake ceilings are
part of the decoder representation, not a post-inference terrain repair.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import scipy.ndimage

from .conditioning import HYDROLOGY_CONDITIONING_CHANNELS
from .training_profile import (
    HYDROLOGY_PROFILE_CHANNELS,
    HydrologyTrainingProfile,
    flow_direction_from_vectors,
)
from .world_plan import D8_DIRECTION_OFFSETS


HYDROLOGY_DECODER_SCHEMA = "terrain-diffusion-hydrology-decoder"
HYDROLOGY_DECODER_VERSION = 4
HYDROLOGY_DECODER_V1_VERSION = 1
HYDROLOGY_DECODER_V2_VERSION = 2
HYDROLOGY_DECODER_V3_VERSION = 3


@dataclass(frozen=True)
class HydrologyDecoderContract:
    """Immutable fine-geometry and constrained-output contract."""

    schema: str = HYDROLOGY_DECODER_SCHEMA
    version: int = HYDROLOGY_DECODER_VERSION
    refinement: int = 3
    coarse_resolution_m: float = 30.0
    fine_resolution_m: float = 10.0
    channel_proximity_scale_m: float = 2000.0
    corridor_half_width_m: float = 60.0
    corridor_truncation: float = 2.5
    minimum_river_width_m: float = 2.0
    width_discharge_coefficient: float = 2.5
    maximum_river_width_m: float = 120.0
    lake_minimum_depth_m: float = 1.0
    anchor_mode: str = "subcell-centerline"
    bed_relative_encoding: str = "corridor-tanh"
    bed_relative_scale_m: float = 32.0
    synthesis_base: str = "conditioned-exact-fine"
    post_model_composition: str = "none"
    residual_support: str = "free-terrain-only"

    def validate(self) -> None:
        if self.schema != HYDROLOGY_DECODER_SCHEMA:
            raise ValueError(f"Unsupported hydrology decoder schema: {self.schema!r}")
        if self.version not in (
            HYDROLOGY_DECODER_V1_VERSION,
            HYDROLOGY_DECODER_V2_VERSION,
            HYDROLOGY_DECODER_V3_VERSION,
            HYDROLOGY_DECODER_VERSION,
        ):
            raise ValueError(
                f"Unsupported hydrology decoder version {self.version}; "
                f"expected {HYDROLOGY_DECODER_V1_VERSION}, "
                f"{HYDROLOGY_DECODER_V2_VERSION}, "
                f"{HYDROLOGY_DECODER_V3_VERSION}, or {HYDROLOGY_DECODER_VERSION}"
            )
        if self.refinement != 3:
            raise ValueError("Decoder V1 requires an exact 30 m -> 10 m refinement of 3")
        if not np.isclose(
            self.coarse_resolution_m,
            self.fine_resolution_m * self.refinement,
        ):
            raise ValueError("Decoder resolutions and refinement do not align")
        if min(
            self.channel_proximity_scale_m,
            self.corridor_half_width_m,
            self.corridor_truncation,
            self.minimum_river_width_m,
            self.width_discharge_coefficient,
            self.maximum_river_width_m,
            self.lake_minimum_depth_m,
            self.bed_relative_scale_m,
        ) <= 0:
            raise ValueError("Hydrology decoder physical scales must be positive")
        if self.maximum_river_width_m < self.minimum_river_width_m:
            raise ValueError("Maximum river width must not be below the minimum")
        if self.anchor_mode != "subcell-centerline":
            raise ValueError("Hydrology decoder requires subcell-centerline anchors")
        if self.version == HYDROLOGY_DECODER_V2_VERSION and (
            self.bed_relative_encoding != "corridor-tanh"
        ):
            raise ValueError("Decoder V2 requires corridor-tanh bed encoding")
        if self.version <= HYDROLOGY_DECODER_V2_VERSION:
            if self.synthesis_base != "conditioned-blurred":
                raise ValueError("Historical decoders require the blurred base")
            if self.post_model_composition != "anchors-and-lake-cap":
                raise ValueError("Historical decoders require post-model composition")
        else:
            if self.synthesis_base != "conditioned-exact-fine":
                raise ValueError("Decoder V3 requires the exact fine synthesis base")
            if self.post_model_composition != "none":
                raise ValueError("Decoder V3 prohibits post-model composition")
        if self.version < HYDROLOGY_DECODER_VERSION:
            if self.residual_support != "unconstrained":
                raise ValueError("Historical decoders require unconstrained residuals")
        elif self.residual_support != "free-terrain-only":
            raise ValueError("Decoder V4 requires free-terrain-only residual support")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        # Preserve exact historical fingerprints after V3 changes the output
        # representation rather than appending another input feature.
        if self.version == HYDROLOGY_DECODER_V1_VERSION:
            values.pop("bed_relative_encoding")
            values.pop("bed_relative_scale_m")
        if self.version <= HYDROLOGY_DECODER_V2_VERSION:
            values.pop("synthesis_base")
            values.pop("post_model_composition")
        if self.version <= HYDROLOGY_DECODER_V3_VERSION:
            values.pop("residual_support")
        if self.version >= HYDROLOGY_DECODER_V3_VERSION:
            values.pop("bed_relative_encoding")
            values.pop("bed_relative_scale_m")
        return values

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "HydrologyDecoderContract":
        contract = cls(**dict(values))
        contract.validate()
        return contract


HYDROLOGY_DECODER_V1 = HydrologyDecoderContract(
    version=HYDROLOGY_DECODER_V1_VERSION,
    synthesis_base="conditioned-blurred",
    post_model_composition="anchors-and-lake-cap",
    residual_support="unconstrained",
)
HYDROLOGY_DECODER_V1.validate()
HYDROLOGY_DECODER_V2 = HydrologyDecoderContract(
    version=HYDROLOGY_DECODER_V2_VERSION,
    synthesis_base="conditioned-blurred",
    post_model_composition="anchors-and-lake-cap",
    residual_support="unconstrained",
)
HYDROLOGY_DECODER_V2.validate()
HYDROLOGY_DECODER_V3 = HydrologyDecoderContract(
    version=HYDROLOGY_DECODER_V3_VERSION,
    residual_support="unconstrained",
)
HYDROLOGY_DECODER_V3.validate()
DEFAULT_HYDROLOGY_DECODER = HydrologyDecoderContract()
DEFAULT_HYDROLOGY_DECODER.validate()


@dataclass(frozen=True)
class FineHydrologyGeometry:
    """Aligned 10 m conditioning and immutable terrain constraints."""

    conditioning: np.ndarray
    profile_conditioning: np.ndarray
    bed_relative_conditioning: np.ndarray
    channel_centerline_mask: np.ndarray
    channel_coverage: np.ndarray
    target_bed_elevation_m: np.ndarray
    lake_mask: np.ndarray
    lake_coverage: np.ndarray
    water_surface_elevation_m: np.ndarray
    freedom_mask: np.ndarray
    river_width_m: np.ndarray

    def validate(self) -> None:
        shape = self.channel_centerline_mask.shape
        if self.conditioning.shape != (len(HYDROLOGY_CONDITIONING_CHANNELS), *shape):
            raise ValueError("Fine hydrology conditioning has an invalid shape")
        if self.profile_conditioning.shape != (
            len(HYDROLOGY_PROFILE_CHANNELS),
            *shape,
        ):
            raise ValueError("Fine profile conditioning has an invalid shape")
        for name in (
            "bed_relative_conditioning",
            "channel_coverage",
            "target_bed_elevation_m",
            "lake_mask",
            "lake_coverage",
            "water_surface_elevation_m",
            "freedom_mask",
            "river_width_m",
        ):
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"Fine hydrology field {name} does not align")
        anchors = np.asarray(self.channel_centerline_mask, dtype=bool)
        if np.any(anchors & ~np.isfinite(self.target_bed_elevation_m)):
            raise ValueError("Every fine river anchor requires a finite bed elevation")
        if np.any(np.asarray(self.lake_mask, dtype=bool) & ~np.isfinite(
            self.water_surface_elevation_m
        )):
            raise ValueError("Every fine lake cell requires a finite water surface")
        if not np.isfinite(self.conditioning).all():
            raise ValueError("Fine hydrology conditioning must be finite")
        if not np.isfinite(self.profile_conditioning).all():
            raise ValueError("Fine profile conditioning must be finite")
        if not np.isfinite(self.bed_relative_conditioning).all():
            raise ValueError("Fine bed-relative conditioning must be finite")
        if np.any(np.abs(self.bed_relative_conditioning) > 1.0 + 1e-6):
            raise ValueError("Fine bed-relative conditioning must lie in [-1, 1]")
        if np.any((self.freedom_mask < 0) | (self.freedom_mask > 1)):
            raise ValueError("Decoder freedom mask must lie in [0, 1]")
        if np.any(self.freedom_mask[anchors] != 0):
            raise ValueError("River anchors must have zero decoder freedom")


def decoder_provenance(
    contract: HydrologyDecoderContract = DEFAULT_HYDROLOGY_DECODER,
) -> dict[str, Any]:
    contract.validate()
    return {
        "hydrology_decoder_schema": contract.schema,
        "hydrology_decoder_version": contract.version,
        "hydrology_decoder_sha256": contract.fingerprint,
        "hydrology_decoder": contract.to_dict(),
    }


def require_matching_decoder(
    values: Mapping[str, Any],
    contract: HydrologyDecoderContract = DEFAULT_HYDROLOGY_DECODER,
) -> None:
    schema = values.get("hydrology_decoder_schema")
    version = values.get("hydrology_decoder_version")
    fingerprint = values.get("hydrology_decoder_sha256")
    if schema != contract.schema or int(version or -1) != contract.version:
        raise ValueError(
            "Hydrology decoder schema/version mismatch: "
            f"artifact={schema!r}/v{version!r}, "
            f"required={contract.schema!r}/v{contract.version}"
        )
    if fingerprint != contract.fingerprint:
        raise ValueError(
            "Hydrology decoder parameters do not match the requested contract"
        )


def decoder_contract_from_provenance(
    values: Mapping[str, Any],
) -> HydrologyDecoderContract:
    """Resolve a supported immutable decoder contract from provenance."""

    version = int(values.get("hydrology_decoder_version") or -1)
    if version == HYDROLOGY_DECODER_V1_VERSION:
        contract = HYDROLOGY_DECODER_V1
    elif version == HYDROLOGY_DECODER_V2_VERSION:
        contract = HYDROLOGY_DECODER_V2
    elif version == HYDROLOGY_DECODER_V3_VERSION:
        contract = HYDROLOGY_DECODER_V3
    elif version == HYDROLOGY_DECODER_VERSION:
        contract = DEFAULT_HYDROLOGY_DECODER
    else:
        raise ValueError(f"Unsupported hydrology decoder version {version}")
    require_matching_decoder(values, contract)
    return contract


def encode_bed_relative_elevation(
    target_bed_elevation_m: np.ndarray,
    channel_centerline_mask: np.ndarray,
    terrain_elevation_30m_m: np.ndarray,
    corridor: np.ndarray,
    *,
    contract: HydrologyDecoderContract = DEFAULT_HYDROLOGY_DECODER,
) -> np.ndarray:
    """Encode the nearest planned bed relative to the aligned terrain prior.

    The signed value is propagated through the same finite valley corridor as
    the profile geometry.  It tells the decoder how far the planned bed lies
    above or below its local bilinearly refined 30 m terrain reference without
    imposing any deterministic bank or valley shape.
    """

    contract.validate()
    anchor = np.asarray(channel_centerline_mask, dtype=bool)
    bed = np.asarray(target_bed_elevation_m, dtype=np.float32)
    corridor_values = np.asarray(corridor, dtype=np.float32)
    if bed.shape != anchor.shape or corridor_values.shape != anchor.shape:
        raise ValueError("Bed, anchor, and corridor fields must align")
    coarse = np.asarray(terrain_elevation_30m_m, dtype=np.float32)
    expected = (
        coarse.shape[0] * contract.refinement,
        coarse.shape[1] * contract.refinement,
    )
    if coarse.ndim != 2 or expected != anchor.shape:
        raise ValueError("Bed-relative terrain reference does not align")
    if not np.isfinite(coarse).all():
        raise ValueError("Bed-relative terrain reference must be finite")
    if not np.any(anchor):
        return np.zeros(anchor.shape, dtype=np.float32)
    if np.any(~np.isfinite(bed[anchor])):
        raise ValueError("Bed-relative conditioning requires finite anchors")
    _, nearest = scipy.ndimage.distance_transform_edt(
        ~anchor, return_indices=True
    )
    nearest_bed = bed[nearest[0], nearest[1]]
    fine_terrain = scipy.ndimage.zoom(
        coarse,
        zoom=contract.refinement,
        order=1,
        mode="nearest",
        grid_mode=True,
        prefilter=False,
    )
    fine_terrain = fine_terrain[: expected[0], : expected[1]]
    relative_m = nearest_bed - fine_terrain
    encoded = np.tanh(relative_m / contract.bed_relative_scale_m)
    return (encoded * corridor_values).astype(np.float32)


def build_fine_hydrology_geometry(
    hydrology_30m: np.ndarray,
    profile_30m: HydrologyTrainingProfile,
    *,
    lake_water_surface_elevation_m: np.ndarray | None = None,
    terrain_elevation_30m_m: np.ndarray | None = None,
    contract: HydrologyDecoderContract = DEFAULT_HYDROLOGY_DECODER,
) -> FineHydrologyGeometry:
    """Rasterize a connected subcell centerline and aligned decoder fields."""

    contract.validate()
    hydrology = np.asarray(hydrology_30m, dtype=np.float32)
    if hydrology.ndim != 3 or hydrology.shape[0] != len(
        HYDROLOGY_CONDITIONING_CHANNELS
    ):
        raise ValueError("Hydrology decoder requires the eight-channel plan")
    coarse_shape = hydrology.shape[1:]
    if profile_30m.channel_mask.shape != coarse_shape:
        raise ValueError("Hydrology profile and conditioning must align")
    refinement = contract.refinement
    fine_shape = tuple(size * refinement for size in coarse_shape)
    channel = np.asarray(profile_30m.channel_mask, dtype=bool)
    flow = flow_direction_from_vectors(hydrology[3], hydrology[4])
    bed = np.asarray(profile_30m.target_bed_elevation_m, dtype=np.float32)

    centerline = np.zeros(fine_shape, dtype=bool)
    fine_bed = np.full(fine_shape, np.nan, dtype=np.float32)
    source_rows = np.full(fine_shape, -1, dtype=np.int32)
    source_cols = np.full(fine_shape, -1, dtype=np.int32)

    def write_segment(
        start: tuple[int, int],
        stop: tuple[int, int],
        start_bed: float,
        stop_bed: float,
        source: tuple[int, int],
    ) -> None:
        steps = max(abs(stop[0] - start[0]), abs(stop[1] - start[1])) + 1
        rows = np.rint(np.linspace(start[0], stop[0], steps)).astype(np.int32)
        cols = np.rint(np.linspace(start[1], stop[1], steps)).astype(np.int32)
        elevations = np.linspace(start_bed, stop_bed, steps, dtype=np.float32)
        valid = (
            (rows >= 0)
            & (rows < fine_shape[0])
            & (cols >= 0)
            & (cols < fine_shape[1])
        )
        for row, col, elevation in zip(rows[valid], cols[valid], elevations[valid]):
            centerline[row, col] = True
            if not np.isfinite(fine_bed[row, col]) or elevation < fine_bed[row, col]:
                fine_bed[row, col] = elevation
                source_rows[row, col] = source[0]
                source_cols[row, col] = source[1]

    height, width = coarse_shape
    for row, col in zip(*np.nonzero(channel)):
        if not np.isfinite(bed[row, col]):
            raise ValueError("Every coarse channel cell requires a finite bed")
        start = (row * refinement + refinement // 2,
                 col * refinement + refinement // 2)
        code = int(flow[row, col])
        stop = start
        stop_bed = float(bed[row, col])
        if code in D8_DIRECTION_OFFSETS:
            delta_row, delta_col = D8_DIRECTION_OFFSETS[code]
            next_row, next_col = row + delta_row, col + delta_col
            if 0 <= next_row < height and 0 <= next_col < width:
                if channel[next_row, next_col] and np.isfinite(bed[next_row, next_col]):
                    stop = (
                        next_row * refinement + refinement // 2,
                        next_col * refinement + refinement // 2,
                    )
                    stop_bed = float(bed[next_row, next_col])
                elif hydrology[7, next_row, next_col] >= 0.5:
                    stop = (
                        next_row * refinement + refinement // 2,
                        next_col * refinement + refinement // 2,
                    )
                    if lake_water_surface_elevation_m is not None:
                        surface = float(lake_water_surface_elevation_m[next_row, next_col])
                        if np.isfinite(surface):
                            stop_bed = min(
                                stop_bed,
                                surface - contract.lake_minimum_depth_m,
                            )
        if stop == start and (row in (0, height - 1) or col in (0, width - 1)):
            incoming = []
            for delta_row, delta_col in D8_DIRECTION_OFFSETS.values():
                upstream_row, upstream_col = row - delta_row, col - delta_col
                if not (
                    0 <= upstream_row < height and 0 <= upstream_col < width
                ):
                    continue
                upstream_code = int(flow[upstream_row, upstream_col])
                if (
                    channel[upstream_row, upstream_col]
                    and D8_DIRECTION_OFFSETS.get(upstream_code)
                    == (delta_row, delta_col)
                ):
                    incoming.append((delta_row, delta_col))
            if incoming:
                delta_row, delta_col = incoming[0]
                stop = (
                    int(np.clip(start[0] + delta_row * refinement, 0, fine_shape[0] - 1)),
                    int(np.clip(start[1] + delta_col * refinement, 0, fine_shape[1] - 1)),
                )
            else:
                edge_candidates = []
                if row == 0:
                    edge_candidates.append((0, start[1]))
                if row == height - 1:
                    edge_candidates.append((fine_shape[0] - 1, start[1]))
                if col == 0:
                    edge_candidates.append((start[0], 0))
                if col == width - 1:
                    edge_candidates.append((start[0], fine_shape[1] - 1))
                stop = min(
                    edge_candidates,
                    key=lambda point: abs(point[0] - start[0]) + abs(point[1] - start[1]),
                )
        write_segment(start, stop, float(bed[row, col]), stop_bed, (row, col))

    if np.any(centerline):
        distance_pixels, nearest = scipy.ndimage.distance_transform_edt(
            ~centerline, return_indices=True
        )
    else:
        distance_pixels = np.full(fine_shape, np.inf, dtype=np.float32)
        nearest = np.zeros((2, *fine_shape), dtype=np.int32)
    distance_m = distance_pixels * contract.fine_resolution_m

    repeated = np.repeat(
        np.repeat(hydrology, refinement, axis=1), refinement, axis=2
    ).astype(np.float32)
    discharge = np.expm1(
        np.clip(repeated[1], 0.0, 1.0) * np.log1p(1000.0)
    )
    width_m = np.clip(
        contract.width_discharge_coefficient * np.sqrt(np.maximum(discharge, 0.0)),
        contract.minimum_river_width_m,
        contract.maximum_river_width_m,
    ).astype(np.float32)
    nearest_width = width_m[nearest[0], nearest[1]] if np.any(centerline) else width_m
    coverage = np.clip(
        (0.5 * nearest_width + 0.5 * contract.fine_resolution_m - distance_m)
        / contract.fine_resolution_m,
        0.0,
        1.0,
    ).astype(np.float32)

    lake_coverage = _refine_lake_coverage(hydrology[7], refinement)
    fine_lake = lake_coverage >= 0.5
    surfaces = _complete_lake_surfaces(
        hydrology[7] >= 0.5,
        lake_water_surface_elevation_m,
        terrain_elevation_30m_m,
    )
    fine_surface = _refine_lake_surfaces(
        hydrology[7] >= 0.5,
        surfaces,
        fine_lake,
        refinement,
    )
    fine_surface[~fine_lake] = np.nan
    if np.any(fine_lake & ~np.isfinite(fine_surface)):
        raise ValueError("Fine lake geometry requires water-surface elevation")

    conditioning = repeated.copy()
    conditioning[2] = np.where(
        np.isfinite(distance_m),
        np.exp(-distance_m / contract.channel_proximity_scale_m),
        0.0,
    )
    conditioning[5] = 0.0
    if np.any(centerline):
        coarse_order = repeated[5]
        conditioning[5, centerline] = coarse_order[centerline]
    conditioning[7] = lake_coverage

    coarse_incision = np.repeat(
        np.repeat(profile_30m.conditioning[1], refinement, axis=0),
        refinement,
        axis=1,
    )
    coarse_grade = np.repeat(
        np.repeat(profile_30m.conditioning[2], refinement, axis=0),
        refinement,
        axis=1,
    )
    corridor = np.where(
        distance_m <= contract.corridor_half_width_m * contract.corridor_truncation,
        np.exp(-0.5 * np.square(distance_m / contract.corridor_half_width_m)),
        0.0,
    ).astype(np.float32)
    fine_profile = np.stack(
        (
            centerline.astype(np.float32),
            coarse_incision * centerline,
            coarse_grade * centerline,
            corridor,
        )
    ).astype(np.float32)
    if contract.version == HYDROLOGY_DECODER_V2_VERSION:
        if terrain_elevation_30m_m is None:
            raise ValueError(
                "Decoder V2 bed-relative conditioning requires 30 m terrain"
            )
        bed_relative = encode_bed_relative_elevation(
            fine_bed,
            centerline,
            terrain_elevation_30m_m,
            corridor,
            contract=contract,
        )
    else:
        bed_relative = np.zeros(fine_shape, dtype=np.float32)
    freedom = np.ones(fine_shape, dtype=np.float32)
    freedom[centerline] = 0.0
    geometry = FineHydrologyGeometry(
        conditioning=conditioning,
        profile_conditioning=fine_profile,
        bed_relative_conditioning=bed_relative,
        channel_centerline_mask=centerline,
        channel_coverage=coverage,
        target_bed_elevation_m=fine_bed,
        lake_mask=fine_lake,
        lake_coverage=lake_coverage,
        water_surface_elevation_m=fine_surface,
        freedom_mask=freedom,
        river_width_m=nearest_width.astype(np.float32),
    )
    geometry.validate()
    return geometry


def constrain_refined_terrain(
    candidate_elevation_m: np.ndarray,
    base_elevation_m: np.ndarray,
    geometry: FineHydrologyGeometry,
    *,
    contract: HydrologyDecoderContract = DEFAULT_HYDROLOGY_DECODER,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the versioned post-model composition contract."""

    contract.validate()
    geometry.validate()
    candidate = np.asarray(candidate_elevation_m, dtype=np.float32)
    base = np.asarray(base_elevation_m, dtype=np.float32).copy()
    if candidate.shape != geometry.freedom_mask.shape or base.shape != candidate.shape:
        raise ValueError("Candidate, base, and fine hydrology geometry must align")
    if contract.post_model_composition == "none":
        if not np.isfinite(candidate).all():
            raise RuntimeError("Hydrology decoder produced non-finite terrain")
        return candidate.copy(), np.zeros(candidate.shape, dtype=np.float32)
    return compose_refinement_training_target(
        candidate,
        base,
        geometry,
        contract=contract,
    )


def parameterize_refinement_residual(
    residual_signed_sqrt: np.ndarray,
    geometry: FineHydrologyGeometry,
    *,
    contract: HydrologyDecoderContract = DEFAULT_HYDROLOGY_DECODER,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict learned residual support to decoder-owned terrain.

    V4 assigns planned river anchors and lake interiors to the deterministic
    synthesis base. The model retains full support on banks, shorelines,
    valleys, and surrounding terrain. This happens in residual space before
    terrain synthesis; it is not an elevation correction or terrain repair.
    """

    contract.validate()
    geometry.validate()
    residual = np.asarray(residual_signed_sqrt, dtype=np.float32)
    if residual.shape != geometry.freedom_mask.shape:
        raise ValueError("Residual and fine hydrology geometry must align")
    if not np.isfinite(residual).all():
        raise RuntimeError("Hydrology decoder produced a non-finite residual")
    if contract.residual_support == "unconstrained":
        return residual.copy(), np.zeros(residual.shape, dtype=np.float32)
    immutable = geometry.channel_centerline_mask | geometry.lake_mask
    supported = residual.copy()
    suppressed = np.zeros(residual.shape, dtype=np.float32)
    suppressed[immutable] = residual[immutable]
    supported[immutable] = 0.0
    return supported, suppressed


def compose_refinement_training_target(
    candidate_elevation_m: np.ndarray,
    base_elevation_m: np.ndarray,
    geometry: FineHydrologyGeometry,
    *,
    contract: HydrologyDecoderContract = DEFAULT_HYDROLOGY_DECODER,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a supervised target with planned anchors and lake-bed ceilings.

    V3 uses this only while preparing its target and exact fine synthesis base;
    it is never an inference-time terrain repair.
    """

    contract.validate()
    geometry.validate()
    candidate = np.asarray(candidate_elevation_m, dtype=np.float32)
    base = np.asarray(base_elevation_m, dtype=np.float32).copy()
    if candidate.shape != geometry.freedom_mask.shape or base.shape != candidate.shape:
        raise ValueError("Candidate, base, and fine hydrology geometry must align")
    anchors = geometry.channel_centerline_mask
    base[anchors] = geometry.target_bed_elevation_m[anchors]
    constrained = base + geometry.freedom_mask * (candidate - base)
    lakes = geometry.lake_mask
    if np.any(lakes):
        maximum_lake_bed = (
            geometry.water_surface_elevation_m - contract.lake_minimum_depth_m
        )
        constrained[lakes] = np.minimum(
            constrained[lakes], maximum_lake_bed[lakes]
        )
    if not np.isfinite(constrained).all():
        raise RuntimeError("Hydrology decoder composition produced non-finite terrain")
    return constrained.astype(np.float32), (candidate - constrained).astype(np.float32)


def _refine_lake_coverage(lake_mask: np.ndarray, refinement: int) -> np.ndarray:
    lakes = np.asarray(lake_mask, dtype=np.float32)
    if not np.any(lakes >= 0.5):
        return np.zeros(
            (lakes.shape[0] * refinement, lakes.shape[1] * refinement),
            dtype=np.float32,
        )
    refined = scipy.ndimage.zoom(
        lakes,
        zoom=refinement,
        order=1,
        mode="nearest",
        grid_mode=True,
        prefilter=False,
    )
    expected = (lakes.shape[0] * refinement, lakes.shape[1] * refinement)
    if refined.shape != expected:
        refined = refined[: expected[0], : expected[1]]
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def _complete_lake_surfaces(
    lake_mask: np.ndarray,
    supplied_surface_m: np.ndarray | None,
    terrain_elevation_m: np.ndarray | None,
) -> np.ndarray:
    lakes = np.asarray(lake_mask, dtype=bool)
    if supplied_surface_m is None:
        surfaces = np.full(lakes.shape, np.nan, dtype=np.float32)
    else:
        surfaces = np.asarray(supplied_surface_m, dtype=np.float32).copy()
        if surfaces.shape != lakes.shape:
            raise ValueError("Lake surfaces and hydrology must align")
    terrain = None
    if terrain_elevation_m is not None:
        terrain = np.asarray(terrain_elevation_m, dtype=np.float32)
        if terrain.shape != lakes.shape:
            raise ValueError("Lake fallback terrain and hydrology must align")
    labels, count = scipy.ndimage.label(lakes)
    for label in range(1, count + 1):
        component = labels == label
        known = surfaces[component & np.isfinite(surfaces)]
        if known.size:
            level = float(np.median(known))
        elif terrain is not None and np.any(component & np.isfinite(terrain)):
            level = float(np.median(terrain[component & np.isfinite(terrain)]))
        else:
            raise ValueError(
                "Every lake component requires a supplied surface or fallback terrain"
            )
        surfaces[component] = level
    surfaces[~lakes] = np.nan
    return surfaces


def _refine_lake_surfaces(
    coarse_lake_mask: np.ndarray,
    coarse_surface_m: np.ndarray,
    fine_lake_mask: np.ndarray,
    refinement: int,
) -> np.ndarray:
    """Extend component levels through anti-aliased fine shoreline cells.

    Fine lake coverage is bilinear, so a fine cell can cross the 0.5 threshold
    just outside the nearest-neighbour footprint of a coarse lake cell. Extend
    every coarse component level to its nearest coarse lake before the 3x
    nearest-neighbour rasterization. This preserves level surfaces and avoids
    inventing a terrain-derived level in the shoreline fringe.
    """

    coarse_lakes = np.asarray(coarse_lake_mask, dtype=bool)
    surfaces = np.asarray(coarse_surface_m, dtype=np.float32)
    fine_lakes = np.asarray(fine_lake_mask, dtype=bool)
    expected = tuple(size * refinement for size in coarse_lakes.shape)
    if surfaces.shape != coarse_lakes.shape or fine_lakes.shape != expected:
        raise ValueError("Lake masks and surfaces do not align for refinement")
    if not np.any(coarse_lakes):
        return np.full(expected, np.nan, dtype=np.float32)
    _, nearest = scipy.ndimage.distance_transform_edt(
        ~coarse_lakes, return_indices=True
    )
    extended = surfaces[nearest[0], nearest[1]]
    if not np.isfinite(extended).all():
        raise ValueError("Completed coarse lake surfaces must be finite on support")
    refined = scipy.ndimage.zoom(
        extended,
        zoom=refinement,
        order=0,
        mode="nearest",
        grid_mode=True,
        prefilter=False,
    ).astype(np.float32)
    refined[~fine_lakes] = np.nan
    return refined
