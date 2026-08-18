"""Expandable, signed-coordinate catalog above immutable hydrology regions.

The atlas does not store large rasters. Each region remains its own resumable
surface/world-plan package; this catalog freezes their identities and the
continental drainage contracts joining them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Literal

import h5py
import numpy as np


ATLAS_SCHEMA_VERSION = 1
ATLAS_MANIFEST = "atlas_manifest.json"
ATLAS_CATALOG = "atlas.sqlite"
REGION_MACRO_CELLS = 256
MACRO_RESOLUTION_M = 7680.0


@dataclass(frozen=True, order=True)
class RegionKey:
    """Signed index of one 1966.08 km square detail region."""

    row: int
    col: int

    @property
    def macro_origin(self) -> tuple[int, int]:
        return self.row * REGION_MACRO_CELLS, self.col * REGION_MACRO_CELLS

    @property
    def planner_origin(self) -> tuple[int, int]:
        macro_row, macro_col = self.macro_origin
        return macro_row * 32, macro_col * 32


@dataclass(frozen=True)
class AtlasManifest:
    atlas_id: str
    world_seed: int
    schema_version: int = ATLAS_SCHEMA_VERSION
    region_macro_cells: int = REGION_MACRO_CELLS
    macro_resolution_m: float = MACRO_RESOLUTION_M
    planner_resolution_m: float = 240.0
    minecraft_horizontal_scale: float = 0.5
    minecraft_vertical_scale: float = 0.5
    minecraft_sea_level_y: int = 63

    def validate(self) -> None:
        if not self.atlas_id or "/" in self.atlas_id:
            raise ValueError("atlas_id must be a non-empty path-safe name")
        if self.schema_version != ATLAS_SCHEMA_VERSION:
            raise ValueError("Unsupported atlas schema version")
        if self.region_macro_cells != REGION_MACRO_CELLS:
            raise ValueError("This schema requires 256 macro cells per region")
        if self.macro_resolution_m != self.planner_resolution_m * 32:
            raise ValueError("Macro and planner grids must nest exactly 32x")


_REGION_TRANSITIONS = {
    "registered": {"surface_complete"},
    "surface_complete": {"routed"},
    "routed": {"validated"},
    "validated": {"published"},
    "published": set(),
}


class HydrologyAtlas:
    """Persistent catalog for progressive, immutable world expansion."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest_path = self.root / ATLAS_MANIFEST
        self.catalog_path = self.root / ATLAS_CATALOG

    @classmethod
    def create(cls, root: str | Path, manifest: AtlasManifest) -> "HydrologyAtlas":
        manifest.validate()
        atlas = cls(root)
        if atlas.root.exists() and any(atlas.root.iterdir()):
            raise FileExistsError(f"Atlas directory is not empty: {atlas.root}")
        atlas.root.mkdir(parents=True, exist_ok=True)
        atlas.manifest_path.write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with atlas.open_catalog() as connection:
            atlas._create_schema(connection)
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('atlas_id', ?)",
                (manifest.atlas_id,),
            )
            connection.commit()
        return atlas

    def read_manifest(self) -> AtlasManifest:
        manifest = AtlasManifest(**json.loads(self.manifest_path.read_text(encoding="utf-8")))
        manifest.validate()
        return manifest

    def open_catalog(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.catalog_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS macro_regions (
                region_row INTEGER NOT NULL,
                region_col INTEGER NOT NULL,
                macro_origin_row INTEGER NOT NULL,
                macro_origin_col INTEGER NOT NULL,
                artifact_path TEXT NOT NULL,
                artifact_sha256 TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('complete','frozen')),
                updated_utc TEXT NOT NULL,
                PRIMARY KEY(region_row, region_col)
            )"""
        )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS macro_drainage_basins (
                basin_id TEXT PRIMARY KEY,
                state TEXT NOT NULL CHECK(state IN ('frozen')),
                outlet_kind TEXT NOT NULL CHECK(outlet_kind IN ('ocean','boundary')),
                outlet_macro_row INTEGER NOT NULL,
                outlet_macro_col INTEGER NOT NULL,
                min_macro_row INTEGER NOT NULL,
                min_macro_col INTEGER NOT NULL,
                max_macro_row INTEGER NOT NULL,
                max_macro_col INTEGER NOT NULL,
                cell_count INTEGER NOT NULL,
                area_km2 REAL NOT NULL,
                maximum_accumulation_km2 REAL NOT NULL,
                frozen_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS macro_drainage_cells (
                basin_id TEXT NOT NULL REFERENCES macro_drainage_basins(basin_id),
                macro_row INTEGER NOT NULL,
                macro_col INTEGER NOT NULL,
                PRIMARY KEY(basin_id, macro_row, macro_col)
            );
            CREATE TABLE IF NOT EXISTS basin_codes (
                basin_id TEXT PRIMARY KEY,
                basin_code INTEGER NOT NULL UNIQUE CHECK(basin_code >= 0)
            );
            """
        )
        return connection

    def register_macro_region(
        self,
        key: RegionKey,
        artifact_path: str | Path,
        artifact_sha256: str,
    ) -> None:
        """Register one immutable 256x256 learned macro field."""

        path = Path(artifact_path).resolve()
        if not path.is_file() or len(artifact_sha256) != 64:
            raise ValueError("Macro artifact and SHA-256 identity are required")
        with self.open_catalog() as connection:
            existing = connection.execute(
                """SELECT artifact_path, artifact_sha256 FROM macro_regions
                   WHERE region_row=? AND region_col=?""",
                (key.row, key.col),
            ).fetchone()
            if existing is not None and existing != (str(path), artifact_sha256):
                raise ValueError(f"Macro region {key} is already immutable")
            connection.execute(
                """INSERT INTO macro_regions
                   (region_row, region_col, macro_origin_row, macro_origin_col,
                    artifact_path, artifact_sha256, state, updated_utc)
                   VALUES (?, ?, ?, ?, ?, ?, 'complete', ?)
                   ON CONFLICT(region_row, region_col) DO UPDATE SET
                     updated_utc=excluded.updated_utc""",
                (
                    key.row, key.col, *key.macro_origin, str(path),
                    artifact_sha256, _now(),
                ),
            )
            connection.commit()

    def register_surface(
        self,
        key: RegionKey,
        surface_file: str | Path,
    ) -> str:
        """Register a completed or in-progress integrated surface artifact."""

        manifest = self.read_manifest()
        surface_path = Path(surface_file).resolve()
        with h5py.File(surface_path, "r") as surface:
            provenance = json.loads(surface.attrs["provenance_json"])
            if int(provenance["seed"]) != manifest.world_seed:
                raise ValueError("Surface seed differs from atlas seed")
            if int(provenance["macro_cells"]) != manifest.region_macro_cells:
                raise ValueError("Surface region geometry differs from atlas geometry")
            complete = bool(surface.attrs.get("complete", False))
            completed_tiles = int(surface.attrs.get("completed_tile_count", 0))
            total_tiles = int(surface["completed_tiles"].size)
        state = "surface_complete" if complete else "registered"
        identity = stable_world_id(
            manifest.world_seed, "region", key.row, key.col
        )
        with self.open_catalog() as connection:
            existing = connection.execute(
                "SELECT surface_path FROM regions WHERE region_row=? AND region_col=?",
                (key.row, key.col),
            ).fetchone()
            if existing is not None and Path(existing[0]) != surface_path:
                raise ValueError(f"Region {key} is already bound to another surface")
            connection.execute(
                """INSERT INTO regions
                   (region_id, region_row, region_col, macro_origin_row,
                    macro_origin_col, state, surface_path, completed_tiles,
                    total_tiles, updated_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(region_row, region_col) DO UPDATE SET
                     state=excluded.state,
                     completed_tiles=excluded.completed_tiles,
                     total_tiles=excluded.total_tiles,
                     updated_utc=excluded.updated_utc""",
                (
                    identity, key.row, key.col, *key.macro_origin, state,
                    str(surface_path), completed_tiles, total_tiles, _now(),
                ),
            )
            connection.commit()
        return identity

    def transition_region(
        self,
        key: RegionKey,
        target: Literal["surface_complete", "routed", "validated", "published"],
        *,
        world_plan_path: str | Path | None = None,
    ) -> None:
        """Advance one immutable artifact through the publication lifecycle."""

        with self.open_catalog() as connection:
            row = connection.execute(
                "SELECT state FROM regions WHERE region_row=? AND region_col=?",
                (key.row, key.col),
            ).fetchone()
            if row is None:
                raise KeyError(key)
            current = row[0]
            if target == current:
                return
            if target not in _REGION_TRANSITIONS[current]:
                raise ValueError(f"Invalid region transition {current!r} -> {target!r}")
            if target == "routed" and world_plan_path is None:
                raise ValueError("Routing requires an immutable world-plan path")
            connection.execute(
                """UPDATE regions SET state=?, world_plan_path=COALESCE(?, world_plan_path),
                   updated_utc=? WHERE region_row=? AND region_col=?""",
                (
                    target,
                    None if world_plan_path is None else str(Path(world_plan_path).resolve()),
                    _now(), key.row, key.col,
                ),
            )
            connection.commit()

    def add_boundary_contract(
        self,
        *,
        source_region: RegionKey,
        destination_region: RegionKey,
        global_planner_row: int,
        global_planner_col: int,
        basin_id: str,
        upstream_area_m2: float,
        mean_discharge_m3s: float,
    ) -> str:
        """Freeze one directed cross-region river connection."""

        distance = abs(source_region.row - destination_region.row) + abs(
            source_region.col - destination_region.col
        )
        if distance != 1:
            raise ValueError("Boundary contracts require edge-adjacent regions")
        if upstream_area_m2 < 0 or mean_discharge_m3s < 0:
            raise ValueError("River contract quantities cannot be negative")
        manifest = self.read_manifest()
        contract_id = stable_world_id(
            manifest.world_seed,
            "portal",
            global_planner_row,
            global_planner_col,
            source_region.row,
            source_region.col,
            destination_region.row,
            destination_region.col,
        )
        with self.open_catalog() as connection:
            existing = connection.execute(
                """SELECT source_region_row, source_region_col,
                          destination_region_row, destination_region_col,
                          global_planner_row, global_planner_col, basin_id,
                          upstream_area_m2, mean_discharge_m3s
                   FROM boundary_contracts WHERE contract_id=?""",
                (contract_id,),
            ).fetchone()
            values = (
                source_region.row, source_region.col,
                destination_region.row, destination_region.col,
                int(global_planner_row), int(global_planner_col), basin_id,
                float(upstream_area_m2), float(mean_discharge_m3s),
            )
            if existing is not None:
                if existing != values:
                    raise ValueError(f"Boundary contract collision: {contract_id}")
                return contract_id
            connection.execute(
                """INSERT INTO boundary_contracts
                   (contract_id, source_region_row, source_region_col,
                    destination_region_row, destination_region_col,
                    global_planner_row, global_planner_col, basin_id,
                    upstream_area_m2, mean_discharge_m3s, frozen_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (contract_id, *values, _now()),
            )
            connection.commit()
        return contract_id

    def basin_code(self, basin_id: str) -> int:
        """Return a stable non-nodata uint32 code for a textual basin ID."""

        maximum = int(np.iinfo(np.uint32).max) - 1
        candidate = int.from_bytes(
            hashlib.blake2b(basin_id.encode("utf-8"), digest_size=4).digest(),
            "little",
        ) % maximum
        with self.open_catalog() as connection:
            existing = connection.execute(
                "SELECT basin_code FROM basin_codes WHERE basin_id=?", (basin_id,)
            ).fetchone()
            if existing is not None:
                return int(existing[0])
            while connection.execute(
                "SELECT 1 FROM basin_codes WHERE basin_code=?", (candidate,)
            ).fetchone() is not None:
                candidate = (candidate + 1) % maximum
            connection.execute(
                "INSERT INTO basin_codes(basin_id, basin_code) VALUES (?, ?)",
                (basin_id, candidate),
            )
            connection.commit()
        return candidate

    def freeze_continental_drainage(
        self,
        drainage,
        *,
        macro_origin_row: int,
        macro_origin_col: int,
    ) -> None:
        """Atomically persist one closed landmass and its macro catchments."""

        import numpy as np

        land = np.asarray(drainage.landmass_mask, dtype=bool)
        catchments = np.asarray(drainage.routing.catchment_id, dtype=np.uint32)
        if land.shape != catchments.shape or not np.any(land):
            raise ValueError("Continental drainage has invalid land/catchment geometry")
        if np.any(land[0]) or np.any(land[-1]) or np.any(land[:, 0]) or np.any(land[:, -1]):
            raise ValueError("Cannot persist a landmass that still touches coverage")
        land_rows, land_cols = np.nonzero(land)
        with self.open_catalog() as connection:
            with connection:
                connection.execute(
                    """INSERT INTO macro_landmasses
                       (landmass_id, state, min_macro_row, min_macro_col,
                        max_macro_row, max_macro_col, cell_count, frozen_utc)
                       VALUES (?, 'frozen', ?, ?, ?, ?, ?, ?)""",
                    (
                        drainage.landmass_id,
                        int(macro_origin_row + land_rows.min()),
                        int(macro_origin_col + land_cols.min()),
                        int(macro_origin_row + land_rows.max()),
                        int(macro_origin_col + land_cols.max()),
                        int(land.sum()),
                        _now(),
                    ),
                )
                for basin in drainage.basins:
                    basin_mask = land & (
                        catchments == int(basin.local_catchment_id)
                    )
                    rows, cols = np.nonzero(basin_mask)
                    connection.execute(
                        """INSERT INTO continental_basins
                           (basin_id, landmass_id, state, outlet_kind,
                            outlet_macro_row, outlet_macro_col,
                            min_macro_row, min_macro_col,
                            max_macro_row, max_macro_col, frozen_utc)
                           VALUES (?, ?, 'frozen', 'ocean', ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            basin.basin_id,
                            drainage.landmass_id,
                            basin.outlet_macro_row,
                            basin.outlet_macro_col,
                            int(macro_origin_row + rows.min()),
                            int(macro_origin_col + cols.min()),
                            int(macro_origin_row + rows.max()),
                            int(macro_origin_col + cols.max()),
                            _now(),
                        ),
                    )
                    connection.executemany(
                        """INSERT INTO basin_macro_cells
                           (basin_id, macro_row, macro_col) VALUES (?, ?, ?)""",
                        (
                            (
                                basin.basin_id,
                                int(macro_origin_row + row),
                                int(macro_origin_col + col),
                            )
                            for row, col in zip(rows, cols)
                        ),
                    )

    def freeze_macro_drainage_basin(
        self,
        basin,
        catchment_id: np.ndarray,
        land_mask: np.ndarray,
        *,
        macro_origin_row: int,
        macro_origin_col: int,
    ) -> None:
        """Persist one basin only after its upstream perimeter is closed."""

        if not basin.closed or basin.outlet_kind != "ocean":
            raise ValueError("Only closed ocean-draining basins can be frozen")
        catchments = np.asarray(catchment_id, dtype=np.uint32)
        land = np.asarray(land_mask, dtype=bool)
        if catchments.shape != land.shape:
            raise ValueError("Catchment and land rasters must align")
        mask = land & (catchments == int(basin.local_catchment_id))
        rows, cols = np.nonzero(mask)
        if rows.size != basin.cell_count:
            raise ValueError("Frozen basin cell count differs from analysis")
        with self.open_catalog() as connection:
            existing = connection.execute(
                "SELECT basin_id FROM macro_drainage_basins WHERE basin_id=?",
                (basin.basin_id,),
            ).fetchone()
            if existing is not None:
                return
            with connection:
                connection.execute(
                    """INSERT INTO macro_drainage_basins
                       (basin_id, state, outlet_kind, outlet_macro_row,
                        outlet_macro_col, min_macro_row, min_macro_col,
                        max_macro_row, max_macro_col, cell_count, area_km2,
                        maximum_accumulation_km2, frozen_utc)
                       VALUES (?, 'frozen', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        basin.basin_id, basin.outlet_kind,
                        basin.outlet_macro_row, basin.outlet_macro_col,
                        basin.min_macro_row, basin.min_macro_col,
                        basin.max_macro_row, basin.max_macro_col,
                        basin.cell_count, basin.area_km2,
                        basin.maximum_accumulation_km2, _now(),
                    ),
                )
                connection.executemany(
                    """INSERT INTO macro_drainage_cells
                       (basin_id, macro_row, macro_col) VALUES (?, ?, ?)""",
                    (
                        (
                            basin.basin_id,
                            int(macro_origin_row + row),
                            int(macro_origin_col + col),
                        )
                        for row, col in zip(rows, cols)
                    ),
                )

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE regions (
                region_id TEXT PRIMARY KEY,
                region_row INTEGER NOT NULL,
                region_col INTEGER NOT NULL,
                macro_origin_row INTEGER NOT NULL,
                macro_origin_col INTEGER NOT NULL,
                state TEXT NOT NULL CHECK(state IN
                    ('registered','surface_complete','routed','validated','published')),
                surface_path TEXT NOT NULL,
                world_plan_path TEXT,
                completed_tiles INTEGER NOT NULL,
                total_tiles INTEGER NOT NULL,
                updated_utc TEXT NOT NULL,
                UNIQUE(region_row, region_col)
            );
            CREATE TABLE continental_basins (
                basin_id TEXT PRIMARY KEY,
                landmass_id TEXT NOT NULL REFERENCES macro_landmasses(landmass_id),
                state TEXT NOT NULL CHECK(state IN ('provisional','frozen')),
                outlet_kind TEXT NOT NULL CHECK(outlet_kind IN ('ocean','endorheic','boundary')),
                outlet_macro_row INTEGER NOT NULL,
                outlet_macro_col INTEGER NOT NULL,
                min_macro_row INTEGER NOT NULL,
                min_macro_col INTEGER NOT NULL,
                max_macro_row INTEGER NOT NULL,
                max_macro_col INTEGER NOT NULL,
                frozen_utc TEXT
            );
            CREATE TABLE macro_landmasses (
                landmass_id TEXT PRIMARY KEY,
                state TEXT NOT NULL CHECK(state IN ('provisional','frozen')),
                min_macro_row INTEGER NOT NULL,
                min_macro_col INTEGER NOT NULL,
                max_macro_row INTEGER NOT NULL,
                max_macro_col INTEGER NOT NULL,
                cell_count INTEGER NOT NULL,
                frozen_utc TEXT
            );
            CREATE TABLE basin_macro_cells (
                basin_id TEXT NOT NULL REFERENCES continental_basins(basin_id),
                macro_row INTEGER NOT NULL,
                macro_col INTEGER NOT NULL,
                PRIMARY KEY(basin_id, macro_row, macro_col)
            );
            CREATE TABLE boundary_contracts (
                contract_id TEXT PRIMARY KEY,
                source_region_row INTEGER NOT NULL,
                source_region_col INTEGER NOT NULL,
                destination_region_row INTEGER NOT NULL,
                destination_region_col INTEGER NOT NULL,
                global_planner_row INTEGER NOT NULL,
                global_planner_col INTEGER NOT NULL,
                basin_id TEXT NOT NULL,
                upstream_area_m2 REAL NOT NULL,
                mean_discharge_m3s REAL NOT NULL,
                frozen_utc TEXT NOT NULL,
                UNIQUE(global_planner_row, global_planner_col, basin_id)
            );
            CREATE TABLE expansions (
                expansion_id TEXT PRIMARY KEY,
                requested_region_row INTEGER NOT NULL,
                requested_region_col INTEGER NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('planning','ready','published','failed')),
                created_utc TEXT NOT NULL,
                completed_utc TEXT
            );
            """
        )


def stable_world_id(seed: int, kind: str, *coordinates: int) -> str:
    """Backend-independent identity stable across expansion order and machines."""

    payload = json.dumps(
        [int(seed) & 0xFFFFFFFFFFFFFFFF, str(kind), *map(int, coordinates)],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
