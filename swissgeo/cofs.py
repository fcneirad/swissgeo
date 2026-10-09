"""COFS (Swiss municipality code) lookup from LV95 coordinates.

A *COFS* code is the 4-digit identifier of a Swiss municipality (Gemeinde), e.g.
``0351`` for Bern. This module resolves an LV95 point to its municipality using
a pre-converted numpy polygon dataset with a spatial grid index — **no geopandas,
no shapely, no network**.

The data is produced by ``scripts/convert_data.py`` from the official swisstopo
*swissBOUNDARIES3D* GeoPackage (layer ``tlm_hoheitsgebiet``, field ``bfs_nummer``).

Point-in-polygon uses the ray-casting (even-odd) algorithm, vectorised with numpy.
A 5 km grid index narrows candidates before the exact test, keeping lookups fast
(~20 candidate polygons on average for a typical Swiss point).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .transforms import is_valid_lv95

__all__ = ["Municipality", "CofsLookup"]


@dataclass(frozen=True)
class Municipality:
    """A Swiss municipality identified by its COFS code."""

    cofs: str
    name: Optional[str] = None
    canton: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        parts = [f"COFS {self.cofs}"]
        if self.name:
            parts.append(self.name)
        if self.canton:
            parts.append(f"(canton {self.canton})")
        return " ".join(parts)


class CofsLookup:
    """Resolve LV95 coordinates to their COFS municipality code.

    Parameters
    ----------
    data_dir:
        Path to the directory produced by ``scripts/convert_data.py`` containing
        ``verts.npy``, ``ring_start.npy``, ``poly_start.npy``, ``bbox.npy``,
        ``cells.npy``, ``cell_start.npy``, ``props.json``, and ``meta.json``.

    Examples
    --------
    >>> lookup = CofsLookup(data_dir="data/cofs_lv95")
    >>> m = lookup.from_lv95(2611464, 1198923)   # Bern
    >>> m.cofs
    '0351'
    """

    def __init__(self, data_dir: Optional[str] = None) -> None:
        if data_dir is None:
            # Default: look relative to the package's parent (project root).
            here = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(os.path.dirname(here), "data", "cofs_lv95")
        self.data_dir = data_dir
        self._load(data_dir)

    # ------------------------------------------------------------------ load
    def _load(self, path: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"COFS data directory not found: {path}\n"
                "Run `python scripts/convert_data.py` to produce it."
            )
        self._verts = np.load(os.path.join(path, "verts.npy"))          # (V, 2) float32
        self._ring_start = np.load(os.path.join(path, "ring_start.npy"))  # (R,) int32
        self._poly_start = np.load(os.path.join(path, "poly_start.npy"))  # (P+1,) int32
        self._bbox = np.load(os.path.join(path, "bbox.npy"))            # (P, 4) float32
        self._cells = np.load(os.path.join(path, "cells.npy"))          # (K,) int32
        self._cell_start = np.load(os.path.join(path, "cell_start.npy"))  # (P+1,) int32
        with open(os.path.join(path, "props.json"), encoding="utf-8") as f:
            self._props: list[dict] = json.load(f)
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            self._meta: dict = json.load(f)

    # ------------------------------------------------------------- grid index
    def _candidate_indices(self, easting: float, northing: float) -> np.ndarray:
        """Return polygon indices whose grid cell contains the point."""
        cell_size = self._meta["cell_size_m"]
        e0 = self._meta["grid_origin_e"]
        n0 = self._meta["grid_origin_n"]
        c = int((easting - e0) // cell_size)
        r = int((northing - n0) // cell_size)
        cell_id = r * 100000 + c

        # Find all polygons that reference this cell.
        # cells is sorted by polygon (CSR-like via cell_start).
        # We need the inverse: for a given cell_id, which polygons contain it?
        # Since cells are stored per-polygon in CSR order, we search with np.where.
        # For performance with ~10k total cell assignments, a direct search is fine.
        mask = self._cells == cell_id
        if not mask.any():
            return np.array([], dtype=np.int32)
        # Map cell positions back to polygon indices.
        # cell_start[i]..cell_start[i+1] are the cells for polygon i.
        # We need: for each position p where cells[p]==cell_id, which polygon owns p?
        # Use searchsorted on cell_start.
        positions = np.where(mask)[0]
        poly_idx = np.searchsorted(self._cell_start, positions, side="right") - 1
        return np.unique(poly_idx)

    # ------------------------------------------------------- point-in-polygon
    @staticmethod
    def _point_in_ring(x: float, y: float, ring: np.ndarray) -> bool:
        """Ray-casting (even-odd rule) for a single ring.

        ``ring`` is an open polygonal chain (N, 2) with no repeated closing vertex.
        """
        n = len(ring)
        if n < 3:
            return False
        inside = False
        x0, y0 = ring[0]
        for i in range(1, n + 1):
            x1, y1 = ring[i % n]
            # Check if the ray from (x,y) going in +x direction crosses edge (x0,y0)-(x1,y1).
            if ((y0 > y) != (y1 > y)):
                x_intersect = (x1 - x0) * (y - y0) / (y1 - y0) + x0
                if x < x_intersect:
                    inside = not inside
            x0, y0 = x1, y1
        return inside

    def _point_in_polygon(self, easting: float, northing: float, poly_idx: int) -> bool:
        """Test if a point is inside polygon ``poly_idx`` (handles holes)."""
        r0 = self._poly_start[poly_idx]
        r1 = self._poly_start[poly_idx + 1]
        # Even-odd rule across all rings: count crossings; odd => inside.
        crossings = 0
        for ri in range(r0, r1):
            vs = int(self._ring_start[ri])
            ve = int(self._ring_start[ri + 1]) if ri + 1 < len(self._ring_start) else len(self._verts)
            ring = self._verts[vs:ve]
            if self._point_in_ring(easting, northing, ring):
                crossings += 1
        return crossings % 2 == 1

    # ------------------------------------------------------------------ public
    def from_lv95(self, easting: float, northing: float) -> Optional[Municipality]:
        """Return the :class:`Municipality` for an LV95 point, or ``None``.

        Returns ``None`` when the point is outside Switzerland or no municipality
        boundary contains it (e.g. open water).
        """
        if not is_valid_lv95(easting, northing):
            return None

        candidates = self._candidate_indices(easting, northing)
        for idx in candidates:
            # Quick bbox reject.
            bx = self._bbox[idx]
            if not (bx[0] <= easting <= bx[2] and bx[1] <= northing <= bx[3]):
                continue
            if self._point_in_polygon(easting, northing, int(idx)):
                p = self._props[int(idx)]
                return Municipality(cofs=p["cofs"], name=p.get("name"), canton=p.get("canton"))
        return None

    def from_wgs84(self, lon: float, lat: float) -> Optional[Municipality]:
        """Convenience wrapper: convert WGS84 to LV95 then look up the COFS code."""
        from .transforms import to_lv95

        e, n = to_lv95(lon, lat)
        return self.from_lv95(e, n)
