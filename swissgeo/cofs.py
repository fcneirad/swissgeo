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

from .transforms import LV95_SWISS_BOUNDS, is_valid_lv95

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
            # Default: resolve via DataStore so $SWISSGEO_DATA is honoured
            # (falls back to the package-relative <project>/data/cofs_lv95).
            from .data import DataStore

            data_dir = DataStore().cofs_path
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

    # ------------------------------------------------- vectorised batch lookup
    def _cell_to_polys(self) -> dict:
        """Lazily build the inverse grid index: cell_id -> polygon indices."""
        inv = getattr(self, "_cell_inv", None)
        if inv is None:
            cells = self._cells
            starts = self._cell_start
            # For each CSR position p, the owning polygon.
            polys_of_pos = (np.searchsorted(starts, np.arange(cells.size), side="right") - 1
                            if cells.size else np.zeros(0, dtype=np.int32))
            inv = {}
            for cell_id in np.unique(cells):
                mask = cells == cell_id
                inv[int(cell_id)] = np.unique(polys_of_pos[mask])
            self._cell_inv = inv
        return inv

    def from_lv95_batch(
        self, eastings, northings
    ) -> list:
        """Vectorised :meth:`from_lv95` for arrays of points.

        Parameters
        ----------
        eastings, northings:
            1-D sequences (or numpy arrays) of LV95 coordinates, same length.

        Returns
        -------
        List of :class:`Municipality` or ``None`` in input order. Points outside
        the Swiss LV95 extent are rejected without any polygon test. The result
        is identical to calling :meth:`from_lv95` per point (first containing
        candidate polygon wins).

        Notes
        -----
        Point-in-polygon uses the same even-odd ray-casting rule as the scalar
        path, vectorised with numpy over (points x edges) blocks.
        """
        e = np.asarray(eastings, dtype=np.float64).ravel()
        n = np.asarray(northings, dtype=np.float64).ravel()
        if e.shape != n.shape:
            raise ValueError("eastings and northings must have the same length")

        out: list = [None] * e.size
        if e.size == 0:
            return out

        b = LV95_SWISS_BOUNDS
        in_bounds = (
            (e >= b["min_e"]) & (e <= b["max_e"])
            & (n >= b["min_n"]) & (n <= b["max_n"])
        )
        idx = np.nonzero(in_bounds)[0]
        if idx.size == 0:
            return out

        # Group in-bounds points by grid cell.
        cell_size = self._meta["cell_size_m"]
        e0 = self._meta["grid_origin_e"]
        n0 = self._meta["grid_origin_n"]
        c = ((e[idx] - e0) // cell_size).astype(np.int64)
        r = ((n[idx] - n0) // cell_size).astype(np.int64)
        cell_ids = r * 100000 + c

        inv = self._cell_to_polys()
        unique_cells, inverse = np.unique(cell_ids, return_inverse=True)

        verts = self._verts
        ring_start = self._ring_start
        # NOTE: ring_start may or may not carry a terminator entry (one past the
        # last vertex); use the same boundary test as the scalar path.
        n_ring_entries = len(ring_start)

        for ci, cell_id in enumerate(unique_cells):
            point_idx = idx[inverse == ci]
            candidates = inv.get(cell_id)
            if candidates is None or candidates.size == 0:
                continue
            pts_e = e[point_idx]
            pts_n = n[point_idx]
            # Candidates are tested in the same order as the scalar path; the
            # first containing polygon wins per point.
            unresolved = np.ones(pts_e.size, dtype=bool)
            for poly in candidates:
                if not unresolved.any():
                    break
                pi = int(poly)
                bx = self._bbox[pi]
                # Bbox reject per (still unresolved) point.
                in_bbox = (
                    (pts_e >= bx[0]) & (pts_e <= bx[2])
                    & (pts_n >= bx[1]) & (pts_n <= bx[3])
                    & unresolved
                )
                if not in_bbox.any():
                    continue
                sub_pos = np.nonzero(in_bbox)[0]
                se = pts_e[sub_pos]
                sn = pts_n[sub_pos]

                parity = np.zeros(se.size, dtype=np.int8)
                r0 = int(self._poly_start[pi])
                r1 = int(self._poly_start[pi + 1])
                for ri in range(r0, r1):
                    vs = int(ring_start[ri])
                    ve = int(ring_start[ri + 1]) if ri + 1 < n_ring_entries else len(verts)
                    ring = verts[vs:ve].astype(np.float64)
                    nv = ring.shape[0]
                    if nv < 3:
                        continue
                    ax = ring[:, 0]
                    ay = ring[:, 1]
                    bx_e = np.roll(ax, -1)
                    by_e = np.roll(ay, -1)
                    # Chunk points to bound the (P x E) temporary.
                    for s in range(0, se.size, 4096):
                        px = se[s:s + 4096][:, None]
                        py = sn[s:s + 4096][:, None]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            x_int = (bx_e - ax) * (py - ay) / (by_e - ay) + ax
                        crosses = ((ay > py) != (by_e > py)) & (px < x_int)
                        parity[s:s + 4096] ^= (crosses.sum(axis=1) % 2).astype(np.int8)

                hit = np.nonzero(parity)[0]
                if hit.size == 0:
                    continue
                p = self._props[pi]
                muni = Municipality(cofs=p["cofs"], name=p.get("name"), canton=p.get("canton"))
                for j in hit:
                    out[int(point_idx[sub_pos[j]])] = muni
                unresolved[sub_pos[hit]] = False

        return out

    def from_wgs84(self, lon: float, lat: float) -> Optional[Municipality]:
        """Convenience wrapper: convert WGS84 to LV95 then look up the COFS code."""
        from .transforms import to_lv95

        e, n = to_lv95(lon, lat)
        return self.from_lv95(e, n)
