"""Elevation (altitude) lookup from LV95 coordinates.

Reads a digital elevation model stored as a memory-mapped numpy array (``.npy``)
in EPSG:2056, produced by ``scripts/convert_data.py`` from the official swisstopo
*swissALTIRegio* GeoTIFF (10 m resolution). Values are bilinearly interpolated to
the requested point. **No rasterio, no network** — only numpy + stdlib at runtime.

The returned :class:`Elevation` carries the height in metres above sea level plus
the source and the native grid resolution.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["Elevation", "ElevationLookup"]


@dataclass(frozen=True)
class Elevation:
    """An elevation measurement in metres above sea level (m a.s.l.)."""

    value: float
    source: str
    resolution_m: Optional[float] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        res = f" @ {self.resolution_m:g} m grid" if self.resolution_m else ""
        return f"{self.value:.2f} m a.s.l. ({self.source}{res})"


class ElevationLookup:
    """Resolve LV95 coordinates to an elevation in metres above sea level.

    Parameters
    ----------
    data_dir:
        Path to the directory produced by ``scripts/convert_data.py`` containing
        ``grid.npy`` (float32, H x W) and ``meta.json``. The grid is opened with
        ``np.memmap`` so only the touched rows are read from disk — a 10 m DEM of
        all Switzerland (~9.5 GB) costs no RAM beyond the queried window.

    Examples
    --------
    >>> lookup = ElevationLookup(data_dir="data/elevation_10m")
    >>> elev = lookup.from_lv95(2611464, 1198923)   # Bern
    >>> round(elev.value)
    791
    """

    def __init__(self, data_dir: Optional[str] = None) -> None:
        if data_dir is None:
            # Default: resolve via DataStore so $SWISSGEO_DATA is honoured
            # (falls back to the package-relative <project>/data/elevation_10m).
            from .data import DataStore

            data_dir = DataStore().elevation_path
        self.data_dir = data_dir
        self._load(data_dir)

    # ------------------------------------------------------------------ load
    def _load(self, path: str) -> None:
        grid_path = os.path.join(path, "grid.npy")
        if not os.path.exists(grid_path):
            raise FileNotFoundError(
                f"Elevation grid not found: {grid_path}\n"
                "Run `python scripts/convert_data.py` to produce it."
            )
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            self._meta: dict = json.load(f)
        # Memory-mapped read-only access via np.load (handles the .npy header and
        # maps only the touched pages): no full-grid RAM cost.
        self._grid = np.load(grid_path, mmap_mode="r")

    # ------------------------------------------------------------------ query
    def _query_local(self, easting: float, northing: float) -> Optional[Elevation]:
        origin_e = self._meta["origin_e"]   # top-left corner easting
        origin_n = self._meta["origin_n"]   # top-left corner northing
        px = self._meta["pixel_size_m"]

        # Fractional pixel coordinates in *corner* space (top-left origin, y down).
        col_f = (easting - origin_e) / px
        row_f = (origin_n - northing) / px

        # Raster values are defined at pixel *centers*: shift into center space.
        cx = col_f - 0.5
        cy = row_f - 0.5

        i0 = int(np.floor(cx))
        j0 = int(np.floor(cy))
        h, w = self._grid.shape
        # Bilinear needs a 2x2 block fully inside the raster.
        if not (0 <= i0 < w - 1 and 0 <= j0 < h - 1):
            return None

        # Read only the 2x2 window from disk.
        win = self._grid[j0:j0 + 2, i0:i0 + 2]
        v00, v01 = float(win[0, 0]), float(win[0, 1])
        v10, v11 = float(win[1, 0]), float(win[1, 1])

        # Fractional offsets within the 2x2 block.
        fx = cx - i0
        fy = cy - j0

        value = _bilinear_nan(v00, v01, v10, v11, fx, fy)
        if value is None:
            return None

        return Elevation(
            value=value,
            source=os.path.basename(self._meta.get("source", "grid.npy")),
            resolution_m=px,
        )

    # ------------------------------------------------------------------ public
    def from_lv95(self, easting: float, northing: float) -> Optional[Elevation]:
        """Return the :class:`Elevation` for an LV95 point, or ``None`` if unknown."""
        return self._query_local(easting, northing)

    def from_wgs84(self, lon: float, lat: float) -> Optional[Elevation]:
        """Convenience wrapper: convert WGS84 to LV95 then read the elevation."""
        from .transforms import to_lv95

        e, n = to_lv95(lon, lat)
        return self.from_lv95(e, n)


def _bilinear_nan(v00: float, v01: float, v10: float, v11: float,
                  fx: float, fy: float) -> Optional[float]:
    """Bilinear interpolation with NaN (nodata) handling; falls back to nearest."""
    vals = [v for v in (v00, v01, v10, v11) if np.isfinite(v)]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]

    def lerp(a: float, b: float, t: float) -> float:
        if not np.isfinite(a):
            return b
        if not np.isfinite(b):
            return a
        return a + (b - a) * t

    top = lerp(v00, v01, fx)
    bot = lerp(v10, v11, fx)
    return lerp(top, bot, fy)
