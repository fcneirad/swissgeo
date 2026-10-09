"""Locate the local swissgeo data directories.

The library is fully offline: all data must be present on disk. The
:class:`DataStore` resolves where the converted datasets live and reports whether
they are ready. It does **not** download anything (the official swisstopo services
are not reachable from this network, and the design goal is zero online requests).

To (re)produce the data from the raw swisstopo files, run::

    python scripts/convert_data.py

which reads ``data/swissBOUNDARIES3D_*.gpkg`` and ``data/swissaltiregio_*.tif``
and writes ``data/cofs_lv95/`` and ``data/elevation_10m/``.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = ["DataStore", "DEFAULT_DATA_DIR"]


def _default_data_dir() -> str:
    """Default data location: $SWISSGEO_DATA, else <project>/data."""
    env = os.environ.get("SWISSGEO_DATA")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "data")


DEFAULT_DATA_DIR = _default_data_dir()


class DataStore:
    """Resolve the local paths of the swissgeo datasets.

    Parameters
    ----------
    data_dir:
        Directory containing ``cofs_lv95/`` and ``elevation_10m/``. Defaults to
        ``$SWISSGEO_DATA`` or the project's ``data/`` directory.

    Examples
    --------
    >>> store = DataStore()
    >>> cofs_dir = store.cofs_path          # .../data/cofs_lv95
    >>> elev_dir = store.elevation_path     # .../data/elevation_10m
    >>> from swissgeo import CofsLookup, ElevationLookup
    >>> lookup = CofsLookup(data_dir=cofs_dir)
    """

    def __init__(self, data_dir: Optional[str] = None) -> None:
        # Resolve the default at construction time so $SWISSGEO_DATA changes
        # (e.g. in tests) are picked up.
        self.data_dir = data_dir or _default_data_dir()

    # ------------------------------------------------------------------ paths
    @property
    def cofs_path(self) -> str:
        """Directory with the converted COFS polygon dataset."""
        return os.path.join(self.data_dir, "cofs_lv95")

    @property
    def elevation_path(self) -> str:
        """Directory with the converted elevation grid."""
        return os.path.join(self.data_dir, "elevation_10m")

    # ------------------------------------------------------------------ status
    def cofs_ready(self) -> bool:
        """True if the COFS dataset is present and complete."""
        d = self.cofs_path
        needed = ("verts.npy", "ring_start.npy", "poly_start.npy", "bbox.npy",
                  "cells.npy", "cell_start.npy", "props.json", "meta.json")
        return os.path.isdir(d) and all(os.path.exists(os.path.join(d, f)) for f in needed)

    def elevation_ready(self) -> bool:
        """True if the elevation grid is present."""
        d = self.elevation_path
        return os.path.isdir(d) and os.path.exists(os.path.join(d, "grid.npy")) \
            and os.path.exists(os.path.join(d, "meta.json"))

    def ensure_cofs(self) -> str:
        """Return the COFS data directory, raising if it is missing."""
        if not self.cofs_ready():
            raise FileNotFoundError(
                f"COFS dataset not found in {self.cofs_path}.\n"
                "Place the raw swisstopo files in the data directory and run:\n"
                "    python scripts/convert_data.py"
            )
        return self.cofs_path

    def ensure_elevation(self) -> str:
        """Return the elevation data directory, raising if it is missing."""
        if not self.elevation_ready():
            raise FileNotFoundError(
                f"Elevation dataset not found in {self.elevation_path}.\n"
                "Place the raw swisstopo files in the data directory and run:\n"
                "    python scripts/convert_data.py"
            )
        return self.elevation_path
