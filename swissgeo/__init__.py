"""swissgeo -- Swiss geodesy toolkit (fully offline, numpy + stdlib only).

Quick start
-----------
>>> from swissgeo import to_lv95, to_wgs84
>>> e, n = to_lv95(7.5892, 46.9413)      # Bern (WGS84) -> LV95
>>> round(e), round(n)
(2611464, 1198923)

Municipality (COFS) and elevation lookups use local data produced by
``scripts/convert_data.py`` from the official swisstopo files:

>>> from swissgeo import CofsLookup, ElevationLookup
>>> lookup = CofsLookup(data_dir="data/cofs_lv95")
>>> m = lookup.from_lv95(2611464, 1198923)   # Bern
>>> m.cofs
'0351'

>>> elev = ElevationLookup(data_dir="data/elevation_10m").from_lv95(2611464, 1198923)
>>> round(elev.value)
791
"""

from __future__ import annotations

from .transforms import (
    LV03_SWISS_BOUNDS,
    LV95_SWISS_BOUNDS,
    is_valid_lv03,
    is_valid_lv95,
    lv03_to_lv95,
    lv03_to_wgs84,
    lv95_to_lv03,
    to_lv03,
    to_lv95,
    to_wgs84,
    transform,
)
from .cofs import CofsLookup, Municipality
from .elevation import Elevation, ElevationLookup
from .data import DataStore

__version__ = "0.2.0"

__all__ = [
    # transforms
    "to_lv95",
    "to_wgs84",
    "to_lv03",
    "lv03_to_wgs84",
    "lv95_to_lv03",
    "lv03_to_lv95",
    "transform",
    "is_valid_lv95",
    "is_valid_lv03",
    "LV95_SWISS_BOUNDS",
    "LV03_SWISS_BOUNDS",
    # cofs
    "CofsLookup",
    "Municipality",
    # elevation
    "ElevationLookup",
    "Elevation",
    # data
    "DataStore",
    "__version__",
]
