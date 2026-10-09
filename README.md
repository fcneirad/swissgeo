# swissgeo

Swiss geodesy toolkit: **WGS84 ↔ LV95 coordinate transforms**, **COFS municipality
lookup**, and **elevation queries** — fully offline, with a runtime that needs only
**numpy + the Python standard library**.

No `pyproj`, `geopandas`, `shapely`, `rasterio`, `fiona`, or network access is used
at runtime. All geospatial data is pre-converted to compact numpy formats and read
locally.

## Features

| Function | Input | Output | Accuracy |
|---|---|---|---|
| `to_lv95(lon, lat)` | WGS84 (EPSG:4326) | LV95 (EPSG:2056) easting/northing | < 1 mm vs pyproj |
| `to_wgs84(e, n)` | LV95 (EPSG:2056) | WGS84 (EPSG:4326) lon/lat | < 1 mm vs pyproj |
| `to_lv03(lon, lat)` | WGS84 (EPSG:4326) | LV03 / CH1903 easting/northing | exact offset from LV95 |
| `lv03_to_wgs84(e, n)` | LV03 / CH1903 | WGS84 (EPSG:4326) lon/lat | < 1 mm vs pyproj |
| `lv95_to_lv03` / `lv03_to_lv95` | LV95 ↔ LV03 | the other grid | exact (±2'000'000 / ±1'000'000 m) |
| `CofsLookup.from_lv95(e, n)` | LV95 point | COFS code + name + canton | exact polygon match (swissBOUNDARIES3D) |
| `ElevationLookup.from_lv95(e, n)` | LV95 point | metres a.s.l. (bilinear) | 10 m grid (swissALTIRegio) |

All functions accept either a single scalar pair or a sequence of pairs.

## Quick start

```python
from swissgeo import to_lv95, to_wgs84, CofsLookup, ElevationLookup

# 1) Coordinate transforms (no data needed)
e, n = to_lv95(7.4475, 46.9481)          # Bern centre -> LV95
print(e, n)                              # ~2600675, ~1199668

lon, lat = to_wgs84(e, n)                # back to WGS84

# 2) COFS municipality code (needs local data)
cofs = CofsLookup(data_dir="data/cofs_lv95")
m = cofs.from_lv95(e, n)
print(m.cofs, m.name, m.canton)          # 0351 Bern 2

# 3) Elevation (needs local data)
elev = ElevationLookup(data_dir="data/elevation_10m")
ev = elev.from_lv95(e, n)
print(ev.value)                          # ~540 m a.s.l.
```

Command line:

```bash
swissgeo transform 7.4475 46.9481                 # WGS84 -> LV95
swissgeo transform 7.4475 46.9481 --to lv03       # WGS84 -> LV03 (CH1903)
swissgeo transform --to wgs84 2600675 1199668     # LV95 -> WGS84
swissgeo transform --from lv03 --to lv95 600675 199668   # LV03 -> LV95
swissgeo cofs 2600675 1199668                     # -> COFS 0351 Bern (canton 2)
swissgeo elevation 2600675 1199668                # -> ~540 m a.s.l.
swissgeo data                                     # check local datasets are present
swissgeo fetch-data                               # one-time: download + convert data
```

## Data

The library is **offline by design**: it never makes network requests. It reads two
pre-converted datasets from `data/`:

* `data/cofs_lv95/` — municipality polygons (numpy arrays + a 5 km spatial grid
  index) derived from the official swisstopo *swissBOUNDARIES3D* GeoPackage
  (layer `tlm_hoheitsgebiet`, field `bfs_nummer` = COFS/SFOS code).
* `data/elevation_10m/` — the swisstopo *swissALTIRegio* DEM as a memory-mapped
  float32 grid (10 m resolution, all of Switzerland).

To (re)produce these from the raw swisstopo files, run the one-time setup command.
It fetches any missing raw files from `data.geo.admin.ch` (existing files are never
overwritten) and converts them:

```bash
swissgeo fetch-data                    # COFS + elevation
swissgeo fetch-data --skip-elevation   # COFS only
swissgeo fetch-data --no-download      # convert already-present raw files only
```

**No extra packages are installed into your environment.** The COFS (GeoPackage)
step uses only the Python standard library (`sqlite3` + `struct`). The elevation
(GeoTIFF) step needs `rasterio`; if it is not importable in your interpreter, a
*throwaway venv* is created on top of it (`--system-site-packages`, so numpy and
swissgeo are inherited), `rasterio` is installed **only** into that temp venv, the
conversion runs there, and the temp venv is deleted afterwards. Your environment —
and its dependency set — is left completely untouched.

The equivalent development script does the same thing:

```bash
python scripts/convert_data.py [--data-dir data] [--skip-elevation] [--no-download]
```

Raw inputs (official swisstopo distribution on `data.geo.admin.ch`):

* `swissBOUNDARIES3D_1_5_LV95_LN02.gpkg` (municipality boundaries, EPSG:2056)
  <https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/swissboundaries3d_2026-01/swissboundaries3d_2026-01_2056_5728.gpkg.zip>
* `swissaltiregio_2056_5728.tif` (elevation, EPSG:2056, 10 m, ~10 GB)
  <https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/swissaltiregio/swissaltiregio_2056_5728.tif>

## How the transform works

The WGS84 → LV95 chain is implemented in pure numpy (`swissgeo/transforms.py`):

```
WGS84 (lon, lat)
  -> ECEF on the WGS84 ellipsoid
  -> Helmert translation to CH1903+   (EPSG:1676: x=674.374, y=15.056, z=405.346)
  -> ECEF on the Bessel 1841 ellipsoid
  -> geodetic (lat, lon)              (iterative ECEF inverse)
  -> LV95 (easting, northing)         (Hotine Oblique Mercator "somerc")
```

The `somerc` equations are taken from the PROJ source (`src/projections/somerc.cpp`)
and validated against pyproj to < 1 mm over the whole Swiss territory. The inverse
chain reverses these steps.

**LV03 (CH1903)** uses the *identical* projection as LV95 — only the
false easting/northing differ (600'000 / 200'000 instead of 2'600'000 / 1'200'000).
It is therefore related to LV95 by a pure, exact offset:
`LV03 = LV95 − (2'000'000, 1'000'000)`. The helpers `to_lv03`, `lv03_to_wgs84`,
`lv95_to_lv03` and `lv03_to_lv95` implement this with no approximation.

## Project layout

```
swissgeo/
  transforms.py    pure-numpy WGS84 <-> LV95 (ECEF + Helmert + somerc)
  cofs.py          COFS lookup: numpy polygons + ray-casting point-in-polygon
  elevation.py     elevation lookup: memory-mapped .npy grid + bilinear interp
  data.py          DataStore: resolves/validates the local data directories
  setup_data.py    one-time download + conversion (shared by CLI and script)
  cli.py           command-line interface
scripts/
  convert_data.py  thin wrapper over swissgeo.setup_data
data/              converted datasets (cofs_lv95/, elevation_10m/) + raw inputs
tests/             pytest suite (synthetic + real-data tests)
```

## Dependencies

* **Runtime:** `numpy` only.
* **One-time data setup** (`swissgeo fetch-data`): no extra packages are installed
  into your environment. The COFS step is stdlib-only; if `rasterio` is missing for
  the elevation step it is provisioned in a throwaway venv that is deleted afterwards.
* **Development:** `pytest` (`pip install -e ".[dev]"`).

## Tests

```bash
python -m pytest
```

The suite validates the transforms against pyproj (when available) over a dense grid,
checks COFS point-in-polygon behaviour on synthetic boxes and a polygon-with-hole,
verifies bilinear interpolation exactly on a planar synthetic surface, and — when the
real data is present — asserts Bern → COFS `0351` and ~540 m elevation.
