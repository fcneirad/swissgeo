"""One-time setup: download raw swisstopo files and convert them to runtime formats.

This module is the single source of truth for the data pipeline. It is used by
both ``scripts/convert_data.py`` (development script) and the
``swissgeo fetch-data`` CLI command.

Design goals
------------
* **COFS** (GeoPackage -> numpy polygons): Python standard library only
  (``sqlite3`` + ``struct``). No extra packages, ever.
* **Elevation** (GeoTIFF -> float32 grid): needs ``rasterio``. If the current
  interpreter does not have rasterio, we do NOT install it into your
  environment: instead a *throwaway venv* is created on top of it
  (``--system-site-packages``, so numpy and swissgeo are inherited), rasterio
  is installed **only** into that temp venv, the conversion runs there in a
  subprocess, and the temp venv is deleted afterwards. Your environment is
  never modified.

Outputs (under ``data/``):

* ``cofs_lv95/``  -- municipality polygons as numpy arrays + a spatial grid index
  * ``verts.npy``       float32 (V, 2)  all ring vertices concatenated
  * ``ring_start.npy``  int32   (R,)    start offset of each ring in ``verts``
  * ``poly_start.npy``  int32   (P+1,)  start offset of each polygon's rings
  * ``bbox.npy``        float32 (P, 4)  per-polygon [min_e, min_n, max_e, max_n]
  * ``cells.npy``       int32   (K,)    flattened cell ids a polygon touches
  * ``cell_start.npy``  int32   (P+1,)  start offset of each polygon's cells
  * ``props.json``      list of {cofs, name, canton} per polygon
  * ``meta.json``       grid parameters + provenance

* ``elevation_10m/`` -- swissALTIRegio DEM as a memory-mappable numpy array
  * ``grid.npy``        float32 (H, W)  metres a.s.l., nodata -> NaN
  * ``meta.json``       origin, pixel size, CRS, provenance

Raw data sources (official swisstopo distribution on ``data.geo.admin.ch``)::

    COFS:      https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/swissboundaries3d_2026-01/swissboundaries3d_2026-01_2056_5728.gpkg.zip
    Elevation: https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/swissaltiregio/swissaltiregio_2056_5728.tif
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Minimal GeoPackage / WKB geometry reader (stdlib only: sqlite3 + struct).
#
# A .gpkg is a SQLite database; each feature's geometry is a BLOB with an 8-byte
# GPKG header (magic "GP", version, SRS id, envelope type), an optional MBR
# envelope, then a WKB geometry. This swisstopo file uses *legacy PostGIS/EWKB*
# Z-type codes (MultiPolygonZ = 1006, PolygonZ = 1003) rather than the standard
# bit-flagged 0x80000006 / 0x80000003, and a non-standard header size, so we
# locate the WKB by scanning for a byte-order byte followed by a recognised
# polygon type code and accept the first parse that consumes the whole blob.
# ---------------------------------------------------------------------------
_WKB_MP_TYPES = {6, 0x80000006, 1006}      # MultiPolygon / Z-flagged / legacy EWKB
_WKB_POLY_TYPES = {3, 0x80000003, 1003}    # Polygon / Z-flagged / legacy EWKB


def _wkb_parse_poly(b: bytes, o: int, bo: str, step: int):
    """Parse one WKB Polygon at offset ``o``; return (rings, new_offset)."""
    (nr,) = struct.unpack(bo + "I", b[o:o+4]); o += 4
    rings = []
    for _ in range(nr):
        (n,) = struct.unpack(bo + "I", b[o:o+4]); o += 4
        ring = []
        for _ in range(n):
            x, y = struct.unpack(bo + "dd", b[o:o+16])
            o += step
            ring.append((x, y))
        rings.append(ring)
    return rings, o


def parse_gpkg_geometry(blob: bytes):
    """Parse a GPKG geometry BLOB into a list of polygons.

    Each polygon is a list of rings; each ring is a list of ``(easting, northing)``
    tuples (Z values are dropped). Returns ``None`` if no valid geometry is found.
    Handles standard and legacy-EWKB type codes, little/big endian, 2D/3D.
    """
    n = len(blob)
    for off in range(8, min(n - 5, 64)):
        bo_byte = blob[off]
        if bo_byte not in (1, 2):
            continue
        bo = "<" if bo_byte == 1 else ">"
        t = struct.unpack(bo + "I", blob[off+1:off+5])[0]
        if t not in _WKB_MP_TYPES and t not in _WKB_POLY_TYPES:
            continue
        has_z = (t & 0x20000000) != 0 or t in (1003, 1006)
        step = 24 if has_z else 16
        o = off + 5
        polys = []
        try:
            if t in _WKB_MP_TYPES:
                (np_,) = struct.unpack(bo + "I", blob[o:o+4]); o += 4
                for _ in range(np_):
                    if blob[o] != bo_byte:
                        raise ValueError
                    pt = struct.unpack(bo + "I", blob[o+1:o+5])[0]
                    if (pt & 0x0FFFFFFF) != 3 and pt not in _WKB_POLY_TYPES:
                        raise ValueError
                    o += 5
                    r, o = _wkb_parse_poly(blob, o, bo, step)
                    polys.append(r)
            else:
                r, o = _wkb_parse_poly(blob, o, bo, step)
                polys.append(r)
        except Exception:
            continue
        if o == n:  # consumed the whole blob -> correct WKB start
            return polys
    return None


def convert_cofs(gpkg_path: str, out_dir: str) -> None:
    """Convert the swissBOUNDARIES3D GeoPackage to numpy polygon arrays.

    Uses only the Python standard library (``sqlite3`` + ``struct``); no geopandas.
    """
    import numpy as np

    t0 = time.time()
    print(f"[cofs] reading {gpkg_path} ...")
    con = sqlite3.connect(gpkg_path)
    cur = con.cursor()

    # Verify the layer's SRS is EPSG:2056 (LV95).
    srs_row = cur.execute(
        "SELECT srs_id FROM gpkg_geometry_columns WHERE table_name=?",
        ("tlm_hoheitsgebiet",),
    ).fetchone()
    if not srs_row or int(srs_row[0]) != 2056:
        con.close()
        raise SystemExit("COFS layer is not EPSG:2056; refusing to convert.")

    rows = cur.execute(
        "SELECT bfs_nummer, name, kantonsnummer, geom "
        "FROM tlm_hoheitsgebiet ORDER BY rowid"
    ).fetchall()
    con.close()

    n_muni = sum(1 for r in rows if r[0] is not None)
    print(f"[cofs] {n_muni} municipalities")

    verts_list: list[np.ndarray] = []
    ring_start: list[int] = []
    poly_start: list[int] = [0]
    bboxes: list[list[float]] = []
    props: list[dict] = []
    n_vert_total = 0

    for bfs_nummer, name, canton, geom_blob in rows:
        if bfs_nummer is None or geom_blob is None:
            continue
        polys = parse_gpkg_geometry(geom_blob)
        if not polys:
            continue
        pminx = pminy = math.inf
        pmaxx = pmaxy = -math.inf
        n_rings_this_poly = 0
        for rings in polys:
            for ring in rings:
                coords = np.asarray(ring, dtype=np.float32)
                # Drop the duplicated closing vertex (WKB rings are closed).
                if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
                    coords = coords[:-1]
                if len(coords) < 3:
                    continue
                ring_start.append(n_vert_total)
                verts_list.append(coords)
                n_vert_total += len(coords)
                pminx = min(pminx, float(coords[:, 0].min()))
                pminy = min(pminy, float(coords[:, 1].min()))
                pmaxx = max(pmaxx, float(coords[:, 0].max()))
                pmaxy = max(pmaxy, float(coords[:, 1].max()))
                n_rings_this_poly += 1
        if n_rings_this_poly == 0:
            continue
        bboxes.append([pminx, pminy, pmaxx, pmaxy])
        code = str(int(bfs_nummer))
        props.append({
            "cofs": code.zfill(4),
            "name": None if name is None else str(name),
            "canton": (None if canton is None or str(canton) == "nan"
                       else str(int(canton))),
        })
        poly_start.append(len(ring_start))

    verts = np.vstack(verts_list).astype(np.float32)
    ring_start_arr = np.asarray(ring_start, dtype=np.int32)
    poly_start_arr = np.asarray(poly_start, dtype=np.int32)
    bbox = np.asarray(bboxes, dtype=np.float32)
    n_polys = len(props)

    # ------------------------------------------------------------- spatial idx
    cell = 5000.0  # metres
    e0 = math.floor(float(bbox[:, 0].min()) / cell) * cell
    n0 = math.floor(float(bbox[:, 1].min()) / cell) * cell
    cells_per_poly: list[np.ndarray] = []
    for (minx, miny, maxx, maxy) in bbox:
        c0 = int((minx - e0) // cell)
        r0 = int((miny - n0) // cell)
        c1 = int((maxx - e0) // cell)
        r1 = int((maxy - n0) // cell)
        ids = [r * 100000 + c for r in range(r0, r1 + 1)
               for c in range(c0, c1 + 1)]
        cells_per_poly.append(np.asarray(ids, dtype=np.int32))

    cell_start = np.zeros(n_polys + 1, dtype=np.int32)
    for i, arr in enumerate(cells_per_poly):
        cell_start[i + 1] = cell_start[i] + len(arr)
    cells = (np.concatenate(cells_per_poly).astype(np.int32)
             if cells_per_poly else np.zeros(0, dtype=np.int32))

    # ------------------------------------------------------------------ write
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "verts.npy"), verts)
    np.save(os.path.join(out_dir, "ring_start.npy"), ring_start_arr)
    np.save(os.path.join(out_dir, "poly_start.npy"), poly_start_arr)
    np.save(os.path.join(out_dir, "bbox.npy"), bbox)
    np.save(os.path.join(out_dir, "cells.npy"), cells)
    np.save(os.path.join(out_dir, "cell_start.npy"), cell_start)
    with open(os.path.join(out_dir, "props.json"), "w", encoding="utf-8") as f:
        json.dump(props, f, ensure_ascii=False)
    meta = {
        "crs": 2056,
        "n_polys": n_polys,
        "n_rings": len(ring_start_arr),
        "n_verts": int(len(verts)),
        "cell_size_m": cell,
        "grid_origin_e": e0,
        "grid_origin_n": n0,
        "source": os.path.basename(gpkg_path),
        "layer": "tlm_hoheitsgebiet",
        "code_field": "bfs_nummer",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    counts = np.diff(cell_start)
    print(f"[cofs] verts={len(verts):,} rings={len(ring_start_arr):,} "
          f"polys={n_polys}")
    if len(counts):
        print(f"[cofs] cell assignments: total={len(cells):,} "
              f"max/poly={int(counts.max())} mean/poly={counts.mean():.1f}")
    print(f"[cofs] wrote {out_dir} in {time.time() - t0:.1f}s")


def convert_elevation(tif_path: str, out_dir: str, chunk_rows: int = 2048) -> None:
    """Convert the swissALTIRegio GeoTIFF to a float32 .npy (nodata -> NaN)."""
    import numpy as np
    import rasterio

    t0 = time.time()
    print(f"[elev] reading {tif_path} ...")
    with rasterio.open(tif_path) as src:
        epsg = None
        try:
            epsg = src.crs.to_epsg()
        except Exception:
            pass
        if epsg != 2056:
            raise SystemExit(f"Elevation raster CRS is {src.crs}; expected EPSG:2056.")
        width, height = src.width, src.height
        transform = src.transform
        nodata = src.nodata
        print(f"[elev] {width}x{height} px, res=({transform.a:.3f}, "
              f"{transform.e:.3f}) m, origin=({transform.c:.1f}, "
              f"{transform.f:.1f}), nodata={nodata}")

        grid = np.empty((height, width), dtype=np.float32)
        for row0 in range(0, height, chunk_rows):
            row1 = min(row0 + chunk_rows, height)
            win = rasterio.windows.Window(0, row0, width, row1 - row0)
            block = src.read(1, window=win).astype(np.float32)
            if nodata is not None:
                block = np.where(block == nodata, np.nan, block)
            grid[row0:row1] = block
            print(f"[elev]   rows {row0}-{row1} ({time.time() - t0:.0f}s)")

    os.makedirs(out_dir, exist_ok=True)
    out_npy = os.path.join(out_dir, "grid.npy")
    print(f"[elev] writing {out_npy} ...")
    np.save(out_npy, grid)
    meta = {
        "crs": 2056,
        "width": width,
        "height": height,
        "origin_e": float(transform.c),   # top-left corner easting
        "origin_n": float(transform.f),   # top-left corner northing
        "pixel_size_m": abs(float(transform.a)),
        "unit": "m",
        "datum": "swissALTIRegio (swisstopo)",
        "source": os.path.basename(tif_path),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    finite = np.isfinite(grid)
    print(f"[elev] valid pixels {finite.sum():,}/{grid.size:,} "
          f"min={np.nanmin(grid):.1f} max={np.nanmax(grid):.1f} m")
    print(f"[elev] wrote {out_dir} in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Optional raw-data download (stdlib urllib only).
# Official swisstopo distribution on data.geo.admin.ch.
# ---------------------------------------------------------------------------
SOURCES = {
    "cofs": (
        "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/"
        "swissboundaries3d_2026-01/swissboundaries3d_2026-01_2056_5728.gpkg.zip"
    ),
    "elevation": (
        "https://data.geo.admin.ch/ch.swisstopo.swissaltiregio/"
        "swissaltiregio/swissaltiregio_2056_5728.tif"
    ),
}


def _download(url: str, dest: str) -> None:
    """Download ``url`` to ``dest`` (via a .part temp file) with progress."""
    import urllib.request

    t0 = time.time()
    state = {"last": 0.0}

    def hook(block_num, block_size, total):
        got = block_num * block_size
        now = time.time()
        if now - state["last"] > 1.0 or (total and got >= total):
            state["last"] = now
            if total:
                print(f"    {got/1e6:9.1f} / {total/1e6:.1f} MB ({now - t0:.0f}s)")
            else:
                print(f"    {got/1e6:9.1f} MB ({now - t0:.0f}s)")

    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    os.replace(tmp, dest)


def ensure_raw_files(data_dir: str, want_cofs: bool, want_elevation: bool) -> None:
    """Download any missing raw input files.

    Never overwrites existing files. The COFS source is a .zip containing the
    GeoPackage; it is extracted into ``data_dir`` and the zip removed.
    """
    import zipfile

    os.makedirs(data_dir, exist_ok=True)
    if want_cofs:
        url = SOURCES["cofs"]
        base = os.path.basename(url)          # ...gpkg.zip
        # The zip member name differs from the URL basename, so check for any
        # boundaries GeoPackage already present (same rule as auto-detect).
        have = [f for f in os.listdir(data_dir)
                if f.lower().endswith(".gpkg")
                and ("swissboundaries3d" in f.lower() or "cofs" in f.lower())]
        if not have:
            zip_path = os.path.join(data_dir, base)
            if os.path.exists(zip_path):
                print(f"[dl] extracting existing {base}")
            else:
                print(f"[dl] downloading {url}")
                _download(url, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                for member in z.namelist():
                    if member.lower().endswith(".gpkg"):
                        target = os.path.join(data_dir, os.path.basename(member))
                        print(f"[dl] extracting {member} -> {target}")
                        with z.open(member) as src, open(target, "wb") as out:
                            out.write(src.read())
            os.remove(zip_path)
    if want_elevation:
        url = SOURCES["elevation"]
        dest = os.path.join(data_dir, os.path.basename(url))
        if not os.path.exists(dest):
            print(f"[dl] downloading {url} (~10 GB, this takes a while)")
            _download(url, dest)


def _find_raw(data_dir: str, patterns: list[str]):
    """Return the first file in ``data_dir`` whose name contains any pattern."""
    for p in patterns:
        cands = [f for f in os.listdir(data_dir) if p.lower() in f.lower()]
        if cands:
            return os.path.join(data_dir, sorted(cands)[0])
    return None


# ---------------------------------------------------------------------------
# Elevation conversion without touching the user's environment.
# ---------------------------------------------------------------------------

def _rasterio_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("rasterio") is not None


def _run_elevation_ephemeral(tif_path: str, out_dir: str) -> None:
    """Run :func:`convert_elevation` in a throwaway venv that has rasterio.

    The current interpreter lacks rasterio. A temporary venv is created on top
    of it (``--system-site-packages``, so numpy and swissgeo are inherited),
    rasterio is installed **only** into the temp venv, the conversion runs there
    in a subprocess, and the temp venv is deleted afterwards. The user's
    environment is never modified.
    """
    import tempfile

    base = sys.executable
    tmp = tempfile.mkdtemp(prefix="swissgeo-rasterio-")
    venv_dir = os.path.join(tmp, "venv")
    print(f"[elev] rasterio not found in {base}")
    print("[elev] creating a throwaway venv (your environment will NOT be modified) ...")
    try:
        subprocess.run(
            [base, "-m", "venv", "--system-site-packages", venv_dir], check=True
        )

        if os.name == 'nt':
            venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_py = os.path.join(venv_dir, "bin", "python")

        uv = shutil.which("uv")
        if uv:
            cmd = [uv, "pip", "install", "--python", venv_py, "rasterio"]
            tool = "uv"
        else:
            cmd = [venv_py, "-m", "pip", "install", "rasterio"]
            tool = "pip"
        print(f"[elev] installing rasterio into the throwaway venv ({tool}) ...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            raise SystemExit(
                "Could not install rasterio into the throwaway venv.\n"
                "Either install it into your environment yourself "
                "(pip install rasterio) or make sure 'uv' or 'pip' is available."
            )

        # Run the conversion inside the temp venv. PYTHONPATH makes `swissgeo`
        # importable even when running from a source checkout (not installed).
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        code = ("import sys; from swissgeo.setup_data import convert_elevation;"
                "convert_elevation(sys.argv[1], sys.argv[2])")
        subprocess.run([venv_py, "-c", code, tif_path, out_dir], check=True, env=env)
    except subprocess.CalledProcessError:
        raise SystemExit(
            "Elevation conversion in the throwaway venv failed.\n"
            "Alternative: install rasterio into your environment "
            "(pip install rasterio) and re-run."
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[elev] throwaway venv removed; your environment was not modified.")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_setup(data_dir: str = "data", want_cofs: bool = True,
              want_elevation: bool = True, download: bool = True,
              cofs_path: str | None = None,
              elevation_path: str | None = None) -> None:
    """Download (optionally) and convert the raw swisstopo datasets.

    Parameters
    ----------
    data_dir:        directory holding the raw files and receiving the
                     converted ``cofs_lv95/`` / ``elevation_10m/`` outputs.
    want_cofs:       convert the municipality boundaries (stdlib only).
    want_elevation:  convert the elevation DEM (rasterio; a throwaway venv is
                     used automatically if rasterio is not installed).
    download:        fetch any missing raw files from data.geo.admin.ch first.
    cofs_path / elevation_path: explicit raw-file overrides (auto-detect otherwise).
    """
    os.makedirs(data_dir, exist_ok=True)
    if download:
        ensure_raw_files(data_dir, want_cofs=want_cofs,
                         want_elevation=want_elevation)

    # COFS: stdlib only; numpy is a runtime dependency so always available.
    if want_cofs:
        gpkg = cofs_path or _find_raw(data_dir, ["swissboundaries3d", "cofs"])
        if gpkg:
            convert_cofs(gpkg, os.path.join(data_dir, "cofs_lv95"))
        else:
            print(f"[cofs] no GeoPackage found in {data_dir}; skipping "
                  "(re-run with download enabled or pass the file path)")

    # Elevation: needs rasterio; use a throwaway venv if it is missing.
    if want_elevation:
        tif = elevation_path or _find_raw(data_dir, ["swissaltiregio", "alti", "dgm"])
        if not tif:
            print(f"[elev] no GeoTIFF found in {data_dir}; skipping "
                  "(re-run with download enabled or pass the file path)")
        elif _rasterio_available():
            convert_elevation(tif, os.path.join(data_dir, "elevation_10m"))
        else:
            _run_elevation_ephemeral(tif, os.path.join(data_dir, "elevation_10m"))
