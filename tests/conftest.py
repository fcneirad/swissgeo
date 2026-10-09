"""Shared pytest fixtures for swissgeo tests.

Builds small *synthetic* datasets directly in the swissgeo **runtime formats**
(numpy arrays + JSON), so the tests need no geopandas/rasterio/shapely:

* a COFS dataset with five municipality polygons (four boxes + one polygon with a
  hole) whose codes and names are known;
* an elevation grid whose value is a deterministic linear function of the LV95
  coordinates, so bilinear interpolation can be verified exactly.

Real-data tests (against ``data/cofs_lv95`` and ``data/elevation_10m``) are added
in the individual test modules and skipped when the data is not present.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic COFS dataset
# ---------------------------------------------------------------------------
# (cofs, name, canton, rings) where each ring is a list of (e, n) vertices.
# Boxes are open chains (no repeated closing vertex), matching convert_data.py.
def _box(e0: float, n0: float, e1: float, n1: float):
    return [(e0, n0), (e1, n0), (e1, n1), (e0, n1)]


SYNTH_MUNICIPALITIES = [
    ("0351", "Bern", "2", [_box(2_600_000, 1_199_000, 2_602_000, 1_201_000)]),
    ("0261", "Zürich", "1", [_box(2_680_000, 1_245_000, 2_684_000, 1_249_000)]),
    ("6621", "Genève", "25", [_box(2_500_000, 1_115_000, 2_503_000, 1_118_000)]),
    ("0701", "Basel", "28", [_box(2_490_000, 1_260_000, 2_493_000, 1_263_000)]),
    # Polygon with a hole: outer box minus inner box.
    ("9999", "Testhole", "0", [
        _box(2_550_000, 1_200_000, 2_554_000, 1_204_000),
        _box(2_551_000, 1_201_000, 2_553_000, 1_203_000),
    ]),
]


def build_cofs_dataset(out_dir: str) -> None:
    """Write a synthetic COFS dataset in the swissgeo runtime format."""
    verts_list = []
    ring_start = []
    poly_start = [0]
    bboxes = []
    props = []
    n_vert = 0

    for cofs, name, canton, rings in SYNTH_MUNICIPALITIES:
        pminx = pminy = math.inf
        pmaxx = pmaxy = -math.inf
        for ring in rings:
            coords = np.asarray(ring, dtype=np.float32)
            ring_start.append(n_vert)
            verts_list.append(coords)
            n_vert += len(coords)
            pminx = min(pminx, float(coords[:, 0].min()))
            pminy = min(pminy, float(coords[:, 1].min()))
            pmaxx = max(pmaxx, float(coords[:, 0].max()))
            pmaxy = max(pmaxy, float(coords[:, 1].max()))
        bboxes.append([pminx, pminy, pmaxx, pmaxy])
        props.append({"cofs": cofs, "name": name, "canton": canton})
        poly_start.append(len(ring_start))

    verts = np.vstack(verts_list).astype(np.float32)
    bbox = np.asarray(bboxes, dtype=np.float32)
    n_polys = len(props)

    # Spatial grid index (same scheme as scripts/convert_data.py).
    cell = 5000.0
    e0 = math.floor(float(bbox[:, 0].min()) / cell) * cell
    n0 = math.floor(float(bbox[:, 1].min()) / cell) * cell
    cells_per_poly = []
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

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "verts.npy"), verts)
    np.save(os.path.join(out_dir, "ring_start.npy"),
            np.asarray(ring_start, dtype=np.int32))
    np.save(os.path.join(out_dir, "poly_start.npy"),
            np.asarray(poly_start, dtype=np.int32))
    np.save(os.path.join(out_dir, "bbox.npy"), bbox)
    np.save(os.path.join(out_dir, "cells.npy"), cells)
    np.save(os.path.join(out_dir, "cell_start.npy"), cell_start)
    with open(os.path.join(out_dir, "props.json"), "w", encoding="utf-8") as f:
        json.dump(props, f, ensure_ascii=False)
    meta = {
        "crs": 2056,
        "n_polys": n_polys,
        "n_rings": len(ring_start),
        "n_verts": int(len(verts)),
        "cell_size_m": cell,
        "grid_origin_e": e0,
        "grid_origin_n": n0,
        "source": "synthetic",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


@pytest.fixture(scope="session")
def cofs_dir(tmp_path_factory):
    """Path to a synthetic COFS dataset in the runtime format."""
    d = str(tmp_path_factory.mktemp("cofs"))
    build_cofs_dataset(d)
    return d


# ---------------------------------------------------------------------------
# Synthetic elevation grid
# ---------------------------------------------------------------------------
# Planar surface: z(e, n) = 100 + e * 1e-4 + n * 2e-4  (metres).
ELEV_ORIGIN_E = 2_600_000.0
ELEV_ORIGIN_N = 1_210_000.0   # top-left corner northing
ELEV_PX = 50.0
ELEV_W = ELEV_H = 400        # 20 km x 20 km window


def elev_surface(e: float, n: float) -> float:
    """The exact planar surface used to fill the synthetic grid."""
    return 100.0 + e * 1e-4 + n * 2e-4


def build_elev_dataset(out_dir: str) -> None:
    """Write a synthetic elevation grid in the swissgeo runtime format."""
    grid = np.empty((ELEV_H, ELEV_W), dtype=np.float32)
    for j in range(ELEV_H):
        n = ELEV_ORIGIN_N - (j + 0.5) * ELEV_PX   # pixel-center northing
        for i in range(ELEV_W):
            e = ELEV_ORIGIN_E + (i + 0.5) * ELEV_PX  # pixel-center easting
            grid[j, i] = elev_surface(e, n)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "grid.npy"), grid)
    meta = {
        "crs": 2056,
        "width": ELEV_W,
        "height": ELEV_H,
        "origin_e": ELEV_ORIGIN_E,
        "origin_n": ELEV_ORIGIN_N,
        "pixel_size_m": ELEV_PX,
        "unit": "m",
        "source": "synthetic",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


@pytest.fixture(scope="session")
def elev_dir(tmp_path_factory):
    """Path to a synthetic elevation dataset in the runtime format."""
    d = str(tmp_path_factory.mktemp("elev"))
    build_elev_dataset(d)
    return d
