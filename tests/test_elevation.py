"""Tests for elevation lookup.

Synthetic tests use a planar surface ``z(e, n) = 100 + e*1e-4 + n*2e-4`` sampled
on a 50 m grid; because the surface is linear, bilinear interpolation reproduces
it exactly (up to float32 rounding). Real-data tests run against
``data/elevation_10m`` when present.
"""

from __future__ import annotations

import os

import pytest

from swissgeo import ElevationLookup, Elevation
from conftest import elev_surface, ELEV_ORIGIN_E, ELEV_ORIGIN_N, ELEV_PX

REAL_ELEV_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "elevation_10m")
has_real_data = os.path.isdir(REAL_ELEV_DIR) and os.path.exists(
    os.path.join(REAL_ELEV_DIR, "grid.npy"))


# ---------------------------------------------------------------------------
# Synthetic dataset (planar surface -> exact bilinear)
# ---------------------------------------------------------------------------
def test_interior_point_exact(elev_dir):
    lk = ElevationLookup(data_dir=elev_dir)
    # A point strictly inside the grid, not on a pixel center.
    e = ELEV_ORIGIN_E + 1234.5
    n = ELEV_ORIGIN_N - 987.25
    ev = lk.from_lv95(e, n)
    assert ev is not None
    expected = elev_surface(e, n)
    # float32 grid -> allow a few centimetres of rounding.
    assert abs(ev.value - expected) < 0.05


def test_pixel_center_exact(elev_dir):
    lk = ElevationLookup(data_dir=elev_dir)
    # Exactly on a pixel center: value should equal the stored cell exactly.
    i, j = 100, 100
    e = ELEV_ORIGIN_E + (i + 0.5) * ELEV_PX
    n = ELEV_ORIGIN_N - (j + 0.5) * ELEV_PX
    ev = lk.from_lv95(e, n)
    assert ev is not None
    assert abs(ev.value - elev_surface(e, n)) < 1e-3


def test_resolution_and_source(elev_dir):
    lk = ElevationLookup(data_dir=elev_dir)
    ev = lk.from_lv95(ELEV_ORIGIN_E + 1000.0, ELEV_ORIGIN_N - 1000.0)
    assert ev is not None
    assert ev.resolution_m == pytest.approx(ELEV_PX)
    assert ev.source == "synthetic"


def test_outside_grid_returns_none(elev_dir):
    lk = ElevationLookup(data_dir=elev_dir)
    # Far outside the 20 km window.
    assert lk.from_lv95(ELEV_ORIGIN_E + 100_000.0, ELEV_ORIGIN_N - 100_000.0) is None


def test_from_wgs84(elev_dir):
    from swissgeo import to_wgs84

    lk = ElevationLookup(data_dir=elev_dir)
    e, n = ELEV_ORIGIN_E + 1234.5, ELEV_ORIGIN_N - 987.25
    lon, lat = to_wgs84(e, n)
    ev = lk.from_wgs84(lon, lat)
    assert ev is not None
    assert abs(ev.value - elev_surface(e, n)) < 0.05


def test_elevation_str(elev_dir):
    lk = ElevationLookup(data_dir=elev_dir)
    ev = lk.from_lv95(ELEV_ORIGIN_E + 1000.0, ELEV_ORIGIN_N - 1000.0)
    s = str(ev)
    assert "m a.s.l." in s


def test_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ElevationLookup(data_dir=str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# Real data (swissALTIRegio) — skipped if not converted
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_elev():
    if not has_real_data:
        pytest.skip("real elevation data not present")
    return ElevationLookup(data_dir=REAL_ELEV_DIR)


class TestRealElevation:
    def test_bern_elevation(self, real_elev):
        from swissgeo import to_lv95

        e, n = to_lv95(7.4475, 46.9481)   # Bern city centre (~540 m a.s.l.)
        ev = real_elev.from_lv95(e, n)
        assert ev is not None
        assert 520 < ev.value < 560

    def test_zurich_elevation(self, real_elev):
        from swissgeo import to_lv95

        e, n = to_lv95(8.5417, 47.3769)   # Zürich (~400 m a.s.l.)
        ev = real_elev.from_lv95(e, n)
        assert ev is not None
        assert 380 < ev.value < 430

    def test_geneva_elevation(self, real_elev):
        from swissgeo import to_lv95

        e, n = to_lv95(6.1432, 46.2044)   # Genève (~375 m a.s.l.)
        ev = real_elev.from_lv95(e, n)
        assert ev is not None
        assert 350 < ev.value < 400

    def test_resolution_is_10m(self, real_elev):
        from swissgeo import to_lv95

        e, n = to_lv95(7.4475, 46.9481)
        ev = real_elev.from_lv95(e, n)
        assert ev.resolution_m == pytest.approx(10.0)
