"""Tests for COFS municipality lookup.

Synthetic tests use a small dataset built in the runtime format (see conftest)
with known boxes and a polygon-with-hole, so point-in-polygon behaviour is
deterministic. Real-data tests run against ``data/cofs_lv95`` when present.
"""

from __future__ import annotations

import os

import pytest

from swissgeo import CofsLookup, Municipality, to_wgs84

REAL_COFS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cofs_lv95")
has_real_data = os.path.isdir(REAL_COFS_DIR) and os.path.exists(
    os.path.join(REAL_COFS_DIR, "verts.npy"))


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
def test_bern_box(cofs_dir):
    lk = CofsLookup(data_dir=cofs_dir)
    m = lk.from_lv95(2_601_000, 1_200_000)   # centre of the Bern box
    assert m is not None
    assert m.cofs == "0351"
    assert m.name == "Bern"
    assert m.canton == "2"


def test_zurich_box(cofs_dir):
    lk = CofsLookup(data_dir=cofs_dir)
    m = lk.from_lv95(2_682_000, 1_247_000)
    assert m is not None and m.cofs == "0261" and m.name == "Zürich"


def test_geneva_box(cofs_dir):
    lk = CofsLookup(data_dir=cofs_dir)
    m = lk.from_lv95(2_501_500, 1_116_500)
    assert m is not None and m.cofs == "6621"


def test_outside_all_polygons(cofs_dir):
    lk = CofsLookup(data_dir=cofs_dir)
    # A point inside the grid but in no municipality.
    assert lk.from_lv95(2_570_000, 1_230_000) is None


def test_outside_swiss_extent(cofs_dir):
    lk = CofsLookup(data_dir=cofs_dir)
    assert lk.from_lv95(1_000_000, 1_200_000) is None


def test_polygon_with_hole(cofs_dir):
    lk = CofsLookup(data_dir=cofs_dir)
    # Inside the outer box but inside the hole -> NOT in the municipality.
    assert lk.from_lv95(2_552_000, 1_202_000) is None
    # Inside the outer box but outside the hole -> in the municipality.
    m = lk.from_lv95(2_550_500, 1_203_500)
    assert m is not None and m.cofs == "9999"


def test_from_wgs84(cofs_dir):
    lk = CofsLookup(data_dir=cofs_dir)
    # WGS84 point that maps into the Bern box.
    e, n = 2_601_000.0, 1_200_000.0
    lon, lat = to_wgs84(e, n)
    m = lk.from_wgs84(lon, lat)
    assert m is not None and m.cofs == "0351"


def test_municipality_str(cofs_dir):
    lk = CofsLookup(data_dir=cofs_dir)
    m = lk.from_lv95(2_601_000, 1_200_000)
    s = str(m)
    assert "0351" in s and "Bern" in s


def test_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CofsLookup(data_dir=str(tmp_path / "does_not_exist"))


# ---------------------------------------------------------------------------
# Batch lookup (from_lv95_batch)
# ---------------------------------------------------------------------------
def _batch_matches_scalar(lk, eastings, northings):
    """Assert batch results equal per-point scalar results."""
    batch = lk.from_lv95_batch(eastings, northings)
    assert len(batch) == len(eastings)
    for i, (e, n) in enumerate(zip(eastings, northings)):
        scalar = lk.from_lv95(float(e), float(n))
        b = batch[i]
        if scalar is None:
            assert b is None, f"point {i} ({e}, {n}): batch={b!r}, scalar=None"
        else:
            assert b is not None, f"point {i} ({e}, {n}): batch=None, scalar={scalar!r}"
            assert b.cofs == scalar.cofs, (
                f"point {i} ({e}, {n}): batch cofs={b.cofs}, scalar cofs={scalar.cofs}")


def test_batch_matches_scalar_synthetic(cofs_dir):
    import numpy as np

    lk = CofsLookup(data_dir=cofs_dir)
    rng = np.random.default_rng(42)
    # Dense grid over the synthetic extent + a few out-of-bounds points.
    es = np.linspace(2_489_000, 2_685_000, 120)
    ns = np.linspace(1_114_000, 1_264_000, 120)
    eastings = rng.uniform(2_489_000, 2_685_000, size=3000)
    northings = rng.uniform(1_114_000, 1_264_000, size=3000)
    eastings = np.concatenate([eastings, [1_000_000.0, 2_600_000.0], es[:5]])
    northings = np.concatenate([northings, [1_200_000.0, 1_500_000.0], ns[:5]])
    _batch_matches_scalar(lk, eastings, northings)


def test_batch_empty_and_mismatch(cofs_dir):
    lk = CofsLookup(data_dir=cofs_dir)
    assert lk.from_lv95_batch([], []) == []
    with pytest.raises(ValueError):
        lk.from_lv95_batch([2_601_000.0], [1_200_000.0, 1_200_500.0])


def test_batch_known_points_synthetic(cofs_dir):
    lk = CofsLookup(data_dir=cofs_dir)
    res = lk.from_lv95_batch(
        [2_601_000, 2_682_000, 2_570_000, 2_552_000, 2_550_500],
        [1_200_000, 1_247_000, 1_230_000, 1_202_000, 1_203_500],
    )
    assert res[0].cofs == "0351"          # Bern box
    assert res[1].cofs == "0261"          # Zürich box
    assert res[2] is None                 # in grid, no municipality
    assert res[3] is None                 # inside the hole
    assert res[4].cofs == "9999"          # outside the hole


class TestRealCofsBatch:
    def test_batch_matches_scalar_real(self, real_cofs):
        import numpy as np

        rng = np.random.default_rng(7)
        eastings = rng.uniform(2_350_000, 2_850_000, size=400)
        northings = rng.uniform(1_050_000, 1_350_000, size=400)
        # Add known city points and out-of-bounds points.
        from swissgeo import to_lv95

        for lon, lat in [(7.4475, 46.9481), (8.5417, 47.3769), (6.1432, 46.2044)]:
            e, n = to_lv95(lon, lat)
            eastings = np.append(eastings, e)
            northings = np.append(northings, n)
        eastings = np.append(eastings, [2_600_000.0, 1_000_000.0])
        northings = np.append(northings, [1_500_000.0, 1_200_000.0])
        _batch_matches_scalar(real_cofs, eastings, northings)

    def test_batch_speed_real(self, real_cofs):
        import numpy as np
        import time

        rng = np.random.default_rng(1)
        n_pts = 20_000
        eastings = rng.uniform(2_350_000, 2_850_000, size=n_pts)
        northings = rng.uniform(1_050_000, 1_350_000, size=n_pts)
        t0 = time.perf_counter()
        res = real_cofs.from_lv95_batch(eastings, northings)
        dt = time.perf_counter() - t0
        found = sum(1 for m in res if m is not None)
        print(f"\nbatch: {n_pts} points in {dt:.2f}s "
              f"({n_pts / dt:,.0f} pts/s), {found} matched")
        # Batch must be dramatically faster than the scalar loop.
        assert dt < 60.0, f"batch lookup too slow: {dt:.1f}s for {n_pts} points"


# ---------------------------------------------------------------------------
# Real data (swissBOUNDARIES3D) — skipped if not converted
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_cofs():
    if not has_real_data:
        pytest.skip("real COFS data not present")
    return CofsLookup(data_dir=REAL_COFS_DIR)


class TestRealCofs:
    def test_bern(self, real_cofs):
        from swissgeo import to_lv95

        e, n = to_lv95(7.4475, 46.9481)   # Bern city centre
        m = real_cofs.from_lv95(e, n)
        assert m is not None
        assert m.cofs == "0351"
        assert m.name == "Bern"

    def test_zurich(self, real_cofs):
        from swissgeo import to_lv95

        e, n = to_lv95(8.5417, 47.3769)
        m = real_cofs.from_lv95(e, n)
        assert m is not None and m.cofs == "0261"

    def test_geneva(self, real_cofs):
        from swissgeo import to_lv95

        e, n = to_lv95(6.1432, 46.2044)
        m = real_cofs.from_lv95(e, n)
        assert m is not None and m.cofs == "6621"

    def test_luzern(self, real_cofs):
        from swissgeo import to_lv95

        e, n = to_lv95(8.3102, 47.0502)
        m = real_cofs.from_lv95(e, n)
        assert m is not None and m.cofs == "1061"

    def test_outside_switzerland(self, real_cofs):
        assert real_cofs.from_lv95(2_600_000, 1_500_000) is None
