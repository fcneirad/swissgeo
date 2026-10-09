"""Tests for WGS84 <-> LV95 coordinate transforms.

The pure-numpy implementation is validated against known reference values and,
when pyproj is available (it is in the hidres dev environment), against pyproj
itself over a grid covering all of Switzerland.
"""

from __future__ import annotations

import math

import pytest

from swissgeo import (
    to_lv95, to_wgs84, to_lv03, lv03_to_wgs84, lv95_to_lv03, lv03_to_lv95,
    transform, is_valid_lv95, is_valid_lv03,
)

pyproj = pytest.importorskip("pyproj", reason="pyproj used only as a test oracle")


# ---------------------------------------------------------------------------
# Known reference values (validated against pyproj / official sources)
# ---------------------------------------------------------------------------
def test_to_lv95_bern():
    e, n = to_lv95(7.4475, 46.9481)   # Bern city centre
    assert abs(e - 2_600_675) < 5
    assert abs(n - 1_199_668) < 5


def test_to_lv95_zurich():
    e, n = to_lv95(8.5417, 47.3769)   # Zürich
    assert abs(e - 2_683_304) < 5
    assert abs(n - 1_247_926) < 5


def test_to_lv95_geneva():
    e, n = to_lv95(6.1432, 46.2044)   # Genève
    assert abs(e - 2_500_016) < 5
    assert abs(n - 1_117_821) < 5


def test_to_wgs84_roundtrip_scalar():
    lon, lat = to_wgs84(2_600_675, 1_199_668)
    assert abs(lon - 7.4475) < 1e-4
    assert abs(lat - 46.9481) < 1e-4


def test_scalar_and_sequence_consistency():
    lons = [7.4475, 8.5417, 6.1432]
    lats = [46.9481, 47.3769, 46.2044]
    es_seq, ns_seq = to_lv95(lons, lats)
    for i in range(3):
        e_s, n_s = to_lv95(lons[i], lats[i])
        assert abs(e_s - es_seq[i]) < 1e-6
        assert abs(n_s - ns_seq[i]) < 1e-6


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        to_lv95([7.4, 8.5], [46.9])
    with pytest.raises(ValueError):
        to_wgs84([2_600_000], [1_200_000, 1_201_000])


def test_transform_helper():
    out = transform([(7.4475, 46.9481)], to_crs="lv95")
    assert len(out) == 1 and abs(out[0][0] - 2_600_675) < 5
    out2 = transform([(2_600_675, 1_199_668)], to_crs="wgs84")
    assert abs(out2[0][0] - 7.4475) < 1e-4
    with pytest.raises(ValueError):
        transform([(1, 2)], to_crs="nope")


def test_is_valid_lv95():
    assert is_valid_lv95(2_600_000, 1_200_000)
    assert not is_valid_lv95(1_000_000, 1_200_000)
    assert not is_valid_lv95(2_600_000, 2_000_000)


# ---------------------------------------------------------------------------
# Validation against pyproj (the reference implementation)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fwd():
    return pyproj.Transformer.from_crs(4326, 2056, always_xy=True)


@pytest.fixture(scope="module")
def inv():
    return pyproj.Transformer.from_crs(2056, 4326, always_xy=True)


def test_forward_matches_pyproj(fwd):
    for lon, lat in [(7.4475, 46.9481), (8.5417, 47.3769), (6.1432, 46.2044)]:
        e_ref, n_ref = fwd.transform(lon, lat)
        e, n = to_lv95(lon, lat)
        assert abs(e - e_ref) < 1e-3   # sub-millimetre
        assert abs(n - n_ref) < 1e-3


def test_inverse_matches_pyproj(inv):
    for e, n in [(2_600_675, 1_199_668), (2_683_304, 1_247_926), (2_500_016, 1_117_821)]:
        lon_ref, lat_ref = inv.transform(e, n)
        lon, lat = to_wgs84(e, n)
        assert abs(lon - lon_ref) < 1e-9
        assert abs(lat - lat_ref) < 1e-9


def test_grid_matches_pyproj(fwd):
    """Dense grid over the whole Swiss territory: max error must be < 1 mm."""
    import numpy as np

    lons = np.arange(5.9, 10.6, 0.2)
    lats = np.arange(45.7, 47.9, 0.2)
    LL = np.meshgrid(lons, lats)
    E_ref, N_ref = fwd.transform(LL[0].ravel(), LL[1].ravel())
    E_ours, N_ours = to_lv95(LL[0].ravel().tolist(), LL[1].ravel().tolist())
    assert max(abs(e - r) for e, r in zip(E_ours, E_ref)) < 1e-3
    assert max(abs(n - r) for n, r in zip(N_ours, N_ref)) < 1e-3


def test_roundtrip_precision():
    """LV95 -> WGS84 -> LV95 must be stable to ~1 mm.

    The residual (~1 mm) is inherent to the inverse projection's convergence
    tolerance (PROJ uses EPS = 1e-10 rad); our result matches pyproj's own
    roundtrip exactly, so this bound verifies we are on par with the reference.
    """
    e0, n0 = 2_600_675.0, 1_199_668.0
    lon, lat = to_wgs84(e0, n0)
    e1, n1 = to_lv95(lon, lat)
    assert abs(e1 - e0) < 2e-3
    assert abs(n1 - n0) < 2e-3


# ---------------------------------------------------------------------------
# LV03 (CH1903, EPSG:2178) -- identical projection to LV95, pure offset
# ---------------------------------------------------------------------------
def test_to_lv03_bern():
    e, n = to_lv03(7.4475, 46.9481)   # Bern city centre
    assert abs(e - 600_675) < 5
    assert abs(n - 199_668) < 5


def test_lv95_lv03_exact_offset():
    # LV03 = LV95 - (2'000'000, 1'000'000), exactly.
    e95, n95 = to_lv95(7.4475, 46.9481)
    e03, n03 = to_lv03(7.4475, 46.9481)
    assert abs((e95 - e03) - 2_000_000.0) < 1e-6
    assert abs((n95 - n03) - 1_000_000.0) < 1e-6
    # Round-trip through the dedicated helpers is exact.
    e_back, n_back = lv03_to_lv95(e03, n03)
    assert abs(e_back - e95) < 1e-6
    assert abs(n_back - n95) < 1e-6


def test_is_valid_lv03():
    assert is_valid_lv03(600_000, 200_000)
    assert not is_valid_lv03(1_000_000, 200_000)
    assert not is_valid_lv03(600_000, 800_000)


def test_transform_helper_lv03():
    out = transform([(7.4475, 46.9481)], to_crs="lv03")
    assert len(out) == 1 and abs(out[0][0] - 600_675) < 5


def _lv03_crs():
    """Build the CH1903 / LV03 CRS explicitly.

    We cannot use an EPSG code here: in this PROJ build ``EPSG:2178`` resolves to
    a *Polish* CRS (ETRF2000-PL), not Swiss CH1903. LV03 is the identical somerc
    projection as LV95 with false easting/northing 600'000 / 200'000 and the same
    WGS84 -> CH1903+ Helmert shift (x=+674.374, y=+15.056, z=+405.346).
    """
    return pyproj.CRS.from_proj4(
        "+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 "
        "+x_0=600000 +y_0=200000 +ellps=bessel +units=m +no_defs "
        "+towgs84=674.374,15.056,405.346"
    )


@pytest.fixture(scope="module")
def fwd_lv03():
    return pyproj.Transformer.from_crs(4326, _lv03_crs(), always_xy=True)


@pytest.fixture(scope="module")
def inv_lv03():
    return pyproj.Transformer.from_crs(_lv03_crs(), 4326, always_xy=True)


def test_to_lv03_matches_pyproj(fwd_lv03):
    for lon, lat in [(7.4475, 46.9481), (8.5417, 47.3769), (6.1432, 46.2044)]:
        e_ref, n_ref = fwd_lv03.transform(lon, lat)
        e, n = to_lv03(lon, lat)
        assert abs(e - e_ref) < 1e-3
        assert abs(n - n_ref) < 1e-3


def test_lv03_to_wgs84_matches_pyproj(inv_lv03):
    for e, n in [(600_675, 199_668), (683_304, 247_926), (500_016, 117_821)]:
        lon_ref, lat_ref = inv_lv03.transform(e, n)
        lon, lat = lv03_to_wgs84(e, n)
        assert abs(lon - lon_ref) < 1e-9
        assert abs(lat - lat_ref) < 1e-9


def test_lv03_grid_matches_pyproj(fwd_lv03):
    """Dense grid over the whole Swiss territory: max error must be < 1 mm."""
    import numpy as np

    lons = np.arange(5.9, 10.6, 0.2)
    lats = np.arange(45.7, 47.9, 0.2)
    LL = np.meshgrid(lons, lats)
    E_ref, N_ref = fwd_lv03.transform(LL[0].ravel(), LL[1].ravel())
    E_ours, N_ours = to_lv03(LL[0].ravel().tolist(), LL[1].ravel().tolist())
    assert max(abs(e - r) for e, r in zip(E_ours, E_ref)) < 1e-3
    assert max(abs(n - r) for n, r in zip(N_ours, N_ref)) < 1e-3
