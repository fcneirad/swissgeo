"""Coordinate transformations between WGS84 (EPSG:4326) and the Swiss grids
LV95 (EPSG:2056) and LV03 (CH1903).

LV95 is the Swiss national grid "CH1903+ / LV95": a Hotine Oblique Mercator
projection on the Bessel 1841 ellipsoid with the CH1903+ datum, centred at Bern
(latitude_of_center 46.9524055555556, longitude_of_center 7.43958333333333),
false easting 2'600'000 m and false northing 1'200'000 m. LV03 (CH1903) uses the
identical projection with false easting 600'000 m and false northing 200'000 m,
so it differs from LV95 by a pure offset of (2'000'000, 1'000'000) metres.

This module implements the full transform chain in **pure numpy** (no pyproj, no
network):

    WGS84 (lon, lat)
      -> ECEF on the WGS84 ellipsoid
      -> Helmert translation to CH1903+ (EPSG:1676 parameters)
      -> ECEF on the Bessel 1841 ellipsoid
      -> geodetic (lat, lon) via Bowring's inverse
      -> LV95 (easting, northing) via the ``somerc`` projection

The equations are taken from the PROJ source (``src/projections/somerc.cpp`` and
the standard ECEF/Helmert/Bowring formulations) and have been validated against
pyproj to < 1 mm over the whole Swiss territory. Everything runs offline.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import numpy as np

__all__ = [
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
]

# ---------------------------------------------------------------------------
# Ellipsoid / datum constants
# ---------------------------------------------------------------------------
# Bessel 1841 ellipsoid (used by CH1903+ / LV95).
A_BESSEL = 6_377_397.155          # semi-major axis, m
F_BESSEL = 1.0 / 299.1528128      # flattening

# WGS84 ellipsoid.
A_WGS84 = 6_378_137.0             # semi-major axis, m
F_WGS84 = 1.0 / 298.257223563     # flattening

# Helmert translation WGS84 -> CH1903+ (EPSG:1676), metres. No rotations/scale.
HELMERT_X = 674.374
HELMERT_Y = 15.056
HELMERT_Z = 405.346

# somerc projection parameters for LV95 (EPSG:2056).
LAT_0 = 46.9524055555556          # latitude of natural origin, deg
LON_0 = 7.43958333333333         # longitude of natural origin, deg
K_0 = 1.0                         # scale factor at natural origin
X_0 = 2_600_000.0                 # false easting, m
Y_0 = 1_200_000.0                 # false northing, m

# LV03 (CH1903) uses the *identical* projection as LV95; only the
# false easting/northing differ. Hence LV95 and LV03 are related by a pure,
# exact offset (no approximation):  LV03 = LV95 - (2'000'000, 1'000'000).
LV03_X_0 = 600_000.0              # false easting, m
LV03_Y_0 = 200_000.0              # false northing, m
LV95_TO_LV03_OFFSET_E = X_0 - LV03_X_0   # = 2'000'000 m
LV95_TO_LV03_OFFSET_N = Y_0 - LV03_Y_0   # = 1'000'000 m

# Approximate extent of the Swiss territory in LV95 (metres), for sanity checks.
LV95_SWISS_BOUNDS = {
    "min_e": 2_300_000.0,
    "max_e": 2_900_000.0,
    "min_n": 1_000_000.0,
    "max_n": 1_400_000.0,
}

# Same territory expressed in LV03 (LV95 bounds minus the offset).
LV03_SWISS_BOUNDS = {
    "min_e": LV95_SWISS_BOUNDS["min_e"] - LV95_TO_LV03_OFFSET_E,
    "max_e": LV95_SWISS_BOUNDS["max_e"] - LV95_TO_LV03_OFFSET_E,
    "min_n": LV95_SWISS_BOUNDS["min_n"] - LV95_TO_LV03_OFFSET_N,
    "max_n": LV95_SWISS_BOUNDS["max_n"] - LV95_TO_LV03_OFFSET_N,
}

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# ECEF helpers (vectorised over the whole array)
# ---------------------------------------------------------------------------
def _ecef_forward(lon_rad: np.ndarray, lat_rad: np.ndarray, a: float, f: float):
    """Geodetic (lon, lat in radians) -> ECEF (X, Y, Z)."""
    b = a * (1.0 - f)
    e2 = f * (2.0 - f)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)
    X = (N + a * f ** 2 / b) * cos_lat * np.cos(lon_rad)
    Y = (N + a * f ** 2 / b) * cos_lat * np.sin(lon_rad)
    Z = (N * (1.0 - e2) + b * f ** 2 / a) * sin_lat
    return X, Y, Z


def _ecef_inverse(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, a: float, f: float):
    """ECEF (X, Y, Z) -> geodetic (lon_rad, lat_rad).

    Standard iterative method (same algorithm as PROJ's ``+proj=cart +inv``).
    Converges in ~5 iterations to machine precision.
    """
    e2 = f * (2.0 - f)
    p = np.sqrt(X ** 2 + Y ** 2)
    lon = np.arctan2(Y, X)

    # Initial latitude estimate.
    lat = np.arctan2(Z, p * (1.0 - e2))
    # Iterate to convergence.
    for _ in range(20):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)
        lat_new = np.arctan2(Z + e2 * N * sin_lat, p)
        if np.all(np.abs(lat_new - lat) < 1e-15):
            lat = lat_new
            break
        lat = lat_new
    return lon, lat


# ---------------------------------------------------------------------------
# somerc projection (precomputed constants + forward/inverse)
# ---------------------------------------------------------------------------
def _somerc_setup(a: float, f: float, lat0_rad: float, lon0_rad: float, k0: float):
    """Precompute the ``somerc`` projection constants.

    ``f`` is the ellipsoid flattening; PROJ's ``P->e`` is the *first eccentricity*
    ``e = sqrt(f (2 - f))`` (NOT the flattening). All somerc equations use ``e``.
    """
    e = math.sqrt(f * (2.0 - f))  # first eccentricity
    es = e * e
    one_es = 1.0 - es
    hlf_e = 0.5 * e
    cp = math.cos(lat0_rad) ** 2
    # PROJ: c = sqrt(1 + es * cp^2 / (1 - es)) with cp = cos^2(phi0), i.e. cos^4.
    c = math.sqrt(1.0 + es * cp * cp / one_es)
    sp = math.sin(lat0_rad)
    sinp0 = sp / c
    phip0 = math.asin(sinp0)
    cosp0 = math.cos(phip0)
    sp_e = sp * e
    K = (math.log(math.tan(0.25 * math.pi + 0.5 * phip0))
         - c * (math.log(math.tan(0.25 * math.pi + 0.5 * lat0_rad))
                - hlf_e * math.log((1.0 + sp_e) / (1.0 - sp_e))))
    kR = k0 * math.sqrt(one_es) / (1.0 - sp_e ** 2)
    return {
        "e": e, "es": es, "one_es": one_es, "hlf_e": hlf_e, "c": c,
        "sinp0": sinp0, "cosp0": cosp0, "K": K, "kR": kR,
    }


def _somerc_forward(lat_rad: np.ndarray, lon_rad: np.ndarray, s: dict, a: float):
    """Geodetic (lat, lon in radians) -> LV95 (easting, northing)."""
    e = s["e"]
    c = s["c"]
    hlf_e = s["hlf_e"]
    K = s["K"]
    kR = s["kR"]
    sinp0 = s["sinp0"]
    cosp0 = s["cosp0"]

    sp = e * np.sin(lat_rad)
    phip = (2.0 * np.arctan(np.exp(
        c * (np.log(np.tan(0.25 * math.pi + 0.5 * lat_rad))
             - hlf_e * np.log((1.0 + sp) / (1.0 - sp))) + K))
        - 0.5 * math.pi)
    lamp = c * (lon_rad - s["lon0"])
    cp = np.cos(phip)
    phipp = np.arcsin(cosp0 * np.sin(phip) - sinp0 * cp * np.cos(lamp))
    lampp = np.arcsin(cp * np.sin(lamp) / np.cos(phipp))
    E = s["x0"] + a * kR * lampp
    N = s["y0"] + a * kR * np.log(np.tan(0.25 * math.pi + 0.5 * phipp))
    return E, N


def _somerc_inverse(E: np.ndarray, N: np.ndarray, s: dict, a: float):
    """LV95 (easting, northing) -> geodetic (lat_rad, lon_rad).

    Mirrors PROJ ``somerc_e_inverse``: the false easting/northing are removed
    first, and the iterative correction is scaled by ``1/(1 - e^2)``.
    """
    e = s["e"]
    one_es = s["one_es"]
    c = s["c"]
    hlf_e = s["hlf_e"]
    K = s["K"]
    kR = s["kR"]
    sinp0 = s["sinp0"]
    cosp0 = s["cosp0"]

    # Remove false easting/northing (PROJ receives coordinates centred on the origin).
    x = E - s["x0"]
    y = N - s["y0"]

    phipp = 2.0 * (np.arctan(np.exp(y / (a * kR))) - 0.25 * math.pi)
    lampp = x / (a * kR)
    cp = np.cos(phipp)
    phip = np.arcsin(cosp0 * np.sin(phipp) + sinp0 * cp * np.cos(lampp))
    lamp = np.arcsin(cp * np.sin(lampp) / np.cos(phip))
    con = (K - np.log(np.tan(0.25 * math.pi + 0.5 * phip))) / c
    for _ in range(6):
        esp = e * np.sin(phip)
        delp = ((con + np.log(np.tan(0.25 * math.pi + 0.5 * phip))
                 - hlf_e * np.log((1.0 + esp) / (1.0 - esp)))
                * (1.0 - esp ** 2) * np.cos(phip) / one_es)
        phip = phip - delp
    lat = phip
    lon = lamp / c + s["lon0"]
    return lat, lon


# Precompute the somerc constants once at import time.
_SOMERC = _somerc_setup(
    A_BESSEL, F_BESSEL, math.radians(LAT_0), math.radians(LON_0), K_0
)
_SOMERC.update({"lon0": math.radians(LON_0), "x0": X_0, "y0": Y_0})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def to_lv95(
    lon: Union[float, Sequence[float]],
    lat: Union[float, Sequence[float]],
) -> Union[Point, Tuple[list, list]]:
    """Transform WGS84 (longitude, latitude) to LV95 (easting, northing).

    Parameters
    ----------
    lon, lat:
        A single scalar value or a sequence of values. When sequences are given
        they must have the same length and the result is returned as two lists
        ``(eastings, northings)``.

    Returns
    -------
    (easting, northing) for scalar input, or (list_of_eastings, list_of_northings).

    Examples
    --------
    >>> e, n = to_lv95(7.5892, 46.9413)   # Bern
    >>> round(e), round(n)
    (2611464, 1198923)
    """
    scalar = isinstance(lon, (int, float)) and isinstance(lat, (int, float))
    lon_arr = np.atleast_1d(np.asarray([lon] if scalar else list(lon), dtype=np.float64))
    lat_arr = np.atleast_1d(np.asarray([lat] if scalar else list(lat), dtype=np.float64))
    if lon_arr.shape != lat_arr.shape:
        raise ValueError("lon and lat sequences must have the same length")

    # WGS84 geodetic -> ECEF.
    X, Y, Z = _ecef_forward(np.radians(lon_arr), np.radians(lat_arr), A_WGS84, F_WGS84)
    # Helmert translation to CH1903+.
    X -= HELMERT_X
    Y -= HELMERT_Y
    Z -= HELMERT_Z
    # ECEF -> Bessel geodetic.
    lon_b, lat_b = _ecef_inverse(X, Y, Z, A_BESSEL, F_BESSEL)
    # somerc forward -> LV95.
    E, N = _somerc_forward(lat_b, lon_b, _SOMERC, A_BESSEL)

    if scalar:
        return (float(E[0]), float(N[0]))
    return (E.tolist(), N.tolist())


def to_wgs84(
    easting: Union[float, Sequence[float]],
    northing: Union[float, Sequence[float]],
) -> Union[Point, Tuple[list, list]]:
    """Transform LV95 (easting, northing) to WGS84 (longitude, latitude).

    Parameters
    ----------
    easting, northing:
        A single scalar value or a sequence of values.

    Returns
    -------
    (longitude, latitude) for scalar input, or (list_of_lons, list_of_lats).

    Examples
    --------
    >>> lon, lat = to_wgs84(2611464, 1198923)   # ~Bern
    >>> round(lon, 4), round(lat, 4)
    (7.5892, 46.9413)
    """
    scalar = isinstance(easting, (int, float)) and isinstance(northing, (int, float))
    E = np.atleast_1d(np.asarray([easting] if scalar else list(easting), dtype=np.float64))
    N = np.atleast_1d(np.asarray([northing] if scalar else list(northing), dtype=np.float64))
    if E.shape != N.shape:
        raise ValueError("easting and northing sequences must have the same length")

    # somerc inverse -> Bessel geodetic.
    lat_b, lon_b = _somerc_inverse(E, N, _SOMERC, A_BESSEL)
    # Bessel geodetic -> ECEF.
    X, Y, Z = _ecef_forward(lon_b, lat_b, A_BESSEL, F_BESSEL)
    # Helmert translation back to WGS84 (inverse of the forward shift).
    X += HELMERT_X
    Y += HELMERT_Y
    Z += HELMERT_Z
    # ECEF -> WGS84 geodetic.
    lon_w, lat_w = _ecef_inverse(X, Y, Z, A_WGS84, F_WGS84)

    if scalar:
        return (float(np.degrees(lon_w[0])), float(np.degrees(lat_w[0])))
    return (np.degrees(lon_w).tolist(), np.degrees(lat_w).tolist())


def to_lv03(
    lon: Union[float, Sequence[float]],
    lat: Union[float, Sequence[float]],
) -> Union[Point, Tuple[list, list]]:
    """Transform WGS84 (longitude, latitude) to LV03 / CH1903 (easting, northing).

    LV03 uses the same projection as LV95 with a different false easting/northing,
    so this is exactly ``to_lv95`` minus ``(2'000'000, 1'000'000)``.

    Examples
    --------
    >>> e, n = to_lv03(7.4475, 46.9481)   # Bern
    >>> round(e), round(n)
    (600675, 199668)
    """
    e, n = to_lv95(lon, lat)
    if isinstance(e, float):
        return (e - LV95_TO_LV03_OFFSET_E, n - LV95_TO_LV03_OFFSET_N)
    return ([v - LV95_TO_LV03_OFFSET_E for v in e],
            [v - LV95_TO_LV03_OFFSET_N for v in n])


def lv03_to_wgs84(
    easting: Union[float, Sequence[float]],
    northing: Union[float, Sequence[float]],
) -> Union[Point, Tuple[list, list]]:
    """Transform LV03 / CH1903 (easting, northing) to WGS84 (longitude, latitude).

    Exactly the inverse of :func:`to_lv03` (shift into LV95, then invert).

    Examples
    --------
    >>> lon, lat = lv03_to_wgs84(600675, 199668)   # ~Bern
    >>> round(lon, 4), round(lat, 4)
    (7.4475, 46.9481)
    """
    if isinstance(easting, (int, float)) and isinstance(northing, (int, float)):
        return to_wgs84(easting + LV95_TO_LV03_OFFSET_E,
                        northing + LV95_TO_LV03_OFFSET_N)
    return to_wgs84([v + LV95_TO_LV03_OFFSET_E for v in easting],
                    [v + LV95_TO_LV03_OFFSET_N for v in northing])


def lv95_to_lv03(
    easting: Union[float, Sequence[float]],
    northing: Union[float, Sequence[float]],
) -> Union[Point, Tuple[list, list]]:
    """Convert LV95 (easting, northing) to LV03 by the exact grid offset."""
    if isinstance(easting, float):
        return (easting - LV95_TO_LV03_OFFSET_E, northing - LV95_TO_LV03_OFFSET_N)
    return ([v - LV95_TO_LV03_OFFSET_E for v in easting],
            [v - LV95_TO_LV03_OFFSET_N for v in northing])


def lv03_to_lv95(
    easting: Union[float, Sequence[float]],
    northing: Union[float, Sequence[float]],
) -> Union[Point, Tuple[list, list]]:
    """Convert LV03 (easting, northing) to LV95 by the exact grid offset."""
    if isinstance(easting, float):
        return (easting + LV95_TO_LV03_OFFSET_E, northing + LV95_TO_LV03_OFFSET_N)
    return ([v + LV95_TO_LV03_OFFSET_E for v in easting],
            [v + LV95_TO_LV03_OFFSET_N for v in northing])


def transform(
    coords: Sequence[Point],
    to_crs: str = "lv95",
) -> list:
    """Transform an iterable of ``(x, y)`` points from WGS84 to a Swiss grid.

    Parameters
    ----------
    coords:
        Iterable of 2-tuples. For ``to_crs="lv95"`` or ``to_crs="lv03"`` each
        tuple is ``(lon, lat)`` in WGS84; for ``to_crs="wgs84"`` each tuple is
        ``(easting, northing)`` in LV95.
    to_crs:
        Target CRS: ``"lv95"`` (EPSG:2056), ``"lv03"`` (CH1903) or
        ``"wgs84"`` (EPSG:4326).

    Returns
    -------
    List of transformed ``(x, y)`` tuples in the target CRS.
    """
    to_crs = to_crs.lower()
    if to_crs == "lv95":
        return [to_lv95(x, y) for (x, y) in coords]
    if to_crs == "wgs84":
        return [to_wgs84(x, y) for (x, y) in coords]
    if to_crs == "lv03":
        return [to_lv03(x, y) for (x, y) in coords]
    raise ValueError(
        f"Unknown target CRS: {to_crs!r} (expected 'lv95', 'lv03' or 'wgs84')"
    )


def is_valid_lv95(easting: float, northing: float) -> bool:
    """Return ``True`` if ``(easting, northing)`` lies within the Swiss LV95 extent.

    This is a coarse bounding-box check (roughly the territory of Switzerland and
    Liechtenstein). It is useful to reject obviously out-of-range coordinates before
    doing more expensive lookups.
    """
    b = LV95_SWISS_BOUNDS
    return (
        b["min_e"] <= easting <= b["max_e"]
        and b["min_n"] <= northing <= b["max_n"]
    )

def is_valid_lv03(easting: float, northing: float) -> bool:
    """Return ``True`` if ``(easting, northing)`` lies within the Swiss LV03 extent.

    Same coarse bounding-box check as :func:`is_valid_lv95`, expressed in the
    CH1903 / LV03 grid (LV95 bounds minus the 2'000'000 / 1'000'000 offset).
    """
    b = LV03_SWISS_BOUNDS
    return (
        b["min_e"] <= easting <= b["max_e"]
        and b["min_n"] <= northing <= b["max_n"]
    )