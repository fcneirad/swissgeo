"""Command-line interface for swissgeo.

Examples
--------
Transform coordinates (WGS84, LV95 and LV03 are interchangeable)::

    swissgeo transform 7.4475 46.9481                    # WGS84 -> LV95
    swissgeo transform 7.4475 46.9481 --to lv03         # WGS84 -> LV03 (CH1903)
    swissgeo transform --to wgs84 2600675 1199668       # LV95 -> WGS84
    swissgeo transform --from lv03 --to lv95 600675 199668   # LV03 -> LV95

Look up COFS / elevation (fully offline, local data)::

    swissgeo cofs 2611464 1198923
    swissgeo elevation 2611464 1198923

Check that the local datasets are present::

    swissgeo data

Download and convert the raw swisstopo datasets (one-time setup)::

    swissgeo fetch-data                 # COFS + elevation
    swissgeo fetch-data --skip-elevation   # COFS only (no rasterio needed)
"""

from __future__ import annotations

import argparse
import json
import sys


_CRS_CHOICES = ("wgs84", "lv95", "lv03")


def _route(x, y, src, dst):
    """Convert a single (x, y) point between WGS84, LV95 and LV03.

    Grid<->grid conversions use the exact LV95/LV03 offset; everything else is
    routed through WGS84. Returns ``(crs, a, b)`` in the target CRS.
    """
    from .transforms import (
        to_lv95, to_wgs84, to_lv03, lv03_to_wgs84, lv95_to_lv03, lv03_to_lv95,
    )

    if src in ("lv95", "lv03") and dst in ("lv95", "lv03"):
        if src == dst:
            return dst, x, y
        e, n = (lv95_to_lv03(x, y) if src == "lv95" else lv03_to_lv95(x, y))
        return dst, e, n

    if src == "wgs84":
        lon, lat = x, y
    elif src == "lv95":
        lon, lat = to_wgs84(x, y)
    else:  # lv03
        lon, lat = lv03_to_wgs84(x, y)

    if dst == "wgs84":
        return "wgs84", lon, lat
    if dst == "lv95":
        e, n = to_lv95(lon, lat)
        return "lv95", e, n
    e, n = to_lv03(lon, lat)
    return "lv03", e, n


def _cmd_transform(args: argparse.Namespace) -> int:
    src = args.src
    dst = args.to
    if src is None:
        # Backward-compatible default: a grid target implies WGS84 input, and the
        # wgs84 target implies LV95 input (the original behaviour).
        src = "lv95" if dst == "wgs84" else "wgs84"

    crs, a, b = _route(args.x, args.y, src, dst)
    if crs == "wgs84":
        print(f"WGS84 lon={a:.6f}  lat={b:.6f}")
    elif crs == "lv95":
        print(f"LV95  E={a:.3f}  N={b:.3f}")
    else:
        print(f"LV03  E={a:.3f}  N={b:.3f}")
    return 0


def _cmd_cofs(args: argparse.Namespace) -> int:
    from .cofs import CofsLookup

    lookup = CofsLookup(data_dir=args.data)
    m = lookup.from_lv95(args.e, args.n)
    if m is None:
        print("No municipality found for this point.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"cofs": m.cofs, "name": m.name, "canton": m.canton}))
    else:
        print(str(m))
    return 0


def _cmd_elevation(args: argparse.Namespace) -> int:
    from .elevation import ElevationLookup

    lookup = ElevationLookup(data_dir=args.data)
    elev = lookup.from_lv95(args.e, args.n)
    if elev is None:
        print("No elevation found for this point.", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"value": elev.value, "source": elev.source}))
    else:
        print(str(elev))
    return 0


def _cmd_fetch_data(args: argparse.Namespace) -> int:
    from .setup_data import run_setup

    try:
        run_setup(
            data_dir=args.data,
            want_cofs=True,
            want_elevation=not args.skip_elevation,
            download=not args.no_download,
            cofs_path=args.gpkg,
            elevation_path=args.tif,
        )
    except SystemExit as e:
        if e.code:
            print(str(e.code), file=sys.stderr)
        return int(e.code or 0)
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from .data import DataStore

    store = DataStore(data_dir=args.data)
    cofs_ok = store.cofs_ready()
    elev_ok = store.elevation_ready()
    print(f"data dir   : {store.data_dir}")
    print(f"cofs       : {'OK' if cofs_ok else 'MISSING'}  ({store.cofs_path})")
    print(f"elevation  : {'OK' if elev_ok else 'MISSING'}  ({store.elevation_path})")
    if not (cofs_ok and elev_ok):
        print("\nRun `python scripts/convert_data.py` to produce the missing data.",
              file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="swissgeo", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transform", help="Transform coordinates between WGS84, LV95 and LV03")
    t.add_argument("x", type=float, help="longitude (WGS84) or easting (LV95/LV03)")
    t.add_argument("y", type=float, help="latitude (WGS84) or northing (LV95/LV03)")
    t.add_argument("--from", dest="src", choices=_CRS_CHOICES, default=None,
                   help="source CRS (default: wgs84 for grid targets, lv95 for --to wgs84)")
    t.add_argument("--to", choices=_CRS_CHOICES, default="lv95")
    t.set_defaults(func=_cmd_transform)

    c = sub.add_parser("cofs", help="Look up COFS municipality code from LV95")
    c.add_argument("e", type=float, help="LV95 easting")
    c.add_argument("n", type=float, help="LV95 northing")
    c.add_argument("--data", default=None, help="path to cofs_lv95 data directory")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=_cmd_cofs)

    e = sub.add_parser("elevation", help="Look up elevation from LV95")
    e.add_argument("e", type=float, help="LV95 easting")
    e.add_argument("n", type=float, help="LV95 northing")
    e.add_argument("--data", default=None, help="path to elevation_10m data directory")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=_cmd_elevation)

    f = sub.add_parser(
        "fetch-data",
        help="Download and convert the raw swisstopo datasets (one-time setup)")
    f.add_argument("--data", default="data",
                   help="directory for raw files + converted outputs (default: data)")
    f.add_argument("--gpkg", default=None,
                   help="path to swissBOUNDARIES3D gpkg (default: auto-detect)")
    f.add_argument("--tif", default=None,
                   help="path to swissALTIRegio tif (default: auto-detect)")
    f.add_argument("--skip-elevation", action="store_true",
                   help="only fetch/convert the COFS boundaries (no rasterio needed)")
    f.add_argument("--no-download", action="store_true",
                   help="do not fetch missing raw files from data.geo.admin.ch")
    f.set_defaults(func=_cmd_fetch_data)

    d = sub.add_parser("data", help="Check that the local datasets are present")
    d.add_argument("--data", default=None, help="path to the data directory")
    d.set_defaults(func=_cmd_data)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
