#!/usr/bin/env python3
"""One-time conversion of raw swisstopo files into swissgeo runtime formats.

Thin wrapper around :mod:`swissgeo.setup_data` (the single source of truth,
also used by the ``swissgeo fetch-data`` CLI command).

The COFS (GeoPackage) step uses only the Python standard library; the
elevation (GeoTIFF) step needs ``rasterio``. If rasterio is not installed in
the current interpreter, a *throwaway venv* is created on top of it, rasterio
is installed **only** there, and the temp venv is deleted afterwards -- your
environment is never modified.

Usage::

    python scripts/convert_data.py [--data-dir data] [--skip-elevation]
                                   [--no-download] [--gpkg PATH] [--tif PATH]

    # fetch missing raw files from data.geo.admin.ch first (default) and convert:
    python scripts/convert_data.py
    # COFS only, no rasterio needed at all:
    python scripts/convert_data.py --skip-elevation
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data",
                    help="directory containing the raw files (default: data)")
    ap.add_argument("--gpkg", default=None,
                    help="path to swissBOUNDARIES3D gpkg (default: auto-detect)")
    ap.add_argument("--tif", default=None,
                    help="path to swissALTIRegio tif (default: auto-detect)")
    ap.add_argument("--skip-elevation", action="store_true",
                    help="only convert the COFS boundaries (no rasterio needed)")
    ap.add_argument("--no-download", action="store_true",
                    help="do not fetch missing raw files from data.geo.admin.ch")
    args = ap.parse_args(argv)

    # Make the project root importable when running from a source checkout
    # (sys.path[0] is this script's directory, not the repo root).
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    # Import lazily so `--help` works even without numpy installed.
    from swissgeo.setup_data import run_setup

    try:
        run_setup(
            data_dir=args.data_dir,
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


if __name__ == "__main__":
    raise SystemExit(main())
