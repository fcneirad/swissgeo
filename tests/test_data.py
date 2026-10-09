"""Tests for DataStore (local data resolution)."""

from __future__ import annotations

import os

import pytest

from swissgeo import DataStore


def test_default_data_dir_exists():
    store = DataStore()
    assert os.path.isdir(store.data_dir)


def test_cofs_path_points_to_directory():
    store = DataStore()
    assert store.cofs_path.endswith("cofs_lv95")
    assert store.elevation_path.endswith("elevation_10m")


def test_env_var_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SWISSGEO_DATA", str(tmp_path))
    store = DataStore()
    assert store.data_dir == str(tmp_path)


def test_ready_flags_and_ensure(tmp_path):
    store = DataStore(data_dir=str(tmp_path))
    # Nothing present yet.
    assert not store.cofs_ready()
    assert not store.elevation_ready()
    with pytest.raises(FileNotFoundError):
        store.ensure_cofs()
    with pytest.raises(FileNotFoundError):
        store.ensure_elevation()
