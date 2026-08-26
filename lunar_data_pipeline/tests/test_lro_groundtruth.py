"""Tests for lro_groundtruth — crater GT loading, filtering, and synthetic generation."""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
try:
    from lunar_data_pipeline.lro_groundtruth import (
        CraterRecord,
        filter_by_region,
        generate_synthetic_craters,
        load_crater_database,
        product_bounds,
    )
except ImportError:
    from ..lro_groundtruth import (  # type: ignore[no-redef]
        CraterRecord,
        filter_by_region,
        generate_synthetic_craters,
        load_crater_database,
        product_bounds,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a small sample crater CSV (Robbins-like format)."""
    csv_file = tmp_path / "test_craters.csv"
    rows = [
        {"Longitude": "25.20", "Latitude": "-13.50", "Diameter_km": "3.2", "Depth_km": "0.5", "Name": "TestA"},
        {"Longitude": "25.25", "Latitude": "-13.10", "Diameter_km": "7.5", "Depth_km": "1.2", "Name": "TestB"},
        {"Longitude": "25.80", "Latitude": "-10.00", "Diameter_km": "5.0", "Depth_km": "0.8", "Name": "TestC"},
        {"Longitude": "100.00", "Latitude": "20.00", "Diameter_km": "12.0", "Depth_km": "2.0", "Name": "TestD"},
        {"Longitude": "25.18", "Latitude": "-13.80", "Diameter_km": "0.5", "Depth_km": "", "Name": "TooSmall"},
    ]
    with csv_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Longitude", "Latitude", "Diameter_km", "Depth_km", "Name"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_file


@pytest.fixture
def lu_format_csv(tmp_path: Path) -> Path:
    """Create a CSV in LU1319373 format (different column names)."""
    csv_file = tmp_path / "lu_craters.csv"
    rows = [
        {"lon": "25.22", "lat": "-13.30", "diameter": "4.0"},
        {"lon": "25.19", "lat": "-13.60", "diameter": "2.5"},
        {"lon": "50.00", "lat": "10.00", "diameter": "8.0"},
    ]
    with csv_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lon", "lat", "diameter"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_file


# ---------------------------------------------------------------------------
# Tests — load_crater_database
# ---------------------------------------------------------------------------

class TestLoadDatabase:

    def test_load_robbins_format(self, sample_csv: Path):
        craters = load_crater_database(sample_csv)
        assert len(craters) == 5
        assert craters[0].lon == pytest.approx(25.20)
        assert craters[0].lat == pytest.approx(-13.50)
        assert craters[0].diameter_km == pytest.approx(3.2)
        assert craters[0].depth_km == pytest.approx(0.5)
        assert craters[0].name == "TestA"

    def test_load_lu_format(self, lu_format_csv: Path):
        craters = load_crater_database(lu_format_csv)
        assert len(craters) == 3
        assert craters[0].diameter_km == pytest.approx(4.0)

    def test_min_diameter_filter(self, sample_csv: Path):
        craters = load_crater_database(sample_csv, min_diameter_km=1.0)
        # "TooSmall" with 0.5 km should be excluded
        assert all(c.diameter_km >= 1.0 for c in craters)
        assert len(craters) == 4

    def test_max_diameter_filter(self, sample_csv: Path):
        craters = load_crater_database(sample_csv, max_diameter_km=6.0)
        assert all(c.diameter_km <= 6.0 for c in craters)

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_crater_database("nonexistent.csv")

    def test_bad_header(self, tmp_path: Path):
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("col_a,col_b,col_c\n1,2,3\n")
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            load_crater_database(csv_file)


# ---------------------------------------------------------------------------
# Tests — filter_by_region
# ---------------------------------------------------------------------------

class TestFilterRegion:

    def test_basic_filter(self, sample_csv: Path):
        craters = load_crater_database(sample_csv)
        region = filter_by_region(craters, lon_min=25.0, lon_max=25.5, lat_min=-14.0, lat_max=-12.8)
        assert len(region) == 3  # TestA, TestB, TooSmall
        assert all(25.0 <= c.lon <= 25.5 for c in region)
        assert all(-14.0 <= c.lat <= -12.8 for c in region)

    def test_empty_region(self, sample_csv: Path):
        craters = load_crater_database(sample_csv)
        region = filter_by_region(craters, lon_min=90.0, lon_max=91.0, lat_min=0.0, lat_max=1.0)
        assert len(region) == 0


# ---------------------------------------------------------------------------
# Tests — synthetic generator
# ---------------------------------------------------------------------------

class TestSyntheticGenerator:

    def test_basic_generation(self):
        craters = generate_synthetic_craters(25.0, 25.5, -14.0, -12.8, count=15)
        assert len(craters) == 15
        assert all(c.lon >= 25.0 and c.lon <= 25.5 for c in craters)
        assert all(c.lat >= -14.0 and c.lat <= -12.8 for c in craters)
        assert all(c.diameter_km >= 1.0 for c in craters)
        assert all(c.diameter_km <= 15.0 for c in craters)

    def test_reproducible_with_seed(self):
        a = generate_synthetic_craters(0, 1, 0, 1, count=5, seed=123)
        b = generate_synthetic_craters(0, 1, 0, 1, count=5, seed=123)
        for ca, cb in zip(a, b):
            assert ca.lat == pytest.approx(cb.lat)
            assert ca.lon == pytest.approx(cb.lon)
            assert ca.diameter_km == pytest.approx(cb.diameter_km)

    def test_names_assigned(self):
        craters = generate_synthetic_craters(0, 1, 0, 1, count=3)
        assert craters[0].name == "SYN-000"
        assert craters[2].name == "SYN-002"


# ---------------------------------------------------------------------------
# Tests — CraterRecord
# ---------------------------------------------------------------------------

class TestCraterRecord:

    def test_radius_px_at(self):
        c = CraterRecord(lat=0, lon=0, diameter_km=2.0)
        # 2 km at 0.2 m/px → radius = 2000 / (2*0.2) = 5000 px
        assert c.radius_px_at(0.2) == pytest.approx(5000.0)

    def test_radius_px_at_tmc2(self):
        c = CraterRecord(lat=0, lon=0, diameter_km=10.0)
        # 10 km at 6.13 m/px → radius = 10000 / (2*6.13) ≈ 815.6 px
        assert c.radius_px_at(6.13) == pytest.approx(815.6, rel=0.01)


# ---------------------------------------------------------------------------
# Tests — product_bounds (with a mock product)
# ---------------------------------------------------------------------------

class TestProductBounds:

    def test_product_bounds(self):
        class MockProduct:
            footprint = [(-13.5, 25.0), (-13.5, 25.5), (-13.0, 25.5), (-13.0, 25.0)]
        lon_min, lon_max, lat_min, lat_max = product_bounds(MockProduct())
        assert lon_min == pytest.approx(25.0)
        assert lon_max == pytest.approx(25.5)
        assert lat_min == pytest.approx(-13.5)
        assert lat_max == pytest.approx(-13.0)
