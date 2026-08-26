"""Tests for csv_geolocation using a synthetic geolocation CSV."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from lunar_data_pipeline.csv_geolocation import (
    fit_local_transforms,
    locate_region,
    read_geolocation_csv,
)


@pytest.fixture()
def synthetic_csv(tmp_path):
    """Grid: lon = 20 + 0.01*pixel + 0.002*scan ; lat = -10 - 0.01*scan."""
    path = tmp_path / "ch2_tmc_x_g_grd_d18.csv"
    lines = ["Longitude,Latitude,Pixel,Scan"]
    for scan in range(0, 201, 5):
        for pixel in range(0, 101, 5):
            lon = 20 + 0.01 * pixel + 0.002 * scan
            lat = -10 - 0.01 * scan
            lines.append(f"{lon:.6f},{lat:.6f},{pixel},{scan}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestReadGeolocationCsv:
    def test_parses_all_rows(self, synthetic_csv):
        grid = read_geolocation_csv(synthetic_csv)
        assert len(grid) == 41 * 21
        assert grid.pixel[0] == 0
        assert grid.scan[-1] == 200

    def test_skips_garbage_rows(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text(
            "Longitude,Latitude,Pixel,Scan\n"
            "20.0,-10.0,0,0\n"
            "not,a,number,row\n"
            "20.05,-10.05,5,5\n"
            "\n",
            encoding="utf-8",
        )
        grid = read_geolocation_csv(path)
        assert len(grid) == 2


class TestLocateRegion:
    def test_bbox_of_points_inside_polygon(self, synthetic_csv):
        grid = read_geolocation_csv(synthetic_csv)
        # lon 20.2..20.4, lat -10.2..-10.5 -> pixel 20..40-ish, scan 20..50-ish
        poly = Polygon([(20.2, -10.2), (20.4, -10.2), (20.4, -10.5), (20.2, -10.5)])
        scan_rng, pix_rng, n = locate_region(grid, poly)
        assert n > 0
        assert scan_rng[0] >= 15 and scan_rng[1] <= 55
        assert pix_rng[0] >= 10 and pix_rng[1] <= 45

    def test_empty_region_raises(self, synthetic_csv):
        grid = read_geolocation_csv(synthetic_csv)
        far = Polygon([(99, 99), (100, 99), (100, 100), (99, 100)])
        with pytest.raises(ValueError):
            locate_region(grid, far)


class TestFitLocalTransforms:
    def test_round_trip_accuracy(self, synthetic_csv):
        grid = read_geolocation_csv(synthetic_csv)
        h_px_to_geo, h_geo_to_px, n = fit_local_transforms(
            grid, (0, 200), (0, 100)
        )
        assert n == 41 * 21

        # Analytic forward mapping for an arbitrary window point.
        px, scan = 63.0, 147.0
        expected_lon = 20 + 0.01 * px + 0.002 * scan
        expected_lat = -10 - 0.01 * scan
        from lunar_data_pipeline.csv_geolocation import _apply

        geo = _apply(h_px_to_geo, np.array([[px, scan]]))[0]
        assert geo[0] == pytest.approx(expected_lon, abs=1e-6)
        assert geo[1] == pytest.approx(expected_lat, abs=1e-6)

        back = _apply(h_geo_to_px, geo.reshape(1, -1))[0]
        assert back[0] == pytest.approx(px, abs=1e-3)
        assert back[1] == pytest.approx(scan, abs=1e-3)

    def test_too_few_points_raises(self, synthetic_csv):
        grid = read_geolocation_csv(synthetic_csv)
        with pytest.raises(ValueError):
            fit_local_transforms(grid, (0, 3), (0, 3), min_points=50)

