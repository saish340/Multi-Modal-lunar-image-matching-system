"""Tests for groundtruth corner-homography pixel correspondence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunar_data_pipeline.groundtruth import (
    PixelTransform,
    build_pixel_transform,
    project_point,
    transform_from_manifest_dict,
    transform_to_manifest_dict,
)
from lunar_data_pipeline.pds4_parser import LunarProduct


def make_product(
    product_id: str,
    lines: int,
    samples: int,
    lat_range=(-10.0, 10.0),
    lon_range=(20.0, 40.0),
) -> LunarProduct:
    """Synthetic product whose footprint is a clean lon/lat rectangle.

    Corners in CORNER_ORDER (UL, UR, LR, LL).
    """
    lat0, lat1 = lat_range
    lon0, lon1 = lon_range
    return LunarProduct(
        instrument="OHRC",
        file_path=Path(f"/fake/{product_id}.img"),
        label_path=Path(f"/fake/{product_id}.xml"),
        lines=lines,
        samples=samples,
        datatype="UnsignedByte",
        footprint=[
            (lat1, lon0),  # upper_left  -> pixel (0, 0)
            (lat1, lon1),  # upper_right -> (W-1, 0)
            (lat0, lon1),  # lower_right -> (W-1, H-1)
            (lat0, lon0),  # lower_left  -> (0, H-1)
        ],
        product_id=product_id,
    )


class TestBuildPixelTransform:
    def test_identity_when_products_identical(self):
        a = make_product("a", lines=100, samples=200)
        b = make_product("b", lines=100, samples=200)

        xy = project_point((50, 25), a, b)

        assert xy == pytest.approx((50, 25), abs=1e-6)

    def test_resolution_scaling_between_same_footprint_images(self):
        # Same geographic rectangle; B has twice the resolution of A.
        # Under the corner-pixel-centre convention, A's centre pixel
        # (49.5, 49.5) must land on B's centre pixel (99.5, 99.5).
        a = make_product("lowres", lines=100, samples=100)
        b = make_product("hires", lines=200, samples=200)

        xy_b = build_pixel_transform(a, b).project((49.5, 49.5))

        assert xy_b == pytest.approx((99.5, 99.5), abs=1e-6)

    def test_corners_map_to_corners(self):
        a = make_product("a", lines=101, samples=201)
        b = make_product("b", lines=51, samples=76)
        transform = build_pixel_transform(a, b)

        corners_a = np.array(
            [[0.0, 0.0], [200.0, 0.0], [200.0, 100.0], [0.0, 100.0]]
        )
        mapped = transform.project_many(corners_a)

        expected = np.array(
            [[0.0, 0.0], [75.0, 0.0], [75.0, 50.0], [0.0, 50.0]]
        )
        assert mapped == pytest.approx(expected, abs=1e-6)

    def test_project_many_matches_single_projection(self):
        a = make_product("a", lines=100, samples=200)
        b = make_product("b", lines=150, samples=300)
        transform = build_pixel_transform(a, b)

        points = np.array([[10.0, 10.0], [150.5, 42.25]])
        batch = transform.project_many(points)
        single = [transform.project((x, y)) for x, y in points]

        assert batch == pytest.approx(np.asarray(single), abs=1e-9)

    def test_incomplete_footprint_raises(self):
        broken = make_product("broken", lines=10, samples=10)
        broken.footprint = []
        ok = make_product("ok", lines=10, samples=10)

        with pytest.raises(ValueError):
            build_pixel_transform(broken, ok)


class TestManifestRoundTrip:
    def test_serialize_deserialize_preserves_projection(self):
        a = make_product("a", lines=120, samples=240)
        b = make_product("b", lines=60, samples=80)
        original = build_pixel_transform(a, b)

        data = transform_to_manifest_dict(original)
        revived = transform_from_manifest_dict(data)

        assert isinstance(revived, PixelTransform)
        point = (37.0, 81.5)
        assert revived.project(point) == pytest.approx(original.project(point))

    def test_serialised_matrices_are_plain_floats(self):
        a = make_product("a", lines=10, samples=10)
        b = make_product("b", lines=10, samples=10)
        data = transform_to_manifest_dict(build_pixel_transform(a, b))

        for matrix_name in ("h_pixel_to_geo_a", "h_geo_to_pixel_b"):
            for row in data[matrix_name]:
                assert all(isinstance(v, float) for v in row)


class TestKnownLimitations:
    def test_non_rectangular_footprint_still_yields_finite_mapping(self):
        """Homography handles non-affine quadrilaterals but accuracy degrades;
        this only asserts numerical sanity, not geodetic correctness."""
        skewed = make_product("skew", lines=100, samples=100)
        skewed.footprint = [
            (-10.0, 20.0),
            (12.0, 41.0),   # pulled corner
            (-8.0, 39.0),
            (-12.0, 21.0),
        ]
        other = make_product("plain", lines=100, samples=100)

        transform = build_pixel_transform(skewed, other)
        xy = transform.project((50.0, 50.0))

        assert np.all(np.isfinite(xy))
