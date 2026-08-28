"""Tests for scorer.py (and scale_pyramid coordinate mapping) using synthetic
data with known correct/incorrect answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunar_data_pipeline.groundtruth import build_pixel_transform
from lunar_data_pipeline.pds4_parser import LunarProduct
from lunar_data_pipeline.scale_pyramid import (
    ScaledImage,
    compute_downsample_factor,
    downsample_to_match,
)
from lunar_data_pipeline.scorer import (
    SYSTEMATIC_BIAS_FRACTION,
    format_report,
    score_correspondences,
)


def make_product(product_id: str) -> LunarProduct:
    """100x100 px product covering lon [0, 10], lat [-10, 0] (clean affine)."""
    return LunarProduct(
        instrument="OHRC",
        file_path=Path(f"/fake/{product_id}.img"),
        label_path=Path(f"/fake/{product_id}.xml"),
        lines=100,
        samples=100,
        datatype="UnsignedByte",
        footprint=[(0.0, 0.0), (0.0, 10.0), (-10.0, 10.0), (-10.0, 0.0)],
        product_id=product_id,
    )


@pytest.fixture()
def transform():
    return build_pixel_transform(make_product("a"), make_product("b"))


class TestScoreCorrespondences:
    def test_perfect_matches_all_correct(self, transform):
        pts_a = np.array([[10.0, 20.0], [50.0, 50.0], [90.0, 80.0]])
        pts_b = transform.project_many(pts_a)

        report = score_correspondences(pts_a, pts_b, transform, tolerance_px=5.0)

        assert report.total_scored == 3
        assert report.correct == 3
        assert report.percent_correct == 100.0
        assert report.stats()["max"] < 1e-9

    def test_tolerance_boundary_counts_as_correct(self, transform):
        pts_a = np.array([[50.0, 50.0]])
        expected_b = transform.project_many(pts_a)[0]
        # Displace by exactly the tolerance along x -> must still count.
        pts_b = expected_b + np.array([10.0, 0.0])

        report = score_correspondences(pts_a, pts_b, transform, tolerance_px=10.0)

        assert report.correct == 1

    def test_known_mix_of_good_and_bad(self, transform):
        pts_a = np.array([[25.0, 25.0], [75.0, 75.0], [40.0, 40.0], [60.0, 60.0]])
        good_b = transform.project_many(pts_a)
        predicted_b = good_b.copy()
        predicted_b[2] += np.array([500.0, -800.0])  # far outlier
        predicted_b[3] += np.array([-2000.0, 1500.0])  # far outlier

        report = score_correspondences(pts_a, predicted_b, transform, tolerance_px=50.0)

        assert report.total_scored == 4
        assert report.correct == 2
        assert report.percent_correct == 50.0
        s = report.stats()
        assert s["min"] < 1e-9
        assert s["max"] > 2000.0

    def test_no_bias_flag_for_unbiased_noise(self, transform):
        # Many near-perfect matches with small alternating noise: means ~ 0.
        rng = np.random.default_rng(42)
        pts_a = rng.uniform([10, 10], [90, 90], size=(40, 2))
        predicted_b = transform.project_many(pts_a)
        predicted_b += rng.uniform(-4.0, 4.0, size=(40, 2))

        report = score_correspondences(pts_a, predicted_b, transform, tolerance_px=50.0)

        assert report.correct == 40
        assert not report.systematic_bias_suspected

    def test_systematic_offset_is_flagged(self, transform):
        pts_a = np.array([[10.0 + i * 8.0, 10.0 + i * 7.0] for i in range(10)])
        good_b = transform.project_many(pts_a)
        shifted = good_b + np.array([30.0, 0.0])  # everyone off by same dx

        report = score_correspondences(pts_a, shifted, transform, tolerance_px=50.0)

        assert report.mean_dx_px == pytest.approx(-30.0)  # expected - predicted
        assert report.systematic_bias_suspected

    def test_empty_input_handled(self, transform):
        report = score_correspondences(
            np.empty((0, 2)), np.empty((0, 2)), transform, tolerance_px=50.0
        )
        assert report.total_scored == 0
        assert report.correct == 0
        assert "no predictions" in format_report(report)

    def test_length_mismatch_raises(self, transform):
        with pytest.raises(ValueError):
            score_correspondences(
                np.zeros((3, 2)), np.zeros((4, 2)), transform, tolerance_px=10.0
            )

    def test_invalid_tolerance_raises(self, transform):
        with pytest.raises(ValueError):
            score_correspondences(
                np.zeros((1, 2)), np.zeros((1, 2)), transform, tolerance_px=0.0
            )


class TestScalePyramidMapping:
    def test_factor_computation_and_clamping(self):
        assert compute_downsample_factor(0.2, 6.13) == pytest.approx(30.65)
        assert compute_downsample_factor(6.13, 0.2) == 1.0  # never upsample
        with pytest.raises(ValueError):
            compute_downsample_factor(0.0, 1.0)

    def test_rounding_aware_coordinate_mapping(self):
        # 101x51 image downsampled by factor 10 -> requested size 11x6.
        img = np.zeros((51, 101), np.uint8)
        scaled = downsample_to_match(img, source_resolution_m=0.2,
                                     target_resolution_m=2.0)
        assert scaled.image.shape == (5, 10)
        assert scaled.downsample_factor == pytest.approx(101 / 10)

        pts = scaled.to_original(np.array([[5.0, 2.5], [0.0, 0.0]]))
        # x is scaled by the samples-factor; y by the (independent) lines-factor.
        assert pts[0] == pytest.approx((5 * 101 / 10, 2.5 * 51 / 5))
        assert pts[1][0] == 0.0

    def test_per_axis_factor_when_non_square(self):
        # Non-square OHRC-style image: downsampled lines and samples round to
        # different factors, so y must use its own factor, not the x factor.
        img = np.zeros((101074, 12000), np.uint8)  # matches real OHRC dims
        scaled = downsample_to_match(img, source_resolution_m=0.2,
                                     target_resolution_m=6.13)
        fx = scaled.downsample_factor
        fy = scaled.downsample_factor_y
        assert fx != fy
        assert fx == pytest.approx(12000 / scaled.image.shape[1])
        assert fy == pytest.approx(101074 / scaled.image.shape[0])

        p = scaled.to_original(np.array([[100.0, 200.0]]))
        assert p[0, 0] == pytest.approx(100.0 * fx)
        assert p[0, 1] == pytest.approx(200.0 * fy)
        # single-point (2,) input maps back to a single point
        q = scaled.to_original(np.array([100.0, 200.0]))
        assert q.shape == (2,)
        assert q[0] == pytest.approx(100.0 * fx)
        assert q[1] == pytest.approx(200.0 * fy)

    def test_scaledimage_never_upsamples(self):
        with pytest.raises(ValueError):
            ScaledImage(image=np.zeros((4, 4), np.uint8),
                        downsample_factor=0.5,
                        original_lines=4, original_samples=4,
                        source_resolution_m=1.0, target_resolution_m=2.0)
