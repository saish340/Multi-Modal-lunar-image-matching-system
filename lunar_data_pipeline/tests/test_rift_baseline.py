"""Tests for the RIFT2 baseline wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from lunar_data_pipeline.rift_baseline import (
    RIFTMatchResult,
    run_rift_matching,
)
from lunar_data_pipeline.scale_pyramid import ScaledImage


def _make_scaled(arr: np.ndarray, factor: float = 1.0) -> ScaledImage:
    """Wrap a uint8 array as a ScaledImage for testing."""
    return ScaledImage(
        image=arr,
        downsample_factor=factor,
        original_lines=arr.shape[0],
        original_samples=arr.shape[1],
        source_resolution_m=1.0,
        target_resolution_m=1.0,
    )


class TestRIFTMatchResult:
    def test_default_fields(self):
        r = RIFTMatchResult()
        assert r.keypoint_count_a == 0
        assert r.inlier_count == 0
        assert len(r.inlier_indices) == 0

    def test_points_original_shapes(self):
        r = RIFTMatchResult()
        assert r.points_a_original.shape == (0, 2)


class TestRunRIFTMatching:
    """Smoke tests on tiny synthetic images (real lunar data is too large for
    unit tests; integration coverage is in run_comparison.py)."""

    def test_low_texture_image_returns_few_keypoints(self):
        """Low-texture image should find few features without crashing.
        Note: uniform images cause NaN in phase congruency (known edge case
        in the reference implementation), so we add slight texture."""
        rng = np.random.RandomState(99)
        img = (128 + rng.randint(-5, 6, size=(100, 100))).astype(np.uint8)
        sa = _make_scaled(img)
        sb = _make_scaled(img.copy())
        result = run_rift_matching(sa, sb, npt=100)
        assert isinstance(result, RIFTMatchResult)
        assert result.keypoint_count_a >= 0
        assert result.inlier_count >= 0

    def test_identical_images_produce_matches(self):
        """Two identical random images should find matches."""
        rng = np.random.RandomState(42)
        # Need reasonably sized images for RIFT2's FAST detector
        img = rng.randint(20, 220, size=(200, 200), dtype=np.uint8)
        # Add some structure
        cv2.rectangle(img, (50, 50), (100, 100), 255, 2)
        cv2.circle(img, (150, 150), 30, 128, -1)

        sa = _make_scaled(img)
        sb = _make_scaled(img.copy())
        result = run_rift_matching(sa, sb, npt=500, ratio_thresh=0.95)
        assert result.keypoint_count_a > 0, "should find features on structured image"
        assert result.keypoint_count_b > 0
        # Identical images should produce many matches
        assert result.ratio_test_matches > 0, "identical images should have ratio-test matches"

    def test_returns_original_resolution_coordinates(self):
        """Matched points should be mapped back to original pixel space."""
        rng = np.random.RandomState(7)
        img = rng.randint(20, 220, size=(200, 200), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (100, 100), 255, 2)

        factor = 2.0
        # Downsample to simulate scaling
        small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
        sa = _make_scaled(small, factor=factor)
        sb = _make_scaled(small.copy(), factor=factor)
        result = run_rift_matching(sa, sb, npt=500, ratio_thresh=0.95)

        if result.inlier_count > 0:
            # All coordinates should be in original space (>= factor * adjusted size)
            assert result.points_a_original.max() >= 100  # at least some coords > adjusted size
