"""SIFT detection + matching + RANSAC filtering between scale-adjusted images.

Pipeline position: consumes :class:`scale_pyramid.ScaledImage` pairs (so the
~30x OHRC/TMC resolution gap is already handled) and returns correspondences
mapped back to ORIGINAL-resolution pixel coordinates, ready for scoring
against the Phase 0 ground truth.

Notes:
- Matching and homography estimation happen in ADJUSTED space where both
  images share comparable m/px; reported point correspondences are mapped
  back before returning.
- The estimated homography describes adjusted-A -> adjusted-B pixel motion;
  a composed original-resolution homography is provided for convenience, but
  remember pushbroom geometry makes any single homography approximate
  (see groundtruth.py limitations).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

try:  # package-style import when used as a module...
    from .scale_pyramid import ScaledImage
except ImportError:  # ...and flat import when scripts run directly
    from scale_pyramid import ScaledImage  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Outcome of one SIFT+RANSAC matching run."""

    # Original-resolution correspondence points, aligned by index.
    points_a_original: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    points_b_original: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    # Subset indices into the arrays above that survived RANSAC.
    inlier_indices: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    keypoint_count_a: int = 0
    keypoint_count_b: int = 0
    raw_match_count: int = 0  # brute-force knn pairs generated
    ratio_test_matches: int = 0  # survivors of Lowe's ratio test
    inlier_count: int = 0  # survivors of RANSAC
    homography_adjusted: np.ndarray | None = None  # adj-A -> adj-B
    homography_original: np.ndarray | None = None  # orig-A -> orig-B

    # Kept for visualisation (adjusted space).
    keypoints_a_adj: list = field(default_factory=list)
    keypoints_b_adj: list = field(default_factory=list)
    good_matches: list = field(default_factory=list)


def _scale_matrix(factor: float) -> np.ndarray:
    return np.array([[factor, 0.0, 0.0], [0.0, factor, 0.0], [0.0, 0.0, 1.0]])


def compose_original_homography(
    homography_adjusted: np.ndarray,
    factor_a: float,
    factor_b: float,
) -> np.ndarray:
    """Compose adj-A -> adj-B homography into orig-A -> orig-B pixels.

    orig = S(f) . adj, so H_orig = S(f_b) . H_adj . S(1/f_a).
    """
    return _scale_matrix(factor_b) @ homography_adjusted @ _scale_matrix(1.0 / factor_a)


def run_sift_matching(
    scaled_a: ScaledImage,
    scaled_b: ScaledImage,
    ratio_thresh: float = 0.7,
    ransac_reproj_thresh: float = 8.0,
    nfeatures: int = 20000,
) -> MatchResult:
    """Detect, match (Lowe ratio test), and RANSAC-filter SIFT features."""
    logger.info(
        "SIFT on A: %dx%d (adj) | SIFT on B: %dx%d (adj)",
        scaled_a.image.shape[1],
        scaled_a.image.shape[0],
        scaled_b.image.shape[1],
        scaled_b.image.shape[0],
    )

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints_a, descriptors_a = sift.detectAndCompute(scaled_a.image, None)
    keypoints_b, descriptors_b = sift.detectAndCompute(scaled_b.image, None)
    logger.info(
        "Keypoints: A=%d B=%d", len(keypoints_a), len(keypoints_b)
    )

    result = MatchResult()
    result.keypoint_count_a = len(keypoints_a)
    result.keypoint_count_b = len(keypoints_b)
    result.keypoints_a_adj = keypoints_a
    result.keypoints_b_adj = keypoints_b

    if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 2 or len(keypoints_b) < 2:
        logger.warning("Not enough keypoints/descriptors to match")
        return result

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    result.raw_match_count = sum(len(pair) for pair in knn)

    good = [
        best
        for best, second in (pair for pair in knn if len(pair) == 2)
        if best.distance < ratio_thresh * second.distance
    ]
    result.ratio_test_matches = len(good)
    result.good_matches = good
    logger.info(
        "Raw knn pairs: %d | after Lowe ratio (< %.2f): %d",
        result.raw_match_count,
        ratio_thresh,
        len(good),
    )

    if len(good) < 4:
        logger.warning("Only %d ratio-test matches; RANSAC needs >= 4", len(good))
        return result

    src_adj = np.float32([keypoints_a[gm.queryIdx].pt for gm in good])
    dst_adj = np.float32([keypoints_b[gm.trainIdx].pt for gm in good])

    homography_adj, inlier_mask = cv2.findHomography(
        src_adj.reshape(-1, 1, 2),
        dst_adj.reshape(-1, 1, 2),
        cv2.RANSAC,
        ransac_reproj_thresh,
        maxIters=10000,
    )
    if homography_adj is None or inlier_mask is None:
        logger.warning("findHomography returned None; treating all matches as outliers")
        inlier_mask = np.zeros((len(good), 1), dtype=np.uint8)
        homography_adj = np.eye(3)

    inlier_mask = inlier_mask.ravel().astype(bool)
    result.inlier_count = int(inlier_mask.sum())
    result.homography_adjusted = homography_adj
    result.homography_original = compose_original_homography(
        homography_adj, scaled_a.downsample_factor, scaled_b.downsample_factor
    )
    logger.info(
        "RANSAC (thresh %.1f px adj): %d/%d inliers (%.1f%%)",
        ransac_reproj_thresh,
        result.inlier_count,
        len(good),
        100.0 * result.inlier_count / max(len(good), 1),
    )

    # Map matched points back to original-resolution pixels, aligned by index.
    pts_a_adj = src_adj
    pts_b_adj = dst_adj
    result.points_a_original = scaled_a.to_original(pts_a_adj)
    result.points_b_original = scaled_b.to_original(pts_b_adj)
    result.inlier_indices = np.flatnonzero(inlier_mask)

    return result
