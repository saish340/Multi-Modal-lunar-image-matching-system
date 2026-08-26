"""RIFT2 detection + matching + RANSAC filtering between scale-adjusted images.

Same interface shape as sift_baseline.py so scorer.py and run_comparison.py
can reuse it.  RIFT2 is NOT scale-invariant, so the caller must still
downsample the higher-resolution image before passing it in (via scale_pyramid).

Notes on RIFT2 vs SIFT:
- Keypoint detection uses phase congruency (illumination invariant) instead of
  DoG (intensity gradient based).  This is the core advantage for cross-sensor
  and cross-illumination matching.
- The descriptor is built from a Maximum Index Map (MIM) constructed from a
  log-Gabor convolution sequence, not from gradient histograms.
- RIFT2 (vs RIFT1) uses a dominant-index rotation invariance technique that
  avoids constructing multiple MIMs, cutting runtime ~3x.
- RIFT2's internal outlier removal uses MAGSAC (via cv2.USAC_MAGSAC), which
  tends to be more robust than plain RANSAC on small inlier-set problems.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Add the cloned reference repo to the path so we can import RIFT2
_REF_DIR = Path(__file__).resolve().parent.parent / "rift_reference"
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))

try:  # package-style import when used as a module...
    from .scale_pyramid import ScaledImage
except ImportError:  # ...and flat import when scripts run directly
    from scale_pyramid import ScaledImage  # type: ignore[no-redef]

from src.RIFT2 import RIFT2 as _RIFT2Core  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class RIFTMatchResult:
    """Outcome of one RIFT2+MAGSAC matching run."""

    points_a_original: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    points_b_original: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    inlier_indices: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    keypoint_count_a: int = 0
    keypoint_count_b: int = 0
    raw_match_count: int = 0
    ratio_test_matches: int = 0
    inlier_count: int = 0
    homography_adjusted: np.ndarray | None = None
    homography_original: np.ndarray | None = None

    keypoints_a_adj: list = field(default_factory=list)
    keypoints_b_adj: list = field(default_factory=list)
    good_matches: list = field(default_factory=list)

    # RIFT-specific timing info
    time_detect_s: float = 0.0
    time_match_s: float = 0.0
    time_ransac_s: float = 0.0


def _scale_matrix(factor: float) -> np.ndarray:
    return np.array([[factor, 0.0, 0.0], [0.0, factor, 0.0], [0.0, 0.0, 1.0]])


def _compose_original_homography(
    homography_adjusted: np.ndarray,
    factor_a: float,
    factor_b: float,
) -> np.ndarray:
    return _scale_matrix(factor_b) @ homography_adjusted @ _scale_matrix(1.0 / factor_a)


def run_rift_matching(
    scaled_a: ScaledImage,
    scaled_b: ScaledImage,
    ratio_thresh: float = 0.75,
    ransac_reproj_thresh: float = 5.0,
    npt: int = 5000,
    patch_size: int = 96,
    is_ori: int = 1,
    nfeatures: int = 0,
) -> RIFTMatchResult:
    """Detect (phase congruency + MIM), match (BF + Lowe ratio), and
    RANSAC-filter RIFT2 features.

    Parameters
    ----------
    scaled_a, scaled_b : ScaledImage
        Cropped + resolution-adjusted images (same as for run_sift_matching).
    ratio_thresh : float
        Lowe's ratio test threshold (RIFT2 descriptors are often matched with
        a more relaxed ratio than SIFT, e.g. 0.75 vs 0.7).
    ransac_reproj_thresh : float
        Reprojection threshold in adjusted-space pixels for MAGSAC.
    npt : int
        Max keypoints per image (RIFT2's FAST detector parameter).
    patch_size : int
        Patch size for MIM descriptor extraction (96 is the authors' default).
    is_ori : int
        Whether to compute keypoint orientation (1=yes, 0=no).
    """
    logger.info(
        "RIFT2 on A: %dx%d (adj) | RIFT2 on B: %dx%d (adj)",
        scaled_a.image.shape[1],
        scaled_a.image.shape[0],
        scaled_b.image.shape[1],
        scaled_b.image.shape[0],
    )

    img_a = scaled_a.image if scaled_a.image.ndim == 2 else cv2.cvtColor(scaled_a.image, cv2.COLOR_BGR2GRAY)
    img_b = scaled_b.image if scaled_b.image.ndim == 2 else cv2.cvtColor(scaled_b.image, cv2.COLOR_BGR2GRAY)

    # Ensure uint8 (RIFT2's phasecong converts internally, but be safe)
    if img_a.dtype != np.uint8:
        img_a = img_a.astype(np.uint8)
    if img_b.dtype != np.uint8:
        img_b = img_b.astype(np.uint8)

    rift2 = _RIFT2Core(npt=npt, patch_size=patch_size, is_ori=is_ori)

    t0 = time.time()
    kp1, des1, kp2, des2 = rift2(img_a, img_b)
    t_detect = time.time() - t0

    result = RIFTMatchResult()
    result.keypoint_count_a = len(kp1)
    result.keypoint_count_b = len(kp2)
    result.keypoints_a_adj = kp1
    result.keypoints_b_adj = kp2
    result.time_detect_s = t_detect

    logger.info("Keypoints: A=%d B=%d (detect: %.1fs)", len(kp1), len(kp2), t_detect)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        logger.warning("Not enough RIFT2 features to match")
        return result

    # --- Matching (BF + Lowe ratio) ---
    t1 = time.time()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    result.raw_match_count = sum(len(pair) for pair in knn)

    good = [
        best
        for best, second in (pair for pair in knn if len(pair) == 2)
        if best.distance < ratio_thresh * second.distance
    ]
    result.ratio_test_matches = len(good)
    result.good_matches = good
    t_match = time.time() - t1
    result.time_match_s = t_match

    logger.info(
        "Raw knn pairs: %d | after Lowe ratio (< %.2f): %d (match: %.3fs)",
        result.raw_match_count,
        ratio_thresh,
        len(good),
        t_match,
    )

    if len(good) < 4:
        logger.warning("Only %d ratio-test matches; MAGSAC needs >= 4", len(good))
        return result

    # --- Outlier removal (MAGSAC, same as RIFT2 reference) ---
    t2 = time.time()
    src_adj = np.float32([kp1[gm.queryIdx].pt for gm in good])
    dst_adj = np.float32([kp2[gm.trainIdx].pt for gm in good])

    homography_adj, inlier_mask = cv2.findHomography(
        src_adj.reshape(-1, 1, 2),
        dst_adj.reshape(-1, 1, 2),
        cv2.USAC_MAGSAC,
        ransac_reproj_thresh,
        maxIters=10000,
    )
    t_ransac = time.time() - t2
    result.time_ransac_s = t_ransac

    if homography_adj is None or inlier_mask is None:
        logger.warning("findHomography returned None; treating all matches as outliers")
        inlier_mask = np.zeros((len(good), 1), dtype=np.uint8)
        homography_adj = np.eye(3)

    inlier_mask = inlier_mask.ravel().astype(bool)
    result.inlier_count = int(inlier_mask.sum())
    result.homography_adjusted = homography_adj
    result.homography_original = _compose_original_homography(
        homography_adj, scaled_a.downsample_factor, scaled_b.downsample_factor
    )

    logger.info(
        "MAGSAC (thresh %.1f px adj): %d/%d inliers (%.1f%%) [%.3fs]",
        ransac_reproj_thresh,
        result.inlier_count,
        len(good),
        100.0 * result.inlier_count / max(len(good), 1),
        t_ransac,
    )

    # Map matched points back to original-resolution pixels
    result.points_a_original = scaled_a.to_original(src_adj)
    result.points_b_original = scaled_b.to_original(dst_adj)
    result.inlier_indices = np.flatnonzero(inlier_mask)

    return result
