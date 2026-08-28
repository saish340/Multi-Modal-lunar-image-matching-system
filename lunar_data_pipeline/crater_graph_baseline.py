"""Crater-neighborhood-graph registration baseline.

Detects craters in both crops, builds a local neighborhood graph in each,
matches the graphs, converts matched crater centers into pixel
correspondences, and fits a RANSAC homography -- mirroring the interface
shape of :mod:`sift_baseline` / :mod:`rift_baseline` so :mod:`scorer` and the
comparison runner work unmodified.

Coordinate bookkeeping (see crater_graph.py and scale_pyramid.py):

- Matching happens in an ADJUSTED space in which both images are at a
  comparable metres-per-pixel (OHRC downsampled toward TMC-2 resolution).
  The OHRC detections live in the downsampled-ScaledImage frame; the TMC-2
  detections live in the crop-local frame.  Both are ~6.13 m/px, so the
  RANSAC homography is fit in this common-metric space.
- Reported point correspondences are mapped back to ORIGINAL-resolution
  pixel coordinates for scoring:
    * image A (OHRC):   points * OHRC downsample factor  (via ScaledImage)
    * image B (TMC-2):  crop-local points + crop pixel/scan offset
  because :func:`scorer.score_correspondences` projects A points through the
  ground-truth transform into FULL TMC-2 image coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from .crater_detector import CraterDetection
from .crater_graph import CraterGraph, build_graph
from .graph_matcher import (
    Correspondence,
    match_consistent,
    match_mutual_neighbor,
    match_nearest_neighbor,
    matches_to_pixel_pairs,
)

logger = logging.getLogger(__name__)


@dataclass
class CraterGraphMatchResult:
    """Outcome of one crater-graph + RANSAC matching run (SIFT-shaped)."""

    # Original-resolution correspondence points, aligned by index.
    points_a_original: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    points_b_original: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    # Subset indices into the arrays above that survived RANSAC.
    inlier_indices: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    # Node / match counts.
    node_count_a: int = 0
    node_count_b: int = 0
    raw_match_count: int = 0
    consistency_match_count: int = 0
    inlier_count: int = 0

    homography_adjusted: np.ndarray | None = None  # adj-A -> adj-B
    homography_original: np.ndarray | None = None  # orig-A -> orig-B

    # Crater centres in adjusted space, aligned by index (for visualisation).
    src_adjusted: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    dst_adjusted: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))

    correspondences: list[Correspondence] = field(default_factory=list)
    graph_a: CraterGraph | None = None
    graph_b: CraterGraph | None = None


def _scale_matrix(fx: float, fy: float | None = None) -> np.ndarray:
    if fy is None:
        fy = fx
    return np.array(
        [[fx, 0.0, 0.0], [0.0, fy, 0.0], [0.0, 0.0, 1.0]]
    )


def compose_original_homography(
    homography_adjusted: np.ndarray,
    factor_a: float,
    factor_b: float,
    factor_a_y: float | None = None,
    factor_b_y: float | None = None,
) -> np.ndarray:
    """Compose adj-A -> adj-B homography into orig-A -> orig-B pixels.

    Factor components follow :func:`ScaledImage.to_original`:
    ``orig = S_b @ adj @ S_a^-1`` where the scale matrices may be anisotropic
    (per-axis factors) when the downsampled images are not square.
    """
    if factor_a_y is None:
        factor_a_y = factor_a
    if factor_b_y is None:
        factor_b_y = factor_b
    return _scale_matrix(factor_b, factor_b_y) @ homography_adjusted @ _scale_matrix(1.0 / factor_a, 1.0 / factor_a_y)


def run_crater_graph_matching(
    craters_a: Sequence[CraterDetection],
    craters_b: Sequence[CraterDetection],
    *,
    meters_per_px_a: float,
    meters_per_px_b: float,
    ohrc_downsample_factor: float,
    tmc_crop_offset: tuple[float, float],
    ohrc_downsample_factor_y: float | None = None,
    k: int = 4,
    min_radius_m: float = 0.0,
    ratio_test: float = 0.6,
    min_similarity: float = 0.25,
    sigma: float = 0.5,
    max_nodes_a: int | None = None,
    max_nodes_b: int | None = None,
    ransac_reproj_thresh: float = 15.0,
    use_consistency: bool = False,
    matcher: str = "neighbor_identity",
    n_candidates_per_node: int = 8,
    min_spectrum_similarity: float = 0.1,
    neighbor_d_tol: float = 0.6,
    neighbor_s_tol: float = 0.6,
    min_mutual_fraction: float = 0.5,
    neighbor_physical: bool = False,
    neighbor_use_ratio_test: bool = True,
) -> CraterGraphMatchResult:
    """Build graphs, match them, and fit a RANSAC homography.

    Parameters
    ----------
    craters_a, craters_b :
        Detected craters for image A (OHRC, in the downsampled frame) and
        image B (TMC-2, in crop-local pixels).
    meters_per_px_a, meters_per_px_b :
        Effective resolutions of the two detection frames (both ~6.13 for our
        pair) used to normalise graph features into physical metres.
    ohrc_downsample_factor :
        ``scaled_a.downsample_factor`` used to map A detections back to
        original OHRC pixels.
    tmc_crop_offset :
        ``(pixel_offset, scan_offset)`` added to crop-local B pixels to get
        full TMC-2 image coordinates.
    """
    result = CraterGraphMatchResult()
    result.node_count_a = len(craters_a)
    result.node_count_b = len(craters_b)

    graph_a = build_graph(
        [c.cx for c in craters_a], [c.cy for c in craters_a],
        [c.radius for c in craters_a], meters_per_px_a,
        k=k, min_radius_m=min_radius_m,
        confidence=[c.confidence for c in craters_a],
    )
    graph_b = build_graph(
        [c.cx for c in craters_b], [c.cy for c in craters_b],
        [c.radius for c in craters_b], meters_per_px_b,
        k=k, min_radius_m=min_radius_m,
        confidence=[c.confidence for c in craters_b],
    )
    result.graph_a = graph_a
    result.graph_b = graph_b
    result.node_count_a = len(graph_a.nodes)
    result.node_count_b = len(graph_b.nodes)

    if matcher == "neighbor_identity":
        matches = match_mutual_neighbor(
            graph_a, graph_b,
            ratio_test=ratio_test,
            sigma=sigma,
            n_candidates_per_node=n_candidates_per_node,
            min_spectrum_similarity=min_spectrum_similarity,
            neighbor_d_tol=neighbor_d_tol,
            neighbor_s_tol=neighbor_s_tol,
            min_mutual_fraction=min_mutual_fraction,
            physical=neighbor_physical,
            use_ratio_test=neighbor_use_ratio_test,
        )
    else:
        matches = match_nearest_neighbor(
            graph_a, graph_b,
            ratio_test=ratio_test,
            min_similarity=min_similarity,
            sigma=sigma,
            max_nodes_a=max_nodes_a,
            max_nodes_b=max_nodes_b,
        )
    result.raw_match_count = len(matches)
    result.correspondences = matches
    logger.info("Graph match: %d raw correspondences (%s)", len(matches), matcher)

    if use_consistency:
        matches = match_consistent(graph_a, graph_b, matches)
        result.consistency_match_count = len(matches)
        result.correspondences = matches
        logger.info("After consistency pruning: %d", len(matches))

    if len(matches) < 4:
        logger.warning("Only %d matches; RANSAC needs >= 4", len(matches))
        return result

    src_adj, dst_adj = matches_to_pixel_pairs(graph_a, graph_b, matches)
    src_adj = np.asarray(src_adj, dtype=float).reshape(-1, 1, 2)
    dst_adj = np.asarray(dst_adj, dtype=float).reshape(-1, 1, 2)
    result.src_adjusted = src_adj.reshape(-1, 2)
    result.dst_adjusted = dst_adj.reshape(-1, 2)

    homography_adj, inlier_mask = cv2.findHomography(
        src_adj, dst_adj, cv2.RANSAC, ransac_reproj_thresh, maxIters=10000
    )
    if homography_adj is None or inlier_mask is None:
        logger.warning("findHomography returned None; treating all as outliers")
        inlier_mask = np.zeros((len(matches), 1), dtype=np.uint8)
        homography_adj = np.eye(3)

    inlier_mask = inlier_mask.ravel().astype(bool)
    result.inlier_count = int(inlier_mask.sum())
    result.homography_adjusted = homography_adj
    result.homography_original = compose_original_homography(
        homography_adj, ohrc_downsample_factor, 1.0,
        ohrc_downsample_factor_y if ohrc_downsample_factor_y is not None else ohrc_downsample_factor,
        1.0,
    )
    logger.info(
        "RANSAC (thresh %.1f px adj): %d/%d inliers (%.1f%%)",
        ransac_reproj_thresh, result.inlier_count, len(matches),
        100.0 * result.inlier_count / max(len(matches), 1),
    )

    # Map to original-resolution pixel coordinates (aligned by index).
    src_flat = src_adj.reshape(-1, 2)
    dst_flat = dst_adj.reshape(-1, 2)
    if ohrc_downsample_factor_y is None:
        ohrc_downsample_factor_y = ohrc_downsample_factor
    result.points_a_original = src_flat * np.array([ohrc_downsample_factor, ohrc_downsample_factor_y])
    px0, sy0 = tmc_crop_offset
    result.points_b_original = dst_flat + np.array([px0, sy0])
    result.inlier_indices = np.flatnonzero(inlier_mask)

    return result
