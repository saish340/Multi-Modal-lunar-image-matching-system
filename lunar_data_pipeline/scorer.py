"""Score predicted pixel correspondences against the Phase 0 ground truth.

A predicted correspondence ``(x_a, y_a) -> (x_b, y_b)`` is evaluated by
projecting ``(x_a, y_a)`` through the ground-truth transform (groundtruth.py)
to get the EXPECTED location in image B, then measuring Euclidean pixel
distance to the prediction. Predictions within a configurable tolerance count
as correct.

Caveat carried over from groundtruth.py: the ground truth is a corner-based
homography approximation (~5% along-track deviation observed on this pair).
Errors are therefore an upper bound on true matching error, and a large
SYSTEMATIC offset usually indicates ground-truth bias rather than bad
matches. Mean dx/dy are reported explicitly so that distinction is visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

try:  # package-style import when used as a module...
    from .groundtruth import PixelTransform
except ImportError:  # ...and flat import when scripts run directly
    from groundtruth import PixelTransform  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

#: Fraction of tolerance above which a consistent mean offset is flagged as
#: likely systematic (ground-truth bias) instead of random matching error.
SYSTEMATIC_BIAS_FRACTION = 0.3


@dataclass
class ScoreReport:
    total_scored: int = 0
    correct: int = 0
    percent_correct: float = 0.0
    tolerance_px: float = 0.0
    errors_px: np.ndarray = None  # type: ignore[assignment]
    mean_dx_px: float = 0.0  # expected - predicted, averaged
    mean_dy_px: float = 0.0
    systematic_bias_suspected: bool = False

    def stats(self) -> dict:
        if self.errors_px is None or len(self.errors_px) == 0:
            return {}
        e = self.errors_px
        return {
            "min": float(e.min()),
            "max": float(e.max()),
            "mean": float(e.mean()),
            "median": float(np.median(e)),
            "p90": float(np.percentile(e, 90)),
        }

    def summary_dict(self) -> dict:
        return {
            "total_scored": self.total_scored,
            "correct_within_tolerance": self.correct,
            "percent_correct": round(self.percent_correct, 3),
            "tolerance_px": self.tolerance_px,
            "error_distribution_px": self.stats(),
            "mean_dx_expected_minus_predicted_px": round(self.mean_dx_px, 3),
            "mean_dy_expected_minus_predicted_px": round(self.mean_dy_px, 3),
            "systematic_bias_suspected": self.systematic_bias_suspected,
        }


def score_correspondences(
    points_a: np.ndarray,
    points_b_predicted: np.ndarray,
    transform: PixelTransform,
    tolerance_px: float = 75.0,
) -> ScoreReport:
    """Score N predicted correspondences against the ground-truth transform.

    ``points_a`` and ``points_b_predicted`` are (N, 2) arrays in ORIGINAL-
    resolution pixel coordinates of images A and B respectively.
    """
    points_a = np.atleast_2d(np.asarray(points_a, dtype=float))
    points_b_predicted = np.atleast_2d(np.asarray(points_b_predicted, dtype=float))
    if len(points_a) != len(points_b_predicted):
        raise ValueError("points_a and points_b_predicted must have equal length")
    if tolerance_px <= 0:
        raise ValueError("tolerance_px must be positive")

    if len(points_a) == 0:
        logger.warning("No correspondences to score")
        return ScoreReport(tolerance_px=tolerance_px, errors_px=np.empty(0))

    expected_b = transform.project_many(points_a)
    diffs = points_b_predicted - expected_b
    errors = np.hypot(diffs[:, 0], diffs[:, 1])

    correct_mask = errors <= tolerance_px
    mean_dx = float(np.mean(expected_b[:, 0] - points_b_predicted[:, 0]))
    mean_dy = float(np.mean(expected_b[:, 1] - points_b_predicted[:, 1]))

    bias_threshold = SYSTEMATIC_BIAS_FRACTION * tolerance_px
    bias = abs(mean_dx) > bias_threshold or abs(mean_dy) > bias_threshold

    report = ScoreReport(
        total_scored=len(errors),
        correct=int(correct_mask.sum()),
        percent_correct=100.0 * correct_mask.sum() / len(errors),
        tolerance_px=tolerance_px,
        errors_px=errors,
        mean_dx_px=mean_dx,
        mean_dy_px=mean_dy,
        systematic_bias_suspected=bias,
    )
    logger.info(
        "Scored %d correspondences: %d correct within %.1f px (%.1f%%)",
        report.total_scored,
        report.correct,
        tolerance_px,
        report.percent_correct,
    )
    return report


def format_report(report: ScoreReport) -> str:
    """Human-readable multi-line summary for CLI output."""
    lines = [
        f"Scored {report.total_scored} predictions against ground truth "
        f"(tolerance {report.tolerance_px:g} TMC-2 px):",
    ]
    if report.total_scored == 0:
        lines.append("  no predictions to evaluate")
        return "\n".join(lines)

    s = report.stats()
    lines.append(
        f"  correct: {report.correct}/{report.total_scored} "
        f"({report.percent_correct:.1f}%)"
    )
    lines.append(
        f"  error px min/median/mean/p90/max: "
        f"{s['min']:.1f} / {s['median']:.1f} / {s['mean']:.1f} / "
        f"{s['p90']:.1f} / {s['max']:.1f}"
    )
    lines.append(
        f"  mean offset (expected - predicted): dx={report.mean_dx_px:+.1f} px, "
        f"dy={report.mean_dy_px:+.1f} px"
    )
    if report.systematic_bias_suspected:
        lines.append(
            "  NOTE: consistent mean offset detected -- likely ground-truth "
            "transform bias (corner-homography approximation), not random "
            "matching failure. Interpret accordingly."
        )
    else:
        lines.append("  no systematic offset detected (errors look unbiased)")
    return "\n".join(lines)
