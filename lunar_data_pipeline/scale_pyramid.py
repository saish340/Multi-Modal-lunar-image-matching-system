"""Handle the resolution gap between OHRC (~0.2 m/px) and TMC-2 (~6 m/px).

The two cameras differ by tens of times in ground sampling distance. SIFT is
scale-invariant in theory, but its practical octave range covers roughly a
10x scale change, so crossing a ~30x baseline gap in one shot wastes most of
the detector's range and hurts matching. We therefore downsample the
high-resolution OHRC image toward the TMC-2 resolution before detection.

Downsampling OHRC (rather than upsampling TMC-2) preserves real information:
every TMC pixel exists in the OHRC scene, never the reverse.

Coordinate bookkeeping matters: keypoint coordinates found in the downsampled
image must be mapped back to original-resolution pixels before scoring
against the Phase 0 ground truth (which is defined in original pixel space).
:class:`ScaledImage` carries the applied factor for exactly this purpose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScaledImage:
    """An image after resolution adjustment, remembering how to get back."""

    image: np.ndarray  # uint8, possibly downsampled
    downsample_factor: float  # >= 1; original px = adjusted px * factor
    original_lines: int
    original_samples: int
    source_resolution_m: float
    target_resolution_m: float

    def to_original(self, points: np.ndarray) -> np.ndarray:
        """Map adjusted-space (x, y) points back to original-resolution pixels."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        return pts * self.downsample_factor

    def __post_init__(self) -> None:
        if self.downsample_factor < 1.0:
            raise ValueError("downsample_factor must be >= 1 (never upsample)")


def compute_downsample_factor(
    source_resolution_m: float, target_resolution_m: float
) -> float:
    """Pixel-merge factor bringing *source_resolution_m* down toward *target*.

    E.g. source=0.2 m/px (OHRC), target=6.13 m/px (TMC-2) -> 6.13/0.2 ~= 30.7,
    i.e. each adjusted pixel merges ~30.7 original pixels and thus spans the
    same ground distance as a TMC-2 pixel. Values below 1 are clamped to 1
    (we never upsample; images already coarser than the target pass through).
    """
    if source_resolution_m <= 0 or target_resolution_m <= 0:
        raise ValueError("resolutions must be positive")
    return max(target_resolution_m / source_resolution_m, 1.0)


def downsample_to_match(
    image: np.ndarray,
    source_resolution_m: float,
    target_resolution_m: float,
    factor_override: float | None = None,
) -> ScaledImage:
    """Downsample *image* so its resolution approaches *target_resolution_m*.

    ``factor_override`` forces an explicit downsample factor instead of the
    resolution-derived one (must still be >= 1).
    """
    lines, samples = image.shape[:2]
    factor = compute_downsample_factor(source_resolution_m, target_resolution_m)
    if factor_override is not None:
        if factor_override < 1.0:
            raise ValueError("factor_override must be >= 1 (never upsample)")
        factor = float(factor_override)

    if factor == 1.0:
        logger.info(
            "No downsampling needed (source %.3f m/px <= target %.3f m/px)",
            source_resolution_m,
            target_resolution_m,
        )
        return ScaledImage(
            image=np.ascontiguousarray(image),
            downsample_factor=1.0,
            original_lines=lines,
            original_samples=samples,
            source_resolution_m=source_resolution_m,
            target_resolution_m=target_resolution_m,
        )

    new_samples = max(int(round(samples / factor)), 1)
    new_lines = max(int(round(lines / factor)), 1)
    # INTER_AREA is the right kernel for decimation (box-like averaging).
    resized = cv2.resize(image, (new_samples, new_lines), interpolation=cv2.INTER_AREA)
    logger.info(
        "Downsampled %dx%d -> %dx%d (factor %.2fx; %.3f -> %.3f m/px effective)",
        samples,
        lines,
        new_samples,
        new_lines,
        factor,
        source_resolution_m,
        source_resolution_m * factor,
    )
    return ScaledImage(
        image=resized,
        downsample_factor=samples / new_samples,  # exact applied factor
        original_lines=lines,
        original_samples=samples,
        source_resolution_m=source_resolution_m,
        target_resolution_m=target_resolution_m,
    )
