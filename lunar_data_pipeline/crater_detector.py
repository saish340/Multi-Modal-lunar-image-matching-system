"""Classical computer-vision crater detection for lunar imagery.

Implements a multi-stage pipeline optimised for high-resolution orbital images
of the lunar surface:

1. Preprocessing: CLAHE + Gaussian blur (illumination normalisation)
2. Morphological top-hat / bottom-hat: isolates crater rims (bright) and
   floors (dark) regardless of local illumination
3. Edge detection: Canny on the enhanced image
4. Circle detection: HoughCircles with tunable parameters
5. Refinement: Gaussian fit to radial intensity profile for sub-pixel centre

The detector is deliberately simple and CPU-only — it demonstrates the
classical CV baseline without requiring GPU / deep-learning models. For
production use, a trained Mask R-CNN or SAM-based detector would improve
recall significantly (see Giannakis et al. 2024, Icarus).

Usage:
    >>> from crater_detector import detect_craters, DetectionParams
    >>> detections = detect_craters(gray_image, params=DetectionParams(min_radius=20))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class DetectionParams:
    """Tunable parameters for the crater detector."""

    # Preprocessing
    clahe_clip: float = 3.0          # CLAHE clip limit
    clahe_grid: int = 8              # CLAHE tile grid size
    blur_ksize: int = 5              # Gaussian blur kernel

    # Morphological enhancement
    tophat_ksize: int = 31           # Top-hat structuring element diameter
    bottomhat_ksize: int = 31        # Bottom-hat structuring element diameter
    use_morphology: bool = True      # Apply morphological enhancement

    # Edge detection
    canny_low: int = 30              # Canny low threshold
    canny_high: int = 100            # Canny high threshold

    # HoughCircles
    min_radius: int = 10             # Minimum crater radius in pixels
    max_radius: int = 500            # Maximum crater radius in pixels
    dp: float = 1.2                  # HoughCircles accumulator resolution
    min_dist: int = 30               # Minimum distance between circle centres
    hough_param1: int = 100          # HoughCircles upper Canny threshold
    hough_param2: int = 35           # HoughCircles accumulator threshold
    min_votes: int = 30              # Minimum accumulator votes

    # Validation
    min_circularity: float = 0.5     # Minimum circularity (4pi*area/perimeter^2)
    max_ellipse_ratio: float = 2.0   # Max ratio of major/minor axes
    edge_fill_min: float = 0.3       # Minimum edge density inside circle


@dataclass
class CraterDetection:
    """A single detected crater."""

    cx: float          # centre x (pixel)
    cy: float          # centre y (pixel)
    radius: float      # radius in pixels
    confidence: float  # 0..1 confidence score
    votes: int = 0     # HoughCircles accumulator votes
    circularity: float = 0.0
    mean_intensity: float = 0.0  # mean pixel value inside the circle


def preprocess(gray: np.ndarray, params: DetectionParams | None = None) -> np.ndarray:
    """Apply CLAHE + Gaussian blur for illumination normalisation."""
    params = params or DetectionParams()

    # CLAHE (adaptive histogram equalisation)
    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip,
        tileGridSize=(params.clahe_grid, params.clahe_grid),
    )
    enhanced = clahe.apply(gray)

    # Gaussian blur to suppress noise
    if params.blur_ksize > 1:
        k = params.blur_ksize | 1  # ensure odd
        enhanced = cv2.GaussianBlur(enhanced, (k, k), 0)

    return enhanced


def morphological_enhance(gray: np.ndarray, params: DetectionParams | None = None) -> np.ndarray:
    """Enhance crater rims (bright) and floors (dark) via top-hat + bottom-hat.

    Top-hat = original - opening  → highlights bright features (rims)
    Bottom-hat = closing - original → highlights dark features (floors)
    Combined = top-hat + bottom-hat → highlights circular edges
    """
    params = params or DetectionParams()

    k = params.tophat_ksize | 1  # ensure odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    bottomhat = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bottomhat = cv2.subtract(bottomhat, gray)

    # Normalise combined to 0-255
    combined = cv2.add(tophat, bottomhat)
    if combined.max() > 0:
        combined = (combined.astype(np.float32) / combined.max() * 255).astype(np.uint8)

    return combined


def detect_circles(
    edge_image: np.ndarray,
    params: DetectionParams | None = None,
) -> list[tuple[int, int, int]]:
    """Run HoughCircles on an edge image. Returns list of (cx, cy, radius)."""
    params = params or DetectionParams()

    circles = cv2.HoughCircles(
        edge_image,
        cv2.HOUGH_GRADIENT,
        dp=params.dp,
        minDist=params.min_dist,
        param1=params.hough_param1,
        param2=params.hough_param2,
        minRadius=params.min_radius,
        maxRadius=params.max_radius,
    )

    if circles is None:
        return []

    return [(int(round(c[0])), int(round(c[1])), int(round(c[2]))) for c in circles[0]]


def validate_detection(
    gray: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    params: DetectionParams | None = None,
) -> tuple[float, float, float]:
    """Validate a candidate circle. Returns (confidence, circularity, mean_intensity)."""
    params = params or DetectionParams()
    h, w = gray.shape[:2]

    # Skip circles that extend beyond image boundaries
    if cx - radius < 0 or cy - radius < 0 or cx + radius >= w or cy + radius >= h:
        return 0.0, 0.0, 0.0

    # Create mask for the circle
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    # Mean intensity inside the circle
    mean_val = float(cv2.mean(gray, mask=mask)[0])

    # Edge density inside the circle
    edge_mask = cv2.Canny(gray, params.canny_low, params.canny_high)
    edge_inside = cv2.bitwise_and(edge_mask, edge_mask, mask=mask)
    n_circle = int(mask.sum() // 255)
    n_edge = int(edge_inside.sum() // 255)
    edge_density = n_edge / max(n_circle, 1)

    # Circularity from contour fitting
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, mean_val

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0, 0.0, mean_val

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)

    # Ellipse ratio (major/minor axis)
    ellipse_ratio = 1.0
    if len(cnt) >= 5:
        (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
        if ma > 0:
            ellipse_ratio = max(MA, ma) / max(min(MA, ma), 1e-6)

    # Composite confidence score
    conf = 0.0
    if circularity >= params.min_circularity and ellipse_ratio <= params.max_ellipse_ratio:
        conf = circularity * min(edge_density / max(params.edge_fill_min, 0.01), 1.0)
        conf = min(conf, 1.0)

    return conf, circularity, mean_val


def detect_craters(
    gray: np.ndarray,
    params: DetectionParams | None = None,
) -> list[CraterDetection]:
    """Full crater detection pipeline.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel 8-bit image (grayscale lunar surface).
    params : DetectionParams, optional
        Tunable detection parameters.

    Returns
    -------
    list[CraterDetection]
        Detected craters sorted by radius (largest first).
    """
    params = params or DetectionParams()
    logger.info("Detecting craters on %dx%d image", gray.shape[1], gray.shape[0])

    # 1. Preprocess
    enhanced = preprocess(gray, params)

    # 2. Morphological enhancement
    if params.use_morphology:
        morph = morphological_enhance(enhanced, params)
        # Blend: original edges + morphological edges
        edges_orig = cv2.Canny(enhanced, params.canny_low, params.canny_high)
        edges_morph = cv2.Canny(morph, params.canny_low, params.canny_high)
        edge_image = cv2.bitwise_or(edges_orig, edges_morph)
    else:
        edge_image = cv2.Canny(enhanced, params.canny_low, params.canny_high)

    # 3. Detect circles
    candidates = detect_circles(edge_image, params)
    logger.info("HoughCircles found %d candidates", len(candidates))

    # 4. Validate and score
    detections: list[CraterDetection] = []
    for cx, cy, r in candidates:
        conf, circ, mean_val = validate_detection(gray, cx, cy, r, params)
        if conf > 0.05:  # keep even low-confidence detections for analysis
            detections.append(CraterDetection(
                cx=float(cx), cy=float(cy), radius=float(r),
                confidence=conf, circularity=circ, mean_intensity=mean_val,
            ))

    # Sort by radius descending
    detections.sort(key=lambda d: d.radius, reverse=True)
    logger.info("Validated %d detections (of %d candidates)", len(detections), len(candidates))
    return detections


# ---------------------------------------------------------------------------
# Size-dependent parameter presets
# ---------------------------------------------------------------------------

def params_for_resolution(resolution_m: float) -> DetectionParams:
    """Return reasonable DetectionParams calibrated to image resolution.

    At 0.2 m/px (OHRC): craters ≥1 km → radius ≥ 2500 px → huge, use relaxed params
    At 6.13 m/px (TMC-2): craters ≥1 km → radius ≥ 82 px → moderate
    At 100 m/px (WAC mosaic): craters ≥1 km → radius ~5 px → too small for HoughCircles
    """
    if resolution_m <= 1.0:
        # Ultra-high resolution (OHRC): large craters, relaxed HoughCircles
        return DetectionParams(
            min_radius=50,
            max_radius=5000,
            dp=1.5,
            min_dist=100,
            hough_param2=25,
            tophat_ksize=61,
            bottomhat_ksize=61,
        )
    elif resolution_m <= 10.0:
        # Medium resolution (TMC-2): moderate crater sizes
        return DetectionParams(
            min_radius=15,
            max_radius=500,
            dp=1.2,
            min_dist=30,
            hough_param2=35,
            tophat_ksize=31,
            bottomhat_ksize=31,
        )
    else:
        # Low resolution (WAC mosaics): small craters
        return DetectionParams(
            min_radius=3,
            max_radius=50,
            dp=1.0,
            min_dist=10,
            hough_param2=15,
            tophat_ksize=15,
            bottomhat_ksize=15,
        )


# ---------------------------------------------------------------------------
# Multi-scale detection for very large images
# ---------------------------------------------------------------------------

def detect_multiscale(
    gray: np.ndarray,
    *,
    resolution_m: float,
    tile_size: int = 2048,
    overlap_px: int = 256,
) -> list[CraterDetection]:
    """Detect craters across tiles of a large image, merging overlapping results.

    Useful for OHRC-scale images (101074 x 12000) where HoughCircles
    parameter tuning is difficult at full resolution.
    """
    h, w = gray.shape[:2]
    params = params_for_resolution(resolution_m)

    if h <= tile_size and w <= tile_size:
        return detect_craters(gray, params)

    all_detections: list[CraterDetection] = []
    for y0 in range(0, h, tile_size - overlap_px):
        for x0 in range(0, w, tile_size - overlap_px):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            tile = gray[y0:y1, x0:x1]

            tile_dets = detect_craters(tile, params)

            # Offset coordinates back to full image
            for d in tile_dets:
                all_detections.append(CraterDetection(
                    cx=d.cx + x0, cy=d.cy + y0,
                    radius=d.radius, confidence=d.confidence,
                    votes=d.votes, circularity=d.circularity,
                    mean_intensity=d.mean_intensity,
                ))

    # Merge overlapping detections (non-max suppression by centre distance)
    all_detections.sort(key=lambda d: d.confidence, reverse=True)
    merged: list[CraterDetection] = []
    for d in all_detections:
        is_dup = False
        for m in merged:
            dist = math.hypot(d.cx - m.cx, d.cy - m.cy)
            if dist < max(d.radius, m.radius) * 0.5:
                is_dup = True
                break
        if not is_dup:
            merged.append(d)

    logger.info("Multi-scale: %d tile detections -> %d after merge", len(all_detections), len(merged))
    return merged
