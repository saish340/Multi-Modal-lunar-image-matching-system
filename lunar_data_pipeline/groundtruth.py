"""Ground-truth pixel correspondence between overlapping products (Phase 0).

Given two products whose footprints share lunar lat/lon space, this module
builds a coordinate transform mapping any pixel of image A onto its expected
pixel in image B:

    pixel(A) --H_A--> (lon, lat) --H_B^-1--> pixel(B)

Each homography is fitted from exactly four control-point correspondences:
the label's footprint corners mapped to the image corners. Pixel coordinates
are zero-based ``(x, y) = (sample, line)``, i.e. upper-left pixel centre is
``(0, 0)`` and lower-right is ``(samples - 1, lines - 1)``.

KNOWN LIMITATIONS -- do not treat outputs as exact:

1. **Pushbroom geometry is not projective.** OHRC and TMC-2 are line-scan
   pushbroom sensors; every image line has its own exterior orientation, so
   no single homography is geometrically exact. A corner-fitted homography is
   a reasonable approximation only where interior distortion across the
   footprint is small relative to the matching tolerance. Expect it to hold
   decently over small footprints (OHRC ~3 km swath) and degrade over long
   TMC-2 strips that span tens of degrees of latitude.
2. **Planar lon/lat assumption.** Transforms operate directly on selenographic
   degrees as if they were planar. Distortion grows toward the poles; near-pole
   products (e.g. |lat| > 80 deg) will show larger errors.
3. **Corner geolocation error.** The corner coordinates themselves come from
   ISRO's system geometry and carry their own uncertainty (note the difference
   between System_Level_Coordinates and Refined_Corner_Coordinates labels).
4. **Refinement path exists.** Each product ships a per-pixel geolocation CSV
   (``geometry/*_g_grd_*.csv``, columns Longitude,Latitude,Pixel,Scan). A later
   phase can interpolate that grid per line for far more accurate ground truth;
   Phase 0 intentionally uses the cheap corner-based approximation.

Usage:
    >>> transform = build_pixel_transform(product_a, product_b)
    >>> xy_in_b = transform.project((x_a, y_a))
    >>> # or one-shot:
    >>> xy_in_b = project_point((x_a, y_a), product_a, product_b)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # package-style import when used as a module...
    from .pds4_parser import CORNER_ORDER, LunarProduct
except ImportError:  # ...and flat import when pipeline.py is run directly
    from pds4_parser import CORNER_ORDER, LunarProduct  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

PixelPoint = tuple[float, float]  # (x, y) = (sample, line), zero-based
GeoPoint = tuple[float, float]  # (lon, lat)


def _solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve a 3x3 planar homography mapping *src* (N,2) onto *dst* (N,2).

    Normalised direct linear transform; requires at least 4 non-degenerate
    point correspondences.
    """
    if src.shape != dst.shape or src.shape[0] < 4:
        raise ValueError("homography needs >= 4 corresponding point pairs")

    def _normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centroid = points.mean(axis=0)
        shifted = points - centroid
        mean_dist = np.mean(np.hypot(shifted[:, 0], shifted[:, 1]))
        scale = np.sqrt(2.0) / mean_dist if mean_dist > 0 else 1.0
        transform = np.array(
            [[scale, 0.0, -scale * centroid[0]],
             [0.0, scale, -scale * centroid[1]],
             [0.0, 0.0, 1.0]]
        )
        homogeneous = np.column_stack([points, np.ones(len(points))])
        return (transform @ homogeneous.T).T[:, :2], transform

    src_n, t_src = _normalise(src)
    dst_n, t_dst = _normalise(dst)

    rows = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
    system = np.asarray(rows)

    try:
        _, _, vt = np.linalg.svd(system)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"homography SVD failed: {exc}") from exc

    h_normalized = vt[-1].reshape(3, 3)
    h = np.linalg.inv(t_dst) @ h_normalized @ t_src
    return h / h[2, 2]


def _apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply homography *h* to an (N,2) array (or (2,) single point)."""
    single = points.ndim == 1
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    projected = (h @ homogeneous.T).T
    projected /= projected[:, 2:3]
    result = projected[:, :2]
    return result[0] if single else result


def _corner_pixel_targets(product: LunarProduct) -> np.ndarray:
    """Image-corner pixel coords matching CORNER_ORDER (UL, UR, LR, LL)."""
    w = max(product.samples - 1, 0)
    hgt = max(product.lines - 1, 0)
    return np.array(
        [
            [0.0, 0.0],  # upper_left
            [float(w), 0.0],  # upper_right
            [float(w), float(hgt)],  # lower_right
            [0.0, float(hgt)],  # lower_left
        ]
    )


def _corner_geo_points(product: LunarProduct) -> np.ndarray:
    """(lon, lat) array in CORNER_ORDER derived from the product footprint."""
    return np.array([(lon, lat) for lat, lon in product.footprint], dtype=float)


@dataclass
class PixelTransform:
    """Reusable pixel(A) -> pixel(B) correspondence map."""

    product_a_id: str
    product_b_id: str
    h_pixel_to_geo_a: np.ndarray  # pixel(A) -> (lon, lat)
    h_geo_to_pixel_b: np.ndarray  # (lon, lat) -> pixel(B)

    def project(self, point_in_a: PixelPoint) -> PixelPoint:
        """Map one pixel of image A to its expected pixel in image B."""
        geo = _apply_homography(self.h_pixel_to_geo_a, np.array(point_in_a))
        out = _apply_homography(self.h_geo_to_pixel_b, geo)
        return float(out[0]), float(out[1])

    def project_many(self, points_in_a: np.ndarray) -> np.ndarray:
        """Vectorised variant of :meth:`project` for an (N,2) array."""
        geo = _apply_homography(self.h_pixel_to_geo_a, points_in_a)
        return _apply_homography(self.h_geo_to_pixel_b, geo)


def build_pixel_transform(product_a: LunarProduct, product_b: LunarProduct) -> PixelTransform:
    """Build the ground-truth transform between two overlapping products."""
    for product in (product_a, product_b):
        if len(product.footprint) != len(CORNER_ORDER):
            raise ValueError(
                f"product {product.product_id} has no complete 4-corner "
                "footprint; cannot build ground-truth transform"
            )

    h_pixel_to_geo_a = _solve_homography(
        _corner_pixel_targets(product_a), _corner_geo_points(product_a)
    )
    h_geo_to_pixel_b = _solve_homography(
        _corner_geo_points(product_b), _corner_pixel_targets(product_b)
    )

    logger.debug(
        "Built transform %s -> %s", product_a.product_id, product_b.product_id
    )
    return PixelTransform(
        product_a_id=product_a.product_id,
        product_b_id=product_b.product_id,
        h_pixel_to_geo_a=h_pixel_to_geo_a,
        h_geo_to_pixel_b=h_geo_to_pixel_b,
    )


def project_point(
    point_in_a: PixelPoint, product_a: LunarProduct, product_b: LunarProduct
) -> PixelPoint:
    """Convenience wrapper: map a single pixel via freshly built transforms.

    For batch evaluation, build once with :func:`build_pixel_transform` and
    reuse :meth:`PixelTransform.project_many` instead.
    """
    return build_pixel_transform(product_a, product_b).project(point_in_a)


def transform_to_manifest_dict(transform: PixelTransform) -> dict:
    """Serialise a transform's matrices for inline storage in the manifest.

    Matrix entries are written at full double precision (json emits an
    exact round-trip repr for floats). Do NOT decimal-round these values:
    the perspective-row coefficients can be O(1e-8), and coarse rounding
    there degrades through the homogeneous division into visible pixel
    error on long image strips.
    """
    return {
        "method": "corner_homography",
        "pixel_convention": "(x=sample, y=line), zero-based, UL=(0,0)",
        "geo_convention": "(lon, lat) selenographic degrees",
        "product_a_id": transform.product_a_id,
        "product_b_id": transform.product_b_id,
        "h_pixel_to_geo_a": [
            [float(v) for v in row] for row in transform.h_pixel_to_geo_a
        ],
        "h_geo_to_pixel_b": [
            [float(v) for v in row] for row in transform.h_geo_to_pixel_b
        ],
    }


def transform_from_manifest_dict(data: dict) -> PixelTransform:
    """Reconstruct a :class:`PixelTransform` from its manifest representation."""
    return PixelTransform(
        product_a_id=data["product_a_id"],
        product_b_id=data["product_b_id"],
        h_pixel_to_geo_a=np.asarray(data["h_pixel_to_geo_a"], dtype=float),
        h_geo_to_pixel_b=np.asarray(data["h_geo_to_pixel_b"], dtype=float),
    )


def validate_transform_pairing(pair_dir_hint: Path | None = None) -> None:
    """Placeholder hook for future per-pixel CSV refinement checks."""
    raise NotImplementedError("CSV-grid refinement is planned for a later phase")
