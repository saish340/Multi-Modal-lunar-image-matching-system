"""Per-pixel geolocation refinement from ISDA ``*_g_grd_*.csv`` tables.

Each Chandrayaan-2 product ships a geometry CSV (defined by the ``*_g_grd_*``
PDS4 label) with columns ``Longitude,Latitude,Pixel,Scan`` giving ISRO's own
selenographic coordinates for a subsampled grid of image pixels.

Why this matters: the Phase 0 corner-based homography is a single projective
approximation of pushbroom geometry. On a 223088-line TMC-2 strip it misplaces
content by thousands of pixels along-track (observed: ~10,000 px / ~4.4% on
the Phase 1 pair). This module uses the CSV grid to:

1. locate the true image-pixel bounding box of a geographic region, and
2. fit a LOCAL homography (pixel,scan) -> (lon,lat) (and its inverse) over a
   restricted window, where the projective approximation is actually valid.

The fits reuse the normalised DLT solver from ``groundtruth.py``; with more
than four points it degrades gracefully into least-squares.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, point

try:  # package-style import when used as a module...
    from .groundtruth import _solve_homography
except ImportError:  # ...and flat import when scripts run directly
    from groundtruth import _solve_homography  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


@dataclass
class GeolocationGrid:
    pixel: np.ndarray  # sample index (x)
    scan: np.ndarray  # line index (y)
    lat: np.ndarray
    lon: np.ndarray

    def __len__(self) -> int:
        return len(self.lat)


def read_geolocation_csv(csv_path: str | Path) -> GeolocationGrid:
    """Parse a ``*_g_grd_*.csv`` (Longitude,Latitude,Pixel,Scan per row)."""
    csv_path = Path(csv_path)
    pixels, scans, lats, lons = [], [], [], []
    with csv_path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        header_l = [h.strip().lower() for h in header]
        expected = ["longitude", "latitude", "pixel", "scan"]
        if header_l[:2] != expected[:2]:
            logger.warning(
                "%s: unexpected header %s (expected %s); parsing by position anyway",
                csv_path.name,
                header,
                expected,
            )
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                lon, lat, pix, scan = (float(v) for v in row[:4])
            except ValueError:
                continue
            lons.append(lon)
            lats.append(lat)
            pixels.append(pix)
            scans.append(scan)

    grid = GeolocationGrid(
        pixel=np.asarray(pixels),
        scan=np.asarray(scans),
        lat=np.asarray(lats),
        lon=np.asarray(lons),
    )
    logger.info("Parsed %s: %d geolocation points", csv_path.name, len(grid))
    return grid


def fit_homography(
    src_points: np.ndarray, dst_points: np.ndarray
) -> np.ndarray:
    """Least-squares homography src(N,2) -> dst(N,2) via the shared DLT."""
    return _solve_homography(np.asarray(src_points, float), np.asarray(dst_points, float))


def fit_local_transforms(
    grid: GeolocationGrid,
    scan_range: tuple[int, int],
    pixel_range: tuple[int, int],
    min_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit local (pixel,scan)<->(lon,lat) homographies over a window.

    Returns ``(H_px_to_geo, H_geo_to_px, n_points)``. Raises ValueError if the
    window contains too few grid points for a stable fit.
    """
    s0, s1 = scan_range
    c0, c1 = pixel_range
    mask = (
        (grid.scan >= s0) & (grid.scan <= s1)
        & (grid.pixel >= c0) & (grid.pixel <= c1)
    )
    n = int(mask.sum())
    if n < min_points:
        raise ValueError(
            f"only {n} geolocation points in window scan{scan_range} "
            f"pixel{pixel_range}; need >= {min_points}"
        )

    px = np.column_stack([grid.pixel[mask], grid.scan[mask]])
    geo = np.column_stack([grid.lon[mask], grid.lat[mask]])

    h_px_to_geo = fit_homography(px, geo)
    h_geo_to_px = fit_homography(geo, px)

    # Residual check: the fit should reproduce the grid closely for a local window.
    fwd = _apply(h_px_to_geo, px)
    residuals = np.hypot(*(fwd - geo).T)
    logger.info(
        "Local fit over %d points: geo residual px mean=%.4f max=%.4f (deg)",
        n,
        residuals.mean(),
        residuals.max(),
    )
    return h_px_to_geo, h_geo_to_px, n


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    out = (h @ homogeneous.T).T
    return out[:, :2] / out[:, 2:3]


def locate_region(
    grid: GeolocationGrid,
    region_lonlat_polygon: Polygon,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    """Bounding (scan_range, pixel_range) of grid points inside a lon/lat polygon.

    The polygon is in (lon, lat) coordinates, matching shapely's x/y convention
    used by ``overlap_finder.footprint_polygon``.
    """
    inside = np.array(
        [
            region_lonlat_polygon.contains(point.Point(lon, lat))
            for lon, lat in zip(grid.lon, grid.lat)
        ]
    )
    n = int(inside.sum())
    if n == 0:
        raise ValueError("no geolocation points fall inside the requested region")

    scan_range = (int(grid.scan[inside].min()), int(grid.scan[inside].max()))
    pixel_range = (int(grid.pixel[inside].min()), int(grid.pixel[inside].max()))
    logger.info(
        "Region located: %d grid pts, scan %s, pixel %s",
        n,
        scan_range,
        pixel_range,
    )
    return scan_range, pixel_range, n
