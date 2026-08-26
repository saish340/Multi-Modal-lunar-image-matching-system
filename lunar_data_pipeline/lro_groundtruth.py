"""Lunar crater ground-truth loading from external catalogs.

Supports multiple crater database formats:
  - Robbins et al. (2018): CSV with columns ``Longitude, Latitude, Diameter_km``
  - LU1319373 Wang & Wu (2021): columns ``Longitude, Latitude, Diameter``
  - IAU Gazetteer: columns ``Feature Name, Center Latitude, Center Longitude, Diameter``
  - Generic: any CSV with recognizable lat/lon/diameter columns

When no external catalog is available, a synthetic generator produces
realistic random craters for testing the detection pipeline.

Usage:
    >>> from lro_groundtruth import load_crater_database, filter_by_region
    >>> craters = load_crater_database("path/to/robbins.csv")
    >>> region_craters = filter_by_region(craters, lon_min=25.0, lon_max=25.5,
    ...                                   lat_min=-14.0, lat_max=-12.8)
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from .groundtruth import _apply_homography, _solve_homography
    from .pds4_parser import LunarProduct
except ImportError:
    from groundtruth import _apply_homography, _solve_homography  # type: ignore[no-redef]
    from pds4_parser import LunarProduct  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


@dataclass
class CraterRecord:
    """A single crater from an external catalog."""

    lat: float  # centre latitude (degrees, planetocentric)
    lon: float  # centre longitude (degrees, East-positive)
    diameter_km: float  # rim-to-rim diameter in kilometres
    depth_km: float | None = None  # depth in km (if available)
    name: str = ""  # IAU name (if available)
    row_id: int = -1  # source row index

    def radius_px_at(self, resolution_m: float) -> float:
        """Approximate radius in pixels at given resolution (metres/pixel)."""
        return (self.diameter_km * 1000.0) / (2.0 * resolution_m)


# ---------------------------------------------------------------------------
# Database loaders
# ---------------------------------------------------------------------------

# Column name variants for auto-detection
_LAT_NAMES = {"latitude", "lat", "center_lat", "center latitude", "clat", "lat_deg"}
_LON_NAMES = {"longitude", "lon", "center_lon", "center longitude", "clon", "lon_deg", "long"}
_DIA_NAMES = {"diameter", "diameter_km", "diam_km", "d_km", "diameter (km)", "diameter_km", "d"}
_DEP_NAMES = {"depth", "depth_km", "d_depth_km", "depth (km)"}
_NAME_NAMES = {"name", "feature_name", "clean_feature_name", "crater_name"}


def _normalise_col(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")


def _detect_columns(header: list[str]) -> dict[str, int | None]:
    """Auto-detect lat/lon/diameter column indices from header row."""
    normed = [_normalise_col(h) for h in header]
    result: dict[str, int | None] = {"lat": None, "lon": None, "diameter": None, "depth": None, "name": None}
    for i, col in enumerate(normed):
        if col in _LAT_NAMES and result["lat"] is None:
            result["lat"] = i
        elif col in _LON_NAMES and result["lon"] is None:
            result["lon"] = i
        elif col in _DIA_NAMES and result["diameter"] is None:
            result["diameter"] = i
        elif col in _DEP_NAMES and result["depth"] is None:
            result["depth"] = i
        elif col in _NAME_NAMES and result["name"] is None:
            result["name"] = i
    return result


def load_crater_database(
    csv_path: str | Path,
    *,
    min_diameter_km: float = 0.0,
    max_diameter_km: float = float("inf"),
) -> list[CraterRecord]:
    """Load a crater database from a CSV file.

    Auto-detects column layout from header. Accepts Robbins, LU1319373,
    IAU Gazetteer, or any CSV with lat/lon/diameter columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Crater database not found: {csv_path}")

    craters: list[CraterRecord] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = _detect_columns(header)

        if cols["lat"] is None or cols["lon"] is None or cols["diameter"] is None:
            raise ValueError(
                f"Cannot auto-detect lat/lon/diameter columns from header: {header}. "
                f"Expected columns matching: lat={_LAT_NAMES}, lon={_LON_NAMES}, diam={_DIA_NAMES}"
            )

        logger.info(
            "Detected columns: lat=%d, lon=%d, diameter=%d, depth=%s, name=%s",
            cols["lat"], cols["lon"], cols["diameter"],
            cols["depth"], cols["name"],
        )

        for row_idx, row in enumerate(reader):
            try:
                lat = float(row[cols["lat"]])
                lon = float(row[cols["lon"]])
                diam = float(row[cols["diameter"]])
            except (ValueError, IndexError):
                continue

            if not (min_diameter_km <= diam <= max_diameter_km):
                continue

            depth = None
            if cols["depth"] is not None:
                try:
                    depth = float(row[cols["depth"]])
                except (ValueError, IndexError):
                    pass

            name = ""
            if cols["name"] is not None:
                try:
                    name = row[cols["name"]].strip()
                except IndexError:
                    pass

            craters.append(CraterRecord(
                lat=lat, lon=lon, diameter_km=diam,
                depth_km=depth, name=name, row_id=row_idx,
            ))

    logger.info("Loaded %d craters from %s (min_d=%.1f km, max_d=%.1f km)", len(craters), csv_path.name, min_diameter_km, max_diameter_km)
    return craters


# ---------------------------------------------------------------------------
# Region filtering
# ---------------------------------------------------------------------------

def filter_by_region(
    craters: list[CraterRecord],
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> list[CraterRecord]:
    """Return craters whose centres fall within a bounding box."""
    result = [
        c for c in craters
        if lon_min <= c.lon <= lon_max and lat_min <= c.lat <= lat_max
    ]
    logger.info(
        "Filtered %d -> %d craters in lon[%.2f,%.2f] lat[%.2f,%.2f]",
        len(craters), len(result), lon_min, lon_max, lat_min, lat_max,
    )
    return result


# ---------------------------------------------------------------------------
# Coordinate conversion: (lon, lat) -> pixel
# ---------------------------------------------------------------------------

def craters_to_pixels(
    craters: list[CraterRecord],
    product: LunarProduct,
) -> list[tuple[float, float, float]]:
    """Convert crater (lon, lat) centres to pixel (x, y) and radius_px for an image product.

    Uses the corner-based homography from groundtruth.py.
    Returns list of (x_px, y_px, radius_px).
    """
    from .pds4_parser import CORNER_ORDER

    pixel_targets = np.array([
        [0.0, 0.0],
        [float(product.samples - 1), 0.0],
        [float(product.samples - 1), float(product.lines - 1)],
        [0.0, float(product.lines - 1)],
    ])
    geo_points = np.array([(lon, lat) for lat, lon in product.footprint], dtype=float)

    h_geo_to_px = _solve_homography(geo_points, pixel_targets)

    results = []
    for c in craters:
        geo_pt = np.array([[c.lon, c.lat]])
        homogeneous = np.column_stack([geo_pt, np.ones((1, 1))])
        projected = (h_geo_to_px @ homogeneous.T).T
        projected /= projected[:, 2:3]
        x_px, y_px = projected[0, 0], projected[0, 1]
        radius_px = (c.diameter_km * 1000.0) / (2.0 * product.pixel_resolution_m)
        results.append((float(x_px), float(y_px), float(radius_px)))

    return results


# ---------------------------------------------------------------------------
# Synthetic crater generator (fallback)
# ---------------------------------------------------------------------------

def generate_synthetic_craters(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    count: int = 20,
    *,
    min_diameter_km: float = 1.0,
    max_diameter_km: float = 15.0,
    seed: int = 42,
) -> list[CraterRecord]:
    """Generate synthetic craters for pipeline testing when no catalog is available.

    Uses a power-law size distribution (N(>D) ~ D^-1.8) typical of the
    lunar cratering record, scaled to the requested count and region.
    """
    rng = np.random.default_rng(seed)

    # Power-law sampling: P(d) ~ d^(-2.8) (pdf for cumulative D^-1.8)
    u = rng.uniform(0, 1, count)
    diameters = min_diameter_km * (1.0 - u) ** (-1.0 / 1.8)
    diameters = np.clip(diameters, min_diameter_km, max_diameter_km)
    diameters = np.sort(diameters)[::-1]  # largest first

    lats = rng.uniform(lat_min, lat_max, count)
    lons = rng.uniform(lon_min, lon_max, count)

    craters = [
        CraterRecord(
            lat=float(lats[i]), lon=float(lons[i]),
            diameter_km=float(diameters[i]),
            name=f"SYN-{i:03d}",
            row_id=i,
        )
        for i in range(count)
    ]
    logger.info("Generated %d synthetic craters in lon[%.2f,%.2f] lat[%.2f,%.2f]", count, lon_min, lon_max, lat_min, lat_max)
    return craters


# ---------------------------------------------------------------------------
# Product bounds helper
# ---------------------------------------------------------------------------

def product_bounds(product: LunarProduct) -> tuple[float, float, float, float]:
    """Return (lon_min, lon_max, lat_min, lat_max) from footprint corners."""
    lons = [lon for lat, lon in product.footprint]
    lats = [lat for lat, lon in product.footprint]
    return min(lons), max(lons), min(lats), max(lats)


def overlap_bounds(product_a: LunarProduct, product_b: LunarProduct) -> tuple[float, float, float, float]:
    """Return the intersection bounding box of two products' footprints."""
    lon_min_a, lon_max_a, lat_min_a, lat_max_a = product_bounds(product_a)
    lon_min_b, lon_max_b, lat_min_b, lat_max_b = product_bounds(product_b)
    return (
        max(lon_min_a, lon_min_b),
        min(lon_max_a, lon_max_b),
        max(lat_min_a, lat_min_b),
        min(lat_max_a, lat_max_b),
    )
