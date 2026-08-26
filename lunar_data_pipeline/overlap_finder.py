"""Footprint overlap detection between parsed lunar products.

Builds shapely polygons from each product's footprint corners (treated as
planar lon/lat degree coordinates) and reports every OHRC x TMC2 pair whose
footprints intersect, ranked by usefulness for image matching.

Area caveat: polygon areas are computed in raw lon/lat degrees, which
exaggerates area near the poles (meridians converge). This is fine for
*detecting* overlap and for coarse ranking, but percentage values for polar
products should be treated as approximate.
"""

from __future__ import annotations

import logging
from typing import Iterable, NamedTuple

from shapely.geometry import Polygon

try:  # package-style import when used as a module...
    from .pds4_parser import LunarProduct
except ImportError:  # ...and flat import when pipeline.py is run directly
    from pds4_parser import LunarProduct  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class OverlapPair(NamedTuple):
    """One confirmed OHRC/TMC2 overlap.

    Unpacking as a plain tuple yields
    ``(ohrc_product, tmc_product, overlap_polygon, overlap_percentage)``.

    ``overlap_percentage`` is the fraction of the *smaller* footprint covered
    by the intersection, expressed as 0-100. Because an OHRC scene (~3 km)
    is tiny compared to a TMC-2 strip (tens of km), this is effectively
    "% of the OHRC footprint covered by TMC-2", which is the most useful
    ranking signal for matching experiments.
    """

    ohrc: LunarProduct
    tmc2: LunarProduct
    overlap_polygon: Polygon
    overlap_percentage: float
    pct_of_ohrc: float
    pct_of_tmc2: float


def footprint_polygon(product: LunarProduct) -> Polygon | None:
    """Build a valid shapely Polygon from a product's footprint corners.

    Returns ``None`` if the product has no usable footprint or the geometry
    cannot be repaired into a valid polygon.
    """
    if len(product.footprint) < 3:
        return None
    # Shapely wants (x, y); we use (lon, lat).
    points = [(lon, lat) for lat, lon in product.footprint]
    polygon = Polygon(points)
    if not polygon.is_valid:
        logger.warning(
            "Invalid footprint polygon for %s; attempting buffer(0) repair",
            product.product_id,
        )
        repaired = polygon.buffer(0)
        if repaired.is_empty or not isinstance(repaired, Polygon):
            logger.error(
                "Could not repair footprint polygon for %s", product.product_id
            )
            return None
        polygon = repaired
    return polygon


def compute_overlap(
    poly_a: Polygon, poly_b: Polygon
) -> tuple[float, float, float] | None:
    """Return ``(pct_of_a, pct_of_b, pct_of_smaller)`` for two polygons.

    Returns ``None`` if the polygons do not intersect with positive area.
    """
    intersection = poly_a.intersection(poly_b)
    if intersection.is_empty or intersection.area <= 0.0:
        return None

    pct_of_a = 100.0 * intersection.area / poly_a.area if poly_a.area > 0 else 0.0
    pct_of_b = 100.0 * intersection.area / poly_b.area if poly_b.area > 0 else 0.0
    denominators = [d for d in (poly_a.area, poly_b.area) if d > 0]
    pct_smaller = (
        100.0 * intersection.area / min(denominators) if denominators else 0.0
    )
    return pct_of_a, pct_of_b, pct_smaller


def find_overlaps(
    ohrc_products: Iterable[LunarProduct],
    tmc_products: Iterable[LunarProduct],
    min_overlap_pct: float = 0.0,
) -> list[OverlapPair]:
    """Find all OHRC x TMC2 pairs with intersecting footprints.

    Results are sorted by ``overlap_percentage`` descending (largest joint
    coverage first). Pairs whose smaller-footprint coverage is below
    ``min_overlap_pct`` are filtered out.
    """
    ohrc_list = list(ohrc_products)
    tmc_list = list(tmc_products)

    ohrc_polys = {p.product_id: footprint_polygon(p) for p in ohrc_list}
    tmc_polys = {p.product_id: footprint_polygon(p) for p in tmc_list}

    pairs: list[OverlapPair] = []
    for ohrc in ohrc_list:
        poly_ohrc = ohrc_polys[ohrc.product_id]
        if poly_ohrc is None:
            logger.warning(
                "OHRC %s has no usable footprint; excluded from pairing",
                ohrc.product_id,
            )
            continue
        for tmc in tmc_list:
            poly_tmc = tmc_polys[tmc.product_id]
            if poly_tmc is None:
                logger.warning(
                    "TMC2 %s has no usable footprint; excluded from pairing",
                    tmc.product_id,
                )
                continue

            result = compute_overlap(poly_ohrc, poly_tmc)
            if result is None:
                continue

            pct_ohrc, pct_tmc, pct_smaller = result
            if pct_smaller < min_overlap_pct:
                logger.debug(
                    "Pair %s x %s below threshold (%.3f%% < %.3f%%)",
                    ohrc.product_id,
                    tmc.product_id,
                    pct_smaller,
                    min_overlap_pct,
                )
                continue

            intersection = poly_ohrc.intersection(poly_tmc)
            pairs.append(
                OverlapPair(
                    ohrc=ohrc,
                    tmc2=tmc,
                    overlap_polygon=intersection,
                    overlap_percentage=pct_smaller,
                    pct_of_ohrc=pct_ohrc,
                    pct_of_tmc2=pct_tmc,
                )
            )
            logger.info(
                "Overlap found: %s x %s (%.3f%% of smaller footprint)",
                ohrc.product_id,
                tmc.product_id,
                pct_smaller,
            )

    pairs.sort(key=lambda pair: pair.overlap_percentage, reverse=True)
    return pairs
