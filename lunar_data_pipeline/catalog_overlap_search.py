"""Search the FULL PRADAN shapefile catalogs for the best OHRC x TMC-2 pairs.

Strategy
--------
1. Load every calibrated catalog entry via ``shapefile_loader`` into the same
   ``LunarProduct`` structure Phase 0 uses.
2. Cheap bbox-level pre-filter with a shapely ``STRtree`` spatial index over
   TMC-2 footprints: each OHRC footprint queries the tree for bounding-box
   candidates instead of testing all ~10k TMC-2 strips individually.
3. Precise polygon intersection + the shared overlap metric from
   ``overlap_finder.compute_overlap`` run only on surviving candidates.
4. Rank: non-polar first (|centroid lat| <= threshold), then by overlap
   percentage of the smaller footprint descending, with a mild preference for
   TMC-2 *nadir* view variants (``_ncn_``) since OHRC is nadir-viewing.
5. Write the top N to ``candidate_pairs.json`` with exact PRADAN download
   filenames, and print a human-readable summary regardless of log level.

Note on shadow avoidance: the catalogs' INC_ANGLE / PHA_ANGLE columns were
observed to be populated with zeros, so terminator/day-night filtering is NOT
possible from these shapefiles; the JSON records the angles when non-zero so
you can spot usable ones manually.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from shapely import STRtree

try:  # package-style import when used as a module...
    from .overlap_finder import OverlapPair, compute_overlap, footprint_polygon
    from .pds4_parser import LunarProduct
    from .shapefile_loader import (
        discover_catalog_files,
        load_catalog,
    )
except ImportError:  # ...and flat import when scripts run directly
    from overlap_finder import OverlapPair, compute_overlap, footprint_polygon  # type: ignore[no-redef]
    from pds4_parser import LunarProduct  # type: ignore[no-redef]
    from shapefile_loader import discover_catalog_files, load_catalog  # type: ignore[no-redef]

logger = logging.getLogger("lunar_data_pipeline.catalog_search")

_TMC_NADIR_PATTERN = re.compile(r"_ncn_")


@dataclass
class CatalogSearchStats:
    ohrc_count: int = 0
    tmc_count: int = 0
    combos_checked: int = 0
    prefilter_candidates: int = 0
    overlapping_pairs: int = 0


@dataclass
class CatalogSearchResult:
    pairs: list[OverlapPair] = field(default_factory=list)
    stats: CatalogSearchStats = field(default_factory=CatalogSearchStats)


def _valid_footprint_polygons(
    products: list[LunarProduct], instrument_label: str
) -> tuple[list[LunarProduct], list]:
    """Products and their valid polygons, kept aligned by index."""
    kept: list[LunarProduct] = []
    polygons: list = []
    for product in products:
        polygon = footprint_polygon(product)
        if polygon is None:
            logger.warning(
                "%s %s has no usable footprint; excluded",
                instrument_label,
                product.product_id,
            )
            continue
        kept.append(product)
        polygons.append(polygon)
    return kept, polygons


def search_overlaps_with_index(
    ohrc_products: list[LunarProduct],
    tmc_products: list[LunarProduct],
) -> CatalogSearchResult:
    """Full-catalog OHRC x TMC-2 overlap search using an STRtree pre-filter.

    Reuses ``compute_overlap`` (the exact Phase 0 metric) for precision; the
    STRtree only decides which pairs are worth evaluating.
    """
    result = CatalogSearchResult()
    result.stats.ohrc_count = len(ohrc_products)
    result.stats.tmc_count = len(tmc_products)

    ohrc_kept, ohrc_polys = _valid_footprint_polygons(ohrc_products, "OHRC")
    tmc_kept, tmc_polys = _valid_footprint_polygons(tmc_products, "TMC2")

    result.stats.combos_checked = len(ohrc_kept) * len(tmc_kept)
    if not tmc_polys or not ohrc_kept:
        logger.warning("Empty catalog side; nothing to search")
        return result

    tree = STRtree(tmc_polys)
    for ohrc_product, ohrc_poly in zip(ohrc_kept, ohrc_polys):
        candidate_indices = tree.query(ohrc_poly)
        for idx in candidate_indices:
            result.stats.prefilter_candidates += 1
            tmc_product = tmc_kept[int(idx)]
            tmc_poly = tmc_polys[int(idx)]

            metrics = compute_overlap(ohrc_poly, tmc_poly)
            if metrics is None:
                continue
            pct_ohrc, pct_tmc, pct_smaller = metrics

            intersection = ohrc_poly.intersection(tmc_poly)
            result.pairs.append(
                OverlapPair(
                    ohrc=ohrc_product,
                    tmc2=tmc_product,
                    overlap_polygon=intersection,
                    overlap_percentage=pct_smaller,
                    pct_of_ohrc=pct_ohrc,
                    pct_of_tmc2=pct_tmc,
                )
            )

    result.stats.overlapping_pairs = len(result.pairs)
    return result


def is_polar(pair: OverlapPair, max_abs_lat: float) -> bool:
    """A pair counts as polar if its overlap centroid leaves the mid-lat band."""
    return abs(pair.overlap_polygon.centroid.y) > max_abs_lat


def rank_pairs(pairs: list[OverlapPair], max_abs_lat: float) -> list[OverlapPair]:
    """Non-polar first, then largest overlap %, then TMC-2 nadir-view first."""

    def sort_key(pair: OverlapPair):
        polar_flag = 1 if is_polar(pair, max_abs_lat) else 0
        nadir_penalty = (
            0 if _TMC_NADIR_PATTERN.search(pair.tmc2.product_id) else 1
        )
        return (polar_flag, -pair.overlap_percentage, nadir_penalty)

    return sorted(pairs, key=sort_key)


def _angles_usable(pair: OverlapPair) -> bool:
    return any(
        isinstance(v, float) and v != 0.0
        for v in (
            pair.ohrc.incidence_angle_deg,
            pair.ohrc.emission_angle_deg,
            pair.ohrc.phase_angle_deg,
        )
    )


def pair_to_json(pair: OverlapPair, rank: int, max_abs_lat: float) -> dict:
    centroid = pair.overlap_polygon.centroid
    angles = {
        "incidence_angle_deg": pair.ohrc.incidence_angle_deg,
        "emission_angle_deg": pair.ohrc.emission_angle_deg,
        "phase_angle_deg": pair.ohrc.phase_angle_deg,
    }

    variant_match = re.search(r"_nc([a-z])_", pair.tmc2.product_id)
    variant = {"a": "aft", "f": "fore", "n": "nadir"}.get(
        variant_match.group(1) if variant_match else "", None
    )

    return {
        "rank": rank,
        "lat_band": "polar" if is_polar(pair, max_abs_lat) else "non-polar",
        "overlap": {
            "percentage_of_smaller_pct": round(pair.overlap_percentage, 6),
            "pct_of_ohrc_footprint": round(pair.pct_of_ohrc, 6),
            "pct_of_tmc2_footprint": round(pair.pct_of_tmc2, 6),
            "center_lonlat": [round(float(centroid.x), 6), round(float(centroid.y), 6)],
        },
        "ohrc": {
            "product_id": pair.ohrc.product_id,
            "download_filename": pair.ohrc.download_filename,
            "imaging_orbit_number": pair.ohrc.imaging_orbit_number,
            "start_time": pair.ohrc.start_time,
        },
        "tmc2": {
            "product_id": pair.tmc2.product_id,
            "download_filename": pair.tmc2.download_filename,
            "view_variant": variant,
            "imaging_orbit_number": pair.tmc2.imaging_orbit_number,
            "start_time": pair.tmc2.start_time,
        },
        "sun_geometry_angles_deg": angles if _angles_usable(pair) else None,
        "angles_note": (
            None
            if _angles_usable(pair)
            else "catalog angle columns are zero-filled; no shadow information available"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find the best downloadable OHRC + TMC-2 pairs across the full "
            "PRADAN shapefile catalogs before committing to large downloads."
        )
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("."),
        help="Folder containing the extracted PRADAN shapefile folders "
        "(default: current directory; searched recursively).",
    )
    parser.add_argument(
        "--ohrc-shapefile",
        action="append",
        default=None,
        metavar="PATH",
        help="Explicit OHRC .shp path(s); overrides auto-discovery.",
    )
    parser.add_argument(
        "--tmc-shapefile",
        action="append",
        default=None,
        metavar="PATH",
        help="Explicit TMC-2 .shp path(s); overrides auto-discovery.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("candidate_pairs.json"),
        help="Output JSON path (default: ./candidate_pairs.json).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of ranked candidates to write (default: 20).",
    )
    parser.add_argument(
        "--max-abs-lat",
        type=float,
        default=60.0,
        help="Overlaps centred beyond this absolute latitude are treated as "
        "polar and sorted after non-polar ones (default: 60 degrees).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    root: Path = args.catalog_root
    if not root.is_dir():
        logger.error("Catalog root does not exist: %s", root)
        return 2

    ohrc_files = (
        [Path(p) for p in args.ohrc_shapefile]
        if args.ohrc_shapefile
        else discover_catalog_files(root, "OHRC")
    )
    tmc_files = (
        [Path(p) for p in args.tmc_shapefile]
        if args.tmc_shapefile
        else discover_catalog_files(root, "TMC2")
    )
    if not ohrc_files or not tmc_files:
        logger.error(
            "Could not locate both catalogs under %s (OHRC files: %d, TMC2 files: %d)",
            root.resolve(),
            len(ohrc_files),
            len(tmc_files),
        )
        return 2

    ohrc_products, _ = load_catalog(ohrc_files, "OHRC")
    tmc_products, _ = load_catalog(tmc_files, "TMC2")
    if not ohrc_products or not tmc_products:
        logger.error("One of the catalogs loaded zero usable rows; aborting")
        return 2

    logger.info(
        "Searching %d OHRC x %d TMC-2 footprints...", len(ohrc_products), len(tmc_products)
    )
    search_result = search_overlaps_with_index(ohrc_products, tmc_products)
    ranked = rank_pairs(search_result.pairs, args.max_abs_lat)

    top = ranked[: args.top]
    payload = {
        "schema_version": "1.0",
        "generated_note": "candidate pairs only; transforms require downloaded labels",
        "catalog_root": str(root.resolve()),
        "stats": {
            "ohrc_products_loaded": len(ohrc_products),
            "tmc2_products_loaded": len(tmc_products),
            "combinations_checked_naive": search_result.stats.combos_checked,
            "spatial_prefilter_candidates": search_result.stats.prefilter_candidates,
            "overlapping_pairs_found": search_result.stats.overlapping_pairs,
            "non_polar_overlapping_pairs": sum(
                1
                for p in search_result.pairs
                if not is_polar(p, args.max_abs_lat)
            ),
        },
        "ranking": "non-polar first, then overlap % desc, then TMC2 nadir-view preferred",
        "candidates": [pair_to_json(p, i + 1, args.max_abs_lat) for i, p in enumerate(top)],
    }

    output_path: Path = args.output
    if output_path.parent and str(output_path.parent) not in ("", "."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    logger.info("Wrote %d candidates to %s", len(top), output_path.resolve())

    stats = search_result.stats
    non_polar_total = payload["stats"]["non_polar_overlapping_pairs"]

    print(f"Checked {stats.ohrc_count} OHRC x {stats.tmc_count} TMC-2 "
          f"combinations ({stats.combos_checked:,} total).")
    print(f"Spatial index pre-filter passed {stats.prefilter_candidates:,} candidate pairs.")
    print(f"Precise intersection confirmed {stats.overlapping_pairs} overlapping pairs "
          f"({non_polar_total} within +/-{args.max_abs_lat:g} deg latitude).")
    print(f"Top {len(top)} candidates written to {output_path.resolve()}")

    if top:
        best = top[0]
        c = best.overlap_polygon.centroid
        print(
            f"Best candidate: OHRC product {best.ohrc.product_id} "
            f"(filename {best.ohrc.download_filename}) overlaps TMC-2 product "
            f"{best.tmc2.product_id} (filename {best.tmc2.download_filename}) by "
            f"{best.overlap_percentage:.2f}% around latitude/longitude "
            f"({c.y:.3f}, {c.x:.3f})."
        )
        if not _angles_usable(best):
            print(
                "Note: catalog sun-angle columns are zero-filled, so "
                "terminator/shadow filtering was not possible; eyeball the "
                "browse images before downloading."
            )
    else:
        print("No overlapping pairs found between the two catalogs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
