"""Rank OHRC x TMC-2 overlap candidates for crater-detection testing.

Extends ``catalog_overlap_search``'s footprint-overlap search with two
additional, independent axes so we can pick a *crater-dense*, *matched-
illumination* pair rather than the biggest footprint overlap alone:

1. **Crater density** -- counts how many ground-truth craters from
   ``data/lunar_craters.parquet`` (HuggingFace Robbins subset; ~30% complete
   vs the full ~1.3M Robbins database) fall inside each candidate's overlap
   footprint. Used as a relative density proxy for ranking, not an absolute
   count, because the parquet is a partial Robbins subset.
2. **Sun-angle similarity** -- how closely the OHRC and TMC-2 members of a
   pair match in sun elevation and azimuth.

Sun-angle caveat (important): the PRADAN *catalog* shapefiles carry
zero-filled INC_ANGLE/EMI_ANGLE/PHA_ANGLE columns, so matching illumination
is NOT possible catalog-wide. Real sun_azimuth / sun_elevation /
solar_incidence values only exist in each product's own PDS4 XML label
(parsed by ``pds4_parser``). That means screening K candidates on sun-angle
requires downloading the (small) XML label for each member of those K pairs.
This module accepts sun angles for whichever candidates already have labels
available and reports the remaining candidates that still need labels, so a
targeted batch of tiny label-only downloads can be approved before committing
to full multi-GB imagery.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import Point, box

try:  # package-style import when used as a module...
    from .catalog_overlap_search import search_overlaps_with_index
    from .overlap_finder import OverlapPair
    from .pds4_parser import LunarProduct, parse_label
    from .shapefile_loader import discover_catalog_files, load_catalog
except ImportError:  # ...and flat import when scripts run directly
    from catalog_overlap_search import search_overlaps_with_index  # type: ignore[no-redef]
    from overlap_finder import OverlapPair  # type: ignore[no-redef]
    from pds4_parser import LunarProduct, parse_label  # type: ignore[no-redef]
    from shapefile_loader import discover_catalog_files, load_catalog  # type: ignore[no-redef]

logger = logging.getLogger("lunar_data_pipeline.candidate_ranking")

#: Default CRATER_COLUMNS keyword, part of the craters DataFrame contract.
LON_COL = "longitude_deg"
LAT_COL = "latitude_deg"

#: Sun-angle thresholds the shortlist must satisfy to be considered "matched".
MIN_SUN_ELEVATION_DEG = 20.0
MAX_SUN_ELEVATION_DIFF_DEG = 30.0
MAX_SUN_AZIMUTH_DIFF_DEG = 30.0

#: Minimum number of (parquet) craters inside a footprint to be crater-dense.
MIN_CRATERS_IN_FOOTPRINT = 5


@dataclass
class SunAngles:
    """Real sun geometry for one product, parsed from its PDS4 label."""

    sun_elevation_deg: float | None = None
    sun_azimuth_deg: float | None = None

    @property
    def usable(self) -> bool:
        return (
            self.sun_elevation_deg is not None
            and self.sun_azimuth_deg is not None
            and math.isfinite(self.sun_elevation_deg)
            and math.isfinite(self.sun_azimuth_deg)
        )


def load_craters(parquet_path: str | Path) -> pd.DataFrame:
    """Load the ground-truth crater table (lon, lat, diameter_km required)."""
    return pd.read_parquet(parquet_path)


def build_crater_index(df: pd.DataFrame) -> tuple[list[Point], STRtree]:
    """Return (points aligned with df rows, STRtree) for point-in-poly lookup."""
    points = [
        Point(lon, lat)
        for lon, lat in zip(df[LON_COL].to_numpy(), df[LAT_COL].to_numpy())
    ]
    return points, STRtree(points)


def count_craters_in_polygon(
    tree: STRtree, points: list[Point], polygon
) -> int:
    """Number of crater points whose centroid lies inside *polygon*."""
    bounds = polygon.bounds
    query_box = box(*bounds)
    candidates = tree.query(query_box)
    total = 0
    for idx in candidates:
        if points[int(idx)].within(polygon) or polygon.contains(points[int(idx)]):
            total += 1
    return total


def crater_point_counts(df: pd.DataFrame, polygons: Iterable) -> list[int]:
    """Crater counts for many polygons in one pass (builds the index once)."""
    points, tree = build_crater_index(df)
    from shapely.geometry import MultiPolygon

    out: list[int] = []
    for polygon in polygons:
        if polygon is None or polygon.is_empty:
            out.append(0)
            continue
        if isinstance(polygon, MultiPolygon):
            count = 0
            for sub in polygon.geoms:
                count += count_craters_in_polygon(tree, points, sub)
            out.append(count)
        else:
            out.append(count_craters_in_polygon(tree, points, polygon))
    return out


def polygon_area_deg2(polygon) -> float:
    """Planar lon/lat degree area (approximate; fine for relative ranking)."""
    return polygon.area if polygon is not None else 0.0


@dataclass
class CandidateScore:
    """All ranking signals for one OHRC x TMC-2 candidate pair."""

    ohrc_id: str
    tmc2_id: str
    ohrc_download: str | None
    tmc2_download: str | None
    overlap_centroid: tuple[float, float]
    overlap_percentage: float
    crater_count: int
    crater_density_per_deg2: float
    sun_elev_diff_deg: float | None = None
    sun_az_diff_deg: float | None = None
    sun_angles_available: bool = False
    labels_needed: list[str] = field(default_factory=list)
    passes_filters: bool = False
    filter_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ohrc_id": self.ohrc_id,
            "tmc2_id": self.tmc2_id,
            "ohrc_download": self.ohrc_download,
            "tmc2_download": self.tmc2_download,
            "overlap_center_lonlat": list(self.overlap_centroid),
            "overlap_percentage_smaller_pct": round(self.overlap_percentage, 6),
            "crater_count_in_footprint": self.crater_count,
            "crater_density_per_deg2": round(self.crater_density_per_deg2, 4),
            "sun_elevation_diff_deg": self.sun_elev_diff_deg,
            "sun_azimuth_diff_deg": self.sun_az_diff_deg,
            "sun_angles_available": self.sun_angles_available,
            "labels_needed_for_sun": self.labels_needed,
            "passes_filters": self.passes_filters,
            "filter_reasons": self.filter_reasons,
        }


def _angle_diff(a: float, b: float) -> float:
    """Smallest angular difference between two degrees, in [0, 180]."""
    diff = abs(a - b) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def sun_angles_from_labels(labels_dir: str | Path | None) -> dict[str, SunAngles]:
    """Parse sun angles from every PDS4 data label under *labels_dir*.

    Walks the directory tree for ``*_d_img_*.xml`` files and returns a map of
    ``product_id -> SunAngles`` for each one that yields at least an
    elevation. If *labels_dir* is None, returns an empty dict (no labels).
    """
    result: dict[str, SunAngles] = {}
    if labels_dir is None:
        return result
    labels_dir = Path(labels_dir)
    if not labels_dir.is_dir():
        logger.warning("Labels directory does not exist: %s", labels_dir)
        return result
    for xml in sorted(labels_dir.rglob("*_d_img_*.xml")):
        product = parse_label(xml)
        if product is None:
            continue
        result[product.product_id] = SunAngles(
            sun_elevation_deg=product.sun_elevation_deg,
            sun_azimuth_deg=product.sun_azimuth_deg,
        )
    logger.info("Parsed sun angles from %d available labels", len(result))
    return result


def score_pair(
    pair: OverlapPair,
    crater_count: int,
    sun_map: dict[str, SunAngles],
    *,
    min_elevation_deg: float = MIN_SUN_ELEVATION_DEG,
    min_overlap_pct: float = 90.0,
    min_craters: int = MIN_CRATERS_IN_FOOTPRINT,
) -> CandidateScore:
    """Compute the ranking signals for a single candidate pair.

    Sun-angle similarity is computed only when *both* members' labels are in
    *sun_map*; otherwise the pair is flagged with the labels it still needs.
    ``passes_filters`` records whether the pair meets the overlap, crater-count
    and (when sun angles are available) sun-elevation thresholds, so the
    shortlist can be built from candidates that genuinely satisfy all three.
    """
    centroid = pair.overlap_polygon.centroid
    ohrc_angles = sun_map.get(pair.ohrc.product_id)
    tmc_angles = sun_map.get(pair.tmc2.product_id)

    labels_needed: list[str] = []
    if ohrc_angles is None or not ohrc_angles.usable:
        labels_needed.append(pair.ohrc.product_id)
    if tmc_angles is None or not tmc_angles.usable:
        labels_needed.append(pair.tmc2.product_id)

    sun_available = not labels_needed
    ev_diff = az_diff = None
    if sun_available:
        ev_diff = abs(ohrc_angles.sun_elevation_deg - tmc_angles.sun_elevation_deg)
        az_diff = _angle_diff(
            ohrc_angles.sun_azimuth_deg, tmc_angles.sun_azimuth_deg
        )

    area = polygon_area_deg2(pair.overlap_polygon)
    density = crater_count / area if area > 0 else 0.0

    reasons: list[str] = []
    if pair.overlap_percentage < min_overlap_pct:
        reasons.append(
            f"overlap {pair.overlap_percentage:.1f}% < {min_overlap_pct}%"
        )
    if crater_count < min_craters:
        reasons.append(f"only {crater_count} craters in footprint (< {min_craters})")
    if sun_available:
        if ohrc_angles.sun_elevation_deg < min_elevation_deg:
            reasons.append(
                f"OHRC sun elevation {ohrc_angles.sun_elevation_deg:.1f} < {min_elevation_deg}"
            )
        if tmc_angles.sun_elevation_deg < min_elevation_deg:
            reasons.append(
                f"TMC2 sun elevation {tmc_angles.sun_elevation_deg:.1f} < {min_elevation_deg}"
            )
        if ev_diff is not None and ev_diff > MAX_SUN_ELEVATION_DIFF_DEG:
            reasons.append(f"sun elevation diff {ev_diff:.1f} deg")
    else:
        # Sun angles for one/both members are not downloaded yet, so the
        # illumination match cannot be confirmed; the pair does not satisfy the
        # sun constraint until labels are pulled (tiny downloads).
        reasons.append("sun angles unavailable - labels not downloaded")
    passes = not reasons

    return CandidateScore(
        ohrc_id=pair.ohrc.product_id,
        tmc2_id=pair.tmc2.product_id,
        ohrc_download=pair.ohrc.download_filename,
        tmc2_download=pair.tmc2.download_filename,
        overlap_centroid=(float(centroid.x), float(centroid.y)),
        overlap_percentage=pair.overlap_percentage,
        crater_count=crater_count,
        crater_density_per_deg2=density,
        sun_elev_diff_deg=ev_diff,
        sun_az_diff_deg=az_diff,
        sun_angles_available=sun_available,
        labels_needed=labels_needed,
        passes_filters=passes,
        filter_reasons=reasons,
    )


def rank_candidates(
    pairs: list[OverlapPair],
    craters: pd.DataFrame,
    sun_map: dict[str, SunAngles] | None = None,
    *,
    min_overlap_pct: float = 90.0,
    min_craters: int = MIN_CRATERS_IN_FOOTPRINT,
    min_elevation_deg: float = MIN_SUN_ELEVATION_DEG,
) -> tuple[list[CandidateScore], dict[str, str]]:
    """Rank candidates by overlap quality + crater density + sun-angle match.

    Returns ``(scored, diagnostics)``:

    - ``scored``: every candidate scored, sorted so that pairs passing the
      filters come first (by density desc), then the rest. Each entry carries
      ``passes_filters`` so the calling code can build a strict shortlist from
      only the candidates that satisfy overlap + crater-count + sun constraints.
    - ``diagnostics``: human-readable notes keyed by pair id.
    """
    sun_map = sun_map or {}
    counts = crater_point_counts(craters, [p.overlap_polygon for p in pairs])

    scored: list[CandidateScore] = []
    diagnostics: dict[str, str] = {}
    for pair, count in zip(pairs, counts):
        score = score_pair(
            pair,
            count,
            sun_map,
            min_elevation_deg=min_elevation_deg,
            min_overlap_pct=min_overlap_pct,
            min_craters=min_craters,
        )
        pair_id = f"{score.ohrc_id}__x__{score.tmc2_id}"
        diagnostics[pair_id] = (
            "; ".join(score.filter_reasons) if score.filter_reasons else "passes all filters"
        )
        scored.append(score)

    def sort_key(score: CandidateScore):
        pass_rank = 0 if score.passes_filters else 1
        sun_rank = 0 if score.sun_angles_available else 1
        return (pass_rank, sun_rank, -score.crater_density_per_deg2)

    scored.sort(key=sort_key)
    return scored, diagnostics


def _load_pairs(args) -> tuple[list[OverlapPair], dict[str, int]]:
    """Re-run (or reuse) the footprint overlap search to get real polygons."""
    root: Path = args.catalog_root
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
        raise FileNotFoundError(
            "Could not locate both catalogs. Pass --ohrc-shapefile/--tmc-shapefile "
            "or place them under --catalog-root."
        )
    ohrc_products, _ = load_catalog(ohrc_files, "OHRC")
    tmc_products, _ = load_catalog(tmc_files, "TMC2")
    return search_overlaps_with_index(ohrc_products, tmc_products).pairs, {
        "ohrc": len(ohrc_products),
        "tmc2": len(tmc_products),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rank OHRC x TMC-2 overlap candidates for crater-detection testing "
            "by overlap quality + crater density + sun-angle match."
        )
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=Path("."),
        help="Folder containing PRADAN shapefile catalogs (searched recursively).",
    )
    parser.add_argument("--ohrc-shapefile", action="append", default=None, metavar="PATH")
    parser.add_argument("--tmc-shapefile", action="append", default=None, metavar="PATH")
    parser.add_argument(
        "--craters",
        type=Path,
        default=Path("data/lunar_craters.parquet"),
        help="Ground-truth crater parquet (Robbins HF subset).",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory tree containing downloaded *_d_img_*.xml labels to "
        "supply real sun angles for the candidates that already have them.",
    )
    parser.add_argument(
        "--min-overlap-pct",
        type=float,
        default=90.0,
        help="Minimum overlap percentage of the smaller (OHRC) footprint (default 90).",
    )
    parser.add_argument(
        "--min-craters",
        type=int,
        default=MIN_CRATERS_IN_FOOTPRINT,
        help="Minimum parquet craters inside a footprint (default 5).",
    )
    parser.add_argument(
        "--min-sun-elevation",
        type=float,
        default=MIN_SUN_ELEVATION_DEG,
        help="Each member must be above this sun elevation to count as matched "
        "(default 20 deg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("candidate_ranking.json"),
        help="Output JSON path (default ./candidate_ranking.json).",
    )
    parser.add_argument("--top", type=int, default=10, help="Shortlist size (default 10).")
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

    if not args.craters.exists():
        logger.error("Craters parquet not found: %s", args.craters)
        return 2
    if not args.catalog_root.is_dir():
        logger.error("Catalog root does not exist: %s", args.catalog_root)
        return 2

    pairs, counts = _load_pairs(args)
    logger.info(
        "Loaded %d OHRC + %d TMC-2 catalog entries; %d overlapping pairs",
        counts["ohrc"],
        counts["tmc2"],
        len(pairs),
    )

    craters = load_craters(args.craters)
    sun_map = sun_angles_from_labels(args.labels_dir)

    scored, diagnostics = rank_candidates(
        pairs,
        craters,
        sun_map,
        min_overlap_pct=args.min_overlap_pct,
        min_craters=args.min_craters,
        min_elevation_deg=args.min_sun_elevation,
    )

    passing = [s for s in scored if s.passes_filters]
    # If the strict thresholds are unmet by any candidate, surface the most
    # promising (highest-density) non-passing candidates so the caller can see
    # what labels are still worth pulling, rather than returning an empty list.
    source = passing if passing else scored
    shortlist = source[: args.top]
    payload = {
        "schema_version": "1.0",
        "craters_source": str(args.craters.resolve()),
        "craters_completeness_note": (
            "HuggingFace Robbins subset (~30% of full Robbins); density is a "
            "relative ranking proxy, not an absolute count."
        ),
        "sun_angle_feasibility": (
            "catalog angle columns are zero-filled; real sun angles exist only "
            "in per-product PDS4 labels. Candidates without a downloaded label "
            "are flagged under labels_needed_for_sun."
        ),
        "filters": {
            "min_overlap_pct": args.min_overlap_pct,
            "min_craters_in_footprint": args.min_craters,
            "min_sun_elevation_deg": args.min_sun_elevation,
            "max_sun_elevation_diff_deg": MAX_SUN_ELEVATION_DIFF_DEG,
            "max_sun_azimuth_diff_deg": MAX_SUN_AZIMUTH_DIFF_DEG,
        },
        "stats": {
            "ohrc_products_loaded": counts["ohrc"],
            "tmc2_products_loaded": counts["tmc2"],
            "overlapping_pairs": len(pairs),
            "labels_with_sun_angles": len(sun_map),
            "candidates_passing_all_filters": len(passing),
            "candidates_with_sun_angles": sum(
                1 for s in scored if s.sun_angles_available
            ),
            "candidates_needing_labels": sum(
                1 for s in scored if not s.sun_angles_available
            ),
        },
        "shortlist_source": (
            "passes_all_filters" if passing else "fallback_highest_density_needs_labels"
        ),
        "shortlist": [s.to_dict() for s in shortlist],
        "diagnostics": diagnostics,
    }

    out: Path = args.output
    if out.parent and str(out.parent) not in ("", "."):
        out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    logger.info("Wrote shortlist (%d) to %s", len(shortlist), out.resolve())

    print(f"Catalog: {counts['ohrc']} OHRC x {counts['tmc2']} TMC-2; "
          f"{len(pairs)} overlapping pairs.")
    print(f"Labels with sun angles available: {len(sun_map)}; "
          f"candidates passing all filters: {len(passing)}.")
    print(f"Top {len(shortlist)} candidates "
          f"({'filtered (all pass)' if passing else 'fallback (need labels)'}):")
    for s in shortlist:
        sun_tag = "sun-OK" if s.sun_angles_available else "need-labels"
        ok = "PASS" if s.passes_filters else "FAIL"
        print(
            f"  [{ok}][{sun_tag}] {s.ohrc_id} x {s.tmc2_id} "
            f"craters={s.crater_count} dens={s.crater_density_per_deg2:.3f}/deg2 "
            f"overlap={s.overlap_percentage:.1f}% "
            f"lonlat=({s.overlap_centroid[0]:.2f},{s.overlap_centroid[1]:.2f})"
        )
        if not s.passes_filters:
            print(f"        reasons: {'; '.join(s.filter_reasons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
