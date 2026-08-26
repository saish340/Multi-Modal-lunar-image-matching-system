"""CLI pipeline: parse PDS4 labels -> find OHRC/TMC-2 overlaps -> write manifest.

Usage:
    python pipeline.py --data-root /path/to/downloaded/products --output manifest.json

Walks *data-root* for PDS4 data labels (``*_d_img_*.xml`` style files),
parses them into product metadata, computes footprint overlaps between OHRC
and TMC-2 products, builds corner-homography ground-truth transforms for each
confirmed pair, and writes everything to a JSON manifest documented in
``manifest_schema.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

try:  # package-style import when used as a module...
    from .groundtruth import transform_to_manifest_dict
    from .overlap_finder import OverlapPair, find_overlaps
    from .pds4_parser import LunarProduct, parse_label
except ImportError:  # ...and flat import when run directly as a script
    from groundtruth import transform_to_manifest_dict  # type: ignore[no-redef]
    from overlap_finder import OverlapPair, find_overlaps  # type: ignore[no-redef]
    from pds4_parser import LunarProduct, parse_label  # type: ignore[no-redef]

logger = logging.getLogger("lunar_data_pipeline")

MANIFEST_SCHEMA_VERSION = "1.0"

#: Filename fragments identifying non-image-product labels to skip early.
_NON_DATA_LABEL_MARKERS = ("_brw", "_grd")

#: Corner order shared with the parser (UL, UR, LR, LL).
CORNER_ORDER = ("upper_left", "upper_right", "lower_right", "lower_left")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def discover_products(data_root: Path) -> tuple[list[LunarProduct], list[dict]]:
    """Recursively find and parse data-product labels under *data_root*.

    Returns ``(products, skipped)`` where *skipped* records every label file
    that was rejected or failed to parse, for diagnostics.
    """
    products: list[LunarProduct] = []
    skipped: list[dict] = []

    xml_files = sorted(data_root.rglob("*.xml"))
    if not xml_files:
        logger.warning("No XML label files found under %s", data_root)

    for xml_path in xml_files:
        lowered = xml_path.name.lower()
        if any(marker in lowered for marker in _NON_DATA_LABEL_MARKERS):
            logger.debug("Skipping non-data label by name: %s", xml_path)
            skipped.append({"label": str(xml_path), "reason": "browse/geometry label"})
            continue

        product = parse_label(xml_path)
        if product is None:
            skipped.append({"label": str(xml_path), "reason": "parse failed or incomplete"})
            continue

        if not product.footprint:
            skipped.append({"label": str(xml_path), "reason": "missing footprint corners"})

        if not product.file_path.exists():
            logger.warning(
                "%s references missing image file %s; keeping metadata anyway",
                product.product_id,
                product.file_path,
            )
        products.append(product)

    return products, skipped


def product_entry(product: LunarProduct, pairing_eligible: bool) -> dict:
    return {
        "product_id": product.product_id,
        "instrument": product.instrument,
        "label_path": str(product.label_path),
        "image_path": str(product.file_path),
        "geometry_csv_path": (
            str(product.geometry_csv_path) if product.geometry_csv_path else None
        ),
        "lines": product.lines,
        "samples": product.samples,
        "datatype": product.datatype,
        "pixel_resolution_m": product.pixel_resolution_m,
        "projection": product.projection,
        "area": product.area,
        "start_time": product.start_time,
        "stop_time": product.stop_time,
        "logical_identifier": product.logical_identifier,
        "footprint_lonlat_corners": [
            [lon, lat] for lat, lon in product.footprint
        ],
        "pairing_eligible": pairing_eligible,
    }


def pair_entry(pair: OverlapPair) -> dict:
    from groundtruth import build_pixel_transform  # local import keeps flat-script mode simple

    ohrc, tmc = pair.ohrc, pair.tmc2
    try:
        transform = build_pixel_transform(ohrc, tmc)
        ground_truth = transform_to_manifest_dict(transform)
    except ValueError as exc:
        logger.error(
            "Could not build ground truth for %s x %s: %s",
            ohrc.product_id,
            tmc.product_id,
            exc,
        )
        ground_truth = {"method": "unavailable", "error": str(exc)}

    polygon_coords = (
        [[round(float(x), 8), round(float(y), 8)] for x, y in pair.overlap_polygon.exterior.coords]
        if not pair.overlap_polygon.is_empty
        else []
    )
    centroid = pair.overlap_polygon.centroid

    return {
        "pair_id": f"{ohrc.product_id}__x__{tmc.product_id}",
        "ohrc": {
            "product_id": ohrc.product_id,
            "label_path": str(ohrc.label_path),
            "image_path": str(ohrc.file_path),
            "geometry_csv_path": str(ohrc.geometry_csv_path)
            if ohrc.geometry_csv_path
            else None,
            "lines": ohrc.lines,
            "samples": ohrc.samples,
        },
        "tmc2": {
            "product_id": tmc.product_id,
            "label_path": str(tmc.label_path),
            "image_path": str(tmc.file_path),
            "geometry_csv_path": str(tmc.geometry_csv_path)
            if tmc.geometry_csv_path
            else None,
            "lines": tmc.lines,
            "samples": tmc.samples,
        },
        "overlap": {
            "percentage_of_smaller_pct": round(pair.overlap_percentage, 6),
            "pct_of_ohrc_footprint": round(pair.pct_of_ohrc, 6),
            "pct_of_tmc2_footprint": round(pair.pct_of_tmc2, 6),
            "polygon_lonlat": polygon_coords,
            "centroid_lonlat": [round(float(centroid.x), 8), round(float(centroid.y), 8)],
        },
        "ground_truth": ground_truth,
    }


def build_manifest(
    data_root: Path,
    products: list[LunarProduct],
    pairs: list[OverlapPair],
    skipped: list[dict],
) -> dict:
    counts = {
        "OHRC": sum(1 for p in products if p.instrument == "OHRC"),
        "TMC2": sum(1 for p in products if p.instrument == "TMC2"),
        "IIRS": sum(1 for p in products if p.instrument == "IIRS"),
        "pairs": len(pairs),
    }

    paired_ohrc_ids = {pair.ohrc.product_id for pair in pairs}
    paired_tmc_ids = {pair.tmc2.product_id for pair in pairs}

    product_entries = {}
    for product in products:
        eligible = bool(product.footprint) and (
            product.product_id in paired_ohrc_ids or product.product_id in paired_tmc_ids
        )
        product_entries[product.product_id] = product_entry(product, eligible)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_root": str(data_root.resolve()),
        "counts": counts,
        "skipped_labels": skipped,
        "products": product_entries,
        "pairs": [pair_entry(pair) for pair in pairs],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an OHRC/TMC-2 overlap manifest with ground-truth pixel "
            "transforms from downloaded Chandrayaan-2 PDS4 products."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Root folder containing unzipped Chandrayaan-2 product folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pair_manifest.json"),
        help="Output manifest JSON path (default: ./pair_manifest.json).",
    )
    parser.add_argument(
        "--min-overlap-pct",
        type=float,
        default=0.0,
        help="Discard pairs covering less than this percent of the smaller "
        "footprint (default: 0, keep any positive-area intersection).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    data_root: Path = args.data_root
    if not data_root.is_dir():
        logger.error("Data root does not exist or is not a directory: %s", data_root)
        return 2

    logger.info("Scanning %s for PDS4 data labels...", data_root.resolve())
    products, skipped = discover_products(data_root)

    ohrc_products = [p for p in products if p.instrument == "OHRC"]
    tmc_products = [p for p in products if p.instrument == "TMC2"]
    iirs_products = [p for p in products if p.instrument == "IIRS"]
    other = [
        p
        for p in products
        if p.instrument not in ("OHRC", "TMC2", "IIRS")
    ]

    no_footprint = [p.product_id for p in products if not p.footprint]
    if no_footprint:
        logger.warning(
            "%d product(s) lack footprint corners and cannot be paired: %s",
            len(no_footprint),
            ", ".join(no_footprint),
        )

    pairs = find_overlaps(
        ohrc_products, tmc_products, min_overlap_pct=args.min_overlap_pct
    )

    manifest = build_manifest(data_root, products, pairs, skipped)

    output_path: Path = args.output
    if output_path.parent and str(output_path.parent) not in ("", "."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    logger.info("Manifest written to %s", output_path.resolve())

    # Required human-readable summary.
    print(
        f"Found {len(ohrc_products)} OHRC products, "
        f"{len(tmc_products)} TMC-2 products, "
        f"{len(pairs)} overlapping pairs."
    )
    if iirs_products:
        print(f"(Also parsed {len(iirs_products)} IIRS product(s); "
              "cross-instrument pairing arrives in a later phase.)")
    if other:
        print(f"(Ignored {len(other)} product(s) from unrecognised instruments.)")
    if not pairs:
        print(
            "No overlapping pairs detected. Check that the downloads actually "
            "cover the same lunar region (compare 'area' and footprint corners "
            "across instruments)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
