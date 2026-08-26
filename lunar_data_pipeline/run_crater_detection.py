"""CLI: run crater detection on an OHRC / TMC-2 pair and evaluate against GT.

Usage:
    # With external crater database:
    python run_crater_detection.py --manifest pair_manifest.json \\
        --crater-db ../data/robbins_lunar_craters.csv

    # With synthetic craters (fallback, no database needed):
    python run_crater_detection.py --manifest pair_manifest.json --synthetic

    # Custom parameters:
    python run_crater_detection.py --manifest pair_manifest.json \\
        --crater-db ../data/robbins_lunar_craters.csv \\
        --min-diameter-km 2.0 --match-radius-px 200 --output-dir crater_output

Steps:
1. Read the pair from the manifest; parse both labels.
2. Load imagery (OHRC full, TMC-2 crop to overlap region).
3. Load or generate crater GT (external CSV or synthetic).
4. Run classical CV detection on each image.
5. Score detections against GT using positional + size matching.
6. Write overlay visualisations and results JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from .crater_detector import (
        CraterDetection,
        DetectionParams,
        detect_craters,
        detect_multiscale,
        params_for_resolution,
    )
    from .csv_geolocation import fit_local_transforms, locate_region, read_geolocation_csv
    from .image_loader import load_image, save_preview, to_uint8
    from .lro_groundtruth import (
        CraterRecord,
        filter_by_region,
        generate_synthetic_craters,
        load_crater_database,
        overlap_bounds,
        product_bounds,
    )
    from .pds4_parser import parse_label
except ImportError:
    from crater_detector import (  # type: ignore[no-redef]
        CraterDetection,
        DetectionParams,
        detect_craters,
        detect_multiscale,
        params_for_resolution,
    )
    from csv_geolocation import fit_local_transforms, locate_region, read_geolocation_csv  # type: ignore[no-redef]
    from image_loader import load_image, save_preview, to_uint8  # type: ignore[no-redef]
    from lro_groundtruth import (  # type: ignore[no-redef]
        CraterRecord,
        filter_by_region,
        generate_synthetic_craters,
        load_crater_database,
        overlap_bounds,
        product_bounds,
    )
    from pds4_parser import parse_label  # type: ignore[no-redef]

logger = logging.getLogger("lunar_data_pipeline.crater_detection")


# ---------------------------------------------------------------------------
# Scoring: detection vs GT
# ---------------------------------------------------------------------------

def score_detections(
    detections: list[CraterDetection],
    gt_craters: list[CraterRecord],
    product,
    *,
    match_radius_px: float = 150,
    match_diameter_ratio: float = 3.0,
) -> dict:
    """Score detections against ground-truth craters.

    A detection matches a GT crater if:
      - centre distance < match_radius_px, AND
      - |log2(det_diameter / gt_diameter)| < log2(match_diameter_ratio)

    Returns dict with TP/FP/FN counts, precision, recall, F1, and per-detection matches.
    """
    from .lro_groundtruth import craters_to_pixels

    gt_pixels = craters_to_pixels(gt_craters, product)  # (x, y, radius_px)
    gt_matched = [False] * len(gt_pixels)
    results = []

    for det in detections:
        best_iou = 0.0
        best_gt_idx = -1
        best_dist = float("inf")

        for i, (gx, gy, gr) in enumerate(gt_pixels):
            dist = math.hypot(det.cx - gx, det.cy - gy)
            if dist > match_radius_px:
                continue

            # Size compatibility check
            det_diam = det.radius * 2
            gt_diam = gr * 2
            if gt_diam > 0:
                size_ratio = max(det_diam, gt_diam) / max(min(det_diam, gt_diam), 1e-6)
                if size_ratio > match_diameter_ratio:
                    continue

            # IoU approximation (circles)
            overlap = max(0, min(det.radius, gr) - dist / 2) ** 2 * math.pi
            union = math.pi * (det.radius ** 2 + gr ** 2) - overlap
            iou = overlap / max(union, 1e-6)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                best_dist = dist

        if best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
            gt_matched[best_gt_idx] = True
            results.append({
                "det_cx": det.cx, "det_cy": det.cy, "det_radius": det.radius,
                "gt_idx": best_gt_idx,
                "gt_cx": gt_pixels[best_gt_idx][0],
                "gt_cy": gt_pixels[best_gt_idx][1],
                "gt_radius": gt_pixels[best_gt_idx][2],
                "centre_dist_px": best_dist,
                "iou": best_iou,
                "match": True,
            })
        else:
            results.append({
                "det_cx": det.cx, "det_cy": det.cy, "det_radius": det.radius,
                "gt_idx": -1, "centre_dist_px": float("inf"),
                "iou": 0.0, "match": False,
            })

    tp = sum(1 for r in results if r["match"])
    fp = sum(1 for r in results if not r["match"])
    fn = sum(1 for m in gt_matched if not m)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "n_gt": len(gt_craters),
        "n_detections": len(detections),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_detection": results,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_overlay(
    gray: np.ndarray,
    detections: list[CraterDetection],
    gt_craters_xy: list[tuple[float, float, float]] | None = None,
    *,
    title: str = "",
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Draw detection overlay: green = detected, blue = GT, red = missed GT."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(gray.shape) == 2 else gray.copy()

    # Draw GT craters (blue circles)
    if gt_craters_xy:
        for gx, gy, gr in gt_craters_xy:
            cv2.circle(vis, (int(gx), int(gy)), int(gr), (255, 180, 0), 2)  # blue in BGR
            cv2.circle(vis, (int(gx), int(gy)), 3, (255, 180, 0), -1)

    # Draw detections (green circles)
    for d in detections:
        color = (0, 220, 0) if d.confidence > 0.3 else (0, 160, 160)  # green / yellow
        cv2.circle(vis, (int(d.cx), int(d.cy)), int(d.radius), color, 2)
        cv2.circle(vis, (int(d.cx), int(d.cy)), 3, color, -1)

    # Title bar
    if title:
        cv2.rectangle(vis, (0, 0), (len(vis[0]), 30), (0, 0, 0), -1)
        cv2.putText(vis, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if output_path:
        cv2.imwrite(str(output_path), vis)
        logger.info("Saved overlay: %s", output_path)

    return vis


# ---------------------------------------------------------------------------
# Image loading helper
# ---------------------------------------------------------------------------

def load_image_region(
    label,
    csv_path: str | Path | None,
    *,
    region_polygon=None,
    center_lonlat: tuple[float, float] | None = None,
    half_width_px: int = 6000,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Load an image region using geolocation CSV refinement.

    ``label`` must be a parsed ``LunarProduct`` (from ``pds4_parser``).
    Returns ``(crop, scan_range, pixel_range)``.
    """
    from shapely.geometry import Polygon, box

    if csv_path and Path(csv_path).exists():
        grid = read_geolocation_csv(csv_path)

        if region_polygon is not None:
            scan_range, pixel_range, _ = locate_region(grid, region_polygon)
        elif center_lonlat is not None:
            clon, clat = center_lonlat
            delta_deg = half_width_px * label.pixel_resolution_m / 30000.0
            region_polygon = box(clon - delta_deg, clat - delta_deg, clon + delta_deg, clat + delta_deg)
            scan_range, pixel_range, _ = locate_region(grid, region_polygon)
        else:
            scan_range = (0, label.lines - 1)
            pixel_range = (0, label.samples - 1)

        s0, s1 = scan_range
        p0, p1 = pixel_range
        logger.info("Loading crop scan[%d,%d] pixel[%d,%d] from %s", s0, s1, p0, p1, label.file_path)
        crop = load_image(label, row_range=(s0, s1 + 1), col_range=(p0, p1 + 1))
        return crop, scan_range, pixel_range

    logger.warning("No geolocation CSV; loading full image (may be very large)")
    return load_image(label), (0, label.lines - 1), (0, label.samples - 1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_crater_detection(manifest_path: str, **kwargs) -> dict:
    """Run the full crater detection + GT evaluation pipeline."""
    manifest_path = Path(manifest_path)
    with manifest_path.open() as f:
        manifest = json.load(f)

    pair = manifest["pairs"][0]
    ohrc_id = pair["ohrc"]["product_id"]
    tmc2_id = pair["tmc2"]["product_id"]
    ohrc_prod = manifest["products"][ohrc_id]
    tmc2_prod = manifest["products"][tmc2_id]

    output_dir = Path(kwargs.get("output_dir", "crater_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse labels
    ohrc_label = parse_label(Path(ohrc_prod["label_path"]))
    tmc2_label = parse_label(Path(tmc2_prod["label_path"]))

    # Determine overlap bounds
    lon_min, lon_max, lat_min, lat_max = overlap_bounds(ohrc_label, tmc2_label)
    logger.info("Overlap bounds: lon[%.4f, %.4f] lat[%.4f, %.4f]", lon_min, lon_max, lat_min, lat_max)

    # Load crater GT
    crater_db_path = kwargs.get("crater_db")
    if crater_db_path and Path(crater_db_path).exists():
        all_craters = load_crater_database(
            crater_db_path,
            min_diameter_km=kwargs.get("min_diameter_km", 1.0),
        )
        gt_craters = filter_by_region(all_craters, lon_min=lon_min, lon_max=lon_max,
                                      lat_min=lat_min, lat_max=lat_max)
        gt_source = f"external ({Path(crater_db_path).name})"
    else:
        gt_craters = generate_synthetic_craters(
            lon_min, lon_max, lat_min, lat_max,
            count=kwargs.get("synthetic_count", 20),
        )
        gt_source = "synthetic"

    logger.info("GT source: %s — %d craters in overlap region", gt_source, len(gt_craters))

    results = {
        "manifest": str(manifest_path),
        "gt_source": gt_source,
        "gt_count": len(gt_craters),
        "overlap_bounds": {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max},
    }

    # Process each instrument — skip OHRC (0.2 m/px) for classical CV detection
    # because craters ≥1 km span 5000+ px radius (full image width).
    # Classical HoughCircles is appropriate for TMC-2 (6.13 m/px).
    instruments = [
        ("TMC-2", tmc2_prod, tmc2_label),
    ]
    if kwargs.get("include_ohrc", False):
        instruments.insert(0, ("OHRC", ohrc_prod, ohrc_label))

    for instrument, prod, label in instruments:
        logger.info("=" * 60)
        logger.info("Processing %s (%s)", instrument, prod["product_id"])
        logger.info("=" * 60)

        csv_path = prod.get("geometry_csv_path")
        if csv_path:
            # Resolve relative path from data_root
            csv_full = Path(manifest.get("data_root", manifest_path.parent)) / csv_path
            if not csv_full.exists():
                csv_full = None
        else:
            csv_full = None

        try:
            # Use overlap region polygon for cropping
            from shapely.geometry import Polygon as ShapelyPolygon
            overlap_poly = ShapelyPolygon(pair["overlap"]["polygon_lonlat"])

            t0 = time.time()
            gray, scan_range, pixel_range = load_image_region(label, csv_full, region_polygon=overlap_poly)
            load_time = time.time() - t0
            logger.info("Loaded %s crop: %dx%d in %.1fs (scan %s, pixel %s)",
                        instrument, gray.shape[1], gray.shape[0], load_time, scan_range, pixel_range)

            # Ensure uint8
            if gray.dtype != np.uint8:
                from .image_loader import to_uint8 as _to8
                gray = _to8(gray)

            # Detect craters
            resolution_m = prod["pixel_resolution_m"]
            t0 = time.time()

            if max(gray.shape) > 4000:
                detections = detect_multiscale(gray, resolution_m=resolution_m)
            else:
                params = params_for_resolution(resolution_m)
                detections = detect_craters(gray, params)

            detect_time = time.time() - t0
            logger.info("Detected %d craters in %s in %.1fs", len(detections), instrument, detect_time)

            # Score against GT
            score = score_detections(
                detections, gt_craters, label,
                match_radius_px=kwargs.get("match_radius_px", 150),
            )
            logger.info(
                "%s: TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f",
                instrument, score["tp"], score["fp"], score["fn"],
                score["precision"], score["recall"], score["f1"],
            )

            # Draw overlay
            from .lro_groundtruth import craters_to_pixels
            gt_px = craters_to_pixels(gt_craters, label)
            vis_path = output_dir / f"{instrument}_detections.png"
            draw_overlay(gray, detections, gt_px, title=f"{instrument}: {len(detections)} detected, GT={len(gt_craters)}", output_path=vis_path)

            # Save preview of raw crop
            crop_path = output_dir / f"{instrument}_crop.png"
            save_preview(gray, str(crop_path))

            results[instrument.lower().replace("-", "")] = {
                "product_id": prod["product_id"],
                "crop_size": [int(gray.shape[1]), int(gray.shape[0])],
                "resolution_m": resolution_m,
                "n_detections": len(detections),
                "load_time_s": round(load_time, 2),
                "detect_time_s": round(detect_time, 2),
                "scoring": score,
            }

        except Exception as exc:
            logger.error("%s failed: %s", instrument, exc, exc_info=True)
            results[instrument.lower().replace("-", "")] = {"error": str(exc)}

    # Save results JSON
    results_path = output_dir / "crater_detection_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved: %s", results_path)

    # Print summary
    print("\n" + "=" * 60)
    print("CRATER DETECTION RESULTS")
    print("=" * 60)
    print(f"GT source: {gt_source} ({len(gt_craters)} craters in overlap region)")
    print(f"Overlap: lon[{lon_min:.4f}, {lon_max:.4f}] lat[{lat_min:.4f}, {lat_max:.4f}]")
    for key in ("ohrc", "tmc2"):
        if key in results and "scoring" in results[key]:
            s = results[key]["scoring"]
            print(f"\n{key.upper()}: {results[key]['n_detections']} detections")
            print(f"  TP={s['tp']}  FP={s['fp']}  FN={s['fn']}")
            print(f"  Precision={s['precision']:.3f}  Recall={s['recall']:.3f}  F1={s['f1']:.3f}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crater detection + GT evaluation for Chandrayaan-2 pair",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Path to pair_manifest.json")
    parser.add_argument("--crater-db", default=None, help="Path to crater CSV database (Robbins/LU1319373/IAU)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic GT (fallback)")
    parser.add_argument("--synthetic-count", type=int, default=20, help="Number of synthetic craters")
    parser.add_argument("--min-diameter-km", type=float, default=1.0, help="Min crater diameter in GT (km)")
    parser.add_argument("--match-radius-px", type=float, default=150, help="Max centre distance for a TP match (px)")
    parser.add_argument("--output-dir", default="crater_output", help="Output directory")
    parser.add_argument("--include-ohrc", action="store_true", help="Also run detection on OHRC (slow, 0.2 m/px)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_crater_detection(
        args.manifest,
        crater_db=args.crater_db,
        synthetic=args.synthetic,
        synthetic_count=args.synthetic_count,
        min_diameter_km=args.min_diameter_km,
        match_radius_px=args.match_radius_px,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
