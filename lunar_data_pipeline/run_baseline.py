"""CLI: run the SIFT+RANSAC baseline on a confirmed pair from the manifest.

Usage:
    python run_baseline.py --manifest pair_manifest.json --tolerance-px 75

Steps:
1. Read the confirmed pair (label paths + ground-truth transform) from the
   Phase 0 manifest.
2. Parse both labels for geometry; compute the TMC-2 crop window from the
   ground-truth projection of the OHRC corners (with generous padding) so
   SIFT never searches the whole 223088x4000 strip.
3. Load imagery, downsample OHRC toward TMC-2 resolution (scale_pyramid),
4. SIFT + Lowe ratio + RANSAC (sift_baseline).
5. Score inlier correspondences against ground truth (scorer), write
   preview/overlay PNGs and a results JSON, print a final summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

try:  # package-style import when used as a module...
    from .csv_geolocation import (
        fit_local_transforms,
        locate_region,
        read_geolocation_csv,
    )
    from .groundtruth import PixelTransform, transform_from_manifest_dict
    from .image_loader import load_image, save_preview, to_uint8
    from .overlap_finder import footprint_polygon
    from .pds4_parser import parse_label
    from .scale_pyramid import downsample_to_match
    from .scorer import format_report, score_correspondences
    from .sift_baseline import run_sift_matching
except ImportError:  # ...and flat import when run directly as a script
    from csv_geolocation import (  # type: ignore[no-redef]
        fit_local_transforms,
        locate_region,
        read_geolocation_csv,
    )
    from groundtruth import PixelTransform, transform_from_manifest_dict  # type: ignore[no-redef]
    from image_loader import load_image, save_preview, to_uint8  # type: ignore[no-redef]
    from overlap_finder import footprint_polygon  # type: ignore[no-redef]
    from pds4_parser import parse_label  # type: ignore[no-redef]
    from scale_pyramid import downsample_to_match  # type: ignore[no-redef]
    from scorer import format_report, score_correspondences  # type: ignore[no-redef]
    from sift_baseline import run_sift_matching  # type: ignore[no-redef]

logger = logging.getLogger("lunar_data_pipeline.baseline")


def compute_crop_window(
    transform,
    ohrc_lines: int,
    ohrc_samples: int,
    tmc_lines: int,
    tmc_samples: int,
    padding_px: int,
) -> tuple[int, int, int, int]:
    """TMC-2 pixel window covering the projected OHRC footprint + padding."""
    corners = np.array(
        [
            [0.0, 0.0],
            [float(ohrc_samples - 1), 0.0],
            [float(ohrc_samples - 1), float(ohrc_lines - 1)],
            [0.0, float(ohrc_lines - 1)],
            [ohrc_samples / 2.0, ohrc_lines / 2.0],
        ]
    )
    projected = transform.project_many(corners)
    x0 = max(int(np.floor(projected[:, 0].min())) - padding_px, 0)
    x1 = min(int(np.ceil(projected[:, 0].max())) + padding_px, tmc_samples)
    y0 = max(int(np.floor(projected[:, 1].min())) - padding_px, 0)
    y1 = min(int(np.ceil(projected[:, 1].max())) + padding_px, tmc_lines)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(f"degenerate crop window ({x0},{y0})-({x1},{y1})")
    return x0, y0, x1, y1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SIFT+RANSAC baseline scored against the manifest ground truth."
    )
    parser.add_argument("--manifest", type=Path, default=Path("pair_manifest.json"))
    parser.add_argument("--pair-index", type=int, default=0,
                        help="Which pair entry from the manifest (default: first).")
    parser.add_argument("--tolerance-px", type=float, default=75.0,
                        help="Ground-truth correctness tolerance in TMC-2 px.")
    parser.add_argument("--target-ratio", type=float, default=None,
                        help="Force OHRC->TMC downsample factor; default: auto "
                        "from the labels' m/px values.")
    parser.add_argument("--ratio-test", type=float, default=0.7)
    parser.add_argument("--ransac-thresh", type=float, default=8.0,
                        help="RANSAC reprojection threshold in adjusted-space px.")
    parser.add_argument("--crop-padding", type=int, default=500,
                        help="Padding around the projected footprint, TMC px.")
    parser.add_argument("--nfeatures", type=int, default=20000)
    parser.add_argument("--max-draw", type=int, default=200,
                        help="Max matches drawn on the overlay PNG.")
    parser.add_argument("--output-dir", type=Path, default=Path("baseline_output"))
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        return 2
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs = manifest.get("pairs", [])
    if not pairs:
        logger.error("Manifest contains no confirmed pairs")
        return 2
    pair = pairs[args.pair_index]
    gt_entry = pair.get("ground_truth", {})
    if gt_entry.get("method") != "corner_homography":
        logger.error("Pair %s lacks a usable ground truth entry", pair.get("pair_id"))
        return 2

    ohrc_label = Path(pair["ohrc"]["label_path"])
    tmc_label = Path(pair["tmc2"]["label_path"])
    ohrc_product = parse_label(ohrc_label)
    tmc_product = parse_label(tmc_label)
    if ohrc_product is None or tmc_product is None:
        logger.error("Failed to parse labels: %s / %s", ohrc_label, tmc_label)
        return 2
    if not ohrc_product.file_path.exists() or not tmc_product.file_path.exists():
        logger.error("Image files missing on disk")
        return 2

    transform = transform_from_manifest_dict(gt_entry)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Crop window + ground truth ------------------------------------------
    # The corner homography misplaces content by thousands of pixels along a
    # full TMC-2 strip (documented Phase 0 limitation). When the TMC product
    # ships a per-pixel geolocation CSV we use it to (a) find the true crop
    # window and (b) fit a LOCAL geo->pixel transform for scoring.
    refined_transform = transform
    ground_truth_mode = "corner_homography"
    tmc_grid = None

    if tmc_product.geometry_csv_path and tmc_product.geometry_csv_path.exists():
        tmc_grid = read_geolocation_csv(tmc_product.geometry_csv_path)
        ohrc_polygon = footprint_polygon(ohrc_product)
        if ohrc_polygon is None:
            logger.error("OHRC product has no usable footprint polygon")
            return 2
        try:
            scan_rng, pix_rng, _ = locate_region(tmc_grid, ohrc_polygon)
        except ValueError as exc:
            logger.error("CSV geolocation cannot locate OHRC region: %s", exc)
            return 2

        s0 = max(scan_rng[0] - args.crop_padding, 0)
        s1 = min(scan_rng[1] + args.crop_padding, tmc_product.lines)
        c0 = max(pix_rng[0] - args.crop_padding, 0)
        c1 = min(pix_rng[1] + args.crop_padding, tmc_product.samples)
        x0, y0, x1, y1 = c0, s0, c1, s1

        h_px_to_geo, h_geo_to_px, fit_points = fit_local_transforms(
            tmc_grid, (y0, y1), (x0, x1)
        )
        refined_transform = PixelTransform(
            product_a_id=ohrc_product.product_id,
            product_b_id=tmc_product.product_id,
            h_pixel_to_geo_a=transform.h_pixel_to_geo_a,  # OHRC corner fit (small footprint)
            h_geo_to_pixel_b=h_geo_to_px,  # CSV-fitted local TMC mapping
        )
        ground_truth_mode = "csv_refined_local_homography"
        logger.warning(
            "CROP (CSV-refined): TMC-2 strip restricted to cols [%d:%d] rows "
            "[%d:%d] (%dx%d of %dx%d; padding %d px). Ground truth for scoring "
            "uses a local homography fitted to %d CSV grid points.",
            x0, x1, y0, y1,
            x1 - x0, y1 - y0,
            tmc_product.samples, tmc_product.lines,
            args.crop_padding,
            fit_points,
        )
    else:
        x0, y0, x1, y1 = compute_crop_window(
            transform,
            ohrc_product.lines,
            ohrc_product.samples,
            tmc_product.lines,
            tmc_product.samples,
            padding_px=args.crop_padding,
        )
        logger.warning(
            "CROP (corner-homography fallback; no TMC geolocation CSV found): "
            "cols [%d:%d] rows [%d:%d] (%dx%d of %dx%d; padding %d px).",
            x0, x1, y0, y1,
            x1 - x0, y1 - y0,
            tmc_product.samples, tmc_product.lines,
            args.crop_padding,
        )

    # --- Load and prepare images ------------------------------------------
    logger.info("Loading OHRC image (%d MB raw)...",
                ohrc_product.file_path.stat().st_size // (1024 * 1024))
    ohrc_img = load_image(ohrc_product)  # uint8 already
    ohrc_scaled = downsample_to_match(
        ohrc_img,
        source_resolution_m=ohrc_product.pixel_resolution_m,
        target_resolution_m=tmc_product.pixel_resolution_m,
        factor_override=args.target_ratio,
    )
    del ohrc_img

    logger.info("Loading TMC-2 crop...")
    tmc_crop_raw = load_image(tmc_product, row_range=(y0, y1), col_range=(x0, x1))
    tmc_crop = to_uint8(tmc_crop_raw)
    del tmc_crop_raw
    tmc_scaled = downsample_to_match(
        tmc_crop,
        source_resolution_m=tmc_product.pixel_resolution_m,
        target_resolution_m=tmc_product.pixel_resolution_m,  # no-op, keeps bookkeeping
    )

    save_preview(ohrc_scaled.image, args.output_dir / "preview_ohrc_adjusted.png")
    save_preview(tmc_scaled.image, args.output_dir / "preview_tmc_crop.png")

    # --- Match -------------------------------------------------------------
    match_result = run_sift_matching(
        ohrc_scaled,
        tmc_scaled,
        ratio_thresh=args.ratio_test,
        ransac_reproj_thresh=args.ransac_thresh,
        nfeatures=args.nfeatures,
    )

    # --- Score ---------------------------------------------------------------
    # --- Score ---------------------------------------------------------------
    # Primary scoring uses the best available ground truth (CSV-refined when
    # possible). The raw corner-homography manifest GT is scored too so the
    # refinement's impact is visible.
    report_inliers = score_correspondences(
        match_result.points_a_original[match_result.inlier_indices],
        match_result.points_b_original[match_result.inlier_indices],
        refined_transform,
        tolerance_px=args.tolerance_px,
    )
    report_all = score_correspondences(
        match_result.points_a_original,
        match_result.points_b_original,
        refined_transform,
        tolerance_px=args.tolerance_px,
    )
    if ground_truth_mode == "csv_refined_local_homography":
        report_corner_inliers = score_correspondences(
            match_result.points_a_original[match_result.inlier_indices],
            match_result.points_b_original[match_result.inlier_indices],
            transform,  # raw manifest corner homography
            tolerance_px=args.tolerance_px,
        )
    else:
        report_corner_inliers = None

    # --- Visualisation -------------------------------------------------------
    if match_result.good_matches:
        draw_n = min(args.max_draw, len(match_result.good_matches))
        overlay = cv2.drawMatches(
            ohrc_scaled.image,
            match_result.keypoints_a_adj,
            tmc_scaled.image,
            match_result.keypoints_b_adj,
            match_result.good_matches[:draw_n],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        overlay_path = args.output_dir / "matches_overlay.png"
        save_preview(overlay, overlay_path, max_dim=2400)
    else:
        overlay_path = None

    results = {
        "pair_id": pair.get("pair_id"),
        "ground_truth_mode": ground_truth_mode,
        "parameters": {
            "tolerance_px": args.tolerance_px,
            "downsample_factor_ohrc": ohrc_scaled.downsample_factor,
            "ratio_test": args.ratio_test,
            "ransac_reproj_thresh_adj_px": args.ransac_thresh,
            "crop_padding_tmc_px": args.crop_padding,
            "nfeatures": args.nfeatures,
        },
        "tmc_crop_window_cols": [x0, x1],
        "tmc_crop_window_rows": [y0, y1],
        "keypoints": {
            "ohrc": match_result.keypoint_count_a,
            "tmc2": match_result.keypoint_count_b,
        },
        "raw_matches": match_result.raw_match_count,
        "ratio_test_matches": match_result.ratio_test_matches,
        "ransac_inliers": match_result.inlier_count,
        "homography_original_ohrc_to_tmc": (
            None
            if match_result.homography_original is None
            else [[float(v) for v in row] for row in match_result.homography_original]
        ),
        "score_all_matches": report_all.summary_dict(),
        "score_ransac_inliers": report_inliers.summary_dict(),
        "score_ransac_inliers_vs_corner_gt": (
            report_corner_inliers.summary_dict()
            if report_corner_inliers is not None
            else None
        ),
        "artifacts": {
            "overlay_png": str(overlay_path) if overlay_path else None,
            "preview_ohrc": str(args.output_dir / "preview_ohrc_adjusted.png"),
            "preview_tmc_crop": str(args.output_dir / "preview_tmc_crop.png"),
        },
    }
    results_path = args.output_dir / "baseline_results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")

    # --- Final summary (always printed) --------------------------------------
    print("=" * 72)
    print(f"Baseline result for pair: {pair.get('pair_id')}")
    print(
        f"Keypoints: OHRC={match_result.keypoint_count_a:,} "
        f"TMC2={match_result.keypoint_count_b:,}"
    )
    print(
        f"Matches: raw={match_result.raw_match_count:,} "
        f"after-ratio-test={match_result.ratio_test_matches:,} "
        f"RANSAC-inliers={match_result.inlier_count:,}"
    )
    print(format_report(report_inliers))
    if report_corner_inliers is not None and report_corner_inliers.total_scored:
        print(
            f"(Same inliers scored against the raw corner-homography manifest GT: "
            f"{report_corner_inliers.correct}/{report_corner_inliers.total_scored} "
            f"correct -- shows why CSV refinement matters.)"
        )
    if match_result.ratio_test_matches > 0:
        print(
            f"(All {report_all.total_scored} ratio-test matches incl. outliers: "
            f"{report_all.correct}/{report_all.total_scored} correct = "
            f"{report_all.percent_correct:.1f}% within tolerance)"
        )
    print(f"Overlay PNG : {overlay_path}")
    print(f"Results JSON: {results_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
