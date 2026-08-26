"""Run both SIFT and RIFT2 on the same cropped image pair and print a
side-by-side comparison table.

Usage:
    python run_comparison.py --manifest pair_manifest.json --tolerance-px 75

Uses the exact same CSV-refined cropping logic as run_baseline.py.
Outputs:
- comparison_overlay_sift.png   (SIFT matches overlay)
- comparison_overlay_rift.png   (RIFT matches overlay)
- comparison_results.json       (structured comparison data)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from .csv_geolocation import (
        fit_local_transforms,
        locate_region,
        read_geolocation_csv,
    )
    from .groundtruth import PixelTransform, transform_from_manifest_dict
    from .image_loader import load_image, save_preview, to_uint8
    from .overlap_finder import footprint_polygon
    from .pds4_parser import parse_label
    from .rift_baseline import run_rift_matching
    from .scale_pyramid import downsample_to_match
    from .scorer import format_report, score_correspondences
    from .sift_baseline import run_sift_matching
except ImportError:
    from csv_geolocation import (  # type: ignore[no-redef]
        fit_local_transforms,
        locate_region,
        read_geolocation_csv,
    )
    from groundtruth import PixelTransform, transform_from_manifest_dict  # type: ignore[no-redef]
    from image_loader import load_image, save_preview, to_uint8  # type: ignore[no-redef]
    from overlap_finder import footprint_polygon  # type: ignore[no-redef]
    from pds4_parser import parse_label  # type: ignore[no-redef]
    from rift_baseline import run_rift_matching  # type: ignore[no-redef]
    from scale_pyramid import downsample_to_match  # type: ignore[no-redef]
    from scorer import format_report, score_correspondences  # type: ignore[no-redef]
    from sift_baseline import run_sift_matching  # type: ignore[no-redef]

logger = logging.getLogger("lunar_data_pipeline.comparison")


def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.1f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SIFT vs RIFT2 side-by-side comparison on the same crop."
    )
    parser.add_argument("--manifest", type=Path, default=Path("pair_manifest.json"))
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--tolerance-px", type=float, default=75.0)
    parser.add_argument("--target-ratio", type=float, default=None)
    parser.add_argument("--ratio-test-sift", type=float, default=0.7)
    parser.add_argument("--ratio-test-rift", type=float, default=0.75)
    parser.add_argument("--ransac-thresh", type=float, default=8.0)
    parser.add_argument("--ransac-thresh-rift", type=float, default=5.0)
    parser.add_argument("--crop-padding", type=int, default=500)
    parser.add_argument("--nfeatures", type=int, default=20000)
    parser.add_argument("--rift-npt", type=int, default=5000)
    parser.add_argument("--max-draw", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_output"))
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    # --- Load manifest and parse labels ---
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
        logger.error("Pair lacks a usable ground truth entry")
        return 2

    ohrc_label = Path(pair["ohrc"]["label_path"])
    tmc_label = Path(pair["tmc2"]["label_path"])
    ohrc_product = parse_label(ohrc_label)
    tmc_product = parse_label(tmc_label)
    if ohrc_product is None or tmc_product is None:
        logger.error("Failed to parse labels")
        return 2
    if not ohrc_product.file_path.exists() or not tmc_product.file_path.exists():
        logger.error("Image files missing on disk")
        return 2

    transform = transform_from_manifest_dict(gt_entry)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Crop window (identical to run_baseline.py) ---
    refined_transform = transform
    ground_truth_mode = "corner_homography"

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
            h_pixel_to_geo_a=transform.h_pixel_to_geo_a,
            h_geo_to_pixel_b=h_geo_to_px,
        )
        ground_truth_mode = "csv_refined_local_homography"
    else:
        corners = np.array([
            [0.0, 0.0],
            [float(ohrc_product.samples - 1), 0.0],
            [float(ohrc_product.samples - 1), float(ohrc_product.lines - 1)],
            [0.0, float(ohrc_product.lines - 1)],
        ])
        projected = transform.project_many(corners)
        x0 = max(int(np.floor(projected[:, 0].min())) - args.crop_padding, 0)
        x1 = min(int(np.ceil(projected[:, 0].max())) + args.crop_padding, tmc_product.samples)
        y0 = max(int(np.floor(projected[:, 1].min())) - args.crop_padding, 0)
        y1 = min(int(np.ceil(projected[:, 1].max())) + args.crop_padding, tmc_product.lines)

    # --- Load imagery (shared between both methods) ---
    ohrc_img = load_image(ohrc_product)
    ohrc_scaled = downsample_to_match(
        ohrc_img,
        source_resolution_m=ohrc_product.pixel_resolution_m,
        target_resolution_m=tmc_product.pixel_resolution_m,
        factor_override=args.target_ratio,
    )
    del ohrc_img

    tmc_crop_raw = load_image(tmc_product, row_range=(y0, y1), col_range=(x0, x1))
    tmc_crop = to_uint8(tmc_crop_raw)
    del tmc_crop_raw
    tmc_scaled = downsample_to_match(
        tmc_crop,
        source_resolution_m=tmc_product.pixel_resolution_m,
        target_resolution_m=tmc_product.pixel_resolution_m,
    )

    save_preview(ohrc_scaled.image, args.output_dir / "preview_ohrc_adjusted.png")
    save_preview(tmc_scaled.image, args.output_dir / "preview_tmc_crop.png")

    # --- Run SIFT ---
    print("\n" + "=" * 72)
    print("Running SIFT...")
    print("=" * 72)
    t_sift_start = time.time()
    sift_result = run_sift_matching(
        ohrc_scaled,
        tmc_scaled,
        ratio_thresh=args.ratio_test_sift,
        ransac_reproj_thresh=args.ransac_thresh,
        nfeatures=args.nfeatures,
    )
    t_sift_total = time.time() - t_sift_start

    sift_inliers_report = score_correspondences(
        sift_result.points_a_original[sift_result.inlier_indices],
        sift_result.points_b_original[sift_result.inlier_indices],
        refined_transform,
        tolerance_px=args.tolerance_px,
    )
    sift_all_report = score_correspondences(
        sift_result.points_a_original,
        sift_result.points_b_original,
        refined_transform,
        tolerance_px=args.tolerance_px,
    )

    # --- Run RIFT2 ---
    print("\n" + "=" * 72)
    print("Running RIFT2...")
    print("=" * 72)
    t_rift_start = time.time()
    rift_result = run_rift_matching(
        ohrc_scaled,
        tmc_scaled,
        ratio_thresh=args.ratio_test_rift,
        ransac_reproj_thresh=args.ransac_thresh_rift,
        npt=args.rift_npt,
    )
    t_rift_total = time.time() - t_rift_start

    rift_inliers_report = score_correspondences(
        rift_result.points_a_original[rift_result.inlier_indices],
        rift_result.points_b_original[rift_result.inlier_indices],
        refined_transform,
        tolerance_px=args.tolerance_px,
    )
    rift_all_report = score_correspondences(
        rift_result.points_a_original,
        rift_result.points_b_original,
        refined_transform,
        tolerance_px=args.tolerance_px,
    )

    # --- Visualisation ---
    sift_overlay_path = None
    rift_overlay_path = None

    if sift_result.good_matches:
        draw_n = min(args.max_draw, len(sift_result.good_matches))
        overlay = cv2.drawMatches(
            ohrc_scaled.image, sift_result.keypoints_a_adj,
            tmc_scaled.image, sift_result.keypoints_b_adj,
            sift_result.good_matches[:draw_n], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        sift_overlay_path = args.output_dir / "comparison_overlay_sift.png"
        save_preview(overlay, sift_overlay_path, max_dim=2400)

    if rift_result.good_matches:
        draw_n = min(args.max_draw, len(rift_result.good_matches))
        overlay = cv2.drawMatches(
            ohrc_scaled.image, rift_result.keypoints_a_adj,
            tmc_scaled.image, rift_result.keypoints_b_adj,
            rift_result.good_matches[:draw_n], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        rift_overlay_path = args.output_dir / "comparison_overlay_rift.png"
        save_preview(overlay, rift_overlay_path, max_dim=2400)

    # --- Side-by-side table ---
    print("\n" + "=" * 72)
    print("SIFT vs RIFT2 Comparison")
    print(f"Pair: {pair.get('pair_id')}")
    print(f"Ground truth mode: {ground_truth_mode}")
    print(f"TMC crop: cols [{x0}:{x1}] rows [{y0}:{y1}]")
    print("=" * 72)

    hdr = f"{'Metric':<42} {'SIFT':>14} {'RIFT2':>14}"
    sep = "-" * 72
    print(hdr)
    print(sep)
    print(f"{'Keypoints (image A / B)':<42} {sift_result.keypoint_count_a:>6,}/{sift_result.keypoint_count_b:<6,} {rift_result.keypoint_count_a:>6,}/{rift_result.keypoint_count_b:<6,}")
    print(f"{'Raw knn pairs':<42} {sift_result.raw_match_count:>14,} {rift_result.raw_match_count:>14,}")
    print(f"{'Ratio-test matches':<42} {sift_result.ratio_test_matches:>14,} {rift_result.ratio_test_matches:>14,}")
    print(f"{'RANSAC/MAGSAC inliers':<42} {sift_result.inlier_count:>14,} {rift_result.inlier_count:>14,}")
    print(sep)

    def _score_line(label: str, report) -> str:
        if report.total_scored == 0:
            return f"{label:<42} {'N/A':>14} {'N/A':>14}"
        return f"{label:<42} {report.correct}/{report.total_scored} ({report.percent_correct:4.1f}%) {report.correct}/{report.total_scored} ({report.percent_correct:4.1f}%)"

    print(_score_line("Correct vs GT (inliers)", sift_inliers_report))
    print(_score_line("Correct vs GT (all matches)", sift_all_report))
    print(sep)

    print(f"{'Total wall time':<42} {_fmt_time(t_sift_total):>14} {_fmt_time(t_rift_total):>14}")
    if rift_result.time_detect_s > 0:
        print(f"  RIFT2 detect+describe: {_fmt_time(rift_result.time_detect_s)}")
        print(f"  RIFT2 match:          {_fmt_time(rift_result.time_match_s)}")
        print(f"  RIFT2 MAGSAC:         {_fmt_time(rift_result.time_ransac_s)}")
    print(sep)

    print(f"Overlay SIFT : {sift_overlay_path}")
    print(f"Overlay RIFT : {rift_overlay_path}")
    print("=" * 72)

    # --- Write JSON results ---
    results = {
        "pair_id": pair.get("pair_id"),
        "ground_truth_mode": ground_truth_mode,
        "tmc_crop_window": {"cols": [x0, x1], "rows": [y0, y1]},
        "tolerance_px": args.tolerance_px,
        "sift": {
            "keypoints_a": sift_result.keypoint_count_a,
            "keypoints_b": sift_result.keypoint_count_b,
            "raw_matches": sift_result.raw_match_count,
            "ratio_test_matches": sift_result.ratio_test_matches,
            "ransac_inliers": sift_result.inlier_count,
            "correct_vs_gt_inliers": sift_inliers_report.summary_dict(),
            "correct_vs_gt_all": sift_all_report.summary_dict(),
            "total_time_s": round(t_sift_total, 3),
            "overlay_path": str(sift_overlay_path),
        },
        "rift": {
            "keypoints_a": rift_result.keypoint_count_a,
            "keypoints_b": rift_result.keypoint_count_b,
            "raw_matches": rift_result.raw_match_count,
            "ratio_test_matches": rift_result.ratio_test_matches,
            "ransac_inliers": rift_result.inlier_count,
            "correct_vs_gt_inliers": rift_inliers_report.summary_dict(),
            "correct_vs_gt_all": rift_all_report.summary_dict(),
            "total_time_s": round(t_rift_total, 3),
            "rift_detect_time_s": round(rift_result.time_detect_s, 3),
            "rift_match_time_s": round(rift_result.time_match_s, 3),
            "rift_ransac_time_s": round(rift_result.time_ransac_s, 3),
            "overlay_path": str(rift_overlay_path),
        },
    }

    results_path = args.output_dir / "comparison_results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")
    print(f"Results JSON: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
