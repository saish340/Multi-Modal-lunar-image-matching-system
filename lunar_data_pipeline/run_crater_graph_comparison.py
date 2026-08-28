"""Run SIFT, RIFT2, and crater-neighborhood-graph matching side by side on the
same cropped OHRC/TMC-2 pair, score all three with scorer.py, and print one
final comparison table.

Usage:
    python run_crater_graph_comparison.py --manifest pair_manifest.json \
        --tolerance-px 75

This is the Phase 4 centerpiece: the three methods are scored with IDENTICAL
tolerance settings against the same CSV-refined ground-truth transform, so the
resulting table is directly apples-to-apples with the Phase 1 (SIFT) and
Phase 2 (RIFT2) 0% baselines.
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
    from .crater_detector import detect_craters, params_for_resolution
    from .crater_graph_baseline import run_crater_graph_matching
    from .csv_geolocation import fit_local_transforms, locate_region, read_geolocation_csv
    from .groundtruth import PixelTransform, transform_from_manifest_dict
    from .image_loader import load_image, save_preview, to_uint8
    from .overlap_finder import footprint_polygon
    from .pds4_parser import parse_label
    from .rift_baseline import run_rift_matching
    from .scale_pyramid import downsample_to_match
    from .scorer import format_report, score_correspondences
    from .sift_baseline import run_sift_matching
except ImportError:
    from crater_detector import detect_craters, params_for_resolution  # type: ignore[no-redef]
    from crater_graph_baseline import run_crater_graph_matching  # type: ignore[no-redef]
    from csv_geolocation import fit_local_transforms, locate_region, read_geolocation_csv  # type: ignore[no-redef]
    from groundtruth import PixelTransform, transform_from_manifest_dict  # type: ignore[no-redef]
    from image_loader import load_image, save_preview, to_uint8  # type: ignore[no-redef]
    from overlap_finder import footprint_polygon  # type: ignore[no-redef]
    from pds4_parser import parse_label  # type: ignore[no-redef]
    from rift_baseline import run_rift_matching  # type: ignore[no-redef]
    from scale_pyramid import downsample_to_match  # type: ignore[no-redef]
    from scorer import format_report, score_correspondences  # type: ignore[no-redef]
    from sift_baseline import run_sift_matching  # type: ignore[no-redef]

logger = logging.getLogger("lunar_data_pipeline.crater_graph_comparison")


def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.1f}s"


def _error_bands(errors_px) -> list[dict]:
    """Bucket errors into tolerance-class bands for honest near/far characterisation."""
    edges = [0, 75, 150, 300, 600, 1200, 2500, float("inf")]
    labels = ["<=75", "75-150", "150-300", "300-600", "600-1200", "1200-2500", ">2500"]
    counts = np.histogram(errors_px, bins=edges)[0]
    return [{"band": lab, "count": int(c)} for lab, c in zip(labels, counts)]


def _save_npz(path, det_a, det_b):
    def col(dets, attr):
        return np.array([getattr(x, attr) for x in dets])
    np.savez(path,
             a_cx=col(det_a, "cx"), a_cy=col(det_a, "cy"),
             a_r=col(det_a, "radius"), a_conf=col(det_a, "confidence"),
             b_cx=col(det_b, "cx"), b_cy=col(det_b, "cy"),
             b_r=col(det_b, "radius"), b_conf=col(det_b, "confidence"))
    logger.info("Saved crater detections to %s", path)


def _from_npz(d, prefix):
    from .crater_detector import CraterDetection
    return [
        CraterDetection(cx=float(d[f"{prefix}_cx"][i]),
                        cy=float(d[f"{prefix}_cy"][i]),
                        radius=float(d[f"{prefix}_r"][i]),
                        confidence=float(d[f"{prefix}_conf"][i]))
        for i in range(d[f"{prefix}_cx"].shape[0])
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SIFT vs RIFT2 vs crater-graph side-by-side comparison."
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
    # crater-graph knobs
    parser.add_argument("--graph-k", type=int, default=4, help="Graph neighbours per node")
    parser.add_argument("--graph-min-radius-m", type=float, default=300.0,
                        help="Drop detections below this physical radius (m) when building graphs")
    parser.add_argument("--graph-ratio-test", type=float, default=0.6)
    parser.add_argument("--graph-min-similarity", type=float, default=0.25)
    parser.add_argument("--graph-sigma", type=float, default=0.5)
    parser.add_argument("--graph-ransac-thresh", type=float, default=15.0)
    parser.add_argument("--graph-consistency", action="store_true",
                        help="Enable the (lossy) graph-consistency pruning pass; "
                             "OFF by default to report the simple match->RANSAC result")
    parser.add_argument("--graph-max-nodes-b", type=int, default=None,
                        help="Upper bound on B nodes (perf safety cap); None = all")
    parser.add_argument("--graph-matcher", default="neighbor_identity",
                        choices=["neighbor_identity", "spectrum"],
                        help="Crater graph matcher (mutual-neighbor-identity vs old sorted-spectrum)")
    parser.add_argument("--graph-n-candidates", type=int, default=8,
                        help="Shortlist size per A node (first-pass filter)")
    parser.add_argument("--graph-min-spectrum-sim", type=float, default=0.1)
    parser.add_argument("--graph-neighbor-d-tol", type=float, default=0.6,
                        help="Log-distance-ratio tolerance for neighbour identity match")
    parser.add_argument("--graph-neighbor-s-tol", type=float, default=0.6,
                        help="Size-ratio tolerance for neighbour identity match")
    parser.add_argument("--graph-min-mutual-fraction", type=float, default=0.5,
                        help="Fraction of A's neighbours that must find B partners")
    parser.add_argument("--graph-neighbor-physical", action="store_true",
                        help="Match neighbours by physical distance (m) not ratio")
    parser.add_argument("--graph-no-ratio-test", action="store_true",
                        help="Disable the Lowe-style ratio test in mutual matching")
    parser.add_argument("--output-dir", type=Path, default=Path("crater_graph_comparison_output"))
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
        logger.error("Pair lacks a usable ground truth entry")
        return 2

    ohrc_label = Path(pair["ohrc"]["label_path"])
    tmc_label = Path(pair["tmc2"]["label_path"])
    ohrc_product = parse_label(ohrc_label)
    tmc_product = parse_label(tmc_label)
    if ohrc_product is None or tmc_product is None:
        logger.error("Failed to parse labels")
        return 2

    transform = transform_from_manifest_dict(gt_entry)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Crop window + refined local ground truth (identical to run_comparison) ---
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
        h_px_to_geo, h_geo_to_px, _ = fit_local_transforms(tmc_grid, (y0, y1), (x0, x1))
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

    # --- Load imagery (shared) ---
    t_load = time.time()
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
    logger.info("Images loaded: OHRC adj %s, TMC crop %s [%.1fs]",
                ohrc_scaled.image.shape, tmc_crop.shape, time.time() - t_load)

    # --- Crater detection (shared, both at ~TMC-2 resolution) ---
    params = params_for_resolution(tmc_product.pixel_resolution_m)
    cache_path = args.output_dir / "crater_detections.npz"
    det_a = det_b = None
    if cache_path.exists():
        try:
            c = np.load(cache_path, allow_pickle=False)
            det_a = _from_npz(c, "a")
            det_b = _from_npz(c, "b")
            logger.info("Loaded cached crater detections: A=%d B=%d",
                        len(det_a), len(det_b))
        except Exception as exc:
            logger.warning("Cache read failed (%s); re-detecting", exc)
            det_a = det_b = None
    if det_a is None or det_b is None:
        t_det = time.time()
        det_a = detect_craters(ohrc_scaled.image, params)
        det_b = detect_craters(tmc_crop, params)
        logger.info("Crater detection: A=%d B=%d [%.1fs]",
                    len(det_a), len(det_b), time.time() - t_det)
        _save_npz(cache_path, det_a, det_b)

    # --- Run SIFT ---
    print("\n" + "=" * 74 + "\nRunning SIFT...\n" + "=" * 74)
    t0 = time.time()
    sift_result = run_sift_matching(
        ohrc_scaled, tmc_scaled,
        ratio_thresh=args.ratio_test_sift, ransac_reproj_thresh=args.ransac_thresh,
        nfeatures=args.nfeatures,
    )
    t_sift = time.time() - t0

    # --- Run RIFT2 ---
    print("\n" + "=" * 74 + "\nRunning RIFT2...\n" + "=" * 74)
    t0 = time.time()
    rift_result = run_rift_matching(
        ohrc_scaled, tmc_scaled,
        ratio_thresh=args.ratio_test_rift, ransac_reproj_thresh=args.ransac_thresh_rift,
        npt=args.rift_npt,
    )
    t_rift = time.time() - t0

    # --- Run crater-graph ---
    print("\n" + "=" * 74 + "\nRunning crater-neighborhood-graph matching...\n" + "=" * 74)
    t0 = time.time()
    graph_result = run_crater_graph_matching(
        det_a, det_b,
        meters_per_px_a=tmc_product.pixel_resolution_m,
        meters_per_px_b=tmc_product.pixel_resolution_m,
        ohrc_downsample_factor=ohrc_scaled.downsample_factor,
        tmc_crop_offset=(float(c0), float(s0)),
        k=args.graph_k,
        min_radius_m=args.graph_min_radius_m,
        ratio_test=args.graph_ratio_test,
        min_similarity=args.graph_min_similarity,
        sigma=args.graph_sigma,
        max_nodes_b=args.graph_max_nodes_b,
        ransac_reproj_thresh=args.graph_ransac_thresh,
        use_consistency=args.graph_consistency,
        matcher=args.graph_matcher,
        n_candidates_per_node=args.graph_n_candidates,
        min_spectrum_similarity=args.graph_min_spectrum_sim,
        neighbor_d_tol=args.graph_neighbor_d_tol,
        neighbor_s_tol=args.graph_neighbor_s_tol,
        min_mutual_fraction=args.graph_min_mutual_fraction,
        neighbor_physical=args.graph_neighbor_physical,
        neighbor_use_ratio_test=not args.graph_no_ratio_test,
    )
    t_graph = time.time() - t0

    # --- Score all three with identical settings ---
    sift_inl = score_correspondences(
        sift_result.points_a_original[sift_result.inlier_indices],
        sift_result.points_b_original[sift_result.inlier_indices],
        refined_transform, tolerance_px=args.tolerance_px)
    sift_all = score_correspondences(
        sift_result.points_a_original, sift_result.points_b_original,
        refined_transform, tolerance_px=args.tolerance_px)

    rift_inl = score_correspondences(
        rift_result.points_a_original[rift_result.inlier_indices],
        rift_result.points_b_original[rift_result.inlier_indices],
        refined_transform, tolerance_px=args.tolerance_px)
    rift_all = score_correspondences(
        rift_result.points_a_original, rift_result.points_b_original,
        refined_transform, tolerance_px=args.tolerance_px)

    graph_inl = score_correspondences(
        graph_result.points_a_original[graph_result.inlier_indices],
        graph_result.points_b_original[graph_result.inlier_indices],
        refined_transform, tolerance_px=args.tolerance_px)
    graph_all = score_correspondences(
        graph_result.points_a_original, graph_result.points_b_original,
        refined_transform, tolerance_px=args.tolerance_px)

    # --- Visualisation overlays ---
    _draw_crater_graph_overlay(
        ohrc_scaled.image, tmc_crop, graph_result,
        args.output_dir / "overlay_crater_graph.png", max_draw=args.max_draw)
    if sift_result.good_matches:
        ov = cv2.drawMatches(ohrc_scaled.image, sift_result.keypoints_a_adj,
                             tmc_scaled.image, sift_result.keypoints_b_adj,
                             sift_result.good_matches[:args.max_draw], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        save_preview(ov, args.output_dir / "overlay_sift.png", max_dim=2400)
    if rift_result.good_matches:
        ov = cv2.drawMatches(ohrc_scaled.image, rift_result.keypoints_a_adj,
                             tmc_scaled.image, rift_result.keypoints_b_adj,
                             rift_result.good_matches[:args.max_draw], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        save_preview(ov, args.output_dir / "overlay_rift.png", max_dim=2400)

    # --- Final table ---
    print("\n" + "=" * 74)
    print("FINAL COMPARISON: SIFT vs RIFT2 vs CRATER-GRAPH")
    print(f"Pair: {pair.get('pair_id')}")
    print(f"Ground truth mode: {ground_truth_mode}")
    print(f"TMC crop: cols [{x0}:{x1}] rows [{y0}:{y1}]  | tolerance {args.tolerance_px:g} px")
    print("=" * 74)

    hdr = f"{'Metric':<40} {'SIFT':>12} {'RIFT2':>12} {'CraterGraph':>14}"
    print(hdr)
    print("-" * 78)
    print(f"{'Detected/matched features':<40}"
          f"{sift_result.keypoint_count_a:>6,}/{sift_result.keypoint_count_b:<6,}"
          f"{rift_result.keypoint_count_a:>6,}/{rift_result.keypoint_count_b:<6,}"
          f"{graph_result.node_count_a:>6,}/{graph_result.node_count_b:<7,}")
    print(f"{'Raw matches':<40}{sift_result.raw_match_count:>12,}{rift_result.raw_match_count:>12,}{graph_result.raw_match_count:>14,}")
    print(f"{'Ratio-test / consistency survivors':<40}{sift_result.ratio_test_matches:>12,}{rift_result.ratio_test_matches:>12,}{graph_result.consistency_match_count:>14,}")
    print(f"{'RANSAC inliers':<40}{sift_result.inlier_count:>12,}{rift_result.inlier_count:>12,}{graph_result.inlier_count:>14,}")
    print("-" * 78)

    def _score_line(label, a, b, g):
        def cell(r):
            return "N/A" if r.total_scored == 0 else f"{r.correct}/{r.total_scored} ({r.percent_correct:.1f}%)"
        print(f"{label:<40}{cell(a):>12}{cell(b):>12}{cell(g):>15}")

    _score_line("Correct vs GT (inliers)", sift_inl, rift_inl, graph_inl)
    _score_line("Correct vs GT (all matches)", sift_all, rift_all, graph_all)
    bands = _error_bands(graph_all.errors_px)
    print("-" * 78)
    print("Crater-graph raw-match error distribution (px): "
          + "  ".join(f"{b['band']}:{b['count']}" for b in bands))
    if graph_inl.total_scored > 0:
        inb = _error_bands(graph_inl.errors_px)
        print("Crater-graph RANSAC-inlier error distribution (px): "
              + "  ".join(f"{b['band']}:{b['count']}" for b in inb))
    print("-" * 78)
    print(f"{'Total wall time':<40}{_fmt_time(t_sift):>12}{_fmt_time(t_rift):>12}{_fmt_time(t_graph):>15}")
    print("=" * 74)
    print(f"\nOverlay crater-graph : {args.output_dir / 'overlay_crater_graph.png'}")
    print(f"Overlay SIFT         : {args.output_dir / 'overlay_sift.png'}")
    print(f"Overlay RIFT2        : {args.output_dir / 'overlay_rift.png'}")
    print("=" * 74)

    results = {
        "pair_id": pair.get("pair_id"),
        "ground_truth_mode": ground_truth_mode,
        "tmc_crop_window": {"cols": [x0, x1], "rows": [y0, y1]},
        "tolerance_px": args.tolerance_px,
        "crater_detection_a": len(det_a),
        "crater_detection_b": len(det_b),
        "sift": {
            "keypoints_a/b": [sift_result.keypoint_count_a, sift_result.keypoint_count_b],
            "raw_matches": sift_result.raw_match_count,
            "ratio_test_matches": sift_result.ratio_test_matches,
            "ransac_inliers": sift_result.inlier_count,
            "correct_vs_gt_inliers": sift_inl.summary_dict(),
            "correct_vs_gt_all": sift_all.summary_dict(),
            "total_time_s": round(t_sift, 3),
        },
        "rift": {
            "keypoints_a/b": [rift_result.keypoint_count_a, rift_result.keypoint_count_b],
            "raw_matches": rift_result.raw_match_count,
            "ratio_test_matches": rift_result.ratio_test_matches,
            "ransac_inliers": rift_result.inlier_count,
            "correct_vs_gt_inliers": rift_inl.summary_dict(),
            "correct_vs_gt_all": rift_all.summary_dict(),
            "total_time_s": round(t_rift, 3),
        },
        "crater_graph": {
            "nodes_a/b": [graph_result.node_count_a, graph_result.node_count_b],
            "raw_matches": graph_result.raw_match_count,
            "consistency_matches": graph_result.consistency_match_count,
            "ransac_inliers": graph_result.inlier_count,
            "correct_vs_gt_inliers": graph_inl.summary_dict(),
            "correct_vs_gt_all": graph_all.summary_dict(),
            "error_bands_all_px": _error_bands(graph_all.errors_px),
            "error_bands_inliers_px": _error_bands(graph_inl.errors_px),
            "total_time_s": round(t_graph, 3),
            "params": {
                "k": args.graph_k,
                "min_radius_m": args.graph_min_radius_m,
                "ratio_test": args.graph_ratio_test,
                "min_similarity": args.graph_min_similarity,
                "sigma": args.graph_sigma,
                "ransac_reproj_thresh": args.graph_ransac_thresh,
                "matcher": args.graph_matcher,
                "n_candidates_per_node": args.graph_n_candidates,
                "min_spectrum_similarity": args.graph_min_spectrum_sim,
                "neighbor_d_tol": args.graph_neighbor_d_tol,
                "neighbor_s_tol": args.graph_neighbor_s_tol,
                "min_mutual_fraction": args.graph_min_mutual_fraction,
                "neighbor_physical": args.graph_neighbor_physical,
                "neighbor_use_ratio_test": not args.graph_no_ratio_test,
            },
        },
    }
    out_path = args.output_dir / "comparison_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults JSON: {out_path}")
    return 0


def _draw_crater_graph_overlay(img_a, img_b, result, out_path, max_draw=200):
    """Side-by-side overlay of crater-graph correspondences (adjusted space)."""
    try:
        ha = result.homography_adjusted
        if ha is None or result.src_adjusted.shape[0] == 0:
            save_preview(np.hstack([img_a, img_b]), out_path, max_dim=2400)
            return
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        # Resize both to a common display height so they can be side by side.
        display_h = min(h_a, h_b, 2000)
        ga = cv2.resize(img_a, (max(w_a * display_h // h_a, 1), display_h),
                        interpolation=cv2.INTER_AREA)
        gb = cv2.resize(img_b, (max(w_b * display_h // h_b, 1), display_h),
                        interpolation=cv2.INTER_AREA)
        ga_scale = w_a / ga.shape[1]
        gb_scale = w_b / gb.shape[1]
        canvas = np.hstack([ga, gb])
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        src = result.src_adjusted
        dst = result.dst_adjusted
        inl = set(result.inlier_indices.tolist())
        n = min(max_draw, len(src))
        for i in range(n):
            a = (int(src[i, 0] / ga_scale), int(src[i, 1] * display_h / h_a))
            b = (int(dst[i, 0] / gb_scale) + ga.shape[1],
                 int(dst[i, 1] * display_h / h_b))
            color = (0, 220, 0) if i in inl else (0, 0, 220)
            cv2.circle(canvas, a, 4, color, -1)
            cv2.circle(canvas, b, 4, color, -1)
            cv2.line(canvas, a, b, color, 1)
        save_preview(canvas, out_path, max_dim=2400)
    except Exception as exc:  # visualisation must never break the run
        logger.warning("Crater-graph overlay failed: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
