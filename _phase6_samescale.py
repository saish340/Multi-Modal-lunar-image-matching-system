"""Phase 6 - same-scale OHRC-OHRC SIFT sanity check, corner-homography GT (v2: cropped).

Runs SIFT on a CORRESPONDING same-scale window of two OHRC products
(~88.4% overlap, both 0.25 m/px, same day) and scores against the full-image
corner-homography ground truth from groundtruth.build_pixel_transform.

SIFT cannot run on the full 1.2 Gpx frames (OpenCV pyramid allocs ~19 GB), so
we crop a tractable 8000x8000 window on A, project it onto B via the corner
transform, crop B there, and shift the returned SIFT points back to absolute
product pixel coordinates before scoring.

Honest same-scale tolerance: 10 px of OHRC (0.25 m/px) = 2.5 m on the ground,
instead of the 75 px (TMC-2 @5.4 m/px) cross-sensor convention.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_pkg = _Path(__file__).resolve().parent / "lunar_data_pipeline"
if str(_pkg) not in _sys.path:
    _sys.path.insert(0, str(_pkg))

try:
    from groundtruth import build_pixel_transform
    from image_loader import load_image, save_preview, to_uint8
    from pds4_parser import parse_label
    from scale_pyramid import ScaledImage
    from scorer import score_correspondences
    from sift_baseline import run_sift_matching
except ImportError:  # pragma: no cover
    from lunar_data_pipeline.groundtruth import build_pixel_transform
    from lunar_data_pipeline.image_loader import load_image, save_preview, to_uint8
    from lunar_data_pipeline.pds4_parser import parse_label
    from lunar_data_pipeline.scale_pyramid import ScaledImage
    from lunar_data_pipeline.scorer import score_correspondences
    from lunar_data_pipeline.sift_baseline import run_sift_matching

A_LABEL = (
    "D:/Multi-Modal-lunar-image-matching-system/"
    "ch2_ohr_ncp_20260103T0609041371_d_img_d18/"
    "data/calibrated/20260103/ch2_ohr_ncp_20260103T0609041371_d_img_d18.xml"
)
B_LABEL = (
    "D:/Multi-Modal-lunar-image-matching-system/"
    "ch2_ohr_ncp_20260103T1005176450_d_img_d18/"
    "data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.xml"
)
OUT_DIR = Path("phase6_ohrc_ohrc_output")

# A central window to match on (samples x lines), large enough to be meaningful,
# small enough for SIFT.
A_COL0, A_COL1 = 2000, 10000          # 8000 wide
A_ROW0, A_ROW1 = 40000, 48000         # 8000 tall
PAD = 200


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a = parse_label(A_LABEL)
    b = parse_label(B_LABEL)
    assert a is not None and b is not None, "parse failed"
    assert abs(a.pixel_resolution_m - b.pixel_resolution_m) < 1e-9, "not same scale!"

    transform = build_pixel_transform(a, b)
    print(f"PAIR: {a.product_id}  vs  {b.product_id}")
    print(f"same scale: A={a.pixel_resolution_m} B={b.pixel_resolution_m} m/px (no gap)")

    # --- corresponding crop window -------------------------------------------
    a_corners = np.array([
        [A_COL0, A_ROW0], [A_COL1, A_ROW0],
        [A_COL1, A_ROW1], [A_COL0, A_ROW1],
    ], float)
    b_proj = transform.project_many(a_corners)
    b_col0 = max(int(np.floor(b_proj[:, 0].min())) - PAD, 0)
    b_col1 = min(int(np.ceil(b_proj[:, 0].max())) + PAD, b.samples)
    b_row0 = max(int(np.floor(b_proj[:, 1].min())) - PAD, 0)
    b_row1 = min(int(np.ceil(b_proj[:, 1].max())) + PAD, b.lines)
    print(f"A crop : cols[{A_COL0}:{A_COL1}] rows[{A_ROW0}:{A_ROW1}] ({A_COL1-A_COL0}x{A_ROW1-A_ROW0})")
    print(f"B crop : cols[{b_col0}:{b_col1}] rows[{b_row0}:{b_row1}] ({b_col1-b_col0}x{b_row1-b_row0})")
    if b_col1 - b_col0 < 8 or b_row1 - b_row0 < 8:
        print("B crop window degenerate - A window likely outside B footprint")
        return 2

    t0 = time.time()
    a_crop = load_image(a, row_range=(A_ROW0, A_ROW1), col_range=(A_COL0, A_COL1))
    print(f"loaded A crop {a_crop.shape} [{time.time()-t0:.1f}s]")
    t1 = time.time()
    b_crop = load_image(b, row_range=(b_row0, b_row1), col_range=(b_col0, b_col1))
    print(f"loaded B crop {b_crop.shape} [{time.time()-t1:.1f}s]")

    a_scaled = ScaledImage(image=to_uint8(a_crop), downsample_factor=1.0,
                           original_lines=a.lines, original_samples=a.samples,
                           source_resolution_m=a.pixel_resolution_m,
                           target_resolution_m=b.pixel_resolution_m,
                           downsample_factor_y=1.0)
    b_scaled = ScaledImage(image=to_uint8(b_crop), downsample_factor=1.0,
                           original_lines=b.lines, original_samples=b.samples,
                           source_resolution_m=b.pixel_resolution_m,
                           target_resolution_m=b.pixel_resolution_m,
                           downsample_factor_y=1.0)
    del a_crop, b_crop

    t2 = time.time()
    mr = run_sift_matching(a_scaled, b_scaled, ratio_thresh=0.7,
                           ransac_reproj_thresh=8.0, nfeatures=50000)
    print(f"SIFT+match done [{time.time()-t2:.0f}s]")
    print(f"keypoints A={mr.keypoint_count_a} B={mr.keypoint_count_b} "
          f"raw={mr.raw_match_count} ratio={mr.ratio_test_matches} inliers={mr.inlier_count}")

    if mr.ratio_test_matches == 0:
        print("\nRESULT: SIFT found NO ratio-test matches on the same-scale pair.")
        return 0

    # Shift crop-relative points back to absolute product coordinates.
    pa = mr.points_a_original + np.array([A_COL0, A_ROW0])
    pb = mr.points_b_original + np.array([b_col0, b_row0])
    pa_inl = pa[mr.inlier_indices]
    pb_inl = pb[mr.inlier_indices]

    print("\n=== SCORING (corner-homography GT, full-image) ===")
    summary = {}
    for tol in (10, 25, 50, 100):
        rep_in = score_correspondences(pa_inl, pb_inl, transform, tolerance_px=tol)
        rep_all = score_correspondences(pa, pb, transform, tolerance_px=tol)
        summary[str(tol)] = {
            "inliers": rep_in.summary_dict(),
            "all": rep_all.summary_dict(),
        }
        print(f"--- tolerance {tol} px (={tol*0.25:.0f} m ground) ---")
        print(f"  inliers vs GT : {rep_in.correct}/{rep_in.total_scored} ({rep_in.percent_correct:.1f}%)")
        print(f"  all vs GT     : {rep_all.correct}/{rep_all.total_scored} ({rep_all.percent_correct:.1f}%)")

    if mr.good_matches:
        draw_n = min(200, len(mr.good_matches))
        overlay = cv2.drawMatches(a_scaled.image, mr.keypoints_a_adj,
                                  b_scaled.image, mr.keypoints_b_adj,
                                  mr.good_matches[:draw_n], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        save_preview(overlay, OUT_DIR / "overlay_sift.png", max_dim=2400)
        print("overlay:", OUT_DIR / "overlay_sift.png")
    save_preview(a_scaled.image, OUT_DIR / "preview_A_window.png")
    save_preview(b_scaled.image, OUT_DIR / "preview_B_window.png")

    results = {
        "pair": [a.product_id, b.product_id],
        "res_m_per_px": [a.pixel_resolution_m, b.pixel_resolution_m],
        "ground_truth_mode": "corner_homography(same-scale, cropped window)",
        "a_window_px": {"cols": [A_COL0, A_COL1], "rows": [A_ROW0, A_ROW1]},
        "b_window_px": {"cols": [b_col0, b_col1], "rows": [b_row0, b_row1]},
        "keypoints": {"A": mr.keypoint_count_a, "B": mr.keypoint_count_b},
        "raw_matches": mr.raw_match_count,
        "ratio_test_matches": mr.ratio_test_matches,
        "ransac_inliers": mr.inlier_count,
        "scoring": summary,
    }
    (OUT_DIR / "baseline_results.json").write_text(_json_dump(results))
    print("wrote", OUT_DIR / "baseline_results.json")
    return 0


def _json_dump(obj):
    import json
    return json.dumps(obj, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
