"""Phase 6 diagnostic: is the 0% GT score a GT error, SIFT error, or shift bug?

Rebuilds the crop, reruns SIFT, then compares:
  A) SIFT-inferred RANSAC homography (crop space) vs corner GT (projected into
     the crop window) by sampling control points - reports drift.
  B) Error of each corresponding point vs GT within the crop window
     (distribution + systematic bias).
  C) Cross-check the GT by using the SIFT inlier model to predict B points and
     measure proximity to the CSV-refined expected positions.
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
    from groundtruth import build_pixel_transform, _apply_homography
    from image_loader import load_image, to_uint8
    from pds4_parser import parse_label
    from scale_pyramid import ScaledImage
    from sift_baseline import run_sift_matching
    from csv_geolocation import read_geolocation_csv, locate_region
except ImportError:  # pragma: no cover
    from lunar_data_pipeline.groundtruth import build_pixel_transform, _apply_homography
    from lunar_data_pipeline.image_loader import load_image, to_uint8
    from lunar_data_pipeline.pds4_parser import parse_label
    from lunar_data_pipeline.scale_pyramid import ScaledImage
    from lunar_data_pipeline.sift_baseline import run_sift_matching
    from lunar_data_pipeline.csv_geolocation import read_geolocation_csv, locate_region

A_LABEL = "D:/Multi-Modal-lunar-image-matching-system/ch2_ohr_ncp_20260103T0609041371_d_img_d18/data/calibrated/20260103/ch2_ohr_ncp_20260103T0609041371_d_img_d18.xml"
B_LABEL = "D:/Multi-Modal-lunar-image-matching-system/ch2_ohr_ncp_20260103T1005176450_d_img_d18/data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.xml"
AC0, AC1, AR0, AR1 = 2000, 10000, 40000, 48000
PAD = 200


def main():
    a = parse_label(A_LABEL); b = parse_label(B_LABEL)
    transform = build_pixel_transform(a, b)

    a_corners = np.array([[AC0, AR0], [AC1, AR0], [AC1, AR1], [AC0, AR1]], float)
    bp = transform.project_many(a_corners)
    BC0 = max(int(np.floor(bp[:,0].min()))-PAD, 0)
    BC1 = min(int(np.ceil(bp[:,0].max()))+PAD, b.samples)
    BR0 = max(int(np.floor(bp[:,1].min()))-PAD, 0)
    BR1 = min(int(np.ceil(bp[:,1].max()))+PAD, b.lines)

    a_crop = to_uint8(load_image(a, row_range=(AR0, AR1), col_range=(AC0, AC1)))
    b_crop = to_uint8(load_image(b, row_range=(BR0, BR1), col_range=(BC0, BC1)))
    a_scaled = ScaledImage(a_crop, 1.0, a.lines, a.samples, a.pixel_resolution_m, b.pixel_resolution_m, 1.0)
    b_scaled = ScaledImage(b_crop, 1.0, b.lines, b.samples, b.pixel_resolution_m, b.pixel_resolution_m, 1.0)

    mr = run_sift_matching(a_scaled, b_scaled, ratio_thresh=0.7, ransac_reproj_thresh=8.0, nfeatures=50000)
    pa = mr.points_a_original + np.array([AC0, AR0])
    pb = mr.points_b_original + np.array([BC0, BR0])
    pa_inl, pb_inl = pa[mr.inlier_indices], pb[mr.inlier_indices]
    print(f"inliers={mr.inlier_count} ratio={mr.ratio_test_matches}")

    # B) direct residual: GT-expected B for each A-inlier vs SIFT B
    expected = transform.project_many(pa_inl)
    res = pb_inl - expected
    dist = np.hypot(res[:,0], res[:,1])
    print("\n[B] inlier residual (SIFT B - GT-expected B), absolute px:")
    print("  count", len(dist), "mean", dist.mean(), "median", np.median(dist),
          "p90", np.percentile(dist,90), "max", dist.max())
    print("  mean dx", res[:,0].mean(), "mean dy", res[:,1].mean())
    # scaling: for a correctly-shifted same-scale match, GT-expected and SIFT-B should be close.
    # compute per-inlier ground distance implied by the residual
    print("  % within 10px:",100*(dist<=10).mean(),"25px:",100*(dist<=25).mean(),
          "50:",100*(dist<=50).mean(),"100:",100*(dist<=100).mean())

    # A) compare SIFT RANSAC H (crop->crop) composed with GT on the window corners
    H = mr.homography_adjusted  # crop A -> crop B
    grid = np.array([[0,0],[8000,0],[8000,8000],[0,8000]], float)
    pred_crop = _apply_homography(H, grid)
    # GT: map these crop coords to absolute A, then project to absolute B, then back to crop
    gt_abs = transform.project_many(grid + np.array([AC0, AR0]))
    gt_crop = gt_abs - np.array([BC0, BR0])
    drift = pred_crop - gt_crop
    dg = np.hypot(drift[:,0], drift[:,1])
    print("\n[A] SIFT-H vs GT drift on window corners (crop px):", dict(zip(['UL','UR','LR','LL'], np.round(dg,1).tolist())))
    print("    SIFT-H corners:", np.round(pred_crop,1).tolist())
    print("    GT     corners:", np.round(gt_crop,1).tolist())


if __name__ == "__main__":
    main()
