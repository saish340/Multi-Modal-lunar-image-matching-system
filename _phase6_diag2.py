"""Phase 6 diag 2: confirm projection + re-score SIFT vs CSV-refined local GT.

The corner-homography GT is invalid for polar-stereographic products (a planar
H cannot represent the nonlinear warp). Here we fit a LOCAL homography from the
product-B per-pixel geolocation CSV restricted to the crop window (the best
available GT, same principle as run_baseline) and score the SIFT inliers (shifted
to absolute B coordinates) against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_pkg = _Path(__file__).resolve().parent / "lunar_data_pipeline"
if str(_pkg) not in _sys.path:
    _sys.path.insert(0, str(_pkg))

try:
    from groundtruth import build_pixel_transform, PixelTransform, _apply_homography
    from image_loader import load_image, to_uint8
    from pds4_parser import parse_label
    from scale_pyramid import ScaledImage
    from sift_baseline import run_sift_matching
    from csv_geolocation import read_geolocation_csv, fit_local_transforms, locate_region
    from scorer import score_correspondences
except ImportError:  # pragma: no cover
    from lunar_data_pipeline.groundtruth import build_pixel_transform, PixelTransform, _apply_homography
    from lunar_data_pipeline.image_loader import load_image, to_uint8
    from lunar_data_pipeline.pds4_parser import parse_label
    from lunar_data_pipeline.scale_pyramid import ScaledImage
    from lunar_data_pipeline.sift_baseline import run_sift_matching
    from lunar_data_pipeline.csv_geolocation import read_geolocation_csv, fit_local_transforms, locate_region
    from lunar_data_pipeline.scorer import score_correspondences

A_LABEL = "D:/Multi-Modal-lunar-image-matching-system/ch2_ohr_ncp_20260103T0609041371_d_img_d18/data/calibrated/20260103/ch2_ohr_ncp_20260103T0609041371_d_img_d18.xml"
B_LABEL = "D:/Multi-Modal-lunar-image-matching-system/ch2_ohr_ncp_20260103T1005176450_d_img_d18/data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.xml"
AC0, AC1, AR0, AR1 = 2000, 10000, 40000, 48000
PAD = 200


def main():
    a = parse_label(A_LABEL); b = parse_label(B_LABEL)
    print("projection A:", a.projection, "| B:", b.projection)
    corner_t = build_pixel_transform(a, b)

    a_corners = np.array([[AC0, AR0], [AC1, AR0], [AC1, AR1], [AC0, AR1]], float)
    bp = corner_t.project_many(a_corners)
    BC0 = max(int(np.floor(bp[:,0].min()))-PAD, 0)
    BC1 = min(int(np.ceil(bp[:,0].max()))+PAD, b.samples)
    BR0 = max(int(np.floor(bp[:,1].min()))-PAD, 0)
    BR1 = min(int(np.ceil(bp[:,1].max()))+PAD, b.lines)

    # CSV-refined local GT over the B crop window
    grid = read_geolocation_csv(b.geometry_csv_path)
    h_px2geo, h_geo2px, n = fit_local_transforms(grid, (BR0, BR1), (BC0, BC1))
    csv_t = PixelTransform(a.product_id, b.product_id,
                           h_pixel_to_geo_a=corner_t.h_pixel_to_geo_a,
                           h_geo_to_pixel_b=h_geo2px)
    print(f"CSV local fit over window: {n} grid points (valid GT for polar-stereographic)")

    a_crop = to_uint8(load_image(a, row_range=(AR0, AR1), col_range=(AC0, AC1)))
    b_crop = to_uint8(load_image(b, row_range=(BR0, BR1), col_range=(BC0, BC1)))
    a_scaled = ScaledImage(a_crop, 1.0, a.lines, a.samples, a.pixel_resolution_m, b.pixel_resolution_m, 1.0)
    b_scaled = ScaledImage(b_crop, 1.0, b.lines, b.samples, b.pixel_resolution_m, b.pixel_resolution_m, 1.0)
    mr = run_sift_matching(a_scaled, b_scaled, ratio_thresh=0.7, ransac_reproj_thresh=8.0, nfeatures=50000)

    pa = mr.points_a_original + np.array([AC0, AR0])
    pb = mr.points_b_original + np.array([BC0, BR0])
    pa_inl, pb_inl = pa[mr.inlier_indices], pb[mr.inlier_indices]
    print(f"ratio={mr.ratio_test_matches} inliers={mr.inlier_count}")

    for tr in (csv_t, corner_t, None):
        for tol in (10, 25, 50):
            rep = score_correspondences(pa_inl, pb_inl, tr, tolerance_px=tol)
            tag = "CSV" if tr is csv_t else ("corner" if tr is corner_t else "corner")
            print(f"[{tag}] tol {tol}px: {rep.correct}/{rep.total_scored} ({rep.percent_correct:.1f}%)")
        if tr is not None:
            repin = score_correspondences(pa_inl, pb_inl, tr, tolerance_px=75)
            repall = score_correspondences(pa, pb, tr, tolerance_px=75)
            tag = "CSV" if tr is csv_t else ("corner" if tr is corner_t else "corner")
            print(f"[{tag}] tol 75px: inliers {repin.correct}/{repin.total_scored} | all {repall.correct}/{repall.total_scored} ({repall.percent_correct:.1f}%)")


if __name__ == "__main__":
    main()
