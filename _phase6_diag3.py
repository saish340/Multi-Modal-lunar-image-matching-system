"""Phase 6 diag 3: geolocation cross-check of SIFT matches via each product's own CSV.

Ground-truth independent: for each SIFT ratio-test correspondence,
  - map the A-side point through A's geolocation CSV -> (lon,lat)_A
  - map the B-side point through B's geolocation CSV -> (lon,lat)_B
  - if the match is CORRECT, (lon,lat)_A should equal (lon,lat)_B.
Measures the lon/lat separation; if small (<< pixel ground distance) SIFT matched
the correct physical features; if large, SIFT latched onto self-similar texture.
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
    from image_loader import load_image, to_uint8
    from pds4_parser import parse_label
    from scale_pyramid import ScaledImage
    from sift_baseline import run_sift_matching
    from csv_geolocation import read_geolocation_csv, fit_local_transforms, _apply
    from groundtruth import build_pixel_transform, _apply_homography
except ImportError:  # pragma: no cover
    from lunar_data_pipeline.image_loader import load_image, to_uint8
    from lunar_data_pipeline.pds4_parser import parse_label
    from lunar_data_pipeline.scale_pyramid import ScaledImage
    from lunar_data_pipeline.sift_baseline import run_sift_matching
    from lunar_data_pipeline.csv_geolocation import read_geolocation_csv, fit_local_transforms, _apply
    from lunar_data_pipeline.groundtruth import build_pixel_transform, _apply_homography

A_LABEL = "D:/Multi-Modal-lunar-image-matching-system/ch2_ohr_ncp_20260103T0609041371_d_img_d18/data/calibrated/20260103/ch2_ohr_ncp_20260103T0609041371_d_img_d18.xml"
B_LABEL = "D:/Multi-Modal-lunar-image-matching-system/ch2_ohr_ncp_20260103T1005176450_d_img_d18/data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.xml"
AC0, AC1, AR0, AR1 = 2000, 10000, 40000, 48000
PAD = 200


def main():
    a = parse_label(A_LABEL); b = parse_label(B_LABEL)
    ga = read_geolocation_csv(a.geometry_csv_path)
    gb = read_geolocation_csv(b.geometry_csv_path)

    # local fits over the windowed pixel ranges (full strip is fine; homography
    # is fitted locally where the matches actually are)
    # Derive the B crop window from the corner projection (same as diag).
    ct = build_pixel_transform(a, b)
    a_corners = np.array([[AC0, AR0], [AC1, AR0], [AC1, AR1], [AC0, AR1]], float)
    bp = ct.project_many(a_corners)
    BC0 = max(int(np.floor(bp[:,0].min()))-PAD, 0)
    BC1 = min(int(np.ceil(bp[:,0].max()))+PAD, b.samples)
    BR0 = max(int(np.floor(bp[:,1].min()))-PAD, 0)
    BR1 = min(int(np.ceil(bp[:,1].max()))+PAD, b.lines)

    # A-side fit over the A window; B-side fit over the B window (local, bounded)
    hA_px2geo, hA_geo2px, nA = fit_local_transforms(ga, (AR0, AR1), (AC0, AC1))
    hB_px2geo, hB_geo2px, nB = fit_local_transforms(gb, (BR0, BR1), (BC0, BC1))
    print(f"A local fit n={nA}, B window ({BC0},{BC1})x({BR0},{BR1}) fit n={nB}")

    a_crop = to_uint8(load_image(a, row_range=(AR0, AR1), col_range=(AC0, AC1)))
    b_crop = to_uint8(load_image(b, row_range=(BR0, BR1), col_range=(BC0, BC1)))
    a_scaled = ScaledImage(a_crop, 1.0, a.lines, a.samples, a.pixel_resolution_m, b.pixel_resolution_m, 1.0)
    b_scaled = ScaledImage(b_crop, 1.0, b.lines, b.samples, b.pixel_resolution_m, b.pixel_resolution_m, 1.0)
    mr = run_sift_matching(a_scaled, b_scaled, ratio_thresh=0.7, ransac_reproj_thresh=8.0, nfeatures=50000)
    print(f"ratio={mr.ratio_test_matches} inliers={mr.inlier_count}")

    pa = mr.points_a_original + np.array([AC0, AR0])   # absolute A
    pb = mr.points_b_original + np.array([BC0, BR0])   # absolute B (window origin)
    mask = np.zeros(len(pa), bool); mask[mr.inlier_indices] = True

    geoA = _apply(hA_px2geo, pa)
    geoB = _apply(hB_px2geo, pb)
    print("A-side matched points geolocate -> "
          f"lon [{geoA[:,0].min():.3f},{geoA[:,0].max():.3f}] lat [{geoA[:,1].min():.3f},{geoA[:,1].max():.3f}]")
    print("  (expected A-window ~ lon [24.96,25.90] lat [-85.05,-84.97])")
    print("B-side matched points geolocate -> "
          f"lon [{geoB[:,0].min():.3f},{geoB[:,0].max():.3f}] lat [{geoB[:,1].min():.3f},{geoB[:,1].max():.3f}]")
    dlon = (geoB[:,0] - geoA[:,0])
    dlat = geoB[:,1] - geoA[:,1]
    # approximate ground distance per degree at ~85S: ~111 km/deg lat, ~111*cos(85) ~9.7km/deg lon
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(85.0))
    dist_km = np.hypot(dlat*km_per_deg_lat, dlon*km_per_deg_lon)
    dist_px = dist_km / (0.25/1000.0)  # px at 0.25 m/px

    for name, sel in [("RATIO-ALL", np.ones(len(pa),bool)),
                      ("INLIERS", mask)]:
        d = dist_px[sel]
        print(f"\n[{name}] n={len(d)}  match geo-separation:")
        print(f"  mean {d.mean():.1f} px  median {np.median(d):.1f} px  "
              f"p10 {np.percentile(d,10):.1f}  p90 {np.percentile(d,90):.1f}  max {d.max():.1f}")
        print(f"  lon mean {dlon[sel].mean():.4f} deg  lat mean {dlat[sel].mean():.4f} deg")
        print(f"  % correct (sep < 25 m = 100px): {(d<100).mean()*100:.1f}%")
        print(f"  % correct (sep < 2.5 m = 10px): {(d<10).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
