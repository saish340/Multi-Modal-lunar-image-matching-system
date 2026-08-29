"""Phase 6 diag 4: geolocation round-trip consistency (rules out coordinate bug).

1. Geocode the A-window center through A's CSV -> (lon,lat).
2. Confirm (lon,lat) is inside B's footprint polygon.
3. Convert that (lon,lat) back to B pixels via B's CSV (local fit) and compare
   with the B-window center (which the corner homography predicted).
   If the round-trip lands in B's window, geolocation is self-consistent and
   the two images DO correspond to the same ground -> any SIFT mismatch is a
   genuine feature-matching failure, not a coordinate bug.
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
    from pds4_parser import parse_label
    from overlap_finder import footprint_polygon
    from csv_geolocation import read_geolocation_csv, fit_local_transforms, _apply
    from groundtruth import build_pixel_transform, _apply_homography
except ImportError:  # pragma: no cover
    from lunar_data_pipeline.pds4_parser import parse_label
    from lunar_data_pipeline.overlap_finder import footprint_polygon
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
    poly_b = footprint_polygon(b)

    ct = build_pixel_transform(a, b)
    a_c = np.array([[AC0, AR0], [AC1, AR0], [AC1, AR1], [AC0, AR1]], float)
    bp = ct.project_many(a_c)
    BC0, BC1 = max(int(np.floor(bp[:,0].min()))-PAD,0), min(int(np.ceil(bp[:,0].max()))+PAD, b.samples)
    BR0, BR1 = max(int(np.floor(bp[:,1].min()))-PAD,0), min(int(np.ceil(bp[:,1].max()))+PAD, b.lines)

    hA, _, nA = fit_local_transforms(ga, (AR0, AR1), (AC0, AC1))
    hB, _, nB = fit_local_transforms(gb, (BR0, BR1), (BC0, BC1))

    # A-window center -> geo
    a_center = np.array([[AC0 + (AC1-AC0)/2, AR0 + (AR1-AR0)/2]])
    geo = _apply(hA, a_center)[0]
    print(f"A-window center pixel ({a_center[0][0]:.0f},{a_center[0][1]:.0f}) -> geo lon={geo[0]:.4f} lat={geo[1]:.4f}")
    print("inside B footprint?", "YES" if poly_b.contains(__import__('shapely').geometry.Point(geo[0], geo[1])) else "NO")

    # geo -> B pixels via B CSV
    b_pix = _apply(hB, np.array([geo]))[0]
    b_center = np.array([BC0+(BC1-BC0)/2, BR0+(BR1-BR0)/2])
    print(f"geo -> B pixel ({b_pix[0]:.0f},{b_pix[1]:.0f})   B-window center ({b_center[0]:.0f},{b_center[1]:.0f})")
    off = np.hypot(*(b_pix - b_center))
    print(f"round-trip offset from B-window center: {off:.1f} px")
    print("\ninterpretation: if small (<= a few px) geolocation is consistent, windows correspond,")
    print("and SIFT mismatch (~27k px) is a genuine feature-matching failure, NOT a coordinate bug.")


if __name__ == "__main__":
    main()
