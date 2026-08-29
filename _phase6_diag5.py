"""Phase 6 diag 5: pure grid-based content-overlap check (no homographies).

Determines whether the two OHRC images genuinely image the same ground:
- geocode N sample pixels of the A-window via A's geolocation grid (nearest-neighbor)
- for each resulting (lon,lat), find the nearest B grid point and its pixel location
- if B's grid covers those (lon,lat) near the B-window, the imagery really overlaps
  there (real same-scale test). Then the SIFT ~27k px mismatch is a genuine failure.
Also reports the per-point A-window -> B-window pixel correspondence (via B grid),
which is itself a homography-free expected mapping to compare SIFT against.
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
    from csv_geolocation import read_geolocation_csv
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    from lunar_data_pipeline.pds4_parser import parse_label
    from lunar_data_pipeline.csv_geolocation import read_geolocation_csv
    from scipy.spatial import cKDTree

A_LABEL = "D:/Multi-Modal-lunar-image-matching-system/ch2_ohr_ncp_20260103T0609041371_d_img_d18/data/calibrated/20260103/ch2_ohr_ncp_20260103T0609041371_d_img_d18.xml"
B_LABEL = "D:/Multi-Modal-lunar-image-matching-system/ch2_ohr_ncp_20260103T1005176450_d_img_d18/data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.xml"
AC0, AC1, AR0, AR1 = 2000, 10000, 40000, 48000


def main():
    a = parse_label(A_LABEL); b = parse_label(B_LABEL)
    ga = read_geolocation_csv(a.geometry_csv_path)
    gb = read_geolocation_csv(b.geometry_csv_path)

    # A-window sample pixels (col,row)
    rng = np.random.default_rng(0)
    n = 400
    cols = rng.integers(AC0, AC1, n); rows = rng.integers(AR0, AR1, n)
    a_samp = np.column_stack([cols, rows])

    # geocode A pixels by nearest A-grid point (grid is dense -> ~nearest px)
    ta = cKDTree(np.column_stack([ga.pixel, ga.scan]))
    _, ia = ta.query(a_samp, k=1)
    a_lonlat = np.column_stack([ga.lon[ia], ga.lat[ia]])

    # find B grid points near those lon/lat
    tb = cKDTree(np.column_stack([gb.lon, gb.lat]))
    d_geo, ib = tb.query(a_lonlat, k=1)
    b_pix = np.column_stack([gb.pixel[ib], gb.scan[ib]])

    # B-window (corner prediction) for reference
    b_win = b_pix.copy()

    print(f"A window {n} sampled pixels:")
    print(f"  A window col range [{cols.min()},{cols.max()}] row [{rows.min()},{rows.max()}]")
    print(f"  geocoded A lon range [{a_lonlat[:,0].min():.3f},{a_lonlat[:,0].max():.3f}] "
          f"lat [{a_lonlat[:,1].min():.3f},{a_lonlat[:,1].max():.3f}]")
    print(f"  B nearest-grid geo distance (deg): mean {d_geo.mean():.5f} max {d_geo.max():.5f}")
    print(f"    (B grid spacing ~0.01 deg? if d_geo small, B images that ground)")
    print(f"  correspondiing B pixel: col [{b_pix[:,0].min():.0f},{b_pix[:,0].max():.0f}] "
          f"row [{b_pix[:,1].min():.0f},{b_pix[:,1].max():.0f}]")

    # B geolocation grid coverage near A's region
    spread_ok = np.hypot(b_pix[:,0]-b_pix[:,0].mean(), b_pix[:,1]-b_pix[:,1].mean())
    print(f"  B pixel spread around centroid: mean {spread_ok.mean():.0f} px, "
          f"std {spread_ok.std():.0f} px")

    # Confirm: does SIFT place A-window content at b_pix? We know SIFT -> ~27k px away
    # from correctness. Report expected B window center from grid correspondence.
    print(f"\n  EXPECTED B-window (grid-correspondence) col [{b_pix[:,0].min():.0f},{b_pix[:,0].max():.0f}] "
          f"row [{b_pix[:,1].min():.0f},{b_pix[:,1].max():.0f}]")
    print(f"  (earlier corner+CSV GT gave col [~2606,11229] row [~35584,44289])")


if __name__ == "__main__":
    main()
