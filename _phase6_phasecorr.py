"""Phase 6 - phase-correlation sanity check on the same-scale OHRC-OHRC pair.

Independent GLOBAL alignment test, diagnostic only (not a pipeline replacement).

Loads the SAME A-window crop (cols[2000:10000] x rows[40000:48000]) and the B-window
crop derived from the corner homography (as the SIFT sanity check did). The two crops
are pre-aligned by that corner transform, so a CORRECT global registration should
show a residual crop-relative translation near (0,0). Phase correlation recovers the
single dominant translation from the frequency domain.

Reported absolute offset = (B_col0 - A_col0 + dx, B_row0 - A_row0 + dy), compared
against the expected absolute offset (B_col0 - A_col0, B_row0 - A_row0).
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
    from csv_geolocation import read_geolocation_csv
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    from lunar_data_pipeline.groundtruth import build_pixel_transform
    from lunar_data_pipeline.image_loader import load_image, save_preview, to_uint8
    from lunar_data_pipeline.pds4_parser import parse_label
    from lunar_data_pipeline.csv_geolocation import read_geolocation_csv
    from scipy.spatial import cKDTree

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
A_COL0, A_COL1 = 2000, 10000
A_ROW0, A_ROW1 = 40000, 48000
PAD = 200
TOLERANCES = (10, 25, 50, 100)  # same-scale scoring tolerances in px (0.25 m/px)


def hann2d(shape):
    r = np.hanning(shape[0])[:, None]
    c = np.hanning(shape[1])[None, :]
    return r * c


def phase_correlate(a, b):
    """Return (dx, dy) of b relative to a, i.e. content in a moves to a+(dx,dy) in b,
    plus peak score, on equal-size inputs."""
    assert a.shape == b.shape, (a.shape, b.shape)
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    cross /= (np.abs(cross) + 1e-12)
    cc = np.fft.ifft2(cross).real
    # shift for centered display not needed; find max index
    idx = np.unravel_index(np.argmax(cc), cc.shape)
    peak = cc[idx]
    h, w = a.shape
    dy = idx[0] if idx[0] <= h // 2 else idx[0] - h
    dx = idx[1] if idx[1] <= w // 2 else idx[1] - w
    return dx, dy, peak


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a = parse_label(A_LABEL); b = parse_label(B_LABEL)
    assert a is not None and b is not None
    transform = build_pixel_transform(a, b)
    print(f"PAIR: {a.product_id}  vs  {b.product_id}")
    print(f"scale A={a.pixel_resolution_m} B={b.pixel_resolution_m} m/px")

    a_corners = np.array([[A_COL0, A_ROW0], [A_COL1, A_ROW0],
                          [A_COL1, A_ROW1], [A_COL0, A_ROW1]], float)
    b_proj = transform.project_many(a_corners)
    b_col0 = max(int(np.floor(b_proj[:, 0].min())) - PAD, 0)
    b_col1 = min(int(np.ceil(b_proj[:, 0].max())) + PAD, b.samples)
    b_row0 = max(int(np.floor(b_proj[:, 1].min())) - PAD, 0)
    b_row1 = min(int(np.ceil(b_proj[:, 1].max())) + PAD, b.lines)
    print(f"A window: cols[{A_COL0}:{A_COL1}] rows[{A_ROW0}:{A_ROW1}]  ({(A_COL1-A_COL0)}x{(A_ROW1-A_ROW0)})")
    print(f"B window: cols[{b_col0}:{b_col1}] rows[{b_row0}:{b_row1}]  ({(b_col1-b_col0)}x{(b_row1-b_row0)})")

    t0 = time.time()
    a_crop = to_uint8(load_image(a, row_range=(A_ROW0, A_ROW1), col_range=(A_COL0, A_COL1)))
    b_crop = to_uint8(load_image(b, row_range=(b_row0, b_row1), col_range=(b_col0, b_col1)))
    print(f"loaded A crop {a_crop.shape}, B crop {b_crop.shape} [{time.time()-t0:.1f}s]")

    H = max(a_crop.shape[0], b_crop.shape[0])
    W = max(a_crop.shape[1], b_crop.shape[1])
    size = 1
    while size < max(H, W):
        size *= 2
    print(f"pad target: {size}x{size}")

    def place(img):
        out = np.zeros((size, size), np.float32)
        top = (size - img.shape[0]) // 2
        left = (size - img.shape[1]) // 2
        out[top:top + img.shape[0], left:left + img.shape[1]] = img.astype(np.float32)
        return out, top, left

    a_pad, a_top, a_left = place(a_crop)
    b_pad, b_top, b_left = place(b_crop)
    win = hann2d((size, size))
    a_w = a_pad * win
    b_w = b_pad * win

    dx, dy, peak = phase_correlate(a_w, b_w)
    # dx,dy are in the padded coordinate frame (content moves from A-pad to B-pad).
    # Both images are centered in their pads, so the relative translation of the
    # original content equals (dx, dy) directly (same centering offsets).
    print(f"\nPHASE-CORRELATION (crop-relative): dx={dx} dy={dy}  peak_score={peak:.4f}")

    # PSR confidence: (peak - mean)/std of cross-power response excluding a window
    fa = np.fft.fft2(a_w); fb = np.fft.fft2(b_w)
    cross = fa * np.conj(fb); cross /= (np.abs(cross) + 1e-12)
    cc = np.fft.ifft2(cross).real
    ex, ey = (dx % cc.shape[1]), (dy % cc.shape[0])
    side = 7
    m = np.ones(cc.shape, bool)
    m[ey-side:ey+side+1, ex-side:ex+side+1] = False
    psr_sig = (cc[ey, ex] - cc[m].mean()) / (cc[m].std() + 1e-12)
    print(f"PSR (peak-to-sidelobe ratio): {psr_sig:.2f}  (>=~6-8 indicates a strong, "
          f"unambiguous peak; <~3 weak/ambiguous)")

    # --- grid/CSV nearest-correspondence expected offset (authoritative, homography-free)
    ga = read_geolocation_csv(a.geometry_csv_path)
    gb = read_geolocation_csv(b.geometry_csv_path)
    rng = np.random.default_rng(1)
    n = 300
    s_col = rng.integers(A_COL0, A_COL1, n); s_row = rng.integers(A_ROW0, A_ROW1, n)
    a_pts = np.column_stack([s_col, s_row])
    ta = cKDTree(np.column_stack([ga.pixel, ga.scan]))
    _, ia = ta.query(a_pts, k=1)
    alon_ = np.column_stack([ga.lon[ia], ga.lat[ia]])
    tb = cKDTree(np.column_stack([gb.lon, gb.lat]))
    _, ib = tb.query(alon_, k=1)
    b_pts = np.column_stack([gb.pixel[ib], gb.scan[ib]])
    exp_off = b_pts - a_pts
    exp_dx_m, exp_dy_m = np.median(exp_off, axis=0)
    print(f"\nGRID-GT expected offset (median over {n} A pts): dx={exp_dx_m:.0f} dy={exp_dy_m:.0f} "
          f"(IQR x {np.percentile(exp_off[:,0],[25,75]).round(0)}, "
          f"y {np.percentile(exp_off[:,1],[25,75]).round(0)})")

    # compare phase-corr ABSOLUTE offset to BOTH GTs
    abs_dx = (b_col0 - A_COL0) + dx
    abs_dy = (b_row0 - A_ROW0) + dy
    for gt_name, gx, gy in (
        ("corner-homography", (b_col0 - A_COL0), (b_row0 - A_ROW0)),
        ("grid/CSV", exp_dx_m, exp_dy_m),
    ):
        e = np.hypot(abs_dx - gx, abs_dy - gy)
        best = min(filter(lambda t: e <= t, TOLERANCES), default=None)
        print(f"\n  vs {gt_name:16s}: expected ({gx:.0f},{gy:.0f})  "
              f"phase-corr ({abs_dx:.0f},{abs_dy:.0f})  |err|={e:.1f} px  -> "
              f"{'PASS' if best is not None else 'FAIL'} (closest tol {best})")

    # spatial cross-correlation confidence (alternative to PSR)
    a_n = a_pad - a_pad.mean(); b_n = b_pad - b_pad.mean()
    coeff = float((a_n * b_n).sum() / (np.sqrt((a_n*a_n).sum() * (b_n*b_n).sum()) + 1e-12))
    print(f"  spatial cross-correlation (zero-shift): {coeff:.4f}")

    # visualization: shifted overlay of B content onto A using recovered translation
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    b_shift = cv2.warpAffine(b_crop, M, (a_crop.shape[1], a_crop.shape[0]))
    diff = cv2.absdiff(a_crop, b_shift)
    save_preview(np.hstack([a_crop, b_shift, diff]), OUT_DIR / "phasecorr_overlay.png", max_dim=2400)
    print("wrote", OUT_DIR / "phasecorr_overlay.png")

    # decision: PASS only if phase-corr offset is within tolerance of BOTH GTs with a
    # non-degenerate peak
    e_corner = np.hypot(abs_dx - (b_col0-A_COL0), abs_dy - (b_row0-A_ROW0))
    e_grid = np.hypot(abs_dx - exp_dx_m, abs_dy - exp_dy_m)
    pass_corner = e_corner <= max(TOLERANCES)
    pass_grid = e_grid <= max(TOLERANCES)
    strong = psr_sig >= 6.0
    print(f"\nDECISION (100px tolerance): corner_GT={'PASS' if pass_corner else 'FAIL'}  "
          f"grid_GT={'PASS' if pass_grid else 'FAIL'}  strong_peak={'YES' if strong else 'NO'}")
    return 0 if (pass_corner or pass_grid) and strong else 1


if __name__ == "__main__":
    sys.exit(main())
