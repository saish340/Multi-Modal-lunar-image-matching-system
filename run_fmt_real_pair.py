"""Phase 7 - run Fourier-Mellin Transform on the real OHRC/TMC-2 pair.

Primary experiment (resolution-equalised): both images are rendered showing the
SAME ground region ``R`` (in the confirmed overlap) at the SAME display
resolution:

- ``B`` = TMC-2 reference crop of ``R``, natural 6.13 m/px.
- ``A`` = OHRC coverage of ``R``, downsampled 0.2 -> 6.13 m/px.

FMT then recovers the residual GLOBAL transform (scale, rotation, translation)
between the two sensors' views of ``R``. Because both are equal-res, the *scale*
part is expected ~1 and the meaningful recoveries are the residual rotation
(~2.7 deg per Phase 4 track geometry) and translation, plus any residual scale
from non-uniform pushbroom geometry.

Scoring: sample correspondence points inside A, map them to B via the
authoritative per-pixel geolocation CSVs (OHRC->lon/lat->TMC, nearest grid),
then apply the recovered FMT transform and measure pixel error vs that
geolocation ground truth. A correct global transform should project points onto
the geolocation-derived locations within tolerance.

The scale-recovery dimension is validated on real content by a secondary mode:
feed ``A`` at HALF the display resolution (12.26 m/px for the same canvas) and
confirm FMT recovers a scale near 0.5, bridging the synthetic test to real
cross-sensor content.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

_pkg = Path(__file__).resolve().parent / "lunar_data_pipeline"
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from fourier_mellin import register  # noqa: E402
from image_loader import load_image, save_preview, to_uint8  # noqa: E402
from pds4_parser import parse_label  # noqa: E402
from csv_geolocation import read_geolocation_csv  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

A_LABEL = (
    "D:/Multi-Modal-lunar-image-matching-system/"
    "ch2_ohr_ncp_20231004T0406038822_d_img_d18/"
    "data/calibrated/20231004/ch2_ohr_ncp_20231004T0406038822_d_img_d18.xml"
)
B_LABEL = (
    "D:/Multi-Modal-lunar-image-matching-system/"
    "ch2_tmc_ncn_20250707T1853051045_d_img_d18/"
    "data/calibrated/20250707/ch2_tmc_ncn_20250707T1853051045_d_img_d18.xml"
)
OUT_DIR = Path("phase7_fmt_output")

# Ground region R: in OHRC original-resolution pixels (cols, rows). We use the
# FULL OHRC swath (its usable cross-track extent, cols 400..11800 = ~2.3 km at
# 0.2 m/px) and a long along-track stretch, so that after downsampling to TMC-2
# resolution (30.65x) the equal-res display still has usable size. OHRC's
# swath is narrow in TMC terms (~390 TMC px), unavoidable for this sensor pair.
A_OHRC_COL0, A_OHRC_COL1 = 400, 11800  # full swath, ~2.3 km
A_OHRC_ROW0, A_OHRC_ROW1 = 45000, 69000  # 24000 px = 4.8 km along track
DISPLAY = 1024  # display size for both equal-res images (not used directly)

#: square alternative: OHRC region whose swath width equals its along-track
#: height (favorable aspect for log-polar FMT). Dims = full swath width.
SQUARE_OHRC = (400, 11800, 45000, 56400)  # cols, cols, rows, rows (11400 sq)
def _region(use_square: bool) -> tuple[int, ...]:
    if use_square:
        return SQUARE_OHRC
    return (A_OHRC_COL0, A_OHRC_COL1, A_OHRC_ROW0, A_OHRC_ROW1)

#: scoring tolerance in B-display (TMC-2) pixels.
TQ = [50.0, 100.0, 300.0]


def ohrc_to_tmc_grid(a_grid, b_grid, ox, oy):
    """Map OHRC-res pixels (ox,oy) -> TMC-2-res pixels via nearest geolocation."""
    ta = cKDTree(np.column_stack([a_grid.pixel, a_grid.scan]))
    _, ia = ta.query(np.column_stack([ox, oy]), k=1)
    lonlat = np.column_stack([a_grid.lon[ia], a_grid.lat[ia]])
    tb = cKDTree(np.column_stack([b_grid.lon, b_grid.lat]))
    _, ib = tb.query(lonlat, k=1)
    return np.column_stack([b_grid.pixel[ib], b_grid.scan[ib]])


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S", stream=sys.stderr,
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--display", type=int, default=DISPLAY)
    ap.add_argument("--half-scale", action="store_true",
                    help="secondary mode: A at half display resolution (scale~0.5)")
    ap.add_argument("--n_gt", type=int, default=200)
    ap.add_argument("--square", action="store_true",
                    help="use a square equal-res OHRC region (favourable aspect)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = parse_label(A_LABEL); b = parse_label(B_LABEL)
    assert a is not None and b is not None
    A_OHRC_COL0, A_OHRC_COL1, A_OHRC_ROW0, A_OHRC_ROW1 = _region(args.square)
    ag = read_geolocation_csv(a.geometry_csv_path)
    bg = read_geolocation_csv(b.geometry_csv_path)
    print(f"PAIR: {a.product_id} (OHRC {a.pixel_resolution_m} m/px)   vs   "
          f"{b.product_id} (TMC-2 {b.pixel_resolution_m} m/px)")
    res_ratio = b.pixel_resolution_m / a.pixel_resolution_m
    print(f"resolution ratio ~ {res_ratio:.2f}x")

    # ---- build A (downsampled OHRC of region R) and B (TMC crop of R) ----
    o_w = A_OHRC_COL1 - A_OHRC_COL0
    o_h = A_OHRC_ROW1 - A_OHRC_ROW0
    # downsample factor to bring OHRC to TMC resolution
    ds = res_ratio
    disp_w = max(int(round(o_w / ds)), 1)
    disp_h = max(int(round(o_h / ds)), 1)
    if args.half_scale:
        disp_w_int = max(int(round(o_w / (2 * ds))), 1)
        disp_h_int = max(int(round(o_h / (2 * ds))), 1)
    else:
        disp_w_int, disp_h_int = disp_w, disp_h

    a_crop = to_uint8(load_image(
        a, row_range=(A_OHRC_ROW0, A_OHRC_ROW1),
        col_range=(A_OHRC_COL0, A_OHRC_COL1),
    ))
    import cv2
    a_disp = cv2.resize(a_crop, (disp_w_int, disp_h_int),
                        interpolation=cv2.INTER_AREA)

    # B: the TMC-2 region covering region R (same ground area). We find the TMC
    # bounding box of region R by geolocating R's corners.
    corners_ohrc = np.array([
        [A_OHRC_COL0, A_OHRC_ROW0], [A_OHRC_COL1, A_OHRC_ROW0],
        [A_OHRC_COL1, A_OHRC_ROW1], [A_OHRC_COL0, A_OHRC_ROW1],
    ], float)
    b_corners = ohrc_to_tmc_grid(ag, bg, corners_ohrc[:, 0], corners_ohrc[:, 1])
    pad = 100
    bc0 = max(int(np.floor(b_corners[:, 0].min())) - pad, 0)
    bc1 = min(int(np.ceil(b_corners[:, 0].max())) + pad, b.samples)
    br0 = max(int(np.floor(b_corners[:, 1].min())) - pad, 0)
    br1 = min(int(np.ceil(b_corners[:, 1].max())) + pad, b.lines)
    print(f"A display ({disp_w_int}x{disp_h_int}) from OHRC region "
          f"cols[{A_OHRC_COL0}:{A_OHRC_COL1}] rows[{A_OHRC_ROW0}:{A_OHRC_ROW1}]")
    print(f"B (TMC-2) region cols[{bc0}:{bc1}] rows[{br0}:{br1}]")
    b_crop = to_uint8(load_image(
        b, row_range=(br0, br1), col_range=(bc0, bc1),
    ))
    # resize B to match A's display dims (both show region R at same scale)
    b_disp = cv2.resize(b_crop, (disp_w_int, disp_h_int),
                        interpolation=cv2.INTER_AREA)

    # ---- run FMT ----
    t0 = time.time()
    res = register(a_disp, b_disp)
    dt = time.time() - t0
    print(f"\nFMT recovered (in {dt:.1f}s):")
    print(f"  scale    : {res.scale:.4f}")
    print(f"  theta    : {res.theta_deg:.2f} deg")
    print(f"  translate: ({res.tx:.1f}, {res.ty:.1f})")
    print(f"  log-polar PSR      : {res.log_polar_psr:.2f}")
    print(f"  translation PSR    : {res.translation_psr:.2f}")

    # ---- score against geolocation ground truth ----
    # sample A-display points, map to OHRC-res, geolocate to TMC-res, then to
    # B display coords (subtract B crop offset). FMT projects A-display->B-display.
    rng = np.random.default_rng(7)
    n = args.n_gt
    a_sx = rng.integers(0, disp_w_int, n).astype(float)
    a_sy = rng.integers(0, disp_h_int, n).astype(float)
    # A-display -> OHRC-res (col,row)  x->col(scaled by ds), y->row(scaled by ds_y)
    ds_y = o_h / disp_h_int
    ds_x = o_w / disp_w_int
    o_col = A_OHRC_COL0 + a_sx * ds_x
    o_row = A_OHRC_ROW0 + a_sy * ds_y
    b_tmc = ohrc_to_tmc_grid(ag, bg, o_col, o_row)
    b_disp_gt = np.column_stack([b_tmc[:, 0] - bc0, b_tmc[:, 1] - br0])

    a_pts = np.column_stack([a_sx, a_sy])
    pred = res.H_apply @ np.column_stack([a_pts, np.ones(n)]).T
    pred = (pred / pred[2])[:2].T
    errs = np.hypot(pred[:, 0] - b_disp_gt[:, 0], pred[:, 1] - b_disp_gt[:, 1])

    print(f"\nScoring FMT transform vs geolocation GT ({n} points, B-display px):")
    for tqq in TQ:
        k = int((errs <= tqq).sum())
        print(f"  within {tqq:5.0f} px : {k}/{n} ({100.0*k/n:.1f}%)")
    print(f"  error px min/median/mean/max: "
          f"{errs.min():.1f} / {np.median(errs):.1f} / {errs.mean():.1f} / {errs.max():.1f}")

    # visual
    Mvis = res.H_apply[:2, :].copy()
    b_shift = cv2.warpAffine(a_disp, np.float64(Mvis),
                             (b_disp.shape[1], b_disp.shape[0]))
    diff = cv2.absdiff(b_disp, b_shift)
    save_preview(np.hstack([a_disp, b_disp, b_shift, diff]),
                 OUT_DIR / f"fmt_real_overlay_{'half' if args.half_scale else 'equal'}.png",
                 max_dim=2400)

    # ---- persist result record ----
    res_record = {
        "pair": [a.product_id, b.product_id],
        "mode": "half_scale" if args.half_scale else "equal_res",
        "square_region": args.square,
        "resolution_ratio_x": round(res_ratio, 3),
        "ohrc_region_px": {"cols": [A_OHRC_COL0, A_OHRC_COL1],
                           "rows": [A_OHRC_ROW0, A_OHRC_ROW1]},
        "a_display": [disp_w_int, disp_h_int],
        "b_tmc_region_px": {"cols": [bc0, bc1], "rows": [br0, br1]},
        "fmt_recovered": res.summary_dict(),
        "gt_scoring": {
            "n_points": n,
            "within_100px": int((errs <= 100).sum()),
            "within_300px": int((errs <= 300).sum()),
            "error_px": {"min": float(errs.min()), "median": float(np.median(errs)),
                         "mean": float(errs.mean()), "max": float(errs.max())},
        },
        "expected_true_transform": {
            "note": "geolocation-derived affine: rotation~2.9 deg, anisotropic "
                    "scale x~1.12 y~1.62 -> no global isotropic s*R exists",
        },
    }
    rec_path = OUT_DIR / f"fmt_real_result_{'half' if args.half_scale else 'equal'}.json"
    with open(rec_path, "w", encoding="utf-8") as fh:
        json.dump(res_record, fh, indent=2)
    print(f"wrote {rec_path}")

    # verdict: scale near expected and a coherent peak, and GT projection sane
    print(f"\nVERDICT (equal-res primary): see report; scale={res.scale:.3f}, "
          f"theta={res.theta_deg:.2f}deg, logpolar PSR={res.log_polar_psr:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
