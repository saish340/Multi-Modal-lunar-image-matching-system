"""Phase 8 - run anisotropic-affine phase-correlation on the real OHRC/TMC-2 pair.

Same experiment setup as ``run_fmt_real_pair.py`` (Phase 7 primary): both images
render the SAME ground region ``R`` at the SAME display resolution, so the
recovered transform is the residual (anisotropic scale, rotation, translation)
between the two sensors' views of ``R``.

The Phase 7 Fourier-Mellin transform assumed a SINGLE isotropic scale; on this
pushbroom pair that assumption is violated (independent along-track/cross-track
scales), so FMT fit a global ``s * Rot`` poorly. Phase 8 instead recovers two
independent scales (``sx``, ``sy``) plus rotation and translation, which should
fit the geolocation ground truth substantially better.

Scoring mirrors ``run_fmt_real_pair.py``: sample correspondence points inside
``A``, map them to ``B`` via the authoritative per-pixel geolocation CSVs, then
apply the recovered affine ``H_apply`` and measure pixel error vs that ground
truth. The geolocation data is used ONLY for scoring, never to seed the search.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_pkg = Path(__file__).resolve().parent / "lunar_data_pipeline"
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from anisotropic_affine_search import search  # noqa: E402
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
OUT_DIR = Path("phase8_aniso_output")

#: Ground region R in OHRC original-resolution pixels (cols, rows). Same region
#: as the Phase 7 runner so results are directly comparable. The square region
#: has a favourable aspect and is the default here.
SQUARE_OHRC = (400, 11800, 45000, 56400)  # cols, cols, rows, rows (11400 sq)
FULL_OHRC = (400, 11800, 45000, 69000)  # full swath, 24000 rows

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
    ap.add_argument("--region", choices=["square", "full"], default="square")
    ap.add_argument("--n_gt", type=int, default=200)
    ap.add_argument("--coarse", type=int, default=7)
    ap.add_argument("--fine", type=int, default=7)
    ap.add_argument(
        "--flip-b", action="store_true",
        help="flip B's scan axis before searching. The raw geolocation CSVs "
             "give opposite along-track latitude ordering between OHRC and "
             "TMC-2 (det<0 residual), which the positive-scale model cannot "
             "represent; this tests the physically-consistent mirror hypothesis.",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = parse_label(A_LABEL); b = parse_label(B_LABEL)
    assert a is not None and b is not None
    region = SQUARE_OHRC if args.region == "square" else FULL_OHRC
    A_OHRC_COL0, A_OHRC_COL1, A_OHRC_ROW0, A_OHRC_ROW1 = region
    ag = read_geolocation_csv(a.geometry_csv_path)
    bg = read_geolocation_csv(b.geometry_csv_path)
    print(f"PAIR: {a.product_id} (OHRC {a.pixel_resolution_m} m/px)   vs   "
          f"{b.product_id} (TMC-2 {b.pixel_resolution_m} m/px)")
    res_ratio = b.pixel_resolution_m / a.pixel_resolution_m
    print(f"resolution ratio ~ {res_ratio:.2f}x")

    # ---- build equal-res A (downsampled OHRC of R) and B (TMC crop of R) ----
    o_w = A_OHRC_COL1 - A_OHRC_COL0
    o_h = A_OHRC_ROW1 - A_OHRC_ROW0
    ds = res_ratio
    disp_w = max(int(round(o_w / ds)), 1)
    disp_h = max(int(round(o_h / ds)), 1)

    a_crop = to_uint8(load_image(
        a, row_range=(A_OHRC_ROW0, A_OHRC_ROW1),
        col_range=(A_OHRC_COL0, A_OHRC_COL1),
    ))
    import cv2
    a_disp = cv2.resize(a_crop, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

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
    print(f"A display ({disp_w}x{disp_h}) from OHRC region "
          f"cols[{A_OHRC_COL0}:{A_OHRC_COL1}] rows[{A_OHRC_ROW0}:{A_OHRC_ROW1}]")
    print(f"B (TMC-2) region cols[{bc0}:{bc1}] rows[{br0}:{br1}]")
    b_crop = to_uint8(load_image(
        b, row_range=(br0, br1), col_range=(bc0, bc1),
    ))
    # Raw crop -> display scale factors. Ground truth must be rescaled by these
    # too, or scoring compares display-space predictions against raw-space truth.
    disp_to_raw_x = (bc1 - bc0) / disp_w
    disp_to_raw_y = (br1 - br0) / disp_h
    b_disp = cv2.resize(b_crop, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # ---- optional mirror handling (see --flip-b) ----
    # If a scan-axis reflection is applied it must apply to B's DISPLAY image
    # only, never to the geolocation ground truth (the GT stays in the raw
    # TMC-2 coordinate frame; a reflected recovered H maps raw-B space, so its
    # display-space projection must be un-flipped before scoring/overlay).
    b_disp_raw = b_disp
    if args.flip_b:
        b_disp = b_disp[::-1, :].copy()
        print("flipping B scan axis for search (mirror hypothesis)")

    # ---- run anisotropic search ----
    t0 = time.time()
    res = search(a_disp, b_disp, coarse_each=args.coarse, fine_each=args.fine)
    dt = time.time() - t0
    print(f"\nanisotropic search recovered (in {dt:.1f}s, {res.n_eval} "
          f"phase-correlations = {res.coarse_count} coarse + {res.fine_count} fine):")
    print(f"  sx       : {res.sx:.4f}")
    print(f"  sy       : {res.sy:.4f}")
    print(f"  theta    : {res.theta_deg:.2f} deg")
    print(f"  translate: ({res.tx:.1f}, {res.ty:.1f})")
    print(f"  PSR            : {res.psr:.2f}")
    print(f"  translation PSR: {res.translation_psr:.2f}")

    # ---- score against geolocation ground truth ----
    rng = np.random.default_rng(7)
    n = args.n_gt
    a_sx = rng.integers(0, disp_w, n).astype(float)
    a_sy = rng.integers(0, disp_h, n).astype(float)
    ds_y = o_h / disp_h
    ds_x = o_w / disp_w
    o_col = A_OHRC_COL0 + a_sx * ds_x
    o_row = A_OHRC_ROW0 + a_sy * ds_y
    b_tmc = ohrc_to_tmc_grid(ag, bg, o_col, o_row)
    # Rescale raw-TMC geolocation into the SAME display space as the search's
    # H_apply operates in; both must be display px to be comparable.
    b_disp_gt = np.column_stack([
        (b_tmc[:, 0] - bc0) / disp_to_raw_x,
        (b_tmc[:, 1] - br0) / disp_to_raw_y,
    ])

    a_pts = np.column_stack([a_sx, a_sy])
    pred = res.H_apply @ np.column_stack([a_pts, np.ones(n)]).T
    pred = (pred / pred[2])[:2].T
    # If B was mirrored in the search, the recovered H lives in the flipped
    # B frame; unflip predictions to compare against raw-frame ground truth.
    if args.flip_b:
        pred = pred.copy()
        pred[:, 1] = (disp_h - 1) - pred[:, 1]
    errs = np.hypot(pred[:, 0] - b_disp_gt[:, 0], pred[:, 1] - b_disp_gt[:, 1])

    print(f"\nScoring anisotropic transform vs geolocation GT ({n} points, "
          f"B-display px):")
    for tqq in TQ:
        k = int((errs <= tqq).sum())
        print(f"  within {tqq:5.0f} px : {k}/{n} ({100.0*k/n:.1f}%)")
    print(f"  error px min/median/mean/max: "
          f"{errs.min():.1f} / {np.median(errs):.1f} / {errs.mean():.1f} / {errs.max():.1f}")

    # ---- visual ----
    Mvis = res.H_apply[:2, :].copy()
    b_shift_disp = cv2.warpAffine(a_disp, np.float64(Mvis),
                                  (b_disp_raw.shape[1], b_disp_raw.shape[0]))
    if args.flip_b:
        b_shift_disp = b_shift_disp[::-1, :].copy()
    diff = cv2.absdiff(b_disp_raw, b_shift_disp)
    save_preview(np.hstack([a_disp, b_disp_raw, b_shift_disp, diff]),
                 OUT_DIR / f"aniso_real_overlay_{args.region}_{'flip' if args.flip_b else 'nof'}.png",
                 max_dim=2400)

    # ---- persist result record ----
    res_record = {
        "pair": [a.product_id, b.product_id],
        "region": args.region,
        "resolution_ratio_x": round(res_ratio, 3),
        "ohrc_region_px": {"cols": [A_OHRC_COL0, A_OHRC_COL1],
                           "rows": [A_OHRC_ROW0, A_OHRC_ROW1]},
        "a_display": [disp_w, disp_h],
        "b_tmc_region_px": {"cols": [bc0, bc1], "rows": [br0, br1]},
        "flip_b": bool(args.flip_b),
        "recovered": res.summary_dict(),
        "gt_scoring": {
            "n_points": n,
            "within_100px": int((errs <= 100).sum()),
            "within_300px": int((errs <= 300).sum()),
            "error_px": {"min": float(errs.min()), "median": float(np.median(errs)),
                         "mean": float(errs.mean()), "max": float(errs.max())},
        },
        "expected_true_transform": {
            "note": "geolocation-derived residual affine for the equal-res "
                    "region has NEGATIVE determinant (the raw per-pixel "
                    "geolocation CSVs give OHRC dlat/dscan>0 but TMC-2 "
                    "dlat/dscan<0: opposite along-track latitude ordering, "
                    "confirmed independently by the PDS4 corner footprints). "
                    "The positive-scale R(theta) diag(sx,sy) model cannot "
                    "represent a reflection, so a scan-axis flip of one image "
                    "is required. Additionally the shared geolocation grid is "
                    "sampled every 100 px, so affine-residual scoring is "
                    "lower-bounded at ~ 40 px display space by grid "
                    "quantization, not by true distortion.",
        },
    }
    rec_path = OUT_DIR / f"aniso_real_result_{args.region}_{'flip' if args.flip_b else 'nof'}.json"
    with open(rec_path, "w", encoding="utf-8") as fh:
        json.dump(res_record, fh, indent=2)
    print(f"wrote {rec_path}")

    print(f"\nVERDICT: sx={res.sx:.3f}, sy={res.sy:.3f}, "
          f"theta={res.theta_deg:.2f} deg, PSR={res.psr:.1f}, "
          f"GT median error={np.median(errs):.1f} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())