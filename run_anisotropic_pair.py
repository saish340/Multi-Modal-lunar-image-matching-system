"""Phase 9 - generalized anisotropic-affine registration runner for an arbitrary
OHRC x TMC-2 pair.

Parameterised version of ``run_anisotropic_real_pair.py`` so the same
mirror-corrected anisotropic-affine search can be run on any usable pair:
takes the two PDS4 labels, an OHRC native-pixel region (the ground region R),
and a ``--flip-b`` mirror flag (determined independently per pair by
``run_p9_mirror_check.py``; only set when the raw geolocation shows opposite
along-track latitude orientation between the two products).

Both views are rendered of the SAME ground region R at the SAME display
resolution, so the recovered transform is the residual (anisotropic scale,
rotation, translation) between the two sensors. Scoring uses the per-pixel
geolocation CSVs as authoritative ground truth, rescaled into the SAME display
space as the search (the Phase 8 fix that had compared raw-TMC truth against
display-space predictions). Geolocation is used ONLY for scoring, never to
seed the search.
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

_pkg = Path(__file__).resolve().parent / "lunar_data_pipeline"
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from anisotropic_affine_search import search  # noqa: E402
from image_loader import load_image, save_preview, to_uint8  # noqa: E402
from pds4_parser import parse_label  # noqa: E402
from csv_geolocation import read_geolocation_csv  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def ohrc_to_tmc_grid(a_grid, b_grid, ox, oy):
    """Map OHRC-res pixels (ox,oy) -> TMC-2-res pixels via nearest geolocation."""
    ta = cKDTree(np.column_stack([a_grid.pixel, a_grid.scan]))
    _, ia = ta.query(np.column_stack([ox, oy]), k=1)
    lonlat = np.column_stack([a_grid.lon[ia], a_grid.lat[ia]])
    tb = cKDTree(np.column_stack([b_grid.lon, b_grid.lat]))
    _, ib = tb.query(lonlat, k=1)
    return np.column_stack([b_grid.pixel[ib], b_grid.scan[ib]])


def main() -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    ap = argparse.ArgumentParser(
        description="Run mirror-corrected anisotropic-affine registration on an "
                    "OHRC x TMC-2 pair using geolocation only for scoring.")
    ap.add_argument("--ohrc-label", required=True)
    ap.add_argument("--tmc-label", required=True)
    ap.add_argument("--name", required=True, help="short pair id used in file names")
    ap.add_argument("--region", required=True,
                    help="OHRC native px region as col0,col1,row0,row1")
    ap.add_argument("--flip-b", action="store_true",
                    help="flip TMC scan axis before search (set per-pair only if "
                         "run_p9_mirror_check.py reports the pair is mirrored)")
    ap.add_argument("--n_gt", type=int, default=300)
    ap.add_argument("--coarse", type=int, default=7)
    ap.add_argument("--fine", type=int, default=7)
    ap.add_argument("--out-dir", default="phase9_output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    col0, col1, row0, row1 = (int(x) for x in args.region.split(","))

    a = parse_label(args.ohrc_label)
    b = parse_label(args.tmc_label)
    if a is None or b is None:
        print("FAIL: could not parse one or both labels")
        return 1
    ag = read_geolocation_csv(a.geometry_csv_path)
    bg = read_geolocation_csv(b.geometry_csv_path)
    res_ratio = b.pixel_resolution_m / a.pixel_resolution_m
    print(f"PAIR {args.name}: {a.product_id} (OHRC {a.pixel_resolution_m} m/px)   vs   "
          f"{b.product_id} (TMC-2 {b.pixel_resolution_m} m/px)")
    print(f"resolution ratio ~ {res_ratio:.2f}x  | flip_b={args.flip_b}")

    # ---- build equal-res A (downsampled OHRC of R) and B (TMC crop of R) ----
    o_w = col1 - col0
    o_h = row1 - row0
    ds = res_ratio
    disp_w = max(int(round(o_w / ds)), 1)
    disp_h = max(int(round(o_h / ds)), 1)

    a_crop = to_uint8(load_image(a, row_range=(row0, row1), col_range=(col0, col1)))
    a_disp = cv2.resize(a_crop, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    corners_ohrc = np.array([
        [col0, row0], [col1, row0], [col1, row1], [col0, row1],
    ], float)
    b_corners = ohrc_to_tmc_grid(ag, bg, corners_ohrc[:, 0], corners_ohrc[:, 1])
    pad = 100
    bc0 = max(int(np.floor(b_corners[:, 0].min())) - pad, 0)
    bc1 = min(int(np.ceil(b_corners[:, 0].max())) + pad, b.samples)
    br0 = max(int(np.floor(b_corners[:, 1].min())) - pad, 0)
    br1 = min(int(np.ceil(b_corners[:, 1].max())) + pad, b.lines)
    disp_to_raw_x = (bc1 - bc0) / disp_w
    disp_to_raw_y = (br1 - br0) / disp_h
    print(f"A display ({disp_w}x{disp_h}) from OHRC region "
          f"cols[{col0}:{col1}] rows[{row0}:{row1}]")
    print(f"B (TMC-2) region cols[{bc0}:{bc1}] rows[{br0}:{br1}]")
    b_crop = to_uint8(load_image(b, row_range=(br0, br1), col_range=(bc0, bc1)))
    b_disp_raw = cv2.resize(b_crop, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    b_disp = b_disp_raw
    if args.flip_b:
        b_disp = b_disp_raw[::-1, :].copy()
        print("flipping B scan axis for search (mirror hypothesis)")

    # ---- run anisotropic search ----
    t0 = time.time()
    res = search(a_disp, b_disp, coarse_each=args.coarse, fine_each=args.fine)
    dt = time.time() - t0
    print(f"\nanisotropic search recovered (in {dt:.1f}s, {res.n_eval} "
          f"phase-correlations):")
    print(f"  sx={res.sx:.4f} sy={res.sy:.4f} theta={res.theta_deg:.2f} "
          f"t=({res.tx:.1f},{res.ty:.1f}) PSR={res.psr:.2f} "
          f"tPSR={res.translation_psr:.2f}")

    # ---- score against geolocation ground truth (display space) ----
    rng = np.random.default_rng(7)
    n = args.n_gt
    a_sx = rng.integers(0, disp_w, n).astype(float)
    a_sy = rng.integers(0, disp_h, n).astype(float)
    ds_y = o_h / disp_h
    ds_x = o_w / disp_w
    o_col = col0 + a_sx * ds_x
    o_row = row0 + a_sy * ds_y
    b_tmc = ohrc_to_tmc_grid(ag, bg, o_col, o_row)
    b_disp_gt = np.column_stack([
        (b_tmc[:, 0] - bc0) / disp_to_raw_x,
        (b_tmc[:, 1] - br0) / disp_to_raw_y,
    ])

    a_pts = np.column_stack([a_sx, a_sy])
    pred = res.H_apply @ np.column_stack([a_pts, np.ones(n)]).T
    pred = (pred / pred[2])[:2].T
    if args.flip_b:
        pred = pred.copy()
        pred[:, 1] = (disp_h - 1) - pred[:, 1]
    errs = np.hypot(pred[:, 0] - b_disp_gt[:, 0], pred[:, 1] - b_disp_gt[:, 1])

    print(f"\nScoring anisotropic transform vs geolocation GT ({n} points, "
          f"B-display px):")
    for tqq in [50.0, 100.0, 300.0]:
        k = int((errs <= tqq).sum())
        print(f"  within {tqq:5.0f} px : {k}/{n} ({100.0 * k / n:.1f}%)")
    print(f"  error px min/median/mean/max: "
          f"{errs.min():.1f} / {np.median(errs):.1f} / {errs.mean():.1f} / "
          f"{errs.max():.1f}")

    # ---- visual ----
    Mvis = res.H_apply[:2, :].copy()
    b_shift_disp = cv2.warpAffine(a_disp, np.float64(Mvis),
                                  (b_disp_raw.shape[1], b_disp_raw.shape[0]))
    if args.flip_b:
        b_shift_disp = b_shift_disp[::-1, :].copy()
    diff = cv2.absdiff(b_disp_raw, b_shift_disp)
    save_preview(np.hstack([a_disp, b_disp_raw, b_shift_disp, diff]),
                 out_dir / f"ov_{args.name}_{'flip' if args.flip_b else 'nof'}.png",
                 max_dim=2400)

    # ---- persist ----
    rec = {
        "pair_id": args.name,
        "products": [a.product_id, b.product_id],
        "flip_b": bool(args.flip_b),
        "resolution_ratio_x": round(res_ratio, 3),
        "ohrc_region_px": {"cols": [col0, col1], "rows": [row0, row1]},
        "a_display": [disp_w, disp_h],
        "b_tmc_region_px": {"cols": [bc0, bc1], "rows": [br0, br1]},
        "recovered": res.summary_dict(),
        "gt_scoring": {
            "n_points": n,
            "within_50px": int((errs <= 50).sum()),
            "within_100px": int((errs <= 100).sum()),
            "within_300px": int((errs <= 300).sum()),
            "error_px": {"min": float(errs.min()), "median": float(np.median(errs)),
                         "mean": float(errs.mean()), "max": float(errs.max())},
        },
    }
    rec_path = out_dir / f"result_{args.name}_{'flip' if args.flip_b else 'nof'}.json"
    with open(rec_path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    print(f"wrote {rec_path}")
    print(f"\nVERDICT {args.name}: sx={res.sx:.3f} sy={res.sy:.3f} "
          f"theta={res.theta_deg:.2f} PSR={res.psr:.1f} "
          f"GT median={np.median(errs):.1f}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
