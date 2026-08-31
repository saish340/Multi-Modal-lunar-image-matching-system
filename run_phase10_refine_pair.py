"""Phase 10 - local refinement on a real OHRC x TMC-2 pair (before/after).

Runs the SAME mirror-corrected anisotropic-affine pipeline twice on one real
pair, with ``search(refine=False)`` (the Phase 8/9 coarse grid search, kept as
the baseline) and ``search(refine=True)`` (Phase 10: continuous Nelder-Mead PSR
refinement on top). Both are scored against the SAME geolocation GT points in
display space (the Phase 8/9 fix), so the two numbers are directly comparable.

This isolates the refinement's contribution: any accuracy change is due to the
refinement stage alone, since the coarse search, mirror handling, region, and
scoring are identical in both runs.
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
    ta = cKDTree(np.column_stack([a_grid.pixel, a_grid.scan]))
    _, ia = ta.query(np.column_stack([ox, oy]), k=1)
    lonlat = np.column_stack([a_grid.lon[ia], a_grid.lat[ia]])
    tb = cKDTree(np.column_stack([b_grid.lon, b_grid.lat]))
    _, ib = tb.query(lonlat, k=1)
    return np.column_stack([b_grid.pixel[ib], b_grid.scan[ib]])


def main() -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    ap = argparse.ArgumentParser(
        description="Phase 10: coarse vs coarse+refine on a real pair.")
    ap.add_argument("--ohrc-label", required=True)
    ap.add_argument("--tmc-label", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--flip-b", action="store_true")
    ap.add_argument("--n_gt", type=int, default=300)
    ap.add_argument("--coarse", type=int, default=7)
    ap.add_argument("--fine", type=int, default=7)
    ap.add_argument("--out-dir", default="phase10_output")
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

    def score(res, a_pts, n, flip):
        pred = res.H_apply @ np.column_stack([a_pts, np.ones(n)]).T
        pred = (pred / pred[2])[:2].T
        if flip:
            pred = pred.copy()
            pred[:, 1] = (disp_h - 1) - pred[:, 1]
        return np.hypot(pred[:, 0] - gt_b_disp[:, 0], pred[:, 1] - gt_b_disp[:, 1])

    # ---- fixed GT points shared by both runs ----
    rng = np.random.default_rng(7)
    n = args.n_gt
    a_sx = rng.integers(0, disp_w, n).astype(float)
    a_sy = rng.integers(0, disp_h, n).astype(float)
    ds_y = o_h / disp_h
    ds_x = o_w / disp_w
    o_col = col0 + a_sx * ds_x
    o_row = row0 + a_sy * ds_y
    b_tmc = ohrc_to_tmc_grid(ag, bg, o_col, o_row)
    gt_b_disp = np.column_stack([
        (b_tmc[:, 0] - bc0) / disp_to_raw_x,
        (b_tmc[:, 1] - br0) / disp_to_raw_y,
    ])
    a_pts = np.column_stack([a_sx, a_sy])

    results = {}
    for mode, refine_flag in [("coarse", False), ("refined", True)]:
        t0 = time.time()
        res = search(a_disp, b_disp, coarse_each=args.coarse, fine_each=args.fine,
                     refine=refine_flag)
        dt = time.time() - t0
        errs = score(res, a_pts, n, args.flip_b)
        results[mode] = {"res": res, "errs": errs}
        tag = args.name + ("_flip" if args.flip_b else "_nof")
        Mvis = res.H_apply[:2, :].copy()
        b_shift = cv2.warpAffine(a_disp, np.float64(Mvis),
                                 (b_disp_raw.shape[1], b_disp_raw.shape[0]))
        if args.flip_b:
            b_shift = b_shift[::-1, :].copy()
        save_preview(
            np.hstack([a_disp, b_disp_raw, b_shift,
                       cv2.absdiff(b_disp_raw, b_shift)]),
            out_dir / f"ov_{tag}_{mode}.png", max_dim=2400)
        print(f"\n[{mode}] sx={res.sx:.4f} sy={res.sy:.4f} theta={res.theta_deg:.2f} "
              f"t=({res.tx:.1f},{res.ty:.1f}) PSR={res.psr:.2f} "
              f"({res.n_eval} evals, {res.n_refine} refine, {dt:.1f}s)")
        print(f"  within 50/100/300: "
              f"{int((errs<=50).sum())}/{int((errs<=100).sum())}/{int((errs<=300).sum())} / {n}"
              f"   min/med/mean/max: {errs.min():.1f}/{np.median(errs):.1f}/"
              f"{errs.mean():.1f}/{errs.max():.1f}")

    crs, rfd = results["coarse"], results["refined"]
    m0 = float(np.median(crs["errs"]))
    m1 = float(np.median(rfd["errs"]))
    print("\n" + "=" * 66)
    print(f"PHASE 10 before/after (median display-px error, {n} shared GT pts):")
    print(f"  coarse (Phase 8/9)  : {m0:.1f} px")
    print(f"  coarse + refinement : {m1:.1f} px   ({m1 - m0:+.1f} px, "
          f"{100.0*(m1-m0)/max(m0,1):+.1f}%)")
    print(f"  geolocation ceiling : see GT-grid quant floor for this pair")
    print("=" * 66)

    rec = {
        "pair_id": args.name,
        "products": [a.product_id, b.product_id],
        "flip_b": bool(args.flip_b),
        "resolution_ratio_x": round(res_ratio, 3),
        "ohrc_region_px": {"cols": [col0, col1], "rows": [row0, row1]},
        "a_display": [disp_w, disp_h],
        "b_tmc_region_px": {"cols": [bc0, bc1], "rows": [br0, br1]},
        "n_gt": n,
        "coarse": {
            "recovered": crs["res"].summary_dict(),
            "gt_scoring": {
                "within_50px": int((crs["errs"] <= 50).sum()),
                "within_100px": int((crs["errs"] <= 100).sum()),
                "within_300px": int((crs["errs"] <= 300).sum()),
                "error_px": {"min": float(crs["errs"].min()),
                             "median": float(np.median(crs["errs"])),
                             "mean": float(crs["errs"].mean()),
                             "max": float(crs["errs"].max())},
            },
        },
        "refined": {
            "recovered": rfd["res"].summary_dict(),
            "gt_scoring": {
                "within_50px": int((rfd["errs"] <= 50).sum()),
                "within_100px": int((rfd["errs"] <= 100).sum()),
                "within_300px": int((rfd["errs"] <= 300).sum()),
                "error_px": {"min": float(rfd["errs"].min()),
                             "median": float(np.median(rfd["errs"])),
                             "mean": float(rfd["errs"].mean()),
                             "max": float(rfd["errs"].max())},
            },
        },
        "before_after": {"coarse_median_px": m0, "refined_median_px": m1,
                         "delta_px": m1 - m0,
                         "delta_pct": 100.0 * (m1 - m0) / max(m0, 1.0)},
    }
    rec_path = out_dir / f"result_{args.name}_{'flip' if args.flip_b else 'nof'}.json"
    with open(rec_path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    print(f"wrote {rec_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
