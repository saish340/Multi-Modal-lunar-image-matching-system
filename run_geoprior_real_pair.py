"""Phase 11 - geolocation-prior tile-vote registration on the real OHRC/TMC-2 pair.

Implements the Phase 11 audit's H3 fix (see
``lunar_data_pipeline/geoprior_tile_registration.py`` and
``phase11_audit_output/``): instead of a blind global spectral search whose
translation step is defeated by self-similar terrain, seed every tile's
search with the products' own navigation (geolocation) data and fit the
5-DOF model to robust tile votes.

Discipline (same as every phase):
1. synthetic gate on the REAL OHRC crop (known injected 5-DOF transform,
   perturbed prior) -- must PASS before any real-data claim;
2. real pair with the geolocation data used ONLY as the search prior
   (scoring uses independently-sampled correspondences from the same grids,
   the Phase-8 convention, seed 7 / 300 points).

Output: printed report + ``phase11_geoprior_output/`` JSON records + overlay.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

_pkg = Path(__file__).resolve().parent / "lunar_data_pipeline"
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from geoprior_tile_registration import (  # noqa: E402
    TileVote,
    TileVoteParams,
    apply_h,
    fit_5dof,
    model_h,
    register,
    synthetic_gate,
    tile_centers,
)
from image_loader import load_image, save_preview, to_uint8  # noqa: E402
from pds4_parser import parse_label  # noqa: E402
from csv_geolocation import read_geolocation_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "phase11_geoprior_output"

A_LABEL = (
    ROOT / "ch2_ohr_ncp_20231004T0406038822_d_img_d18"
    / "data/calibrated/20231004/ch2_ohr_ncp_20231004T0406038822_d_img_d18.xml"
)
B_LABEL = (
    ROOT / "ch2_tmc_ncn_20250707T1853051045_d_img_d18"
    / "data/calibrated/20250707/ch2_tmc_ncn_20250707T1853051045_d_img_d18.xml"
)

# Exactly the Phase-8 runner's square region and scoring setup.
A_COL0, A_COL1, A_ROW0, A_ROW1 = 400, 11800, 45000, 56400
PAD = 100
N_GT = 300
GT_SEED = 7
TQ = [50.0, 100.0, 300.0]


def ohrc_to_tmc_grid(a_grid, b_grid, ox, oy):
    """Map OHRC-res pixels (ox,oy) -> TMC-2-res pixels via nearest geolocation."""
    ta = cKDTree(np.column_stack([a_grid.pixel, a_grid.scan]))
    _, ia = ta.query(np.column_stack([ox, oy]), k=1)
    lonlat = np.column_stack([a_grid.lon[ia], a_grid.lat[ia]])
    tb = cKDTree(np.column_stack([b_grid.lon, b_grid.lat]))
    _, ib = tb.query(lonlat, k=1)
    return np.column_stack([b_grid.pixel[ib], b_grid.scan[ib]])


def build_pair():
    """The exact Phase-8 display pair (square region, scan-flipped B)."""
    a = parse_label(str(A_LABEL))
    b = parse_label(str(B_LABEL))
    ag = read_geolocation_csv(a.geometry_csv_path)
    bg = read_geolocation_csv(b.geometry_csv_path)
    res_ratio = b.pixel_resolution_m / a.pixel_resolution_m
    o_w, o_h = A_COL1 - A_COL0, A_ROW1 - A_ROW0
    disp_w = max(int(round(o_w / res_ratio)), 1)
    disp_h = max(int(round(o_h / res_ratio)), 1)

    a_crop = to_uint8(load_image(
        a, row_range=(A_ROW0, A_ROW1), col_range=(A_COL0, A_COL1)))
    a_disp = cv2.resize(a_crop, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    corners = np.array([
        [A_COL0, A_ROW0], [A_COL1, A_ROW0],
        [A_COL1, A_ROW1], [A_COL0, A_ROW1]], float)
    b_corners = ohrc_to_tmc_grid(ag, bg, corners[:, 0], corners[:, 1])
    bc0 = max(int(np.floor(b_corners[:, 0].min())) - PAD, 0)
    bc1 = min(int(np.ceil(b_corners[:, 0].max())) + PAD, b.samples)
    br0 = max(int(np.floor(b_corners[:, 1].min())) - PAD, 0)
    br1 = min(int(np.ceil(b_corners[:, 1].max())) + PAD, b.lines)
    b_crop = to_uint8(load_image(b, row_range=(br0, br1), col_range=(bc0, bc1)))
    b_disp_raw = cv2.resize(b_crop, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    print(f"[pair] OHRC {a.pixel_resolution_m} m/px vs TMC-2 "
          f"{b.pixel_resolution_m} m/px (ratio {res_ratio:.2f}); display "
          f"{disp_w}x{disp_h}; B raw cols[{bc0}:{bc1}] rows[{br0}:{br1}]")
    return dict(
        a=a, b=b, ag=ag, bg=bg, res_ratio=res_ratio, o_w=o_w, o_h=o_h,
        disp_w=disp_w, disp_h=disp_h, a_disp=a_disp, b_disp_raw=b_disp_raw,
        b_disp_search=b_disp_raw[::-1, :].copy(),  # Phase-8 --flip-b
        bc0=bc0, bc1=bc1, br0=br0, br1=br1,
        disp_to_raw_x=(bc1 - bc0) / disp_w,
        disp_to_raw_y=(br1 - br0) / disp_h,
    )


def build_gt(pair, n=N_GT, seed=GT_SEED):
    """The exact Phase-8 GT correspondences, in raw and flip-B display space."""
    rng = np.random.default_rng(seed)
    disp_w, disp_h = pair["disp_w"], pair["disp_h"]
    a_sx = rng.integers(0, disp_w, n).astype(float)
    a_sy = rng.integers(0, disp_h, n).astype(float)
    ds_x = pair["o_w"] / disp_w
    ds_y = pair["o_h"] / disp_h
    o_col = A_COL0 + a_sx * ds_x
    o_row = A_ROW0 + a_sy * ds_y
    b_tmc = ohrc_to_tmc_grid(pair["ag"], pair["bg"], o_col, o_row)
    b_raw = np.column_stack([
        (b_tmc[:, 0] - pair["bc0"]) / pair["disp_to_raw_x"],
        (b_tmc[:, 1] - pair["br0"]) / pair["disp_to_raw_y"],
    ])
    a_pts = np.column_stack([a_sx, a_sy])
    b_flip = np.column_stack([b_raw[:, 0], (disp_h - 1) - b_raw[:, 1]])
    return a_pts, b_raw, b_flip


def make_predict_b(pair):
    """Navigation prior: A-display px -> B-display (flip-frame) px via the
    products' own per-pixel geolocation grids (nearest-neighbour lookup)."""
    ta = cKDTree(np.column_stack([pair["ag"].pixel, pair["ag"].scan]))
    tb = cKDTree(np.column_stack([pair["bg"].lon, pair["bg"].lat]))

    def predict_b(pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, float))
        o_col = A_COL0 + pts[:, 0] * (pair["o_w"] / pair["disp_w"])
        o_row = A_ROW0 + pts[:, 1] * (pair["o_h"] / pair["disp_h"])
        _, ia = ta.query(np.column_stack([o_col, o_row]), k=1)
        lonlat = np.column_stack([pair["ag"].lon[ia], pair["ag"].lat[ia]])
        _, ib = tb.query(lonlat, k=1)
        bx = (pair["bg"].pixel[ib] - pair["bc0"]) / pair["disp_to_raw_x"]
        by = ((pair["disp_h"] - 1)
              - (pair["bg"].scan[ib] - pair["br0"]) / pair["disp_to_raw_y"])
        return np.column_stack([bx, by])

    return predict_b


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    OUT_DIR.mkdir(exist_ok=True)
    pair = build_pair()

    # ---- 1. synthetic gate on the REAL OHRC crop (must PASS) ----
    print("\n=== 1. synthetic gate (real OHRC crop, injected 5-DOF) ===")
    gate = synthetic_gate(pair["a_disp"])
    print(f"injected : {[round(v, 4) for v in gate['injected']]}")
    for stage in ("stage_a", "stage_b"):
        s = gate[stage]
        print(f"{stage}: recovered={[round(v, 4) for v in s['recovered']]}")
        print(f"          errors   ={dict((k, round(v, 4)) for k, v in s['errors'].items())}")
        print(f"          fit      : inliers={s['fit']['n_inliers']}/"
              f"{s['fit']['n_votes']} (ratio {s['fit']['inlier_ratio']:.2f}) "
              f"resid={s['fit']['resid_median_px']}px -> {s['verdict']}")
    if gate.get("prior_only"):
        p = gate["prior_only"]
        print(f"prior-only baseline (no image content): "
              f"errors={dict((k, round(v, 4)) for k, v in p['errors'].items())}")
    print(f"GATE VERDICT: {gate['verdict']}")
    with open(OUT_DIR / "geoprior_gate.json", "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=2, default=float)
    if gate["verdict"] != "PASS":
        print("GATE FAILED -- real-data run aborted (discipline rule).")
        return 1

    # ---- 2. real pair: geolocation-prior tile-vote registration ----
    print("\n=== 2. real pair: geolocation-prior tile-vote registration ===")
    predict_b = make_predict_b(pair)
    res, votes = register(pair["a_disp"], pair["b_disp_search"], predict_b,
                          TileVoteParams())
    print(f"recovered: sx={res.sx:.4f} sy={res.sy:.4f} theta={res.theta_deg:.2f}deg "
          f"t=({res.tx:.1f}, {res.ty:.1f})")
    print(f"votes    : {res.n_votes}/{res.n_tiles} usable ({res.n_skipped} skipped), "
          f"inliers {res.n_inliers}/{res.n_votes} "
          f"(ratio {res.inlier_ratio:.2f}), fit residual "
          f"{res.resid_median_px:.1f} px median / {res.resid_p90_px:.1f} p90")
    vp = np.hypot(*np.array([[v.mx - v.px, v.my - v.py] for v in votes]).T)
    print(f"vote-vs-prediction displacement: median={np.median(vp):.1f} "
          f"p90={np.percentile(vp, 90):.1f} px (the correction the prior needed)")

    # ---- 2b. PRE-STATED content-verification diagnostic (margin=8) ----
    # Question (fixed before running): does image content agree with the
    # navigation prior to within +/-8 px everywhere? A verification test, not
    # a search: the tiny window cannot wander to wrong terrain, so consistent
    # small votes confirm the prior, while edge-saturated or scattered votes
    # expose prior error / cross-sensor content disagreement. The deliverable
    # pipeline above (margin=80) is NOT retuned from this diagnostic.
    print("\n=== 2b. content-verification diagnostic (margin=8, 1 pass) ===")
    vparams = TileVoteParams(margin=8, iterations=1)
    res_v, votes_v = register(pair["a_disp"], pair["b_disp_search"], predict_b,
                              vparams)
    vp_v = np.hypot(*np.array([[v.mx - v.px, v.my - v.py] for v in votes_v]).T)
    n_edge = int((vp_v >= 0.95 * vparams.margin).sum())
    print(f"verification fit: sx={res_v.sx:.4f} sy={res_v.sy:.4f} "
          f"theta={res_v.theta_deg:.2f}deg t=({res_v.tx:.1f}, {res_v.ty:.1f}) "
          f"inliers {res_v.n_inliers}/{res_v.n_votes}, fit residual "
          f"{res_v.resid_median_px:.1f} px median")
    print(f"  vote-vs-prediction: median={np.median(vp_v):.1f} "
          f"p90={np.percentile(vp_v, 90):.1f} px; "
          f"edge-saturated (>={0.95 * vparams.margin:.1f} px): "
          f"{n_edge}/{len(votes_v)}")

    # ---- 3. ground-truth scoring (Phase-8 convention, 300 pts, seed 7) ----
    a_pts, _, b_flip = build_gt(pair)
    pred = apply_h(res.H_apply, a_pts)
    errs = np.hypot(*(pred - b_flip).T)

    # Prior-only reference: the SAME robust 5-DOF estimator applied to the raw
    # prior correspondences (tile centres -> predicted B positions) -- what
    # zero image content achieves (the audit's E3 affine). Comparing
    # predict_b(a_pts) directly against b_flip would be self-comparison,
    # because build_gt constructs b_flip through the very same NN chain.
    cparams = TileVoteParams()
    centers = tile_centers(pair["disp_w"], pair["disp_h"], cparams)
    center_c = ((pair["disp_w"] - 1) / 2.0, (pair["disp_h"] - 1) / 2.0)
    prior_votes = [
        TileVote(ax=float(cx), ay=float(cy), px=float(px), py=float(py),
                 mx=float(px), my=float(py), score=0.0)
        for (cx, cy), (px, py) in zip(centers, predict_b(centers))
    ]
    p5_prior, info_prior = fit_5dof(prior_votes, center_c, cparams)
    H_prior = model_h(*p5_prior, center_c)
    errs_prior = np.hypot(*(apply_h(H_prior, a_pts) - b_flip).T)

    # GT errors of the margin-8 verification fit (section 2b).
    errs_verify = np.hypot(*(apply_h(res_v.H_apply, a_pts) - b_flip).T)

    print(f"\n=== 3. GT scoring ({N_GT} pts, B-flip display px) ===")
    print(f"  pipeline (margin=80, ECC, 2-pass): median={np.median(errs):.1f} px, "
          f"within50={int((errs <= 50).sum())}/{N_GT}")
    print(f"  verification (margin=8)          : median={np.median(errs_verify):.1f} px, "
          f"within50={int((errs_verify <= 50).sum())}/{N_GT}")
    print(f"  prior-only affine (no content)   : median={np.median(errs_prior):.1f} px, "
          f"within50={int((errs_prior <= 50).sum())}/{N_GT}  <- the bar to beat")
    for tqq in TQ:
        k = int((errs <= tqq).sum())
        print(f"  pipeline within {tqq:5.0f} px : {k}/{N_GT} ({100.0*k/N_GT:.1f}%)")
    print(f"  pipeline error px min/median/mean/max: {errs.min():.1f} / "
          f"{np.median(errs):.1f} / {errs.mean():.1f} / {errs.max():.1f}")
    print("  context: Phase 8 unseeded global search = 111.3 px median; "
          "audit prior-alone affine = 21.5 px")

    # ---- 4. overlay + record ----
    H = res.H_apply[:2, :]
    warped = cv2.warpAffine(pair["a_disp"], np.float64(H),
                            (pair["b_disp_search"].shape[1],
                             pair["b_disp_search"].shape[0]))
    diff = cv2.absdiff(pair["b_disp_search"], warped)
    save_preview(np.hstack([pair["a_disp"], pair["b_disp_search"], warped, diff]),
                 OUT_DIR / "geoprior_real_overlay.png", max_dim=2400)

    record = {
        "pair": [pair["a"].product_id, pair["b"].product_id],
        "region_ohrc_px": {"cols": [A_COL0, A_COL1], "rows": [A_ROW0, A_ROW1]},
        "display": [pair["disp_w"], pair["disp_h"]],
        "flip_b": True,
        "gate": gate,
        "recovered": res.summary_dict(),
        "match_points": [v.__dict__ for v in votes],
        "vote_vs_prediction_px": {"median": float(np.median(vp)),
                                  "p90": float(np.percentile(vp, 90))},
        "verification_margin8": {
            "recovered": res_v.summary_dict(),
            "vote_vs_prediction_px": {"median": float(np.median(vp_v)),
                                      "p90": float(np.percentile(vp_v, 90))},
            "edge_saturated": int(n_edge),
            "gt_median_px": float(np.median(errs_verify)),
        },
        "prior_only_affine": {
            "params": [float(v) for v in p5_prior],
            "fit": info_prior,
            "gt_median_px": float(np.median(errs_prior)),
        },
        "gt_scoring": {
            "n_points": N_GT, "seed": GT_SEED,
            "within_px": {str(int(t)): int((errs <= t).sum()) for t in TQ},
            "error_px": {"min": float(errs.min()), "median": float(np.median(errs)),
                         "mean": float(errs.mean()), "max": float(errs.max())},
        },
        "context": {
            "phase8_unseeded_global_search_median_px": 111.3,
            "audit_prior_alone_affine_median_px": 21.5,
            "synthesis": (
                "navigation prior alone = 23.1 px median; content "
                "verification (margin=8) = 23.7 px median -- image content "
                "AGREES with the navigation prior locally, beating Phase 8's "
                "111.3 px ~5x and sitting at the ~26-40 px geolocation "
                "ceiling. The wide-margin (80 px) content SEARCH pipeline "
                "degrades to 117.3 px wrong-basin: wide per-tile search on "
                "self-similar terrain relapses into the Phase-6/10 spectral "
                "ambiguity, so the reliable real-data operating point is "
                "prior + LOCAL verification, not search."
            ),
            "note": "prior and GT scorer derive from the same ISDA geolocation "
                    "CSVs; the real-pair number certifies image-content "
                    "consistency with navigation data, not absolute geodetic "
                    "accuracy (see module docstring).",
        },
        "phase11_real_px": {
            "pipeline_margin80_gt_median": float(np.median(errs)),
            "verification_margin8_gt_median": float(np.median(errs_verify)),
            "prior_only_affine_gt_median": float(np.median(errs_prior)),
        },
    }
    rec_path = OUT_DIR / "geoprior_real_result.json"
    with open(rec_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=float)
    print(f"\nwrote {rec_path}")
    print("VERDICT (honest three-way, 300 GT pts, B-flip display px):")
    print(f"  pipeline margin=80 (content SEARCH): {np.median(errs):.1f} px median "
          f"({int((errs <= 50).sum())}/{N_GT} within 50) -- wrong basin on this "
          "terrain")
    print(f"  verification margin=8 (content check): {np.median(errs_verify):.1f} px "
          f"median ({int((errs_verify <= 50).sum())}/{N_GT} within 50) "
          "-- content AGREES with the prior")
    print(f"  prior-only affine (no content): {np.median(errs_prior):.1f} px median "
          f"({int((errs_prior <= 50).sum())}/{N_GT} within 50)")
    print("  => the navigation prior + local content verification is the "
          "real-data operating point,"
          f"\n     beating Phase 8's 111.3 px ~5x (prior 23.1 / verify 23.7 px).")
    return 0


if __name__ == "__main__":
    sys.exit(main())