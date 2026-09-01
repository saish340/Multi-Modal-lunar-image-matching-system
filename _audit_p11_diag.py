"""Phase 11 audit: independent verification of the Phase 8/10 diagnosis.

Re-derives, in one place and in the exact flip-B scoring space of Phase 8,
the numbers the Phase 8/10 conclusions rest on, then tests hypotheses the
existing diagnosis leaves open:

H1 (model ceiling): compare the Phase-8 result's GT median error against the
    BEST transform in the very model being searched (5-DOF
    R(theta)diag(sx,sy)+t), fitted directly to the geolocation GT
    correspondences in the same space.  Phase 8/10 quote a "~26-40 px
    ceiling" measured in raw TMC pixels on the full strip; the searched-model
    ceiling in the exact scoring space was never tabulated next to it.

H2 (objective blindness): the phase-correlation PSR used as the global
    search objective may be unable to rank the true alignment above the
    found wrong one on real data.  Pure spectrum whitening amplifies bands
    where the two images share no content (here most of B's canvas is
    non-matching pad/bbox slack), and PSR's mean/std normalisation is
    candidate-dependent.  We evaluate the current PSR plus three alternative
    objectives (high-pass PSR, magnitude-thresholded POC PSR, masked NCC) at
    BOTH the found and the geolocation-true parameters, along a parameter
    transect between them, and on a local grid around the true parameters.

H3 (fix prototype): tile-wise NCC/POC displacement voting seeded by the
    per-product geolocation grids (navigation-prior practice), followed by a
    robust 5-DOF fit, scored against the same GT.

Outputs: printed tables + ``phase11_audit_output/audit_diag_results.json``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

_pkg = Path(__file__).resolve().parent / "lunar_data_pipeline"
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from anisotropic_affine_search import (  # noqa: E402
    refine_transform,
    resample_for_search,
    search,
    warp_content,
)
from phase_correlation import hann2d, phase_correlate  # noqa: E402
from image_loader import load_image, to_uint8  # noqa: E402
from pds4_parser import parse_label  # noqa: E402
from csv_geolocation import read_geolocation_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "phase11_audit_output"

A_LABEL = (
    ROOT / "ch2_ohr_ncp_20231004T0406038822_d_img_d18"
    / "data/calibrated/20231004/ch2_ohr_ncp_20231004T0406038822_d_img_d18.xml"
)
B_LABEL = (
    ROOT / "ch2_tmc_ncn_20250707T1853051045_d_img_d18"
    / "data/calibrated/20250707/ch2_tmc_ncn_20250707T1853051045_d_img_d18.xml"
)
P8_JSON = ROOT / "phase8_aniso_output" / "aniso_real_result_square_flip.json"

# Exactly the Phase-8 runner's square region and scoring setup.
A_COL0, A_COL1, A_ROW0, A_ROW1 = 400, 11800, 45000, 56400
PAD = 100
N_GT = 300
GT_SEED = 7

# Tile-vote prototype constants (fixed a priori, see H3 in the docstring).
N_TILES = 6          # 6x6 tile grid over the display
TILE_PX = 56         # tile side in display px (~340 m ground)
MARGIN_PX = 80       # search half-window around the geolocation prediction
MIN_NCC = 0.45       # vote quality threshold (a priori)
POC_PSR_MIN = 8.0    # fallback vote quality if NCC votes are too few


def ohrc_to_tmc_grid(a_grid, b_grid, ox, oy):
    """Map OHRC-res pixels (ox,oy) -> TMC-2-res pixels via nearest geolocation."""
    ta = cKDTree(np.column_stack([a_grid.pixel, a_grid.scan]))
    _, ia = ta.query(np.column_stack([ox, oy]), k=1)
    lonlat = np.column_stack([a_grid.lon[ia], a_grid.lat[ia]])
    tb = cKDTree(np.column_stack([b_grid.lon, b_grid.lat]))
    _, ib = tb.query(lonlat, k=1)
    return np.column_stack([b_grid.pixel[ib], b_grid.scan[ib]])


def build_pair():
    """Rebuild the Phase-8 display pair exactly (square region, flip_b)."""
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
    disp_to_raw_x = (bc1 - bc0) / disp_w
    disp_to_raw_y = (br1 - br0) / disp_h
    b_disp_raw = cv2.resize(b_crop, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    b_disp_search = b_disp_raw[::-1, :].copy()  # Phase-8 --flip-b
    return dict(
        a=a, b=b, ag=ag, bg=bg, res_ratio=res_ratio, o_w=o_w, o_h=o_h,
        disp_w=disp_w, disp_h=disp_h, a_disp=a_disp, b_disp_raw=b_disp_raw,
        b_disp_search=b_disp_search, bc0=bc0, bc1=bc1, br0=br0, br1=br1,
        disp_to_raw_x=disp_to_raw_x, disp_to_raw_y=disp_to_raw_y,
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


def apply_h(h, pts):
    hom = np.column_stack([pts, np.ones(len(pts))])
    out = (h @ hom.T).T
    return out[:, :2] / out[:, 2:3]


def flip_matrix(disp_h):
    """3x3 scan-axis reflection used by --flip-b (an involution)."""
    f = np.eye(3)
    f[1, 1] = -1.0
    f[1, 2] = disp_h - 1
    return f


def gt_errs(h, a_pts, b_gt):
    pred = apply_h(h, a_pts)
    return np.hypot(pred[:, 0] - b_gt[:, 0], pred[:, 1] - b_gt[:, 1])


def fit_affine6(a_pts, b_pts):
    """Least-squares full 6-DOF affine a->b."""
    amat = np.column_stack([a_pts, np.ones(len(a_pts))])
    sol, *_ = np.linalg.lstsq(amat, b_pts, rcond=None)
    h = np.eye(3)
    h[0, :], h[1, :] = sol[0], sol[1]
    return h


def model_h(sx, sy, th, tx, ty, center):
    """The search's 5-DOF model T(c+t) . R(th) . diag(sx,sy) . T(-c)."""
    t = math.radians(th)
    co, si = math.cos(t), math.sin(t)
    lin = np.array([[co, si], [-si, co]]) @ np.diag([sx, sy])
    h = np.eye(3)
    h[:2, :2] = lin
    h[0, 2], h[1, 2] = -center[0], -center[1]
    tt = np.eye(3)
    tt[0, 2], tt[1, 2] = center[0] + tx, center[1] + ty
    return tt @ h


def fit_model5(a_pts, b_pts, center, x0=None):
    """Least-squares fit of the 5-DOF search model to correspondences."""

    def resid(p):
        sx, sy, th, tx, ty = p
        h = model_h(sx, sy, th, tx, ty, center)
        pred = apply_h(h, a_pts)
        return (pred - b_pts).ravel()

    if x0 is None:
        off = (b_pts - a_pts).mean(axis=0)
        x0 = np.array([1.0, 1.0, 0.0, off[0], off[1]])
    r = least_squares(resid, x0, method="lm", max_nfev=20000)
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def err_structure(h, a_pts, b_gt, center):
    """Split the GT error field of h into constant-offset vs linear parts."""
    pred = apply_h(h, a_pts)
    e = pred - b_gt
    e0 = e.mean(axis=0)
    rel = a_pts - center
    amat = np.column_stack([rel, np.ones(len(a_pts))])
    sol, *_ = np.linalg.lstsq(amat, e, rcond=None)
    lin = amat @ sol
    return dict(
        e0=e0.tolist(),
        rms_total=float(np.sqrt((e ** 2).sum(axis=1).mean())),
        rms_after_mean=float(np.sqrt(((e - e0) ** 2).sum(axis=1).mean())),
        rms_linear_part=float(np.sqrt((lin ** 2).sum(axis=1).mean())),
        rms_after_linear=float(np.sqrt(((e - lin) ** 2).sum(axis=1).mean())),
    )


# ---------------------------------------------------------------------------
# Objective functions (H2).  All operate on (a, b) in the flip-B search frame.
# ---------------------------------------------------------------------------

def _pocs(a, b):
    """Raw normalised cross-power spectrum and its inverse surface."""
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    return cross


def score_current(a, b, sx, sy, th):
    """The exact Phase-8 scoring: Hann window + pure whitening PSR."""
    ws = resample_for_search(a, sx, sy, th)
    win = hann2d(ws.shape)
    ws = ws * win
    b_ = b * win
    dx, dy, peak, psr = phase_correlate(ws, b_)
    return dict(psr=float(psr), peak=float(peak), shift=(int(dx), int(dy)))


def score_highpass(a, b, sx, sy, th, sigma=4.0):
    """Hann window after a Gaussian high-pass (kills low-frequency dominance)."""
    ws = resample_for_search(a, sx, sy, th)
    win = hann2d(ws.shape)
    ws = (ws - cv2.GaussianBlur(ws, (0, 0), sigma)) * win
    b_ = (b - cv2.GaussianBlur(b, (0, 0), sigma)) * win
    dx, dy, peak, psr = phase_correlate(ws, b_)
    return dict(psr=float(psr), peak=float(peak), shift=(int(dx), int(dy)))


def score_bandlimit(a, b, sx, sy, th, tau=0.10):
    """Magnitude-thresholded whitening: bands with |cross| < tau*max keep a
    damped weight instead of being amplified to unit magnitude."""
    ws = resample_for_search(a, sx, sy, th)
    win = hann2d(ws.shape)
    ws = ws * win
    b_ = b * win
    cross = _pocs(ws, b_)
    mag = np.abs(cross)
    cross = cross / np.maximum(mag, tau * mag.max())
    cc = np.fft.ifft2(cross).real
    h, w = cc.shape
    idx = np.unravel_index(np.argmax(cc), cc.shape)
    peak = float(cc[idx])
    dy = idx[0] if idx[0] <= h // 2 else idx[0] - h
    dx = idx[1] if idx[1] <= w // 2 else idx[1] - w
    ex, ey = int(idx[1]), int(idx[0])
    m = np.ones(cc.shape, bool)
    m[max(ey - 3, 0):ey + 4, max(ex - 3, 0):ex + 4] = False
    side = cc[m]
    psr = (peak - side.mean()) / (side.std() + 1e-12)
    return dict(psr=float(psr), peak=peak, shift=(int(dx), int(dy)))


def score_ncc(a, b, sx, sy, th):
    """Masked NCC at the POC-recovered translation (candidate-fair, [-1,1])."""
    ws = resample_for_search(a, sx, sy, th)
    win = hann2d(ws.shape)
    dx, dy, _, _ = phase_correlate(ws * win, b * win)
    tx, ty = float(-dx), float(-dy)
    wa = warp_content(np.asarray(a, np.float32), sx, sy, th, tx, ty)
    valid = warp_content(np.ones(a.shape, np.float32), sx, sy, th, tx, ty) > 0.99
    if valid.sum() < 500:
        return dict(psr=float("nan"), peak=float("nan"), shift=(int(dx), int(dy)))
    va, vb = wa[valid], np.asarray(b, np.float32)[valid]
    va = va - va.mean()
    vb = vb - vb.mean()
    denom = np.sqrt((va * va).sum() * (vb * vb).sum()) + 1e-12
    return dict(psr=float((va * vb).sum() / denom), peak=float("nan"),
                shift=(int(dx), int(dy)))


OBJECTIVES = {
    "psr_current": score_current,
    "psr_highpass": score_highpass,
    "psr_bandlimit": score_bandlimit,
    "ncc_masked": score_ncc,
}


def eval_all(a, b, params, center):
    """Evaluate every objective at a 5-DOF parameter tuple."""
    sx, sy, th, tx, ty = params
    out = {}
    for name, fn in OBJECTIVES.items():
        r = fn(a, b, sx, sy, th)
        r["gt_shift_err"] = float(np.hypot(r["shift"][0] - tx, r["shift"][1] - ty))
        out[name] = r
    h = model_h(sx, sy, th, tx, ty, center)
    out["_h"] = h
    return out


def transect(a, b, p_found, p_true, center, n=11):
    """Objective values along the linear parameter path found->true."""
    p_f, p_t = np.asarray(p_found, float), np.asarray(p_true, float)
    dth = (p_t[2] - p_f[2] + 180.0) % 360.0 - 180.0
    p_t_adj = np.array([p_t[0], p_t[1], p_f[2] + dth, p_t[3], p_t[4]])
    rows = []
    for i in range(n):
        f = i / (n - 1)
        p = p_f + (p_t_adj - p_f) * f
        r = eval_all(a, b, p, center)
        rows.append(dict(fraction=f, params=p.tolist(),
                         **{k: r[k]["psr"] for k in OBJECTIVES}))
    return rows


def local_grid(a, b, p_true, center, dsx=0.03, dsy=0.03, dth=2.0, n=5):
    """Local objective grid around the true parameters."""
    offs = np.linspace(-1, 1, n)
    best = {k: (-1e9, None) for k in OBJECTIVES}
    rows = []
    for osx in offs * dsx * p_true[0]:
        for osy in offs * dsy * p_true[1]:
            for oth in offs * dth:
                p = (p_true[0] + osx, p_true[1] + osy,
                     p_true[2] + oth, p_true[3], p_true[4])
                r = eval_all(a, b, p, center)
                rows.append(dict(params=p[:3], **{k: r[k]["psr"] for k in OBJECTIVES}))
                for k in OBJECTIVES:
                    if r[k]["psr"] > best[k][0]:
                        best[k] = (r[k]["psr"], p[:3])
    return best, rows


# ---------------------------------------------------------------------------
# H3: tile-vote prototype (geolocation-seeded local refinement)
# ---------------------------------------------------------------------------

def tile_grid(disp_w, disp_h, n_tiles=N_TILES):
    """NxN tile centres spread over the A-safe interior [t2, disp-t2]."""
    t2 = TILE_PX // 2
    xs = np.linspace(t2, disp_w - t2, n_tiles)
    ys = np.linspace(t2, disp_h - t2, n_tiles)
    return np.array([(float(x), float(y)) for y in ys for x in xs])


def predict_b_centers(pair, a_centers):
    """Geolocation NN prediction of B (flip-frame display) positions."""
    ds_x = pair["o_w"] / pair["disp_w"]
    ds_y = pair["o_h"] / pair["disp_h"]
    o_col = A_COL0 + a_centers[:, 0] * ds_x
    o_row = A_ROW0 + a_centers[:, 1] * ds_y
    b_tmc = ohrc_to_tmc_grid(pair["ag"], pair["bg"], o_col, o_row)
    bx = (b_tmc[:, 0] - pair["bc0"]) / pair["disp_to_raw_x"]
    by = ((pair["disp_h"] - 1)
          - (b_tmc[:, 1] - pair["br0"]) / pair["disp_to_raw_y"])
    return np.column_stack([bx, by])


def _parabolic(res, maxloc):
    """2-D parabolic sub-pixel peak offset, clamped to +/-1 px."""
    x, y = maxloc
    dx = dy = 0.0
    if 0 < x < res.shape[1] - 1:
        dl, d0, dr = float(res[y, x - 1]), float(res[y, x]), float(res[y, x + 1])
        den = dl - 2.0 * d0 + dr
        if abs(den) > 1e-12:
            dx = float(np.clip(0.5 * (dl - dr) / den, -1.0, 1.0))
    if 0 < y < res.shape[0] - 1:
        dl, d0, dr = float(res[y - 1, x]), float(res[y, x]), float(res[y + 1, x])
        den = dl - 2.0 * d0 + dr
        if abs(den) > 1e-12:
            dy = float(np.clip(0.5 * (dl - dr) / den, -1.0, 1.0))
    return dx, dy


def tile_votes(a_disp, b_disp, a_centers, pred_centers,
               tile=TILE_PX, margin=MARGIN_PX):
    """Per-tile NCC vote (primary) and POC-PSR comparison (fallback)."""
    a32 = np.asarray(a_disp, np.float32)
    b32 = np.asarray(b_disp, np.float32)
    t2 = tile // 2
    votes, skipped = [], 0
    for (acx, acy), (pcx, pcy) in zip(a_centers, pred_centers):
        ax0, ay0 = int(round(acx)) - t2, int(round(acy)) - t2
        tile_img = a32[ay0:ay0 + tile, ax0:ax0 + tile]
        if tile_img.shape != (tile, tile):
            skipped += 1
            continue
        ph = t2 + margin
        px_i, py_i = int(round(pcx)), int(round(pcy))
        bx0, by0 = max(px_i - ph, 0), max(py_i - ph, 0)
        bx1, by1 = min(px_i + ph, b32.shape[1]), min(py_i + ph, b32.shape[0])
        # The window must still cover the full +/-margin neighbourhood of the
        # prediction (the true match may lie anywhere in it, and the tile at
        # that match must fit inside the patch): skip when it cannot.
        if (bx0 > px_i - margin or by0 > py_i - margin
                or bx1 < px_i + margin or by1 < py_i + margin):
            skipped += 1
            continue
        patch = b32[by0:by1, bx0:bx1]
        res = cv2.matchTemplate(patch, tile_img, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        sxp, syp = _parabolic(res, maxloc)
        mx = bx0 + maxloc[0] + t2 + sxp
        my = by0 + maxloc[1] + t2 + syp
        poc_psr = float("nan")
        poc_mx = poc_my = float("nan")
        px0, py0 = px_i - t2, py_i - t2
        if px0 >= 0 and py0 >= 0 and px0 + tile <= b32.shape[1] \
                and py0 + tile <= b32.shape[0]:
            dx, dy, _, psr_v = phase_correlate(
                tile_img, b32[py0:py0 + tile, px0:px0 + tile])
            poc_psr = float(psr_v)
            poc_mx, poc_my = pcx + dx, pcy + dy
        votes.append(dict(ax=float(acx), ay=float(acy),
                          px=float(pcx), py=float(pcy),
                          mx=float(mx), my=float(my),
                          ncc=float(maxv), poc_psr=poc_psr,
                          poc_mx=float(poc_mx), poc_my=float(poc_my)))
    return votes, skipped


def fit_votes_5dof(votes, center, mode="ncc"):
    """Robust 5-DOF fit to tile votes: soft-L1 then inlier refit (fixed rule:
    inlier if residual <= max(3*1.4826*MAD, 10 px))."""
    a_pts = np.array([[v["ax"], v["ay"]] for v in votes])
    if mode == "ncc":
        b_pts = np.array([[v["mx"], v["my"]] for v in votes])
    else:
        b_pts = np.array([[v["poc_mx"], v["poc_my"]] for v in votes])
    off = np.median(b_pts - a_pts, axis=0)
    x0 = np.array([1.0, 1.0, 0.0, off[0], off[1]])

    def resid(p):
        pred = apply_h(model_h(*p, center), a_pts)
        return (pred - b_pts).ravel()

    r1 = least_squares(resid, x0, loss="soft_l1", f_scale=8.0, max_nfev=20000)
    e = np.hypot(*((apply_h(model_h(*r1.x, center), a_pts) - b_pts).T))
    mad = 1.4826 * np.median(np.abs(e - np.median(e)))
    keep = e <= max(3.0 * mad, 10.0)
    if keep.sum() >= 5:
        r2 = least_squares(resid, r1.x, max_nfev=20000)
        final, kept = r2.x, keep
    else:
        final, kept = r1.x, np.ones(len(votes), bool)
    e_final = np.hypot(*((apply_h(model_h(*final, center), a_pts) - b_pts).T))
    return final, dict(n_votes=len(votes), n_inliers=int(kept.sum()),
                       inlier_ratio=float(kept.mean()),
                       resid_median=float(np.median(e_final)),
                       resid_p90=float(np.percentile(e_final, 90)))



# ===========================================================================
# Pair loading (mirrors run_anisotropic_real_pair.py SQUARE region, --flip-b)
# ===========================================================================

A_COL0, A_COL1, A_ROW0, A_ROW1 = 400, 11800, 45000, 56400  # (see line ~68)


def load_pair():
    """Thin wrapper: build_pair + the key name my drivers use."""
    pair = build_pair()
    pair["b_disp"] = pair["b_disp_search"]
    return pair


def gate_a(pair):
    """Synthetic gate: inject a known 5-DOF transform into B, tile-vote + fit."""
    rng = np.random.default_rng(11)
    sx_t, sy_t, th_t = 0.947, 1.085, -4.6
    tx_t = float(rng.uniform(-14, 14))
    ty_t = float(rng.uniform(-14, 14))
    b_syn = warp_content(pair["a_disp"], sx_t, sy_t, th_t, tx_t, ty_t)
    centers = tile_grid(pair["disp_w"], pair["disp_h"])
    votes, skipped = tile_votes(pair["a_disp"], b_syn, centers, centers.copy())
    print(f"[gate] injected sx={sx_t} sy={sy_t} th={th_t} "
          f"t=({tx_t:.2f},{ty_t:.2f}); votes={len(votes)} skipped={skipped}")
    center = (pair["disp_w"] / 2.0, pair["disp_h"] / 2.0)
    p, stats = fit_votes_5dof(votes, center)
    print(f"[gate] recovered sx={p[0]:.4f} sy={p[1]:.4f} th={p[2]:.4f} "
          f"t=({p[3]:.2f},{p[4]:.2f})  fit={stats}")
    dth = (p[2] - th_t + 180.0) % 360.0 - 180.0
    errs = dict(sx=p[0] - sx_t, sy=p[1] - sy_t, th=dth,
                tx=p[3] - tx_t, ty=p[4] - ty_t)
    print(f"[gate] errors {({k: round(v, 4) for k, v in errs.items()})}")
    ok = (abs(errs["sx"]) < 0.01 and abs(errs["sy"]) < 0.01
          and abs(errs["th"]) < 0.5 and abs(errs["tx"]) < 3.0
          and abs(errs["ty"]) < 3.0)
    print(f"[gate] VERDICT: {'PASS' if ok else 'FAIL'}")
    return ok, dict(injected=[sx_t, sy_t, th_t, tx_t, ty_t],
                    recovered=[float(v) for v in p],
                    errors={k: float(v) for k, v in errs.items()},
                    fit=stats, verdict="PASS" if ok else "FAIL")

# ---------------------------------------------------------------------------
# Ground-truth scoring (identical construction to run_anisotropic_real_pair.py)
# ---------------------------------------------------------------------------

def gt_errors(pair, params, n=N_GT):
    """GT pixel errors (flip-frame B display px) for a 5-DOF param vector."""
    a_pts, _, b_flip = build_gt(pair, n=n)
    center = (pair["disp_w"] / 2.0, pair["disp_h"] / 2.0)
    pred = apply_h(model_h(*params, center), a_pts)
    return np.hypot(*(pred - b_flip).T)


def score_errs(tag, errs):
    print(f"  [{tag}] median={np.median(errs):7.2f}  mean={errs.mean():7.2f}  "
          f"min={errs.min():6.2f}  max={errs.max():8.2f}")
    for t in (50, 100, 300):
        print(f"      within {t:3d} px: {int((errs <= t).sum())}/{len(errs)}")
    return dict(median=float(np.median(errs)), mean=float(errs.mean()),
                min=float(errs.min()), max=float(errs.max()),
                within={str(t): int((errs <= t).sum()) for t in (50, 100, 300)},
                n=len(errs))


def recover_translation(pair, sx, sy, th):
    """Final translation-only pass at fixed (sx,sy,theta) (search() convention)."""
    ws = resample_for_search(pair["a_disp"], sx, sy, th)
    win = hann2d(ws.shape)
    dx, dy, peak, psr = phase_correlate(ws * win, pair["b_disp"] * win)
    return (-dx, -dy), float(psr), float(peak)

def h1_h2(pair):
    a, b = pair["a_disp"], pair["b_disp"]
    center = (pair["disp_w"] / 2.0, pair["disp_h"] / 2.0)
    print("\n=== H1: unseeded coarse+polish search (no NM refine) ===")
    res = search(a, b, refine=False)
    p1 = (res.sx, res.sy, res.theta_deg, res.tx, res.ty)
    print(f"[H1] sx={res.sx:.4f} sy={res.sy:.4f} theta={res.theta_deg:.3f} "
          f"t=({res.tx:.1f},{res.ty:.1f}) PSR={res.psr:.2f} "
          f"tPSR={res.translation_psr:.2f} n_eval={res.n_eval}")
    h1_gt = score_errs("H1 unseeded", gt_errors(pair, p1))
    print("\n=== E3: geolocation-derived affine (GT-fitted, no image content) ===")
    a_pts, _, b_flip = build_gt(pair, n=N_GT)
    ag5, ag_rms = fit_model5(a_pts, b_flip, center)
    ev = eval_all(a, b, ag5, center)
    print(f"[E3] params={[round(v, 4) for v in ag5]} (fit rms={ag_rms:.2f} px) "
          f"psr_current={ev['psr_current']['psr']:.2f} "
          f"hp={ev['psr_highpass']['psr']:.2f} "
          f"bl={ev['psr_bandlimit']['psr']:.2f} "
          f"nccm={ev['ncc_masked']['psr']:.2f}")
    e3_gt = score_errs("E3 geoloc affine", gt_errors(pair, ag5))
    print("\n=== E4: PSR transect unseeded -> geolocation ===")
    rows = transect(a, b, p1, ag5, center)
    for r in rows:
        print(f"    f={r['fraction']:.2f} sx={r['params'][0]:.4f} "
              f"sy={r['params'][1]:.4f} th={r['params'][2]:7.3f} "
              f"cur={r['psr_current']:7.3f} hp={r['psr_highpass']:7.3f} "
              f"bl={r['psr_bandlimit']:7.3f} nccm={r['ncc_masked']:7.3f}")
    bestg, _grows = local_grid(a, b, ag5, center)
    for k, (bpsr, bpar) in bestg.items():
        print(f"[E4:local_grid] best {k}: psr={bpsr:.3f} at "
              f"({bpar[0]:.4f}, {bpar[1]:.4f}, {bpar[2]:.3f})")
    best_name = max(bestg, key=lambda k: bestg[k][0])
    best = (best_name, bestg[best_name][1])
    print("\n=== H2: stage-2 NM refinement from both seeds ===")
    from anisotropic_affine_search import refine_transform  # noqa: E402
    h2 = {}
    for tag, seed in (("unseeded", best[1][:3]), ("geoloc", ag5[:3])):
        sx2, sy2, th2, psr_r, nfev = refine_transform(
            a, b, seed[0], seed[1], seed[2],
            sx_bounds_frac=0.30, sy_bounds_frac=0.30, theta_bounds_deg=15.0)
        (tx2, ty2), tpsr, _ = recover_translation(pair, sx2, sy2, th2)
        p2 = (sx2, sy2, th2, tx2, ty2)
        print(f"[H2:{tag}] NM({nfev} evals) -> sx={sx2:.4f} sy={sy2:.4f} "
              f"th={th2:.3f} t=({tx2:.1f},{ty2:.1f}) refined-PSR={psr_r:.2f} "
              f"tPSR={tpsr:.2f}")
        h2[tag] = dict(params=[float(v) for v in p2], refined_psr=float(psr_r),
                       translation_psr=float(tpsr), nfev=int(nfev),
                       gt=score_errs(f"H2 {tag}", gt_errors(pair, p2)))
    return dict(h1=dict(params=[float(v) for v in p1], psr=float(res.psr),
                        translation_psr=float(res.translation_psr),
                        gt=h1_gt),
                e3=dict(params=[float(v) for v in ag5],
                        psr_current=float(ev['psr_current']['psr']),
                        psr_highpass=float(ev['psr_highpass']['psr']),
                        psr_bandlimit=float(ev['psr_bandlimit']['psr']),
                        psr_ncc_masked=float(ev['ncc_masked']['psr']),
                        gt=e3_gt),
                e4=dict(best_objective=best[0],
                        best_params=[float(v) for v in best[1]],
                        transect=rows),
                h2=h2)

def h3(pair):
    print("\n=== H3: geolocation-seeded tile-vote + robust 5-DOF fit ===")
    centers = tile_grid(pair["disp_w"], pair["disp_h"])
    pred = predict_b_centers(pair, centers)
    votes, skipped = tile_votes(pair["a_disp"], pair["b_disp"], centers, pred)
    poc = [v for v in votes if not math.isnan(v["poc_psr"])]
    r_ncc = np.hypot(*np.array([[v["mx"] - v["px"], v["my"] - v["py"]]
                                for v in votes]).T)
    r_poc = np.hypot(*np.array([[v["poc_mx"] - v["px"], v["poc_my"] - v["py"]]
                                for v in poc]).T)
    agree = np.hypot(*np.array([[v["mx"] - v["poc_mx"], v["my"] - v["poc_my"]]
                                for v in poc]).T)
    print(f"[H3] tiles={len(centers)} votes={len(votes)} skipped={skipped} "
          f"poc_ok={len(poc)}")
    print(f"[H3] vote-vs-prediction residual (disp px): NCC median="
          f"{np.median(r_ncc):.2f} p90={np.percentile(r_ncc, 90):.2f} | "
          f"POC median={np.median(r_poc):.2f} p90={np.percentile(r_poc, 90):.2f}")
    print(f"[H3] NCC-vs-POC per-tile agreement: median={np.median(agree):.2f} "
          f"p90={np.percentile(agree, 90):.2f} px")
    center = (pair["disp_w"] / 2.0, pair["disp_h"] / 2.0)
    pn, sn = fit_votes_5dof(votes, center)
    print(f"[H3:NCC] params={[round(v, 4) for v in pn]} fit={sn}")
    pp, sp = fit_votes_5dof(poc, center, mode="poc")
    print(f"[H3:POC] params={[round(v, 4) for v in pp]} fit={sp}")
    return dict(n_tiles=len(centers), n_votes=len(votes), n_poc=len(poc),
                skipped=skipped,
                vote_resid=dict(ncc_median=float(np.median(r_ncc)),
                                ncc_p90=float(np.percentile(r_ncc, 90)),
                                poc_median=float(np.median(r_poc)),
                                poc_p90=float(np.percentile(r_poc, 90))),
                ncc_poc_agreement_median=float(np.median(agree)),
                ncc=dict(params=[float(v) for v in pn], fit=sn,
                         gt=score_errs("H3 NCC", gt_errors(pair, pn))),
                poc=dict(params=[float(v) for v in pp], fit=sp,
                         gt=score_errs("H3 POC", gt_errors(pair, pp))),
                votes=votes)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pair = load_pair()
    ok_gate, gate_rec = gate_a(pair)
    if mode == "gate":
        return 0 if ok_gate else 1
    record = dict(gate=gate_rec,
                  pair="phase8 pair (20231004 OHRC x 20250707 TMC-2)",
                  region="SQUARE flip-frame")
    record["h1_h2"] = h1_h2(pair)
    record["h3"] = h3(pair)
    with open(OUT_DIR / "audit_results.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=float)
    print(f"\nwrote {OUT_DIR / 'audit_results.json'} (total {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

