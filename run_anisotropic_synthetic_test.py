"""Phase 8 - synthetic gate: does the anisotropic-affine search recover a KNOWN
anisotropic transform (independent x/y scale) applied to a real lunar crop?

Takes a real OHRC lunar crop (the same scene the Phase 7/8 real work uses),
warp-seams it with a known anisotropic affine -- different x-scale and y-scale
(1.3 vs 0.8) plus a rotation (10 deg) and translation -- then verifies the
coarse-to-fine search in :mod:`anisotropic_affine_search` recovers those exact
parameters, and that its assembled ``H_apply`` projects ground-truth points back
to within sub-pixel accuracy of the injected transform.

Padded injection keeps the expanding-x content inside the canvas so the
spectrum relationship is not cut off (the Phase 7 note that scale>1 clips);
phase-correlation scoring makes this forgiving in any case.

GATE: recovers (sx, sy, theta) near the injected values and projects GT points
with small RMS -> PASS, then and only then should the real pair be attempted.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

_pkg = Path(__file__).resolve().parent / "lunar_data_pipeline"
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from anisotropic_affine_search import (
    _assemble_affine,
    search,
    warp_content,
)  # noqa: E402
from image_loader import load_image, to_uint8  # noqa: E402
from pds4_parser import parse_label  # noqa: E402

# Real lunar crop source (same OHRC South Pole scene used by Phase 7 unit tests).
A_LABEL = (
    "D:/Multi-Modal-lunar-image-matching-system/"
    "ch2_ohr_ncp_20260103T0609041371_d_img_d18/"
    "data/calibrated/20260103/ch2_ohr_ncp_20260103T0609041371_d_img_d18.xml"
)

# INJECTED ground-truth anisotropic transform (independent x/y scales).
INJ_SX = 1.3
INJ_SY = 0.8
INJ_THETA = 10.0
INJ_TX = 15.0
INJ_TY = -12.0

# Search ranges: broad enough to demonstrate generality (we do NOT seed with
# the answer). sx/sy in [0.5, 2.0], theta in [-30, 30].
SX_RANGE = (0.5, 2.0)
SY_RANGE = (0.5, 2.0)
TH_RANGE = (-30.0, 30.0)

CROP = 384  # enough texture for reliable phase-correlation scoring


def inject(a: np.ndarray, sx: float, sy: float, theta_deg: float,
           tx: float, ty: float) -> np.ndarray:
    """Warp A's content by the anisotropic affine (sx, sy, theta) + (tx, ty)
    using the search's own coordinate-consistent convention (so the injected
    "true" parameters are geometrically reachable by the search)."""
    return warp_content(a, sx, sy, theta_deg, tx, ty)


def main() -> int:
    p = parse_label(A_LABEL)
    if p is None:
        print("FAIL: could not parse real label (data missing?)")
        return 1
    crop = to_uint8(load_image(
        p, row_range=(40000, 40000 + CROP), col_range=(2000, 2000 + CROP)
    )).astype(np.float32)
    a = (crop - crop.mean()) / (crop.std() + 1e-9)

    b = inject(a, INJ_SX, INJ_SY, INJ_THETA, INJ_TX, INJ_TY)

    print(f"crop: {CROP}x{CROP} at cols[2000] rows[40000] (real OHRC scene)")
    print(f"injected: sx={INJ_SX}, sy={INJ_SY}, theta={INJ_THETA} deg, "
          f"t=({INJ_TX},{INJ_TY})")
    print(f"search range: sx{list(SX_RANGE)} sy{list(SY_RANGE)} "
          f"theta{list(TH_RANGE)} deg (not seeded with the answer)")

    t0 = time.time()
    res = search(
        a, b,
        sx_range=SX_RANGE, sy_range=SY_RANGE, theta_range=TH_RANGE,
        coarse_each=7, fine_each=7,
    )
    dt = time.time() - t0

    print(f"\nrecovered (in {dt:.1f}s, {res.n_eval} phase-correlations "
          f"= {res.coarse_count} coarse + {res.fine_count} fine):")
    print(f"  sx      : {res.sx:.3f}   (err {res.sx - INJ_SX:+.3f})")
    print(f"  sy      : {res.sy:.3f}   (err {res.sy - INJ_SY:+.3f})")
    print(f"  theta   : {res.theta_deg:.2f} deg (err {res.theta_deg - INJ_THETA:+.2f})")
    print(f"  tx, ty  : ({res.tx:.1f}, {res.ty:.1f})   "
          f"expected ({INJ_TX},{INJ_TY})")
    print(f"  PSR     : {res.psr:.2f}")

    # Ground-truth projection check: sample A-space points, map with BOTH the
    # injected H and the recovered H, and compare.
    rng = np.random.default_rng(0)
    n = 60
    pts = rng.uniform([10, 10], [CROP - 10, CROP - 10], (n, 2))
    c = ((CROP - 1) / 2.0, (CROP - 1) / 2.0)
    H_true = _assemble_affine(INJ_SX, INJ_SY, INJ_THETA, INJ_TX, INJ_TY, c)
    true_b = (H_true @ np.column_stack([pts, np.ones(n)]).T)[:2].T
    rec_b = res.project_many(pts)
    rms = float(np.sqrt(np.mean(np.sum((rec_b - true_b) ** 2, axis=1))))

    print(f"\nGT projection RMS: {rms:.3f} px (recovered H vs injected H over "
          f"{n} points)")

    tol_sx, tol_sy, tol_th = 0.06, 0.06, 1.5
    ok_sx = abs(res.sx - INJ_SX) <= tol_sx
    ok_sy = abs(res.sy - INJ_SY) <= tol_sy
    ok_th = abs(res.theta_deg - INJ_THETA) <= tol_th
    ok_rms = rms <= 2.0
    ok_t = np.hypot(res.tx - INJ_TX, res.ty - INJ_TY) <= 2.0

    passed = ok_sx and ok_sy and ok_th and ok_rms and ok_t
    print(f"\nSYNTHETIC GATE: {'PASS' if passed else 'FAIL'}")
    print(f"  sx  within {tol_sx}: {ok_sx}   sy within {tol_sy}: {ok_sy}   "
          f"theta within {tol_th}: {ok_th}   tx,ty within 2px: {ok_t}   RMS<=2px: {ok_rms}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
