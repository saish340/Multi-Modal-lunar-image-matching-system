"""Phase 7 - synthetic Fourier-Mellin self-test (gate BEFORE real data).

Isolates "does the FMT implementation work at all" from "does it solve our
real cross-sensor problem". We take a real downloaded lunar crop, apply a
KNOWN scale + rotation (+ translation) to create a synthetic "second image",
then run FMT and confirm it recovers the known transform within tolerance.

This must pass cleanly before ``run_fmt_real_pair.py`` is trusted on the real
OHRC/TMC-2 pair.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_pkg = Path(__file__).resolve().parent / "lunar_data_pipeline"
if str(_pkg) not in sys.path:
    sys.path.insert(0, str(_pkg))

from phase_correlation import phase_correlate  # noqa: E402
from fourier_mellin import register  # noqa: E402
from image_loader import load_image, to_uint8  # noqa: E402
from pds4_parser import parse_label  # noqa: E402

A_LABEL = (
    "D:/Multi-Modal-lunar-image-matching-system/"
    "ch2_ohr_ncp_20260103T0609041371_d_img_d18/"
    "data/calibrated/20260103/ch2_ohr_ncp_20260103T0609041371_d_img_d18.xml"
)

#: Known injected transform (scale, deg, tx, ty). B = scale*R(theta)*A + t.
#: SCALE < 1 keeps the transformed content fully inside the same-size canvas
#: (no clipping), which is the geometry FMT actually measures -- it models the
#: fine->coarse case where the fine-resolution content appears *smaller*
#: relative to the coarse canvas. The mechanism is direction-agnostic; the
#: log-polar phase correlation finds the relative scale either way.
KNOWN_SCALE = 0.7
KNOWN_THETA = 12.0
KNOWN_TX = 20.0
KNOWN_TY = -25.0
#: Tolerance on each recovered parameter.
SCALE_TOL = 0.06      # relative
THETA_TOL = 1.0       # degrees
TRANS_TOL = 6.0       # px


def make_synthetic_b(a: np.ndarray) -> np.ndarray:
    """Create B = scale*R(theta)*A + t, drawn into a same-size canvas."""
    h, w = a.shape[:2]
    center = (w / 2.0 - 0.5, h / 2.0 - 0.5)
    M = cv2.getRotationMatrix2D(center, KNOWN_THETA, KNOWN_SCALE)
    M[0, 2] += KNOWN_TX
    M[1, 2] += KNOWN_TY
    b = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return b


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S", stream=sys.stderr,
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512, help="square crop size")
    ap.add_argument("--col0", type=int, default=2000)
    ap.add_argument("--row0", type=int, default=40000)
    args = ap.parse_args()

    a = parse_label(A_LABEL)
    assert a is not None
    crop = to_uint8(load_image(
        a, row_range=(args.row0, args.row0 + args.size),
        col_range=(args.col0, args.col0 + args.size),
    )).astype(np.float32)

    b = make_synthetic_b(crop)
    # the injected full affine M (matches H_apply convention used for scoring)
    center = (crop.shape[1] / 2.0 - 0.5, crop.shape[0] / 2.0 - 0.5)
    M = cv2.getRotationMatrix2D(center, KNOWN_THETA, KNOWN_SCALE)
    M[0, 2] += KNOWN_TX
    M[1, 2] += KNOWN_TY
    M_inj = np.vstack([M, [0, 0, 1]])
    print(f"=== FMT synthetic test ===")
    print(f"crop: {args.size}x{args.size} at cols[{args.col0}] rows[{args.row0}]")
    print(f"injected: scale={KNOWN_SCALE}, theta={KNOWN_THETA} deg, "
          f"t=({KNOWN_TX},{KNOWN_TY})")

    t0 = time.time()
    res = register(crop, b)
    dt = time.time() - t0

    s_err = abs(res.scale - KNOWN_SCALE) / KNOWN_SCALE
    th_err = abs(res.theta_deg - KNOWN_THETA)
    t_err = math.hypot(res.tx - KNOWN_TX, res.ty - KNOWN_TY)

    # gold-standard: does the reconstructed global transform reproduce the
    # injected affine on a grid of points (this is what real scoring uses)?
    pts = np.array([[5, 5], [30, 40], [250, 250], [400, 100], [60, 450],
                    [500, 30], [200, 480]], float)
    inj = M_inj @ np.column_stack([pts, np.ones(len(pts))]).T
    inj = (inj / inj[2])[:2].T
    rec = res.H_apply @ np.column_stack([pts, np.ones(len(pts))]).T
    rec = (rec / rec[2])[:2].T
    h_rms = float(np.sqrt(np.mean(np.sum((inj - rec) ** 2, axis=1))))

    print(f"\nrecovered (in {dt:.2f}s):")
    print(f"  scale    : {res.scale:.4f}   (rel.err {s_err:.4f})  "
          f"{'PASS' if s_err <= SCALE_TOL else 'FAIL'}")
    print(f"  theta    : {res.theta_deg:.2f} deg  (err {th_err:.2f})  "
          f"{'PASS' if th_err <= THETA_TOL else 'FAIL'}")
    print(f"  translate: ({res.tx:.1f},{res.ty:.1f})  (err {t_err:.1f} px)  "
          f"{'PASS' if t_err <= TRANS_TOL else 'FAIL'}")
    print(f"  H_apply vs injected affine (RMS over {len(pts)} pts): {h_rms:.2f} px  "
          f"{'PASS' if h_rms <= TRANS_TOL else 'FAIL'}")
    print(f"  log-polar PSR  : {res.log_polar_psr:.2f}")
    print(f"  translation PSR: {res.translation_psr:.2f}")

    ok = (s_err <= SCALE_TOL and th_err <= THETA_TOL and t_err <= TRANS_TOL
          and h_rms <= TRANS_TOL)
    print(f"\nSYNTHETIC GATE: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("diagnostic: if scale is the reciprocal or theta sign flipped, "
              "check the log-polar shift conventions in fourier_mellin.py.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
