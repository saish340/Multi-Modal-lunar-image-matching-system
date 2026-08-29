"""Anisotropic-affine registration via coarse-to-fine phase-correlation search
(Phase 8).

Phase 7's Fourier-Mellin transform assumes a single isotropic scale factor; on
the real OHRC/TMC-2 pair that assumption is violated because both instruments
are pushbroom sensors whose along-track (scan) scale and cross-track (pixel)
scale differ independently, giving an *anisotropic* linear relationship (an
affine with 2 independent scale parameters + rotation + translation rather than
1 scale).

This module recovers that anisotropic transform by an outer grid search over
the ``(scale_x, scale_y, rotation)`` degrees of freedom, wrapping the ordinary
phase-correlation used in Phases 6/7 as an inner scoring step:

1. For a candidate ``(sx, sy, theta)``, resample image ``A`` by that
   anisotropic transform (about the image centre), producing an image whose
   content should match ``B``'s orientation/scale up to a residual translation.
2. Phase-correlate the resampled ``A`` against ``B``; the phase-correlation
   *PSR* (peak-to-sidelobe ratio, from :mod:`phase_correlation`) scores how
   well that candidate aligns the two views.
3. A coarse pass over a broad grid finds the promising region; a fine pass
   around the coarse best result refines it (instead of one huge uniform grid).
4. With the best ``(sx, sy, theta)`` fixed, one final phase correlation recovers
   the translation, and everything is assembled into an affine ``H_apply`` that
   maps A-space points into B-space (reusing the phase-correlation convention
   established in Phase 7).

The geolocation CSV data is used ONLY as validation ground truth by the runners
-- never as a seed or narrowing of the search range here.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from itertools import product
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from .phase_correlation import hann2d, phase_correlate
except ImportError:  # pragma: no cover
    from phase_correlation import hann2d, phase_correlate  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


def _linear_matrix(sx: float, sy: float, theta_deg: float) -> np.ndarray:
    """2x2 linear part mapping A-content vectors to B-content vectors.

    ``L = Rot(theta) @ diag(sx, sy)`` using the cv2/image y-down rotation
    orientation ``[[cos, sin], [-sin, cos]]`` (same convention as Phase 7's
    ``_assemble_affine``). This applies an independent x/y scaling and then a
    rotation, the anisotropic pushbroom model.
    """
    th = math.radians(theta_deg)
    co, si = math.cos(th), math.sin(th)
    R = np.array([[co, si], [-si, co]])
    S = np.diag([sx, sy])
    return R @ S


def _sampling_matrix(sx: float, sy: float, theta_deg: float,
                     center: tuple[float, float]) -> np.ndarray:
    """2x3 warpAffine matrix sampling A to draw A's content warped forward by
    ``(sx, sy, theta)`` about ``center`` into a same-size output canvas.

    The forward content mapping is ``p' = L @ (p - c) + c`` (``L`` maps A-content
    to B-content); warpAffine needs output->input, i.e. the inverse.
    """
    L = _linear_matrix(sx, sy, theta_deg)
    Linv = np.linalg.inv(L)
    cx, cy = center
    v = np.array([cx, cy])
    t = v - Linv @ v
    M = np.eye(3, dtype=np.float64)
    M[:2, :2] = Linv
    M[0, 2], M[1, 2] = t[0], t[1]
    return M[:2]


def _assemble_affine(sx: float, sy: float, theta_deg: float,
                     tx: float, ty: float,
                     center: tuple[float, float]) -> np.ndarray:
    """3x3 affine mapping A-space point -> B-space point: ``T(c+t) . L . T(-c)``
    (centre-of-rotation composition, matching Phase 7's convention)."""
    L = _linear_matrix(sx, sy, theta_deg)
    cx, cy = center
    H = np.eye(3, dtype=np.float64)
    H[:2, :2] = L
    H[0, 2], H[1, 2] = -cx, -cy
    Tt = np.eye(3, dtype=np.float64)
    Tt[0, 2], Tt[1, 2] = cx + tx, cy + ty
    return Tt @ H


def resample_for_search(image: np.ndarray, sx: float, sy: float,
                        theta_deg: float) -> np.ndarray:
    """Resample ``image`` by the anisotropic transform ``(sx, sy, theta)`` about
    its own centre, returning a same-size array with the content warped forward
    (matching B's orientation/scale) with **no** translation.

    This is intentionally the EXACT same affine construction as
    :func:`warp_content` (``_assemble_affine`` + inverse sampling on the
    unpadded image), so an injected ``warp_content(...)`` B is guaranteed
    geometrically consistent with what the search sees -- this was the source
    of the earlier synthetic-gate failure (a divergent padded construction).
    """
    a = np.asarray(image, dtype=np.float32)
    return warp_content(a, sx, sy, theta_deg, 0.0, 0.0)


def warp_content(image: np.ndarray, sx: float, sy: float,
                 theta_deg: float, tx: float = 0.0, ty: float = 0.0,
                 center: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Coordinate-consistent forward warp of ``image``'s content by the
    anisotropic transform ``(sx, sy, theta)`` plus translation ``(tx, ty)``.

    This is exactly the model the search assumes for ``B``: content at ``p`` in
    A appears at ``T(t) (T(c) L T(-c)) p`` in B, using the same centre/padding
    convention as :func:`resample_for_search`. Runners / synthetic tests MUST
    build injected ``B`` with this (or the equivalent ``_assemble_affine``
    inverse) so the "true" parameters actually align -- a pad/centre mismatch
    otherwise makes the ground truth geometrically unreachable.
    """
    a = np.asarray(image, dtype=np.float32)
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float32)
    h, w = a.shape[:2]
    if center is None:
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
    H = _assemble_affine(sx, sy, theta_deg, tx, ty, center)
    return cv2.warpAffine(a, H[:2].astype(np.float64), (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE).astype(np.float32)


@dataclass
class AnisotropicAffineResult:
    """Recovered anisotropic transform mapping ``A``'s content onto ``B``."""

    sx: float
    sy: float
    theta_deg: float
    tx: float
    ty: float
    psr: float
    peak_score: float
    translation_psr: float
    n_eval: int
    runtime_s: float
    coarse_count: int
    fine_count: int
    H_apply: np.ndarray  # 3x3 affine, maps A-space -> B-space

    def project(self, point: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(point, dtype=float))
        hom = np.column_stack([pts, np.ones(len(pts))])
        out = (self.H_apply @ hom.T).T
        return out[0, :2] if np.asarray(point).ndim == 1 else out[:, :2]

    def project_many(self, points: np.ndarray) -> np.ndarray:
        return self.project(points)

    def summary_dict(self) -> dict:
        return {
            "sx": round(float(self.sx), 5),
            "sy": round(float(self.sy), 5),
            "theta_deg": round(float(self.theta_deg), 3),
            "tx": float(self.tx),
            "ty": float(self.ty),
            "psr": round(float(self.psr), 2),
            "peak_score": round(float(self.peak_score), 4),
            "translation_psr": round(float(self.translation_psr), 2),
            "n_eval": self.n_eval,
            "runtime_s": round(float(self.runtime_s), 2),
            "coarse_count": self.coarse_count,
            "fine_count": self.fine_count,
            "H_apply": [[float(v) for v in row] for row in self.H_apply],
        }


def _evaluate(a: np.ndarray, b: np.ndarray, sx: float, sy: float,
              theta_deg: float, window: bool):
    """Resample ``a`` by ``(sx,sy,theta)`` and phase-correlate against ``b``.

    Returns ``(psr, peak_score, dx, dy)``.
    """
    ws = resample_for_search(a, sx, sy, theta_deg)
    if window:
        win = hann2d(ws.shape)
        ws = ws * win
        b_ = b * win
    else:
        b_ = b
    dx, dy, peak, psr = phase_correlate(ws, b_)
    return psr, peak, dx, dy, ws


def grid_values(start: float, stop: float, n: int) -> np.ndarray:
    """Uniform grid of ``n`` points over ``[start, stop]``."""
    if n == 1:
        return np.array([0.5 * (start + stop)])
    return np.linspace(start, stop, n)


def search(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    sx_range: Tuple[float, float] = (0.5, 2.0),
    sy_range: Tuple[float, float] = (0.5, 2.0),
    theta_range: Tuple[float, float] = (-30.0, 30.0),
    coarse_each: int = 7,
    fine_each: int = 7,
    fine_passes: int = 2,
    fine_fraction_sx: float = 0.12,
    fine_fraction_sy: float = 0.12,
    fine_theta_deg: float = 3.0,
    fine_theta_shrink: float = 0.25,
    polish: bool = True,
    polish_each: int = 9,
    polish_passes: int = 2,
    polish_fraction_sx: float = 0.25,
    polish_fraction_sy: float = 0.25,
    polish_theta_deg: float = 0.05,
    window: bool = True,
) -> AnisotropicAffineResult:
    """Coarse-to-fine grid search over ``(sx, sy, theta)`` scored by PSR.

    ``image_a`` and ``image_b`` must be 2-D, same size, showing the same ground
    region at the same resolution (the caller handles cropping/downsampling;
    see the runners). The geolocation data must NOT be used to seed the ranges
    -- a broad range is the default so the method generalises to pairs where the
    answer is not known in advance.
    """
    a = np.asarray(image_a, dtype=np.float32)
    b = np.asarray(image_b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("images must be 2-D")
    if a.shape != b.shape:
        raise ValueError(f"images must be same size, got {a.shape} vs {b.shape}")

    t0 = time.time()
    total = 0
    # coarser/finer split is done via wrapping grid evaluation
    cur = _grid_pass(a, b, sx_range, sy_range, theta_range,
                     coarse_each, window)
    total += cur["n"]
    best = cur
    bx, by, bt = best["sx"], best["sy"], best["theta"]

    # Running search spans. Each refinement pass covers the PREVIOUS span
    # (scaled by the fractions / narrowed for theta), then the next pass
    # shrinks again -- true geometric coarse-to-fine. A fixed-width pass
    # otherwise grid-locks: the coarse grid's best point becomes unreachable
    # by narrower passes that sit around it but cannot move to the true peak.
    sx_span = float(sx_range[1] - sx_range[0])
    sy_span = float(sy_range[1] - sy_range[0])
    # First fine pass spans: scale spans shrink from the coarse spans, theta
    # starts at fine_theta_deg (its own "coarse" span) and shrinks per pass.
    th_span = fine_theta_deg

    for _ in range(fine_passes):
        sx_span = max(fine_fraction_sx * sx_span, 1e-6)
        sy_span = max(fine_fraction_sy * sy_span, 1e-6)
        th_span = max(th_span * fine_theta_shrink, 1e-6)
        cur = _grid_pass(
            a, b,
            (bx - sx_span, bx + sx_span),
            (by - sy_span, by + sy_span),
            (bt - th_span, bt + th_span),
            fine_each, window,
        )
        total += cur["n"]
        best = cur
        bx, by, bt = best["sx"], best["sy"], best["theta"]

    # ---- sub-pixel polish of (sx, sy, theta) ----
    # The final translation (a phase-correlation needle) is only recovered
    # exactly when scale/rotation are sub-pixel accurate (~0.001 in scale,
    # ~0.05 deg). We keep narrowing the (now small) span geometrically so the
    # grid can actually move toward the true peak instead of grid-locking.
    if polish:
        for _ in range(polish_passes):
            sx_span = max(polish_fraction_sx * sx_span, 1e-6)
            sy_span = max(polish_fraction_sy * sy_span, 1e-6)
            th_span = max(th_span * 0.3, polish_theta_deg)
            cur = _grid_pass(
                a, b,
                (bx - sx_span, bx + sx_span),
                (by - sy_span, by + sy_span),
                (bt - th_span, bt + th_span),
                polish_each, window,
            )
            total += cur["n"]
            best = cur
            bx, by, bt = best["sx"], best["sy"], best["theta"]

    # ---- final translation at best (sx, sy, theta) ----
    psr, peak, dx, dy, ws = _evaluate(a, b, bx, by, bt, window)
    # ``warp_content`` places injected content at exactly ``H . p`` (forward),
    # so ``phase_correlate`` returns the *negation* of the recovery: content
    # displaced by ``+(tx, ty)`` comes back as ``-(dx, dy)``. Negating here
    # yields an H_apply whose projection matches ``H_true`` to ~0 px.
    tx, ty = float(-dx), float(-dy)
    c = ((b.shape[1] - 1) / 2.0, (b.shape[0] - 1) / 2.0)
    H = _assemble_affine(bx, by, bt, tx, ty, c)
    runtime = time.time() - t0

    return AnisotropicAffineResult(
        sx=bx, sy=by, theta_deg=bt, tx=tx, ty=ty,
        psr=float(best["psr"]), peak_score=float(best["peak"]),
        translation_psr=float(psr),
        n_eval=total, runtime_s=runtime,
        coarse_count=coarse_each ** 3,
        fine_count=(fine_passes * (fine_each ** 3)
                    + (polish_passes * (polish_each ** 3) if polish else 0)),
        H_apply=H,
    )


def _grid_pass(a: np.ndarray, b: np.ndarray,
               sx_range, sy_range, theta_range, n_each: int, window: bool) -> dict:
    """Evaluate an n_each^3 uniform grid, returning the best candidate."""
    sxs = grid_values(sx_range[0], sx_range[1], n_each)
    sys_ = grid_values(sy_range[0], sy_range[1], n_each)
    ths = grid_values(theta_range[0], theta_range[1], n_each)
    best = None
    count = 0
    for sx, sy, th in product(sxs, sys_, ths):
        psr, peak, dx, dy, _ = _evaluate(a, b, sx, sy, th, window)
        count += 1
        if best is None or psr > best["psr"]:
            best = dict(sx=sx, sy=sy, theta=th, psr=psr, peak=peak, dx=dx, dy=dy)
    best["n"] = count
    return best
