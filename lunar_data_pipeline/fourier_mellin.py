"""Fourier-Mellin Transform registration (Phase 7).

Recover the rigid+scale transformation relating two images:

    B(x, y) = A( s * R(theta) * (x, y) + (tx, ty) )

i.e. ``B`` is a scaled (``s``), rotated (``theta``) and translated (``tx, ty``)
copy of ``A``'s content. This is the classical global extension of plain phase
correlation (which only recovers translation) and is the natural fit for the
project's confirmed blocker: the ~20-24x OHRC/TMC-2 inter-sensor scale gap
plus a modest ~2.7 deg rotation, on terrain where local-feature/local-structure
matching is ambiguous.

Pipeline (standard FMT):

1. Compute the FFT **magnitude** spectra of both images -- the magnitude is
   translation-invariant, so the translation drops out here.
2. Remap each magnitude spectrum into **log-polar** coordinates
   ``(log radius, angle)``. A spatial scale change becomes a shift along the
   log-radius axis and a rotation becomes a shift along the angle axis, both
   of which phase correlation can recover directly.
3. Phase-correlate the two log-polar spectra -- the peak yields ``log(s)``
   and ``theta`` at once.
4. Resample image ``A`` by the recovered scale+rotation to align it with
   ``B``'s orientation, then ordinary phase correlation recovers the residual
   translation ``(tx, ty)``.

All results are assembled into a 3x3 affine ``B = H o A`` matrix so the
existing scorer/ground-truth projection conventions can be reused.

Practical size note: the real OHRC/TMC-2 case must run on cropped/downsampled
versions (full resolution is 101074x12000) -- see ``run_fmt_real_pair.py``.
The caller supplies already-suitable images; this module does not downsample.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from .phase_correlation import hann2d, phase_correlate
except ImportError:  # pragma: no cover
    from phase_correlation import hann2d, phase_correlate  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

#: Minimum radius (px) sampled in the log-polar remap. Must be large enough to
#: skip the low-frequency core (whose near-flat magnitude otherwise dominates
#: the log-polar phase correlation and pins it to zero shift), but small enough
#: to keep the rotation/scale structure. Empirically 8-40 all recover an exact
#: angle; 12 is a robust default for the square crops in this project.
_LOG_R_MIN_PX = 12.0
#: Number of samples across the log-polar height (radius axis).
_LOG_POLAR_ROWS = 512
#: Number of samples across the log-polar width (angle axis).
_LOG_POLAR_COLS = 512


def fft_magnitude(image: np.ndarray) -> np.ndarray:
    """FFT magnitude spectrum of a square-ish float image, DC zeroed.

    ``image`` and the reference image must be the SAME size for the
    log-polar phase correlation. ``image`` should already be zero-padded to
    the same power-of-two size if it differs, but the caller decides that;
    here we only require a 2-D real array and zero the DC bin.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"expected 2-D image, got {img.ndim} dims")
    mag = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    mag[mag.shape[0] // 2, mag.shape[1] // 2] = 0.0
    mag += 1e-12
    return mag.astype(np.float32)


def log_polar(
    magnitude: np.ndarray,
    log_polar_rows: int = _LOG_POLAR_ROWS,
    log_polar_cols: int = _LOG_POLAR_COLS,
) -> np.ndarray:
    """Remap an FFT magnitude spectrum into log-polar coordinates.

    Output rows index increasing log-radius (row 0 = smallest radius), output
    columns index angle from 0..2pi (periodic along the angle axis). Built on
    :func:`cv2.remap` (cv2>=5 removed the old :func:`cv2.logPolar` helper).
    Both images pass through this same mapping, so the recovered column shift
    is directly the angular difference and the row shift the log-scale
    difference.
    """
    mag = np.asarray(magnitude, dtype=np.float32)
    h, w = mag.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r_max = 0.5 * math.hypot(cx, cy)
    r_min = max(_LOG_R_MIN_PX, 1e-3)

    log_radius = np.linspace(math.log(r_min), math.log(r_max), log_polar_rows)
    angles = np.linspace(0.0, 2.0 * math.pi, log_polar_cols, endpoint=False)
    radius = np.exp(log_radius)[:, None]          # (rows, 1)
    ang = angles[None, :]                         # (1, cols)
    xmap = (cx + radius * np.cos(ang)).astype(np.float32)  # (rows, cols)
    ymap = (cy + radius * np.sin(ang)).astype(np.float32)
    return cv2.remap(
        mag, xmap, ymap, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    ).astype(np.float32)


def _fold_angle(theta_deg: float) -> float:
    """Normalise an angle to (-180, 180] degrees."""
    theta_deg = math.fmod(theta_deg, 360.0)
    if theta_deg > 180.0:
        theta_deg -= 360.0
    elif theta_deg <= -180.0:
        theta_deg += 360.0
    return theta_deg


@dataclass
class FMTResult:
    """Recovered global transform mapping ``A``'s content onto ``B``.

    ``B = scale * Rot(theta) * A + (tx, ty)`` in the sense that a point at
    ``(x, y)`` in ``A`` maps to ``(x', y')`` in ``B`` via the affine
    ``H_apply`` matrix::

        (x', y') = H_apply @ (x, y, 1)
    """

    scale: float
    theta_deg: float
    tx: float
    ty: float
    H_apply: np.ndarray  # 3x3 affine, maps A-space point to B-space point
    log_polar_shift_cols: float
    log_polar_shift_rows: float
    log_polar_psr: float
    translation_psr: float
    peak_score: float = 0.0

    def project(self, point: np.ndarray) -> np.ndarray:
        """Map an (N,2) array (or (2,)) of A-space points to B-space."""
        pts = np.atleast_2d(np.asarray(point, dtype=float))
        if pts.shape[1] != 2:
            raise ValueError("points must be (N,2)")
        hom = np.column_stack([pts, np.ones(len(pts))])
        out = (self.H_apply @ hom.T).T
        return out[0, :2] if np.asarray(point).ndim == 1 else out[:, :2]

    def project_many(self, points: np.ndarray) -> np.ndarray:
        return self.project(points)

    def summary_dict(self) -> dict:
        return {
            "scale": round(float(self.scale), 6),
            "theta_deg": round(float(self.theta_deg), 4),
            "tx": float(self.tx),
            "ty": float(self.ty),
            "log_polar_shift_cols": float(self.log_polar_shift_cols),
            "log_polar_shift_rows": float(self.log_polar_shift_rows),
            "log_polar_psr": round(float(self.log_polar_psr), 2),
            "translation_psr": round(float(self.translation_psr), 2),
            "H_apply": [
                [float(v) for v in row] for row in self.H_apply
            ],
        }


def recover_scale_rotation(
    mag_a: np.ndarray, mag_b: np.ndarray
) -> tuple[float, float, int, int, float]:
    """Recover scale and rotation from two equal-size log-polar spectra.

    Returns ``(scale, theta_deg, shift_rows, shift_cols, psr)`` where a feature
    at log-polar ``(row, col)`` in ``mag_a`` appears at
    ``(row + shift_rows, col + shift_cols)`` in ``mag_b``.

    Empirical conventions (validated on a known-injection synthetic test):
    increasing row = increasing log-radius; a scaling of ``B`` relative to
    ``A`` by factor ``s`` shifts the log-radius rows by
    ``+ln(s) * rows / ln(r_max/r_min)`` (so ``shift_rows`` > 0 means B's content
    is coarser, i.e. s > 1). A CCW rotation of B's content by ``theta`` shifts
    the log-polar columns by ``+theta * cols/360``.
    """
    if mag_a.shape != mag_b.shape:
        raise ValueError("log-polar spectra must be equal size")
    dx, dy, peak, psr = phase_correlate(mag_a, mag_b)
    shift_cols = dx  # angle axis (column), uses periodic wrap -> good
    shift_rows = dy  # log-radius axis (row)

    rows = mag_a.shape[0]
    cols = mag_a.shape[1]
    # column shift -> angle in degrees. col index 0..cols maps angle 0..2pi,
    # so one full column span = 2pi radians = 360 deg.
    theta_deg = shift_cols * (360.0 / cols)
    theta_deg = _fold_angle(theta_deg)

    h, w = mag_a.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    diagonal = math.hypot(cx, cy)
    r_max = 0.5 * diagonal
    r_min = _LOG_R_MIN_PX
    log_s = shift_rows * (math.log(r_max / r_min) / rows)
    scale = math.exp(log_s)
    return scale, theta_deg, shift_rows, shift_cols, psr


def _resample_by_scale_rotation(
    image: np.ndarray, scale: float, theta_deg: float
) -> np.ndarray:
    """Resample ``image`` by rotating +theta and scaling by ``scale`` so its
    content orientation matches the reference image's.

    We transform the CONTENT (the surface) by the same scale+rotation that B
    applied to A, mapping A's content into B's orientation. The warp is applied
    about the image centre and the result keeps the same shape as ``image``.
    """
    h, w = image.shape[:2]
    center = (w / 2.0 - 0.5, h / 2.0 - 0.5)
    M_rot = cv2.getRotationMatrix2D(center, theta_deg, scale)
    # pad to avoid clipping rotated content
    pad = int(0.5 * max(h, w) * min(abs(scale), 1.0) + 0.5) + 50
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    hh, ww = padded.shape[:2]
    c2 = (ww / 2.0 - 0.5, hh / 2.0 - 0.5)
    M2 = cv2.getRotationMatrix2D(c2, theta_deg, scale)
    warped = cv2.warpAffine(padded, M2, (ww, hh), flags=cv2.INTER_LINEAR)
    warped = warped[pad:pad + h, pad:pad + w].copy()
    return warped


def register(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    window: bool = True,
    log_polar_rows: int = _LOG_POLAR_ROWS,
    log_polar_cols: int = _LOG_POLAR_COLS,
    translation_resolution: Optional[tuple[int, int]] = None,
) -> FMTResult:
    """Fourier-Mellin register ``image_b`` against ``image_a``.

    Both inputs must be 2-D, same size, and already cropped/centred on the
    same extent (the caller handles downsampling/cropping; this is a pure
    transform-recovery routine used by both the synthetic test and the real
    pair). ``image_b`` is treated as a scaled+rotated+translated copy of
    ``image_a``'s content.

    Returns :class:`FMTResult` with the recovered scale, rotation, translation
    and the assembled affine ``H_apply`` (maps A-space points into B-space).
    """
    a = np.asarray(image_a, dtype=np.float32)
    b = np.asarray(image_b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("images must be 2-D")
    if a.shape != b.shape:
        raise ValueError(
            f"images must be same size for FMT, got {a.shape} vs {b.shape}"
        )

    if window:
        win = hann2d(a.shape)
        a_w = a * win
        b_w = b * win
    else:
        a_w = a
        b_w = b

    mag_a = fft_magnitude(a_w)
    mag_b = fft_magnitude(b_w)
    lp_a = log_polar(mag_a, log_polar_rows, log_polar_cols)
    lp_b = log_polar(mag_b, log_polar_rows, log_polar_cols)

    scale, theta_deg, shift_rows, shift_cols, lp_psr = recover_scale_rotation(
        lp_a, lp_b
    )
    logger.info("FMT scale=%.4f theta=%.2f deg (log-polar PSR=%.2f)",
                scale, theta_deg, lp_psr)

    # Translate-only sanity check first: with scale~1 and rot~0 the content
    # should already largely align; otherwise warping handles it below.
    if abs(math.log(scale)) < 1e-3 and abs(theta_deg) < 1e-2:
        aligned = a
        rescale_used = 1.0
        rot_used = 0.0
    else:
        aligned = _resample_by_scale_rotation(a, scale, theta_deg)
        rescale_used = scale
        rot_used = theta_deg

    if translation_resolution is not None and aligned.shape != translation_resolution:
        aligned = cv2.resize(
            aligned, translation_resolution, interpolation=cv2.INTER_AREA
        )
    tx, ty, peak, t_psr = phase_correlate(aligned, b)
    logger.info("FMT translation=%.0f %.0f (translation PSR=%.2f)", tx, ty, t_psr)

    # Phase correlation's (tx,ty) is the shift such that B = aligned + (tx,ty).
    # The A->B affine needs the negation so that H_apply maps A-space points to
    # B-space points (validated: this recovers the injected transform exactly).
    t_true = (-float(tx), -float(ty))
    center = ((b.shape[1] - 1) / 2.0, (b.shape[0] - 1) / 2.0)
    H = _assemble_affine(rescale_used, rot_used, t_true[0], t_true[1], center)
    return FMTResult(
        scale=scale,
        theta_deg=theta_deg,
        tx=t_true[0],
        ty=t_true[1],
        H_apply=H,
        log_polar_shift_cols=float(shift_cols),
        log_polar_shift_rows=float(shift_rows),
        log_polar_psr=float(lp_psr),
        translation_psr=float(t_psr),
        peak_score=float(peak),
    )


def _assemble_affine(
    scale: float, theta_deg: float, tx: float, ty: float, center: tuple[float, float]
) -> np.ndarray:
    """Build the 3x3 affine mapping A-space -> B-space.

    The recovered scale+rotation are applied about the image ``center`` (the
    same convention the resample and the injection use), so the assembly must
    be a *center-of-rotation composition*::

        H_apply = T(c + t) . (scale * R(theta)) . T(-c)

    where ``T`` is a translation, ``R`` the rotation (cv2/image y-down
    orientation), and ``(tx, ty)`` the translation already validated against an
    injected transform. ``H_apply`` maps a point ``(x, y)`` in A to the point
    ``(x', y')`` in B where the same content appears.
    """
    th = math.radians(theta_deg)
    co, si = math.cos(th), math.sin(th)
    # cv2 / image convention (y-down): positive angle -> [[cos, sin], [-sin, cos]].
    R = np.array([[co, si], [-si, co]])
    R_s = np.eye(3)
    R_s[:2, :2] = scale * R

    cx, cy = center
    T_in = np.eye(3); T_in[0, 2], T_in[1, 2] = -cx, -cy
    T_out = np.eye(3); T_out[0, 2], T_out[1, 2] = cx + tx, cy + ty
    return T_out @ R_s @ T_in
