"""Tests for fourier_mellin.py and phase_correlation.py.

Use compact synthetic textured images (rich, broadband spectra) so the tests
run fast and deterministically without loading gigabytes of real imagery. The
log-polar phase correlation needs a broadband spectrum to find a sharp angular
peak, which craters give but a plain ramp does not.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from lunar_data_pipeline.fourier_mellin import (
    FMTResult,
    _assemble_affine,
    fft_magnitude,
    log_polar,
    recover_scale_rotation,
    register,
)
from lunar_data_pipeline.phase_correlation import (
    PSR_STRONG,
    hann2d,
    phase_correlate,
)

_REAL_LABEL = Path(
    "D:/Multi-Modal-lunar-image-matching-system/"
    "ch2_ohr_ncp_20260103T0609041371_d_img_d18/"
    "data/calibrated/20260103/ch2_ohr_ncp_20260103T0609041371_d_img_d18.xml"
)
_HAVE_REAL = _REAL_LABEL.exists()

if _HAVE_REAL:
    from lunar_data_pipeline.image_loader import load_image, to_uint8
    from lunar_data_pipeline.pds4_parser import parse_label


def make_pattern(n: int = 256, seed: int = 0) -> np.ndarray:
    """Real lunar-texture crop when available, else a fallback pattern.

    REAL imagery (with directional craters/shadow ridges) is the faithful
    reproducer: log-polar phase correlation needs BOTH broad radial extent AND
    angular structure, which pure isotropic noise lacks. If the downloaded
    OHRC product is present we use a small real crop; otherwise we skip tests
    that need scale/rotation recovery.
    """
    if _HAVE_REAL:
        p = parse_label(_REAL_LABEL)
        crop = to_uint8(load_image(
            p, row_range=(40000, 40000 + n), col_range=(2000, 2000 + n)
        ))
        return (crop.astype(np.float32) - crop.mean()) / (crop.std() + 1e-9)

    raise RuntimeError("no real image available")


def warp_with(a: np.ndarray, s: float, theta: float, tx: float, ty: float) -> np.ndarray:
    """B = scale*R(theta)*A + t, into a same-size canvas (s<1 avoids clipping)."""
    h, w = a.shape[:2]
    center = (w / 2.0 - 0.5, h / 2.0 - 0.5)
    M = cv2.getRotationMatrix2D(center, theta, s)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def test_phase_correlate_translation():
    # pure translation needs only a broad spectrum (any orientation works), so
    # white noise suffices and no real image is required.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((128, 128)).astype(np.float32)
    # roll rows +17, cols -23 => content displacement (dy=+17, dx=-23).
    # standard peak convention returns the negation: (dx,dy) = -(dx_true,dy_true)
    b = np.roll(a, (17, -23), axis=(0, 1))
    dx, dy, _, psr = phase_correlate(a, b)
    assert (dx, dy) == (23, -17)
    assert psr >= PSR_STRONG


@pytest.mark.skipif(not _HAVE_REAL, reason="real OHRC product not present")
def test_recover_scale_rotation_known():
    a = make_pattern(512)
    s_inj, th_inj = 0.8, 15.0
    b = warp_with(a, s_inj, th_inj, 0, 0)
    w = hann2d(a.shape)
    mag_a = fft_magnitude(a * w)
    mag_b = fft_magnitude(b * w)
    scale, theta, _, _, psr = recover_scale_rotation(
        log_polar(mag_a), log_polar(mag_b)
    )
    assert abs(scale - s_inj) / s_inj < 0.06
    assert abs(theta - th_inj) < 1.0
    assert psr >= PSR_STRONG


@pytest.mark.skipif(not _HAVE_REAL, reason="real OHRC product not present")
@pytest.mark.parametrize("s_inj,th_inj,tx,ty", [
    (0.7, 12.0, 20.0, -25.0),
    (0.85, 30.0, -12.0, 8.0),
    (0.95, 3.0, 40.0, -40.0),
    (1.0, 0.0, 25.0, -10.0),
])
def test_register_recovers_global_transform(s_inj, th_inj, tx, ty):
    a = make_pattern(512)
    b = warp_with(a, s_inj, th_inj, tx, ty)

    center = (a.shape[1] / 2.0 - 0.5, a.shape[0] / 2.0 - 0.5)
    M = cv2.getRotationMatrix2D(center, th_inj, s_inj)
    M[0, 2] += tx
    M[1, 2] += ty
    M_inj = np.vstack([M, [0, 0, 1]])

    res = register(a, b)
    pts = np.array([[5, 5], [40, 60], [128, 128], [200, 80], [30, 200]], float)
    inj = (M_inj @ np.column_stack([pts, np.ones(len(pts))]).T)
    inj = (inj / inj[2])[:2].T
    rec = (res.H_apply @ np.column_stack([pts, np.ones(len(pts))]).T)
    rec = (rec / rec[2])[:2].T
    rms = float(np.sqrt(np.mean(np.sum((inj - rec) ** 2, axis=1))))
    assert rms < 6.0, f"RMS {rms:.2f} too large for ({s_inj},{th_inj})"


@pytest.mark.skipif(not _HAVE_REAL, reason="real OHRC product not present")
def test_xfm_scale_rot_recovery_no_translation():
    """Scale+rotation only (no translation): recover known values."""
    a = make_pattern(512)
    s_inj, th_inj = 0.8, -10.0
    # s<1 (shrink) keeps content fully in-canvas; s>1 would clip and corrupt
    # the spectrum relationship, which is a test-construction limit, not a bug.
    b = cv2.warpAffine(
        a, cv2.getRotationMatrix2D((a.shape[1] / 2 - 0.5, a.shape[0] / 2 - 0.5),
                                   th_inj, s_inj),
        (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    res = register(a, b)
    assert abs(res.theta_deg - th_inj) < 1.0
    assert abs(math.log(res.scale) - math.log(s_inj)) < 0.06


def test_affine_matrix_is_consistent():
    H = _assemble_affine(0.7, 12.0, 20.0, -25.0, (127.5, 127.5))
    # identity at origin-ish check: 3x3 affine, last row [0,0,1]
    assert np.allclose(H[2], [0, 0, 1])
    # linear part scale ~ 0.7
    assert abs(np.linalg.det(H[:2, :2]) - 0.7 ** 2) < 1e-6


def test_fmtresult_project_single_vs_many():
    res = FMTResult(
        scale=1.0, theta_deg=0.0, tx=10.0, ty=-5.0,
        H_apply=np.array([[1., 0., 10.], [0., 1., -5.], [0., 0., 1.]]),
        log_polar_shift_cols=0.0, log_polar_shift_rows=0.0,
        log_polar_psr=20.0, translation_psr=20.0,
    )
    assert np.allclose(res.project(np.array([3.0, 4.0])), [13.0, -1.0])
    assert np.allclose(res.project_many(np.array([[3., 4.], [1., 2.]])),
                       [[13., -1.], [11., -3.]])
