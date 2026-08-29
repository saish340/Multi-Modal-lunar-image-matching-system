"""Tests for anisotropic_affine_search.py (Phase 8).

The anisotropic-affine search recovers an independent x/y scale plus rotation
and translation between two equal-res views. The synthetic gate uses a REAL
lunar crop (when the downloaded product is present) warped with a KNOWN
anisotropic transform, and checks the search recovers it to tight tolerance
--- the same criterion ``run_anisotropic_synthetic_test.py`` gates on.

The second class of tests pins down the Phase 7/8 convention-facing behaviour:
``warp_content`` must place injected content exactly at ``H_apply . p``
(forward consistency), and determinant sign handling, so that any future
change to the composition convention is caught as a regression.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from lunar_data_pipeline.anisotropic_affine_search import (
    _assemble_affine,
    _linear_matrix,
    resample_for_search,
    search,
    warp_content,
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


def real_crop(n: int = 384, seed: int = 0) -> np.ndarray:
    """A real lunar-texture crop (or raise if product absent)."""
    p = parse_label(_REAL_LABEL)
    crop = to_uint8(load_image(
        p, row_range=(40000, 40000 + n), col_range=(2000, 2000 + n)
    ))
    return (crop.astype(np.float32) - crop.mean()) / (crop.std() + 1e-9)


def test_linear_matrix_is_rot_diag():
    """L = R(theta) . diag(sx, sy), y-down rotation convention."""
    L = _linear_matrix(1.3, 0.8, 90.0)
    # R(90) diag(1.3,0.8); R(90) = [[0,1],[-1,0]] -> [[0,0.8],[-1.3,0]]
    assert np.allclose(L, [[0.0, 0.8], [-1.3, 0.0]])


def test_affine_matrix_consistent_positive_det():
    H = _assemble_affine(1.3, 0.8, 10.0, 15.0, -12.0, (191.5, 191.5))
    assert np.allclose(H[2], [0, 0, 1])
    # linear part det = sx*sy (positive scales => positive determinant, no mirror)
    assert np.allclose(np.linalg.det(H[:2, :2]), 1.3 * 0.8)


def test_warp_content_forward_consistency():
    """warp_content must place content at exactly H_apply.hom(p): the model
    the search assumes for B. A regression here breaks the synthetic gate."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((96, 96)).astype(np.float32)
    sx, sy, th, tx, ty = 1.2, 0.9, 12.0, 4.0, -6.0
    b = warp_content(a, sx, sy, th, tx, ty)
    H = _assemble_affine(sx, sy, th, tx, ty, ((96 - 1) / 2, (96 - 1) / 2))
    # A bright impulse at the image centre should reappear at H applied to it,
    # i.e. at (cx, cy) + (tx, ty) in the warped canvas.
    a0 = np.zeros((96, 96), np.float32)
    cx, cy = 48.0, 48.0
    a0[int(cx), int(cy)] = 1.0
    b0 = warp_content(a0, sx, sy, th, tx, ty)
    ht = H @ np.array([cx, cy, 1.0])
    ht = (ht / ht[2])[:2]
    # The impulse should land near (cx+tx, cy+ty) +/- 1 px (bilinear + border).
    assert abs(np.argmax(b0) % 96 - ht[0]) <= 1.0
    assert abs(np.argmax(b0) // 96 - ht[1]) <= 1.0


def test_resample_for_search_is_t_zero_warp():
    a = np.random.default_rng(1).standard_normal((64, 64)).astype(np.float32)
    r = resample_for_search(a, 1.1, 0.9, 5.0)
    w = warp_content(a, 1.1, 0.9, 5.0, 0.0, 0.0)
    assert np.allclose(r, w)


@pytest.mark.skipif(not _HAVE_REAL, reason="real OHRC product not present")
def test_synthetic_gate_recovers_anisotropic_transform():
    """A real lunar crop warped by a KNOWN anisotropic affine must be
    recovered to gate tolerances (unseeded ranges). This is the Phase 8
    validated win."""
    a = real_crop(384)
    sx, sy, th, tx, ty = 1.3, 0.8, 10.0, 15.0, -12.0
    b = warp_content(a, sx, sy, th, tx, ty)
    res = search(a, b)
    assert abs(res.sx - sx) <= 0.06, f"sx {res.sx} vs {sx}"
    assert abs(res.sy - sy) <= 0.06, f"sy {res.sy} vs {sy}"
    assert abs(res.theta_deg - th) <= 1.5, f"theta {res.theta_deg} vs {th}"
    # GT-projection RMS at a handful of distributed A-points.
    pts = np.array([[5, 5], [60, 120], [192, 192], [320, 100], [40, 360]], float)
    H_true = _assemble_affine(sx, sy, th, tx, ty, ((384 - 1) / 2, (384 - 1) / 2))
    gt = (H_true @ np.column_stack([pts, np.ones(len(pts))]).T)
    gt = (gt / gt[2])[:2].T
    rec = (res.H_apply @ np.column_stack([pts, np.ones(len(pts))]).T)
    rec = (rec / rec[2])[:2].T
    rms = float(np.sqrt(np.mean(np.sum((gt - rec) ** 2, axis=1))))
    assert rms <= 2.0, f"GT-projection RMS {rms} px"


def test_determinant_sign_distinguishes_positive_scale_from_mirror():
    """Positive-scale anisotropic model: every representable L has det>0.
    A reflected pair (det<0) is therefore OUT OF MODEL -- it needs an explicit
    axis flip before search, which callers must apply (see the runner's
    --flip-b path and the Phase 8 findings)."""
    for sx in (0.7, 1.0, 1.6):
        for sy in (0.7, 1.0, 1.6):
            assert np.linalg.det(_linear_matrix(sx, sy, 20.0)) > 0
