"""Tests for geoprior_tile_registration.py (Phase 11).

The geolocation-prior tile-vote module (Phase 11's answer to the Phase 10
blocker -- a full-canvas phase-correlation translation that is ~60-90 px
wrong even at the correct linear parameters) registers a pair by seeding
each tile's local search with the products' own geolocation data and
robustly fitting the Phase-8 5-DOF model to the tile votes.

The most important tests pin the *gates*:

- ``fit_5dof`` recovers a known affine from clean correspondences and
  ignores gross outliers;
- a real lunar crop (when present), warped by a KNOWN anisotropic
  transform, is recovered to the audit's Stage-A tolerance using an
  *identity* prior (predictions = A positions);
- ``synthetic_gate`` reports PASS on a real lunar crop, pinning the
  two-stage gate that the real-pair runner aborts on.

Plus small convention pins: the 5-DOF model keeps the Phase-8 parameter
convention, and tile centres form a uniform lattice inside the safe
interior (the uniform-distribution requirement from the problem statement).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from lunar_data_pipeline.geoprior_tile_registration import (
    TileVote,
    TileVoteParams,
    apply_h,
    fit_5dof,
    model_h,
    register,
    synthetic_gate,
    tile_centers,
)

_REAL_LABEL = Path(
    "D:/Multi-Modal-lunar-image-matching-system/"
    "ch2_ohr_ncp_20231004T0406038822_d_img_d18/"
    "data/calibrated/20231004/ch2_ohr_ncp_20231004T0406038822_d_img_d18.xml"
)
_HAVE_REAL = _REAL_LABEL.exists()

if _HAVE_REAL:
    from lunar_data_pipeline.image_loader import load_image, to_uint8
    from lunar_data_pipeline.pds4_parser import parse_label


def real_crop(n: int = 384) -> np.ndarray:
    """A real lunar-texture crop (or raise if product absent)."""
    p = parse_label(_REAL_LABEL)
    crop = to_uint8(load_image(
        p, row_range=(45000, 45000 + n), col_range=(400, 400 + n)
    ))
    return (crop.astype(np.float32) - crop.mean()) / (crop.std() + 1e-9)


def synthetic_texture(n: int = 256, seed: int = 0) -> np.ndarray:
    """Structured texture with sharp correlation peaks (blobs + noise).

    Pure random noise gives ambiguity-free but peakless NCC; a handful of
    gaussian blobs on noise gives the tile-vote machinery a sharp, localised
    correlation peak to lock on without real-data dependency.
    """
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((n, n)).astype(np.float32)
    for _ in range(24):
        cx, cy = rng.uniform(20, n - 20, 2)
        r = rng.uniform(4, 14)
        yy, xx = np.mgrid[0:n, 0:n]
        amp = rng.uniform(2.0, 4.0)
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    return (img - img.mean()) / (img.std() + 1e-9)


def _lattice_points(n: int, m: int = 8) -> np.ndarray:
    xs = np.linspace(8.0, n - 8.0, m)
    return np.array([(float(x), float(y)) for y in xs for x in xs])


# ---------------------------------------------------------------------------
# 5-DOF model convention pins
# ---------------------------------------------------------------------------

def test_model_h_convention() -> None:
    """The 5-DOF model keeps the Phase-8 convention: T(c+t) R(th) diag(sx,sy)
    T(-c); every representable linear part has det>0 (no mirror)."""
    c = (127.5, 127.5)
    sx, sy, th, tx, ty = 1.3, 0.8, 10.0, 15.0, -12.0
    H = model_h(sx, sy, th, tx, ty, c)
    assert np.allclose(H[2], [0, 0, 1])
    assert np.allclose(np.linalg.det(H[:2, :2]), sx * sy)
    # The 5-DOF model maps p -> L.p + t (H[:2, 2] == t): the origin goes to t.
    assert np.allclose(apply_h(H, np.array([[0.0, 0.0]]))[0], [tx, ty])
    # Rotation convention: R(90) diag(sx,sy) in image (y-down) coords.
    L = model_h(1.3, 0.8, 90.0, 0.0, 0.0, (0.0, 0.0))[:2, :2]
    assert np.allclose(L, [[0.0, 0.8], [-1.3, 0.0]])


def test_apply_h_accepts_2x3_and_3x3() -> None:
    H = model_h(1.0, 1.0, 0.0, 5.0, -3.0, (10.0, 10.0))
    pts = np.array([[0.0, 0.0], [4.0, 7.0]])
    r3 = apply_h(H, pts)
    r23 = apply_h(H[:2], pts)
    assert np.allclose(r3, r23)
    assert np.allclose(r3[0], [5.0, -3.0])  # origin -> t = (5, -3)
# ---------------------------------------------------------------------------
# Robust 5-DOF fit
# ---------------------------------------------------------------------------

def test_fit_5dof_recovers_known_affine() -> None:
    """Clean correspondences (lattice mapped by a known 5-DOF transform,
    plus 1 px noise) must be recovered to tight tolerance by the estimator
    that the tile-vote stage feeds."""
    rng = np.random.default_rng(0)
    n = 256
    a_pts = _lattice_points(n, m=10)
    center = ((n - 1) / 2.0, (n - 1) / 2.0)
    p_true = (1.3, 0.8, 10.0, 15.0, -12.0)
    b_pts = apply_h(model_h(*p_true, center), a_pts)
    b_pts = b_pts + rng.normal(0, 1.0, b_pts.shape)
    votes = [
        TileVote(ax=float(ax), ay=float(ay), px=float(bx), py=float(by),
                 mx=float(bx), my=float(by), score=1.0)
        for (ax, ay), (bx, by) in zip(a_pts, b_pts)
    ]
    p5, info = fit_5dof(votes, center, TileVoteParams())
    assert info["n_inliers"] == len(votes)
    # Tolerances reflect the 1 px noise floor over a ~128 px lever arm.
    assert abs(p5[0] - p_true[0]) < 1e-2
    assert abs(p5[1] - p_true[1]) < 1e-2
    assert abs(p5[2] - p_true[2]) < 1e-1
    assert abs(p5[3] - p_true[3]) < 1.0
    assert abs(p5[4] - p_true[4]) < 1.0


# ---------------------------------------------------------------------------
# register + gate (real-crop guarded)
# ---------------------------------------------------------------------------

def test_register_raises_when_predictions_unusable() -> None:
    """If the navigation prior puts every prediction outside the canvas, no
    tile has a usable search window and register must raise rather than
    return a silently garbage transform."""
    img = synthetic_texture(256)
    predict_b = lambda pts: np.full_like(np.atleast_2d(pts), 1e6)  # noqa: E731
    with pytest.raises(ValueError):
        register(img, img, predict_b, TileVoteParams())


@pytest.mark.skipif(not _HAVE_REAL, reason="real OHRC product not present")
def test_register_recovers_injected_transform_on_real_crop() -> None:
    """The Phase-11-audit Stage-A replication: a real lunar crop warped by a
    KNOWN anisotropic transform is recovered with an *identity* prior at the
    audit's exact configuration (tile=56, margin=80, NCC, single pass) and
    tolerance (0.01/0.01/0.5 deg/3/3 px)."""
    from lunar_data_pipeline.geoprior_tile_registration import warp_content

    a = real_crop(384)
    params = TileVoteParams(tile=56, margin=80, mode="ncc",
                            refine_ecc=False, iterations=1)
    b = warp_content(a, 0.947, 1.085, -4.6, 8.0, -6.0)
    center = ((a.shape[1] - 1) / 2.0, (a.shape[0] - 1) / 2.0)

    res, _ = register(a, b, lambda pts: np.atleast_2d(pts).astype(float),
                      params, center)
    dth = (res.theta_deg + 4.6 + 180.0) % 360.0 - 180.0
    assert abs(res.sx - 0.947) < 0.01
    assert abs(res.sy - 1.085) < 0.01
    assert abs(dth) < 0.5
    assert abs(res.tx - 8.0) < 3.0
    assert abs(res.ty - (-6.0)) < 3.0


@pytest.mark.skipif(not _HAVE_REAL, reason="real OHRC product not present")
def test_synthetic_gate_passes_on_real_crop() -> None:
    """The full two-stage gate must PASS on a real lunar crop. This pins the
    audit replication (Stage A) AND the perturbed-prior pipeline (Stage B)
    that the real-pair runner aborts on if it ever regresses."""
    gate = synthetic_gate(real_crop(384))
    assert gate["verdict"] == "PASS"
    assert gate["stage_a"]["verdict"] == "PASS"
    assert gate["stage_b"]["verdict"] == "PASS"
def test_fit_5dof_excludes_gross_outliers() -> None:
    """One grossly wrong correspondence must be excluded by the MAD inlier
    rule (fixed rule, no tuning), driving the residual down vs a plain fit."""
    rng = np.random.default_rng(1)
    n = 256
    a_pts = _lattice_points(n, m=8)
    center = ((n - 1) / 2.0, (n - 1) / 2.0)
    p_true = (0.95, 1.1, -4.0, 8.0, -6.0)
    b_pts = apply_h(model_h(*p_true, center), a_pts)
    b_pts = b_pts + rng.normal(0, 0.5, b_pts.shape)
    b_pts[0] += [60.0, 40.0]  # a gross outlier
    votes = [
        TileVote(ax=float(ax), ay=float(ay), px=float(bx), py=float(by),
                 mx=float(bx), my=float(by), score=0.0)
        for (ax, ay), (bx, by) in zip(a_pts, b_pts)
    ]
    p5, info = fit_5dof(votes, center, TileVoteParams())
    assert info["n_inliers"] < info["n_votes"]
    # The fit should still be close to the truth: residual stays small.
    assert info["resid_median"] < 2.0
    assert abs(p5[3] - p_true[3]) < 2.0
    assert abs(p5[4] - p_true[4]) < 2.0


def test_tile_centers_form_uniform_interior_lattice() -> None:
    """Tile centres cover the canvas uniformly (the problem statement's
    uniform-distribution requirement) and stay inside the safe interior."""
    params = TileVoteParams()
    w, h = 384, 384
    centers = tile_centers(w, h, params)
    assert len(centers) == params.n_tiles ** 2
    t2 = params.tile // 2
    assert centers[:, 0].min() >= t2 and centers[:, 0].max() <= w - t2
    assert centers[:, 1].min() >= t2 and centers[:, 1].max() <= h - t2
    # Uniform spacing: consecutive lattice steps are equal.
    step_x = np.diff(np.unique(centers[:, 0]))
    step_y = np.diff(np.unique(centers[:, 1]))
    assert np.allclose(step_x, step_x[0])
    assert np.allclose(step_y, step_y[0])
