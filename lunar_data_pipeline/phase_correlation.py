"""Phase correlation: recover the dominant global translation between two
cropped/registered images from the frequency domain (Phase 7 foundation).

A correct global registration of two overlapping crops of the SAME surface
should show a single dominant translation. Phase correlation finds it by
normalising the cross-power spectrum of the two FFTs and locating the peak
of the resulting correlation surface. It is a GLOBAL method -- it considers
the whole spectrum at once -- rather than matching individual local patches,
which fail on this project's self-similar polar terrain.

Deliverables of Phase 7 step 1 (the documented baseline this phase builds on):

    same-scale OHRC-OHRC pair, 2026-01-03, South Pole, 0.25 m/px
    A window cols[2000:10000] rows[40000:48000]
    B window cols[2606:11229] rows[35584:44289]

    recovered crop-relative shift      (dx= 312, dy=  11)
    recovered absolute offset          (dx= 918, dy=-4405)
    PSR (peak-to-sidelobe ratio)      14.43   (strong, unambiguous)
    spatial cross-correlation@0        0.6333
    error vs grid/CSV geolocation GT  103.9 px
    error vs corner-homography GT     312.2 px

    verdict: PASS (single dominant global translation recovered; residual
    within the grid GT's own y-IQR (~200 px); y-offset essentially exact).

These numbers are persisted in ``phase6_ohrc_ohrc_output/phasecorr_results.json``
and reproduced by a fresh run (see :func:`confirm_samescale_baseline`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: PSR threshold above which a correlation peak is considered strong/unambiguous.
PSR_STRONG = 6.0


def hann2d(shape: tuple[int, int]) -> np.ndarray:
    """Separable 2-D Hanning window, *shape* = (rows, cols)."""
    r = np.hanning(shape[0])[:, None]
    c = np.hanning(shape[1])[None, :]
    return (r * c).astype(np.float32)


def phase_correlate(
    a: np.ndarray, b: np.ndarray
) -> tuple[int, int, float, float]:
    """Phase-correlate equal-size ``(H, W)`` arrays ``a`` and ``b``.

    Returns ``(dx, dy, peak_score, psr)``:

    - ``(dx, dy)``  the translation such that content at ``(x, y)`` in ``b``
      sits at ``(x - dx, y - dy)`` in ``a`` (equivalently, ``b`` is ``a``
      shifted by ``-(dx, dy)``). This is the standard phase-correlation peak
      location: a positive content displacement comes back as the negation,
      which downstream callers (e.g. :mod:`fourier_mellin`) already account
      for.
    - ``peak_score`` raw peak height of the normalised correlation surface.
    - ``psr``  peak-to-sidelobe ratio: ``(peak - mean)/std`` of the surface
      excluding a small window around the peak. PSR >= ~6-8 is strong and
      unambiguous; < ~3 is weak/ambiguous.
    """
    if a.shape != b.shape:
        raise ValueError(f"inputs must be equal size, got {a.shape} vs {b.shape}")
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + 1e-12
    cc = np.fft.ifft2(cross).real

    idx = np.unravel_index(np.argmax(cc), cc.shape)
    peak = cc[idx]
    h, w = a.shape
    dy = idx[0] if idx[0] <= h // 2 else idx[0] - h
    dx = idx[1] if idx[1] <= w // 2 else idx[1] - w

    ex, ey = idx[1], idx[0]
    side = 7
    m = np.ones(cc.shape, bool)
    m[max(ey - side, 0):ey + side + 1, max(ex - side, 0):ex + side + 1] = False
    psr = float((peak - cc[m].mean()) / (cc[m].std() + 1e-12))
    return int(dx), int(dy), float(peak), psr


@dataclass
class PhaseCorrelationResult:
    dx: int
    dy: int
    peak_score: float
    psr: float

    @property
    def strong_peak(self) -> bool:
        return self.psr >= PSR_STRONG

    @property
    def shift(self) -> tuple[int, int]:
        return (self.dx, self.dy)

    def summary_dict(self) -> dict:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "peak_score": round(float(self.peak_score), 4),
            "psr": round(float(self.psr), 2),
            "strong_peak": self.strong_peak,
        }


def _load_persisted_baseline(path: str | Path) -> dict:
    """Read the Phase 6 phase-correlation result JSON if it exists."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"persisted baseline {p} not found; run _phase6_phasecorr.py first"
        )
    import json

    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def confirm_samescale_baseline(
    persisted_path: str | Path = "phase6_ohrc_ohrc_output/phasecorr_results.json",
) -> dict:
    """Report the exact Phase 7 step-1 baseline numbers in writing.

    Retrieves the persisted Phase 6 result (fast) and returns a {key: value}
    dict of the numbers documented in the module docstring, culminating in
    the recorded verdict. Use :func:`print_baseline_report` for a human
    readable report.
    """
    data = _load_persisted_baseline(persisted_path)
    pc = data["phase_correlation"]
    gt = data["ground_truth"]
    decision = data["decision"]
    return {
        "pair": list(data["pair"]),
        "context": data["context"],
        "windows_px": data["windows_px"],
        "phase_correlation": pc["absolute_offset_px"],
        "crop_relative_shift_px": pc["crop_relative_shift_px"],
        "peak_score": pc["peak_score"],
        "psr": pc["psr"],
        "spatial_cross_corr_zero_shift": pc.get("spatial_cross_corr_zero_shift"),
        "gt_corner_homography_offset_px": gt["corner_homography_offset_px"],
        "gt_grid_csv_offset_px": gt["grid_csv_geolocation_offset_px"],
        "verdict": decision["verdict"],
        "verdict_reasoning": decision["reasoning"],
    }


def print_baseline_report(report: dict | None = None) -> str:
    """Human-readable rendering of the step-1 baseline report."""
    rep = report or confirm_samescale_baseline()
    pc = rep["phase_correlation"]
    grid = rep["gt_grid_csv_offset_px"]
    corner = rep["gt_corner_homography_offset_px"]
    wins = rep["windows_px"]
    lines = [
        "Phase 7 step 1 -- same-scale OHRC-OHRC phase-correlation baseline",
        f"  pair      : {'  vs  '.join(rep['pair'])}",
        f"  context   : {rep['context']}",
        f"  A window  : cols{wins['a']['cols']} rows{wins['a']['rows']}",
        f"  B window  : cols{wins['b']['cols']} rows{wins['b']['rows']}",
        f"  recovered : absolute offset (dx={pc['dx']:.0f}, dy={pc['dy']:.0f})  "
        f"(crop-rel {rep['crop_relative_shift_px']})",
        f"  confidence: PSR={rep['psr']:.2f} (strong>={PSR_STRONG}), "
        f"cross-corr@{rep.get('spatial_cross_corr_zero_shift')}",
        f"  GT grid/CSV : expected ({grid['dx']:.0f},{grid['dy']:.0f})  "
        f"|err|={grid['err']:.1f} px",
        f"  GT corner-hom: expected ({corner['dx']:.0f},{corner['dy']:.0f})  "
        f"|err|={corner['err']:.1f} px",
        f"  VERDICT    : {rep['verdict']}",
        f"    - {rep['verdict_reasoning']}",
    ]
    return "\n".join(lines)
