# Phase 10 Findings: Local Refinement to Close the Precision Gap

This phase adds a refinement stage on top of the Phase 8/9 coarse-to-fine
anisotropic-affine search and tests whether it closes (or narrows) the gap
between the recovered registration (111 px / 171 px median display-space error)
and the geolocation-derived accuracy floor (26 px / 40 px).

**Headline result (honest): refinement does NOT close the real-data gap
(111.3 → 111.2 px, 170.7 → 170.6 px). The reason is diagnostic and important:
the gap is NOT a coarse-grid precision (quantization) problem. It is a
wrong-basin problem — the phase-correlation PSR optimum already sits far from
the geolocation truth, and no local refinement from that optimum can cross to
the true basin.** A local optimizer only refines *within* the basin the coarse
search selected; it cannot re-select the basin.

---

## 1. Approach chosen: continuous local optimizer (scipy Nelder-Mead)

We chose a **continuous local optimizer** over a finer local grid, and
implemented it as `refine_transform()` in `lunar_data_pipeline/anisotropic_affine_search.py`
(Nelder-Mead via `scipy.optimize.minimize`).

**Justification.**
- The coarse-to-fine search is *quantized*: its best `(sx, sy, theta)` can only
  land on a grid node. A finer local grid would still be quantized to its own
  step size, whereas a per-parameter continuous move can land on non-grid values.
- Phase-correlation PSR is a continuous (but **non-smooth / noisy**) function of
  `(sx, sy, theta)`. Gradient-based methods are unstable on such an objective;
  **Nelder-Mead is derivative-free and robust** to this roughness, which is why
  it is preferred here.
- Translation stays as its own final phase-correlation pass (a separate integer
  shift recovery), consistent with the existing pipeline, and is re-fitting
  implicitly at every PSR evaluation anyway (each phase-correlate returns the
  best integer shift at the candidate `(sx,sy,theta)`).

Implementation notes:
- `sx`, `sy` are optimized in **log space** (multiplicative scales are better
  conditioned than additive deltas); `theta` in degrees.
- Bounds deliberately narrow: ±15% scale, ±8° rotation — i.e. a *refinement*
  that stays in the coarse basin (this is the honest definition of refinement;
  re-searching the whole space would be a new coarse pass, not refinement).
- `refine` is additive and **default-off is not required**: `search(refine=False)`
  reproduces the exact Phase 8/9 coarse pipeline as the baseline, and
  `search(refine=True)` (the default) appends the optimizer. The coarse search
  is never removed — it is the required first step that seeds refinement.

## 2. Synthetic validation gate — PASS (with an instructive caveat)

`run_phase10_synthetic_refine.py` injects a known anisotropic transform
(sx=1.3, sy=0.8, θ=10°, t=(15,−12)) into a real lunar crop, then refines from a
**deliberately coarse (perturbed) estimate** that simulates an off-grid-node
coarse result (sx=1.32, sy=0.83, θ=8.5°).

| quantity | value |
|---|---|
| perturbed start, GT projection RMS | 14.53 px |
| after `refine` (224 PC evals) | **3.76 px** (−74%) |
| end-to-end `search` (coarse→polish→refine) | **0.002 px** |

Gate: refine cuts the start's RMS by >60%, and end-to-end stays ≤2 px → **PASS**.

**Instructive caveat (this is the finding that predicts the real-data outcome).**
We verified the 3.76 px plateau is a *genuine PSR-landscape limit*, not a
termination issue: tightening `xatol/fatol` and raising iterations to 8000/3000
changes the converged optimum not at all (sx=1.2807, sy=0.8154, θ=9.04° every
time). The PSR surface is simply **flat near the optimum** for a 384 px crop, so
the optimizer cannot distinguish (1.28, 0.82, 9.04°) from the true (1.3, 0.8,
10.0°). When seeded by the actual coarse/polish result (which is already near
the true basin), refinement yields an **exact** answer (0.002 px).

**Conclusion from synthetic:** refinement reliably improves precision *within a
basin*, but it cannot move a solution to a *different* basin. Its practical value
is therefore entirely determined by whether the coarse search landed in the right
basin.

## 3. Real-pair before/after

`run_phase10_refine_pair.py` runs the full mirror-corrected pipeline twice per
pair — `refine=False` (Phase 8/9 baseline) and `refine=True` (Phase 10) — scoring
both against the **same** geolocation GT points in display space. scipy was
already in requirements.txt; no dependency change was needed.

**Phase 8 pair** (20231004 × ncn_20250707, square region, mirrored → `--flip-b`):

| mode | sx | sy | θ° | PSR | median err | within 100 | within 300 |
|---|---|---|---|---|---|---|---|
| coarse (baseline) | 0.3059 | 0.4472 | 20.51 | 12.58 | **111.3 px** | 131/300 | 300/300 |
| coarse + refine | 0.3062 | 0.4474 | 20.50 | 12.58 | **111.2 px** | 131/300 | 300/300 |

**60°N pair** (20250516 × nca_20200607, no flip):

| mode | sx | sy | θ° | PSR | median err | within 100 | within 300 |
|---|---|---|---|---|---|---|---|
| coarse (baseline) | 0.7497 | 1.0013 | −19.98 | 11.56 | **170.7 px** | 49/300 | 294/300 |
| coarse + refine | 0.7496 | 1.0013 | −19.98 | 11.56 | **170.6 px** | 49/300 | 294/300 |

## 4. Comparison against each pair's geolocation ceiling

The relevant ceilings (documented in Phase 8/9) are the geolocation-derived
*achievable* transform accuracy:

- Phase 8 pair: geo affine scores ~26 px median; grid quant floor ~40 px.
- 60°N pair: geo affine scores ~40 px median; GT-grid quant floor P50 ≈ 27.5 px.

In both cases refinement lands **nowhere near** the ceiling:

| pair | coarse | coarse+refine | ceiling | gap closed |
|---|---|---|---|---|
| Phase 8 | 111.3 px | 111.2 px | ~26–40 px | **0%** |
| 60°N | 170.7 px | 170.6 px | ~40 px | **0%** |

## 5. Interpretation: why refinement cannot help here

The refinement stage works exactly as designed — it maximises PSR continuously,
converges in a few hundred evaluations, and never *degrades* a result. The reason
it closes none of the real-data gap is that **the coarse search was already at
the local PSR basin maximum**, so the added optimizer immediately re-converges to
(almost) that same point (all four numbers change by <0.0005 in scale). Refinement
was aimed at a "coarse-grid precision gap" that, on these real pairs, does not
exist.

The actual bottleneck is **basin selection**, not intra-basin precision:

- 60°N: the search recovers sy≈1.001 and θ≈−20°, but the geolocation-derived
  residual needs sy≈0.71 and θ≈−172° (≈θ+180). A 40%-off sy is **not** a
  quantization miss — it is a fundamentally different transform that maximises
  global spectral phase-correlation alignment. With ±15% scale bounds the
  refine stage cannot even represent sy=0.71, let alone reach it.
- Phase 8: same story with a different `(sx, sy, θ)`; the recovered transform is
  a coherent PSR optimum that disagrees with per-pixel geolocation truth at the
  ~110 px scale.

This is the **repetitive / self-similar polar-terrain ambiguity resurfacing at a
finer scale**, exactly as feared in the brief: phase correlation optimises a
*global, spectral* alignment that is insensitive to which local features pair up,
so on repetitive terrain it can lock onto a wrong-but-strong global shift. The PSR
landscape near that optimum is smooth (confirmed synthetically), so refinement
faithfully returns you to the *wrong* optimum rather than escaping it.

**So:** Phase 10 does not degrade the baseline (good — the additive stage is
safe), and it conclusively rules out "the precision gap is just grid resolution."
The remaining gap is a **coarse-search objective mismatch** (PSR-vs-geolocation),
which a local re-optimizer is structurally incapable of fixing. Closing it would
require seeding or re-scoring the coarse search with geolocation-informed
geometric priors (e.g. constrain sx/sy from the sensor resolution ratio and orbit
geometry) rather than pure spectral PSR.

## 6. Artifacts

- `run_phase10_refine_pair.py` — coarse vs coarse+refine before/after runner (shared GT scoring).
- `run_phase10_synthetic_refine.py` — synthetic refinement gate (PASS).
- `lunar_data_pipeline/anisotropic_affine_search.py` — `refine_transform()` + `refine` alias; `search(refine=...)`, `n_refine` field.
- `lunar_data_pipeline/tests/test_anisotropic_affine_search.py` — 3 new regression tests.
- `phase10_output/` — `result_p8_sq_flip.json`, `result_p60n_nof.json`, overlay PNGs (`ov_*_coarse`, `ov_*_refined`).

Test suite: **181 passed** (178 prior + 3 new).
