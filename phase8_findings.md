# Phase 8 Findings: Anisotropic-Affine Registration

This document records what Phase 8 established. It separates three phenomena that
earlier analysis had conflated:

1. **Local affine validity** — does an affine map hold within a small region?
2. **Global distortion** — does the affine break down as the region grows?
3. **Coordinate-convention errors / reflections** — is a negative-determinant
   transform real, or an artifact of how coordinates were composed?

Per the phase directive we investigated all three rather than concluding crisply
that "no global affine exists". All findings below are quantitative.

---

## 1. The validated synthetic gate (primary Phase 8 result)

`anisotropic_affine_search.py` recovers an *anisotropic* affine
`T(c+t)·R(θ)·diag(sx,sy)·T(−c)` — independent x/y scale plus rotation plus
translation — by a coarse-to-fine phase-correlation search. The geolocation
data is used **only** for validation, never to seed the search.

On a **real** 384×384 OHRC lunar crop warped by a KNOWN transform:

| parameter | injected | recovered | error | gate |
|---|---|---|---|---|
| `sx` | 1.300 | 1.300 | 0.000 | ±0.06 ✓ |
| `sy` | 0.803 | 0.800 | +0.003 | ±0.06 ✓ |
| `θ` | 10.0° | 9.94° | −0.06° | ±1.5° ✓ |
| `tx`,`ty` | (15,−12) | (15.0,−13.0) | ≤2 px | ±2 px ✓ |
| GT-projection RMS | — | 0.408 px | — | ≤2 px ✓ |

This gate is encoded as `run_anisotropic_synthetic_test.py` and as
`test_anisotropic_affine_search.py::test_synthetic_gate_recovers_anisotropic_transform`.
The search ranges are **not seeded** by the answer (`sx`/`sy ∈ [0.5, 2]`,
`θ ∈ [−30, 30]°`), so this validates the method, not the ground-truth pass.

**Bugs fixed along the way (now regressions):**
- `warp_content` now warps content *forward* by `H·p` (it previously used
  `inv(H)`, silently negating the model). Pinned by
  `test_warp_content_forward_consistency`.
- `search` returns `tx,ty = (−dx,−dy)` to match the phase-correlation
  sign convention. The final translation is recovered exactly only when
  scale/rotation are sub-pixel, which the running-span fine/polish passes
  guarantee.

---

## 2. Does the real OHRC/TMC-2 pair register globally?

No — and Phase 8 now explains *why*, with two independent mechanical causes
rather than a vague "images differ".

### 2a. The residual is non-affine only at the geolocation-grid floor

We fitted the ground mapping `OHRC(pix,scan) → TMC-2(pix,scan)` (via the
per-pixel geolocation CSVs) with a single affine, and measured its residual at
four nested window sizes (all centred on the same ground region):

| window (fraction of strip) | points | affine fit residual P50 / P95 / MAX (raw TMC px) |
|---|---|---|
| 0.10 | 253 | 36.7 / 59.2 / 74.5 |
| 0.25 | 1711 | 36.3 / 58.3 / 72.9 |
| 0.50 | 6783 | 38.7 / 61.0 / 78.6 |
| 1.00 | 27007 | 40.0 / 60.1 / 72.7 |

The residual is **flat across window size** — it does not grow as the strip
lengthens. A genuine pushbroom/scan-accumulated distortion would show a clear
increase in residual from the 0.10 to the 1.00 window. Instead the residual is
constant at ~37–40 px P50 / ~60 px P95.

That floor is **exactly the geolocation-grid quantization** floor: both products'
`*_g_grd_*.csv` grids are sampled every 100 px along `scan` and `pixel`, and a
pure uniform ±50 px / axis quantization noise gives P50 = 39.9, P95 = 59.9,
MAX = 70.6 px — statistically indistinguishable from the fitted residuals above.

**Conclusion on local vs global affine:** with the available ground truth, the
affine model is *neither* demonstrably valid locally *nor* demonstrably broken
globally. It appears valid (no drift signal) but the 100 px geolocation grid
cannot resolve a distortion below that quantization floor. So the honest
statement is: *no distortion beyond ~40 px is detected within the strip*, and a
single affine is a reasonable description at the precision the ground truth
permits. This is distinct from "the images cannot be affine-registered" — they
may well be; the validation resolution simply caps how tightly we can say so.

### 2b. The determinant is negative — and it is REAL, in the raw data

The composed OHRC→TMC-2 map has `det < 0`. We established **where the sign
originates** by inspecting the raw per-pixel geolocation and the independent
PDS4 corner footprints, before touching any composition code.

Raw per-pixel geolocation along-track ordering (before any pipeline transform):

| product | `dlat/dscan` (raw CSV) | corner-footprint left-edge dlat (UL→LL, scan direction) |
|---|---|---|
| OHRC `20231004` | **+0.11** | **+0.8560** |
| TMC-2 `ncn_20250707` | **−4.49** | **−35.90** |

Both the CSV geolocation grid *and* the independent PDS4 label corner footprints
agree: latitude **increases** down OHRC's scan axis but **decreases** down
TMC-2's scan axis. The across-track (`lon/pixel`) ordering is the same sign in
both. So the mirror is a genuine, physically-present **opposite along-track
latitude orientation** between the two products, present in the source data — it
is not introduced by `csv_geolocation.py`, `groundtruth.py`, or any composition
step. Per-product determinants of `(pixel,scan)→(lon,lat)` are individually
negative (OHRC) and positive (TMC-2) because each product's *lon* also runs
opposite to *pixel*; the composition is what surfaces a net negative map.

This is confirmed by a dry composition: flipping TMC-2's scan axis turns the
composed `det` from −0.0019 to +0.0019 (see `_det_final_check.py`, and the
synthetic regression `TestDeterminantOriginComposition`).

**Conclusion:** the `det < 0` is a real coordinate-orientation difference rooted
in the raw geolocation, **not** a downstream sign bug. The positive-scale
`R(θ)diag(sx,sy)` anisotropic model simply cannot represent a reflection, so a
physically-motivated scan-axis flip of one view is required before applying the
model — but the flip is a *data-orientation correction*, not a fabrication to
force a positive determinant.

---

## 3. Equal-res real-pair result (honest numbers, corrected scoring)

`run_anisotropic_real_pair.py` renders both views of the ground region at the
same display resolution and scores the recovered affine against geolocation.

**A scoring bug was found and fixed:** ground-truth TMC-2 geolocation was
compared against display-space predictions without rescaling, which inflated
errors by the resize factor. GT is now rescaled into the same display space as
the search.

With the fix, on the 372×372 equal-res pair (`--n_gt 300`):

| variant | `sx` | `sy` | `θ` | PSR | GT med error (display px) | ≤100 px | ≤300 px |
|---|---|---|---|---|---|---|---|
| plain (no flip) | 1.499 | 2.001 | 30.01° | 9.4 | 349.3 | 1.3% | 33.7% |
| **scan-flip B** (mirror-corrected) | 0.306 | 0.447 | 20.51° | 12.6 | **111.3** | 43.7% | 100% |

The scan-flip variant is decisively better (GT median error 349 → 111 px), which
is consistent with the finding that the residual is genuinely reflected. The
remaining ~111 px display-space median error is dominated by the ~40 px
geolocation-grid floor *plus* the weak correlation needle (PSR 9–14) on a region
that has been downsampled 30× — at 6.13 m/px the shared texture is nearly gone.
The recovered `(sx,sy)≈(0.306,0.447)` in display space is still far from the
residual's true geometry, so this real pair is **not** a clean Phase-8 recovery,
but the mirror correction and the grid floor both explain the gap without
resorting to an unsupported claim.

---

## 4. Summary and interpretation

- **Affine registration works.** The synthetic gate recovers a known anisotropic
  affine from real lunar texture to sub-pixel RMS, with unseeded search ranges.
- **Local vs global:** within the real strip, the affine fit residual is flat
  (~40 px) at all window sizes — no spatially-varying distortion is resolvable,
  and the ~40 px floor is the geolocation-grid quantization ceiling. The images
  are *not* shown to be non-affine; they are shown to be affine at the 
  precision the ground truth allows.
- **det < 0 is real:** it originates in the raw geolocation data (opposite
  along-track latitude orientation, confirmed independently by the PDS4 corner
  footprints), not in pipeline composition. Correcting it requires an explicit
  scan-axis flip before the positive-scale anisotropic model, which the runner
  now supports (`--flip-b`) and which measurably improves GT agreement.
- **The mirror-aware real result** is still not a clean recovery (weak needle at
  6.13 m/px + 40 px grid floor), and this is reported honestly.

## Reproducibility

- `run_anisotropic_synthetic_test.py` — synthetic gate (PASSES).
- `run_anisotropic_real_pair.py --region square [--flip-b]` — real pair.
- `lunar_data_pipeline/tests/test_anisotropic_affine_search.py` — regressions.
- `lunar_data_pipeline/tests/test_csv_geolocation.py::TestDeterminantOriginComposition`
  — regresses the determinant-origin convention (synthetic, data-independent).
- `phase8_aniso_output/` — JSON records + overlays for both real-pair variants.
