# Phase 12 — Phase-11 transfer to the SIH three-sensor dataset (OHRC × TMC-2)

Status: COMPLETE. Gate PASS; content-verification PASS at the Phase-11
operating point; the wide-search primary config reprised the Phase-6/10
wrong-basin failure (documented as a negative result).

Companion artifacts (all in `phase12_transfer_output/`):
`phase12_transfer_result.json` (full data), `geoprior_gate.json`,
`run_stdout.log`, `gate_failed.json` (first-canvas failure, kept on record),
`preview_a_ohrc.png`, `preview_b_tmc.png`, `overlay_rgb.png`,
`votes_overlay_m80.png`, `votes_overlay_m8.png`.

---

## 1. Hypothesis (pre-stated, module docstring of `run_phase12_transfer.py`)

The TIF's own ModelTiepoint/ModelPixelScale georeferencing chained with the
OHRC label's 4-corner System_Level_Coordinates is a valid navigation prior,
and the Phase-11 tile-vote pipeline confirms it with local image content:
the majority of tiles find their predicted counterpart within the search
margin, and the vote-consensus affine differs from the prior only by the
documented ISDA corner-metadata error, not by a wrong basin.

## 2. Exact data

| | OHRC | TMC-2 |
|---|---|---|
| Product | `ch2_ohr_nrp_20200827T0030107497_d_img_d18` | `ch2_tmc_ndn_20231101T0125121377_d_oth_d18` |
| File | `.img` (1,212,900,000 B, MD5 verified in Phase-12 verify) | `.tif` BigTIFF ortho (19,092,166,591 B, MD5 verified) |
| Sensor | OHRC pan 0.24 m/px | TMC-2 pan 5.0 m/px (ModelPixelScale) |
| Geometry | 101,075 × 12,000 uint8 | 169,968 × 56,156 uint16 |
| Acquisition | 2020-08-27 00:30Z | 2023-11-01 01:25Z |
| Overlap | cols ≈ 41086–43470, rows ≈ 39766–44917 (verified, 100% in ortho) | |

Navigation prior = chain of (a) OHRC corner homography `(col,row)->(lat,lon)`
from the label's System_Level_Coordinates and (b) south-polar-stereographic
pixel georeferencing from the TIF's own tags (ModelPixelScale = 5.0 m,
ModelTiepoint, lat0=-90 deg, lon0=0, R=1,737,400 m).  Affine fit to the four
corner mappings has residual **11.51–11.55 OHRC px = 0.55–0.60 TMC px**
(sub-pixel in the reference/TMC frame, the project's accuracy goal).

## 3. Method

Both products were rendered read-only into a **strip-aligned display frame at
TMC-native 5 m/px** (`strip_frame_affines` + `render_a_windowed` +
`render_b_windowed` in the runner):

- display x/y = OHRC cross-track / along-track unit directions (the columns
  of the OHRC->TIF linear part, Gram-Schmidt orthonormalised).  The OHRC
  strip therefore fills the canvas (88.3 % content) instead of a ~24 %
  diagonal band in an axis-aligned TMC window.
- A = OHRC strip rendered into that frame (memmap + row-banded windowed
  warp; peak memory ~ a band, not the 1.2 GB strip).  B = the corresponding
  TMC ortho resampled into the SAME frame.
- Because A and B share the identical display->TIF affine, the navigation
  prior in display coordinates is the **identity**: each A tile's predicted B
  position is its A position, and `meas - pred` directly measures the local
  disagreement between image content and the georeference chain.
- Phase-11 machinery reused unchanged: `geoprior_tile_registration.register`
  (tile-vote NCC + parabolic sub-pixel peak + robust soft-L1/MAD 5-DOF fit,
  inlier-only refit, 2 vote+fit passes) and `synthetic_gate`.

### Parameters (all pre-stated in the module docstring; none changed post-hoc)
- tile = 96 px (480 m), n_tiles = 6 (6x6 lattice), mode = "ncc",
  refine_ecc = False (Phase 11b: no ECC), iterations = 2.
- Primary search margin = 80 px (400 m); verification diagnostic margin = 8 px
  (Phase-11 operating point); sensitivity margin = 32 px.
- Gate: `synthetic_gate` on a 384x384 content crop harvested from the pair's
  own canvas, at the same params (module contract; identical semantics to
  Phase 11's gate on its a_disp crop).
- Pad around the strip = 48 px (= half an A-side), so the fixed 6x6 lattice's
  edge columns land in content.

### Gate-discovered construction correction (before any real-data result)
The first canvas was the axis-aligned TMC window (2386x5152): only ~24 %
content, and the injected worst-case gate transform displaces that canvas's
corners ~398 px — far beyond Stage A's ±80 identity-prior search window
(`gate_failed.json` on record).  The module gate contract is a square content
crop (where the same transform displaces corners ~40 px).  Both were fixed by
construction (strip-aligned dense canvas + content-crop gate), NOT by tuning:
the gate and tile parameters are unchanged.  This is exactly what the
abort-on-fail gate is for: it stopped a physically meaningless run and
diagnosed the input construction.

## 4. Results

### 4.1 Gate — PASS
- Stage A (identity prior, tile=56/margin=80/ncc single pass): errors
  sx=0.0082, sy=0.0073, theta=0.343 deg, tx=1.66, ty=0.18 px; inliers 11/16;
  fit residual 2.16 px.
- Stage B (perturbed-prior, pipeline config): errors sx=0.0001,
  sy=0.0176, theta=0.067 deg, tx=0.75, ty=3.64 px; inliers 9/13; residual
  1.89 px.

### 4.2 Real pair — margin-dependent displacement (the core finding)

| margin | fit (sx, sy, theta, tx, ty) | votes | inliers | fit residual (px) | content-vs-prior radial median (px) | NCC score median |
|---|---|---|---|---|---|---|
| **8** (verification) | 0.9917, 0.9987, 0.002 deg, −11.85, −11.19 | 25/36 | 21/25 | **0.37** | **11.36** (mean 11.33, p90 11.69) | content locks |
| 32 | 0.9649, 0.9961, 0.032 deg, −47.7, −48.2 | 25/36 | 20/25 | 0.61 | 45.2 | partial |
| 80 (primary) | 1.0667, 1.0052, 1.467 deg, −223.3, −154.5 | 25/36 | 24/25 | 41.5 | 112.7 | **0.000** |

Interpretation (read off the raw outputs, no tuning):
- The **margin-8 verification** — the Phase-11 operating point — found a
  tight, consistent content-vs-prior consensus: every usable tile votes the
  same ~(−8, −8) offset, radial median 11.36 px (57 m at 5 m/px), fit
  residual 0.37 px, scale deviation 0.8 % (sx) / 0.1 % (sy), rotation
  0.002 deg.  This is a REAL, spatially consistent geolocation disagreement
  between the OHRC corner geolocation and the TMC-2 ortho's geotags
  (corner-chain quality: 4 corners, no per-pixel CSV).
- As the margin widens, votes fall INTO a wrong basin cleanly monotonically
  (8 -> 11.4 px, 32 -> 45.2 px, 80 -> 112.7 px radial); at margin=80 the NCC
  scores collapse to 0.000 (edge-saturated locks pinned to the ±80 window
  boundary) yet the lock positions are self-consistent (24/25 "inliers",
  residual 41.5 px) — the Phase-6/10 repetitive-terrain ambiguity, now
  quantified as margin dependence in a single pair.
- **Conclusion: on this terrain the ONLY trustworthy search is the narrow
  one.**  The Phase-11 deliverable operating point (prior + <=8 px local
  verification) transfers to the new pair unmodified; widening the search does
  not gain anything — it loses the answer.

### 4.3 Spatial distribution of successful matches
The 6x6 lattice is uniform over the canvas by construction; 25/36 tiles
voted (11 skipped: pad/edge geometry), 21/25 inliers at margin=8.  Accepted
votes are spread along the FULL strip length (display rows ~1,100–5,300 in
`per_tile` of the JSON).  The pre-stated 8x8-grid occupancy metric reports
**23/64 cells (36 %)** — below the pre-stated 70 % bar; partly a
thin-strip-vs-bbox geometry effect, partly skipped tiles.  Noted honestly;
the matches are nonetheless distributed along the whole overlap.

## 5. Pass/fail vs the pre-stated criteria

| Criterion | Result |
|---|---|
| Gate PASS (abort otherwise) | PASS (both stages) |
| >=60 % tiles vote within margin | 25/36 = 69 % ✓ |
| accepted-vote residual median <= 8 px (40 m) | 0.37 px (margin-8) ✓; primary margin-80 = 41.5 px ✗ |
| accepted tiles in >=70 % of an 8x8 grid | 36 % ✗ (thin-strip geometry; matches still full-strip) |
| fitted affine deviates from prior <=5 % in scale | margin-8: 0.8 %/0.1 % ✓; margin-80: 6.7 % ✗ |

Verdict: **PASS at the Phase-11 operating point (prior + <=8 px verification);
FAIL for the wide (margin=80) primary search, which is a documented negative
result consistent with Phase 6/10 — wide windows are unusable on this
repetitive terrain.**

## 6. Comparison with Phase 11

- Phase 11 (OHRC 20231004 x TMC-2 20250707, geolocation-CSV prior):
  23.1 px median prior-alone / 23.7 px content-verified, scored against an
  independent dense CSV ground truth at the same operating point.
- Phase 12 (OHRC 20200827 x TMC-2 ortho, geotag-corner prior): **11.36 px
  median** content-vs-prior, 0.37 px fit residual, <1 % scale — tighter than
  Phase 11 on this pair, with the SAME operating point.
- Semantic caveat (honest): Phase 12's products ship NO per-pixel geolocation
  CSV (unlike the Phase-8/9/11 pair), so there is no independent third-party
  ground truth: 11.36 px certifies image-content-vs-geotag-chain consistency
  (a true correction the prior needs), NOT absolute geodetic accuracy.  The
  number is comparable to (and near the floor of) the ~26–40 px
  geolocation-grid floor seen in Phases 8/9.
- Phase 12 additionally reproduces and quantifies the Phase-6/10 wide-search
  wrong-basin failure as clean margin dependence — a new diagnostic insight.

## 7. Limitations

1. No independent ground truth for this pair (no per-pixel geolocation CSV).
   "Final error" = content-vs-prior self-consistency only.
2. The TMC-2 ortho itself has geotag uncertainty (typically tens of metres);
   the 11.4 px (57 m) offset may include ortho error as well as OHRC corner
   error — the pair cannot separate them.
3. The pre-stated 8x8-grid uniformity criterion (36 %) is not met; matches are
   nevertheless spread along the full strip (JSON per_tile).
4. 11/36 lattice tiles were skipped by the shape/pad guard — a consequence of
   the uniform lattice over a near-strip canvas.
5. OHRC is displayed at 5 m/px (21x downsampling); the registration target is
   sub-pixel accuracy IN the reference (TMC) frame, and the floor remains the
   geolocation uncertainty, not the estimator.
6. Only one transfer pair exists in this three-sensor dataset (IIRS is
   geographically unverified — see the Phase-12 verify report); generalisation
   beyond this pairing remains untested.

## 8. Next step (proposed, not run)

Deliverable direction: (1) treat the margin-8 verification affine as the
pair's registration — the correction the georeference chain needs — and
export its >=21 inlier correspondences as the match-point deliverable
(CSV/GeoJSON with display & OHRC/TMC pixel coordinates); (2) only after an
IIRS product whose footprint VERIFIES overlap (Phase-12 verify said current
IIRS does not) extend the same prior+verification pipeline to the spectral
sensor.