# Phase 9 Findings: Generalization of Anisotropic-Affine Registration

This document reports whether the Phase 8 method (mirror-corrected
anisotropic-affine phase-correlation registration) generalizes across the
downloaded OHRC × TMC-2 pairs, and what pattern explains any variation.

## 1. Pair inventory: only two overlapping OHRC × TMC-2 pairs exist

Before any search, we enumerated every OHRC × TMC-2 combination across the
downloaded products (workspace + `Downloads`) and measured geographic overlap
using the per-pixel geolocation CSVs (more reliable than corner polygons for
pushbroom strips).

| OHRC | lat range | TMC-2 | lat range | overlap |
|---|---|---|---|---|
| ohr_20231004 | −13.8…−12.9° | tmc_ncn_20250707 | −34.9…+1.0° | **yes** (Phase 8) |
| ohr_20250516 | +60.0…+60.9° | tmc_nca_20200607 | +31.1…+63.2° | **yes** (60°N) |
| ohr_20260103a/b | −85.4…−84.5° | any TMC-2 | — | **no** (all zero) |
| any OHRC | — | tmc_ncn_20260813a/b | −27.9…−3.7° | **no** (all zero) |

Every other combination has **zero** grid points in overlap. So the Phase 9
generalization test is limited to **one genuinely new pair** (60°N) in
addition to the Phase 8 baseline. The south-pole OHRC products (20260103) and
the 20260813 TMC strips are regionally disjoint from everything else in the
dataset, so no third usable pair exists without downloading new data.

## 2. Independent mirror check per pair (two-source method)

We checked each pair independently with the exact Phase 8 method: raw
per-pixel geolocation CSV (`dlat/dscan` along-track sign) **and** independent
PDS4 corner-footprint orientation. We also read the PDS4 tag
`isda:orbit_limb_direction` for each product.

| Pair | OHRC orbit | TMC-2 orbit | CSV dlat/dscan (O vs T) | Corner left-edge (O vs T) | Mirror? |
|---|---|---|---|---|---|
| Phase 8 (20231004 × ncn_20250707) | **Ascending** | Descending | +0.0008 vs −0.0161 (opposite) | +0.856 vs −35.90 (opposite) | **YES** |
| 60°N (20250516 × nca_20200607) | Descending | Descending | −0.0009 vs −0.0163 (same) | −0.830 vs −32.14 (same) | **NO** |

**The mirror is NOT a fixed instrument property.** It is present in Phase 8's
pair but NOT in the 60°N pair. In both pairs the CSV and the independent corner
footprint agree, so this is data, not a pipeline artifact.

**Orbit-direction correlation.** The mirror appears exactly when the two
instruments were imaging with **opposite `orbit_limb_direction`**:

- Phase 8: OHRC **Ascending** vs TMC **Descending** → opposite along-track
  latitude ordering → mirror.
- 60°N: OHRC **Descending** vs TMC **Descending** → same along-track latitude
  ordering → no mirror.

This is physically sensible: an ascending vs descending pass crosses lines of
latitude in opposite directions along the scan axis, so when one member
ascends and the other descends over the same ground, their along-track
latitude orientation is reversed. The across-track (`lon/pixel`) ordering was
the same sign in *both* pairs, so the mirror (when present) is purely
along-track. **The correct mirror setting therefore depends on the pair's
orbit combination, not on the instruments themselves** — a per-pair
determination (via `run_p9_mirror_check.py`) is required and sufficient.

## 3. Search results

Equal-res display pairs, geolocation used only for scoring (display-space
scoring fix from Phase 8). 300 GT points per run.

| Pair | mirror setting used | sx | sy | θ (deg) | PSR | GT median | ≤100 px | ≤300 px |
|---|---|---|---|---|---|---|---|---|
| Phase 8 (20231004 × 20250707) | **flip-B** (mirrored) | 0.306 | 0.447 | 20.5 | 12.6 | 111.3 | 43.7% | 100% |
| 60°N (20250516 × 20200607) | **no flip** (not mirrored) | 0.750 | 1.001 | −20.0 | 11.6 | 170.7 | 16.3% | 98.0% |

The 60°N pair was run **without** a flip because the independent mirror check
(section 2) showed it is not mirrored; applying the Phase 8 flip verbatim
would be wrong for this pair.

**Comparison with the geolocation-derived affine (accuracy ceiling).** For the
60°N equal-res pair, the geolocation-derived affine itself scores median
**26.3 px / 100% within 50 px** (residual at the ~26 px grid floor). The
search recovered the dominant scale (sx=0.750 vs geo's 0.748) and the
rotation-180-equivalent (−20° ≡ geo's −172°) but missed the anisotropic
second scale (sy=1.00 vs geo's ~0.71) and the translation, landing at
170 px — well above the achievable 26 px ceiling. So on this pair the search
found a plausible-but-imperfect anisotropic needle rather than a clean one.

## 4. Synthesis: does the method generalize?

**Honest verdict: partially. The method generalizes in mechanics but not yet
in accuracy, and the mirror handling is orbit-dependent rather than fixed.**

What generalizes cleanly:
- **The mirror determination generalizes.** The two-source check correctly
  identified mirror vs no-mirror on both independently, and — critically —
  showed the mirror is governed by the pair's orbit-direction combination
  (opposite Ascending/Descending → mirror; same → none). This answers the
  prompt's concern: the flip is **not** a fixed instrument property; it is a
  per-pair, orbit-dependent datum that must be re-established each time
  (`run_p9_mirror_check.py`), and it does correlate with ascending/descending
  orbit metadata.
- The anisotropic-affine *search* runs on both pairs and lands in the
  "coarse ~100–170 px, mostly within 300 px" regime on both.

Where it does NOT yet generalize:
- Accuracy on the 60°N pair (170 px median) is worse than Phase 8 (111 px)
  and far above the ~26 px geolocation-achievable ceiling, so neither pair
  reaches ground-truth-limited accuracy. The method is not yet a *precise*
  registration on either real pair; it reliably finds the dominant
  scale/rotation axis but not the full anisotropic + translation needle.
- Only two overlapping pairs exist in the dataset, so the sample is small and
  cannot distinguish "pair-specific fragility" from "needs refinement" with
  confidence. The fact that we cannot test a third overlapping pair (no data)
  is itself a limitation.

**Bottom line for whether Phase 8 is ready to be the headline result:**
The mirror-handling insight (orbit-dependent, per-pair) is robust and
generalizes. But as a *precision registration* the method is not yet
production-ready — on both real pairs it recovers only a coarse
correct-ish transform (100–170 px), far above the ~26–40 px geolocation
floor, and anisotropy/translation are imprecise. It is a strong partial
registration and an accurate synthetic-gate method, not yet a
ground-truth-limited real registration. More work (a finer search, or a third
overlapping pair) is needed before adopting it as the headline result.

## Reproducibility

- `run_p9_mirror_check.py` — independent mirror check (CSV + corners) and orbit
  direction, per pair.
- `run_anisotropic_pair.py` — generalized equal-res anisotropic search +
  display-space geolocation scoring, takes labels/region/`--flip-b`.
- `phase9_output/result_pair2_60n_nof.json`, `phase9_output/ov_pair2_60n_nof.png`
  — 60°N run artifacts.
- `phase8_findings.md` / `phase8_aniso_output/` — Phase 8 pair artifacts.
