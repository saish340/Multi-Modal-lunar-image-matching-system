# Phase 3 Findings: OHRC/TMC-2 illumination mismatch, and the small-N crater detection result

This document records the conclusions of Phase 3. It is factual project
rationale: where a result is weak, it says so; where a limitation is
structural, it quantifies it. Nothing here is overstated in either direction.

---

## 1. OHRC systematically acquires at low sun elevation

All three downloaded OHRC products have very low sun elevation; their TMC-2
partners are at high elevation:

| Product (instrument) | Sun elevation (deg) |
| --- | ---: |
| `ch2_ohr_ncp_20231004T0406038822` (OHRC) | 9.5 |
| `ch2_ohr_ncp_20260103T0609041371` (OHRC) | 6.7 |
| `ch2_ohr_ncp_20260103T1005176450` (OHRC) | 5.5 |
| `ch2_tmc_ncn_20250707T1853051045` (TMC-2) | 69.4 |
| `ch2_tmc_ncn_20260813T0627378557` (TMC-2) | 50.9 |
| `ch2_tmc_ncn_20260813T1023298745` (TMC-2) | 50.8 |

**Interpretation.** OHRC is a high-resolution (~0.2–0.25 m/px) nadir camera
intended to reveal landing-site terrain texture. Low solar incidence is
actually desirable for that use case because grazing light carves shadows
that make metre-scale relief visible. The consistent ~5–10° elevation across
three acquisitions spanning 2023–2026, and across two different orbital
regimes, is consistent with OHRC operations deliberately favouring low sun,
not an accidental sampling of our three downloads.

## 2. Consequence: 0 / 9 sun-matched OHRC×TMC-2 pairs

Using the sun-match criterion (both members' sun elevation > 20°, azimuth
difference < 30°), a free table of every downloadable OHRC×TMC-2 combination
(sun angle parsed from each product's own PDS4 label) gives:

- **0 of 9 combinations pass.** Every OHRC member fails the elevation test;
  most also fail the azimuth test by a large margin (49°–157° separation).

This makes illumination mismatch the *expected* condition for a general
OHRC/TMC-2 pairing, not an edge case. It is driven by the two instruments'
complementary design purposes (landing-terrain shadows vs. broad-region
nadir imaging), so it is unlikely to disappear by simply downloading more
products.

### Blocker this created

Fair crater detection needs like illumination across the pair, otherwise
shadows in one image masquerade as structure the other does not show. With
no matched pair available, we cannot run a *well-illuminated, well-matched*
cross-instrument comparison from the current data.

## 3. Small-N real-ground-truth crater detection result

We ran the classical detector (`crater_detector.py`) on the original
confirmed pair (OHRC `20231004` × TMC-2 `ncn_20250707`), scored against the
**real** ground truth from the HuggingFace Robbins subset
(`data/lunar_craters.parquet`, ~384k craters ≥ 1 km, ~30% complete vs. the
full Robbins catalog). Three real craters fall in the overlap footprint
(diameters 2.17, 9.54, 26.66 km).

**This is a small-sample result.** It is not a statistically meaningful
estimate of detector quality; it is a per-crater inspection of 3 craters
across 2 cameras. Read it as such.

### TMC-2 — full detection run

| Crater | diameter | GT centre (crop px) | radius (px) | result |
| --- | ---: | ---: | ---: | --- |
| 11-0-05812 | 2.17 km | (383.8, 4281.0) | 177 | **FOUND** — detection at (311.0, 4361.0), centre distance **108 px**, IoU 0.245 |
| 11-0-05828 | 9.54 km | (724.9, 4019.5) | 778 | MISSED — radius exceeds detector's 500-px cap |
| 11-3-00076 | 26.66 km | (608.8, 2318.0) | 2175 | MISSED — radius exceeds detector's 500-px cap |

Aggregate for TMC-2: TP=1, FP=2052, FN=2 → **P ≈ 0.001, R = 0.33, F1 ≈ 0.001**.

Honest reading:
- The one true positive is weak but real: a detection centred 108 px (≈660 m)
  from the 2.17 km crater's GT centre with IoU 0.245. It is not a clean hit.
- The two misses are **structural**: at 6.13 m/px these craters have radii of
  778 and 2175 px, both beyond the detector's configured max radius (500 px).
  The detector was not set up to find craters this large. So this run does not
  tell us whether the detector could find a 9–27 km crater given an able
  configuration.
- Precision is essentially nil: the simple HoughCircles baseline returns
  thousands of false positives (2053 detections for 3 GT). This is expected
  of the baseline and is the reason it is a *baseline*.

### OHRC — geometric feasibility analysis

A full HoughCircles run on the OHRC crop is not practically runnable here,
and it is unnecessary: the crater sizes make detection structurally impossible
at OHRC's 0.2 m/px resolution.

| Crater | diameter | radius (px @ 0.2 m/px) | full-image centre (px) | feasibility |
| --- | ---: | ---: | ---: | --- |
| 11-0-05812 | 2.17 km | 5,425 | (819, 26022) | radius just above detector cap (5000 px) |
| 11-0-05828 | 9.54 km | 23,850 | (9951, 31331) | radius ≫ frame width (12000 px) |
| 11-3-00076 | 26.66 km | 66,650 | (4823, 63493) | crater diameter (53 km) exceeds along-track frame (20 km) |

**All three are undetectable in OHRC**, not because of illumination but
because the craters are too large for the detector's radius range, and the
two largest are larger than the OHRC frame itself. This is a resolution-vs-
crater-size mismatch.

### Detector differences between OHRC and TMC-2

For these same 3 craters the classical detector is only operable on TMC-2;
OHRC cannot represent them at all. That contrast is itself the useful signal:
the two cameras differ not only in illumination (Section 1) but by ~30× in
ground resolution, so a single crater-size threshold cannot serve both. Any
cross-instrument method must be scale-agnostic rather than fixed-radius.

## 4. What this motivates

Taken together, these findings do not prove the classical detector is good
(N=3, weak hit) and do not prove it is bad (two of three misses were
structural, not algorithmic). They show that a *fixed-radius HoughCircles*
pipeline built around one instrument's resolution is the wrong tool for a
multi-instrument, variable-scale, adverse-illumination setting.

This is exactly the problem a **crater-neighborhood-graph** approach is meant
to solve, and it now has an evidence base:

1. Illumination mismatch is the norm, so a matching method must not depend on
   consistent shadowing — a graph of relative positions/sizes is illumination-
   robust where raw intensity is not.
2. Resolution differs ~30× between cameras, so the representation must be
   scale-invariant (edges/nodes in a normalised graph) rather than tied to a
   pixel-radius range.
3. The same small ground truth is shared by both cameras (3 craters here) —
   a graph matched by neighbourhood topology can exploit craters that a
   fixed-radius detector would treat as out of range.

The next phase should therefore pivot from per-crater Hough matching to
construction and matching of crater neighbourhood graphs, motivated above all
by the fact that consistent illumination between OHRC and TMC-2 is not
available in practice.
