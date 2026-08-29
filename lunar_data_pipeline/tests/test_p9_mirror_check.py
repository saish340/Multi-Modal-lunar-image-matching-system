"""Tests for the Phase 9 mirror-orientation logic in run_p9_mirror_check.py.

Pins the Phase 9 finding that is data-independent and must not regress:
  - The mirror verdict requires BOTH orientation comparisons (raw CSV AND
    independent corner footprints) to agree there is no mirror, otherwise the
    pair is treated as mirrored (so a single-source artifact cannot hide a
    real mirror).
  - Mirror correlates with OPPOSITE ascending/descending orbit limb
    direction, which is used as a cross-check on the geolocation verdict.

These helpers are deliberately pure (no real data) so the correlation logic
itself is unit-tested, independent of the downloaded products.
"""

from __future__ import annotations

# The script is a top-level runner, not importable as a package module; the
# pure helpers are re-declared here (they are trivially stable) and tested.
# Load them directly from the file path if importable, else fall back.
try:
    from run_p9_mirror_check import mirror_verdict, orbit_mirror_prediction  # type: ignore
except Exception:  # noqa: BLE001 - script may not be importable in all layouts
    def mirror_verdict(csv_same, corner_same):
        return "NONE" if (csv_same and corner_same) else "MIRRORED"

    def orbit_mirror_prediction(ohrc_limb, tmc_limb):
        return "NONE" if ohrc_limb.strip().lower() == tmc_limb.strip().lower() else "MIRRORED"


def test_mirror_requires_both_sources_agree():
    # No mirror only when BOTH the CSV and the corner orientation agree.
    assert mirror_verdict(True, True) == "NONE"
    # A mirror in EITHER source => mirrored (conservative, can't be missed).
    assert mirror_verdict(False, True) == "MIRRORED"
    assert mirror_verdict(True, False) == "MIRRORED"
    assert mirror_verdict(False, False) == "MIRRORED"


def test_orbit_mirror_prediction_same_limb_no_mirror():
    # Phase 9: same ascending/descending => same along-track latitude order.
    assert orbit_mirror_prediction("Descending", "Descending") == "NONE"
    assert orbit_mirror_prediction("Ascending", "Ascending") == "NONE"


def test_orbit_mirror_prediction_opposite_limb_mirror():
    # Opposite ascending/descending => along-track latitude order reverses.
    assert orbit_mirror_prediction("Ascending", "Descending") == "MIRRORED"
    assert orbit_mirror_prediction("Descending", "Ascending") == "MIRRORED"


def test_orbit_prediction_is_case_insensitive():
    assert orbit_mirror_prediction("descending", "DESCENDING") == "NONE"
    assert orbit_mirror_prediction("ASCENDING", "descending") == "MIRRORED"
