"""Phase 5 - verify real sun angles for the selected candidate pair.

Reads the two PRADAN PDS4 XML labels the user downloads into ``phase5_labels/``
and reports whether they satisfy the matched-illumination criterion
(both members sun elevation > 20 deg, azimuth diff < 30 deg).

Usage:
    python phase5_verify_sun.py [--labels-dir phase5_labels]

Expected files (drop the XML labels here with these names):
    ch2_ohr_ncp_20220320T2133025866_d_img_d18.xml
    ch2_tmc_nca_20230711T2159032567_d_img_d32.xml
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from lunar_data_pipeline.pds4_parser import parse_label

OHRC_ID = "ch2_ohr_ncp_20220320T2133025866_d_img_d18"
TMC_ID = "ch2_tmc_nca_20230711T2159032567_d_img_d32"

MIN_ELEV = 20.0
MAX_AZ_DIFF = 30.0


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels-dir", default="phase5_labels")
    args = p.parse_args(argv)
    labels_dir = Path(args.labels_dir)

    found = {}
    for target in (OHRC_ID, TMC_ID):
        label = labels_dir / f"{target}.xml"
        if not label.exists():
            print(f"[missing] {target}.xml  (drop it into {labels_dir})")
            continue
        prod = parse_label(label)
        if prod is None:
            print(f"[error] could not parse {label}")
            continue
        found[target] = prod
        print(
            f"[parsed] {target}: sun_elevation={prod.sun_elevation_deg} "
            f"sun_azimuth={prod.sun_azimuth_deg}"
        )

    if not (OHRC_ID in found and TMC_ID in found):
        print("\nBoth labels required. Download from PRADAN and retry.")
        return 1

    o, t = found[OHRC_ID], found[TMC_ID]
    if o.sun_elevation_deg is None or t.sun_elevation_deg is None:
        print("\n[error] sun angles missing from one/both labels.")
        return 1

    ev_ohrc, ev_tmc = o.sun_elevation_deg, t.sun_elevation_deg
    az_ohrc, az_tmc = o.sun_azimuth_deg, t.sun_azimuth_deg
    az_diff = angle_diff(az_ohrc, az_tmc)

    print("\n=== Sun match check ===")
    print(f"OHRC sun elevation : {ev_ohrc:.2f} deg  (need > {MIN_ELEV})")
    print(f"TMC  sun elevation : {ev_tmc:.2f} deg  (need > {MIN_ELEV})")
    print(f"Sun azimuth diff   : {az_diff:.2f} deg  (need < {MAX_AZ_DIFF})")

    reasons = []
    if ev_ohrc < MIN_ELEV:
        reasons.append("OHRC elev too low")
    if ev_tmc < MIN_ELEV:
        reasons.append("TMC elev too low")
    if az_diff > MAX_AZ_DIFF:
        reasons.append("azimuth diff too large")

    if not reasons:
        print("\nPASS: matched illumination. Safe to download the full products.")
        return 0
    print(f"\nFAIL: {'; '.join(reasons)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
