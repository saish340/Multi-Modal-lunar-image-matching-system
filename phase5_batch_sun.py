"""Phase 5 - batch verify real sun angles for OHRC/TMC labels in phase5_labels/.

Scans every *_d_img_*.xml under the labels dir, parses weather/sun angles, and
prints them grouped by instrument. Flags any OHRC product with sun elevation
> MIN (candidate to pursue) and any TMC product with elevation > MIN.

Usage:
    python phase5_batch_sun.py [--labels-dir phase5_labels]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lunar_data_pipeline.pds4_parser import parse_label

MIN_ELEV = 20.0
TOO_HIGH = 75.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels-dir", default="phase5_labels")
    args = p.parse_args(argv)
    labels_dir = Path(args.labels_dir)

    ohrc, tmc = [], []
    for xml in sorted(labels_dir.rglob("*_d_img_*.xml")):
        prod = parse_label(xml)
        if prod is None:
            print(f"[unparsed] {xml}")
            continue
        entry = {
            "id": prod.product_id,
            "elev": prod.sun_elevation_deg,
            "az": prod.sun_azimuth_deg,
            "orbit": getattr(prod, "imaging_orbit_number", None),
        }
        if prod.instrument.upper() == "OHRC":
            ohrc.append(entry)
        else:
            tmc.append(entry)

    print(f"\n--- OHRC labels ({len(ohrc)}) ---")
    seen_oho = set()
    for e in sorted(ohrc, key=lambda x: -(x["elev"] or -999)):
        ok = "HIGH-SUN" if (e["elev"] or 0) > MIN_ELEV else ""
        if e["elev"] and e["elev"] > TOO_HIGH:
            ok = "TOO-HIGH(??)"
        print(f"  {e['id']}  elev={e['elev']}  az={e['az']}  {ok}")
        seen_oho.add(e["id"])
    if not ohrc:
        print("  (none)")

    print(f"\n--- TMC labels ({len(tmc)}) ---")
    for e in sorted(tmc, key=lambda x: -(x["elev"] or -999)):
        ok = "HIGH-SUN" if (e["elev"] or 0) > MIN_ELEV else ""
        print(f"  {e['id']}  elev={e['elev']}  az={e['az']}  {ok}")
    if not tmc:
        print("  (none)")

    high_ohrc = [e for e in ohrc if (e["elev"] or 0) > MIN_ELEV]
    high_tmc = [e for e in tmc if (e["elev"] or 0) > MIN_ELEV]
    print(f"\nHigh-sun OHRC (> {MIN_ELEV}): {len(high_ohrc)}")
    for e in high_ohrc:
        print(f"   -> {e['id']} elev={e['elev']} az={e['az']}")
    if not high_ohrc:
        print("   NONE — suggests OHRC imagery in this catalog may be uniformly low-sun (pivot point).")
    print(f"High-sun TMC  (> {MIN_ELEV}): {len(high_tmc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
