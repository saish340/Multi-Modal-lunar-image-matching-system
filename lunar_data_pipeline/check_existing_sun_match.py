"""Check sun-angle match among the already-downloaded OHRC / TMC-2 products.

Free check: no new downloads. Parses the real sun elevation/azimuth out of
every downloaded ``*_d_img_*.xml`` PDS4 label (via :func:`pds4_parser.parse_label`,
which was extended in Phase 3b to read ``isda:sun_elevation`` /
``isda:sun_azimuth``), then compares every OHRC x TMC-2 combination found.

Sun-match criterion (same constants as :mod:`candidate_ranking`): a pair is
"sun-matched" when BOTH members' sun elevations are above 20 deg and their
azimuths are within 30 deg of each other. Only combinations formed from labels
that actually exist on disk are reported (no catalog-wide guessing).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:  # package-style import when used as a module...
    from .candidate_ranking import (
        MAX_SUN_AZIMUTH_DIFF_DEG,
        MIN_SUN_ELEVATION_DEG,
        SunAngles,
        _angle_diff,
    )
    from .pds4_parser import LunarProduct, parse_label
except ImportError:  # ...and flat import when scripts run directly
    from candidate_ranking import (  # type: ignore[no-redef]
        MAX_SUN_AZIMUTH_DIFF_DEG,
        MIN_SUN_ELEVATION_DEG,
        SunAngles,
        _angle_diff,
    )
    from pds4_parser import LunarProduct, parse_label  # type: ignore[no-redef]

logger = logging.getLogger("lunar_data_pipeline.check_sun_match")


@dataclass
class CheckedPair:
    """Result of comparing one downloaded OHRC and one downloaded TMC-2 product."""

    ohrc_id: str
    tmc2_id: str
    ohrc_elevation_deg: float | None
    ohrc_azimuth_deg: float | None
    tmc2_elevation_deg: float | None
    tmc2_azimuth_deg: float | None
    azimuth_diff_deg: float | None = None
    sun_matched: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ohrc_id": self.ohrc_id,
            "tmc2_id": self.tmc2_id,
            "ohrc_sun_elevation_deg": self.ohrc_elevation_deg,
            "ohrc_sun_azimuth_deg": self.ohrc_azimuth_deg,
            "tmc2_sun_elevation_deg": self.tmc2_elevation_deg,
            "tmc2_sun_azimuth_deg": self.tmc2_azimuth_deg,
            "azimuth_diff_deg": self.azimuth_diff_deg,
            "sun_matched": self.sun_matched,
            "reasons": self.reasons,
        }


def discover_downloaded_labels(root: str | Path) -> list[Path]:
    """All ``*_d_img_*.xml`` data labels under *root* (parsable, product labels)."""
    labels: list[Path] = []
    for xml in sorted(Path(root).rglob("*_d_img_*.xml")):
        product = parse_label(xml)
        if product is not None:
            labels.append(xml)
        elif Path(xml).stat().st_size > 0:
            logger.warning("Unparsable or skipped label: %s", xml)
    return labels


def collect_sun_angles(labels: list[Path]) -> dict[str, SunAngles]:
    """Map product_id -> SunAngles for every parsed label with geometry data."""
    out: dict[str, SunAngles] = {}
    for xml in labels:
        product: LunarProduct | None = parse_label(xml)
        if product is None:
            continue
        out[product.product_id] = SunAngles(
            sun_elevation_deg=product.sun_elevation_deg,
            sun_azimuth_deg=product.sun_azimuth_deg,
        )
    return out


def _needs_label(sa: SunAngles | None) -> str:
    if sa is None:
        return "NO-ANGLES"
    if sa.usable:
        return ""
    return "NO-ANGLES"


def check_pair(
    ohrc_id: str,
    tmc2_id: str,
    ohrc: SunAngles,
    tmc2: SunAngles,
    *,
    min_elevation_deg: float = MIN_SUN_ELEVATION_DEG,
    max_azimuth_diff_deg: float = MAX_SUN_AZIMUTH_DIFF_DEG,
) -> CheckedPair:
    """Evaluate one OHRC/TMC-2 combination against the sun-match criterion."""
    reasons: list[str] = []
    az_diff = None
    elev_ok = True
    if not ohrc.usable or not tmc2.usable:
        reason = _needs_label(ohrc) or _needs_label(tmc2)
        reasons.append(f"{reason} (sun angles not all present)")
        elev_ok = False

    if elev_ok:
        az_diff = _angle_diff(ohrc.sun_azimuth_deg, tmc2.sun_azimuth_deg)
        if ohrc.sun_elevation_deg <= min_elevation_deg:
            reasons.append(
                f"OHRC elevation {ohrc.sun_elevation_deg:.1f} < {min_elevation_deg}"
            )
        if tmc2.sun_elevation_deg <= min_elevation_deg:
            reasons.append(
                f"TMC-2 elevation {tmc2.sun_elevation_deg:.1f} < {min_elevation_deg}"
            )
        if az_diff > max_azimuth_diff_deg:
            reasons.append(f"azimuth diff {az_diff:.1f} > {max_azimuth_diff_deg}")

    sun_matched = not reasons
    return CheckedPair(
        ohrc_id=ohrc_id,
        tmc2_id=tmc2_id,
        ohrc_elevation_deg=ohrc.sun_elevation_deg,
        ohrc_azimuth_deg=ohrc.sun_azimuth_deg,
        tmc2_elevation_deg=tmc2.sun_elevation_deg,
        tmc2_azimuth_deg=tmc2.sun_azimuth_deg,
        azimuth_diff_deg=az_diff,
        sun_matched=sun_matched,
        reasons=reasons,
    )


def check_all_pairs(
    angles: dict[str, SunAngles],
    *,
    min_elevation_deg: float = MIN_SUN_ELEVATION_DEG,
    max_azimuth_diff_deg: float = MAX_SUN_AZIMUTH_DIFF_DEG,
) -> tuple[list[CheckedPair], list[str], list[str]]:
    """Compare every OHRC x TMC-2 combination among downloaded labels.

    Returns ``(pairs, ohrc_ids, tmc2_ids)``.
    """
    ohrc_ids = sorted(pid for pid in angles if pid.lower().startswith("ch2_ohr"))
    tmc_ids = sorted(pid for pid in angles if pid.lower().startswith("ch2_tmc"))

    pairs: list[CheckedPair] = []
    for oid in ohrc_ids:
        for tid in tmc_ids:
            pair = check_pair(
                oid,
                tid,
                angles[oid],
                angles[tid],
                min_elevation_deg=min_elevation_deg,
                max_azimuth_diff_deg=max_azimuth_diff_deg,
            )
            pairs.append(pair)
    return pairs, ohrc_ids, tmc_ids


def _render(pairs, ohrc_ids, tmc_ids):
    out = []
    for p in pairs:
        match = "YES" if p.sun_matched else "NO "
        az = p.azimuth_diff_deg if p.azimuth_diff_deg is not None else float("nan")
        out.append(
            f"{p.ohrc_id:55} {p.tmc2_id:55} O={p.ohrc_elevation_deg:6.1f}"
            f" | T={p.tmc2_elevation_deg:6.1f} | az={az:6.1f}"
            f" | {match}"
        )
        if p.reasons:
            out.append("      - " + "; ".join(p.reasons))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Free sun-angle match check across already-downloaded products."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root folder under which downloaded product labels are searched "
        "(default current directory, recursive).",
    )
    parser.add_argument(
        "--min-elevation",
        type=float,
        default=MIN_SUN_ELEVATION_DEG,
        help="Both members must exceed this sun elevation (deg).",
    )
    parser.add_argument(
        "--max-azimuth-diff",
        type=float,
        default=MAX_SUN_AZIMUTH_DIFF_DEG,
        help="Azimuths must be within this many degrees (deg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the full comparison table.",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )

    labels = discover_downloaded_labels(args.root)
    if not labels:
        logger.error("No downloaded *_d_img_*.xml labels found under %s", args.root)
        return 2

    angles = collect_sun_angles(labels)
    pairs, ohrc_ids, tmc_ids = check_all_pairs(
        angles,
        min_elevation_deg=args.min_elevation,
        max_azimuth_diff_deg=args.max_azimuth_diff,
    )

    print(f"Downloaded labels parsed: {len(angles)}  ({len(ohrc_ids)} OHRC, "
          f"{len(tmc_ids)} TMC-2)")
    print(f"Sun-match criterion: both elevations >{args.min_elevation:.1f} deg "
          f"AND azimuth diff <{args.max_azimuth_diff:.1f} deg")
    print()
    print(f"{'OHRC product':55} {'TMC-2 product':55} O-elev | T-elev | az-diff | Match")
    print("-" * 140)
    for line in _render(pairs, ohrc_ids, tmc_ids):
        print(line)
    print("-" * 140)

    matched = [p for p in pairs if p.sun_matched]
    print(f"\nSun-matched pairs found: {len(matched)} of {len(pairs)} checked.")
    if matched:
        for p in matched:
            print(f"  MATCH: {p.ohrc_id} x {p.tmc2_id} "
                  f"(O-elev {p.ohrc_elevation_deg:.1f}, T-elev {p.tmc2_elevation_deg:.1f}, "
                  f"az-diff {p.azimuth_diff_deg:.1f})")

    if args.output:
        payload = {
            "schema_version": "1.0",
            "parsed_labels": len(angles),
            "criterion": {
                "min_sun_elevation_deg": args.min_elevation,
                "max_azimuth_diff_deg": args.max_azimuth_diff,
            },
            "sun_matched_pairs": [p.to_dict() for p in matched],
            "all_combinations": [p.to_dict() for p in pairs],
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote comparison to %s", args.output.resolve())

    return 0


if __name__ == "__main__":
    sys.exit(main())
