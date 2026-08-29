"""Phase 6 - build manifest for the OHRC-OHRC same-scale sanity pair.

Pair: OHRC 20260103T0609041371 (product A) x OHRC 20260103T1005176450 (B),
~88.4% overlap of the smaller, same sensor, same 0.25 m/px scale, same day.
Built with the pipeline's own build_pixel_transform / transform_to_manifest_dict
so the ground-truth format is identical to prior phases.
"""

from __future__ import annotations

import json
import sys

from lunar_data_pipeline.groundtruth import build_pixel_transform, transform_to_manifest_dict
from lunar_data_pipeline.pds4_parser import parse_label

A_LABEL = (
    "D:/Multi-Modal-lunar-image-matching-system/"
    "ch2_ohr_ncp_20260103T0609041371_d_img_d18/"
    "data/calibrated/20260103/ch2_ohr_ncp_20260103T0609041371_d_img_d18.xml"
)
B_LABEL = (
    "D:/Multi-Modal-lunar-image-matching-system/"
    "ch2_ohr_ncp_20260103T1005176450_d_img_d18/"
    "data/calibrated/20260103/ch2_ohr_ncp_20260103T1005176450_d_img_d18.xml"
)
OUT = "pair_manifest_phase6_ohrc_ohrc.json"


def main() -> int:
    a = parse_label(A_LABEL)
    b = parse_label(B_LABEL)
    if a is None or b is None:
        print("Failed to parse one or both labels")
        return 2
    print(f"A OHRC {a.product_id} res={a.pixel_resolution_m} geo_csv_exists={a.geometry_csv_path and a.geometry_csv_path.exists()}")
    print(f"B OHRC {b.product_id} res={b.pixel_resolution_m} geo_csv_exists={b.geometry_csv_path and b.geometry_csv_path.exists()}")

    transform = build_pixel_transform(a, b)
    gt = transform_to_manifest_dict(transform)

    pair = {
        "pair_id": f"{a.product_id}__x__{b.product_id}",
        "ohrc": {
            "product_id": a.product_id,
            "label_path": A_LABEL,
            "image_path": str(a.file_path),
            "geometry_csv_path": str(a.geometry_csv_path),
            "lines": a.lines,
            "samples": a.samples,
        },
        "tmc2": {
            "product_id": b.product_id,
            "label_path": B_LABEL,
            "image_path": str(b.file_path),
            "geometry_csv_path": str(b.geometry_csv_path),
            "lines": b.lines,
            "samples": b.samples,
        },
        "ground_truth": gt,
    }

    manifest = {
        "schema_version": "1.0",
        "data_root": "D:\\Multi-Modal-lunar-image-matching-system",
        "counts": {"OHRC": 2, "TMC2": 0, "IIRS": 0, "pairs": 1},
        "products": {},
        "pairs": [pair],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
