"""Phase 5 - build the corner-homography manifest for the 60N sun-matched pair.

Constructs ground truth with the pipeline's own build_pixel_transform /
transform_to_manifest_dict, then writes a manifest the crater-graph runner can
consume (it will refine the TMC side via CSV when present).
"""

from __future__ import annotations

import json
import sys

from lunar_data_pipeline.groundtruth import build_pixel_transform, transform_to_manifest_dict
from lunar_data_pipeline.pds4_parser import parse_label

OHRC_LABEL = (
    "C:/Users/SAISH MALVANKAR/Downloads/"
    "ch2_ohr_ncp_20250516T1342347288_d_img_d18/"
    "data/calibrated/20250516/ch2_ohr_ncp_20250516T1342347288_d_img_d18.xml"
)
TMC_LABEL = (
    "C:/Users/SAISH MALVANKAR/Downloads/"
    "ch2_tmc_nca_20200607T2239162106_d_img_d18/"
    "data/calibrated/20200607/ch2_tmc_nca_20200607T2239162106_d_img_d18.xml"
)
OUT = "pair_manifest_phase5_60n.json"


def main() -> int:
    ohrc = parse_label(OHRC_LABEL)
    tmc = parse_label(TMC_LABEL)
    if ohrc is None or tmc is None:
        print("Failed to parse one or both labels")
        return 2
    print(f"OHRC {ohrc.product_id} geo_csv={ohrc.geometry_csv_path}")
    print(f"TMC  {tmc.product_id} geo_csv={tmc.geometry_csv_path}")
    print(f"OHRC sun elev={ohrc.sun_elevation_deg} az={ohrc.sun_azimuth_deg}")
    print(f"TMC  sun elev={tmc.sun_elevation_deg} az={tmc.sun_azimuth_deg}")

    transform = build_pixel_transform(ohrc, tmc)
    gt = transform_to_manifest_dict(transform)

    pair = {
        "pair_id": f"{ohrc.product_id}__x__{tmc.product_id}",
        "ohrc": {
            "product_id": ohrc.product_id,
            "label_path": OHRC_LABEL,
            "image_path": str(ohrc.file_path),
            "geometry_csv_path": str(ohrc.geometry_csv_path),
            "lines": ohrc.lines,
            "samples": ohrc.samples,
        },
        "tmc2": {
            "product_id": tmc.product_id,
            "label_path": TMC_LABEL,
            "image_path": str(tmc.file_path),
            "geometry_csv_path": str(tmc.geometry_csv_path),
            "lines": tmc.lines,
            "samples": tmc.samples,
        },
        "ground_truth": gt,
    }

    manifest = {
        "schema_version": "1.0",
        "data_root": "D:\\Multi-Modal-lunar-image-matching-system",
        "counts": {"OHRC": 1, "TMC2": 1, "IIRS": 0, "pairs": 1},
        "products": {},
        "pairs": [pair],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
