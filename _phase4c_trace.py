"""Phase 4c: trace known crater control points through every coordinate stage."""
import json
import math
from pathlib import Path

import numpy as np

from lunar_data_pipeline.pds4_parser import parse_label
from lunar_data_pipeline.groundtruth import (
    transform_from_manifest_dict, build_pixel_transform, _apply_homography,
)
from lunar_data_pipeline.csv_geolocation import (
    fit_local_transforms, locate_region, read_geolocation_csv,
)
from lunar_data_pipeline.overlap_finder import footprint_polygon
from lunar_data_pipeline.scale_pyramid import downsample_to_match

ROOT = Path(r"D:\Multi-Modal-lunar-image-matching-system")
manifest = json.loads((ROOT / "pair_manifest.json").read_text(encoding="utf-8"))
pair = manifest["pairs"][0]
gt_entry = pair["ground_truth"]

t = transform_from_manifest_dict(gt_entry)
ohrc = parse_label(Path(pair["ohrc"]["label_path"]))
tmc = parse_label(Path(pair["tmc2"]["label_path"]))

print("OHRC", ohrc.samples, "samples x", ohrc.lines, "lines, res", ohrc.pixel_resolution_m)
print("TMC ", tmc.samples, "samples x", tmc.lines, "lines, res", tmc.pixel_resolution_m)

# --- Refined local GT (as the runner builds it) ---
tmc_grid = read_geolocation_csv(tmc.geometry_csv_path)
ohrc_polygon = footprint_polygon(ohrc)
scan_rng, pix_rng, _ = locate_region(tmc_grid, ohrc_polygon)
crop_padding = 500
s0 = max(scan_rng[0] - crop_padding, 0); s1 = min(scan_rng[1] + crop_padding, tmc.lines)
c0 = max(pix_rng[0] - crop_padding, 0); c1 = min(pix_rng[1] + crop_padding, tmc.samples)
x0, y0, x1, y1 = c0, s0, c1, s1
h_px_to_geo, h_geo_to_px, fit_points = fit_local_transforms(tmc_grid, (y0, y1), (x0, x1))
refined = build_pixel_transform(ohrc, tmc)
refined.h_pixel_to_geo_a = t.h_pixel_to_geo_a
refined.h_geo_to_pixel_b = h_geo_to_px
print(f"\nCrop window: cols[{x0}:{x1}] rows[{y0}:{y1}]  -> offset (x0,y0)=({c0},{s0})")

# --- Downsample A (as the runner does) ---
from lunar_data_pipeline.image_loader import load_image
ohrc_img = load_image(ohrc)
scaled_a = downsample_to_match(
    ohrc_img, ohrc.pixel_resolution_m, tmc.pixel_resolution_m)
print(f"\nOHRC downsampled shape (H,W)={scaled_a.image.shape} factor={scaled_a.downsample_factor:.6f}")
print(f"  x-factor samples/new=", ohrc.samples / scaled_a.image.shape[1],
      " y-factor lines/new=", ohrc.lines / scaled_a.image.shape[0])

# The 3 real craters: name, lon, lat, OHRC full px, TMC crop px
craters = [
    ("11-0-05812", 25.249, -13.556, (819.2, 26022.2), (383.8, 4281.0)),
    ("11-0-05828", 25.178, -13.512, (9950.8, 31330.9), (724.9, 4019.5)),
    ("11-3-00076", 25.212, -13.239, (4822.5, 63493.1), (608.8, 2318.0)),
]

print("\n=== STAGE-BY-STAGE TRACE (crater control points) ===")
hdr = f"{'crater':<12}{'infullOHRC':>16}{'->geo(lon,lat)':>20}{'->TMCfull':>14}{'->crop-loc':>14}{'known crop':>14}{'crop dxy':>14}"
print(hdr)
print("-" * len(hdr))
for name, lon, lat, (oh_px, oh_py), (crop_x, crop_y) in craters:
    geo = _apply_homography(t.h_pixel_to_geo_a, np.array([oh_px, oh_py]))
    tmc_full = refined.project((oh_px, oh_py))
    crop_local = (tmc_full[0] - x0, tmc_full[1] - y0)
    print(f"{name:<12}{('('+f'{oh_px:.1f},{oh_py:.1f}'+')'):>16}"
          f"{('('+f'{geo[0]:.4f},{geo[1]:.4f}'+')'):>20}"
          f"{('('+f'{tmc_full[0]:.1f},{tmc_full[1]:.1f}'+')'):>14}"
          f"{('('+f'{crop_local[0]:.1f},{crop_local[1]:.1f}'+')'):>14}"
          f"{('('+f'{crop_x:.1f},{crop_y:.1f}'+')'):>14}"
          f"{('('+f'{crop_local[0]-crop_x:+.1f},{crop_local[1]-crop_y:+.1f}'+')'):>14}")

# Also test raw corner-homography transform (not refined)
print("\n=== Compare corner-homography vs refined-local (lat/lon -> TMC full) ===")
for name, lon, lat, (oh_px, oh_py), (crop_x, crop_y) in craters:
    t_full = t.project((oh_px, oh_py))
    t_ref = refined.project((oh_px, oh_py))
    print(f"{name:<12} corner->TMCfull={t_full}  refined->TMCfull={t_ref}  dy={t_full[1]-t_ref[1]:+.1f}")

# --- Verify downsample mapping on A: known OHRC pixel -> downsampled -> back ---
print("\n=== Downsample round-trip (OHRC full px == full px *factor?) ===")
for name, lon, lat, (oh_px, oh_py), _ in craters:
    adj_x, adj_y = oh_px / scaled_a.downsample_factor, oh_py / scaled_a.downsample_factor
    back_x, back_y = scaled_a.to_original(np.array([adj_x, adj_y]))
    print(f"{name:<12} full=({oh_px:.1f},{oh_py:.1f})  adj=({adj_x:.2f},{adj_y:.2f})  "
          f"to_original=({back_x:.1f},{back_y:.1f})  roundtrip_err=({back_x-oh_px:+.2f},{back_y-oh_py:+.2f})")
