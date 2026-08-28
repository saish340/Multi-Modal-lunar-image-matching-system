"""Phase 4c: trace actual crater-graph RANSAC inlier matches through full mapping."""
import json
from pathlib import Path

import numpy as np

from lunar_data_pipeline.pds4_parser import parse_label
from lunar_data_pipeline.groundtruth import transform_from_manifest_dict, build_pixel_transform
from lunar_data_pipeline.csv_geolocation import (
    fit_local_transforms, locate_region, read_geolocation_csv,
)
from lunar_data_pipeline.overlap_finder import footprint_polygon
from lunar_data_pipeline.image_loader import load_image
from lunar_data_pipeline.scale_pyramid import downsample_to_match
from lunar_data_pipeline.crater_detector import CraterDetection

ROOT = Path(r"D:\Multi-Modal-lunar-image-matching-system")
manifest = json.loads((ROOT / "pair_manifest.json").read_text(encoding="utf-8"))
pair = manifest["pairs"][0]
gt_entry = pair["ground_truth"]
t = transform_from_manifest_dict(gt_entry)

ohrc = parse_label(Path(pair["ohrc"]["label_path"]))
tmc = parse_label(Path(pair["tmc2"]["label_path"]))

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

ohrc_img = load_image(ohrc)
scaled_a = downsample_to_match(ohrc_img, ohrc.pixel_resolution_m, tmc.pixel_resolution_m)
factor = scaled_a.downsample_factor
print(f"factor={factor:.4f}, A-downshape={scaled_a.image.shape}")

# Load cached detections
c = np.load(ROOT / "crater_graph_comparison_output_mni" / "crater_detections.npz", allow_pickle=False)
def load(prefix):
    return [CraterDetection(cx=float(c[f"{prefix}_cx"][i]), cy=float(c[f"{prefix}_cy"][i]),
                            radius=float(c[f"{prefix}_r"][i]), confidence=float(c[f"{prefix}_conf"][i]))
            for i in range(c[f"{prefix}_cx"].shape[0])]
det_a, det_b = load("a"), load("b")
print("det A", len(det_a), "det B", len(det_b))

# Rebuild graphs + match + RANSAC exactly as the baseline does
from lunar_data_pipeline.crater_graph import build_graph
from lunar_data_pipeline.graph_matcher import match_mutual_neighbor, matches_to_pixel_pairs
k=4; min_radius_m=300.0; ransac_thresh=15.0
ga = build_graph([d.cx for d in det_a],[d.cy for d in det_a],[d.radius for d in det_a],6.13,k=k,min_radius_m=min_radius_m)
gb = build_graph([d.cx for d in det_b],[d.cy for d in det_b],[d.radius for d in det_b],6.13,k=k,min_radius_m=min_radius_m)
print("nodes A", len(ga.nodes), "B", len(gb.nodes))
matches = match_mutual_neighbor(ga, gb, ratio_test=0.6, sigma=0.5, n_candidates_per_node=8,
    min_spectrum_similarity=0.1, neighbor_d_tol=0.6, neighbor_s_tol=0.6,
    min_mutual_fraction=0.5, physical=True, use_ratio_test=True)
print("matches", len(matches))
import cv2
src_adj, dst_adj = matches_to_pixel_pairs(ga, gb, matches)
if len(src_adj) >= 4:
    H, mask = cv2.findHomography(src_adj.reshape(-1,1,2), dst_adj.reshape(-1,1,2), cv2.RANSAC, ransac_thresh, maxIters=10000)
    mask = mask.ravel().astype(bool)
    print("RANSAC inliers:", int(mask.sum()))
    print(f"\n{'A-down(adj)':>20}{'A-orig':>16}{'B-crop':>16}{'B-full':>16}{'GT-exp B':>16}{'err(dx,dy)':>20}")
    for i in np.flatnonzero(mask):
        a_adj = src_adj[i]; b_crop = dst_adj[i]
        a_orig = a_adj * factor
        b_full = b_crop + np.array([x0, y0])
        gt_b = refined.project_many(a_orig.reshape(1,2))[0]
        err = b_full - gt_b
        print(f"{str(tuple(np.round(a_adj,1))):>20}{str(tuple(np.round(a_orig,1))):>16}"
              f"{str(tuple(np.round(b_crop,1))):>16}{str(tuple(np.round(b_full,1))):>16}"
              f"{str(tuple(np.round(gt_b,1))):>16}{str(tuple(np.round(err,1))):>20}")
