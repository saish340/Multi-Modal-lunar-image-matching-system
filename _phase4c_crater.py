"""Phase 4c: verify detector positions of a known real crater against GT-predicted positions."""
import json
from pathlib import Path

import numpy as np

from lunar_data_pipeline.pds4_parser import parse_label
from lunar_data_pipeline.groundtruth import transform_from_manifest_dict, build_pixel_transform
from lunar_data_pipeline.csv_geolocation import (
    fit_local_transforms, locate_region, read_geolocation_csv,
)
from lunar_data_pipeline.overlap_finder import footprint_polygon
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
s0 = max(scan_rng[0]-crop_padding,0); s1 = min(scan_rng[1]+crop_padding,tmc.lines)
c0 = max(pix_rng[0]-crop_padding,0); c1 = min(pix_rng[1]+crop_padding,tmc.samples)
x0,y0,x1,y1 = c0,s0,c1,s1
h_px_to_geo, h_geo_to_px, _ = fit_local_transforms(tmc_grid,(y0,y1),(x0,x1))
refined = build_pixel_transform(ohrc, tmc)
refined.h_pixel_to_geo_a = t.h_pixel_to_geo_a
refined.h_geo_to_pixel_b = h_geo_to_px

factor = 30.612245

c = np.load(ROOT/"crater_graph_comparison_output_mni"/"crater_detections.npz", allow_pickle=False)
def load(prefix):
    return [CraterDetection(cx=float(c[f"{prefix}_cx"][i]), cy=float(c[f"{prefix}_cy"][i]),
                            radius=float(c[f"{prefix}_r"][i]), confidence=float(c[f"{prefix}_conf"][i]))
            for i in range(c[f"{prefix}_cx"].shape[0])]
det_a, det_b = load("a"), load("b")

# Real crater 11-0-05812
oh_full = np.array([819.2, 26022.2])
gt_full_b = refined.project((oh_full[0], oh_full[1]))
oh_down = oh_full / factor
b_crop_expected = gt_full_b - np.array([x0, y0])
print(f"Crater 11-0-05812: OHRC full={oh_full}")
print(f"  downsampled A pos expected = ({oh_down[0]:.1f},{oh_down[1]:.1f})")
print(f"  GT TMC full = ({gt_full_b[0]:.1f},{gt_full_b[1]:.1f})  B-crop expected = ({b_crop_expected[0]:.1f},{b_crop_expected[1]:.1f})")

def nearest(dets, pos, r=200):
    arr = np.array([[d.cx, d.cy] for d in dets])
    d2 = np.sum((arr-pos)**2, axis=1)
    i = int(np.argmin(d2))
    return int(np.sqrt(d2[i])), (arr[i][0], arr[i][1])

d_a, pa = nearest(det_a, oh_down)
d_b, pb = nearest(det_b, b_crop_expected)
print(f"\nNearest A-down detection to expected ({oh_down[0]:.1f},{oh_down[1]:.1f}): dist={d_a:.1f}px at {pa}")
print(f"Nearest B-crop detection to expected ({b_crop_expected[0]:.1f},{b_crop_expected[1]:.1f}): dist={d_b:.1f}px at {pb}")

# Also for craters 11-0-05828 and 11-3-00076
craters = [
    ("11-0-05828", (9950.8, 31330.9)),
    ("11-3-00076", (4822.5, 63493.1)),
]
for nm, oh in craters:
    oh = np.array(oh, float)
    gt = refined.project((oh[0], oh[1]))
    down = oh/factor
    bc = gt - np.array([x0,y0])
    dd_a, pa = nearest(det_a, down)
    dd_b, pb = nearest(det_b, bc)
    print(f"\n{nm}: A-down exp {tuple(np.round(down,1))} nearest dist={dd_a:.1f} at {pa}")
    print(f"     B-crop exp {tuple(np.round(bc,1))} nearest dist={dd_b:.1f} at {pb}")

# Does the matcher pair the correct A detection to the correct B detection?
print("\n=== Does graph matcher pair the detected known crate correctly? ===")
from lunar_data_pipeline.crater_graph import build_graph
from lunar_data_pipeline.graph_matcher import match_mutual_neighbor, match_nearest_neighbor
ga = build_graph([d.cx for d in det_a],[d.cy for d in det_a],[d.radius for d in det_a],6.13,k=4,min_radius_m=300.0)
gb = build_graph([d.cx for d in det_b],[d.cy for d in det_b],[d.radius for d in det_b],6.13,k=4,min_radius_m=300.0)
# find A node nearest to known crater down pos
arr_a = np.array([[d.cx,d.cy] for d in det_a])
iA = int(np.argmin(np.sum((arr_a-oh_down)**2,axis=1)))
print("A node id for known crater:", iA, "pos", arr_a[iA])
m = match_mutual_neighbor(ga, gb, physical=True)  # default params
# find match involving iA
hits = [x for x in m if x.idx_a==iA]
print("matches for this A node:", [(x.idx_a, x.idx_b, round(x.mutual_fraction,2)) for x in hits])
if hits:
    bnode = hits[0].idx_b
    bpos = np.array([gb.node(bnode).cx, gb.node(bnode).cy])
    print(f"  matched B node {bnode} at crop {bpos}, full {bpos+np.array([x0,y0])}, GT-expect crop {b_crop_expected}")
    print(f"  error vs GT (crop): {bpos-b_crop_expected}")
