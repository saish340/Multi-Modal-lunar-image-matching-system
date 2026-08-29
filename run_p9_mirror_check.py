"""Phase 9 - independent mirror/orientation check for the 60N pair,
using the exact two-source method from Phase 8:
  1) raw per-pixel geolocation CSV: dlat/dscan sign (along-track),
     dlon/dpix sign (across-track)
  2) independent PDS4 corner-footprint orientation (top-edge / left-edge dlat)
Also recompute for the Phase 8 pair for side-by-side, and record each
product's orbit_limb_direction (ascending/descending) to correlate with any
mirror flip.
No search is run here - this is the pre-search orientation check.
"""
import sys
import re
import numpy as np

sys.path.insert(0, "lunar_data_pipeline")
from pds4_parser import parse_label
from csv_geolocation import read_geolocation_csv

WS = "D:/Multi-Modal-lunar-image-matching-system"
DL = "C:/Users/SAISH MALVANKAR/Downloads"

PRODUCTS = {
    # Phase 8 pair
    "ohr_20231004": f"{WS}/ch2_ohr_ncp_20231004T0406038822_d_img_d18/data/calibrated/20231004/ch2_ohr_ncp_20231004T0406038822_d_img_d18.xml",
    "tmc_ncn_20250707": f"{WS}/ch2_tmc_ncn_20250707T1853051045_d_img_d18/data/calibrated/20250707/ch2_tmc_ncn_20250707T1853051045_d_img_d18.xml",
    # 60N pair
    "ohr_20250516": f"{DL}/ch2_ohr_ncp_20250516T1342347288_d_img_d18/data/calibrated/20250516/ch2_ohr_ncp_20250516T1342347288_d_img_d18.xml",
    "tmc_nca_20200607": f"{DL}/ch2_tmc_nca_20200607T2239162106_d_img_d18/data/calibrated/20200607/ch2_tmc_nca_20200607T2239162106_d_img_d18.xml",
}


def orbit_dir(label_path):
    m = re.search(r"orbit_limb_direction>\s*([^<]+)", open(label_path, encoding="utf-8").read())
    return m.group(1).strip() if m else "?"


def mirror_verdict(csv_same: bool, corner_same: bool) -> str:
    """Mirror classification from two orientation comparisons (data-independent).

    Phase 9 finding: the mirror is governed by the pair's along-track latitude
    orientation. If EITHER the raw-CSV comparison OR the independent corner
    comparison says the products' along-track latitude ordering is OPPOSITE,
    the pair is mirrored and needs an explicit scan-axis flip before the
    positive-scale anisotropic model. Both sources must be consulted so a
    mirror cannot be missed by a single-source artifact.
    """
    if csv_same and corner_same:
        return "NONE"
    return "MIRRORED"


def orbit_mirror_prediction(ohrc_limb: str, tmc_limb: str) -> str:
    """Predict whether a pair is mirrored from orbit limb direction alone.

    Phase 9 finding: the along-track mirror appears precisely when the two
    instruments imaged with OPPOSITE ascending/descending limb directions
    (an ascending vs descending pass crosses latitude in opposite directions
    along the scan axis). Same limb -> no mirror; opposite limb -> mirror.
    Used to cross-check (never to override) the geolocation-derived verdict.
    """
    o = ohrc_limb.strip().lower()
    t = tmc_limb.strip().lower()
    if o == t:
        return "NONE"
    return "MIRRORED"


def analyze(name, g, p):
    # along-track: mean dlat/dscan (over distinct scans)
    scans = np.unique(g.scan)
    sl = np.array([np.mean(g.lat[g.scan == s]) for s in scans])
    lat_along = np.mean(np.gradient(sl))
    # across-track: mean dlon/dpix
    pix = np.unique(g.pixel)
    pl = np.array([np.mean(g.lon[g.pixel == x]) for x in pix])
    lon_across = np.mean(np.gradient(pl))
    # corner footprint
    fp = p.footprint  # (lat, lon) in UL,UR,LR,LL
    lats = [c[0] for c in fp]
    lons = [c[1] for c in fp]
    # handle lon wrap for the 60N pair near 355 (tetrailic)
    def unwrap(a):
        a = np.asarray(a, float)
        return a if np.max(np.abs(np.diff(a))) < 180 else np.where(a < 180, a + 360, a)
    lons_u = unwrap(lons)
    top_lon_span = lons_u[1] - lons_u[0]      # rightward along top edge (pixel+)
    left_lat_span = lats[3] - lats[0]          # downward along left edge (scan+)
    return dict(
        lat_along=lat_along, lon_across=lon_across,
        corner_top_lon=top_lon_span, corner_left_lat=left_lat_span,
    )


print("=" * 84)
print("Independent mirror check: raw CSV geolocation + PDS4 corner footprints")
print("=" * 84)
results = {}
for name, lab in PRODUCTS.items():
    p = parse_label(lab)
    g = read_geolocation_csv(p.geometry_csv_path)
    r = analyze(name, g, p)
    results[name] = r
    od = orbit_dir(lab)
    print(f"\n{name}  [orbit_limb_direction={od}]")
    print(f"  raw CSV  : dlat/dscan={r['lat_along']:+.6f}   dlon/dpix={r['lon_across']:+.6f}")
    print(f"  corner   : left-edge dlat(scan+)={r['corner_left_lat']:+.6f}  "
          f"top-edge dlon(pix+)={r['corner_top_lon']:+.6f}")

print("\n" + "=" * 84)
print("Per-pair mirror determination (along-track latitude orientation)")
print("=" * 84)
for pname, (o_name, t_name) in {
    "Phase-8 pair": ("ohr_20231004", "tmc_ncn_20250707"),
    "60N pair": ("ohr_20250516", "tmc_nca_20200607"),
}.items():
    o, t = results[o_name], results[t_name]
    o_al, t_al = o["lat_along"], t["lat_along"]
    o_corner, t_corner = o["corner_left_lat"], t["corner_left_lat"]
    # CSV-based
    csv_same = (np.sign(o_al) == np.sign(t_al))
    # Corner-based
    cor_same = (np.sign(o_corner) == np.sign(t_corner))
    mirror = mirror_verdict(csv_same, cor_same)
    # Orbit-direction cross-check: mirror <=> opposite ascending/descending
    o_orb = orbit_dir(PRODUCTS[o_name])
    t_orb = orbit_dir(PRODUCTS[t_name])
    orbit_pred = orbit_mirror_prediction(o_orb, t_orb)
    match = "MATCH" if (orbit_pred == mirror) else "MISMATCH"
    print(f"\n{pname}: OHRC dlat/dscan={o_al:+.3f} vs TMC dlat/dscan={t_al:+.3f} "
          f"-> CSV {'same' if csv_same else 'OPPOSITE'}; "
          f"corner left-edge {o_corner:+.3f} vs {t_corner:+.3f} "
          f"-> {'same' if cor_same else 'OPPOSITE'}")
    print(f"  VERDICT: {mirror}   [orbit limb {o_orb} vs {t_orb} "
          f"-> predicts {orbit_pred}: {match}]")
