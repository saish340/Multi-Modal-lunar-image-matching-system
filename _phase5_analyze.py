import logging, json, sys
from pathlib import Path
import numpy as np
from datetime import datetime

logging.disable(logging.CRITICAL)

from lunar_data_pipeline.shapefile_loader import discover_catalog_files, load_catalog
from lunar_data_pipeline.catalog_overlap_search import search_overlaps_with_index
from lunar_data_pipeline.candidate_ranking import load_craters, crater_point_counts, polygon_area_deg2

root = Path(r"D:\Multi-Modal-lunar-image-matching-system")

ohrc_files = discover_catalog_files(root, "OHRC")
tmc_files = discover_catalog_files(root, "TMC2")
ohrc_products, _ = load_catalog(ohrc_files, "OHRC")
tmc_products, _ = load_catalog(tmc_files, "TMC2")
print(f"OHRC {len(ohrc_products)}  TMC2 {len(tmc_products)}")

def tsec(ts):
    if not ts: return None
    try: return datetime.fromisoformat(str(ts).replace("Z","")).timestamp()
    except Exception: return None

pairs = search_overlaps_with_index(ohrc_products, tmc_products).pairs
print(f"overlapping pairs: {len(pairs)}")

# crater density
craters = load_craters(root / "data" / "lunar_craters.parquet")
counts = crater_point_counts(craters, [p.overlap_polygon for p in pairs])

rows = []
for p, cnt in zip(pairs, counts):
    o, t = p.ohrc, p.tmc2
    o_t = tsec(o.start_time); t_t = tsec(t.start_time)
    dt = abs(o_t - t_t) if (o_t and t_t) else None
    # time-of-day difference (fractional day) -> proxy for sun elevation/azimuth match
    tod_diff = None
    if o_t and t_t:
        fr_o = (o_t % 86400)/86400; fr_t = (t_t % 86400)/86400
        d = abs(fr_o-fr_t); tod_diff = min(d, 1-d)*86400
    area = polygon_area_deg2(p.overlap_polygon)
    rows.append({
        "ohrc": o.product_id, "ohrc_orbit": o.imaging_orbit_number, "ohrc_start": o.start_time,
        "tmc": t.product_id, "tmc_orbit": t.imaging_orbit_number, "tmc_start": t.start_time,
        "dt_sec": dt, "tod_diff_sec": tod_diff,
        "overlap": round(p.overlap_percentage,1),
        "lonlat": [round(float(p.overlap_polygon.centroid.x),3), round(float(p.overlap_polygon.centroid.y),3)],
        "craters": cnt, "density": round(cnt/area,3) if area>0 else 0,
    })

# save all
json.dump(rows, open(r"_phase5_allpairs.json","w"), indent=1)

# Co-temporal = same orbit (OHRC and TMC acquired on same pass). Filter: dt_sec < 3600 (same pass) or same ohrc orbit appearing with tmc of similar orbit set
print("\n=== CO-TEMPORAL (|OHRC start - TMC start| < 1 hr): sun guaranteed matched ===")
co = [r for r in rows if r["dt_sec"] is not None and r["dt_sec"] < 3600]
co.sort(key=lambda r: -r["craters"])
for r in co:
    print(f"ohrc {r['ohrc']} (orb {r['ohrc_orbit']} {r['ohrc_start']})  x  tmc {r['tmc']} (orb {r['tmc_orbit']} {r['tmc_start']})  dt={r['dt_sec']}s  overlap={r['overlap']}% craters={r['craters']} lonlat={r['lonlat']}")
print(f"total co-temporal pairs: {len(co)}")

print("\n=== NEAR-co-temporal (< 6 hr) with craters>=5 ===")
nc = [r for r in rows if r["dt_sec"] is not None and r["dt_sec"] < 21600 and r["craters"]>=3]
nc.sort(key=lambda r: -r["craters"])
for r in nc[:40]:
    print(f"ohrc {r['ohrc']} {r['ohrc_start']}  x  tmc {r['tmc']} {r['tmc_start']}  dt={r['dt_sec']}s  tod_diff={r['tod_diff_sec']}s  overlap={r['overlap']}% craters={r['craters']} lonlat={r['lonlat']}")
