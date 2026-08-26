# Pair Manifest Schema (`pair_manifest.json`)

Version: **1.0** (see `schema_version` field). Written by `pipeline.py`.

The manifest records every confirmed OHRC x TMC-2 footprint overlap together
with the inline ground-truth transform used to predict corresponding pixel
locations between the two images. All paths are absolute strings; all
coordinates are selenographic degrees unless stated otherwise.

## Top-level object

```jsonc
{
  "schema_version": "1.0",
  "generated_utc": "2026-08-26T12:34:56+00:00",
  "data_root": "D:/path/to/downloaded/products",
  "counts": {
    "OHRC": 2,          // parsed products per instrument
    "TMC2": 2,
    "IIRS": 0,
    "pairs": 1           // confirmed overlapping pairs
  },
  "skipped_labels": [    // label XMLs rejected during discovery (diagnostics)
    { "label": ".../browse_x.xml", "reason": "browse/geometry label" },
    { "label": ".../broken.xml",   "reason": "parse failed or incomplete" },
    { "label": ".../nofp.xml",     "reason": "missing footprint corners" }
  ],
  "products": { ... },   // product_id -> ProductEntry (below), all instruments
  "pairs": [ /* PairEntry, sorted by overlap % descending */ ]
}
```

## `products[product_id]` — ProductEntry

```jsonc
{
  "product_id": "ch2_ohr_ncp_20260103T0609041371_d_img_d18",
  "instrument": "OHRC",                  // OHRC | TMC2 | IIRS
  "label_path": "/abs/path/to/data.xml",
  "image_path": "/abs/path/to/img.img",  // may not exist on disk (flagged in logs)
  "geometry_csv_path": null,             // path to *_g_grd_*.csv if discovered
  "lines": 101074,
  "samples": 12000,
  "datatype": "UnsignedByte",            // PDS4 Element_Array/data_type
  "pixel_resolution_m": 0.25,
  "projection": "Polar stereographic",
  "area": "South Pole",
  "start_time": "2026-01-03T06:09:04.1371Z",
  "stop_time": "2026-01-03T06:09:20.5206Z",
  "logical_identifier": "urn:isro:isda:...",
  "footprint_lonlat_corners":            // [[lon, lat] x 4] in order
    [[27.72, -85.32], [26.57, -85.36],   // UL, UR
     [23.01, -84.58], [24.02, -84.55]],  // LR, LL
  "pairing_eligible": true               // had corners AND appears in >= 1 pair
}
```

Corner order everywhere is **UL, UR, LR, LL** as read from the label's
`Refined_Corner_Coordinates` (fallback: `System_Level_Coordinates`).

## `pairs[i]` — PairEntry

```jsonc
{
  "pair_id": "<ohrc_product_id>__x__<tmc2_product_id>",

  "ohrc": {
    "product_id": "...", "label_path": "...", "image_path": "...",
    "geometry_csv_path": "...|null",
    "lines": 101074, "samples": 12000
  },
  "tmc2": { /* same fields */ },

  "overlap": {
    "percentage_of_smaller_pct": 100.0,   // ranking metric (see note)
    "pct_of_ohrc_footprint": 100.0,       // intersection area / OHRC area
    "pct_of_tmc2_footprint": 0.0012,      // intersection area / TMC2 area
    "polygon_lonlat": [[lon, lat], ...],  // exterior ring of the intersection
    "centroid_lonlat": [lon, lat]
  },

  "ground_truth": {
    "method": "corner_homography",
    "pixel_convention": "(x=sample, y=line), zero-based, UL=(0,0)",
    "geo_convention": "(lon, lat) selenographic degrees",
    "product_a_id": "<ohrc id>",
    "product_b_id": "<tmc2 id>",
    "h_pixel_to_geo_a": [[...], [...], [...]],  // 3x3 row-major, pixel(A)->geo
    "h_geo_to_pixel_b": [[...], [...], [...] ]  // 3x3 row-major, geo->pixel(B)
  }
}
```

### Overlap percentage semantics

`percentage_of_smaller_pct = 100 * inter_area / min(area_a, area_b)`.
An OHRC scene (~3 km) is tiny next to a TMC-2 strip (tens of km), so this is
effectively *"% of the OHRC footprint covered by TMC-2"* — the most useful
signal for ranking which pairs to spend matching effort on.

Areas are computed in raw lon/lat degrees (planar assumption). Near the poles
meridian convergence inflates degree-based areas, so treat polar percentages
as approximate; boolean overlap detection remains reliable.

### Ground-truth usage and accuracy

To predict where pixel `(x_a, y_a)` of image A should match in image B:

```
[lon, lat, w] = H_pixel_to_geo_a @ [x_a, y_a, 1]
[x_b, y_b, w'] = H_geo_to_pixel_b @ [lon/w, lat/w, 1]
```

Both matrices are normalised so their `[2][2]` entry equals 1. The transform
is a **4-corner homography approximation**: pushbroom line-scan geometry is
not projective, so accuracy degrades over large footprints (long TMC strips)
and toward the poles. Score matches with a tolerance (e.g. within N pixels)
rather than expecting exact hits. Each product's per-pixel geolocation CSV
(`geometry/*_g_grd_*.csv`, columns Longitude,Latitude,Pixel,Scan) can refine
this in a later phase.
