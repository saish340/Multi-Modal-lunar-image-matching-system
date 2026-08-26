# Crater Detection Schema

## Input Data

### Crater Ground Truth Database

External crater catalogs accepted (auto-detected from CSV header):

| Format | Source | Columns | Notes |
|--------|--------|---------|-------|
| Robbins (2018) | USGS Astropedia | `Longitude, Latitude, Diameter_km, Depth_km` | ~1.3M craters ≥1 km. Download: `lunar_crater_database_robbins_2018` |
| LU1319373 Wang & Wu (2021) | Zenodo (4983248) | `lon, lat, diameter` | 1.32M craters ≥1 km with 3D morphometry |
| IAU Gazetteer | planetarynames.wr.usgs.gov | `Feature Name, Center Latitude, Center Longitude, Diameter` | Named features only (subset) |
| Synthetic (fallback) | Generated | `lat, lon, diameter_km, name` | Power-law size distribution, seed-reproducible |

Column auto-detection is case-insensitive and matches common aliases (e.g. `Center Lat`, `lat_deg`, `clat`).

### Image Products

Loaded via existing `image_loader.py` + `csv_geolocation.py` pipeline. Crops to overlap region defined by `pair_manifest.json`.

## Detection Pipeline

```
Raw grayscale → CLAHE + blur → Top-hat + Bottom-hat → Canny edges → HoughCircles → Validation
```

### DetectionParams

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clahe_clip` | 3.0 | CLAHE clip limit |
| `tophat_ksize` | 31 | Top-hat structuring element diameter |
| `canny_low` / `canny_high` | 30 / 100 | Canny thresholds |
| `min_radius` | 10 | Min crater radius (px) |
| `max_radius` | 500 | Max crater radius (px) |
| `dp` | 1.2 | HoughCircles accumulator resolution |
| `hough_param2` | 35 | Accumulator vote threshold |

### Resolution-Preset Params

| Resolution | Preset | Min Radius | Max Radius |
|------------|--------|------------|------------|
| ≤1.0 m/px (OHRC) | Ultra-high | 50 px | 5000 px |
| 1–10 m/px (TMC-2) | Medium | 15 px | 500 px |
| >10 m/px (WAC) | Low | 3 px | 50 px |

## Scoring

### Match Criteria

A detection matches a GT crater if **both**:
1. Centre distance < `match_radius_px` (default 150 px)
2. Diameter ratio < `match_diameter_ratio` (default 3.0×)

### Metrics

| Metric | Formula |
|--------|---------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 × P × R / (P + R) |

### Output JSON Schema

```json
{
  "manifest": "pair_manifest.json",
  "gt_source": "external|synthetic",
  "gt_count": 15,
  "overlap_bounds": {"lon_min": 25.15, "lon_max": 25.26, "lat_min": -13.78, "lat_max": -12.92},
  "ohrc": {
    "product_id": "...",
    "crop_size": [12000, 101074],
    "resolution_m": 0.2,
    "n_detections": 8,
    "scoring": {
      "tp": 5, "fp": 3, "fn": 10,
      "precision": 0.625, "recall": 0.333, "f1": 0.435,
      "per_detection": [...]
    }
  },
  "tmc2": { ... }
}
```

## Visualization

Overlay PNGs drawn with OpenCV:
- **Green circles**: detected craters
- **Blue circles**: GT craters
- **Circle radius**: detected radius (green) or GT diameter/2 (blue)
