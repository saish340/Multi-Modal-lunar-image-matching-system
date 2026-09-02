"""Quick overlap verification: project OHRC/IIRS footprint corners into TMC GeoTIFF coords."""
import struct, math, re

R = 1737400.0  # Moon radius (meters)
# TMC GeoTIFF: Polar Stereographic South, center lat=-90, central lon=0, scale=1
# ModelTiepoint: pixel (0,0) -> (X=24877.14, Y=829679.19)
# ModelPixelScale: 5.0 m
TMC_X0, TMC_Y0, TMC_PS = 24877.140625, 829679.1875, 5.0

def latlon_to_tmc_xy(lat_deg, lon_deg):
    """South Polar Stereographic (spherical, tangent at pole, Y north-positive)."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    rho = 2 * R * math.tan(math.pi / 4 + phi / 2)
    x = rho * math.sin(lam)
    y = rho * math.cos(lam)  # positive = north, away from pole
    return x, y

def tmc_xy_to_pixel(x, y):
    col = (x - TMC_X0) / TMC_PS
    row = (TMC_Y0 - y) / TMC_PS  # row 0 = top (max Y), rows increase downward
    return col, row

def in_tmc_bounds(col, row):
    return -500 <= col <= 56155 + 500 and -500 <= row <= 169968 + 500

# --- TMC bounds ---
tmc_cols, tmc_rows = 56156, 169968
print("=== TMC ortho bounds ===")
x_min = TMC_X0
x_max = TMC_X0 + (tmc_cols - 1) * TMC_PS
y_min = TMC_Y0 - (tmc_rows - 1) * TMC_PS
y_max = TMC_Y0
print(f"Projected X: [{x_min:.0f}, {x_max:.0f}]")
print(f"Projected Y: [{y_min:.0f}, {y_max:.0f}]")

# --- OHRC footprint ---
ohr_corners = [
    (-68.023374, 20.770117, "UL"),
    (-68.017825, 21.034219, "UR"),
    (-68.862640, 20.807854, "LL"),
    (-68.857015, 21.081631, "LR"),
]
print("\n=== OHRC corners vs TMC ===")
ohr_pixels = []
for lat, lon, label in ohr_corners:
    x, y = latlon_to_tmc_xy(lat, lon)
    col, row = tmc_xy_to_pixel(x, y)
    ok = in_tmc_bounds(col, row)
    ohr_pixels.append((col, row))
    print(f"OHRC {label} ({lat:.4f}, {lon:.4f}) -> TMC pixel ({col:.1f}, {row:.1f})  in_bounds={ok}")

cols = [p[0] for p in ohr_pixels]
rows = [p[1] for p in ohr_pixels]
print(f"OHRC TMC pixel range: col [{min(cols):.0f}, {max(cols):.0f}], row [{min(rows):.0f}, {max(rows):.0f}]")
print(f"TMC image bounds:     col [0, {tmc_cols-1}], row [0, {tmc_rows-1}]")
ohr_overlap = all(in_tmc_bounds(c, r) for c, r in ohr_pixels)
print(f"OHRC fully within TMC bounds: {ohr_overlap}")

# --- IIRS footprint ---
iirs_corners = [
    (-71.435003, 21.144358, "UL"),
    (-71.463998, 19.775695, "UR"),
    (-35.529045, 17.900171, "LL"),
    (-35.539624, 17.412499, "LR"),
]
print("\n=== IIRS corners vs TMC ===")
iirs_pixels = []
for lat, lon, label in iirs_corners:
    x, y = latlon_to_tmc_xy(lat, lon)
    col, row = tmc_xy_to_pixel(x, y)
    ok = in_tmc_bounds(col, row)
    iirs_pixels.append((col, row))
    print(f"IIRS {label} ({lat:.4f}, {lon:.4f}) -> TMC pixel ({col:.1f}, {row:.1f})  in_bounds={ok}")

cols_i = [p[0] for p in iirs_pixels]
rows_i = [p[1] for p in iirs_pixels]
print(f"IIRS TMC pixel range: col [{min(cols_i):.0f}, {max(cols_i):.0f}], row [{min(rows_i):.0f}, {max(rows_i):.0f}]")
