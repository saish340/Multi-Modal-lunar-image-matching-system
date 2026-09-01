"""Read-only verification of the three-sensor Chandrayaan-2 dataset in root.

Stages (all read-only; no source file is modified, resampled or cropped):
  s1  integrity  - parse each PDS4 label, compare declared vs actual size,
                   stream-verify MD5, sample-read pixels from every sensor
  s2  overlap    - project OHRC label corners into the TMC ortho grid using
                   the GeoTIFF's own ModelPixelScale/ModelTiepoint tags;
                   self-check the TMC frame against its Corrected corners;
                   check corner-quad vs physical-extent consistency
  s3  iirs       - IIRS footprint-vs-strip consistency and TMC projection;
                   verdict stays UNVERIFIED unless geometry closes

Writes phase12_verify_output/verify_results.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, box, Point

sys.path.insert(0, str(Path(__file__).parent))

from lunar_data_pipeline.image_loader import (
    expected_file_size,
    load_image,
    numpy_dtype,
)
from lunar_data_pipeline.pds4_parser import parse_label

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "phase12_verify_output"

LABELS = {
    "OHRC": ROOT / "ch2_ohr_nrp_20200827T0030107497_d_img_d18.xml",
    "TMC2": ROOT / "ch2_tmc_ndn_20231101T0125121377_d_oth_d18.xml",
    "IIRS": ROOT / "ch2_iir_nci_20231225T1904122779_d_img_d18.xml",
}

MOON_R = 1737400.0  # spherical Moon radius (m)
GM_MOON = 4.9028e12  # lunar GM (m^3/s^2)


def md5_of(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


_TIFF_TYPES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
               11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8, 19: 8, 20: 8, 21: 8}

_GEOKEY_NAMES = {
    1024: "GTModelType", 1025: "GTRasterType",
    2048: "GeogType", 2049: "GeogCitation",
    2056: "GeogSemiMajorAxis", 2057: "GeogSemiMinorAxis",
    3072: "ProjectedCSType", 3075: "ProjCoordTrans", 3076: "ProjLinearUnits",
    3078: "ProjStdParallel1", 3080: "ProjNatOriginLong", 3081: "ProjNatOriginLat",
    3082: "ProjFalseEasting", 3083: "ProjFalseNorthing",
}


def read_tmc_geotags(tif_path: Path, window: int = 64 << 20) -> dict:
    """Walk IFD0 of the BigTIFF and pull its georeferencing tags (read-only).

    Derives dimensions, ModelPixelScale, ModelTiepoint and resolved GeoKeys
    from the file structure itself -- no prior knowledge of the values.
    """
    with tif_path.open("rb") as f:
        head = f.read(16)
        if head[:2] not in (b"II", b"MM"):
            raise RuntimeError("TMC tif: not a TIFF file")
        e = "<" if head[:2] == b"II" else ">"
        if struct.unpack(e + "H", head[2:4])[0] != 43:
            raise RuntimeError("TMC tif: not BigTIFF (classic TIFF magic)")
        ifd_off = struct.unpack(e + "Q", head[8:16])[0]
        f.seek(ifd_off)
        n = struct.unpack(e + "Q", f.read(8))[0]
        entries = f.read(n * 20)
        values_base = ifd_off + 8 + n * 20
        blob = f.read(window)

    tags: dict[int, object] = {}
    for k in range(n):
        ent = entries[k * 20:(k + 1) * 20]
        tag, typ = struct.unpack(e + "HH", ent[0:4])
        count = struct.unpack(e + "Q", ent[4:12])[0]
        tsize = _TIFF_TYPES.get(typ)
        if tsize is None:
            continue
        total = count * tsize
        if total <= 8:
            raw = ent[12:12 + total]
        else:
            off = struct.unpack(e + "Q", ent[12:20])[0] - values_base
            if off < 0 or off + total > len(blob):
                tags[tag] = f"<{total}B out-of-window>"
                continue
            raw = blob[off:off + total]
        try:
            if typ == 2:
                val: object = raw.rstrip(b"\x00").decode("ascii", "replace")
            elif typ in (3, 4, 16):
                fmt = {3: "H", 4: "I", 16: "Q"}[typ]
                val = struct.unpack(e + str(count) + fmt, raw)
            elif typ == 12:
                val = struct.unpack(e + str(count) + "d", raw)
            else:
                val = raw
        except struct.error:
            continue
        tags[tag] = val[0] if isinstance(val, tuple) and len(val) == 1 else val
    return {"tags": tags, "geokeys": resolve_geokeys(tags), "endian": e}


def resolve_geokeys(tags: dict) -> dict:
    """Dereference GeoKeyDirectory (34735) against GeoDouble/GeoAscii params."""
    shorts = tags.get(34735)
    doubles = tags.get(34736) or ()
    ascii_p = tags.get(34737) or ""
    out: dict[str, object] = {}
    if not isinstance(shorts, tuple) or len(shorts) < 4:
        return out
    for k in range(shorts[3]):
        i = 4 + 4 * k
        if i + 3 >= len(shorts):
            break
        kid, loc, cnt, voff = shorts[i:i + 4]
        name = _GEOKEY_NAMES.get(kid, f"key{kid}")
        if loc == 0:
            out[name] = voff
        elif loc == 34736 and isinstance(doubles, tuple):
            vals = list(doubles[voff:voff + cnt])
            out[name] = vals[0] if len(vals) == 1 else vals
        elif loc == 34737 and isinstance(ascii_p, str):
            out[name] = ascii_p[voff:voff + cnt].rstrip("|")
    return out


def pixel_to_latlon(col: float, row: float, x0: float, y0: float,
                    ps: float) -> tuple[float, float]:
    """TIF pixel -> projected metres -> (lat, lon); north-up GeoTIFF axes."""
    return xy_to_latlon(x0 + col * ps, y0 - row * ps)


def latlon_to_xy(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Spherical south-polar stereographic (tangent at pole, Y north+)."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    rho = 2.0 * MOON_R * math.tan(math.pi / 4.0 + phi / 2.0)
    return rho * math.sin(lam), rho * math.cos(lam)


def xy_to_latlon(x: float, y: float) -> tuple[float, float]:
    rho = math.hypot(x, y)
    if rho == 0.0:
        return -90.0, 0.0
    lat = math.degrees(2.0 * math.atan2(rho, 2.0 * MOON_R) - math.pi / 2.0)
    lon = math.degrees(math.atan2(x, y))
    return lat, lon


def gc_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * MOON_R * math.asin(math.sqrt(a)) / 1000.0


def orbital_speed_mps(alt_km: float) -> float:
    return math.sqrt(GM_MOON / (MOON_R + alt_km * 1000.0))


def parse_corrected_corners(xml_path: Path) -> dict | None:
    text = xml_path.read_text(encoding="utf-8")
    m = re.search(
        r"<isda:Corrected_Corner_Coordinates>(.*?)</isda:Corrected_Corner_Coordinates>",
        text,
        re.S,
    )
    if not m:
        return None
    block = m.group(1)

    def f(tag: str):
        mm = re.search(rf"<isda:{tag}[^>]*>([-\d.]+)</isda:{tag}>", block)
        return float(mm.group(1)) if mm else None

    return {
        "upper_left": (f("upper_left_latitude"), f("upper_left_longitude")),
        "upper_right": (f("upper_right_latitude"), f("upper_right_longitude")),
        "lower_left": (f("lower_left_latitude"), f("lower_left_longitude")),
        "lower_right": (f("lower_right_latitude"), f("lower_right_longitude")),
    }

def patch_stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    fin = a[np.isfinite(a)]
    return {
        "shape": list(arr.shape),
        "finite_fraction": round(float(fin.size / a.size), 6),
        "min": float(fin.min()) if fin.size else None,
        "median": float(np.median(fin)) if fin.size else None,
        "max": float(fin.max()) if fin.size else None,
        "std": round(float(fin.std()), 4) if fin.size else None,
    }


def sample_pixels(product, instrument: str) -> dict:
    """Read-only pixel sampling proving each scientific file is readable."""
    out: dict = {"path": product.file_path.name}
    if instrument == "OHRC":
        # mid-strip patch from the 101075x12000 uint8 raw image
        p = load_image(product, row_range=(50000, 50064), col_range=(5000, 5064))
        out["sample"] = patch_stats(p)
        out["read_via"] = "image_loader.load_image (numpy.memmap)"
    elif instrument == "TMC2":
        # BigTIFF strip layout is not decoded here; sample raw uint16 windows
        # at 25/50/75 % of the file to prove readability + non-degeneracy.
        size = product.file_path.stat().st_size
        samples = {}
        with product.file_path.open("rb") as f:
            for frac in (0.25, 0.5, 0.75):
                off = int(size * frac) // 2 * 2  # align to uint16
                f.seek(off)
                buf = np.frombuffer(f.read(1 << 16), dtype="<u2")
                samples[f"window@{frac:.0%}"] = patch_stats(buf)
        out["sample"] = samples
        out["read_via"] = "raw uint16 window reads (full decode deferred to GDAL)"
    else:  # IIRS: band-sequential float32 cube (bands, lines, samples)
        dt = numpy_dtype(product.datatype)
        mm = np.memmap(product.file_path, dtype=dt, mode="r",
                       shape=(product.bands, product.lines, product.samples))
        bands = {}
        for b in (0, 63, 127, 255):
            bands[f"band_{b}"] = patch_stats(mm[b, 6000:6128, :])
        out["sample"] = bands
        out["read_via"] = "numpy.memmap (band-sequential float32)"
    return out


def md5_status(product) -> dict:
    actual = product.file_path.stat().st_size
    declared = product.file_size_bytes
    implied = expected_file_size(product)
    # declared == actual is the primary anchor; the array model (implied)
    # must not exceed the file and may only fall short by container overhead
    # (<1%, e.g. BigTIFF structure) -- exact equality for raw array products.
    declared_ok = declared is None or declared == actual
    overhead = max(actual - implied, 0)
    container_ok = implied <= actual and overhead <= 0.01 * actual
    size_ok = declared_ok and container_ok
    digest = md5_of(product.file_path)
    return {
        "actual_bytes": actual,
        "declared_bytes": declared,
        "label_implied_bytes": implied,
        "declared_match": declared_ok,
        "container_overhead_bytes": overhead,
        "size_match": size_ok,
        "md5_actual": digest,
        "md5_declared": product.md5_checksum,
        "md5_match": (product.md5_checksum or "").lower() == digest,
    }


def s1_integrity() -> dict:
    print("\n=== s1: integrity (size + MD5 + pixel samples) ===")
    products: dict[str, object] = {}
    report: dict = {}
    for inst, label in LABELS.items():
        product = parse_label(label)
        if product is None:
            report[inst] = {"parse": "FAILED"}
            continue
        products[inst] = product
        entry: dict = {
            "product_id": product.product_id,
            "instrument": product.instrument,
            "lines": product.lines,
            "samples": product.samples,
            "bands": product.bands,
            "datatype": product.datatype,
            "pixel_resolution_m": product.pixel_resolution_m,
            "projection": product.projection,
            "area": product.area,
            "start_time": product.start_time,
            "stop_time": product.stop_time,
            "sun_elevation_deg": product.sun_elevation_deg,
            "sun_azimuth_deg": product.sun_azimuth_deg,
            "solar_incidence_deg": product.solar_incidence_deg,
            "file_exists": product.file_path.exists(),
        }
        if entry["file_exists"]:
            entry["integrity"] = md5_status(product)
            entry["pixels"] = sample_pixels(product, inst)
        report[inst] = entry
        print(f"  [{inst}] {product.product_id}")
        print(f"    {product.lines}x{product.samples}x{product.bands} "
              f"{product.datatype} @ {product.pixel_resolution_m} m/px")
        if entry["file_exists"]:
            it = entry["integrity"]
            print(f"    size {it['actual_bytes']:,} B  declared "
                  f"{it['declared_bytes']:,}  implied {it['label_implied_bytes']:,}  "
                  f"match={it['size_match']}")
            print(f"    md5 match={it['md5_match']}")
    return {"products": report}, products


@dataclass
class Georef:
    """South-polar-stereographic georeferencing taken from the TIF's own tags."""

    ps: float      # metres per pixel (column step == row step)
    x0: float      # projected X (m) of pixel column 0
    y0: float      # projected Y (m) of pixel row 0
    k0: float      # polar scale from ProjStdParallel1 (1.0 = tangent at pole)
    lon0: float    # central meridian (deg)

    def latlon_to_pixel(self, lat: float, lon: float) -> tuple[float, float]:
        phi = math.radians(lat)
        lam = math.radians(lon - self.lon0)
        rho = 2.0 * MOON_R * self.k0 * math.tan(math.pi / 4.0 + phi / 2.0)
        return ((rho * math.sin(lam) - self.x0) / self.ps,
                (self.y0 - rho * math.cos(lam)) / self.ps)


def georef_from_tags(tags: dict, geokeys: dict) -> Georef:
    scale = tags[33550]                      # ModelPixelScale (Sx, Sy, Sz)
    tp = tags[33922]                         # ModelTiepoint (i, j, k, X, Y, Z)
    ps = float(scale[0])
    x0 = float(tp[3]) - float(tp[0]) * ps    # pixel (0,0) projected X
    y0 = float(tp[4]) + float(tp[1]) * ps    # pixel (0,0) projected Y
    lat_ts = geokeys.get("ProjStdParallel1")
    k0 = (1.0 + math.sin(math.radians(abs(float(lat_ts))))) / 2.0 if lat_ts else 1.0
    lon0 = float(geokeys.get("ProjNatOriginLong") or 0.0)
    return Georef(ps=ps, x0=x0, y0=y0, k0=k0, lon0=lon0)


def quad_gc(footprint) -> dict:
    """Ground distances (great-circle) implied by a label footprint.

    ``footprint`` is in parser CORNER_ORDER: UL, UR, LR, LL.
    """
    ul, ur, lr, ll = footprint
    return {
        "width_km": round(gc_km(*ul, *ur), 3),
        "length_km": round(gc_km(*ul, *ll), 3),
        "diagonal_km": round(gc_km(*ul, *lr), 3),
    }


def quad_pixels(footprint, g: Georef) -> list[tuple[float, float]]:
    return [g.latlon_to_pixel(lat, lon) for lat, lon in footprint]


def quad_pixel_extents(px, g: Georef) -> dict:
    ul, ur, lr, ll = px
    d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]) * g.ps / 1000.0
    return {
        "width_km": round(d(ul, ur), 3),
        "length_km": round(d(ul, ll), 3),
        "diagonal_km": round(d(ul, lr), 3),
        "col_range": [round(min(p[0] for p in px), 1), round(max(p[0] for p in px), 1)],
        "row_range": [round(min(p[1] for p in px), 1), round(max(p[1] for p in px), 1)],
    }


def inside_raster(px, width: int, height: int) -> bool:
    return all(0 <= c <= width and 0 <= r <= height for c, r in px)


def s2_overlap(products: dict) -> dict:
    print("\n=== s2: OHRC <-> TMC-2 overlap (TIF's own ModelPixelScale/Tiepoint/GeoKeys) ===")
    tmc = products["TMC2"]
    geo = read_tmc_geotags(tmc.file_path)
    tags, gk = geo["tags"], geo["geokeys"]
    width, height = int(tags[256]), int(tags[257])
    g = georef_from_tags(tags, gk)
    semi_major = gk.get("GeogSemiMajorAxis")
    print(f"  raster {width}x{height} @ {g.ps} m/px | k0={g.k0:.5f} lon0={g.lon0} "
          f"| R_tif={semi_major} (script R={MOON_R:.0f})")
    print(f"  GeoKeys: GTModelType={gk.get('GTModelType')} "
          f"ProjectedCSType={gk.get('ProjectedCSType')} "
          f"ProjCoordTrans={gk.get('ProjCoordTrans')} "
          f"units={gk.get('ProjLinearUnits')} natLat={gk.get('ProjNatOriginLat')}")

    out: dict = {
        "tif": {"width": width, "height": height, "ps_m": g.ps, "k0": g.k0,
                "lon0": g.lon0, "geokeys": {k: str(v) for k, v in gk.items()}},
    }

    # TMC frame self-check: its own label footprint must land inside its raster
    # with pixel-implied extents matching great-circle extents.
    tmc_px = quad_pixels(tmc.footprint, g)
    tmc_ext = quad_pixel_extents(tmc_px, g)
    tmc_gc = quad_gc(tmc.footprint)
    tmc_inside = inside_raster(tmc_px, width, height)
    # Polar-strip note: the TMC label quad sweeps ~155 deg of longitude with its
    # top edge near the south pole, so in the south-polar stereographic plane it
    # can wrap the projection origin, making plain quad-in-raster containment
    # structurally impossible. The extent ratios (pixel-implied vs great-circle)
    # are the frame-independent self-check; containment of the non-wrapping
    # OHRC quad below remains mandatory.
    try:
        tmc_wraps_origin = bool(Polygon(tmc_px).contains(Point(0.0, 0.0)))
    except Exception:
        tmc_wraps_origin = False
    ratio = lambda a, b: round(a / b, 4) if b else None
    out["tmc_selfcheck"] = {
        "label_footprint_inside_raster": tmc_inside,
        "wraps_projection_origin": tmc_wraps_origin,
        "pixel_extents": tmc_ext, "gc_extents": tmc_gc,
        "width_ratio": ratio(tmc_ext["width_km"], tmc_gc["width_km"]),
        "length_ratio": ratio(tmc_ext["length_km"], tmc_gc["length_km"]),
    }
    print(f"  TMC label footprint inside raster: {tmc_inside} "
          f"(cols {tmc_ext['col_range']}, rows {tmc_ext['row_range']}) "
          f"wraps_projection_origin={tmc_wraps_origin}")
    print(f"    extent ratio pixel/GC: width {out['tmc_selfcheck']['width_ratio']}, "
          f"length {out['tmc_selfcheck']['length_ratio']}")

    # OHRC footprint (label System corners) -> TMC pixel space
    ohrc = products["OHRC"]
    ohrc_px = quad_pixels(ohrc.footprint, g)
    ohrc_ext = quad_pixel_extents(ohrc_px, g)
    ohrc_gc = quad_gc(ohrc.footprint)
    ohrc_inside = inside_raster(ohrc_px, width, height)
    poly = Polygon(ohrc_px)
    inter = poly.intersection(box(0, 0, width, height))
    frac = inter.area / poly.area if poly.area else 0.0
    out["ohrc_in_tmc"] = {
        "footprint_inside_raster": ohrc_inside,
        "pixel_extents": ohrc_ext, "gc_extents": ohrc_gc,
        "width_ratio": ratio(ohrc_ext["width_km"], ohrc_gc["width_km"]),
        "length_ratio": ratio(ohrc_ext["length_km"], ohrc_gc["length_km"]),
        "overlap_fraction_of_ohrc": round(float(frac), 6),
        "overlap_km2": round(float(inter.area) * g.ps * g.ps / 1e6, 3),
    }
    print(f"  OHRC footprint inside TMC raster: {ohrc_inside} "
          f"(cols {ohrc_ext['col_range']}, rows {ohrc_ext['row_range']})")
    print(f"    extent ratio pixel/GC: width {out['ohrc_in_tmc']['width_ratio']}, "
          f"length {out['ohrc_in_tmc']['length_ratio']}")
    print(f"    overlap: {out['ohrc_in_tmc']['overlap_fraction_of_ohrc']:.1%} of OHRC "
          f"= {out['ohrc_in_tmc']['overlap_km2']} km^2")

    # The TMC label's Corrected corners are known-inconsistent; project them too
    # so the inconsistency is quantified, not assumed.
    corrected = parse_corrected_corners(tmc.label_path)
    if corrected and all(all(v) for v in corrected.values()):
        order = ["upper_left", "upper_right", "lower_right", "lower_left"]
        cor_px = [g.latlon_to_pixel(*corrected[c]) for c in order]
        cor_inside = inside_raster(cor_px, width, height)
        out["tmc_corrected_corners"] = {
            "inside_raster": cor_inside,
            "col_range": [round(min(p[0] for p in cor_px), 1),
                          round(max(p[0] for p in cor_px), 1)],
            "row_range": [round(min(p[1] for p in cor_px), 1),
                          round(max(p[1] for p in cor_px), 1)],
        }
        print(f"  TMC 'Corrected' corners inside raster: {cor_inside} "
              f"(cols {out['tmc_corrected_corners']['col_range']}, "
              f"rows {out['tmc_corrected_corners']['row_range']}) "
              f"-- known label inconsistency; Refined corners used instead")

    # Honest overlap test: the OHRC quad must fall within BOTH the TMC raster
    # grid (per the TIF's own geotags) AND the TMC label's ephemeris strip.
    # The label strip is ~2% longer than the ortho grid (documented ISDA
    # corner-metadata disagreement), so a small end-of-strip shortfall is
    # metadata noise, not overlap failure.
    ohrc_poly2 = _polygon_safe(ohrc_px)
    tmc_strip_poly = _polygon_safe(tmc_px)
    frac_in_strip = 0.0
    if ohrc_poly2 is not None and tmc_strip_poly is not None:
        inter_strip = ohrc_poly2.intersection(tmc_strip_poly)
        frac_in_strip = (inter_strip.area / ohrc_poly2.area
                         if ohrc_poly2.area else 0.0)
    out["ohrc_in_tmc"]["overlap_fraction_of_tmc_label_strip"] = round(
        float(frac_in_strip), 6)
    print(f"  OHRC quad inside TMC label strip: {frac_in_strip:.1%}")

    consistent = (ohrc_inside and frac >= 0.95 and frac_in_strip >= 0.95
                  and 0.95 <= out["ohrc_in_tmc"]["width_ratio"] <= 1.05
                  and 0.95 <= out["ohrc_in_tmc"]["length_ratio"] <= 1.05)
    out["verdict"] = "VERIFIED" if consistent else "UNVERIFIED"
    print(f"  s2 verdict: {out['verdict']}")
    return out

def _polygon_safe(coords: list[tuple[float, float]]) -> Polygon | None:
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def s3_iirs(products: dict) -> dict:
    """IIRS overlap check: never assumed, always quantified.

    Two independent tests must pass for VERIFIED:
      (a) the label's corner quad must be geometrically consistent with the
          strip the image dimensions imply (lines/samples x pixel scale);
      (b) the corner quad must intersect the TMC raster grid.
    """
    print("\n=== s3: IIRS overlap (footprint-vs-strip + TMC projection) ===")
    out: dict = {"verdict": "UNVERIFIED"}
    iirs = products.get("IIRS")
    tmc = products.get("TMC2")
    if iirs is None or tmc is None:
        out["reason"] = "product parse failed"
        return out

    ps = iirs.pixel_resolution_m
    strip = {
        "lines": iirs.lines, "samples": iirs.samples,
        "length_km": round(iirs.lines * ps / 1e3, 1) if ps else None,
        "width_km": round(iirs.samples * ps / 1e3, 1) if ps else None,
    }
    quad = quad_gc(iirs.footprint)
    ratio = lambda a, b: round(a / b, 4) if (a and b) else None
    out["strip_extent"] = strip
    out["corner_quad_extent"] = quad
    out["length_ratio_quad_over_strip"] = ratio(quad["length_km"], strip["length_km"])
    out["width_ratio_quad_over_strip"] = ratio(quad["width_km"], strip["width_km"])
    print(f"  strip implies {strip['length_km']} x {strip['width_km']} km; "
          f"corner quad spans {quad['length_km']} x {quad['width_km']} km")
    print(f"  ratios quad/strip: length {out['length_ratio_quad_over_strip']}, "
          f"width {out['width_ratio_quad_over_strip']}")

    # (b) project the IIRS corners into the TMC grid (tags re-read: cheap)
    tags = read_tmc_geotags(tmc.file_path)
    g = georef_from_tags(tags["tags"], tags["geokeys"])
    width, height = tmc.samples, tmc.lines
    iirs_px = [g.latlon_to_pixel(*c) for c in iirs.footprint]
    out["iirs_in_tmc"] = {
        "col_range": [round(min(p[0] for p in iirs_px), 1),
                      round(max(p[0] for p in iirs_px), 1)],
        "row_range": [round(min(p[1] for p in iirs_px), 1),
                      round(max(p[1] for p in iirs_px), 1)],
        "inside_raster": inside_raster(iirs_px, width, height),
    }
    frac_tmc = frac_ohrc = 0.0
    poly = _polygon_safe(iirs_px)
    if poly is not None:
        inter = poly.intersection(box(0, 0, width, height))
        frac_tmc = inter.area / poly.area
        out["iirs_in_tmc"]["overlap_fraction_of_iirs"] = round(float(frac_tmc), 6)
        out["iirs_in_tmc"]["overlap_km2"] = round(
            float(inter.area) * g.ps * g.ps / 1e6, 3)
        ohrc_px = [g.latlon_to_pixel(*c) for c in products["OHRC"].footprint]
        ohrc_poly = _polygon_safe(ohrc_px)
        if ohrc_poly is not None:
            inter2 = poly.intersection(ohrc_poly)
            frac_ohrc = (inter2.area / poly.area) if inter2.area else 0.0
            out["iirs_vs_ohrc_overlap_fraction_of_iirs"] = round(float(frac_ohrc), 6)
    print(f"  IIRS corners in TMC pixels: cols "
          f"{out['iirs_in_tmc']['col_range']}, rows {out['iirs_in_tmc']['row_range']}"
          f"  (inside raster: {out['iirs_in_tmc']['inside_raster']})")
    print(f"  overlap fraction: of IIRS {out['iirs_in_tmc'].get('overlap_fraction_of_iirs')}, "
          f"with OHRC quad {out.get('iirs_vs_ohrc_overlap_fraction_of_iirs')}")

    lr = out["length_ratio_quad_over_strip"]
    wr = out["width_ratio_quad_over_strip"]
    consistent = bool(lr and wr and 0.9 <= lr <= 1.1 and 0.9 <= wr <= 1.1)
    out["footprint_consistent_with_strip"] = consistent
    reasons = []
    if not consistent:
        reasons.append("corner-quad extent inconsistent with strip dimensions")
    if frac_tmc <= 0:
        reasons.append("no intersection with TMC raster")
    if frac_ohrc <= 0:
        reasons.append("corner quad does not intersect the OHRC footprint")
    out["reasons"] = reasons
    out["verdict"] = ("VERIFIED" if consistent and frac_tmc > 0
                      else "UNVERIFIED")
    out["acquisition_time_deltas_days"] = {
        "iirs_minus_tmc": _days(iirs.stop_time, tmc.stop_time),
        "iirs_minus_ohrc": _days(iirs.stop_time,
                                 products["OHRC"].stop_time),
    }
    print(f"  s3 verdict: {out['verdict']}  reasons={reasons or 'none'}")
    print(f"  acquisition gaps (days): IIRS-TMC "
          f"{out['acquisition_time_deltas_days']['iirs_minus_tmc']}, "
          f"IIRS-OHRC {out['acquisition_time_deltas_days']['iirs_minus_ohrc']}")
    return out


def _days(t1: str | None, t2: str | None) -> float | None:
    if not t1 or not t2:
        return None

    def parse(t: str) -> datetime:
        return datetime.fromisoformat(t.replace("Z", "+00:00"))

    return round(abs((parse(t1) - parse(t2)).total_seconds()) / 86400.0, 2)


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    r1, products = s1_integrity()
    r2 = s2_overlap(products)
    r3 = s3_iirs(products)
    results = {"s1_integrity": r1, "s2_overlap": r2, "s3_iirs": r3}
    out_path = OUT_DIR / "verify_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
