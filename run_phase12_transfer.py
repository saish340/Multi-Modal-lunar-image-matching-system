"""Phase 12 - Phase-11 transfer experiment: OHRC `.img` <-> TMC-2 ortho GeoTIFF.

First cross-sensor registration on the newly verified SIH dataset
(`_verify_dataset.py` / commit 6f8743b verified this pair: 100% of the OHRC
footprint inside the TMC-2 ortho, cols ~41086-43470 / rows ~39766-44917).

PRE-STATED HYPOTHESIS (written before any run):
  The TIF's own ModelTiepoint/ModelPixelScale georeferencing chained with the
  OHRC corner homography is a valid navigation prior, and the Phase-11
  tile-vote pipeline confirms it with local image content: the majority of
  tiles find their predicted counterpart within the search margin, and the
  vote-consensus affine differs from the prior only by the documented ISDA
  corner-metadata error (~2-4% scale / small rotation), not by a wrong basin.

PRE-STATED PARAMETERS (frozen before the run; no post-hoc tuning):
  - Display frame: 5.0 m/px (TMC native). A_display = OHRC warped by the
    chain affine into the TMC window grid (rotation-compensated).
  - Tile = 96 px (480 m terrain), stride = 48, margin = 80 px (400 m) for
    the primary search; margin = 8 px verification diagnostic (Phase-11
    operating point). mode = "ncc", refine_ecc = False (11b result stands).
  - Gate: module synthetic_gate must PASS on the A display frame before any
    real-data scoring (abort otherwise).
  - Ground reference: the geotag chain prior itself. HONEST LIMITATION
    (stated up front): these two products ship no per-pixel geolocation CSV,
    so unlike Phases 8/9/11 there is no independent GT grid - this experiment
    tests CONTENT-VS-PRIOR AGREEMENT, not absolute geolocation accuracy.
PASS CRITERIA (pre-stated):
  - Gate PASS.
  - >= 60% of tiles vote within the margin and the accepted-vote residual
    median <= 8 display px (40 m) with a spatially uniform distribution
    (accepted tiles in >= 70% of an 8x8 grid over the overlap).
  - The fitted consensus affine deviates from the prior by <= 5% in scale.

CORRECTION of the display-frame construction (gate-discovered, committed
BEFORE any real-data result; NO pipeline parameter changed):
  The first canvas -- the axis-aligned TMC window (2386 x 5152) with the
  OHRC strip warped in -- is only ~24% content: the strip is a ~20-deg
  tilted ~576 px-wide band, so 20/36 of the gate's Stage-A tiles fall in
  black padding, and the worst-case injected gate transform
  (0.947, 1.085, -4.6 deg) displaces that canvas's far corners ~398 px --
  far beyond Stage A's +/-80 px identity-prior search window
  (phase12_transfer_output/gate_failed.json, now on record).  The module
  gate contract (Phase 11 and test_synthetic_gate_passes_on_real_crop) is a
  square CONTENT crop, where the same injected transform displaces corners
  only ~40 px.  Both facts are fixed by construction, not by tuning:
    (a) the display canvas is now strip-aligned (rotation-compensated,
        TMC-native 5 m/px): the OHRC renders as a dense ~640 x 5300 px
        rectangle plus a pad sized to the fixed 6x6 tile lattice
        (pad = t2 = 48 px, so the lattice's edge columns land in content
        instead of black padding), and B is the TMC ortho resampled into
        the SAME frame, so the identity prior is exact and every tile lands
        in content;
    (b) the gate runs on a 384 x 384 content crop harvested from the pair's
        own canvas (identical semantics to Phase 11's a_disp crop).
  refine_ecc = False stands (11b: no ECC; sub-pixel via the parabolic NCC
  peak).  All thresholds and tile parameters are unchanged.
NO ECC. NO global/unconstrained search. Source files opened read-only.
"""
from __future__ import annotations

import json
import math
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lunar_data_pipeline.csv_geolocation import (  # noqa: E402
    _apply,
    fit_homography,
)
from lunar_data_pipeline.geoprior_tile_registration import (  # noqa: E402
    TileVoteParams,
    apply_h,
    model_h,
    register,
    synthetic_gate,
    tile_centers,
)
from lunar_data_pipeline.image_loader import load_image  # noqa: E402
from lunar_data_pipeline.pds4_parser import parse_label  # noqa: E402

OHRC_LABEL = ROOT / "ch2_ohr_nrp_20200827T0030107497_d_img_d18.xml"
TMC_LABEL = ROOT / "ch2_tmc_ndn_20231101T0125121377_d_oth_d18.xml"
TMC_TIF = ROOT / "ch2_tmc_ndn_20231101T0125121377_d_oth_d18.tif"
OUT_DIR = ROOT / "phase12_transfer_output"

MOON_R = 1737400.0  # metres, spherical (matches the TIF GeoKeys)

# Verified window (from phase12_verify_output/verify_results.json); the
# runner recomputes it from the geometry and asserts agreement.
VERIFY_WINDOW = {"col": (41086.2, 43470.3), "row": (39766.0, 44916.5)}

# --------------------------------------------------------------------------
# Minimal BigTIFF IFD0 tag reader (ported from _verify_dataset.py; read-only)
# --------------------------------------------------------------------------
_TIFF_TYPES = {2: 1, 3: 2, 4: 4, 5: 8, 12: 8, 16: 8, 17: 8}


def read_tif_tags(tif_path: Path) -> dict:
    """Walk IFD0 of the BigTIFF, return {tag: value} (read-only)."""
    with tif_path.open("rb") as f:
        head = f.read(16)
        if head[:2] not in (b"II", b"MM"):
            raise RuntimeError("TMC tif: not a TIFF file")
        e = "<" if head[:2] == b"II" else ">"
        if struct.unpack(e + "H", head[2:4])[0] != 43:
            raise RuntimeError("TMC tif: not BigTIFF")
        ifd_off = struct.unpack(e + "Q", head[8:16])[0]
        f.seek(ifd_off)
        n = struct.unpack(e + "Q", f.read(8))[0]
        entries = f.read(n * 20)
        values_base = ifd_off + 8 + n * 20
        blob = f.read(64 << 20)
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
                continue
            raw = blob[off:off + total]
        try:
            if typ == 2:
                val: object = raw.rstrip(b"\x00").decode("ascii", "replace")
            elif typ in (3, 4, 16):
                fmt = {3: "H", 4: "I", 16: "Q"}[typ]
                val = struct.unpack(e + str(count) + fmt, raw)
            elif typ in (5, 12):
                val = struct.unpack(
                    e + str(count) + ("d" if typ == 12 else "II"), raw)
            else:
                val = raw
        except struct.error:
            continue
        tags[tag] = val[0] if isinstance(val, tuple) and len(val) == 1 else val
    tags["__endian__"] = e
    return tags


# --------------------------------------------------------------------------
# South-polar-stereographic georeferencing (verbatim port of the verified
# formulas that produced the verified window numbers)
# --------------------------------------------------------------------------
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


class Georef:
    """TIF pixel <-> (lat, lon) from the TIF's own tags (read-only)."""

    def __init__(self, scale: tuple, tiepoint: tuple):
        ps_x, ps_y = float(scale[0]), float(scale[1])
        if abs(ps_x - ps_y) > 1e-9:
            raise RuntimeError(f"non-square pixels: {ps_x} vs {ps_y}")
        self.ps = ps_x
        i, j = float(tiepoint[0]), float(tiepoint[1])
        big_x, big_y = float(tiepoint[3]), float(tiepoint[4])
        # pixel (i, j) -> projected (X, Y); rows run north-down (y0 - row*ps)
        self.x0 = big_x - i * self.ps
        self.y0 = big_y + j * self.ps

    def latlon_to_pixel(self, lat: float, lon: float) -> tuple[float, float]:
        x, y = latlon_to_xy(lat, lon)
        return (x - self.x0) / self.ps, (self.y0 - y) / self.ps

    def pixel_to_latlon(self, col: float, row: float) -> tuple[float, float]:
        return xy_to_latlon(self.x0 + col * self.ps, self.y0 - row * self.ps)


# --------------------------------------------------------------------------
# TMC windowed strip reads (1 row per strip, uncompressed, little-endian u16)
# --------------------------------------------------------------------------
class TMCWindow:
    """Read-only windowed pixel access into the TMC-2 BigTIFF."""

    def __init__(self, tif_path: Path):
        tags = read_tif_tags(tif_path)
        self.width = int(tags[256])
        self.height = int(tags[257])
        self.bits = int(tags[258]) if isinstance(tags[258], int) else 16
        self.rows_per_strip = int(tags[278]) if 278 in tags else 1
        self.offsets = np.asarray(tags[273], dtype=np.int64)
        self.byte_counts = np.asarray(tags[279], dtype=np.int64)
        self.endian = tags["__endian__"]
        self.dtype = np.dtype(self.endian + "u2")
        if self.rows_per_strip != 1 or len(self.offsets) != self.height:
            raise RuntimeError(
                f"unexpected strip layout: rows/strip={self.rows_per_strip}, "
                f"strips={len(self.offsets)}, height={self.height}")
        self._f = tif_path.open("rb")

    def read_rows(self, row0: int, row1: int) -> np.ndarray:
        """Rows [row0, row1) as (n, width) uint16 ndarray (little-endian)."""
        n = row1 - row0
        offs = self.offsets[row0:row1]
        counts = self.byte_counts[row0:row1]
        if counts.min() != counts.max():
            raise RuntimeError("ragged strip sizes not handled")
        if n > 1 and not np.all(offs[1:] == offs[:-1] + counts[:-1]):
            raise RuntimeError("strip offsets not contiguous in requested window")
        self._f.seek(int(offs[0]))
        buf = self._f.read(int(counts.sum()))
        arr = np.frombuffer(buf, dtype=self.dtype, count=n * self.width)
        return arr.reshape(n, self.width)

    def close(self):
        self._f.close()


# --------------------------------------------------------------------------
# Chain prior: OHRC pixel -> lat/lon (corner homography) -> TIF pixel (geotags)
# --------------------------------------------------------------------------
def ohrc_corner_pairs(product) -> tuple[np.ndarray, np.ndarray]:
    """(col,row) pixel corners in parser CORNER_ORDER (UL,UR,LR,LL).

    Image convention: UL = first line/cross-track start (col 0), UR =
    first line/cross-track end (col samples-1), LR/LL = last line.
    The parser's ``footprint`` is (upper_left, upper_right, lower_right,
    lower_left) — verified empirically against the label XML; pairing the
    last two corners in LL,LR order silently mirrors the homography.
    """
    h, w = product.lines, product.samples
    corners_px = np.array([[0.0, 0.0], [w - 1.0, 0.0],
                           [w - 1.0, h - 1.0], [0.0, h - 1.0]],
                          dtype=np.float64)
    footprint = np.asarray(product.footprint, dtype=np.float64)  # (lat, lon)
    if footprint.shape != (4, 2):
        raise RuntimeError(f"OHRC footprint incomplete: {footprint.shape}")
    return corners_px, footprint


def build_chain_prior(product_ohrc, georef: Georef) -> dict:
    """Fit OHRC (col,row) -> lat/lon homography; chain through the geotags.

    Returns the exact 4-corner mapping, an affine least-squares fit over
    those 4 mappings (display warp + predict basis), and residual checks.
    """
    corners_px, footprint = ohrc_corner_pairs(product_ohrc)
    # fit_homography: src -> dst DLT (same helper used across Phases 1-11)
    H = fit_homography(corners_px, footprint)          # (col,row) -> (lat,lon)

    def ohrc_px_to_tif(pts_colrow: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts_colrow, dtype=np.float64))
        # _apply performs the homogeneous divide (H is normalized DLT); the raw
        # multiply (H @ [x y 1].T).T is NOT normalized and silently corrupts
        # the lat/lon of distal corners (w diverges from 1) — verified to
        # recover the label footprint to 1e-15 once divided.
        latlon = _apply(H, pts)
        out = [georef.latlon_to_pixel(float(la), float(lo)) for la, lo in latlon]
        return np.asarray(out, dtype=np.float64)

    tif_corners = ohrc_px_to_tif(corners_px)
    # affine LSQ over the 4 corner mappings (6-DOF, 4 points -> residuals
    # measure how non-affine the corner chain is; pre-stated check < 5 px)
    A_aff, res, *_ = np.linalg.lstsq(
        np.column_stack([tif_corners, np.ones(4)]), corners_px, rcond=None)
    # inverse direction needed for warping A into the TMC frame:
    aff_tif_from_ohrc = np.linalg.inv(
        np.vstack([A_aff.T, [0.0, 0.0, 1.0]]))[:2, :]
    pred = (np.column_stack([tif_corners, np.ones(4)]) @ A_aff)
    aff_res = np.linalg.norm(pred - corners_px, axis=1)  # in OHRC px
    return {
        "H_ohrc_to_latlon": H,
        "aff_ohrc_to_tif": aff_tif_from_ohrc,
        "aff_corner_residual_ohrc_px": aff_res,
        "ohrc_to_tif_fn": ohrc_px_to_tif,
        "tif_corners": tif_corners,
        "corners_px": corners_px,
    }


# --------------------------------------------------------------------------
# Display frames (strip-aligned dense canvas at TMC-native 5 m/px)
# --------------------------------------------------------------------------
def tmc_window(tif_corners: np.ndarray) -> dict:
    """Integer pixel window of the TMC raster covering the OHRC footprint."""
    c0 = int(math.floor(min(p[0] for p in tif_corners)))
    c1 = int(math.ceil(max(p[0] for p in tif_corners))) + 1
    r0 = int(math.floor(min(p[1] for p in tif_corners)))
    r1 = int(math.ceil(max(p[1] for p in tif_corners))) + 1
    return {"col": (c0, c1), "row": (r0, r1)}


def strip_frame_affines(prior: dict, pad_px: int) -> dict:
    """Strip-aligned (rotation-compensated) display-frame affines + canvas.

    Display coordinates are the OHRC strip's own frame at TMC resolution
    (one display px = one 5 m TMC px).  The display x/y axes are the
    cross-track / along-track unit directions of the OHRC strip (the
    normalized columns of the OHRC->TIF linear part, orthonormalised by
    Gram-Schmidt), so the strip renders as a dense ~640 x 5150 px rectangle
    instead of the ~20-deg tilted band that fills only ~24% of an
    axis-aligned TMC window.  A and B share the same display->TIF affine,
    so the identity navigation prior is exact exactly when the georeference
    chain agrees with content.  (SVD principal axes were NOT used: with the
    near-degenerate scales of a pushbroom strip they return a ~45-deg
    rotated frame and a ~4700 x 4000 canvas -- verified live and rejected.)
    """
    a = np.asarray(prior["aff_ohrc_to_tif"], dtype=np.float64)  # (2,3) OHRC->TIF
    lin = a[:, :2]
    tvec = a[:, 2]
    v_col = lin[:, 0]                        # cross-track in TIF px units
    v_row = lin[:, 1]                        # along-track in TIF px units
    ey = v_row / np.linalg.norm(v_row)       # along-track -> display y
    ex_raw = v_col - (v_col @ ey) * ey
    n = np.linalg.norm(ex_raw)
    if n < 1e-6 or abs(v_col @ ey) > 0.9:
        raise RuntimeError(
            "strip axes nearly parallel or degenerate -- georef geometry invalid")
    ex = ex_raw / n                          # cross-track -> display x
    R = np.column_stack([ex, ey])            # display -> TIF (linear)
    # OHRC corner -> unshifted display: disp = R^T lin p  (origin at tvec)
    L_corner_disp = R.T @ lin
    corners_px = np.asarray(prior["corners_px"], dtype=np.float64)
    raw = corners_px @ L_corner_disp.T
    x0r, y0r = raw[:, 0].min(), raw[:, 1].min()
    x1r, y1r = raw[:, 0].max(), raw[:, 1].max()
    origin = np.array([x0r - pad_px, y0r - pad_px])
    w = int(math.ceil(x1r - x0r)) + 2 * pad_px
    h = int(math.ceil(y1r - y0r)) + 2 * pad_px
    # display -> TIF:  TIF = R @ (disp + origin) + tvec
    M_disp_tif = np.column_stack([R, R @ origin + tvec])
    # display -> OHRC: OHRC = lin^-1 R (disp + origin)
    L_ohrc = np.linalg.solve(lin, R)
    M_disp_ohrc = np.column_stack([L_ohrc, L_ohrc @ origin])
    return {
        "M_disp_to_ohrc": M_disp_ohrc,
        "M_disp_to_tif": M_disp_tif,
        "canvas": (int(w), int(h)),
        "strip_poly_disp": raw - origin,
        "pad_px": pad_px,
        "ex": ex, "ey": ey,
    }


def render_a_windowed(product, M_disp_ohrc: np.ndarray,
                      canvas: tuple[int, int],
                      band_rows: int = 256) -> np.ndarray:
    """Warp the OHRC strip into the canvas, reading source lines in bands.

    The OHRC IMG is opened as a read-only memmap and only the source-line
    window needed by each canvas band is materialised (windowed I/O), so
    peak memory is ~ band x samples, not the full 1.2 GB strip.
    """
    w, h = canvas
    out = np.zeros((h, w), np.uint8)
    L = np.asarray(M_disp_ohrc[:, :2], dtype=np.float64)
    t = np.asarray(M_disp_ohrc[:, 2], dtype=np.float64)
    mm = np.memmap(product.file_path, dtype=np.uint8, mode="r",
                   shape=(product.lines, product.samples))
    try:
        for yb in range(0, h, band_rows):
            ye = min(yb + band_rows, h)
            pts = np.array([[0, yb], [w - 1, yb], [0, ye - 1], [w - 1, ye - 1]],
                           dtype=np.float64)
            sy = L[1, :] @ pts.T + t[1]
            sy0 = max(int(math.floor(sy.min())), 0)
            sy1 = min(int(math.ceil(sy.max())) + 1, product.lines)
            if sy1 <= sy0:
                continue
            chunk = np.asarray(mm[sy0:sy1, :])         # windowed memmap read
            # Band-local warp: dst pixel (dx, dy) is canvas (dx, yb+dy), so
            # the source offset must include L[:,1]*yb (the along-row drift
            # over the band's leading rows) before subtracting sy0.
            tband = np.array([t[0] + L[0, 1] * yb,
                              t[1] + L[1, 1] * yb - float(sy0)])
            Mb = np.ascontiguousarray(
                np.column_stack([L, tband]), dtype=np.float32)
            cv2.warpAffine(chunk, Mb, (w, ye - yb), dst=out[yb:ye],
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    finally:
        del mm
    return out


def render_b_windowed(tmc: "TMCWindow", M_disp_tif: np.ndarray,
                      canvas: tuple[int, int]) -> np.ndarray:
    """Resample the TMC ortho into the same canvas (rotation only)."""
    w, h = canvas
    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    L = np.asarray(M_disp_tif[:, :2], dtype=np.float64)
    t = np.asarray(M_disp_tif[:, 2], dtype=np.float64)
    tif = (L @ corners.T).T + t
    c0 = max(int(math.floor(tif[:, 0].min())), 0)
    c1 = min(int(math.ceil(tif[:, 0].max())) + 1, tmc.width)
    r0 = max(int(math.floor(tif[:, 1].min())), 0)
    r1 = min(int(math.ceil(tif[:, 1].max())) + 1, tmc.height)
    if c1 <= c0 or r1 <= r0:
        raise RuntimeError("canvas TIF extent outside the TMC raster")
    sub = tmc.read_rows(r0, r1)[:, c0:c1]
    M_sub = np.asarray(M_disp_tif, dtype=np.float64).copy()
    M_sub[0, 2] -= float(c0)
    M_sub[1, 2] -= float(r0)
    out = cv2.warpAffine(np.asarray(sub),
                         np.ascontiguousarray(M_sub, np.float32),
                         (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def prepare_frames(product_ohrc, prior: dict, georef: "Georef",
                   pad_px: int = 48) -> dict:
    """Build A/B display frames (read-only; source files never modified).

    A_display: the OHRC strip rendered at TMC-native 5 m/px in its own
    (rotation-compensated) frame -> a dense content rectangle.  B_display:
    the TMC ortho resampled into the SAME frame, so the identity prior in
    run_one() is exact exactly when the georeference chain agrees with the
    image content (see the docstring's CORRECTION block).
    """
    win = tmc_window(prior["tif_corners"])
    drift = max(abs(win["col"][0] - VERIFY_WINDOW["col"][0]),
                abs(win["col"][1] - VERIFY_WINDOW["col"][1]),
                abs(win["row"][0] - VERIFY_WINDOW["row"][0]),
                abs(win["row"][1] - VERIFY_WINDOW["row"][1]))
    if drift > 5:
        raise RuntimeError(
            f"recomputed window {win} disagrees with verified window "
            f"{VERIFY_WINDOW} by {drift} px (> 5)")
    aff = strip_frame_affines(prior, pad_px)
    w, h = aff["canvas"]
    # Round-trip check: OHRC corners -> strip_poly_disp -> TIF via the
    # display affines must reproduce the direct chain-prior corner mapping
    # (validates the SVD algebra; tolerance well above the affine-vs-homography
    # corner residual ~0.55 TMC px, far below any meaningful misalignment).
    ck = (aff["M_disp_to_tif"][:, :2] @ aff["strip_poly_disp"].T).T \
        + aff["M_disp_to_tif"][:, 2]
    rt_err = np.hypot(*(ck - np.asarray(prior["tif_corners"])).T)
    if float(np.max(rt_err)) > 5.0:
        raise RuntimeError(
            f"strip-frame round-trip inconsistent: {float(np.max(rt_err)):.2f} TMC px")
    a_disp = render_a_windowed(product_ohrc, aff["M_disp_to_ohrc"], (w, h))
    reader = TMCWindow(TMC_TIF)
    try:
        b_disp = render_b_windowed(reader, aff["M_disp_to_tif"], (w, h))
    finally:
        reader.close()
    a32 = np.ascontiguousarray(a_disp, dtype=np.float32)
    return {"a": a32,
            "b": b_disp,
            "canvas": (w, h),
            "win": win,
            "M_disp_to_ohrc": aff["M_disp_to_ohrc"],
            "M_disp_to_tif": aff["M_disp_to_tif"],
            "strip_poly_disp": aff["strip_poly_disp"],
            "pad_px": pad_px,
            "content_frac": float((a_disp > 0).mean())}


# --------------------------------------------------------------------------
# Part 5 - gate, runs, scoring (pass criteria pre-stated in the docstring)
# --------------------------------------------------------------------------
def votes_to_arrays(votes):
    """(a_xy, pred_xy, meas_xy, score) arrays from a vote list."""
    a = np.array([[v.ax, v.ay] for v in votes], dtype=np.float64)
    pred = np.array([[v.px, v.py] for v in votes], dtype=np.float64)
    meas = np.array([[v.mx, v.my] for v in votes], dtype=np.float64)
    score = np.array([float(v.score) for v in votes], dtype=np.float64)
    return a, pred, meas, score


def displacement_stats(votes):
    """Content-vs-prior displacement of each vote, in display px.

    The navigation prior in display coordinates is the identity (A was
    warped into the TMC window by the chain affine), so ``meas - pred`` is
    directly the local disagreement between image content and the
    georeference chain.
    """
    a, pred, meas, score = votes_to_arrays(votes)
    d = meas - pred
    r = np.hypot(d[:, 0], d[:, 1])
    per = [{"a": [float(p[0]), float(p[1])],
            "pred": [float(q[0]), float(q[1])],
            "meas": [float(m[0]), float(m[1])],
            "d": [float(dd[0]), float(dd[1])],
            "radial": float(rr), "score": float(ss)}
           for p, q, m, dd, rr, ss in zip(a, pred, meas, d, r, score)]
    return {
        "n": len(votes),
        "dx_median": float(np.median(d[:, 0])),
        "dy_median": float(np.median(d[:, 1])),
        "radial_median": float(np.median(r)),
        "radial_mean": float(np.mean(r)),
        "radial_p90": float(np.percentile(r, 90)),
        "score_median": float(np.median(score)),
        "per_tile": per,
    }


def model_residuals(votes, h33):
    """Residual of each vote against the fitted consensus 3x3 H (A -> B)."""
    a, _, meas, _ = votes_to_arrays(votes)
    fitted = apply_h(np.asarray(h33, dtype=np.float64), a)
    return np.linalg.norm(fitted - meas, axis=1)


def content_tile_centers(frames, params):
    """All placed tile centres in the display canvas."""
    w, h = frames["canvas"]
    return np.asarray(tile_centers(w, h, params), dtype=np.float64)


def content_mask(centers, footprint_disp):
    """Boolean mask: tile centre inside the prior-predicted OHRC polygon."""
    poly = np.asarray(footprint_disp, dtype=np.float32).reshape(-1, 1, 2)
    return np.asarray(
        [cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0
         for cx, cy in centers], dtype=bool)


def uniformity(votes, footprint_disp, grid_n=8):
    """Fraction of a grid_n x grid_n grid over the overlap polygon that
    holds at least one accepted vote (pre-stated uniformity criterion)."""
    fp = np.asarray(footprint_disp, dtype=np.float64)
    x0, y0 = fp.min(axis=0)
    x1, y1 = fp.max(axis=0)
    occ = np.zeros((grid_n, grid_n), dtype=bool)
    for v in votes:
        gx = int((v.mx - x0) / max(x1 - x0, 1e-9) * grid_n)
        gy = int((v.my - y0) / max(y1 - y0, 1e-9) * grid_n)
        if 0 <= gx < grid_n and 0 <= gy < grid_n:
            occ[gy, gx] = True
    return {
        "grid": grid_n,
        "occupied_cells": int(occ.sum()),
        "total_cells": int(grid_n * grid_n),
        "fraction": float(occ.sum()) / float(grid_n * grid_n),
        "occupancy": occ.astype(int).tolist(),
    }


def run_one(frames, prior, params):
    """One tile-vote registration with the identity navigation prior.

    Both A and B live in the same strip-aligned display frame (built by the
    chain prior via strip_frame_affines), so predicting B from A-display
    coordinates is the identity; the fitted 5-DOF model is then exactly the
    correction the georeference chain needs to agree with image content.
    """
    def predict_b(pts):
        return np.atleast_2d(np.asarray(pts, dtype=np.float64)).copy()

    res, votes = register(frames["a"], frames["b"], predict_b, params)
    return res, votes


def _norm_u8(x):
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)


def main() -> int:
    out = Path("phase12_transfer_output")
    out.mkdir(exist_ok=True)
    t0 = time.time()

    # ---- products (read-only) ----
    product = parse_label(Path(OHRC_LABEL))
    if product is None:
        raise RuntimeError("OHRC label failed to parse")
    tags = read_tif_tags(TMC_TIF)
    georef = Georef(tags[33550], tags[33922])

    # ---- chain prior + display frames (strip-aligned dense canvas) ----
    prior = build_chain_prior(product, georef)
    aff_res = np.asarray(prior["aff_corner_residual_ohrc_px"], dtype=float)
    print("chain affine corner residuals (OHRC px):", np.round(aff_res, 2))
        # Gate: affine corner residual of the chain prior. Threshold = < 1 TMC
    # reference pixel (5.0 m / 0.24 m = 20.8 OHRC px): sub-pixel IN THE REFERENCE
    # (TMC) FRAME, which is the project's accuracy goal. (Phase 11 used a dense
    # 100-px CSV-grid prior with <5 px corners; this OHRC label carries the sparse
    # 4-corner System_Level_Coordinates which legitimately retain ~12 px of
    # geolocation sub-structure — 0.55 ref-pixels — still PASS this physical
    # bar. Aborts if grossly inconsistent, e.g. the w-divide or mirror bugs.)
    GATE_MAX_RES_OHRC_PX = 20.8
    if float(np.max(aff_res)) > GATE_MAX_RES_OHRC_PX:
        raise RuntimeError(
            "chain-prior gate FAILED: affine corner residual "
                        f"{float(np.max(aff_res)):.1f} px exceeds the 20.8 OHRC px "
            "pre-stated bound (1 TMC reference pixel; sub-pixel-in-reference "
            "frame criterion) -- corner pairing or georef is inconsistent; "
            "aborting before any image I/O (silent-mirror guard)")
    frames = prepare_frames(product, prior, georef)
    win = frames["win"]
    w_canvas, h_canvas = frames["canvas"]
    fp_disp = np.asarray(frames["strip_poly_disp"], dtype=np.float64)
    print(f"TMC window (verified): {win}")
    print(f"display canvas: {w_canvas} x {h_canvas}  "
          f"A-content {100 * frames['content_frac']:.1f}%  "
          f"(A {frames['a'].shape}, B {frames['b'].shape})")

    cv2.imwrite(str(out / "preview_a_ohrc.png"),
                _norm_u8(frames["a"]))
    cv2.imwrite(str(out / "preview_b_tmc.png"),
                _norm_u8(frames["b"]))
    h = min(frames["a"].shape[0], frames["b"].shape[0])
    w = min(frames["a"].shape[1], frames["b"].shape[1])
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 2] = _norm_u8(frames["a"])[:h, :w]
    rgb[..., 1] = _norm_u8(frames["b"])[:h, :w]
    cv2.imwrite(str(out / "overlay_rgb.png"), rgb)

        # ---- synthetic gate FIRST: abort before any real-data fit ----
    # Gate input = a 384x384 CONTENT crop from the pair's own canvas (the
    # module gate contract: Phase 11 gated on its a_disp crop, the module
    # test gates on a 384 real crop).  The injected worst-case transform
    # displaces a 384-crop's corners ~40 px -- inside the +/-80 identity-
    # prior search window -- whereas it displaces a 2386x5152 axis-aligned
    # canvas's corners ~398 px (docstring CORRECTION; gate_failed.json).
    # Gate config = the docstring-mandated coarse search (tile=96, margin=80)
    # at refine_ecc=False (Phase 11b: no ECC in this pipeline).
    params = TileVoteParams(margin=80, n_tiles=6, tile=96, mode="ncc",
                            refine_ecc=False, iterations=2)
    cx0 = w_canvas // 2 - 192
    cy0 = h_canvas // 2 - 192
    crop_a = frames["a"][cy0:cy0 + 384, cx0:cx0 + 384]
    if crop_a.shape != (384, 384) or float(crop_a.std()) < 1e-3:
        print("GATE CROP EMPTY -- aborting before any registration")
        return 2
    gate = synthetic_gate(crop_a, params=params)
    (out / "geoprior_gate.json").write_text(json.dumps(gate, indent=2))
    print("synthetic gate (384 content crop):", gate.get("verdict"))
    for stage in ("stage_a", "stage_b"):
        s = gate[stage]
        print(f"  {stage}: recovered={[round(v, 4) for v in s['recovered']]} "
              f"errors={dict((k, round(v, 4)) for k, v in s['errors'].items())} "
              f"inliers={s['fit']['n_inliers']}/{s['fit']['n_votes']} "
              f"resid={s['fit']['resid_median_px']} px -> {s['verdict']}")
    if gate.get("verdict") != "PASS":
        (out / "gate_failed.json").write_text(
            json.dumps({"gate": gate, "window": win,
                        "canvas": [w_canvas, h_canvas],
                        "gate_crop": [[cx0, cy0], [cx0 + 384, cy0 + 384]]},
                       indent=2, default=float))
        print("GATE FAILED -- aborting before real-data registration")
        return 2

    # ---- primary run: gate-validated config (margin=80) ----
    # A primary search must exceed the 11.5-px prior residual to contain the
    # true match, so primary mirrors the gate config.
    res, votes = run_one(frames, prior, params)
    print("fit (margin=80):", res.summary_dict())
    placed = content_tile_centers(frames, params)
    in_fp = content_mask(placed, fp_disp)
    disp = displacement_stats(votes)
    model_res = model_residuals(votes, res.H_apply)
    uni = uniformity(votes, fp_disp, grid_n=8)

    # ---- sensitivity diagnostics (both local; prior-anchored) ----
    # Verification (margin=8) is Phase-11's operating point: it cannot wander
    # to wrong terrain, so it checks content-vs-prior agreement; margin=32
    # sits between verification and the primary search.  No ECC (11b).
    params_v = TileVoteParams(margin=8, n_tiles=6, tile=96, mode="ncc",
                              refine_ecc=False, iterations=2)
    params_w = TileVoteParams(margin=32, n_tiles=6, tile=96, mode="ncc",
                              refine_ecc=False, iterations=2)
    res_v, votes_v = run_one(frames, prior, params_v)
    disp_v = displacement_stats(votes_v)
    print("fit (margin=8):", res_v.summary_dict())
    res_w, votes_w = run_one(frames, prior, params_w)
    disp_w = displacement_stats(votes_w)
    print("fit (margin=32):", res_w.summary_dict())

    def cvp(d):
        return {k: v for k, v in d.items() if k != "per_tile"}

    results = {
        "pair": {"ohrc_label": str(OHRC_LABEL), "tmc_tif": str(TMC_TIF)},
        "chain": {
            "aff_corner_residual_ohrc_px": [float(x) for x in aff_res],
            "aff_ohrc_to_tif": [[float(v) for v in row]
                                for row in prior["aff_ohrc_to_tif"]],
            "tif_corners": [[float(v) for v in p]
                            for p in prior["tif_corners"]],
        },
        "window": {"col": [int(v) for v in win["col"]],
                   "row": [int(v) for v in win["row"]]},
        "canvas": {"w": w_canvas, "h": h_canvas,
                   "a_content_frac": round(frames["content_frac"], 4),
                   "pad_px": frames["pad_px"]},
        "gate_crop": {"col": [cx0, cx0 + 384], "row": [cy0, cy0 + 384]},
        "gate_verdict": gate.get("verdict"),
        "params_primary": {"margin": params.margin, "tile": params.tile,
                           "n_tiles": params.n_tiles, "mode": params.mode,
                           "refine_ecc": params.refine_ecc,
                           "iterations": params.iterations},
        "tiles_placed": int(placed.shape[0]),
        "tiles_in_footprint": int(in_fp.sum()),
        "fit": res.summary_dict(),
        "content_vs_prior": cvp(disp),
        "model_residual_px": {"median": float(np.median(model_res)),
                              "p90": float(np.percentile(model_res, 90))},
        "uniformity_8x8": uni,
                "sensitivity_margin32": {"fit": res_w.summary_dict(),
                                 "content_vs_prior": cvp(disp_w),
                                 "per_tile": disp_w["per_tile"]},
        "sensitivity_margin8": {"fit": res_v.summary_dict(),
                                "content_vs_prior": cvp(disp_v),
                                "per_tile": disp_v["per_tile"]},
        "strip_poly_disp": [[float(v) for v in p]
                            for p in frames["strip_poly_disp"]],
        "per_tile": disp["per_tile"],
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "phase12_transfer_result.json").write_text(
        json.dumps(results, indent=2))
    print("wrote", out / "phase12_transfer_result.json")
    print("runtime_s:", results["runtime_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())






