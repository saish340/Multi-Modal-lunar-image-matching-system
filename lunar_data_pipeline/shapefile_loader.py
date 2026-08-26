"""Load ISRO PRADAN shapefile footprint catalogs into LunarProduct objects.

PRADAN distributes per-instrument ESRI shapefiles that index every archived
product as a footprint polygon plus an attribute table. The OHRC and TMC-2
``cal`` catalogs observed in the wild share this schema:

    PRODUCT_ID, OBS_ST_TIME, OBS_ED_TIME,
    UL_LAT, UL_LON, UR_LAT, UR_LON, BL_LAT, BL_LON, BR_LAT, BR_LON,
    DOWNLOAD, BROWSE, EMI_ANGLE, INC_ANGLE, PHA_ANGLE,
    IM_ORB_NUM, DM_ORB_NUM, L0_ID, DOP, (+ instrument-specific fields)

Rows are converted into the same :class:`LunarProduct` structure produced by
``pds4_parser.parse_label`` so ``overlap_finder`` works on both unchanged.
Catalog entries carry ``lines=samples=0`` (dimensions are unknown until the
actual product label is parsed after download) and fill the extra optional
fields (download_filename, angles, orbit numbers).

Known real-world quirks handled here (observed in ISRO's distributed files):

- The ``DOP`` date column contains year-0000 values that crash GDAL-based
  readers. All datetime-typed columns are detected via ``pyogrio.read_info``
  and excluded before reading; string time columns (OBS_ST_TIME/OBS_ED_TIME)
  are used instead.
- The polar variants (``*_sp`` / ``*_np``) contain rows with physically
  impossible coordinates (e.g. latitude 733744 degrees). Rows whose corner or
  geometry coordinates fall outside valid selenographic ranges are skipped
  with a warning.
- Product IDs can repeat across catalog variants (41 duplicates between
  ``ch2_ohr_cal`` and ``ch2_ohr_cal_sp``); deduplication keeps first seen.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pyogrio

try:  # package-style import when used as a module...
    from .pds4_parser import CORNER_ORDER, LunarProduct
except ImportError:  # ...and flat import when scripts run directly
    from pds4_parser import CORNER_ORDER, LunarProduct  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

#: Corner attribute columns per footprint corner, matching CORNER_ORDER
#: (upper_left, upper_right, lower_right, lower_left).
CORNER_COLUMNS: dict[str, tuple[str, str]] = {
    "upper_left": ("UL_LAT", "UL_LON"),
    "upper_right": ("UR_LAT", "UR_LON"),
    "lower_right": ("BR_LAT", "BR_LON"),
    "lower_left": ("BL_LAT", "BL_LON"),
}

#: Glob patterns for auto-discovering calibrated catalogs per instrument.
CATALOG_FILE_PATTERNS = {
    "OHRC": "ch2_ohr_cal*.shp",
    "TMC2": "ch2_tmc_cal*.shp",
}

_VALID_LAT_RANGE = (-90.0, 90.0)
_VALID_LON_RANGE = (-180.0, 180.0)


def _datetime_column_names(shapefile_path: Path) -> set[str]:
    """Field names typed as dates -- these must be excluded from reads.

    ISRO catalogs contain year-0000 dates that crash pyogrio/GDAL during
    field conversion ("year must be in 1..9999"), so we never read them.
    """
    info = pyogrio.read_info(str(shapefile_path))
    return {
        field
        for field, dtype in zip(info["fields"], info["dtypes"])
        if "datetime" in str(dtype).lower()
    }


def read_catalog_dataframe(shapefile_path: str | Path) -> gpd.GeoDataFrame:
    """Read a catalog shapefile into a GeoDataFrame, skipping datetime columns."""
    shapefile_path = Path(shapefile_path)
    for suffix in (".shp", ".dbf", ".shx"):
        if not shapefile_path.with_suffix(suffix).exists():
            logger.warning(
                "%s missing sidecar %s (shapefile may load incompletely)",
                shapefile_path.name,
                suffix,
            )
    datetime_columns = _datetime_column_names(shapefile_path)
    if datetime_columns:
        logger.info(
            "%s: excluding datetime column(s) %s (known year-0000 corruption)",
            shapefile_path.name,
            ", ".join(sorted(datetime_columns)),
        )
    info = pyogrio.read_info(str(shapefile_path))
    columns = [
        field for field in info["fields"] if field not in datetime_columns
    ]
    frame = gpd.read_file(shapefile_path, columns=columns)
    logger.info("Loaded %s: %d rows", shapefile_path.name, len(frame))
    return frame


def _valid_lat(lat: float) -> bool:
    return lat is not None and math.isfinite(lat) and -90.0 <= lat <= 90.0


def _valid_lon(lon: float) -> bool:
    return lon is not None and math.isfinite(lon) and -180.0 <= lon <= 180.0


def _corners_from_row(row: dict) -> list[tuple[float, float]] | None:
    """Extract footprint corners [(lat, lon)] in CORNER_ORDER from a row.

    Returns None if any corner column is missing/NaN/out of physical range.
    """
    corners: list[tuple[float, float]] = []
    for corner in CORNER_ORDER:
        lat_col, lon_col = CORNER_COLUMNS[corner]
        try:
            lat, lon = float(row[lat_col]), float(row[lon_col])
        except (KeyError, TypeError, ValueError):
            return None
        if not (_valid_lat(lat) and _valid_lon(lon)):
            return None
        corners.append((lat, lon))
    return corners


def _footprint_from_geometry(geometry) -> list[tuple[float, float]] | None:
    """Fallback footprint from a polygon's bounding box (axis-aligned).

    Only used when corner attributes are absent/unusable; loses along-track
    skew but keeps overlap detection functional.
    """
    if geometry is None or geometry.is_empty:
        return None
    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    if not (
        _valid_lat(min_lat)
        and _valid_lat(max_lat)
        and _valid_lon(min_lon)
        and _valid_lon(max_lon)
    ):
        return None
    return [
        (max_lat, min_lon),  # upper_left
        (max_lat, max_lon),  # upper_right
        (min_lat, max_lon),  # lower_right
        (min_lat, min_lon),  # lower_left
    ]


def _optional_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _is_missing(value) -> bool:
    """True for None / NaN / blank values coming out of the DBF."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return not str(value).strip()


def _text_field(row: dict, *names: str) -> str | None:
    """First non-missing value among *names* (handles truncated DBF names,
    e.g. some writers clip OBS_ST_TIME to OBS_ST_TIM)."""
    for name in names:
        value = row.get(name)
        if not _is_missing(value):
            return str(value).strip()
    return None


def _optional_int(value) -> int | None:
    value_f = _optional_float(value)
    return int(round(value_f)) if value_f is not None else None


def _row_to_product(
    row: dict, instrument: str, source_shapefile: Path, row_index: int
) -> LunarProduct | None:
    product_id = row.get("PRODUCT_ID")
    if _is_missing(product_id):
        logger.warning(
            "%s row %d: empty PRODUCT_ID; skipping", source_shapefile.name, row_index
        )
        return None
    product_id = str(product_id).strip()

    footprint = _corners_from_row(row)
    footprint_source = "corner_columns"
    if footprint is None:
        footprint = _footprint_from_geometry(row.get("geometry"))
        footprint_source = "geometry_bounds"
        if footprint is not None:
            logger.debug(
                "%s row %d (%s): corner columns unusable; using geometry bounds",
                source_shapefile.name,
                row_index,
                product_id,
            )

    if footprint is None or row.get("geometry") is None:
        logger.warning(
            "%s row %d (%s): no valid footprint (corrupt coordinates?); skipping",
            source_shapefile.name,
            row_index,
            product_id,
        )
        return None

    download_name = _text_field(row, "DOWNLOAD")
    browse_name = _text_field(row, "BROWSE")

    return LunarProduct(
        instrument=instrument,
        file_path=Path(download_name) if download_name else Path(f"{product_id}"),
        label_path=source_shapefile.resolve(),
        lines=0,  # dimensions unknown until the actual product is downloaded
        samples=0,
        datatype="",
        footprint=footprint,
        product_id=product_id,
        logical_identifier=None,
        start_time=_text_field(row, "OBS_ST_TIME", "OBS_ST_TIM"),
        stop_time=_text_field(row, "OBS_ED_TIME", "OBS_ED_TIM"),
        geometry_csv_path=None,
        download_filename=download_name,
        browse_filename=browse_name,
        incidence_angle_deg=_optional_float(row.get("INC_ANGLE")),
        emission_angle_deg=_optional_float(row.get("EMI_ANGLE")),
        phase_angle_deg=_optional_float(row.get("PHA_ANGLE")),
        imaging_orbit_number=_optional_int(row.get("IM_ORB_NUM")),
        dumping_orbit_number=_optional_int(row.get("DM_ORB_NUM")),
    )


def load_catalog_shapefile(
    shapefile_path: str | Path, instrument: str
) -> list[LunarProduct]:
    """Load one PRADAN catalog shapefile into validated LunarProduct objects.

    Rows with missing IDs, corrupt/impossible coordinates, or null geometries
    are skipped with warnings; duplicate PRODUCT_IDs within the file keep the
    first occurrence.
    """
    if instrument not in CATALOG_FILE_PATTERNS and instrument != "IIRS":
        raise ValueError(f"unsupported instrument {instrument!r}")

    frame = read_catalog_dataframe(Path(shapefile_path))
    products: list[LunarProduct] = []
    seen_ids: set[str] = set()

    records = frame.to_dict("records")
    for index, record in enumerate(records):
        product = _row_to_product(record, instrument, Path(shapefile_path), index)
        if product is None:
            continue
        if product.product_id in seen_ids:
            logger.debug(
                "Duplicate PRODUCT_ID %s in %s; keeping first occurrence",
                product.product_id,
                Path(shapefile_path).name,
            )
            continue
        seen_ids.add(product.product_id)
        products.append(product)

    logger.info(
        "%s: %d usable %s catalog entries (%d rows read)",
        Path(shapefile_path).name,
        len(products),
        instrument,
        len(records),
    )
    return products


def discover_catalog_files(root: str | Path, instrument: str) -> list[Path]:
    """Find calibrated catalog shapefiles for *instrument* under *root*."""
    pattern = CATALOG_FILE_PATTERNS[instrument]
    matches = sorted(Path(root).rglob(pattern))
    if not matches:
        logger.warning(
            "No files matching %s under %s for %s", pattern, root, instrument
        )
    return matches


def load_catalog(
    paths: Iterable[str | Path], instrument: str
) -> tuple[list[LunarProduct], dict[str, int]]:
    """Load several catalog files for one instrument, deduplicating across them.

    Returns ``(products, per_file_counts)`` where counts include skipped-row
    diagnostics keyed by filename.
    """
    products: list[LunarProduct] = []
    seen_ids: set[str] = set()
    counts: dict[str, int] = {}

    for path in paths:
        loaded = load_catalog_shapefile(path, instrument)
        kept = 0
        for product in loaded:
            if product.product_id in seen_ids:
                continue
            seen_ids.add(product.product_id)
            products.append(product)
            kept += 1
        counts[Path(path).name] = kept

    logger.info(
        "%s catalog total: %d unique products from %d file(s)",
        instrument,
        len(products),
        len(list(paths)),
    )
    return products, counts
