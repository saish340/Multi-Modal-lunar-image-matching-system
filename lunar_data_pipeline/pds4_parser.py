"""Parse Chandrayaan-2 (ISDA/PRADAN) PDS4 product labels into structured metadata.

The ISDA PDS4 profile stores almost everything Phase 0 needs inside the data
label itself (``*_d_img_*.xml``):

- Instrument identification: ``Observing_System_Component[type=Instrument]/name``
- Footprint corners (selenographic lat/lon), under
  ``Mission_Area/isda:Geometry_Parameters`` -- ``isda:Refined_Corner_Coordinates``
  preferred, falling back to ``isda:System_Level_Coordinates``
- Image dimensions: ``File_Area_Observational/Array_2D_Image/Axis_Array``
  entries named ``Line`` and ``Sample``
- Pixel datatype: ``Element_Array/data_type`` (e.g. UnsignedByte, UnsignedLSB2)

A companion per-pixel geolocation table normally lives under the product's
``geometry/`` folder (``*_g_grd_*.csv``; columns Longitude,Latitude,Pixel,Scan,
per the ``Table_Delimited`` definition in the ``*_g_grd_*.xml`` label).
:func:`find_geometry_csv` locates that file but does not parse it; Phase 0
ground truth uses corner coordinates only.

All element lookups match on *local* tag names (namespace-agnostic) so that
PDS4 namespace version bumps do not break parsing. Malformed or incomplete
labels never raise: they are skipped with a logged warning, because real
archive labels are inconsistent across product versions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

#: Canonical footprint corner order produced by this parser.
CORNER_ORDER = ("upper_left", "upper_right", "lower_right", "lower_left")

_INSTRUMENT_PATTERNS = {
    "OHRC": re.compile(r"high\s+resolution\s+camera|\bohrc\b|\bohr\b", re.IGNORECASE),
    "TMC2": re.compile(r"terrain\s+mapping\s+camera|\btmc\b", re.IGNORECASE),
    "IIRS": re.compile(
        r"imaging\s+infrared\s+spectrometer|\biirs\b|\biir\b", re.IGNORECASE
    ),
}

_FILENAME_PREFIXES = {
    "ch2_ohr": "OHRC",
    "ch2_tmc": "TMC2",
    "ch2_iir": "IIRS",
}


@dataclass
class LunarProduct:
    """Structured metadata for a single Chandrayaan-2 image product."""

    instrument: str  # "OHRC" | "TMC2" | "IIRS"
    file_path: Path  # resolved path to the image data file (.img)
    label_path: Path  # path to the PDS4 XML label this metadata came from
    lines: int
    samples: int
    datatype: str  # PDS4 Element_Array/data_type, e.g. "UnsignedByte"
    footprint: list[tuple[float, float]] = field(default_factory=list)
    # footprint: corner (lat, lon) tuples in CORNER_ORDER; empty if unknown.
    product_id: str = ""
    logical_identifier: str | None = None
    pixel_resolution_m: float | None = None
    projection: str | None = None
    area: str | None = None
    start_time: str | None = None
    stop_time: str | None = None
    geometry_csv_path: Path | None = None
    # Extra fields populated only for catalog entries loaded from PRADAN
    # shapefiles (see shapefile_loader.py); None for label-parsed products.
    download_filename: str | None = None
    browse_filename: str | None = None
    incidence_angle_deg: float | None = None
    emission_angle_deg: float | None = None
    phase_angle_deg: float | None = None
    imaging_orbit_number: int | None = None
    dumping_orbit_number: int | None = None


def _local(tag: str) -> str:
    """Strip the ``{namespace}`` prefix from an XML tag."""
    return tag.rsplit("}", 1)[-1]


def _first(root: ET.Element, name: str) -> ET.Element | None:
    """First element (anywhere in the tree) whose local tag name equals *name*."""
    for el in root.iter():
        if _local(el.tag) == name:
            return el
    return None


def _all(root: ET.Element, name: str) -> list[ET.Element]:
    return [el for el in root.iter() if _local(el.tag) == name]


def _child_text(el: ET.Element, name: str) -> str | None:
    for child in el:
        if _local(child.tag) == name:
            text = (child.text or "").strip()
            return text or None
    return None


def infer_instrument(text: str) -> str | None:
    """Map free text (instrument name, LID, filename...) to OHRC / TMC2 / IIRS."""
    for instrument, pattern in _INSTRUMENT_PATTERNS.items():
        if pattern.search(text):
            return instrument
    return None


def _instrument_from_filename(filename: str) -> str | None:
    lowered = filename.lower()
    for prefix, instrument in _FILENAME_PREFIXES.items():
        if lowered.startswith(prefix + "_"):
            return instrument
    return None


def _detect_instrument(root: ET.Element, label_path: Path) -> str | None:
    """Instrument inference: explicit component name -> LID -> filename -> full text."""
    for component in _all(root, "Observing_System_Component"):
        comp_type = (_child_text(component, "type") or "").strip().lower()
        if comp_type != "instrument":
            continue
        name = _child_text(component, "name") or ""
        instrument = infer_instrument(name)
        if instrument:
            return instrument

    lid_el = _first(root, "logical_identifier")
    if lid_el is not None and lid_el.text:
        instrument = infer_instrument(lid_el.text)
        if instrument:
            return instrument

    instrument = _instrument_from_filename(label_path.name)
    if instrument:
        return instrument

    full_text = " ".join(
        (el.text or "").strip() for el in root.iter() if el.text
    )
    return infer_instrument(full_text)


def _parse_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_corners_block(root: ET.Element) -> ET.Element | None:
    geom = _first(root, "Geometry_Parameters")
    if geom is None:
        return None
    for preferred in ("Refined_Corner_Coordinates", "System_Level_Coordinates"):
        for el in geom.iter():
            if _local(el.tag) == preferred:
                return el
    return None


def _extract_footprint(root: ET.Element) -> list[tuple[float, float]]:
    """Return corner (lat, lon) tuples in CORNER_ORDER, or [] if unavailable."""
    block = _extract_corners_block(root)
    if block is None:
        return []

    corners: list[tuple[float, float]] = []
    for corner in CORNER_ORDER:
        lat = _parse_float(_child_text(block, f"{corner}_latitude"))
        lon = _parse_float(_child_text(block, f"{corner}_longitude"))
        if lat is None or lon is None:
            logger.warning(
                "Incomplete footprint: missing %s latitude/longitude", corner
            )
            return []
        corners.append((lat, lon))
    return corners


def _extract_dimensions(root: ET.Element) -> tuple[int | None, int | None, str]:
    """Extract (lines, samples, datatype) from Array_2D_Image."""
    array = _first(root, "Array_2D_Image")
    lines: int | None = None
    samples: int | None = None
    datatype = ""

    if array is not None:
        for axis in _all(array, "Axis_Array"):
            axis_name = (_child_text(axis, "axis_name") or "").strip()
            elements = _child_text(axis, "elements")
            try:
                count = int(elements) if elements is not None else None
            except ValueError:
                count = None
            if axis_name.lower() == "line":
                lines = count
            elif axis_name.lower() == "sample":
                samples = count
        element_array = _first(array, "Element_Array")
        if element_array is not None:
            datatype = (_child_text(element_array, "data_type") or "").strip()

    return lines, samples, datatype


def _resolve_image_path(root: ET.Element, label_path: Path) -> Path | None:
    file_el = _first(root, "File")
    if file_el is None:
        return None
    file_name = _child_text(file_el, "file_name")
    if not file_name:
        return None
    return (label_path.parent / file_name).resolve()


def find_geometry_csv(label_path: Path) -> Path | None:
    """Locate the per-pixel geolocation CSV for a data label, if present.

    Uses the ISDA naming convention ``<stem with _d_img_ -> _g_grd_>.csv``
    searched under the nearest ancestor ``geometry/`` directory; falls back to
    a unique ``*_g_grd_*.csv`` in that directory tree.
    """
    stem = label_path.stem
    expected = stem.replace("_d_img_", "_g_grd_") + ".csv"

    candidates_parents: list[Path] = [label_path.parent, *label_path.parents[:6]]
    for parent in candidates_parents:
        geometry_dir = parent / "geometry"
        if not geometry_dir.is_dir():
            continue
        exact = sorted(geometry_dir.rglob(expected))
        if exact:
            return exact[0]
        fallback = sorted(geometry_dir.rglob("*_g_grd_*.csv"))
        if len(fallback) == 1:
            return fallback[0]
        if len(fallback) > 1:
            logger.warning(
                "Ambiguous geometry CSVs for %s under %s; skipping association",
                label_path.name,
                geometry_dir,
            )
            return None
    return None


def parse_label(label_path: str | Path) -> LunarProduct | None:
    """Parse one PDS4 data-product label.

    Returns a :class:`LunarProduct`, or ``None`` if the label cannot be parsed
    or lacks fields this pipeline considers essential (instrument, dimensions,
    resolvable image file). A label whose footprint is missing yields a product
    with an empty ``footprint`` list so callers can decide whether to keep it.
    """
    label_path = Path(label_path)
    try:
        root = ET.parse(label_path).getroot()
    except (ET.ParseError, OSError) as exc:
        logger.warning("Skipping unparseable label %s: %s", label_path, exc)
        return None

    if _local(root.tag) != "Product_Observational":
        logger.debug(
            "%s is not a Product_Observational label (%s); skipping",
            label_path.name,
            _local(root.tag),
        )
        return None

    instrument = _detect_instrument(root, label_path)
    if instrument is None:
        logger.warning("Could not identify instrument for %s; skipping", label_path)
        return None

    lines, samples, datatype = _extract_dimensions(root)
    if lines is None or samples is None:
        logger.warning(
            "Missing Line/Sample dimensions in %s; skipping", label_path.name
        )
        return None

    image_path = _resolve_image_path(root, label_path)
    if image_path is None:
        logger.warning("No <File><file_name> in %s; skipping", label_path.name)
        return None

    footprint = _extract_footprint(root)
    if not footprint:
        logger.warning(
            "No usable footprint corners in %s (pairing will exclude it)",
            label_path.name,
        )

    product_params = _first(root, "Product_Parameters")
    lid_el = _first(root, "logical_identifier")

    product = LunarProduct(
        instrument=instrument,
        file_path=image_path,
        label_path=label_path.resolve(),
        lines=lines,
        samples=samples,
        datatype=datatype,
        footprint=footprint,
        product_id=label_path.stem,
        logical_identifier=lid_el.text.strip() if lid_el is not None and lid_el.text else None,
        pixel_resolution_m=_parse_float(_child_text(product_params, "pixel_resolution"))
        if product_params is not None
        else None,
        projection=(_child_text(product_params, "projection") or None)
        if product_params is not None
        else None,
        area=(_child_text(product_params, "area") or None)
        if product_params is not None
        else None,
        start_time=(_child_text(root, "start_date_time")),
        stop_time=(_child_text(root, "stop_date_time")),
        geometry_csv_path=find_geometry_csv(label_path),
    )

    if product.file_path.exists():
        logger.debug("Parsed %s [%s]", product.product_id, instrument)
    else:
        logger.warning(
            "Image file referenced by %s not found on disk: %s",
            label_path.name,
            product.file_path,
        )

    return product
