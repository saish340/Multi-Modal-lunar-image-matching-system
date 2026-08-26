"""Load actual pixel data from downloaded Chandrayaan-2 PDS4 products.

The ``data/`` folder of each product contains a raw binary array (no
container format), described by the PDS4 label fields that
``pds4_parser.parse_label`` already extracted:

- ``datatype``: PDS4 element type, e.g. ``UnsignedByte`` (OHRC calibrated),
  ``UnsignedLSB2`` (TMC-2 calibrated)
- shape: ``lines`` x ``samples`` from the Axis_Array entries
- byte order: suffix (LSB/MSB); calibrated OHRC/TMC products observed so far
  have zero-offset, band-sequential, single-band arrays

Loading uses :class:`numpy.memmap` so callers can crop huge strips (e.g. a
223088x4000 TMC-2 product is 1.8 GB) without reading every byte.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

try:  # package-style import when used as a module...
    from .pds4_parser import LunarProduct
except ImportError:  # ...and flat import when scripts run directly
    from pds4_parser import LunarProduct  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

#: PDS4 Element_Array/data_type -> numpy dtype (explicit byte order).
PDS4_NUMPY_DTYPES = {
    "UnsignedByte": np.dtype("u1"),
    "SignedByte": np.dtype("i1"),
    "UnsignedLSB2": np.dtype("<u2"),
    "SignedLSB2": np.dtype("<i2"),
    "UnsignedMSB2": np.dtype(">u2"),
    "SignedMSB2": np.dtype(">i2"),
    "UnsignedLSB4": np.dtype("<u4"),
    "SignedLSB4": np.dtype("<i4"),
    "UnsignedMSB4": np.dtype(">u4"),
    "SignedMSB4": np.dtype(">i4"),
    "IEEEReal4": np.dtype("<f4"),
    "IEEEReal8": np.dtype("<f8"),
}


def numpy_dtype(pds4_data_type: str) -> np.dtype:
    """Map a PDS4 data_type string to a numpy dtype."""
    try:
        return PDS4_NUMPY_DTYPES[pds4_data_type]
    except KeyError:
        raise ValueError(
            f"unsupported PDS4 data_type {pds4_data_type!r}; known types: "
            f"{sorted(PDS4_NUMPY_DTYPES)}"
        ) from None


def expected_file_size(product: LunarProduct) -> int:
    itemsize = numpy_dtype(product.datatype).itemsize
    return product.lines * product.samples * itemsize


def load_image(
    product: LunarProduct,
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
    offset: int = 0,
) -> np.ndarray:
    """Load the raw image array for *product*, optionally cropped by index.

    ``row_range``/``col_range`` are half-open ``(start, stop)`` index ranges
    in original-resolution pixels; ``None`` means the full axis. Returns a
    regular (non-memmap) ndarray with native byte order.
    """
    dtype = numpy_dtype(product.datatype)
    r0, r1 = row_range if row_range else (0, product.lines)
    c0, c1 = col_range if col_range else (0, product.samples)

    if not (0 <= r0 < r1 <= product.lines):
        raise ValueError(f"invalid row range {row_range} for {product.lines} lines")
    if not (0 <= c0 < c1 <= product.samples):
        raise ValueError(f"invalid col range {col_range} for {product.samples} samples")

    actual_size = product.file_path.stat().st_size
    expected_full = expected_file_size(product) + offset
    if actual_size < expected_full:
        raise ValueError(
            f"image file {product.file_path} is {actual_size:,} bytes but the "
            f"label implies {expected_full:,}; refusing to load garbage"
        )
    if actual_size != expected_full:
        logger.warning(
            "%s size %d != label-implied %d (continuing with label geometry)",
            product.file_path.name,
            actual_size,
            expected_full,
        )

    mm = np.memmap(product.file_path, dtype=dtype, mode="r", offset=offset,
                   shape=(product.lines, product.samples))
    patch = np.asarray(mm[r0:r1, c0:c1])
    if dtype.byteorder not in ("=", "|"):
        patch = patch.astype(dtype.newbyteorder("="))
    logger.debug(
        "Loaded %s rows %s cols %s -> %s %s",
        product.product_id,
        (r0, r1),
        (c0, c1),
        patch.shape,
        patch.dtype,
    )
    return patch


def to_uint8(image: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """Convert any integer image to uint8 via a robust percentile stretch.

    uint8 input is returned unchanged; wider types (e.g. 16-bit TMC-2 DNs)
    are linearly stretched between the given percentiles to preserve
    contrast for feature detection.
    """
    if image.dtype == np.uint8:
        return image
    finite = image.astype(np.float64)
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    if hi <= lo:
        logger.warning("Degenerate intensity range (%.3f, %.3f); returning mid-grey", lo, hi)
        return np.full(image.shape, 128, np.uint8)
    scaled = (finite - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def save_preview(image: np.ndarray, out_path: str | Path, max_dim: int = 1400) -> Path:
    """Save a preview PNG, downscaled so the longest side is <= max_dim."""
    h, w = image.shape[:2]
    factor = max(h, w) / float(max_dim)
    if factor > 1.0:
        image = cv2.resize(
            image, (int(round(w / factor)), int(round(h / factor))),
            interpolation=cv2.INTER_AREA,
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)
    logger.info("Preview written: %s", out_path)
    return out_path
