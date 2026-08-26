"""lunar_data_pipeline: Chandrayaan-2 PDS4 parsing, overlap finding, and
corner-based ground-truth correspondence generation (Phase 0)."""

from .groundtruth import (
    PixelTransform,
    build_pixel_transform,
    project_point,
)
from .overlap_finder import OverlapPair, find_overlaps, footprint_polygon
from .pds4_parser import LunarProduct, parse_label

__version__ = "0.1.0"

__all__ = [
    "LunarProduct",
    "OverlapPair",
    "PixelTransform",
    "build_pixel_transform",
    "find_overlaps",
    "footprint_polygon",
    "parse_label",
    "project_point",
]
