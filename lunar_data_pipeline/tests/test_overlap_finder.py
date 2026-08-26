"""Tests for overlap_finder using synthetic footprint polygons."""

from __future__ import annotations

from pathlib import Path

import pytest

from lunar_data_pipeline.overlap_finder import (
    compute_overlap,
    find_overlaps,
    footprint_polygon,
)
from lunar_data_pipeline.pds4_parser import LunarProduct


def make_product(
    instrument: str,
    product_id: str,
    footprint: list[tuple[float, float]],
    lines: int = 100,
    samples: int = 100,
) -> LunarProduct:
    return LunarProduct(
        instrument=instrument,
        file_path=Path(f"/fake/{product_id}.img"),
        label_path=Path(f"/fake/{product_id}.xml"),
        lines=lines,
        samples=samples,
        datatype="UnsignedByte",
        footprint=list(footprint),
        product_id=product_id,
    )


def square(lat0: float, lat1: float, lon0: float, lon1: float):
    """Corners (lat, lon) in CORNER_ORDER: UL, UR, LR, LL."""
    return [(lat1, lon0), (lat1, lon1), (lat0, lon1), (lat0, lon0)]


class TestFootprintPolygon:
    def test_valid_polygon_from_square(self):
        product = make_product("OHRC", "a", square(-85, -84, 23, 24))

        poly = footprint_polygon(product)

        assert poly is not None
        assert poly.is_valid
        assert poly.area == pytest.approx(1.0)  # 1 deg x 1 deg

    def test_none_for_missing_footprint(self):
        product = make_product("OHRC", "empty", [])

        assert footprint_polygon(product) is None


class TestComputeOverlap:
    def test_partial_overlap_percentages(self):
        a = footprint_polygon(make_product("OHRC", "a", square(0, 2, 0, 2)))
        b = footprint_polygon(make_product("TMC2", "b", square(1, 3, 1, 3)))

        pct_a, pct_b, pct_smaller = compute_overlap(a, b)

        # Intersection is a 1x1 square inside two 2x2 squares -> 25% each way.
        assert pct_a == (pct_b)
        assert abs(pct_smaller - 25.0) < 1e-9

    def test_containment_gives_hundred_percent_of_inner(self):
        outer = footprint_polygon(
            make_product("TMC2", "strip", square(-30, 0, 140, 143))
        )
        inner = footprint_polygon(make_product("OHRC", "scene", square(-5, -4, 141, 142)))

        _, _, pct_smaller = compute_overlap(inner, outer)

        assert pct_smaller == 100.0

    def test_disjoint_polygons_return_none(self):
        a = footprint_polygon(make_product("OHRC", "a", square(0, 1, 0, 1)))
        b = footprint_polygon(make_product("TMC2", "b", square(50, 51, 50, 51)))

        assert compute_overlap(a, b) is None


class TestFindOverlaps:
    def test_pairs_ranked_descending_and_tuple_unpackable(self):
        ohrc = [make_product("OHRC", "scene", square(-5, -4, 141, 142))]
        tmc_strips = [
            make_product("TMC2", "far_strip", square(40, 41, 10, 11)),
            make_product("TMC2", "partial_strip", square(-6, 6, 140.5, 141.5)),
            make_product("TMC2", "covering_strip", square(-30, 0, 140, 143)),
        ]

        pairs = find_overlaps(ohrc, tmc_strips)

        assert [p.tmc2.product_id for p in pairs] == [
            "covering_strip",
            "partial_strip",
        ]
        # Tuple-unpacking compatibility with the spec'd 4-tuple shape.
        first = pairs[0]
        ohrc_prod, tmc_prod, polygon, pct = first[:4]
        assert ohrc_prod.product_id == "scene"
        assert tmc_prod.product_id == "covering_strip"
        assert not polygon.is_empty
        assert pct == 100.0

    def test_min_overlap_pct_filters_weak_pairs(self):
        ohrc = [make_product("OHRC", "scene", square(0, 2, 0, 2))]
        tmc = [make_product("TMC2", "barely_touching", square(1.99, 3.99, 1.99, 3.99))]

        pairs_all = find_overlaps(ohrc, tmc, min_overlap_pct=0.0)
        pairs_strict = find_overlaps(ohrc, tmc, min_overlap_pct=5.0)

        assert len(pairs_all) == 1
        assert pairs_strict == []

    def test_products_without_footprint_are_excluded(self):
        broken = make_product("OHRC", "no_corners", [])
        good = make_product("TMC2", "strip", square(-1, 1, -1, 1))
        also_good = make_product("OHRC", "scene", square(-1, 1, -1, 1))

        pairs = find_overlaps([broken, also_good], [good])

        assert len(pairs) == 1
        assert pairs[0].ohrc.product_id == "scene"
