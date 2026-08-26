"""Tests for shapefile_loader using small synthetic PRADAN-style catalogs.

These write real (tiny) shapefiles via geopandas so the loader's full
read -> validate -> convert path runs, including the datetime-column
exclusion that protects against ISRO's year-0000 corruption.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from lunar_data_pipeline.catalog_overlap_search import (
    rank_pairs,
    search_overlaps_with_index,
)
from lunar_data_pipeline.pds4_parser import CORNER_ORDER, LunarProduct
from lunar_data_pipeline.shapefile_loader import (
    load_catalog,
    load_catalog_shapefile,
)


def rect(lat0: float, lat1: float, lon0: float, lon1: float) -> Polygon:
    return Polygon([(lon0, lat1), (lon1, lat1), (lon1, lat0), (lon0, lat0)])


def catalog_row(
    product_id: str,
    lat0: float,
    lat1: float,
    lon0: float,
    lon1: float,
    *,
    download: str | None = None,
    inc_angle: float = 0.0,
    orbit: float = 1000.0,
    skip_corners: bool = False,
) -> dict:
    """One attribute-table row mimicking PRADAN's cal-catalog schema."""
    row = {
        "PRODUCT_ID": product_id,
        "OBS_ST_TIME": "2021-04-05T04:42:09Z",
        "OBS_ED_TIME": "2021-04-05T04:42:25Z",
        "UL_LAT": float("nan") if skip_corners else lat1,
        "UL_LON": float("nan") if skip_corners else lon0,
        "UR_LAT": float("nan") if skip_corners else lat1,
        "UR_LON": float("nan") if skip_corners else lon1,
        "BL_LAT": float("nan") if skip_corners else lat0,
        "BL_LON": float("nan") if skip_corners else lon0,
        "BR_LAT": float("nan") if skip_corners else lat0,
        "BR_LON": float("nan") if skip_corners else lon1,
        "DOWNLOAD": download or f"{product_id}.zip",
        "BROWSE": f"{product_id}_b_brw.png",
        "EMI_ANGLE": 0.0,
        "INC_ANGLE": inc_angle,
        "PHA_ANGLE": 0.0,
        "IM_ORB_NUM": orbit,
        "DM_ORB_NUM": orbit + 6,
        "L0_ID": "level0job",
        # Deliberately include a datetime column like the corrupted DOP.
        "DOP": pd.Timestamp("2021-04-05"),
        "geometry": rect(lat0, lat1, lon0, lon1),
    }
    return row


def write_catalog(path: Path, rows: list[dict]) -> Path:
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path)
    return path


class TestLoadCatalogShapefile:
    def test_converts_rows_to_lunar_products(self, tmp_path):
        shp = write_catalog(
            tmp_path / "ch2_ohr_cal.shp",
            [catalog_row("ch2_ohr_ncp_a_d_img_d18", -20, -18, 30, 32)],
        )

        products = load_catalog_shapefile(shp, "OHRC")

        assert len(products) == 1
        p = products[0]
        assert isinstance(p, LunarProduct)
        assert p.instrument == "OHRC"
        assert p.product_id == "ch2_ohr_ncp_a_d_img_d18"
        # Footprint order must match CORNER_ORDER (UL, UR, LR, LL).
        expected = [(-18, 30), (-18, 32), (-20, 32), (-20, 30)]
        assert p.footprint == pytest.approx(expected)
        assert p.download_filename == "ch2_ohr_ncp_a_d_img_d18.zip"
        assert p.imaging_orbit_number == 1000
        assert p.start_time == "2021-04-05T04:42:09Z"

    def test_excludes_datetime_columns_without_crashing(self, tmp_path):
        # The synthetic file contains a valid-dated DOP column; the loader
        # excludes ALL datetime columns regardless of content.
        shp = write_catalog(
            tmp_path / "ch2_tmc_cal.shp",
            [catalog_row("ch2_tmc_ncn_b_d_img_d18", 10, 12, -40, -38)],
        )

        products = load_catalog_shapefile(shp, "TMC2")

        assert len(products) == 1
        assert products[0].instrument == "TMC2"

    def test_skips_corrupt_coordinates_null_ids_and_duplicates(self, tmp_path):
        rows = [
            catalog_row("good_product", -10, -9, 50, 51),
            catalog_row("corrupt_lat", 733744.6, 733745.0, 20, 21),  # polar junk
            {**catalog_row("null_id", -5, -4, 60, 61), "PRODUCT_ID": None},
            catalog_row("dup_product", 15, 16, -70, -69),  # first occurrence wins
            catalog_row("dup_product", 25, 26, -75, -74),
        ]
        shp = write_catalog(tmp_path / "ch2_ohr_cal_sp.shp", rows)

        products = load_catalog_shapefile(shp, "OHRC")

        ids = [p.product_id for p in products]
        assert ids == ["good_product", "dup_product"]
        kept = next(p for p in products if p.product_id == "dup_product")
        assert kept.footprint[0] == pytest.approx((16.0, -70.0))

    def test_geometry_bounds_fallback_when_corners_missing(self, tmp_path):
        rows = [
            catalog_row(
                "no_corners_product", -33, -31, 140, 142, skip_corners=True
            )
        ]
        shp = write_catalog(tmp_path / "ch2_ohr_cal_np.shp", rows)

        products = load_catalog_shapefile(shp, "OHRC")

        assert len(products) == 1
        expected = [(-31, 140), (-31, 142), (-33, 142), (-33, 140)]
        assert products[0].footprint == pytest.approx(expected)

    def test_rejects_unsupported_instrument(self, tmp_path):
        shp = write_catalog(
            tmp_path / "ch2_ohr_cal.shp", [catalog_row("x", -1, 1, 1, 2)]
        )

        with pytest.raises(ValueError):
            load_catalog_shapefile(shp, "HRSC")


class TestLoadCatalogMultiFile:
    def test_loads_and_dedupes_across_files(self, tmp_path):
        global_shp = write_catalog(
            tmp_path / "global" / "ch2_ohr_cal.shp",
            [
                catalog_row("ohr_global", -20, -18, 30, 32),
                catalog_row("ohr_other", 40, 41, -10, -9),
            ],
        )
        sp_shp = write_catalog(
            tmp_path / "sp" / "ch2_ohr_cal_sp.shp",
            [catalog_row("ohr_global", -20, -18, 30, 32)],  # duplicate across files
        )

        products, counts = load_catalog([global_shp, sp_shp], "OHRC")

        ids = sorted(p.product_id for p in products)
        assert ids == ["ohr_global", "ohr_other"]
        assert counts["ch2_ohr_cal.shp"] == 2
        assert counts["ch2_ohr_cal_sp.shp"] == 0  # only contained the duplicate


class TestCatalogSearchIntegration:
    def _build_pair_catalogs(self, tmp_path: Path) -> tuple[Path, Path]:
        ohrc_shp = write_catalog(
            tmp_path / "ohrc" / "ch2_ohr_cal.shp",
            [
                catalog_row("ch2_ohr_scene", -22, -20, 141, 143),
                catalog_row("ch2_ohr_polar_scene", 70, 71, 10, 11),
            ],
        )
        tmc_shp = write_catalog(
            tmp_path / "tmc" / "ch2_tmc_cal.shp",
            [
                catalog_row("ch2_tmc_ncn_strip", -30, -10, 139, 145),
                catalog_row("ch2_tmc_ncn_polar_strip", 69, 90, 8, 14),
                catalog_row("far_away_strip", -80, -79, -170, -169),
            ],
        )
        return ohrc_shp, tmc_shp

    def test_search_prefilter_ranking_and_stats(self, tmp_path):
        ohrc_shp, tmc_shp = self._build_pair_catalogs(tmp_path)
        ohrc = load_catalog_shapefile(ohrc_shp, "OHRC")
        tmc = load_catalog_shapefile(tmc_shp, "TMC2")

        result = search_overlaps_with_index(ohrc, tmc)

        assert result.stats.combos_checked == 2 * 3
        assert result.stats.overlapping_pairs == 2

        ranked = rank_pairs(result.pairs, max_abs_lat=60.0)
        # The polar overlap covers a bigger share of its tiny OHRC footprint,
        # but ranking must still put the non-polar pair first.
        assert ranked[0].ohrc.product_id == "ch2_ohr_scene"
        assert ranked[0].tmc2.product_id == "ch2_tmc_ncn_strip"
        assert abs(ranked[0].overlap_polygon.centroid.y) <= 60.0
        assert abs(ranked[1].overlap_polygon.centroid.y) > 60.0
