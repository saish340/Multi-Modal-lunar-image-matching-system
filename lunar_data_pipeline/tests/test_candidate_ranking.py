"""Unit tests for candidate_ranking sun-angle + crater-density scoring."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from shapely.geometry import Polygon

from lunar_data_pipeline.candidate_ranking import (
    MIN_SUN_ELEVATION_DEG,
    SunAngles,
    _angle_diff,
    build_crater_index,
    count_craters_in_polygon,
    crater_point_counts,
    load_craters,
    rank_candidates,
    score_pair,
    sun_angles_from_labels,
)
from lunar_data_pipeline.overlap_finder import OverlapPair
from lunar_data_pipeline.pds4_parser import LunarProduct


def _sample_craters() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crater_id": [f"c{i}" for i in range(6)],
            "latitude_deg": [-13.0, -13.2, -13.4, -13.6, 0.0, 45.0],
            "longitude_deg": [25.0, 25.2, 25.4, 25.1, 0.5, 0.5],
            "diameter_km": [2.0, 3.0, 4.0, 5.0, 1.0, 1.0],
            "depth_km": [0.1] * 6,
            "size_class": ["small"] * 6,
        }
    )


def _product(product_id: str, corners: list[tuple[float, float]]) -> LunarProduct:
    return LunarProduct(
        instrument="OHRC",
        file_path=__import__("pathlib").Path(f"{product_id}.img"),
        label_path=__import__("pathlib").Path(f"{product_id}.xml"),
        lines=100,
        samples=100,
        datatype="UnsignedByte",
        footprint=corners,  # (lat, lon)
        product_id=product_id,
        download_filename=f"{product_id}.zip",
    )


def _pair(product_id: str = "ohrc_x") -> OverlapPair:
    ohrc = _product("ohrc_x", [(-13.8, 24.8), (-13.8, 25.6), (-12.9, 25.6), (-12.9, 24.8)])
    tmc = _product("tmc_y", [(-14.0, 24.0), (-14.0, 26.0), (-12.0, 26.0), (-12.0, 24.0)])
    overlap = Polygon([(24.8, -13.8), (25.6, -13.8), (25.6, -12.9), (24.8, -12.9)])
    return OverlapPair(
        ohrc=ohrc,
        tmc2=tmc,
        overlap_polygon=overlap,
        overlap_percentage=100.0,
        pct_of_ohrc=100.0,
        pct_of_tmc2=50.0,
    )


def _write_label(tmp_path, stem: str, elev: float, azim: float):
    label_dir = tmp_path / stem
    label_dir.mkdir(parents=True, exist_ok=True)
    path = label_dir / f"{stem}.xml"
    path.write_text(
        f"""<?xml version="1.0"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
    xmlns:isda="https://isda.issdc.gov.in/pds4/isda/v1">
  <Identification_Area>
    <logical_identifier>{stem}</logical_identifier>
  </Identification_Area>
  <Observation_Area>
    <isda:sun_elevation unit="deg">{elev}</isda:sun_elevation>
    <isda:sun_azimuth unit="deg">{azim}</isda:sun_azimuth>
    <isda:solar_incidence unit="deg">80.5</isda:solar_incidence>
    <Mission_Area>
      <isda:Geometry_Parameters>
        <isda:Refined_Corner_Coordinates>
          <isda:upper_left_latitude>-13.8</isda:upper_left_latitude>
          <isda:upper_left_longitude>24.8</isda:upper_left_longitude>
          <isda:upper_right_latitude>-13.8</isda:upper_right_latitude>
          <isda:upper_right_longitude>25.6</isda:upper_right_longitude>
          <isda:lower_right_latitude>-12.9</isda:lower_right_latitude>
          <isda:lower_right_longitude>25.6</isda:lower_right_longitude>
          <isda:lower_left_latitude>-12.9</isda:lower_left_latitude>
          <isda:lower_left_longitude>24.8</isda:lower_left_longitude>
        </isda:Refined_Corner_Coordinates>
      </isda:Geometry_Parameters>
    </Mission_Area>
  </Observation_Area>
  <File_Area_Observational>
    <File><file_name>{stem}.img</file_name></File>
    <Array_2D_Image>
      <Element_Array><data_type>UnsignedByte</data_type></Element_Array>
      <Axis_Array><axis_name>Line</axis_name><elements>100</elements></Axis_Array>
      <Axis_Array><axis_name>Sample</axis_name><elements>100</elements></Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
</Product_Observational>
""",
        encoding="utf-8",
    )
    (label_dir / f"{stem}.img").write_bytes(b"\x00" * 8)
    return path


class TestCraters:
    def test_load_craters(self, tmp_path):
        p = tmp_path / "c.parquet"
        _sample_craters().to_parquet(p)
        df = load_craters(p)
        assert len(df) == 6
        assert "longitude_deg" in df.columns

    def test_crater_point_counts_inside_polygon(self):
        df = _sample_craters()
        poly = Polygon([(24.0, -14.0), (26.0, -14.0), (26.0, -12.0), (24.0, -12.0)])
        counts = crater_point_counts(df, [poly])
        # Four craters are inside the [-14,-12] x [24,26] box; the two at
        # (0,0) and (45,0) are not.
        assert counts == [4]
        # count_craters_in_polygon agrees.
        points, tree = build_crater_index(df)
        assert count_craters_in_polygon(tree, points, poly) == 4

    def test_crater_point_counts_outside(self):
        df = _sample_craters()
        far = Polygon([(0.0, 44.0), (1.0, 44.0), (1.0, 46.0), (0.0, 46.0)])
        assert crater_point_counts(df, [far]) == [1]


class TestSunAngles:
    def test_sun_angles_from_labels(self, tmp_path):
        _write_label(tmp_path, "ch2_ohr_ncp_x_d_img_d18", 45.0, 90.0)
        _write_label(tmp_path, "ch2_tmc_ncn_y_d_img_d18", 50.0, 95.0)
        sun = sun_angles_from_labels(tmp_path)
        assert set(sun) == {"ch2_ohr_ncp_x_d_img_d18", "ch2_tmc_ncn_y_d_img_d18"}
        assert sun["ch2_ohr_ncp_x_d_img_d18"].sun_elevation_deg == pytest.approx(45.0)
        assert sun["ch2_ohr_ncp_x_d_img_d18"].sun_azimuth_deg == pytest.approx(90.0)
        assert sun["ch2_ohr_ncp_x_d_img_d18"].usable

    def test_sun_angles_no_labels_dir(self):
        assert sun_angles_from_labels(None) == {}
        assert sun_angles_from_labels("does_not_exist") == {}

    def test_sun_angles_usable_false_when_missing(self):
        assert SunAngles().usable is False
        assert SunAngles(sun_elevation_deg=1.0, sun_azimuth_deg=None).usable is False
        assert SunAngles(sun_elevation_deg=1.0, sun_azimuth_deg=2.0).usable is True


class TestAngleDiff:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (0.0, 10.0, 10.0),
            (350.0, 10.0, 20.0),  # wraps through 360
            (10.0, 350.0, 20.0),
            (100.0, 100.0, 0.0),
            (0.0, 180.0, 180.0),
            (0.0, 181.0, 179.0),
        ],
    )
    def test_wraparound(self, a, b, expected):
        assert _angle_diff(a, b) == pytest.approx(expected)


class TestScorePair:
    def test_with_sun_angles_available(self):
        pair = _pair()
        ohrc = _product("ohrc_x", pair.ohrc.footprint or [])
        tmc = _product("tmc_y", pair.tmc2.footprint or [])
        sun = {
            "ohrc_x": SunAngles(sun_elevation_deg=45.0, sun_azimuth_deg=90.0),
            "tmc_y": SunAngles(sun_elevation_deg=50.0, sun_azimuth_deg=95.0),
        }
        score = score_pair(pair, crater_count=4, sun_map=sun)
        assert score.sun_angles_available is True
        assert score.sun_elev_diff_deg == pytest.approx(5.0)
        assert score.sun_az_diff_deg == pytest.approx(5.0)
        assert score.labels_needed == []

    def test_needs_labels_when_angles_missing(self):
        pair = _pair()
        score = score_pair(pair, crater_count=4, sun_map={})
        assert score.sun_angles_available is False
        assert score.sun_elev_diff_deg is None
        assert set(score.labels_needed) == {"ohrc_x", "tmc_y"}


class TestRankCandidates:
    def test_sun_known_ranked_first_then_density(self):
        df = _sample_craters()
        pair_a = _pair()  # covers the 4-crater cluster
        # B far way, zero craters
        ohrc_b = _product(
            "ohrc_b", [(-90.0, 0.0), (-90.0, 1.0), (-89.0, 1.0), (-89.0, 0.0)]
        )
        tmc_b = _product(
            "tmc_b", [(-91.0, -1.0), (-91.0, 2.0), (-88.0, 2.0), (-88.0, -1.0)]
        )
        poly_b = Polygon([(0.0, -90.0), (1.0, -90.0), (1.0, -89.0), (0.0, -89.0)])
        pair_b = OverlapPair(
            ohrc=ohrc_b,
            tmc2=tmc_b,
            overlap_polygon=poly_b,
            overlap_percentage=100.0,
            pct_of_ohrc=100.0,
            pct_of_tmc2=10.0,
        )
        sun = {
            "ohrc_x": SunAngles(sun_elevation_deg=45.0, sun_azimuth_deg=90.0),
            "tmc_y": SunAngles(sun_elevation_deg=50.0, sun_azimuth_deg=95.0),
            "ohrc_b": SunAngles(sun_elevation_deg=70.0, sun_azimuth_deg=0.0),
            "tmc_b": SunAngles(sun_elevation_deg=75.0, sun_azimuth_deg=5.0),
        }
        scored, diagnostics = rank_candidates(
            [pair_a, pair_b], df, sun, min_craters=1
        )
        assert scored[0].ohrc_id == "ohrc_x"
        assert scored[0].crater_count == 4
        assert scored[1].ohrc_id == "ohrc_b"
        assert scored[1].crater_count == 0
        assert "passes all filters" in diagnostics["ohrc_x__x__tmc_y"]

    def test_flags_below_min_craters(self):
        df = _sample_craters()
        ohrc = _product(
            "ohrc_z", [(-90.0, 0.0), (-90.0, 1.0), (-89.0, 1.0), (-89.0, 0.0)]
        )
        tmc = _product(
            "tmc_z", [(-91.0, -1.0), (-91.0, 2.0), (-88.0, 2.0), (-88.0, -1.0)]
        )
        poly = Polygon([(0.0, -90.0), (1.0, -90.0), (1.0, -89.0), (0.0, -89.0)])
        pair = OverlapPair(
            ohrc=ohrc, tmc2=tmc, overlap_polygon=poly,
            overlap_percentage=100.0, pct_of_ohrc=100.0, pct_of_tmc2=10.0,
        )
        scored, diagnostics = rank_candidates([pair], df, {}, min_craters=5)
        assert scored[0].crater_count == 0
        assert "only 0 craters" in diagnostics["ohrc_z__x__tmc_z"]

    def test_to_dict_contains_download_filenames(self):
        pair = _pair()
        score = score_pair(pair, crater_count=4, sun_map={})
        d = score.to_dict()
        assert d["ohrc_download"] == "ohrc_x.zip"
        assert d["tmc2_download"] == "tmc_y.zip"
        assert d["overlap_center_lonlat"]
        assert d["sun_angles_available"] is False

    def test_passes_filters_true_when_all_constraints_met(self):
        pair = _pair()
        ohrc = _product("ohrc_x", pair.ohrc.footprint or [])
        tmc = _product("tmc_y", pair.tmc2.footprint or [])
        sun = {
            "ohrc_x": SunAngles(sun_elevation_deg=60.0, sun_azimuth_deg=90.0),
            "tmc_y": SunAngles(sun_elevation_deg=65.0, sun_azimuth_deg=95.0),
        }
        score = score_pair(
            pair,
            crater_count=10,
            sun_map=sun,
            min_overlap_pct=90.0,
            min_craters=5,
            min_elevation_deg=20.0,
        )
        assert score.passes_filters is True
        assert score.filter_reasons == []

    def test_passes_filters_false_on_low_craters(self):
        pair = _pair()
        score = score_pair(
            pair,
            crater_count=1,
            sun_map={},
            min_craters=5,
        )
        assert score.passes_filters is False
        assert any("craters" in r for r in score.filter_reasons)

    def test_shortlist_selection_prefers_passing(self):
        df = _sample_craters()
        pair_a = _pair()  # 4 craters, sun available -> passes (min_craters=1)
        ohrc_b = _product(
            "ohrc_b", [(-90.0, 0.0), (-90.0, 1.0), (-89.0, 1.0), (-89.0, 0.0)]
        )
        tmc_b = _product(
            "tmc_b", [(-91.0, -1.0), (-91.0, 2.0), (-88.0, 2.0), (-88.0, -1.0)]
        )
        poly_b = Polygon([(0.0, -90.0), (1.0, -90.0), (1.0, -89.0), (0.0, -89.0)])
        pair_b = OverlapPair(
            ohrc=ohrc_b, tmc2=tmc_b, overlap_polygon=poly_b,
            overlap_percentage=100.0, pct_of_ohrc=100.0, pct_of_tmc2=10.0,
        )
        sun = {
            "ohrc_x": SunAngles(sun_elevation_deg=60.0, sun_azimuth_deg=90.0),
            "tmc_y": SunAngles(sun_elevation_deg=65.0, sun_azimuth_deg=95.0),
            "ohrc_b": SunAngles(sun_elevation_deg=60.0, sun_azimuth_deg=0.0),
            "tmc_b": SunAngles(sun_elevation_deg=65.0, sun_azimuth_deg=5.0),
        }
        scored, _ = rank_candidates([pair_a, pair_b], df, sun, min_craters=1)
        passing = [s for s in scored if s.passes_filters]
        assert [s.ohrc_id for s in passing] == ["ohrc_x"]
        assert scored[0] is passing[0]
