"""Unit tests for check_existing_sun_match.py (no network)."""

from __future__ import annotations

import pytest

from lunar_data_pipeline import check_existing_sun_match as mod
from lunar_data_pipeline.candidate_ranking import SunAngles
from lunar_data_pipeline.check_existing_sun_match import (
    CheckedPair,
    check_all_pairs,
    check_pair,
    collect_sun_angles,
)


def _label_xml(stem: str, elev: float, azim: float):
    return f"""<?xml version="1.0"?>
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
"""


def _write(tmp_path, stem: str, elev: float, azim: float):
    d = tmp_path / stem
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.xml").write_text(_label_xml(stem, elev, azim), encoding="utf-8")
    (d / f"{stem}.img").write_bytes(b"\x00" * 8)


OHRC = "ch2_ohr_ncp_20231004T0406038822_d_img_d18"
TMC = "ch2_tmc_ncn_20250707T1853051045_d_img_d18"


class TestCheckPair:
    def test_matched_pair(self):
        ohrc = SunAngles(sun_elevation_deg=45.0, sun_azimuth_deg=90.0)
        tmc = SunAngles(sun_elevation_deg=50.0, sun_azimuth_deg=95.0)
        p = check_pair(OHRC, TMC, ohrc, tmc)
        assert p.sun_matched is True
        assert p.azimuth_diff_deg == pytest.approx(5.0)
        assert p.reasons == []

    def test_low_elevation_fails(self):
        ohrc = SunAngles(sun_elevation_deg=9.5, sun_azimuth_deg=90.0)
        tmc = SunAngles(sun_elevation_deg=50.0, sun_azimuth_deg=95.0)
        p = check_pair(OHRC, TMC, ohrc, tmc)
        assert p.sun_matched is False
        assert any("OHRC elevation" in r for r in p.reasons)

    def test_azimuth_gap_fails(self):
        ohrc = SunAngles(sun_elevation_deg=50.0, sun_azimuth_deg=10.0)
        tmc = SunAngles(sun_elevation_deg=55.0, sun_azimuth_deg=120.0)
        p = check_pair(OHRC, TMC, ohrc, tmc)
        assert p.sun_matched is False
        assert any("azimuth diff" in r for r in p.reasons)

    def test_missing_angles_fails(self):
        ohrc = SunAngles()  # not usable
        tmc = SunAngles(sun_elevation_deg=50.0, sun_azimuth_deg=95.0)
        p = check_pair(OHRC, TMC, ohrc, tmc)
        assert p.sun_matched is False
        assert p.azimuth_diff_deg is None

    def test_azimuth_wraparound_match(self):
        ohrc = SunAngles(sun_elevation_deg=40.0, sun_azimuth_deg=5.0)
        tmc = SunAngles(sun_elevation_deg=45.0, sun_azimuth_deg=355.0)
        p = check_pair(OHRC, TMC, ohrc, tmc)
        assert p.sun_matched is True
        assert p.azimuth_diff_deg == pytest.approx(10.0)


class TestCollectAndCheckAll:
    def test_collect_sun_angles_from_labels(self, tmp_path):
        _write(tmp_path, OHRC, 45.0, 90.0)
        _write(tmp_path, TMC, 50.0, 95.0)
        labels = mod.discover_downloaded_labels(tmp_path)
        angles = collect_sun_angles(labels)
        assert set(angles) == {OHRC, TMC}
        assert angles[OHRC].sun_elevation_deg == pytest.approx(45.0)

    def test_check_all_pairs_cartesian(self):
        angles = {
            "ch2_ohr_a_d_img_d18": SunAngles(sun_elevation_deg=45.0, sun_azimuth_deg=90.0),
            "ch2_ohr_b_d_img_d18": SunAngles(sun_elevation_deg=10.0, sun_azimuth_deg=90.0),
            "ch2_tmc_x_d_img_d18": SunAngles(sun_elevation_deg=50.0, sun_azimuth_deg=95.0),
            "ch2_tmc_y_d_img_d18": SunAngles(sun_elevation_deg=55.0, sun_azimuth_deg=270.0),
        }
        pairs, ohrc_ids, tmc_ids = check_all_pairs(angles)
        assert len(pairs) == 4  # 2 OHRC x 2 TMC
        matched = [p for p in pairs if p.sun_matched]
        assert [p.ohrc_id for p in matched] == ["ch2_ohr_a_d_img_d18"]
        # only the a-x combination has both high-elevation and close azimuths
        assert len(matched) == 1
        assert matched[0].tmc2_id == "ch2_tmc_x_d_img_d18"

    def test_to_dict_structure(self):
        ohrc = SunAngles(sun_elevation_deg=45.0, sun_azimuth_deg=90.0)
        tmc = SunAngles(sun_elevation_deg=50.0, sun_azimuth_deg=95.0)
        p = check_pair(OHRC, TMC, ohrc, tmc)
        d = p.to_dict()
        assert d["ohrc_id"] == OHRC
        assert d["sun_matched"] is True
        assert "azimuth_diff_deg" in d
        assert "reasons" in d
