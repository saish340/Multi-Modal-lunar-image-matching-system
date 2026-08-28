"""Unit tests for pds4_parser using synthetic PDS4-style labels.

The XML snippets below mimic the ISDA Chandrayaan-2 profile (pds + isda
namespaces) but are hand-written minimal examples, not real archive labels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lunar_data_pipeline.pds4_parser import (
    CORNER_ORDER,
    LunarProduct,
    find_geometry_csv,
    infer_instrument,
    parse_label,
)

NS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
                       xmlns:isda="https://isda.issdc.gov.in/pds4/isda/v1">
  <Identification_Area>
    <logical_identifier>{lid}</logical_identifier>
    <version_id>1.0</version_id>
    <product_class>Product_Observational</product_class>
  </Identification_Area>
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>2026-01-03T06:09:04.137Z</start_date_time>
      <stop_date_time>2026-01-03T06:09:20.520Z</stop_date_time>
    </Time_Coordinates>
    {sun_angles}
    {observing_system}
    <Mission_Area>
      <isda:Product_Parameters>
        <isda:pixel_resolution unit="m/pixel">0.25</isda:pixel_resolution>
        <isda:projection>Polar stereographic</isda:projection>
        <isda:area>South Pole</isda:area>
      </isda:Product_Parameters>
      <isda:Geometry_Parameters>
{corner_blocks}
      </isda:Geometry_Parameters>
    </Mission_Area>
  </Observation_Area>
  <File_Area_Observational>
    <File>
      <file_name>{img_name}</file_name>
    </File>
    <Array_2D_Image>
      <Element_Array>
        <data_type>{datatype}</data_type>
      </Element_Array>
      <Axis_Array>
        <axis_name>Line</axis_name>
        <elements>{lines}</elements>
      </Axis_Array>
      <Axis_Array>
        <axis_name>Sample</axis_name>
        <elements>{samples}</elements>
      </Axis_Array>
    </Array_2D_Image>
  </File_Area_Observational>
</Product_Observational>
"""

OBSERVING_SYSTEM_OHRC = """
    <Observing_System>
      <Observing_System_Component>
        <name>Chandrayaan 2 Orbiter</name>
        <type>Spacecraft</type>
      </Observing_System_Component>
      <Observing_System_Component>
        <name>orbiter high resolution camera</name>
        <type>Instrument</type>
      </Observing_System_Component>
    </Observing_System>"""

OBSERVING_SYSTEM_TMC = """
    <Observing_System>
      <Observing_System_Component>
        <name>terrain mapping camera</name>
        <type>Instrument</type>
      </Observing_System_Component>
    </Observing_System>"""


def corners_xml(prefix: str, values: dict[str, tuple[float, float]]) -> str:
    """Render a Refined/System corner-coordinates block."""
    lines = [f"        <isda:{prefix}>"]
    for corner in ("upper_left", "upper_right", "lower_left", "lower_right"):
        lat, lon = values[corner]
        lines.append(
            f"          <isda:{corner}_latitude unit=\"deg\">{lat}</isda:{corner}_latitude>"
        )
        lines.append(
            f"          <isda:{corner}_longitude unit=\"deg\">{lon}</isda:{corner}_longitude>"
        )
    lines.append(f"        </isda:{prefix}>")
    return "\n".join(lines)


def write_label(
    tmp_path: Path,
    *,
    stem: str = "ch2_ohr_ncp_20260103T0609041371_d_img_d18",
    img_name: str | None = None,
    create_image: bool = True,
    lid: str = "urn:isro:isda:ch2_cho.ohr:data_calibrated:x",
    observing_system: str = OBSERVING_SYSTEM_OHRC,
    datatype: str = "UnsignedByte",
    lines: int = 101074,
    samples: int = 12000,
    refined: dict[str, tuple[float, float]] | None = None,
    system_level: dict[str, tuple[float, float]] | None = None,
    raw_xml: str | None = None,
    sun_angles: str = "",
) -> Path:
    label_dir = tmp_path / stem
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{stem}.xml"

    if raw_xml is not None:
        label_path.write_text(raw_xml, encoding="utf-8")
    else:
        blocks = []
        if refined is not None:
            blocks.append(corners_xml("Refined_Corner_Coordinates", refined))
        if system_level is not None:
            blocks.append(corners_xml("System_Level_Coordinates", system_level))
        label_path.write_text(
            NS_TEMPLATE.format(
                lid=lid,
                sun_angles=sun_angles,
                observing_system=observing_system,
                corner_blocks="\n".join(blocks),
                img_name=img_name or f"{stem}.img",
                datatype=datatype,
                lines=lines,
                samples=samples,
            ),
            encoding="utf-8",
        )

    image_name = img_name or f"{stem}.img"
    if create_image:
        (label_dir / image_name).write_bytes(b"\x00" * 16)
    return label_path


REFINED = {
    "upper_left": (-85.323534, 27.723846),
    "upper_right": (-85.365795, 26.571730),
    "lower_left": (-84.552455, 24.029151),
    "lower_right": (-84.588687, 23.016449),
}

SYSTEM_LEVEL = {
    "upper_left": (-85.0, 27.0),
    "upper_right": (-85.0, 26.0),
    "lower_left": (-84.0, 24.0),
    "lower_right": (-84.0, 23.0),
}


class TestParseLabel:
    def test_parses_complete_ohrc_label(self, tmp_path):
        label = write_label(tmp_path, refined=REFINED)

        product = parse_label(label)

        assert isinstance(product, LunarProduct)
        assert product.instrument == "OHRC"
        assert product.lines == 101074
        assert product.samples == 12000
        assert product.datatype == "UnsignedByte"
        assert product.product_id == label.stem
        assert product.pixel_resolution_m == pytest.approx(0.25)
        assert product.projection == "Polar stereographic"
        assert product.area == "South Pole"
        assert len(product.footprint) == 4

        # Footprint must be in CORNER_ORDER with (lat, lon) tuples.
        expected_order = [REFINED[c] for c in CORNER_ORDER]
        assert product.footprint == pytest.approx(expected_order)
        assert product.file_path.exists()
        assert product.file_path.name.endswith(".img")

    def test_parses_sun_geometry_from_label(self, tmp_path):
        label = write_label(
            tmp_path,
            refined=REFINED,
            sun_angles=(
                '  <isda:sun_elevation unit="deg">9.488661</isda:sun_elevation>\n'
                '  <isda:sun_azimuth unit="deg">272.874656</isda:sun_azimuth>\n'
                '  <isda:solar_incidence unit="deg">80.511339</isda:solar_incidence>'
            ),
        )

        product = parse_label(label)

        assert product is not None
        assert product.sun_elevation_deg == pytest.approx(9.488661)
        assert product.sun_azimuth_deg == pytest.approx(272.874656)
        assert product.solar_incidence_deg == pytest.approx(80.511339)

    def test_sun_geometry_defaults_to_none(self, tmp_path):
        label = write_label(tmp_path, refined=REFINED, sun_angles="")

        product = parse_label(label)

        assert product is not None
        assert product.sun_elevation_deg is None
        assert product.sun_azimuth_deg is None
        assert product.solar_incidence_deg is None

    def test_prefers_refined_over_system_corners(self, tmp_path):
        label = write_label(
            tmp_path, refined=REFINED, system_level=SYSTEM_LEVEL
        )

        product = parse_label(label)

        assert product.footprint[0] == pytest.approx(REFINED["upper_left"])
        assert product.footprint[0] != pytest.approx(SYSTEM_LEVEL["upper_left"])

    def test_falls_back_to_system_level_corners(self, tmp_path):
        label = write_label(tmp_path, refined=None, system_level=SYSTEM_LEVEL)

        product = parse_label(label)

        expected_order = [SYSTEM_LEVEL[c] for c in CORNER_ORDER]
        assert product.footprint == pytest.approx(expected_order)

    def test_instrument_inferred_from_filename_when_component_generic(self, tmp_path):
        label = write_label(
            tmp_path,
            stem="ch2_tmc_ncn_20260813T0627378557_d_img_d18",
            observing_system="",
            lid="urn:isro:isda:some:generic:id",
            datatype="UnsignedLSB2",
            lines=148108,
            samples=4000,
            refined={
                "upper_left": (-3.6, 142.7),
                "upper_right": (-3.6, 141.9),
                "lower_left": (-27.7, 141.7),
                "lower_right": (-27.7, 140.9),
            },
        )

        product = parse_label(label)

        assert product.instrument == "TMC2"
        assert product.datatype == "UnsignedLSB2"

    def test_malformed_xml_returns_none_not_raise(self, tmp_path):
        label_dir = tmp_path / "bad_product"
        label_dir.mkdir()
        bad = label_dir / "bad_product.xml"
        bad.write_text("<Product_Observational><unclosed>", encoding="utf-8")

        assert parse_label(bad) is None

    def test_missing_footprint_yields_empty_list(self, tmp_path):
        label = write_label(tmp_path, refined=None, system_level=None)

        product = parse_label(label)

        assert product is not None
        assert product.footprint == []

    def test_missing_dimensions_return_none(self, tmp_path):
        raw = NS_TEMPLATE.replace("<elements>101074</elements>", "").format(
            lid="x",
            sun_angles="",
            observing_system=OBSERVING_SYSTEM_OHRC,
            corner_blocks=corners_xml("System_Level_Coordinates", SYSTEM_LEVEL),
            img_name="x.img",
            datatype="UnsignedByte",
            lines="",
            samples="",
        )
        label = write_label(tmp_path, stem="dims_less", raw_xml=raw, create_image=False)

        assert parse_label(label) is None

    def test_non_observational_label_skipped(self, tmp_path):
        label = write_label(
            tmp_path,
            stem="weird",
            raw_xml="<?xml version='1.0'?><Product_Browse><x/></Product_Browse>",
            create_image=False,
        )

        assert parse_label(label) is None


class TestInstrumentInference:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("orbiter high resolution camera", "OHRC"),
            ("terrain mapping camera", "TMC2"),
            ("imaging infrared spectrometer", "IIRS"),
        ],
    )
    def test_component_names(self, text, expected):
        assert infer_instrument(text) == expected

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("ch2_ohr_ncp_x.xml", "OHRC"),
            ("ch2_tmc_ncn_x.xml", "TMC2"),
            ("ch2_iir_sss_x.xml", "IIRS"),
        ],
    )
    def test_filename_prefixes(self, filename, expected):
        from lunar_data_pipeline.pds4_parser import _instrument_from_filename

        assert _instrument_from_filename(filename) == expected


class TestGeometryCsvDiscovery:
    def test_find_geometry_csv_by_convention(self, tmp_path):
        stem = "ch2_ohr_ncp_20260103T0609041371_d_img_d18"
        csv_rel = Path("geometry") / "calibrated" / "20260103" / (
            stem.replace("_d_img_", "_g_grd_") + ".csv"
        )
        csv_path = tmp_path / csv_rel
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text("Longitude,Latitude,Pixel,Scan\n", encoding="utf-8")
        data_path = tmp_path / "data" / "calibrated" / "20260103" / f"{stem}.xml"
        data_path.parent.mkdir(parents=True)
        data_path.write_text("<dummy/>", encoding="utf-8")

        found = find_geometry_csv(data_path)

        assert found is not None
        assert found.resolve() == csv_path.resolve()

    def test_returns_none_when_absent(self, tmp_path):
        label = tmp_path / "solo.xml"
        label.write_text("<dummy/>", encoding="utf-8")

        assert find_geometry_csv(label) is None
