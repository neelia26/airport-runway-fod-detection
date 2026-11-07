from pathlib import Path

import pytest

from fod_detection.data.yolo_conversion import (
    BoundingBox,
    read_voc_annotation,
    voc_box_to_yolo,
    yolo_box_to_voc,
)


def test_voc_yolo_round_trip():
    box = BoundingBox("Bolt", 30, 60, 150, 180)

    converted = voc_box_to_yolo(box, width=300, height=300)
    reconstructed = yolo_box_to_voc(converted, width=300, height=300)

    assert converted == pytest.approx((0.3, 0.4, 0.4, 0.4))
    assert reconstructed == pytest.approx((30, 60, 150, 180))


def test_annotation_parser_rejects_invalid_box(tmp_path: Path):
    xml = tmp_path / "000001.xml"
    xml.write_text(
        """<annotation><size><width>300</width><height>300</height></size>
        <object><name>Bolt</name><bndbox><xmin>200</xmin><ymin>20</ymin>
        <xmax>100</xmax><ymax>40</ymax></bndbox></object></annotation>"""
    )

    with pytest.raises(ValueError, match="Invalid box"):
        read_voc_annotation(xml)
