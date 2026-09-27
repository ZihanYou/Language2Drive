"""Bounded, source-only extraction of OpenDRIVE speed and signal metadata.

This module does not match OpenDRIVE roads/lanes to ClipGT IDs, interpret speed
values, or establish rule applicability.  Attributes are XML-decoded strings,
not a byte-for-byte copy of the XML (XML attribute normalization still applies).
"""

from __future__ import annotations

from typing import Any
from xml.etree.ElementTree import TreeBuilder
from xml.parsers import expat


MAX_XODR_BYTES = 8 * 1024 * 1024
MAX_XML_ELEMENTS = 100_000
MAX_XML_DEPTH = 64
SCHEMA_VERSION = "l2d.realsim.raw_xodr_speed.v0.2"


def _parse_attributes_xml(data: bytes):
    """Parse without text/external resources; reject DTDs at parser level.

    Expat's declaration callbacks work for UTF-16 as well as UTF-8, unlike a
    byte-substring check. No external entity resolver or network IO is used.
    """
    parser = expat.ParserCreate()
    builder = TreeBuilder()
    count = depth = 0

    def reject_declaration(*_args):
        raise ValueError("OpenDRIVE DOCTYPE/ENTITY declarations are forbidden")

    def start(name, attributes):
        nonlocal count, depth
        count += 1
        depth += 1
        if count > MAX_XML_ELEMENTS or depth > MAX_XML_DEPTH:
            raise ValueError("OpenDRIVE XML exceeds element/depth budget")
        builder.start(name, attributes)

    def end(name):
        nonlocal depth
        builder.end(name)
        depth -= 1

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.StartDoctypeDeclHandler = reject_declaration
    parser.EntityDeclHandler = reject_declaration
    parser.ExternalEntityRefHandler = reject_declaration
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    try:
        parser.Parse(data, True)
    except expat.ExpatError as error:
        raise ValueError("invalid OpenDRIVE XML") from error
    root = builder.close()
    if root.tag != "OpenDRIVE":
        raise ValueError("expected an unprefixed OpenDRIVE root")
    return root


def _speeds(element) -> list[dict]:
    # Direct children only: vendor extensions and unrelated speed tags do not
    # silently become road/lane speed rules. Empty list means absent element;
    # [{}] means a speed element was present but had no attributes.
    return [dict(speed.attrib) for speed in element.findall("speed")]


def _raw_attribute_tree(element) -> dict:
    """Keep ordered XML attribute subtrees, without interpreting extensions.

    In particular userData/originalFeatureId, validity and dependency are not
    promoted to verified identity/control bindings. XML text is not captured.
    """
    return {
        "tag": element.tag,
        "raw_attributes": dict(element.attrib),
        "children": [_raw_attribute_tree(child) for child in element],
    }


def read_xodr_speed_metadata(zip_file: Any) -> dict:
    """Read exactly one bounded root ``map.xodr`` from an already-open ZIP.

    Raises ValueError for missing/duplicate/oversized members or unsafe/invalid
    XML. ZIP decompression/CRC errors propagate. Reads no USD or lane parquet.
    Repeated IDs/elements stay in source order; missing attributes stay absent.
    ``raw_attributes`` retains header version/vendor/date, road IDs, type ``s``,
    laneSection ``s``, lane IDs, and speed ``max``/``unit``/``sOffset`` verbatim
    at the XML attribute-value level. v0.2 adds ordered ``signal_groups`` on
    roads and root ``controllers`` with their raw attribute subtrees; all v0.1
    speed fields keep their shape. No default units or values are inserted.
    ``dynamic=yes`` is only an attribute, never a time-varying signal state.
    """
    members = [member for member in zip_file.infolist() if member.filename == "map.xodr"]
    if len(members) != 1:
        raise ValueError("map.xodr is missing or duplicated")
    member = members[0]
    if member.file_size > MAX_XODR_BYTES:
        raise ValueError("map.xodr exceeds metadata byte budget")
    with zip_file.open(member) as stream:
        data = stream.read(MAX_XODR_BYTES + 1)
    if len(data) > MAX_XODR_BYTES:
        raise ValueError("map.xodr exceeds metadata byte budget")
    root = _parse_attributes_xml(data)
    roads = []
    for road_index, road in enumerate(root.findall("road")):
        sections = []
        for lanes_index, lanes in enumerate(road.findall("lanes")):
            for section_index, section in enumerate(lanes.findall("laneSection")):
                lane_records = []
                for group_index, group in enumerate(section):
                    if group.tag not in {"left", "center", "right"}:
                        continue
                    for lane_index, lane in enumerate(group.findall("lane")):
                        lane_records.append({
                            "group_index": group_index,
                            "group": group.tag,
                            "raw_group_attributes": dict(group.attrib),
                            "lane_index": lane_index,
                            "raw_attributes": dict(lane.attrib),
                            "speeds": _speeds(lane),
                        })
                sections.append({
                    "lanes_index": lanes_index,
                    "raw_lanes_attributes": dict(lanes.attrib),
                    "lane_section_index": section_index,
                    "raw_attributes": dict(section.attrib),
                    "lanes": lane_records,
                })
        roads.append({
            "road_index": road_index,
            "raw_attributes": dict(road.attrib),
            "types": [{"type_index": index, "raw_attributes": dict(kind.attrib),
                       "speeds": _speeds(kind)}
                      for index, kind in enumerate(road.findall("type"))],
            "lane_sections": sections,
            "signal_groups": [
                {
                    "signal_group_index": group_index,
                    "raw_attributes": dict(group.attrib),
                    "elements": [
                        {"source_child_index": child_index,
                         **_raw_attribute_tree(child)}
                        for child_index, child in enumerate(group)
                        if child.tag in {"signal", "signalReference"}
                    ],
                }
                for group_index, group in enumerate(road.findall("signals"))
            ],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "preserved_raw_metadata",
        "source": {
            "artifact_path": str(zip_file.filename) if zip_file.filename is not None else None,
            "member": member.filename,
            "member_size_bytes": member.file_size,
            "raw_root_attributes": dict(root.attrib),
            "headers": [dict(header.attrib) for header in root.findall("header")],
        },
        "roads": roads,
        "controllers": [
            {"source_root_child_index": child_index, **_raw_attribute_tree(child)}
            for child_index, child in enumerate(root) if child.tag == "controller"
        ],
        "clipgt_lane_mapping": None,
        "legality_verdict": None,
    }
