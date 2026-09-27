"""Preserve bounded, raw ClipGT static traffic-control topology.

The ClipGT association schema is clip-scoped and does not contain a
``map_id_version``. This reader resolves typed IDs only inside the selected
artifact/clip, then separately checks versions and coordinate sources carried
by the resolved entity records. It never infers a traffic-light phase binding
or a formal safety verdict.
"""

from __future__ import annotations

import base64
import io
from typing import Any


ASSOCIATION_KINDS = ("LIGHT_TO_LANE", "WAIT_LINE_TO_LANE")
TABLE_MEMBERS = {
    "clip": "clip.parquet",
    "association": "association.parquet",
    "lane": "lane.parquet",
    "traffic_light": "traffic_light.parquet",
    "wait_line": "wait_line.parquet",
}
MAX_TABLE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 24 * 1024 * 1024


def _selected_members(zip_file: Any, map_directory: str) -> dict[str, Any]:
    selected = {}
    for table_name, filename in TABLE_MEMBERS.items():
        member_name = f"{map_directory}/{filename}"
        matches = [item for item in zip_file.infolist() if item.filename == member_name]
        if len(matches) != 1:
            raise ValueError(f"selected map {table_name} table is missing or duplicated")
        member = matches[0]
        if member.file_size > MAX_TABLE_BYTES:
            raise ValueError(f"selected map {table_name} table exceeds topology byte budget")
        selected[table_name] = member
    if sum(item.file_size for item in selected.values()) > MAX_TOTAL_BYTES:
        raise ValueError("selected map topology tables exceed total byte budget")
    return selected


def _table_payload(zip_file: Any, member: Any) -> tuple[Any, list[dict]]:
    import pyarrow.parquet as pq

    table = pq.ParquetFile(io.BytesIO(zip_file.read(member))).read(use_threads=False)
    return table, table.to_pylist()


def _schema_evidence(table: Any) -> dict:
    return {
        "arrow_schema_base64": base64.b64encode(
            table.schema.serialize().to_pybytes()
        ).decode("ascii"),
        "parquet_schema_metadata_base64": [
            {
                "key_base64": base64.b64encode(key).decode("ascii"),
                "value_base64": base64.b64encode(value).decode("ascii"),
            }
            for key, value in (table.schema.metadata or {}).items()
        ],
    }


def _entity_records(
    rows: list[dict], entity_field: str
) -> tuple[list[dict], dict[Any, list[int]]]:
    records = []
    by_id: dict[Any, list[int]] = {}
    for source_row_index, row in enumerate(rows):
        key = row.get("key") if isinstance(row, dict) else None
        entity = row.get(entity_field) if isinstance(row, dict) else None
        record = {
            "source_row_index": source_row_index,
            "raw_record": row,
            "raw_clip_id": key.get("clip_id") if isinstance(key, dict) else None,
            "raw_label_class_id": (
                key.get("label_class_id") if isinstance(key, dict) else None
            ),
            "raw_map_id": key.get("map_id") if isinstance(key, dict) else None,
            "raw_map_id_version": (
                key.get("map_id_version") if isinstance(key, dict) else None
            ),
            "raw_egomotion_label_class_id": (
                entity.get("egomotion_label_class_id")
                if isinstance(entity, dict)
                else None
            ),
            "raw_version": row.get("version") if isinstance(row, dict) else None,
        }
        records.append(record)
        identifier = record["raw_map_id"]
        # Missing identifiers must never become join keys.  In particular, an
        # empty string is a serialized missing value in some parquet exports;
        # indexing it would let two absent identities appear to match uniquely.
        if identifier not in (None, ""):
            by_id.setdefault(identifier, []).append(source_row_index)
    return records, by_id


def _raw_ids(association: Any, field: str) -> Any:
    if not isinstance(association, dict) or field not in association:
        return None
    value = association[field]
    return list(value) if isinstance(value, (list, tuple)) else value


def _resolve(
    raw_ids: Any,
    records: list[dict],
    rows_by_id: dict[Any, list[int]],
    *,
    selected_clip_id: Any,
    association_clip_id: Any,
    entity_table: str,
) -> list[dict]:
    if not isinstance(raw_ids, list):
        return []
    resolved = []
    for position, identifier in enumerate(raw_ids):
        raw_candidates = list(rows_by_id.get(identifier, []))
        clip_candidates = [
            index
            for index in raw_candidates
            if records[index]["raw_clip_id"] == selected_clip_id
        ]
        if association_clip_id in (None, ""):
            resolution = "unresolved"
            reason = "association_clip_id_missing"
            matches = []
        elif association_clip_id != selected_clip_id:
            resolution = "unresolved"
            reason = "association_clip_id_mismatch"
            matches = []
        elif not raw_candidates:
            resolution = "unresolved"
            reason = "no_raw_map_id_candidate"
            matches = []
        elif not clip_candidates:
            resolution = "unresolved"
            reason = "candidate_clip_id_mismatch"
            matches = []
        elif len(clip_candidates) == 1:
            resolution = "unique"
            reason = "unique_typed_clip_identity_match"
            matches = clip_candidates
        else:
            resolution = "ambiguous"
            reason = "multiple_typed_clip_identity_matches"
            matches = clip_candidates
        resolved.append(
            {
                "source_position": position,
                "raw_id": identifier,
                "entity_table": entity_table,
                "selected_clip_id": selected_clip_id,
                "association_clip_id": association_clip_id,
                "raw_candidate_record_indices": raw_candidates,
                "typed_clip_candidate_record_indices": clip_candidates,
                "entity_record_indices": matches,
                # Compatibility alias for the initial payload shape.
                "endpoint_record_indices": matches,
                "resolution": resolution,
                "resolution_reason": reason,
            }
        )
    return resolved


def _field_match(lane: dict, endpoint: dict, field: str) -> dict:
    lane_value = lane.get(field)
    endpoint_value = endpoint.get(field)
    if lane_value in (None, "") or endpoint_value in (None, ""):
        status = "missing"
    elif lane_value == endpoint_value:
        status = "match"
    else:
        status = "mismatch"
    return {"lane_value": lane_value, "endpoint_value": endpoint_value, "status": status}


def _binding(
    lane_resolution: list[dict],
    endpoint_resolution: dict,
    lane_records: list[dict],
    endpoint_records: list[dict],
    *,
    subject_matches_key: bool,
) -> dict:
    result = {
        "source_lane_structure_valid": bool(
            subject_matches_key
            and len(lane_resolution) == 1
            and lane_resolution[0]["resolution"] == "unique"
        ),
        "endpoint_identity_unique": endpoint_resolution["resolution"] == "unique",
        "map_id_version": None,
        "egomotion_label_class_id": None,
        "entity_identity_compatible": False,
        "compatibility_reason": "identity_not_unique_or_subject_key_mismatch",
    }
    if not result["source_lane_structure_valid"] or not result["endpoint_identity_unique"]:
        return result
    lane = lane_records[lane_resolution[0]["entity_record_indices"][0]]
    endpoint = endpoint_records[endpoint_resolution["entity_record_indices"][0]]
    result["map_id_version"] = _field_match(lane, endpoint, "raw_map_id_version")
    result["egomotion_label_class_id"] = _field_match(
        lane, endpoint, "raw_egomotion_label_class_id"
    )
    statuses = (
        result["map_id_version"]["status"],
        result["egomotion_label_class_id"]["status"],
    )
    if statuses == ("match", "match"):
        result["entity_identity_compatible"] = True
        result["compatibility_reason"] = "entity_version_and_coordinate_source_match"
    elif "missing" in statuses:
        result["compatibility_reason"] = "entity_version_or_coordinate_source_missing"
    else:
        result["compatibility_reason"] = "entity_version_or_coordinate_source_mismatch"
    return result


def read_static_topology(zip_file: Any, map_directory: str, *, scene_id: str) -> dict:
    """Read bounded ClipGT tables and preserve typed static associations."""
    members = _selected_members(zip_file, map_directory)
    tables = {}
    rows = {}
    for name, member in members.items():
        tables[name], rows[name] = _table_payload(zip_file, member)

    if len(rows["clip"]) != 1:
        raise ValueError("selected map clip table must contain exactly one row")
    clip_row = rows["clip"][0]
    clip_key = clip_row.get("key") if isinstance(clip_row, dict) else None
    selected_clip_id = clip_key.get("clip_id") if isinstance(clip_key, dict) else None
    expected_clip_id = scene_id.removeprefix("clipgt-")
    if selected_clip_id in (None, "") or selected_clip_id != expected_clip_id:
        raise ValueError("selected artifact clip identity does not match scene_id")

    entity_records = {}
    entity_indices = {}
    for name, field in (
        ("lane", "lane"),
        ("traffic_light", "traffic_light"),
        ("wait_line", "wait_line"),
    ):
        entity_records[name], entity_indices[name] = _entity_records(rows[name], field)

    associations = []
    kind_counts = {kind: 0 for kind in ASSOCIATION_KINDS}
    resolution_counts = {"unique": 0, "ambiguous": 0, "unresolved": 0}
    resolution_reason_counts: dict[str, int] = {}
    compatibility_counts = {"compatible": 0, "incompatible": 0}
    for source_row_index, row in enumerate(rows["association"]):
        key = row.get("key") if isinstance(row, dict) else None
        kind = key.get("kind") if isinstance(key, dict) else None
        if kind not in ASSOCIATION_KINDS:
            continue
        association = row.get("association") if isinstance(row, dict) else None
        raw_subjects = _raw_ids(association, "subjects")
        raw_objects = _raw_ids(association, "objects")
        endpoint_table = "traffic_light" if kind == "LIGHT_TO_LANE" else "wait_line"
        association_clip_id = key.get("clip_id") if isinstance(key, dict) else None
        association_map_id = key.get("map_id") if isinstance(key, dict) else None
        subject_resolution = _resolve(
            raw_subjects,
            entity_records["lane"],
            entity_indices["lane"],
            selected_clip_id=selected_clip_id,
            association_clip_id=association_clip_id,
            entity_table="lane",
        )
        object_resolution = _resolve(
            raw_objects,
            entity_records[endpoint_table],
            entity_indices[endpoint_table],
            selected_clip_id=selected_clip_id,
            association_clip_id=association_clip_id,
            entity_table=endpoint_table,
        )
        subject_matches_key = (
            isinstance(raw_subjects, list)
            and len(raw_subjects) == 1
            and raw_subjects[0] == association_map_id
        )
        bindings = [
            {
                "object_source_position": item["source_position"],
                "raw_object_id": item["raw_id"],
                **_binding(
                    subject_resolution,
                    item,
                    entity_records["lane"],
                    entity_records[endpoint_table],
                    subject_matches_key=subject_matches_key,
                ),
            }
            for item in object_resolution
        ]
        for item in subject_resolution + object_resolution:
            resolution_counts[item["resolution"]] += 1
            reason = item["resolution_reason"]
            resolution_reason_counts[reason] = resolution_reason_counts.get(reason, 0) + 1
        for binding in bindings:
            key_name = "compatible" if binding["entity_identity_compatible"] else "incompatible"
            compatibility_counts[key_name] += 1
        kind_counts[kind] += 1
        associations.append(
            {
                "source_row_index": source_row_index,
                "raw_record": row,
                "raw_kind": kind,
                "raw_clip_id": association_clip_id,
                "raw_label_class_id": (
                    key.get("label_class_id") if isinstance(key, dict) else None
                ),
                "raw_map_id": association_map_id,
                "raw_map_id_version": (
                    key.get("map_id_version") if isinstance(key, dict) else None
                ),
                "raw_version": row.get("version") if isinstance(row, dict) else None,
                "selected_clip_identity_match": association_clip_id == selected_clip_id,
                "raw_subject_matches_key_map_id": subject_matches_key,
                "raw_direction": {
                    "subject_field": "association.subjects",
                    "raw_subjects": raw_subjects,
                    "object_field": "association.objects",
                    "raw_objects": raw_objects,
                },
                "subject_table": "lane",
                "endpoint_table": endpoint_table,
                "endpoint_resolution": {
                    "subjects": subject_resolution,
                    "objects": object_resolution,
                },
                "typed_bindings": bindings,
            }
        )

    return {
        "schema_version": "l2d.realsim.raw_static_topology.v0.3",
        "status": "preserved_raw_static_topology",
        "source": {
            "artifact_path": str(zip_file.filename) if zip_file.filename is not None else None,
            "scene_id": scene_id,
            "map_directory": map_directory,
            "selected_clip_id": selected_clip_id,
            "raw_clip_record": clip_row,
            "tables": {
                name: {
                    "member": member.filename,
                    "member_size_bytes": member.file_size,
                    **_schema_evidence(tables[name]),
                }
                for name, member in members.items()
            },
        },
        "association_kinds_retained": list(ASSOCIATION_KINDS),
        "associations": associations,
        "entity_records": entity_records,
        # Compatibility alias for consumers of the initial endpoint-only payload.
        "endpoint_records": {
            name: entity_records[name] for name in ("traffic_light", "wait_line")
        },
        "summary": {
            "source_row_counts": {name: len(value) for name, value in rows.items()},
            "retained_association_count": len(associations),
            "retained_kind_counts": kind_counts,
            "entity_record_counts": {
                name: len(value) for name, value in entity_records.items()
            },
            "endpoint_record_counts": {
                name: len(entity_records[name]) for name in ("traffic_light", "wait_line")
            },
            "typed_resolution_counts": resolution_counts,
            "endpoint_resolution_counts": resolution_counts,
            "typed_resolution_reason_counts": resolution_reason_counts,
            "endpoint_resolution_reason_counts": resolution_reason_counts,
            "object_entity_compatibility_counts": compatibility_counts,
        },
        "stopline_signal_phase_binding": None,
        "formal_safety_verdict": None,
        "limitations": [
            "Association subjects and objects are preserved in source order and are not reinterpreted.",
            "Typed map_id resolution is scoped to the selected artifact clip; association rows do not define map_id_version.",
            "Resolved lane and endpoint records must independently match in map_id_version and egomotion coordinate source before compatibility is reported.",
            "A lane linked separately to a wait line and traffic lights does not establish a signal phase binding.",
            "These static tables contain no time-varying traffic-light state.",
        ],
    }
