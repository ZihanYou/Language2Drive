"""Preserve ClipGT lane metadata on newly loaded runtime maps.

The upstream geometry adapter drops lane.speed_limit.  This process-local
adapter retains raw records without interpreting units, zero, lane direction,
or applicability.  It does not change upstream files or certify legality.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from functools import wraps
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any


METADATA_ATTRIBUTE = "l2d_raw_lane_metadata"
MAX_LANE_TABLE_BYTES = 8 * 1024 * 1024
MAX_CACHE_BYTES = 32 * 1024 * 1024
CACHE_SCHEMA = "l2d.realsim.raw_lane_metadata_cache.v0.1"
logger = logging.getLogger(__name__)


def read_lane_metadata(zip_file: Any, map_directory: str, *, scene_id: str) -> dict:
    """Read only the selected map's bounded lane table, preserving raw values.

    All physical rows survive, including duplicate IDs and null IDs.  Source
    fields retain missing versus explicit null; geometry remains in the
    upstream VectorMap and is not duplicated in this metadata payload.
    """
    import pyarrow.parquet as pq

    member_name = f"{map_directory}/lane.parquet"
    members = [item for item in zip_file.infolist() if item.filename == member_name]
    if len(members) != 1:
        raise ValueError("selected map lane table is missing or duplicated")
    member = members[0]
    if member.file_size > MAX_LANE_TABLE_BYTES:
        raise ValueError("selected map lane table exceeds metadata byte budget")
    parquet = pq.ParquetFile(io.BytesIO(zip_file.read(member)))
    table = parquet.read(use_threads=False)
    records = []
    for index, row in enumerate(table.to_pylist()):
        lane = row.get("lane")
        lane_fields = (
            {key: value for key, value in lane.items()
             if key not in {"left_rail", "right_rail"}}
            if isinstance(lane, dict) else lane
        )
        records.append({
            "source_row_index": index,
            "source_fields": {key: value for key, value in row.items() if key != "lane"},
            "raw_lane_fields": lane_fields,
            "speed_limit_present": isinstance(lane, dict) and "speed_limit" in lane,
            "raw_speed_limit": lane.get("speed_limit") if isinstance(lane, dict) else None,
        })
    schema_metadata = [
        {"key_base64": base64.b64encode(key).decode("ascii"),
         "value_base64": base64.b64encode(value).decode("ascii")}
        for key, value in (table.schema.metadata or {}).items()
    ]
    return {
        "schema_version": "l2d.realsim.raw_lane_metadata.v0.1",
        "status": "preserved_raw_metadata",
        "source": {
            "artifact_path": str(zip_file.filename) if zip_file.filename is not None else None,
            "scene_id": scene_id,
            "map_directory": map_directory,
            "member": member_name,
            "member_size_bytes": member.file_size,
            "parquet_schema_metadata_base64": schema_metadata,
            "arrow_schema_base64": base64.b64encode(
                table.schema.serialize().to_pybytes()
            ).decode("ascii"),
        },
        "records": records,
        "legality_verdict": None,
    }


def attach_lane_metadata(vector_map: Any, metadata: dict, *, road_lane_type: Any) -> None:
    """Attach every raw row and retain ambiguous per-lane matches as a list."""
    payload = deepcopy(metadata)
    scene_id = payload["source"]["scene_id"]
    clip_id = scene_id.removeprefix("clipgt-")
    lanes = vector_map.elements.get(road_lane_type, {})
    matches: dict[Any, list] = {}
    for record in payload["records"]:
        key = record["source_fields"].get("key")
        if not isinstance(key, dict) or key.get("clip_id") != clip_id:
            continue
        identifier = key.get("map_id")
        if identifier is not None and identifier in lanes:
            matches.setdefault(identifier, []).append(record)
    for identifier, records in matches.items():
        # Do not overwrite raw values with a float, or collapse duplicate IDs.
        setattr(lanes[identifier], METADATA_ATTRIBUTE, deepcopy(records))
    setattr(vector_map, METADATA_ATTRIBUTE, payload)


def _metadata_cache_path(vector_map_path: Path) -> Path:
    return vector_map_path.with_name(vector_map_path.name + ".l2d_lane_metadata.json")


def _bounded_cache_bytes(path: Path) -> bytes:
    with path.open("rb") as source:
        data = source.read(MAX_CACHE_BYTES + 1)
    if len(data) > MAX_CACHE_BYTES:
        raise ValueError("map metadata cache byte budget exceeded")
    return data


def save_cached_lane_metadata(vector_map: Any, vector_map_path: Path) -> None:
    """Write raw metadata beside a newly written native trajdata protobuf.

    The digest binds only this bounded map cache, never an ASL or a model
    artifact. It prevents an old metadata sidecar attaching to changed map
    geometry. It is not semantic authority or a legality certification.
    """
    metadata = getattr(vector_map, METADATA_ATTRIBUTE, None)
    if metadata is None:
        return
    vector_map_path = Path(vector_map_path)
    cache = {
        "schema_version": CACHE_SCHEMA,
        "map_id": vector_map.map_id,
        "vector_map_sha256": hashlib.sha256(_bounded_cache_bytes(vector_map_path)).hexdigest(),
        "metadata": metadata,
    }
    encoded = json.dumps(cache, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_CACHE_BYTES:
        raise ValueError("map metadata cache byte budget exceeded")
    # Refuse silent conversion of an unsupported source type (e.g. tuple keys).
    if json.loads(encoded)["metadata"] != metadata:
        raise ValueError("raw lane metadata cannot be losslessly encoded as JSON")
    destination = _metadata_cache_path(vector_map_path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".",
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(encoded)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore_cached_lane_metadata(vector_map: Any, vector_map_path: Path, *,
                                 road_lane_type: Any) -> bool:
    """Restore a matching optional metadata cache; old caches remain usable."""
    vector_map_path = Path(vector_map_path)
    sidecar = _metadata_cache_path(vector_map_path)
    if not sidecar.exists():
        return False
    cache = json.loads(_bounded_cache_bytes(sidecar))
    if (cache.get("schema_version") != CACHE_SCHEMA
            or cache.get("map_id") != vector_map.map_id
            or cache.get("vector_map_sha256") != hashlib.sha256(
                _bounded_cache_bytes(vector_map_path)).hexdigest()):
        raise ValueError("raw lane metadata cache does not match vector map")
    attach_lane_metadata(vector_map, cache["metadata"], road_lane_type=road_lane_type)
    return True


def _install_cache_adapter() -> None:
    from trajdata.caching.df_cache import DataFrameCache
    from trajdata.maps.map_api import MapAPI
    from trajdata.maps.vec_map_elements import MapElementType

    original_save = DataFrameCache.finalize_and_cache_map
    if not getattr(original_save, "_l2d_preserves_raw_lane_metadata", False):
        @wraps(original_save)
        def save(cache_path: Path, vector_map: Any, map_params: dict) -> Any:
            _, map_path, _, _ = DataFrameCache.get_map_paths(
                cache_path, vector_map.env_name, vector_map.map_name)
            # This cache is explicitly being rewritten by the caller. Remove
            # its previous optional payload even when the new map lacks one;
            # identical geometry alone must never resurrect stale metadata.
            _metadata_cache_path(map_path).unlink(missing_ok=True)
            result = original_save(cache_path, vector_map, map_params)
            try:
                save_cached_lane_metadata(vector_map, map_path)
            except Exception as error:
                logger.warning("Raw lane metadata cache unavailable for %s: %s", map_path, error)
            return result

        save._l2d_preserves_raw_lane_metadata = True
        DataFrameCache.finalize_and_cache_map = staticmethod(save)

    original_get = MapAPI.get_map
    if not getattr(original_get, "_l2d_preserves_raw_lane_metadata", False):
        @wraps(original_get)
        def get(api: Any, map_id: str, *args: Any, **kwargs: Any) -> Any:
            vector_map = original_get(api, map_id, *args, **kwargs)
            if not hasattr(vector_map, METADATA_ATTRIBUTE):
                map_path = (api.unified_cache_path / vector_map.env_name / "maps"
                            / f"{vector_map.map_name}.pb")
                try:
                    restore_cached_lane_metadata(vector_map, map_path,
                                                 road_lane_type=MapElementType.ROAD_LANE)
                except Exception as error:
                    logger.warning("Raw lane metadata cache not restored for %s: %s", map_path, error)
            return vector_map

        get._l2d_preserves_raw_lane_metadata = True
        MapAPI.get_map = get


def install() -> None:
    """Enable Artifact loading and native cache preservation in this process.

    Missing/unsupported metadata is recorded on the map and logged, without
    adding a new run admission gate. No metadata is inferred for old caches.
    """
    from alpasim_utils.artifact import Artifact
    from trajdata.maps.vec_map_elements import MapElementType

    _install_cache_adapter()
    original = Artifact._load_parquet_map
    if getattr(original, "_l2d_preserves_raw_lane_metadata", False):
        return

    @wraps(original)
    def load_with_metadata(artifact: Any, zip_file: Any, map_directory: str) -> Any:
        vector_map = original(artifact, zip_file, map_directory)
        if vector_map is None:
            return None
        try:
            metadata = read_lane_metadata(zip_file, map_directory, scene_id=artifact.scene_id)
            attach_lane_metadata(vector_map, metadata, road_lane_type=MapElementType.ROAD_LANE)
        except Exception as error:
            # Preserve upstream runtime behavior while exposing the exact gap.
            setattr(vector_map, METADATA_ATTRIBUTE, {
                "schema_version": "l2d.realsim.raw_lane_metadata.v0.1",
                "status": "metadata_unavailable",
                "source": {"artifact_path": str(artifact.source),
                           "scene_id": artifact.scene_id,
                           "map_directory": map_directory},
                "records": [],
                "error": f"{type(error).__name__}: {error}",
                "legality_verdict": None,
            })
            logger.warning("Raw lane metadata unavailable for %s/%s: %s",
                           artifact.source, map_directory, error)
        # Keep independent source failures separate: an absent XODR must not
        # discard ClipGT lane metadata (or vice versa). All supplementary
        # records travel through the same map cache and runtime sidecar.
        from .raw_static_topology import read_static_topology
        from .raw_xodr_speed import read_xodr_speed_metadata

        supplemental = {}
        readers = {
            "static_topology": lambda: read_static_topology(
                zip_file, map_directory, scene_id=artifact.scene_id),
            "xodr_speed": lambda: read_xodr_speed_metadata(zip_file),
        }
        for name, reader in readers.items():
            try:
                supplemental[name] = reader()
            except Exception as error:
                supplemental[name] = {
                    "status": "metadata_unavailable",
                    "error": f"{type(error).__name__}: {error}",
                }
                logger.warning("Supplementary raw map source %s unavailable for %s: %s",
                               name, artifact.source, error)
        getattr(vector_map, METADATA_ATTRIBUTE)["supplemental_raw_sources"] = supplemental
        return vector_map

    load_with_metadata._l2d_preserves_raw_lane_metadata = True
    Artifact._load_parquet_map = load_with_metadata
