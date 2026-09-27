"""Persist the runtime's raw map metadata beside a newly executed rollout.

This is a capture adapter, not a semantic provider. It consumes the same
rollout-metadata broadcast as the ASL writer and never infers speed units,
signal states, lane occupancy, or a legality verdict from raw map values.
"""

from __future__ import annotations

from functools import wraps
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from .raw_lane_metadata import METADATA_ATTRIBUTE, MAX_CACHE_BYTES


logger = logging.getLogger(__name__)
FILENAME = "runtime-map-evidence.private.json"
SCHEMA = "l2d.realsim.runtime_raw_map_evidence.v0.1"


def _publish_new_json(path: Path, payload: dict) -> None:
    """Atomically publish without replacing any pre-existing evidence file."""
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_CACHE_BYTES:
        raise ValueError("runtime map evidence exceeds byte budget")
    if json.loads(encoded) != payload:
        raise ValueError("runtime map evidence cannot be losslessly encoded as JSON")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(encoded)
        # Unlike replace(), link() refuses existing regular files and symlinks.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RuntimeMapEvidenceHandler:
    """Capture map identity when the real runtime broadcasts its metadata.

    Failed capture is explicit in the log and the handler's status, but does
    not disrupt model inference or the existing ASL handler. Missing or invalid
    evidence is never promoted to a completed formal semantic capture.
    """

    def __init__(self, unbound: Any, rollout_dir: str) -> None:
        self.unbound = unbound
        self.path = Path(rollout_dir) / FILENAME
        self.status = "awaiting_rollout_metadata"
        self.error: str | None = None

    async def on_message(self, message: Any) -> None:
        if message.WhichOneof("log_entry") != "rollout_metadata":
            return
        if self.status != "awaiting_rollout_metadata":
            return
        try:
            payload = self._payload(message.rollout_metadata)
            _publish_new_json(self.path, payload)
            self.status = payload["status"]
        except Exception as error:
            self.status = "capture_failed"
            self.error = f"{type(error).__name__}: {error}"
            logger.warning("Runtime raw map capture failed for %s: %s", self.path, self.error)

    def _payload(self, metadata: Any) -> dict:
        from google.protobuf.json_format import MessageToDict

        session = metadata.session_metadata
        if (session.session_uuid != str(self.unbound.rollout_uuid)
                or session.scene_id != self.unbound.scene_id):
            raise ValueError("runtime metadata does not match rollout identity")
        vector_map = self.unbound.vector_map
        raw = getattr(vector_map, METADATA_ATTRIBUTE, None)
        status = "raw_metadata_unavailable"
        if raw is not None:
            if not isinstance(raw, dict):
                raise ValueError("raw map metadata is not an object")
            if raw.get("legality_verdict") is not None:
                raise ValueError("raw metadata must not contain a legality verdict")
            if raw.get("status") == "preserved_raw_metadata":
                if (raw.get("schema_version") != "l2d.realsim.raw_lane_metadata.v0.1"
                        or not isinstance(raw.get("source"), dict)
                        or not isinstance(raw.get("records"), list)):
                    raise ValueError("unsupported or malformed raw map metadata")
                if raw["source"].get("scene_id") != session.scene_id:
                    raise ValueError("raw map metadata belongs to a different scene")
                status = "captured_raw_metadata"
        return {
            "schema_version": SCHEMA,
            "status": status,
            "capture_phase": "rollout_metadata_broadcast",
            "capture_source": "alpasim_runtime.EventBasedRollout.broadcaster",
            "session_metadata": MessageToDict(session, preserving_proto_field_name=True),
            "version_ids": MessageToDict(metadata.version_ids, preserving_proto_field_name=True),
            "rollout_identity": {
                "session_uuid": session.session_uuid,
                "scene_id": session.scene_id,
                "session_seed": int(self.unbound.session_seed),
                "vector_map_id": None if vector_map is None else vector_map.map_id,
            },
            "simulation_timing_us": {
                "egomotion_context_start": int(self.unbound.egomotion_context_start_us),
                "first_policy_timestamp": int(self.unbound.first_policy_timestamp_us),
                "closed_loop_start": int(self.unbound.closed_loop_start_us),
                "planned_end": int(self.unbound.end_timestamp_us),
            },
            "asl_relative_path": "rollout.asl",
            "raw_lane_metadata": raw,
            "dynamic_signal_states": None,
            "directed_lane_occupancy": None,
            "stopline_control_bindings": None,
            "speed_limit_units_and_applicability": None,
            "legality_verdict": None,
            "formal_score_eligible": False,
            "rollout_completion_certified": False,
        }


def install() -> None:
    """Attach to future EventBasedRollout objects in this interpreter only."""
    from alpasim_runtime.event_loop import EventBasedRollout

    original = EventBasedRollout.__post_init__
    if getattr(original, "_l2d_runtime_map_evidence", False):
        return

    @wraps(original)
    def initialize(rollout: Any) -> None:
        original(rollout)
        rollout.broadcaster.handlers.append(
            RuntimeMapEvidenceHandler(rollout.unbound, rollout._rollout_dir())
        )

    initialize._l2d_runtime_map_evidence = True
    EventBasedRollout.__post_init__ = initialize
