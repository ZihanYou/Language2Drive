"""Trusted language sidecar; only text/version/time cross the model boundary."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .io import load_json


@dataclass(frozen=True)
class InstructionSnapshot:
    text: str
    version: int
    issued_at_us: int


class LanguageConditionProvider:
    def get_instruction(self, session_uuid: str, time_query_us: int) -> InstructionSnapshot:
        raise NotImplementedError


class SidecarLanguageConditionProvider(LanguageConditionProvider):
    """Reads a private session map and returns the latest issued instruction."""

    def __init__(self, sidecar: Dict[str, Any]):
        if sidecar.get("schema_version") != "l2d.realsim.sidecar.v0.1":
            raise ValueError("invalid sidecar schema")
        sessions = sidecar.get("sessions")
        if not isinstance(sessions, dict):
            raise ValueError("sidecar.sessions must be an object")
        self._sessions = sessions

    @classmethod
    def from_path(cls, path: Path) -> "SidecarLanguageConditionProvider":
        return cls(load_json(path))

    def get_instruction(self, session_uuid: str, time_query_us: int) -> InstructionSnapshot:
        if session_uuid not in self._sessions:
            raise KeyError("unknown opaque session")
        schedule = self._sessions[session_uuid].get("instruction_schedule", [])
        current: Optional[Dict[str, Any]] = None
        for issuance in schedule:
            if int(issuance["issued_at_us"]) <= int(time_query_us):
                current = issuance
            else:
                break
        if current is None:
            return InstructionSnapshot(text="", version=0, issued_at_us=0)
        return InstructionSnapshot(
            text=str(current["text"]),
            version=int(current["version"]),
            issued_at_us=int(current["issued_at_us"]),
        )

    @staticmethod
    def model_payload(snapshot: InstructionSnapshot) -> Dict[str, Any]:
        """The complete model-visible language payload; no private IDs are included."""

        return {
            "text": snapshot.text,
            "version": snapshot.version,
            "issued_at_us": snapshot.issued_at_us,
        }
