"""Keep unreviewed CPU search inputs distinct from prepared live runtimes."""
from pathlib import Path

from .io import load_json

OFFLINE_SCOPE = "unreviewed_offline_pair_search_input"


def load_search_input(run_dir, offline_candidate=None):
    if offline_candidate is None:
        launch = load_json(Path(run_dir) / "launch.private.json")
        if launch.get("scope") != "exploratory_runtime_not_oracle_witness":
            raise ValueError("prepared exploratory runtime required")
        return launch["candidate"], False
    payload = load_json(offline_candidate)
    candidate = payload["candidate"]
    if (payload.get("scope") != OFFLINE_SCOPE
        or payload.get("human_reviewed") is not False
        or payload.get("qualified") is not False
        or candidate.get("review_revision") is not None
        or "commands" in payload or "driver_binding" in payload):
        raise ValueError("unreviewed offline inputs cannot claim runtime or review authority")
    bindings = payload.get("source_bindings", [])
    if not bindings:
        raise ValueError("offline source bindings required")
    for binding in bindings:
        stat = Path(binding["path"]).stat()
        if stat.st_size != binding["size_bytes"] or str(stat.st_mtime_ns) != binding["mtime_ns"]:
            raise ValueError("offline source changed")
    return candidate, True
