"""Bind genuine speed-run previews while preserving missing recorded previews."""
from pathlib import Path

from .io import load_json


def bind_actual_speed_preview(root, candidate, run_dir):
    from .agent_curation import validate_agent_runs
    from .feasibility_summary import summarize_speed_pilot
    root, run = Path(root).resolve(), Path(run_dir).resolve()
    if not run.is_relative_to(root / ".runtime") or validate_agent_runs(root, [run]) != candidate:
        raise ValueError("matching project-local delegated actual speed run required")
    summary = summarize_speed_pilot(run)
    preview = root / ".runtime/curation_review/agent-feasibility" / candidate["proposal_key"] / run.name
    path = preview / "inventory.private.json"
    packet = load_json(path)
    if (packet.get("scope") != "human_review_packet_not_qualification"
        or packet.get("hypothesis_kind") != "speed" or packet.get("proposal_key") != candidate["proposal_key"]
        or packet.get("seed") != summary["pairs"][0]["seed"]
        or packet.get("curation_authority") != "agent_under_user_delegation_pending_final_review"
        or packet.get("final_human_review_pending") is not True
        or any(packet.get(k) is not False for k in
               ("continuous_support_reviewed", "legality_reviewed", "render_quality_reviewed"))):
        raise ValueError("matching explicitly unapproved actual preview packet required")
    frames = packet.get("frames", [])
    if not frames or {f.get("arm") for f in frames} != {"R0", "R1"}:
        raise ValueError("both actual arms require preview evidence")
    seen = set()
    for frame in frames:
        image = (preview / frame["preview_file"]).resolve()
        asl = Path(frame["source_asl"]).resolve()
        arm = summary["pairs"][0]["arms"][frame["arm"]]
        if (not image.is_relative_to(preview) or not image.is_file() or image in seen
            or image.stat().st_size != frame["preview_size_bytes"]
            or frame.get("preview_size_wh") != [576, 320]
            or not asl.is_relative_to(run / "rollouts") or not asl.is_file()
            or asl.parent.name != arm["session_uuid"]):
            raise ValueError("original exported frame and actual session binding required")
        seen.add(image)
    stat = path.stat()
    return {"scope": "actual_run_preview_not_original_recorded_preview_or_approval",
            "inventory_binding": {"path": str(path), "size_bytes": stat.st_size,
                                  "mtime_ns": str(stat.st_mtime_ns)},
            "run_dir": str(run), "preview_frames": len(frames),
            "original_recorded_preview_available": False, "human_reviewed": False,
            "qualified": False}
