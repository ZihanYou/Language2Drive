"""Explicit A4 pilot protocol choices, fixed before actual execution."""
from copy import deepcopy
import math
from pathlib import Path

from .io import load_json

LEGACY_VERSION = "bilateral_original_three_lane_exclusive_dwell_v1"
EXPLICIT_VERSION = "bilateral_original_three_lane_preselected_dwell_v2"


def validate_dwell_start(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 1.7 <= value <= 6.:
        raise ValueError("one-second dwell must start between intervention and six seconds")
    return float(value)


def bilateral_protocol(search):
    if "bilateral_protocol" not in search:
        if search.get("semantic_prefilter_version") != LEGACY_VERSION:
            raise ValueError("legacy A4 protocol version required")
        return None
    p = search["bilateral_protocol"]
    if (search.get("semantic_prefilter_version") != EXPLICIT_VERSION or not isinstance(p, dict)
        or set(p) != {"dwell_start_s", "post_task_return_arms"}):
        raise ValueError("exact explicit A4 protocol and version required")
    validate_dwell_start(p["dwell_start_s"])
    arms = p["post_task_return_arms"]
    if arms not in ([], ["R0"], ["R1"], ["R0", "R1"]):
        raise ValueError("ordered distinct A4 return arms required")
    ret = search.get("return_hypothesis")
    if bool(arms) != (ret is not None):
        raise ValueError("return arms and post-task return hypothesis differ")
    if ret is not None:
        if (not isinstance(ret, dict) or set(ret) != {"start_s", "duration_s"}
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in ret.values())
            or ret["start_s"] < 7.1 or ret["duration_s"] <= 0
            or ret["start_s"] + ret["duration_s"] > 13.):
            raise ValueError("return must follow the full task and finish within thirteen seconds")
    return deepcopy(p)


def validate_bilateral_runtime_protocol(run_dir):
    """Recheck bound regenerated plans/protocol before RPC or actual diagnostics."""
    run = Path(run_dir).resolve()
    launch = load_json(run / "launch.private.json")
    sidecar = load_json(run / "lane-pair-sidecar.private.json")
    plan = load_json(run / "lane-pair-plan.private.json")
    ref = plan["diagnostic_search_reference"]
    path = Path(ref["path"]).resolve()
    if path != run / "bilateral-preparation-recheck.private.json":
        raise ValueError("run-local regenerated A4 search required")
    stat = path.stat()
    if (stat.st_size, str(stat.st_mtime_ns)) != (ref["size_bytes"], str(ref["mtime_ns"])):
        raise ValueError("regenerated A4 evidence changed")
    search = load_json(path)
    p = bilateral_protocol(search)
    from .bilateral_pair_smoke import validate_bilateral_search
    plans = validate_bilateral_search(search, launch["candidate"])
    for document in (sidecar, plan, launch["bilateral_hypothesis"]):
        if document.get("bilateral_protocol") != p:
            raise ValueError("prepared A4 protocol binding differs")
    sessions = sidecar["sessions"]
    if sorted(v["arm_id"] for v in sessions.values()) != ["R0", "R1"]:
        raise ValueError("exact A4 session pair required")
    for private in sessions.values():
        if (private["oracle_plan"] != plans[private["arm_id"]]
            or private["start_timestamp_us"] != search["start_timestamp_us"]
            or private["instruction_issued_at_us"] != search["start_timestamp_us"] + 1700000):
            raise ValueError("A4 session plans/timing differ from regenerated evidence")
    if p is not None:
        import yaml
        config = yaml.safe_load((run / "user-config.yaml").read_text())["simulation_config"]
        if config["n_sim_steps"] != 70 or config["control_timestep_us"] != 100000 or config["force_gt_duration_us"] != 1700000:
            raise ValueError("explicit A4 protocol requires unchanged full task and prefix")
    return p
