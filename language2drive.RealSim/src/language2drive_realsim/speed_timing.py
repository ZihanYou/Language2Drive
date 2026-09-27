"""Explicit speed-pilot horizons; old evidence keeps its seven-second policy."""
from pathlib import Path
import math

import numpy as np
import yaml

from .io import load_json


def speed_timing(payload):
    timing = payload.get("speed_timing", {
        "rollout_duration_s": 7., "plan_horizon_s": 13., "intervention_s": 1.7})
    keys = {"rollout_duration_s", "plan_horizon_s", "intervention_s"}
    if (not isinstance(timing, dict) or set(timing) != keys
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in timing.values())):
        raise ValueError("explicit finite speed timing required")
    duration, horizon = timing["rollout_duration_s"], timing["plan_horizon_s"]
    if (not 7 <= duration <= 13 or abs(duration*10-round(duration*10)) > 1e-8
        or horizon != duration + 6. or timing["intervention_s"] != 1.7):
        raise ValueError("speed timing requires a 7–13s dense task and full six-second future")
    return dict(timing)


def validate_speed_runtime_timing(run_dir, sidecar):
    """Recheck opt-in timing against prepared runtime, plans and source stats.

    This reads small private evidence only. It grants no actual ASL authority.
    Missing opt-in timing retains the original seven-second compatibility path.
    """
    timing = speed_timing(sidecar)
    from .speed_scope_binding import selected_speed_scope
    scope_key, scope_reference = selected_speed_scope(sidecar)
    if "speed_timing" not in sidecar:
        if scope_reference is not None:
            raise ValueError("preselected landmark scope requires explicit speed timing")
        return timing
    root = Path(run_dir)
    launch = load_json(root / "launch.private.json")
    if launch.get("speed_hypothesis", {}).get("speed_timing") != timing:
        raise ValueError("speed runtime timing differs from launch")
    config = yaml.safe_load((root / "user-config.yaml").read_text())["simulation_config"]
    if (config.get("control_timestep_us") != 100000
        or config.get("n_sim_steps") != round(timing["rollout_duration_s"]*10)
        or config.get("force_gt_duration_us") != 1700000):
        raise ValueError("speed runtime configuration differs from full timing")
    binding = launch["candidate"]["scene_execution_proposal_not_frozen"]
    if binding["start_timestamp_us"] + round(timing["plan_horizon_s"]*1e6) >= binding["recording_end_timestamp_us"]:
        raise ValueError("speed full future exceeds original recording")
    reference = load_json(root / "speed-pair-plan.private.json")["diagnostic_search_reference"]
    path = Path(reference["path"])
    stat = path.stat()
    if stat.st_size != reference["size_bytes"] or str(stat.st_mtime_ns) != str(reference["mtime_ns"]):
        raise ValueError("speed timing source changed")
    search = load_json(path)
    if (search.get("speed_timing") != timing
        or selected_speed_scope(search) != (scope_key, scope_reference)
        or any(search.get(k) != v for k, v in timing.items())
        or search["start_timestamp_us"] != binding["start_timestamp_us"]):
        raise ValueError("speed timing differs from prepared search")
    if scope_reference is not None:
        scope_stat = Path(scope_reference["path"]).stat()
        if scope_stat.st_size != scope_reference["size_bytes"] or str(scope_stat.st_mtime_ns) != scope_reference["mtime_ns"]:
            raise ValueError("preselected landmark scope changed after preparation")
    for private in sidecar["sessions"].values():
        times = np.asarray([w["time_s"] for w in private["oracle_plan"]["waypoints"]], dtype=float)
        if (len(times) < 2 or not np.isfinite(times).all() or times[0] != 0
            or times[-1] < timing["plan_horizon_s"] or np.any(np.diff(times) <= 0)
            or np.any(np.diff(times) > .200001) or not np.any(times == 1.7)
            or private["start_timestamp_us"] != binding["start_timestamp_us"]
            or private["instruction_issued_at_us"] != binding["start_timestamp_us"] + 1700000):
            raise ValueError("speed plan lacks complete bound timing")
    return timing
