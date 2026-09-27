"""Explicit keep/change plan hypotheses; no change to map or acceptance gates."""
from .lane_path_search import retimed_lane_path
from .bilateral_lane import hold_completed_stop_heading


def plan_keep_change_pair(times, recorded_xyz, recorded_yaws, source_corridor,
                          target_corridor, parameters, *, return_timing=None,
                          hold_stop_heading=False):
    """Generate independent source/adjacent plans with unchanged observed prefix.

    Heading holding is a new, explicit plan, not a correction of actual state.
    It applies only after each blend completes and forbids post-task returns.
    Every resulting plan still needs full road, turning and traffic checks.
    """
    if type(hold_stop_heading) is not bool:
        raise ValueError("stop-heading hypothesis must be an explicit boolean")
    if hold_stop_heading and return_timing is not None:
        raise ValueError("stop-heading hypothesis cannot be combined with a return")
    keep_parameters = dict(parameters, lane_change_start_s=parameters["intervention_s"])
    try:
        keep = retimed_lane_path(times, recorded_xyz, recorded_yaws, source_corridor, **keep_parameters)
    except ValueError as error:
        # The low-level routine calls either supplied lane its "target". Name
        # the actual arm so a source-lane endpoint is not misread as a failed
        # adjacent-lane or post-task-return hypothesis.
        raise ValueError(f"R0 keep/source plan rejected: {error}") from error
    returning = ({"return_path_xyz": keep[0], "return_start_s": return_timing["start_s"],
                  "return_duration_s": return_timing["duration_s"]} if return_timing is not None else {})
    try:
        change = retimed_lane_path(times, recorded_xyz, recorded_yaws, target_corridor,
                                   **parameters, **returning)
    except ValueError as error:
        raise ValueError(f"R1 change/adjacent plan rejected: {error}") from error
    paths = {"R0": keep, "R1": change}
    diagnostic = {}
    if hold_stop_heading:
        for arm, start in (("R0", parameters["intervention_s"]),
                           ("R1", parameters["lane_change_start_s"])):
            xyz, yaw = paths[arm]
            heading, record = hold_completed_stop_heading(
                times, xyz, yaw, start + parameters["lane_change_duration_s"])
            paths[arm] = xyz, heading
            diagnostic[arm] = record
    return paths, diagnostic
