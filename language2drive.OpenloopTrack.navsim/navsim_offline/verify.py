"""Contract verifier for NAVSIM local rear-axle poses."""
import math

DT = 0.5
TIMES = [DT * i for i in range(1, 9)]


def normalize_prediction(value):
    """Accept NAVSIM poses or the shared output schema with optional headings."""
    if not isinstance(value, dict):
        return value
    if value.get("coordinate_frame") != "ego_at_anchor" or value.get("reference_point") != "rear_axle":
        raise ValueError("wrong coordinate frame or reference point")
    if value.get("timestamps") != TIMES:
        raise ValueError("timestamps must be 0.5 through 4.0 seconds")
    xy = value.get("positions_xy")
    if not isinstance(xy, list) or len(xy) != 8 or any(len(p) != 2 for p in xy):
        raise ValueError("positions_xy must have eight pairs")
    headings = value.get("headings")
    if headings is None:
        previous = [0.0, 0.0]
        headings = []
        for x, y in xy:
            headings.append(math.atan2(y - previous[1], x - previous[0]))
            previous = [x, y]
    if len(headings) != 8:
        raise ValueError("headings must have eight values")
    return [[x, y, heading] for (x, y), heading in zip(xy, headings)]


def validate_trajectory(poses, expected=8):
    if len(poses) != expected or any(len(p) != 3 for p in poses):
        raise ValueError("unexpected number or shape of (x, y, heading) poses")
    if any(not math.isfinite(float(x)) for p in poses for x in p):
        raise ValueError("trajectory contains non-finite values")
    speeds = []
    previous = (0.0, 0.0)
    for x, y, _ in poses:
        speed = math.hypot(x - previous[0], y - previous[1]) / DT
        speeds.append(speed)
        previous = (x, y)
    return speeds


def trajectory_quality(poses, expected=8):
    """Return basic planned-motion quality separately from language success."""
    speeds = validate_trajectory(poses, expected)
    accelerations = [(b - a) / DT for a, b in zip(speeds, speeds[1:])]
    yaw_rates = [abs(math.remainder(b[2] - a[2], 2 * math.pi)) / DT
                 for a, b in zip(poses, poses[1:])]
    max_acceleration = max((abs(value) for value in accelerations), default=0.0)
    max_yaw_rate = max(yaw_rates, default=0.0)
    return {"trajectory_valid": True,
            "max_speed_mps": max(speeds, default=0.0),
            "max_abs_acceleration_mps2": max_acceleration,
            "max_abs_yaw_rate_rps": max_yaw_rate,
            "kinematically_feasible": max_acceleration <= 3.5 and max_yaw_rate <= 0.8}


def verify_speed_band(poses, contract, horizon=4.0):
    expected = int(horizon / DT)
    speeds = validate_trajectory(poses[:expected], expected)
    start = contract["start_s"]
    end = min(contract["end_s"], horizon)
    selected = [v for t, v in zip(TIMES, speeds) if start <= t <= end]
    return bool(selected) and all(contract["min_mps"] <= v <= contract["max_mps"] for v in selected)


def verify_lane_region(poses, contract, horizon=4.0):
    """Require the full rear-axle-referenced vehicle footprint in a map lane."""
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon, box

    expected = int(horizon / DT)
    validate_trajectory(poses[:expected], expected)
    lane = Polygon(contract["polygon_xy"])
    if not lane.is_valid or lane.is_empty:
        return False
    selected = [(t, pose) for t, pose in zip(TIMES[:expected], poses[:expected])
                if contract["start_s"] <= t <= min(contract["end_s"], horizon)]
    if not selected or poses[expected - 1][0] < contract.get("min_progress_m", 0):
        return False
    footprint = box(-1.4, -1.0, 3.4, 1.0)
    for _, (x, y, heading) in selected:
        vehicle = translate(rotate(footprint, heading, origin=(0, 0), use_radians=True), x, y)
        if not lane.buffer(0.05).covers(vehicle):
            return False
    return True


def verify_stop_region(poses, contract, horizon=4.0):
    expected = int(horizon / DT)
    speeds = validate_trajectory(poses[:expected], expected)
    end = min(contract["end_s"], horizon)
    selected = [(pose, speed) for t, pose, speed in zip(TIMES[:expected], poses[:expected], speeds)
                if contract["start_s"] <= t <= end]
    if not selected:
        return False
    cx, cy = contract["center_xy"]
    radius = contract["radius_m"]
    return all(math.hypot(pose[0] - cx, pose[1] - cy) <= radius
               and speed <= contract["max_mps"] for pose, speed in selected)


def verify_goal_region(poses, contract, horizon=4.0):
    expected = int(horizon / DT)
    validate_trajectory(poses[:expected], expected)
    pose = poses[expected - 1]
    cx, cy = contract["center_xy"]
    position_ok = math.hypot(pose[0] - cx, pose[1] - cy) <= contract["radius_m"]
    heading_ok = abs(math.remainder(pose[2] - contract["heading_rad"], 2 * math.pi)) <= contract["heading_tolerance_rad"]
    return position_ok and heading_ok and pose[0] >= contract.get("min_progress_m", 0.0)


def verify_order(poses, contract, horizon=4.0):
    """Require the first low-speed and target-lane events in a fixed order."""
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon, box

    expected = int(horizon / DT)
    speeds = validate_trajectory(poses[:expected], expected)
    lane = Polygon(contract["target_polygon_xy"])
    if not lane.is_valid or lane.is_empty:
        return False
    footprint = box(-1.4, -1.0, 3.4, 1.0)
    low_time = None
    lane_time = None
    for time_s, speed, (x, y, heading) in zip(TIMES[:expected], speeds, poses[:expected]):
        if time_s < contract.get("start_s", 0.5) or time_s > contract.get("end_s", horizon):
            continue
        if low_time is None and speed <= contract["low_speed_mps"]:
            low_time = time_s
        if lane_time is None:
            vehicle = translate(rotate(footprint, heading, origin=(0, 0), use_radians=True), x, y)
            if lane.buffer(0.05).covers(vehicle):
                lane_time = time_s
    if low_time is None or lane_time is None:
        return False
    gap = contract.get("min_gap_s", 0.5)
    if contract["first"] == "slow":
        return low_time + gap <= lane_time
    if contract["first"] == "lane":
        return lane_time + gap <= low_time
    raise ValueError("unknown order event: " + str(contract["first"]))


def verify_object_speed(poses, contract, horizon=4.0):
    """Verify a speed band while the ego remains behind and near one tracked object."""
    expected = int(horizon / DT)
    speeds = validate_trajectory(poses[:expected], expected)
    targets = contract["target_positions_xy"][:expected]
    if len(targets) != expected:
        return False
    selected = [(pose, speed, target) for t, pose, speed, target in
                zip(TIMES[:expected], poses[:expected], speeds, targets)
                if contract["start_s"] <= t <= min(contract["end_s"], horizon)]
    if not selected:
        return False
    return all(contract["min_mps"] <= speed <= contract["max_mps"]
               and contract["min_gap_m"] <= math.hypot(target[0] - pose[0],
                                                        target[1] - pose[1]) <= contract["max_gap_m"]
               and target[0] - pose[0] >= contract["min_longitudinal_gap_m"]
               for pose, speed, target in selected)


def verify_pass_follow(poses, contract, horizon=4.0):
    """Verify pass or follow relative to one tracked lead object plus lane occupancy."""
    expected = int(horizon / DT)
    validate_trajectory(poses[:expected], expected)
    lane_contract = contract["lane_contract"]
    if not verify_lane_region(poses, lane_contract, horizon):
        return False
    targets = contract["target_positions_xy"][:expected]
    if len(targets) != expected:
        return False
    indices = [i for i, t in enumerate(TIMES[:expected])
               if contract["start_s"] <= t <= min(contract["end_s"], horizon)]
    if not indices:
        return False
    if contract["mode"] == "pass":
        i = indices[-1]
        return poses[i][0] - targets[i][0] >= contract["min_ahead_m"]
    if contract["mode"] == "follow":
        return all(contract["min_gap_m"] <= targets[i][0] - poses[i][0] <= contract["max_gap_m"]
                   for i in indices)
    raise ValueError("unknown pass/follow mode: " + str(contract["mode"]))


def contract_margin(poses, contract, horizon=4.0):
    """Signed near-miss margin for diagnostics; positive implies own-contract slack."""
    from shapely.affinity import rotate, translate
    from shapely.geometry import Point, Polygon, box

    expected = int(horizon / DT)
    speeds = validate_trajectory(poses[:expected], expected)
    end = min(contract.get("end_s", horizon), horizon)
    selected = [(t, pose, speed) for t, pose, speed in
                zip(TIMES[:expected], poses[:expected], speeds)
                if contract.get("start_s", 0.0) <= t <= end]
    if not selected:
        return -horizon
    kind = contract.get("type")
    if kind == "speed_band":
        return min(min(speed - contract["min_mps"], contract["max_mps"] - speed)
                   for _, _, speed in selected)
    if kind == "stop_region":
        center = Point(*contract["center_xy"])
        margins = [min(contract["radius_m"] - center.distance(Point(pose[0], pose[1])),
                       contract["max_mps"] - speed)
                   for _, pose, speed in selected]
        return min(margins)
    if kind == "goal_region":
        pose = poses[expected - 1]
        position_margin = contract["radius_m"] - math.hypot(
            pose[0] - contract["center_xy"][0], pose[1] - contract["center_xy"][1])
        heading_margin = contract["heading_tolerance_rad"] - abs(math.remainder(
            pose[2] - contract["heading_rad"], 2 * math.pi))
        return min(position_margin, heading_margin,
                   pose[0] - contract.get("min_progress_m", 0.0))
    if kind == "lane_region":
        lane = Polygon(contract["polygon_xy"])
        footprint = box(-1.4, -1.0, 3.4, 1.0)
        margins = []
        for _, (x, y, heading), _ in selected:
            vehicle = translate(rotate(footprint, heading, origin=(0, 0),
                                       use_radians=True), x, y)
            if lane.buffer(0.05).covers(vehicle):
                margins.append(lane.boundary.distance(vehicle))
            else:
                margins.append(-math.sqrt(max(vehicle.difference(lane.buffer(0.05)).area, 0.0)))
        margins.append(poses[expected - 1][0] - contract.get("min_progress_m", 0.0))
        return min(margins)
    if kind == "order":
        lane = Polygon(contract["target_polygon_xy"])
        footprint = box(-1.4, -1.0, 3.4, 1.0)
        low_time = lane_time = None
        for time_s, (x, y, heading), speed in selected:
            if low_time is None and speed <= contract["low_speed_mps"]:
                low_time = time_s
            if lane_time is None:
                vehicle = translate(rotate(footprint, heading, origin=(0, 0),
                                           use_radians=True), x, y)
                if lane.buffer(0.05).covers(vehicle):
                    lane_time = time_s
        if low_time is None or lane_time is None:
            return -horizon
        observed = lane_time - low_time if contract["first"] == "slow" else low_time - lane_time
        return observed - contract.get("min_gap_s", 0.5)
    if kind == "object_speed":
        speed_margin = min(min(speed - contract["min_mps"], contract["max_mps"] - speed)
                           for _, pose, speed in selected)
        gap_margin = min(min(math.hypot(target[0] - pose[0], target[1] - pose[1]) - contract["min_gap_m"],
                             contract["max_gap_m"] - math.hypot(target[0] - pose[0], target[1] - pose[1]),
                             target[0] - pose[0] - contract["min_longitudinal_gap_m"])
                         for (_, pose, speed), target in zip(selected, contract["target_positions_xy"][
                             max(0, int(contract["start_s"] / DT) - 1):expected]))
        return min(speed_margin, gap_margin)
    if kind == "pass_follow":
        lane_margin = contract_margin(poses, contract["lane_contract"], horizon)
        indices = [i for i, t in enumerate(TIMES[:expected])
                   if contract["start_s"] <= t <= min(contract["end_s"], horizon)]
        if contract["mode"] == "pass":
            relation_margin = poses[indices[-1]][0] - contract["target_positions_xy"][indices[-1]][0] - contract["min_ahead_m"]
        else:
            relation_margin = min(min(contract["target_positions_xy"][i][0] - poses[i][0] - contract["min_gap_m"],
                                      contract["max_gap_m"] - (contract["target_positions_xy"][i][0] - poses[i][0]))
                                  for i in indices)
        return min(lane_margin, relation_margin)
    raise ValueError("unknown contract type: " + str(kind))


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


def _range_completion(value, lower, upper, scale=None):
    """Soft membership in a closed interval, with exact score 1 inside it."""
    scale = max(float(scale if scale is not None else upper - lower), 1e-6)
    if value < lower:
        return _clip01(1.0 - (lower - value) / scale)
    if value > upper:
        return _clip01(1.0 - (value - upper) / scale)
    return 1.0


def _lower_completion(value, threshold, scale):
    if value >= threshold:
        return 1.0
    return _clip01(1.0 - (threshold - value) / max(float(scale), 1e-6))


def _upper_completion(value, threshold, scale):
    if value <= threshold:
        return 1.0
    return _clip01(1.0 - (value - threshold) / max(float(scale), 1e-6))


def _lane_completion(poses, contract, horizon):
    """Continuous lane score from vehicle-footprint overlap and forward progress."""
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon, box

    expected = int(horizon / DT)
    validate_trajectory(poses[:expected], expected)
    lane = Polygon(contract["polygon_xy"])
    if not lane.is_valid or lane.is_empty:
        return 0.0
    selected = [(t, pose) for t, pose in zip(TIMES[:expected], poses[:expected])
                if contract["start_s"] <= t <= min(contract["end_s"], horizon)]
    if not selected:
        return 0.0
    footprint = box(-1.4, -1.0, 3.4, 1.0)
    buffered = lane.buffer(0.05)
    overlap = []
    for _, (x, y, heading) in selected:
        vehicle = translate(rotate(footprint, heading, origin=(0, 0), use_radians=True), x, y)
        overlap.append(_clip01(buffered.intersection(vehicle).area / vehicle.area))
    minimum_progress = float(contract.get("min_progress_m", 0.0))
    progress = (1.0 if minimum_progress <= 0 else
                _clip01(poses[expected - 1][0] / minimum_progress))
    return min(min(overlap), progress)


def task_completion(poses, contract, horizon=4.0):
    """Return frozen graded task completion C(contract, trajectory) in [0, 1].

    A contract that passes the binary verifier always receives 1. Outside the
    feasible set, primitive constraint scores decay linearly over one native
    tolerance/range and conjunctive constraints use their minimum. This makes
    the score a graded relaxation of the released binary contract.
    """
    expected = int(horizon / DT)
    speeds = validate_trajectory(poses[:expected], expected)
    end = min(contract.get("end_s", horizon), horizon)
    indices = [i for i, t in enumerate(TIMES[:expected])
               if contract.get("start_s", 0.0) <= t <= end]
    if not indices:
        return 0.0
    kind = contract.get("type")
    if kind == "speed_band":
        width = max(contract["max_mps"] - contract["min_mps"], 1.0)
        return min(_range_completion(speeds[i], contract["min_mps"],
                                     contract["max_mps"], width) for i in indices)
    if kind == "lane_region":
        return _lane_completion(poses, contract, horizon)
    if kind == "stop_region":
        cx, cy = contract["center_xy"]
        radius = max(float(contract["radius_m"]), 1e-6)
        values = []
        for i in indices:
            distance = math.hypot(poses[i][0] - cx, poses[i][1] - cy)
            position = _upper_completion(distance, radius, radius)
            stopped = _upper_completion(speeds[i], contract["max_mps"], 1.0)
            values.append(min(position, stopped))
        return min(values)
    if kind == "goal_region":
        pose = poses[expected - 1]
        radius = max(float(contract["radius_m"]), 1e-6)
        distance = math.hypot(pose[0] - contract["center_xy"][0],
                              pose[1] - contract["center_xy"][1])
        position = _upper_completion(distance, radius, radius)
        tolerance = max(float(contract["heading_tolerance_rad"]), 1e-6)
        heading_error = abs(math.remainder(
            pose[2] - contract["heading_rad"], 2 * math.pi))
        heading = _upper_completion(heading_error, tolerance, tolerance)
        minimum_progress = float(contract.get("min_progress_m", 0.0))
        progress = (1.0 if minimum_progress <= 0 else
                    _clip01(pose[0] / minimum_progress))
        return min(position, heading, progress)
    if kind == "order":
        from shapely.affinity import rotate, translate
        from shapely.geometry import Polygon, box

        lane = Polygon(contract["target_polygon_xy"])
        if not lane.is_valid or lane.is_empty:
            return 0.0
        footprint = box(-1.4, -1.0, 3.4, 1.0)
        low_time = lane_time = None
        for i in indices:
            if low_time is None and speeds[i] <= contract["low_speed_mps"]:
                low_time = TIMES[i]
            if lane_time is None:
                x, y, heading = poses[i]
                vehicle = translate(rotate(footprint, heading, origin=(0, 0),
                                           use_radians=True), x, y)
                if lane.buffer(0.05).covers(vehicle):
                    lane_time = TIMES[i]
        if low_time is None and lane_time is None:
            return 0.0
        if low_time is None or lane_time is None:
            return 0.5
        observed = (lane_time - low_time if contract["first"] == "slow"
                    else low_time - lane_time)
        gap = float(contract.get("min_gap_s", 0.5))
        return _clip01((observed + DT) / (gap + DT))
    if kind == "object_speed":
        targets = contract["target_positions_xy"][:expected]
        if len(targets) != expected:
            return 0.0
        speed_width = max(contract["max_mps"] - contract["min_mps"], 1.0)
        gap_width = max(contract["max_gap_m"] - contract["min_gap_m"], 1.0)
        values = []
        for i in indices:
            target = targets[i]
            gap = math.hypot(target[0] - poses[i][0], target[1] - poses[i][1])
            longitudinal = target[0] - poses[i][0]
            values.append(min(
                _range_completion(speeds[i], contract["min_mps"],
                                  contract["max_mps"], speed_width),
                _range_completion(gap, contract["min_gap_m"],
                                  contract["max_gap_m"], gap_width),
                _lower_completion(longitudinal, contract["min_longitudinal_gap_m"],
                                  max(contract["max_gap_m"], 1.0))))
        return min(values)
    if kind == "pass_follow":
        lane_score = _lane_completion(poses, contract["lane_contract"], horizon)
        targets = contract["target_positions_xy"][:expected]
        if len(targets) != expected:
            return 0.0
        if contract["mode"] == "pass":
            i = indices[-1]
            delta = poses[i][0] - targets[i][0]
            relation = _lower_completion(delta, contract["min_ahead_m"],
                                         max(abs(targets[indices[0]][0]),
                                             contract["min_ahead_m"], 1.0))
        elif contract["mode"] == "follow":
            width = max(contract["max_gap_m"] - contract["min_gap_m"], 1.0)
            relation = min(_range_completion(targets[i][0] - poses[i][0],
                                              contract["min_gap_m"],
                                              contract["max_gap_m"], width)
                           for i in indices)
        else:
            raise ValueError("unknown pass/follow mode: " + str(contract["mode"]))
        return min(lane_score, relation)
    raise ValueError("unknown contract type: " + str(kind))


def verify_contract(poses, contract, horizon=4.0):
    kind = contract.get("type")
    if kind == "speed_band":
        return verify_speed_band(poses, contract, horizon)
    if kind == "lane_region":
        return verify_lane_region(poses, contract, horizon)
    if kind == "stop_region":
        return verify_stop_region(poses, contract, horizon)
    if kind == "goal_region":
        return verify_goal_region(poses, contract, horizon)
    if kind == "order":
        return verify_order(poses, contract, horizon)
    if kind == "object_speed":
        return verify_object_speed(poses, contract, horizon)
    if kind == "pass_follow":
        return verify_pass_follow(poses, contract, horizon)
    raise ValueError("unknown contract type: " + str(kind))


def exclusive_success(poses, own, opposite, horizon=4.0):
    validate_trajectory(poses)
    return verify_contract(poses, own, horizon) and not verify_contract(poses, opposite, horizon)


def score_pcus(sidecars, predictions, horizon=4.0):
    """predictions: {pcu_id: {"0": poses, "1": poses}}."""
    results = {}
    for pcu in sidecars:
        pcu_id = pcu["pcu_id"]
        arms = predictions.get(pcu_id, {})
        successes = []
        details = []
        for arm in (0, 1):
            try:
                poses = normalize_prediction(arms[str(arm)])
                own = pcu["contracts"][arm]
                opposite = pcu["contracts"][1 - arm]
                expected = int(horizon / DT)
                quality = trajectory_quality(poses[:expected], expected)
                own_success = verify_contract(poses, own, horizon)
                opposite_success = verify_contract(poses, opposite, horizon)
                success = int(own_success and not opposite_success)
                if success:
                    failure_code = None
                elif own_success and opposite_success:
                    failure_code = "O3_AMBIGUOUS"
                elif opposite_success:
                    failure_code = "O2_WRONG_SEMANTIC"
                elif own.get("type") == "speed_band":
                    failure_code = "O5_QUANTITATIVE"
                elif own.get("type") == "order":
                    failure_code = "O6_SCOPE_ORDER"
                else:
                    failure_code = "O1_NO_COMMIT"
                successes.append(success)
                details.append({"semantic_success": bool(success),
                                "own_contract": own_success,
                                "opposite_contract": opposite_success,
                                "contract_margin": contract_margin(poses, own, horizon),
                                "failure_code": failure_code,
                                **quality})
            except (KeyError, TypeError, ValueError, IndexError, ImportError):
                successes.append(0)
                details.append({"semantic_success": False, "own_contract": False,
                                "opposite_contract": False, "contract_margin": None,
                                "failure_code": "O0_INVALID_OUTPUT",
                                "trajectory_valid": False,
                                "kinematically_feasible": False})
        results[pcu_id] = {"arms": successes, "paired": successes[0] * successes[1],
                           "arm_details": details}
    n = len(results)
    arm_details = [detail for row in results.values() for detail in row["arm_details"]]
    margins = [detail["contract_margin"] for detail in arm_details
               if detail["contract_margin"] is not None]
    failure_counts = {}
    for detail in arm_details:
        code = detail["failure_code"] or "SUCCESS"
        failure_counts[code] = failure_counts.get(code, 0) + 1
    return {
        "n_pcu": n,
        "ol_sem_sisr": sum(sum(x["arms"]) for x in results.values()) / (2 * n) if n else None,
        "ol_sem_cfps": sum(x["paired"] for x in results.values()) / n if n else None,
        "trajectory_valid_rate": (sum(d["trajectory_valid"] for d in arm_details) /
                                  len(arm_details)) if arm_details else None,
        "kinematic_feasibility_rate": (sum(d["kinematically_feasible"] for d in arm_details) /
                                       len(arm_details)) if arm_details else None,
        "contract_margin": {"mean": sum(margins) / len(margins), "minimum": min(margins)}
                           if margins else {"mean": None, "minimum": None},
        "failure_counts": failure_counts,
        "pcus": results,
    }
