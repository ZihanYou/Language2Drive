"""Original sign-relative B3 drafting, never implicit visual or legal approval."""
from pathlib import Path
import math

from .io import load_json
from .crosswalk_authoring import validate_actual_approach_frames
from .speed_scope_binding import selected_speed_scope, read_bound_sign_scope
from .speed_timing import validate_speed_runtime_timing


def sign_speed_utterances(lower_mps, upper_mps, far_m, near_m):
    if (not all(math.isfinite(v) for v in (lower_mps, upper_mps, far_m, near_m))
        or not 0 <= lower_mps < upper_mps or not 0 < near_m < far_m):
        raise ValueError("ordered finite speed and sign approach bands required")
    lower, upper = lower_mps * 3.6, upper_mps * 3.6
    return [
        f"As you approach the speed-limit sign ahead, keep {lower:g}–{upper:g} km/h in the current lane's stretch from approximately {far_m:g} to {near_m:g} metres before it.",
        f"On the approach to the upcoming speed-limit sign, maintain between {lower:g} and {upper:g} km/h along the current lane, from about {far_m:g} metres to {near_m:g} metres before the sign.",
        f"While travelling along the current lane's {far_m:g}-to-{near_m:g}-metre approach to the speed-limit sign ahead, hold a speed of {lower:g}–{upper:g} km/h.",
    ]


def validate_sign_authoring_inputs(root, candidate, native, approach_path, summary):
    path = Path(approach_path).resolve()
    reference = native.get("sign_scope_reference")
    if not reference or Path(reference["path"]).resolve() != path:
        raise ValueError("actual native sign scope must bind the same preselected approach")
    approach, _ = read_bound_sign_scope(reference, candidate)
    if (native["source_lane_id"] != approach["source_lane_ids"][0]
        or native["segment"] != approach["segment"]
        or native["annotations_draft"]["triggers"].get("draft-sign-activation") != approach["segment"]["scope_polygon"]
        or set(native["contracts_draft"]) != {"R0", "R1"}
        or any(c["kind"] != "triggered_speed_band" or c.get("trigger_id") != "draft-sign-activation"
               for c in native["contracts_draft"].values())):
        raise ValueError("actual native sign geometry and triggered contracts must match")
    for pair in summary["pairs"]:
        sidecar = load_json(Path(pair["run_dir"]) / "speed-pair-sidecar.private.json")
        timing = validate_speed_runtime_timing(pair["run_dir"], sidecar)
        if (selected_speed_scope(sidecar) != ("sign_scope_reference", reference)
            or native.get("speed_timing") != timing):
            raise ValueError("all actual pairs must bind the original sign and full timing")
    refs, bindings = validate_actual_approach_frames(root, candidate, native, summary)
    return approach, refs, bindings
