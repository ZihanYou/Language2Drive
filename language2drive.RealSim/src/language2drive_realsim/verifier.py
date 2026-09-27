"""Deterministic semantic verifiers for Layer A/B/C PCUs."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .events import SemanticEvent, derive_events, validate_trace
from .render_validity import NATIVE_MODE


@dataclass
class VerifierResult:
    passed: bool
    failure: Optional[str]
    first_commitment: Optional[Dict[str, Any]] = None
    secondary_tags: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _between(value: float, lower: float, upper: float, epsilon: float) -> bool:
    return lower - epsilon <= value <= upper + epsilon


def _unsafe_result(rows: Sequence[Dict[str, Any]], start_s: float) -> Optional[VerifierResult]:
    for row in rows:
        if row.get("infrastructure_failure"):
            return VerifierResult(False, "INVALID_SIM", diagnostics={"time_s": row["time_s"]})
        if not row["render_valid"]:
            if row.get("render_validity_mode") == NATIVE_MODE and row.get("image_black") is True:
                # Do not let a policy-induced unsupported view disappear into
                # infrastructure retries. Explicit infrastructure_failure above
                # remains the only override in the native-only protocol.
                return VerifierResult(
                    False, "F0_PREINTERVENTION" if float(row["time_s"]) < start_s else "F3_UNSAFE_FOLLOW",
                    secondary_tags=["native_black_image"], diagnostics={"time_s": row["time_s"]})
            # Leaving the curator-frozen reconstruction support can be caused
            # by the policy's semantic choice, so it is a scored unsafe outcome
            # and must never become a retryable infrastructure exclusion.
            if row.get("reconstruction_support_valid") is False:
                return VerifierResult(
                    False,
                    "F3_UNSAFE_FOLLOW",
                    secondary_tags=["outside_reconstruction_support"],
                    diagnostics={
                        "time_s": row["time_s"],
                        "reconstruction_deviation_m": row.get(
                            "reconstruction_deviation_m"
                        ),
                    },
                )
            return VerifierResult(False, "INVALID_SIM", diagnostics={"time_s": row["time_s"], "reason": "invalid_render"})
        if row["collision"] or row["offroad"]:
            failure = "F0_PREINTERVENTION" if float(row["time_s"]) < start_s else "F3_UNSAFE_FOLLOW"
            tags = []
            if row["collision"]:
                tags.append("collision")
            if row["offroad"]:
                tags.append("offroad")
            return VerifierResult(False, failure, secondary_tags=tags, diagnostics={"time_s": row["time_s"]})
    return None


def _progress_failure(rows: Sequence[Dict[str, Any]], minimum: float) -> Optional[VerifierResult]:
    maximum = max((float(row.get("progress_m", 0.0)) for row in rows), default=0.0)
    if maximum < minimum:
        return VerifierResult(False, "F10_ANTI_PROGRESS_PATTERN", diagnostics={"max_progress_m": maximum, "required_m": minimum})
    return None


def _first_event(
    events: Iterable[SemanticEvent], kind: str, subjects: Iterable[str], start_s: float
) -> Optional[SemanticEvent]:
    allowed = set(subjects)
    return next(
        (event for event in events if event.kind == kind and event.subject in allowed and event.time_s >= start_s),
        None,
    )


def _gate_verifier(
    contract: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    events: Sequence[SemanticEvent],
    window: Dict[str, Any],
    layer: str,
) -> VerifierResult:
    start = float(window["start_s"])
    deadline = min(float(window["deadline_s"]), float(contract["deadline_s"]))
    gates = list(window["commitment_gates"])
    first = _first_event(events, "CrossGate", gates, start)
    if first is None:
        return VerifierResult(False, "F1_NO_COMMIT")
    commitment = {"kind": first.kind, "subject": first.subject, "time_s": first.time_s}
    if first.time_s > deadline:
        return VerifierResult(False, "F6_ORDER_SCOPE", first_commitment=commitment, secondary_tags=["late"])
    if first.subject != contract["target_gate"]:
        failure = "F4_WRONG_REFERENT_OUTCOME" if layer == "C" else "F2_WRONG_SEMANTIC"
        return VerifierResult(False, failure, first_commitment=commitment)
    return VerifierResult(True, None, first_commitment=commitment)


def _stable_interval(
    rows: Sequence[Dict[str, Any]], lane: str, scope: str, required_s: float
) -> Optional[Tuple[float, float]]:
    start = None
    last = None
    for row in rows:
        in_scope = scope in row.get("scopes", [])
        occupied = lane in row.get("lane_ids", [])
        current = float(row["time_s"])
        if in_scope and occupied:
            if start is None:
                start = current
            last = current
            if last - start >= required_s:
                return start, last
        else:
            start = None
            last = None
    return None


def _lane_verifier(
    contract: Dict[str, Any], opposing: Dict[str, Any], rows: Sequence[Dict[str, Any]]
) -> VerifierResult:
    scope = contract["scope"]
    required = float(contract["stable_duration_s"])
    target = _stable_interval(rows, contract["target_lane"], scope, required)
    opposite = _stable_interval(rows, opposing["target_lane"], scope, required)
    if target is None and opposite is None:
        if not any(scope in row.get("scopes", []) for row in rows):
            return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["trigger_avoidance"])
        return VerifierResult(False, "F1_NO_COMMIT", secondary_tags=["partial_or_oscillation"])
    if opposite is not None and (target is None or opposite[1] <= target[1]):
        return VerifierResult(
            False,
            "F2_WRONG_SEMANTIC",
            first_commitment={"kind": "StableLane", "subject": opposing["target_lane"], "time_s": opposite[1]},
        )
    if target[1] > float(contract["deadline_s"]):
        return VerifierResult(
            False,
            "F6_ORDER_SCOPE",
            first_commitment={"kind": "StableLane", "subject": contract["target_lane"], "time_s": target[1]},
            secondary_tags=["late"],
        )
    return VerifierResult(
        True,
        None,
        first_commitment={"kind": "StableLane", "subject": contract["target_lane"], "time_s": target[1]},
    )


def _scoped_lane_verifier(contract, opposing, rows, events, window):
    """B2 retains one lane through the complete shared segment, not a short dwell."""
    entry_id, exit_id = contract.get("entry_gate"), contract.get("exit_gate")
    if not entry_id or not exit_id or entry_id == exit_id:
        return VerifierResult(False, "INVALID_SIM", diagnostics={"reason": "missing_distinct_lane_scope_gates"})
    start = float(window["start_s"])
    deadline = min(float(window["deadline_s"]), float(contract["deadline_s"]))
    entry = _first_event(events, "CrossGate", [entry_id], start)
    exit_event = _first_event(events, "CrossGate", [exit_id], start)
    if entry is None:
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["entry_not_crossed"])
    if exit_event is None:
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["incomplete_scope"])
    if exit_event.time_s <= entry.time_s:
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["wrong_scope_order"])
    if exit_event.time_s > deadline:
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["late"])
    scope = contract["scope"]
    traversal = [r for r in rows if entry.time_s <= float(r["time_s"]) < exit_event.time_s]
    if not traversal or any(scope not in r.get("scopes", []) for r in traversal):
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["left_scope_before_exit"])
    checked = [r for r in rows if entry.time_s <= float(r["time_s"]) <= exit_event.time_s
               and scope in r.get("scopes", [])]
    if len(checked) < 2 or exit_event.time_s - entry.time_s < float(contract["stable_duration_s"]):
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["insufficient_scope_duration"])
    violations = [r["time_s"] for r in checked if contract["target_lane"] not in r.get("lane_ids", [])
                  or opposing["target_lane"] in r.get("lane_ids", [])]
    if violations:
        return VerifierResult(False, "F2_WRONG_SEMANTIC", diagnostics={"nonexclusive_lane_times_s": violations})
    return VerifierResult(True, None, diagnostics={"entry_time_s": entry.time_s, "exit_time_s": exit_event.time_s,
        "shared_scope_traversed": True, "exclusive_lane_through_scope_exit": True, "evaluated_samples": len(checked)})


def _speed_verifier(
    contract: Dict[str, Any], rows: Sequence[Dict[str, Any]], epsilon: float,
    events: Sequence[SemanticEvent], window: Dict[str, Any],
) -> VerifierResult:
    # A few valid samples do not prove traversal of the frozen shared segment.
    # These registered private gates never enter the model-visible route.
    entry_id, exit_id = contract.get("entry_gate"), contract.get("exit_gate")
    if not entry_id or not exit_id or entry_id == exit_id:
        return VerifierResult(False, "INVALID_SIM", diagnostics={"reason": "missing_distinct_speed_scope_gates"})
    start = float(window["start_s"])
    deadline = min(float(window["deadline_s"]), float(contract["deadline_s"]))
    entry = _first_event(events, "CrossGate", [entry_id], start)
    exit_event = _first_event(events, "CrossGate", [exit_id], start)
    if entry is None:
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["entry_not_crossed"])
    if exit_event is None:
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["incomplete_scope"])
    if exit_event.time_s <= entry.time_s:
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["wrong_scope_order"])
    if exit_event.time_s > deadline:
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["late"])
    scope = contract["scope"]
    # The exit-crossing sample may lie just outside the boundary. All earlier
    # samples from entry onward must stay in scope; leaving and re-entering
    # cannot hide a violation or substitute a different exit.
    traversal = [row for row in rows if entry.time_s <= float(row["time_s"]) < exit_event.time_s]
    if not traversal or any(scope not in row.get("scopes", []) for row in traversal):
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["left_scope_before_exit"])
    scoped = [row for row in rows if entry.time_s <= float(row["time_s"]) <= exit_event.time_s
              and scope in row.get("scopes", [])]
    activation_time = None
    if contract["kind"] == "triggered_speed_band":
        # B3 binds a fixed visible referent to an activation region upstream of
        # (or at) the common speed-scope entry. Activation is latched: leaving
        # the referent's small region cannot hide later in-scope violations.
        trigger_id = contract["trigger_id"]
        activations = [float(row["time_s"]) for row in rows
                       if start <= float(row["time_s"]) <= entry.time_s
                       and trigger_id in row.get("triggers", [])]
        if not activations:
            return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["trigger_avoidance"])
        activation_time = min(activations)
    if not scoped:
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["trigger_avoidance"])
    if len(scoped) < 2:
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["insufficient_scope_samples"])
    missing = [row["time_s"] for row in scoped if row.get("speed_mps") is None]
    if missing:
        return VerifierResult(False, "INVALID_SIM", diagnostics={"missing_speed_at": missing})
    lower = float(contract["speed_min_mps"])
    upper = float(contract["speed_max_mps"])
    violating = [
        (float(row["time_s"]), float(row["speed_mps"]))
        for row in scoped
        if not _between(float(row["speed_mps"]), lower, upper, epsilon)
    ]
    if violating:
        return VerifierResult(False, "F5_QUANTITATIVE", diagnostics={"violations": violating, "band_mps": [lower, upper]})
    diagnostics = {"evaluated_samples": len(scoped), "band_mps": [lower, upper],
                          "entry_time_s": entry.time_s, "exit_time_s": exit_event.time_s,
                          "shared_scope_traversed": True}
    if activation_time is not None:
        diagnostics.update(trigger_activation_time_s=activation_time, trigger_latched_through_scope_exit=True)
    return VerifierResult(True, None, diagnostics=diagnostics)


def _dwell_interval(
    rows: Sequence[Dict[str, Any]], region: str, maximum_speed: float, dwell_s: float,
    *, scope: str,
) -> Optional[Tuple[float, float]]:
    start = None
    for row in rows:
        # Keep out-of-scope samples: removing them would stitch disjoint stops.
        valid = (
            scope in row.get("scopes", [])
            and region in row.get("regions", [])
            and float(row.get("speed_mps", float("inf"))) <= maximum_speed
        )
        current = float(row["time_s"])
        if valid:
            if start is None:
                start = current
            if current - start >= dwell_s:
                return start, current
        else:
            start = None
    return None


def _stop_verifier(
    contract: Dict[str, Any], opposing: Dict[str, Any], rows: Sequence[Dict[str, Any]], events: Sequence[SemanticEvent]
) -> VerifierResult:
    kind = contract["kind"]
    if not any(contract["scope"] in row.get("scopes", []) for row in rows):
        return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["trigger_avoidance"])
    if kind == "stop_region":
        interval = _dwell_interval(
            rows,
            contract["region"],
            float(contract["max_speed_mps"]),
            float(contract["dwell_s"]),
            scope=contract["scope"],
        )
        opposing_commitment = None
        if opposing.get("kind") == "continue_region" and opposing.get("target_gate"):
            opposing_commitment = _first_event(
                events, "CrossGate", [opposing["target_gate"]], 0.0
            )
        elif opposing.get("kind") == "stop_region":
            opposing_interval = _dwell_interval(
                rows,
                opposing["region"],
                float(opposing["max_speed_mps"]),
                float(opposing["dwell_s"]),
                scope=contract["scope"],
            )
            if opposing_interval is not None:
                return VerifierResult(
                    False,
                    "F2_WRONG_SEMANTIC",
                    first_commitment={"kind": "StopIn", "subject": opposing["region"], "time_s": opposing_interval[1]},
                )
        if opposing_commitment is not None and (
            interval is None or opposing_commitment.time_s <= interval[1]
        ):
            return VerifierResult(
                False,
                "F2_WRONG_SEMANTIC",
                first_commitment={"kind": "CrossGate", "subject": opposing_commitment.subject, "time_s": opposing_commitment.time_s},
            )
        if interval is None:
            return VerifierResult(False, "F1_NO_COMMIT", secondary_tags=["slow_creep_or_no_stop"])
        if interval[1] > float(contract["deadline_s"]):
            return VerifierResult(False, "F6_ORDER_SCOPE", secondary_tags=["late"])
        return VerifierResult(True, None, first_commitment={"kind": "StopIn", "subject": contract["region"], "time_s": interval[1]})
    # A continue arm commits by crossing its neutral progress gate without a
    # stop in the forbidden region. The target gate remains private.
    gate = contract.get("target_gate")
    first = _first_event(events, "CrossGate", [gate], 0.0) if gate else None
    forbidden = opposing.get("region")
    stopped = None
    if forbidden:
        stopped = _dwell_interval(
            rows,
            forbidden,
            float(opposing.get("max_speed_mps", 0.3)),
            float(opposing.get("dwell_s", 1.0)),
            scope=contract["scope"],
        )
    if stopped is not None and (first is None or stopped[1] <= first.time_s):
        return VerifierResult(False, "F2_WRONG_SEMANTIC", first_commitment={"kind": "StopIn", "subject": forbidden, "time_s": stopped[1]})
    if first is None:
        return VerifierResult(False, "F1_NO_COMMIT")
    if first.time_s > float(contract["deadline_s"]):
        return VerifierResult(False, "F6_ORDER_SCOPE", first_commitment={"kind": "CrossGate", "subject": first.subject, "time_s": first.time_s}, secondary_tags=["late"])
    return VerifierResult(True, None, first_commitment={"kind": "CrossGate", "subject": first.subject, "time_s": first.time_s})


def _pass_side_verifier(
    contract: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    window: Dict[str, Any],
) -> VerifierResult:
    actor = contract["actor_id"]
    start_s = float(window["start_s"])
    samples = [
        {
            "time_s": float(row["time_s"]),
            "longitudinal_m": float(row["relative_longitudinal_m"][actor]),
            "side": row.get("relative_side", {}).get(actor),
            "clearance_m": row.get("clearance_m", {}).get(actor),
        }
        for row in rows
        if float(row["time_s"]) >= start_s
        and actor in row.get("relative_longitudinal_m", {})
    ]
    if not samples:
        return VerifierResult(
            False,
            "INVALID_SIM",
            diagnostics={"reason": "missing_actor_longitudinal_state", "actor_id": actor},
        )

    ahead_index = next(
        (index for index, sample in enumerate(samples) if sample["longitudinal_m"] > 0.0),
        None,
    )
    if ahead_index is None:
        return VerifierResult(False, "F1_NO_COMMIT")

    behind_index = next(
        (
            index
            for index in range(ahead_index + 1, len(samples))
            if samples[index]["longitudinal_m"] < 0.0
        ),
        None,
    )
    if behind_index is None:
        return VerifierResult(False, "F1_NO_COMMIT")

    passing_samples = samples[ahead_index : behind_index + 1]
    crossing = min(passing_samples, key=lambda sample: abs(sample["longitudinal_m"]))
    commitment = {
        "kind": "PassActor",
        "subject": actor,
        "value": crossing["side"],
        "time_s": crossing["time_s"],
    }
    if crossing["side"] not in {"LEFT", "RIGHT"}:
        return VerifierResult(
            False, "INVALID_SIM", diagnostics={"reason": "missing_passing_side"}
        )
    if crossing["side"] != contract["side"]:
        return VerifierResult(False, "F2_WRONG_SEMANTIC", first_commitment=commitment)
    if crossing["time_s"] > float(contract["deadline_s"]):
        return VerifierResult(False, "F6_ORDER_SCOPE", first_commitment=commitment, secondary_tags=["late"])
    clearances = [
        float(sample["clearance_m"])
        for sample in passing_samples
        if sample["clearance_m"] is not None
    ]
    if len(clearances) != len(passing_samples):
        return VerifierResult(
            False,
            "INVALID_SIM",
            first_commitment=commitment,
            diagnostics={"reason": "missing_actor_polygon_clearance"},
        )
    if min(clearances) < float(contract["min_clearance_m"]):
        return VerifierResult(False, "F3_UNSAFE_FOLLOW", first_commitment=commitment, secondary_tags=["clearance"])
    return VerifierResult(True, None, first_commitment=commitment, diagnostics={"minimum_clearance_m": min(clearances)})


def verify_trace(
    pcu: Dict[str, Any],
    arm_id: str,
    trace: Iterable[Dict[str, Any]],
    annotations: Optional[Dict[str, Any]] = None,
    threshold_profile: Optional[Dict[str, Any]] = None,
) -> VerifierResult:
    """Evaluate one complete rollout; no learned judge is involved."""

    rows = validate_trace(trace)
    if not rows:
        return VerifierResult(False, "INVALID_SIM", diagnostics={"reason": "empty_trace"})
    maximum_gap = (threshold_profile or {}).get("maximum_sample_gap_s")
    if maximum_gap is not None:
        gaps = [
            float(current["time_s"]) - float(previous["time_s"])
            for previous, current in zip(rows, rows[1:])
        ]
        excessive = [gap for gap in gaps if gap > float(maximum_gap)]
        if excessive:
            return VerifierResult(
                False,
                "INVALID_SIM",
                diagnostics={
                    "reason": "trace_sample_gap",
                    "maximum_observed_gap_s": max(excessive),
                    "maximum_allowed_gap_s": float(maximum_gap),
                },
            )
    window = pcu["decision_window"]
    unsafe = _unsafe_result(rows, float(window["start_s"]))
    if unsafe is not None:
        return unsafe
    minimum_progress = float(pcu["hard_constraints"].get("minimum_progress_m", 0.0))
    progress = _progress_failure(rows, minimum_progress)
    if progress is not None:
        return progress
    arm = pcu["arms"][arm_id]
    contract = arm["semantic_contract"]
    opposing = arm["opposing_contract"]
    events = derive_events(rows, annotations or {})
    kind = contract["kind"]
    if kind == "gate":
        return _gate_verifier(contract, rows, events, window, pcu["primary_layer"])
    if kind == "lane":
        return _lane_verifier(contract, opposing, rows)
    if kind == "scoped_lane":
        return _scoped_lane_verifier(contract, opposing, rows, events, window)
    if kind in {"speed_band", "triggered_speed_band"}:
        epsilon = float((threshold_profile or {}).get("speed_epsilon_mps", 1e-6))
        return _speed_verifier(contract, rows, epsilon, events, window)
    if kind in {"stop_region", "continue_region"}:
        return _stop_verifier(contract, opposing, rows, events)
    if kind == "pass_side":
        return _pass_side_verifier(contract, rows, window)
    return VerifierResult(False, "INVALID_SIM", diagnostics={"reason": "unsupported_contract_kind"})
