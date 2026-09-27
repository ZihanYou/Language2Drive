"""Join existing exploratory evidence, never create qualified witnesses."""
from pathlib import Path

from .constants import COUPLED_SEEDS
from .io import load_json
from .speed_timing import validate_speed_runtime_timing


def summarize_lane_seeds(run_dirs):
    return summarize_pair_seeds(run_dirs, "lane")


def summarize_speed_seeds(run_dirs):
    return summarize_pair_seeds(run_dirs, "speed")


def summarize_speed_pilot(run_dir):
    """One actual exploratory seed, explicitly not three-seed completion."""
    return _summarize_single_pilot(run_dir, "speed")


def summarize_lane_pilot(run_dir):
    """One actual lane pilot; retain later full-segment failures and all safety checks."""
    return _summarize_single_pilot(run_dir, "lane")


def _summarize_single_pilot(run_dir, pair_kind):
    sidecar = load_json(Path(run_dir) / (pair_kind + "-pair-sidecar.private.json"))
    seeds = {p["seed"] for p in sidecar["sessions"].values()}
    if len(seeds) != 1 or any(type(s) is not int or s not in COUPLED_SEEDS for s in seeds):
        raise ValueError("one supported actual coupled seed required")
    return summarize_pair_seeds([run_dir], pair_kind, expected_seeds=tuple(seeds))


def summarize_pair_seeds(run_dirs, pair_kind, *, expected_seeds=COUPLED_SEEDS):
    """Check small reports and file stats; do not read/hash ASLs or load models.

    This is a diagnostic index, not an independent semantic or integrity audit.
    Repeated deterministic trajectories across seeds remain one candidate.
    """
    semantic_scopes = {"lane": "lane_semantic_diagnostic_not_qualification",
                       "speed": "actual_speed_semantic_diagnostic_not_qualification"}
    if pair_kind not in semantic_scopes:
        raise ValueError("unsupported diagnostic pair kind")
    if expected_seeds != COUPLED_SEEDS and (len(expected_seeds) != 1 or type(expected_seeds[0]) is not int or expected_seeds[0] not in COUPLED_SEEDS):
        raise ValueError("only a single supported actual pilot may use the partial diagnostic path")
    if len(run_dirs) != len(expected_seeds):
        raise ValueError("exactly three paired run directories are required")
    rows, seeds, sessions_seen = [], set(), set()
    shared_candidate, shared_plans = None, None
    shared_speed_timing = None
    shared_crosswalk_scope = None
    bilateral_protocols = []
    for directory in map(lambda p: Path(p).resolve(), run_dirs):
        launch = load_json(directory / "launch.private.json")
        candidate = launch["candidate"]
        prefix = pair_kind + "-pair"
        sidecar = load_json(directory / (prefix + "-sidecar.private.json"))
        inspection = load_json(directory / (prefix + "-inspection.private.json"))
        semantic = load_json(directory / (prefix + "-semantics.private.json"))
        duration = 7.
        if pair_kind == "speed":
            timing = validate_speed_runtime_timing(directory, sidecar)
            duration = timing["rollout_duration_s"]
            if (shared_speed_timing is not None and shared_speed_timing != timing
                or "speed_timing" in sidecar and semantic.get("speed_timing") != timing):
                raise ValueError("speed timing differs across seeds or actual semantics")
            shared_speed_timing = timing
            scope_reference = sidecar.get("crosswalk_scope_reference")
            if rows and shared_crosswalk_scope != scope_reference:
                raise ValueError("preselected crosswalk scope differs across seeds")
            shared_crosswalk_scope = scope_reference
        if pair_kind == "lane":
            if "keep_change_hypothesis" in launch or "keep_change_search_reference" in sidecar:
                from .keep_change_pair_smoke import validate_keep_change_runtime_protocol
                validate_keep_change_runtime_protocol(directory)
                if (semantic.get("keep_change_search_reference") != sidecar["keep_change_search_reference"]
                    or semantic.get("linked_region_is_diagnostic_not_frozen_contract") is not True):
                    raise ValueError("keep/change actual semantics lack the exact bound planning-region diagnostic")
            recheck = directory / (prefix + "-semantics-linear-v2.private.json")
            if recheck.exists():
                latest = load_json(recheck)
                if (latest.get("proposal_key") != candidate["proposal_key"]
                    or any(latest.get(k, {}).get("passed") is not True for k in ("planned", "actual"))):
                    raise ValueError("later linear-v2 recheck prevents promotion of the old lane pilot")
                semantic = latest
        returned = load_json(directory / (prefix + "-return.private.json"))
        if any(r.get("hypothesis_kind", "lane") != pair_kind for r in (sidecar, inspection, semantic)):
            raise ValueError("mixed diagnostic pair kinds")
        if sidecar["execution_scope"] != "real_counterfactual_pair_integration_not_qualified_oracle":
            raise ValueError("only exploratory evidence can enter this diagnostic index")
        binding = {key: candidate[key] for key in (
            "proposal_key", "review_revision", "artifact_uuid", "scene_id_private", "selected_arm_pair")}
        if "curation_authority" in candidate:
            binding["curation_authority"] = candidate["curation_authority"]
            binding["human_reviewed"] = candidate["human_reviewed"]
        if shared_candidate is not None and shared_candidate != binding:
            raise ValueError("candidate, selected pair or review revision differs across seeds")
        shared_candidate = binding
        sessions = sidecar["sessions"]
        if sorted(p["arm_id"] for p in sessions.values()) != ["R0", "R1"]:
            raise ValueError("each run must contain exactly R0 and R1")
        pair_seeds = {p["seed"] for p in sessions.values()}
        if len(pair_seeds) != 1:
            raise ValueError("mixed coupled seeds within a pair")
        seed = pair_seeds.pop()
        if seed not in expected_seeds or seed in seeds:
            raise ValueError("missing, duplicate or unsupported seed")
        seeds.add(seed)
        plan_fields = ("oracle_plan", "start_timestamp_us", "instruction_issued_at_us")
        if pair_kind == "speed":
            plan_fields += ("target_speed_kph", "diagnostic_speed_band_mps")
        plans = {p["arm_id"]: {k: p[k] for k in plan_fields} for p in sessions.values()}
        if shared_plans is not None and plans != shared_plans:
            raise ValueError("reference paths or intervention timing differs across seeds")
        shared_plans = plans
        if (inspection.get("scope") != "actual_paired_ASL_diagnostics_not_qualified_oracle"
            or semantic.get("scope") != semantic_scopes[pair_kind]
            or any(r.get("qualified") is not False or r.get("proposal_key") != binding["proposal_key"]
                   for r in (inspection, semantic))):
            raise ValueError("unexpected diagnostic scope or candidate binding")
        if pair_kind == "lane":
            semantics_passed = all(semantic[k]["passed"] is True for k in ("planned", "actual"))
            if candidate["selected_arm_pair"].get("kind") == "left_vs_right":
                protocol = None
                if ("bilateral_protocol" in sidecar or "bilateral_protocol" in semantic
                    or (directory / "bilateral-preparation-recheck.private.json").exists()):
                    from .bilateral_protocol import validate_bilateral_runtime_protocol
                    protocol = validate_bilateral_runtime_protocol(directory)
                if semantic.get("bilateral_protocol") != protocol or (bilateral_protocols and bilateral_protocols[0] != protocol):
                    raise ValueError("A4 protocol differs across actual diagnostics or seeds")
                bilateral_protocols.append(protocol)
                if protocol is not None:
                    start = protocol["dwell_start_s"]
                    if (semantic["planned"].get("dwell_interval_s") != [start, start + 1.]
                        or semantic["actual"].get("final_lane_occupancy", {}).get("dwell_interval_s") != [start, start + 1.]):
                        raise ValueError("actual A4 dwell differs from prepared selection")
                if (sidecar.get("lane_pair_kind") != "left_vs_right"
                    or semantic.get("lane_pair_kind") != "left_vs_right"
                    or semantic.get("actual", {}).get("raw_preintervention_pose_rows_equal") is not True
                    or semantic.get("planned", {}).get("initial_and_instruction_in_original_middle") is not True
                    or not semantic.get("middle_corridor_lane_ids")):
                    raise ValueError("A4 requires bilateral actual semantics and original middle binding, not a keep/change report")
        else:
            results = semantic.get("arms", {})
            semantics_passed = (
                semantic.get("actual_speed_prefilter_passed") is True
                and semantic.get("preintervention_non_language_equal") is True
                and semantic.get("sampled_runtime_safety_passed") is True
                and set(results) == {"R0", "R1"}
                and all(results[p["arm_id"]].get("passed") is True
                        and results[p["arm_id"]].get("opposing_band_passed") is False
                        and results[p["arm_id"]].get("band_mps") == p["diagnostic_speed_band_mps"]
                        and results[p["arm_id"]].get("session_uuid") == session
                        for session, p in sessions.items()))
        if (inspection["errors"] or inspection["preintervention_non_language_equal"] is not True
            or inspection["sampled_runtime_safety_passed"] is not True
            or not semantics_passed):
            raise ValueError("a pair has not passed its exploratory checks")
        returns = returned["rollout_returns"]
        if (len(returns) != 2 or {r["session_uuid"] for r in returns} != set(sessions)
            or any(r["success"] is not True or r["error_code"] != 0 for r in returns)):
            raise ValueError("missing, mismatched or unsuccessful runtime return")
        arms = inspection["arms"]
        if set(arms) != {"R0", "R1"}:
            raise ValueError("missing actual arm inspection")
        for session, private in sessions.items():
            if session in sessions_seen:
                raise ValueError("session reused across seeds")
            sessions_seen.add(session)
            arm = arms[private["arm_id"]]
            asl = Path(arm["source_asl"]).resolve()
            expected = directory / "rollouts" / binding["scene_id_private"] / session / "rollout.asl"
            if asl != expected or arm["session_uuid"] != session:
                raise ValueError("ASL session/path mismatch")
            stat = asl.stat()
            if stat.st_size != arm["source_size_bytes"] or stat.st_mtime_ns != arm["source_mtime_ns"]:
                raise ValueError("ASL changed since diagnostic inspection")
            if "metrics_size_bytes" in arm:
                stat = asl.with_name("metrics.parquet").stat()
                if (stat.st_size != arm["metrics_size_bytes"]
                    or stat.st_mtime_ns != arm["metrics_mtime_ns"]):
                    raise ValueError("metrics changed since diagnostic inspection")
            if (arm["samples"] < 2 or arm["final_time_s"] < duration
                or any(arm[k] != 0 for k in ("collision_samples", "offroad_samples", "black_image_samples"))
                or arm["prefix"]["instruction_issued_at_us"] != private["instruction_issued_at_us"]):
                raise ValueError("incomplete or unsafe actual arm evidence")
        prefixes = [arms[a]["prefix"]["preintervention_non_language_sha256"] for a in ("R0", "R1")]
        if not prefixes[0] or prefixes[0] != prefixes[1]:
            raise ValueError("paired input-prefix reports disagree")
        review_base = "/feasibility/"
        if "agent_candidate_source" in launch:
            review_base = "/agent-feasibility/"
        elif launch.get("review_batch") is not None:
            review_base = "/additional-feasibility/" + launch["review_batch"] + "/"
        rows.append({"seed": seed, "run_dir": str(directory), "packet_name": directory.name,
                     "review_path": review_base + binding["proposal_key"] + "/" + directory.name + "/index.private.html",
                     "arms": {a: {k: arms[a][k] for k in (
                         "session_uuid", "samples", "maximum_reference_tracking_error_m",
                         "collision_samples", "offroad_samples", "black_image_samples")} for a in ("R0", "R1")}})
    return {"scope": "three_seed_diagnostic_index_not_qualification" if len(expected_seeds) == 3 else "single_seed_pilot_diagnostic_not_qualification", "qualified": False,
            "actual_pilot_seeds": sorted(seeds), "three_seed_pilot_complete": len(seeds) == 3,
            "hypothesis_kind": pair_kind,
            "candidate_binding": shared_candidate, "candidate_count": 1,
            "exploratory_rollout_count": len(sessions_seen), "formal_witnesses_created": 0,
            "checks": "existing_reports_and_file_stats_not_independent_integrity_attestation",
            "remaining_gates": ["human_legal_render_observability_review", "continuous_reconstruction_support",
                                "frozen_language_and_semantic_contracts", "formal_oracle_provenance_and_qualification"],
            "pairs": sorted(rows, key=lambda r: r["seed"])}
