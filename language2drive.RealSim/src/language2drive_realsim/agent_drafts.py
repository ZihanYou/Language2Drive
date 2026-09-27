"""Asset-bound authoring drafts, deliberately not compilable release PCUs.

The user authorizes independent authoring followed by one final human review.
An allocation is not a referent, a safe witness, or proof of task-family fit.
"""
from collections import Counter
from pathlib import Path
import uuid
from copy import deepcopy

from .constants import FAMILY_TARGETS, LAYER_TARGETS, COUPLED_SEEDS
from .curation_review import arm_pair_candidates
from .io import load_json
from .family_allocation import family_targets
from .release_profile import release_layer_targets


TEMPLATES = {
    "A1": ("在这个路口左转。", "在这个路口右转。"),
    "A2": ("在这个路口直行。", "在这个路口向{转弯方向}转弯。"),
    "A3": ("走分岔的左侧道路。", "走分岔的右侧道路。"),
    "A4": ("安全地换到左侧相邻车道。", "安全地换到右侧相邻车道。"),
    "A5": ("保持当前车道。", "安全地换到{相邻方向}侧车道。"),
    "A6": ("在前方合法停车区域靠边停下。", "继续沿当前道路行驶，不在该区域停车。"),
    "B1": ("通过指定路段时保持 23–27 km/h。", "通过同一路段时保持 28–32 km/h。"),
    "B2": ("在指定路段内优先使用{第一目标车道}。", "在同一路段内优先使用{第二目标车道}。"),
    "B3": ("经过{可见触发物}后，在指定路段保持{较低速度带}。", "经过同一触发物后，在同一路段保持{较高速度带}。"),
    "B4": ("经过{静态参照物}时保持{第一净距}。", "经过同一参照物时保持{第二净距}。"),
    "B5": ("停在{静态参照物}之前的合法区域。", "停在同一参照物之后的合法区域。"),
    "C1": ("沿指向{实际可读地名甲}的道路行驶。", "沿指向{实际可读地名乙}的道路行驶。"),
    "C2": ("驶入标有{实际可读道路或出口文字甲}的道路。", "驶入标有{实际可读道路或出口文字乙}的道路。"),
    "C3": ("走靠近{真实可见地标甲}的分支。", "走靠近{真实可见地标乙}的分支。"),
    "C4": ("驶向靠近{实际复合属性地标甲}的道路。", "驶向靠近{实际复合属性地标乙}的道路。"),
}


def validate_draft_batch(batch):
    rows = batch["drafts"]
    from .draft_coverage import draft_family_targets
    targets = draft_family_targets(batch)
    layers = release_layer_targets(batch)
    count = sum(layers.values())
    if "family_targets" in batch and batch["family_targets"] != targets:
        raise ValueError("displayed family targets disagree with declared allocation")
    if "layer_targets" in batch and batch["layer_targets"] != layers:
        raise ValueError("displayed layer targets disagree with declared allocation")
    if (batch.get("scope") != "asset_bound_authoring_drafts_not_qualified_pcus"
        or len(rows) != count or Counter(r["task_family"] for r in rows) != Counter(targets)
        or Counter(r["primary_layer"] for r in rows) != Counter(layers)
        or len({r["draft_id"] for r in rows}) != count
        or len({r["source_proposal_key"] for r in rows}) != count):
        raise ValueError("draft identity or exact task allocation mismatch")
    for row in rows:
        if (row["qualified"] is not False or row["final_human_review"] != "pending"
            or row["execution_ready"] is not False or row["formal_witnesses"] != []
            or set(row["arms"]) != {"R0", "R1"}
            or row["coupled_seeds"] != list(COUPLED_SEEDS)):
            raise ValueError("authoring draft must not fabricate execution or approval")
        if row["primary_layer"] == "C" and row.get("grounding_bindings") != {"R0": None, "R1": None}:
            raise ValueError("uninspected text/landmarks must remain explicitly unbound")


def build_draft_batch(root, *, c_family_reallocation=None, b4_family_reallocation=None, release_profile=None):
    root = Path(root).resolve()
    policy = load_json(root / "config/curation_workflow.private.json")
    if (policy["mode"] != "agent_authoring_then_final_human_review"
        or policy["agent_may_author_drafts"] is not True
        or policy["require_final_human_review_before_release"] is not True):
        raise ValueError("explicit agent-authoring authority required")
    inventory = load_json(root / "generated/curation_review_inventory.private.json")
    acquisition = load_json(root / "generated/acquisition_report.json")
    reviews = load_json(root / "data/private/curation_decisions.private.json")["decisions"]
    allocation = {"c_family_reallocation": c_family_reallocation} if c_family_reallocation is not None else {}
    if b4_family_reallocation is not None:
        allocation["b4_family_reallocation"] = b4_family_reallocation
    if release_profile is not None:
        allocation["release_profile"] = deepcopy(release_profile)
    targets = family_targets(allocation)
    layers = release_layer_targets(allocation)
    artifacts = {a["artifact_uuid"]: a["inspection"] for a in acquisition["artifacts"]}
    pool = []
    for scene in inventory["scenes"]:
        artifact = artifacts.get(scene["artifact_uuid"], {})
        if not artifact.get("identity_matches_catalog") or artifact.get("errors"):
            continue
        path = Path(artifact["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != artifact["size_bytes"]:
            continue
        if artifact["metadata"]["scene_id"] != scene["scene_id_private"]:
            raise ValueError("artifact scene mismatch")
        for proposal in scene["proposals"]:
            if reviews.get(proposal["proposal_key"], {}).get("decision") == "reject":
                continue
            if not proposal.get("evidence_rendered") or not proposal.get("map_evidence_rendered"):
                continue
            if any(not (root / proposal[key]).is_file() for key in ("evidence_sheet", "map_evidence_svg")):
                continue
            for pair in arm_pair_candidates(proposal):
                pool.append((scene, proposal, pair, artifact))
    used, scene_uses, rows = set(), Counter(), []
    # Allocate scarce branch leads before the more numerous lane opportunities.
    families = sorted(targets, key=lambda f: (not f.startswith(("C", "A1", "A2", "A3")), f))
    for family in families:
        code = family.split("_")[0]
        branch = code.startswith("C") or code in {"A1", "A2", "A3"}
        kinds = ({"recorded_vs_counterfactual_branch", "unobserved_branch_pair"} if branch else
                 {"left_vs_right"} if code == "A4" else {"keep_vs_left", "keep_vs_right"})
        for _ in range(targets[family]):
            choices = [v for v in pool if v[1]["proposal_key"] not in used and v[2]["kind"] in kinds]
            if not choices:
                raise ValueError("insufficient distinct real opportunities for " + family)
            scene, proposal, pair, artifact = min(choices, key=lambda v: (
                not v[1]["timing_review"]["automatic_timing_prefilter_pass"],
                v[1]["suggested_start_time_offset_us_not_qualified"] + 13000000 >= v[3]["metadata"]["duration_s"] * 1e6,
                scene_uses[v[0]["scene_id_private"]],
                reviews.get(v[1]["proposal_key"], {}).get("decision") != "keep",
                v[1]["proposal_key"], v[2]["arm_pair_id"]))
            used.add(proposal["proposal_key"])
            scene_uses[scene["scene_id_private"]] += 1
            suggested = proposal["suggested_start_time_offset_us_not_qualified"]
            missing = ["task_family_and_arm_mapping", "window_and_spatial_contract", "legal_action_and_balance",
                       "reconstruction_support", "safe_plans_all_seeds", "formal_oracle_witnesses",
                       "final_human_review"]
            if code.startswith("C"):
                missing += ["visible_semantic_referent_pair_at_576x320", "referent_to_distinct_gate_mapping"]
            if code in {"B1", "B3"}:
                missing += ["posted_speed_limit_basis", "shared_entry_exit_gates"]
            if code in {"A6", "B4", "B5"}:
                missing += ["static_object_or_legal_stopping_region"]
            row = {"draft_id": uuid.uuid4().hex, "primary_layer": family[0], "task_family": family,
                "source_proposal_key": proposal["proposal_key"], "scene_id_private": scene["scene_id_private"],
                "artifact_uuid": scene["artifact_uuid"], "artifact_path": artifact["path"],
                "artifact_size_bytes": artifact["size_bytes"],
                "suggested_start_offset_us_not_frozen": suggested,
                "recording_start_timestamp_us": artifact["metadata"]["start_timestamp_us"],
                "recording_duration_s": artifact["metadata"]["duration_s"],
                "original_timing_prefilter_passed": proposal["timing_review"]["automatic_timing_prefilter_pass"],
                "planning_horizon_13s_available_at_suggested_window": suggested + 13000000 < artifact["metadata"]["duration_s"] * 1e6,
                "topology_pair_hypothesis": pair, "family_binding_status": "unverified_hypothesis",
                "arms": {arm: {"instruction_template_not_executable": template,
                                "instruction_schedule": None, "semantic_contract": None}
                         for arm, template in zip(("R0", "R1"), TEMPLATES[code])},
                "grounding_bindings": {"R0": None, "R1": None},
                "evidence_sheet": proposal["evidence_sheet"], "map_evidence_svg": proposal["map_evidence_svg"],
                "coupled_seeds": list(COUPLED_SEEDS), "required_bindings": missing,
                "initial_human_decision_reference": reviews.get(proposal["proposal_key"], {}).get("decision"),
                "final_human_review": "pending", "formal_witnesses": [], "qualified": False,
                "execution_ready": False, "authoring_authority": "user_authorized_agent_not_human_signoff"}
            rows.append(row)
    batch = {"scope": "asset_bound_authoring_drafts_not_qualified_pcus", "drafts": rows,
             "draft_count": sum(layers.values()), "fully_instantiated_pcu_count": 0, "qualified_pcu_count": 0,
             "instruction_templates_are_not_model_inputs": True,
             "family_targets": targets, "layer_targets": layers, **deepcopy(allocation),
             "final_human_review_required": True, "authoring_policy": "config/curation_workflow.private.json"}
    validate_draft_batch(batch)
    return batch


def bind_lane_draft_contracts(root, batch):
    """Bind A4/A5 language and lane geometry, without claiming feasibility."""
    from shapely.geometry import Polygon, mapping
    from shapely.ops import unary_union

    validate_draft_batch(batch)
    result = deepcopy(batch)
    inventory = load_json(Path(root) / "generated/curation_review_inventory.private.json")
    proposals = {p["proposal_key"]: p for s in inventory["scenes"] for p in s["proposals"]}
    count = 0
    for row in result["drafts"]:
        if not row["task_family"].startswith(("A4_", "A5_")):
            continue
        proposal = proposals[row["source_proposal_key"]]
        pair = row["topology_pair_hypothesis"]
        if pair not in arm_pair_candidates(proposal):
            raise ValueError("draft pair no longer matches original topology")
        expected = {"left_vs_right"} if row["task_family"].startswith("A4_") else {"keep_vs_left", "keep_vs_right"}
        if pair.get("kind") not in expected:
            raise ValueError("lane pair does not express the allocated task family")
        geometry = proposal["topology_private"]["lane_geometry_private_not_qualified"]
        lanes = {}
        for arm in pair["arms"]:
            key, role = arm["lane_id"], arm["role"]
            region = geometry[key]["lane_region_polygon_xy_not_qualified"]
            polygon = Polygon(region)
            if not polygon.is_valid or polygon.area <= 0:
                raise ValueError("invalid original lane region")
            lanes[key] = region
            language = {"keep": "Keep your current lane.",
                        "change_left": "Safely change to the adjacent lane on your left.",
                        "change_right": "Safely change to the adjacent lane on your right."}
            if role not in language:
                raise ValueError("unexpected lane arm role")
            row["arms"][arm["arm"]].update(
                instruction_schedule=[{"issue_at_s": 1.7, "utterances": [language[role]]}],
                semantic_contract={"kind": "lane", "target_lane": key, "scope": "draft_shared_scope",
                                   "stable_duration_s": 1., "deadline_s": 7.})
        row["map_bound_contract_draft"] = {
            "lanes": lanes,
            "scopes": {"draft_shared_scope": mapping(unary_union([Polygon(v) for v in lanes.values()]))},
            "coordinate_frame": "nre_local", "position_reference": "ego_aabb_center",
            "frozen": False, "support_region": None}
        row["family_binding_status"] = "map_bound_lane_contract_pending_feasibility"
        row["required_bindings"].remove("task_family_and_arm_mapping")
        count += 1
    result["map_bound_lane_draft_count"] = count
    validate_draft_batch(result)
    return result


def swap_pending_lane_opportunities(batch, first_id, second_id, *, reason):
    """Exchange uninstantiated A5/B2 leads, never relabel an authored PCU.

    Family quotas and slot identities stay fixed. Discard the old family
    contracts, preserve both original rows, and require fresh family evidence.
    This does not establish the common end marker required for B2.
    """
    validate_draft_batch(batch)
    if first_id == second_id or not isinstance(reason, str) or not reason.strip():
        raise ValueError("two distinct pending slots and an explicit reason required")
    selected = [r for r in batch["drafts"] if r["draft_id"] in {first_id, second_id}]
    if len(selected) != 2 or {r["task_family"] for r in selected} != {
        "A5_lane_change_keep", "B2_scoped_lane_preference"
    }:
        raise ValueError("exactly one A5 and one B2 pending slot required")
    protected = {"authored_pcu_path", "exploratory_evidence", "agent_pilot_evidence",
                 "assembled_actual_contract_audit", "actual_pilot_seeds"}
    for row in selected:
        if (row.get("family_binding_status") not in {
                "unverified_hypothesis", "map_bound_lane_contract_pending_feasibility"}
            or protected.intersection(row)
            or row.get("grounding_bindings") != {"R0": None, "R1": None}
            or row.get("topology_pair_hypothesis", {}).get("kind") not in {"keep_vs_left", "keep_vs_right"}
            or row.get("initial_human_decision_reference") == "reject"):
            raise ValueError("preserve authored, grounded, rejected or actual-pilot evidence")
    replacements = {}
    for target, source in (selected, selected[::-1]):
        row = deepcopy(source)
        family = target["task_family"]
        row.update(draft_id=target["draft_id"], task_family=family, primary_layer=family[0],
                   family_binding_status="unverified_hypothesis")
        row.pop("map_bound_contract_draft", None)
        row["arms"] = {arm: {"instruction_template_not_executable": template,
                              "instruction_schedule": None, "semantic_contract": None}
                       for arm, template in zip(("R0", "R1"), TEMPLATES[family.split("_")[0]])}
        row["required_bindings"] = ["task_family_and_arm_mapping", "window_and_spatial_contract",
            "legal_action_and_balance", "reconstruction_support", "safe_plans_all_seeds",
            "formal_oracle_witnesses", "final_human_review"]
        if family.startswith("B2_"):
            row["required_bindings"] += ["same_main_route", "common_observable_scope_end", "shared_entry_exit_gates"]
        replacements[row["draft_id"]] = row
    result = deepcopy(batch)
    result["drafts"] = [replacements.get(r["draft_id"], r) for r in result["drafts"]]
    result["map_bound_lane_draft_count"] = sum("map_bound_contract_draft" in r for r in result["drafts"])
    if "summary" in result:
        result["summary"]["map_bound_lane_drafts"] = result["map_bound_lane_draft_count"]
    result.setdefault("opportunity_swap_history", []).append({
        "reason": reason, "original_rows": deepcopy(selected), "qualified": False,
        "mapping": [{"draft_id": key, "new_proposal_key": row["source_proposal_key"]}
                    for key, row in replacements.items()],
        "family_contracts_reset": True, "human_approval_inherited": False})
    validate_draft_batch(result)
    return result


def replace_lane_draft_with_agent_pilot(root, batch, draft_id, run_dirs):
    """Bind one untouched A4/A5 slot to its matching actual pilot, not eligibility."""
    from .agent_curation import validate_agent_runs
    from .feasibility_summary import summarize_lane_seeds, summarize_lane_pilot
    from shapely.geometry import Polygon, mapping
    from shapely.ops import unary_union

    root = Path(root).resolve()
    validate_draft_batch(batch)
    candidate = validate_agent_runs(root, run_dirs)
    pair = candidate["selected_arm_pair"]
    family = {"left_vs_right": "A4_lane_change_left_right", "keep_vs_left": "A5_lane_change_keep",
              "keep_vs_right": "A5_lane_change_keep"}.get(pair["kind"])
    if family is None:
        raise ValueError("A4 requires left/right; A5 requires keep versus adjacent lane")
    summary = summarize_lane_pilot(run_dirs[0]) if len(run_dirs) == 1 else summarize_lane_seeds(run_dirs)
    selected = [r for r in batch["drafts"] if r["draft_id"] == draft_id]
    if len(selected) != 1 or selected[0]["task_family"] != family:
        raise ValueError("select exactly one " + family[:2] + " authoring slot")
    if (selected[0].get("family_binding_status") not in {"unverified_hypothesis", "map_bound_lane_contract_pending_feasibility"}
        or any(k in selected[0] for k in ("authored_pcu_path", "exploratory_evidence", "agent_pilot_evidence"))):
        raise ValueError("preserve previously bound or authored lane evidence")
    key = candidate["proposal_key"]
    if any(r["source_proposal_key"] == key and r["draft_id"] != draft_id for r in batch["drafts"]):
        raise ValueError("pilot opportunity already allocated; do not double-count")
    launch = load_json(Path(run_dirs[0]) / "launch.private.json")
    offline = load_json(launch["agent_candidate_source"])
    proposals = []
    for binding in offline["source_bindings"]:
        path = Path(binding["path"])
        if path.suffix == ".json":
            inventory = load_json(path)
            proposals.extend(p for scene in inventory.get("scenes", [])
                             if isinstance(scene, dict) and scene.get("artifact_uuid") == candidate["artifact_uuid"]
                             for p in scene.get("proposals", []) if p["proposal_key"] == key)
    if len(proposals) != 1:
        raise ValueError("one original asset-bound proposal required")
    proposal = proposals[0]
    scene = candidate["scene_execution_proposal_not_frozen"]
    artifact_path = Path(scene["artifact_path"])
    row = deepcopy(selected[0])
    # Reset source-specific fields; old notes and evidence must not follow a replacement.
    for field in ("agent_visual_observations", "visual_grounding_lead", "visual_observations",
                  "map_bound_contract_draft"):
        row.pop(field, None)
    row.update(source_proposal_key=key, scene_id_private=candidate["scene_id_private"],
               artifact_uuid=candidate["artifact_uuid"], artifact_path=str(artifact_path),
               artifact_size_bytes=artifact_path.stat().st_size,
               suggested_start_offset_us_not_frozen=scene["start_time_offset_us"],
               recording_start_timestamp_us=scene["recording_start_timestamp_us"],
               recording_duration_s=(scene["recording_end_timestamp_us"] - scene["recording_start_timestamp_us"]) / 1e6,
               planning_horizon_13s_available_at_suggested_window=scene["start_timestamp_us"] + 13000000 < scene["recording_end_timestamp_us"],
               original_timing_prefilter_passed=proposal["timing_review"]["automatic_timing_prefilter_pass"],
               topology_pair_hypothesis=pair, evidence_sheet=proposal["evidence_sheet"],
               map_evidence_svg=proposal["map_evidence_svg"], initial_human_decision_reference=None,
               family_binding_status=("three_seed_lane_pilot_passed_spatial_and_release_gates_pending" if summary["three_seed_pilot_complete"]
                                      else "single_seed_lane_pilot_passed_spatial_and_release_gates_pending"),
               exploratory_evidence=summary,
               required_bindings=["window_and_spatial_contract", "legal_action_and_balance",
                                  "reconstruction_support", "formal_oracle_witnesses", "final_human_review"])
    lanes = {}
    for arm in pair["arms"]:
        identifier, role = arm["lane_id"], arm["role"]
        lanes[identifier] = candidate["lane_geometry_private_not_qualified"][identifier]["lane_region_polygon_xy_not_qualified"]
        language = {"keep": "Keep your current lane.", "change_left": "Safely change to the adjacent lane on your left.",
                    "change_right": "Safely change to the adjacent lane on your right."}[role]
        row["arms"][arm["arm"]] = {"instruction_schedule": [{"issue_at_s": 1.7, "utterances": [language]}],
            "semantic_contract": {"kind": "lane", "target_lane": identifier, "scope": "draft_shared_scope",
                                  "stable_duration_s": 1., "deadline_s": 7.}}
    row["map_bound_contract_draft"] = {"lanes": lanes,
        "scopes": {"draft_shared_scope": mapping(unary_union([Polygon(p) for p in lanes.values()]))},
        "coordinate_frame": "nre_local", "position_reference": "ego_aabb_center",
        "frozen": False, "support_region": None, "actual_contract_verification_pending": True}
    result = deepcopy(batch)
    result["drafts"] = [row if r["draft_id"] == draft_id else r for r in result["drafts"]]
    result["map_bound_lane_draft_count"] = sum("map_bound_contract_draft" in r for r in result["drafts"])
    result.setdefault("replacement_history", []).append({"draft_id": draft_id,
        "old_proposal_key": selected[0]["source_proposal_key"], "new_proposal_key": key,
        "replaced_row": deepcopy(selected[0]), "actual_pilot_seeds": summary["actual_pilot_seeds"],
        "reason": "replace untested hypothesis with actual lane pilot; no qualification claimed"})
    result["summary"] = {"draft_count": len(result["drafts"]),
        "scene_count": len({r["scene_id_private"] for r in result["drafts"]}),
        "layer_counts": dict(Counter(r["primary_layer"] for r in result["drafts"])),
        "current_window_has_13s_horizon": sum(r["planning_horizon_13s_available_at_suggested_window"] for r in result["drafts"]),
        "map_bound_lane_drafts": result["map_bound_lane_draft_count"],
        "three_seed_exploratory_candidates": sum(r.get("exploratory_evidence", {}).get("three_seed_pilot_complete") is True for r in result["drafts"]),
        "single_seed_exploratory_candidates": sum(r.get("exploratory_evidence", {}).get("three_seed_pilot_complete") is False for r in result["drafts"]),
        "qualified_pcus": 0}
    validate_draft_batch(result)
    return result
