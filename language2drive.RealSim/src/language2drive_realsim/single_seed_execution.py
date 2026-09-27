"""Single-seed launch plans and manifest gates, separate from legacy execution.

Planning is pure: it never submits runs, approves reused traces, or signs model
scores. Small protocol/configuration bindings are retained; no ASLs, USDZs or
checkpoint shards are read here.
"""

from collections import Counter
from copy import deepcopy
from pathlib import Path

from .compiler import compile_source
from .io import stable_hash
from .model_evaluation import MODEL_PREFIXES
from .model_identity import checkpoint_identity
from .runtime_client import build_run_plan, create_run_manifest
from .schema import validate_source
from .single_seed_metrics import single_seed_slots


SCHEMA = "l2d.realsim.single_seed_run.v0.1"
ADAPTERS = {
    "Language2Drive/SimLingo-Posttrained-epoch004": ("SimLingo-L2D-Posttrained", "l2d_simlingo"),
    "nvidia/Alpamayo2-Super": ("Alpamayo2-Super", "l2d_alpamayo2_super"),
    "nvidia/Alpamayo-1.5-10B": ("Alpamayo 1.5", "l2d_alpamayo1_5"),
    "RenzKa/simlingo": ("SimLingo", "l2d_simlingo"),
    "Qwen/Qwen-Drive-1.0-4B": ("Qwen-Drive-1.0-4B", "l2d_qwen_drive"),
    "Zewei-Zhou/AutoVLA": ("AutoVLA", "l2d_autovla"),
    "OpenDriveVLA/OpenDriveVLA-0.5B": ("OpenDriveVLA", "l2d_opendrivevla"),
    "OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0": (
        "LMDrive synthetic-null-LiDAR",
        "l2d_lmdrive_null_lidar",
    ),
}

_THREE_VIEW_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
]
_TRAJECTORY_SELECTION = {
    "strategy": "ALWAYS_FIRST",
    "max_num_distance_points": 64,
    "skip_first_n_distance_points": 0,
}
_LMDRIVE_CAMERAS = [
    ("camera_front_wide_120fov", 900, 1200),
    ("camera_cross_left_120fov", 300, 400),
    ("camera_cross_right_120fov", 300, 400),
    ("camera_rear_left_70fov", 300, 400),
]
_LMDRIVE_POSES = [
    ([1.3, 0.0, 2.3], [0.5, -0.5, 0.5, -0.5]),
    ([1.3, 0.0, 2.3], [0.683012701892219, -0.183012701892219,
                        0.183012701892219, -0.683012701892219]),
    ([1.3, 0.0, 2.3], [0.183012701892219, -0.683012701892219,
                        0.683012701892219, -0.183012701892219]),
    ([-1.3, 0.0, 2.3], [-0.5, -0.5, 0.5, 0.5]),
]


def _lmdrive_camera_definition(camera, pose):
    logical_id, height, width = camera
    focal = 503.459778706368 if width == 1200 else 167.819926235456
    return {
        "logical_id": logical_id,
        "rig_to_camera": {"translation_m": pose[0], "rotation_xyzw": pose[1]},
        "intrinsics": {
            "model": "opencv_pinhole",
            "opencv_pinhole": {
                "focal_length": [focal, focal],
                "principal_point": [width / 2.0, height / 2.0],
                "radial": [0.0] * 6,
                "tangential": [0.0] * 2,
                "thin_prism": [0.0] * 4,
            },
        },
        "resolution_hw": [height, width],
        "shutter_type": "GLOBAL",
    }


_LMDRIVE_RUNTIME = {
    "simulation_config": {
        "skip_driver_during_force_gt": True,
        "cameras": [
            {
                "logical_id": logical_id,
                "height": height,
                "width": width,
                "frame_interval_us": 100_000,
                "shutter_duration_us": 30_000,
            }
            for logical_id, height, width in _LMDRIVE_CAMERAS
        ],
    },
    "extra_cameras": [
        _lmdrive_camera_definition(camera, pose)
        for camera, pose in zip(_LMDRIVE_CAMERAS, _LMDRIVE_POSES)
    ],
}
STRICT_ADAPTER_PROTOCOLS = {
    "OpenDILabCommunity/LMDrive-llava-v1.5-7b-v1.0": {
        # This protocol is a separately named synthetic-null-LiDAR ablation,
        # not an assertion of official LMDrive input equivalence.
        "model": {
            "model_type": "l2d_lmdrive_null_lidar",
            "device": "cuda",
            "force_determinism": True,
            "image_decode_device": "cpu",
            "tokenizer_path": None,
            "num_trajectory_samples": 1,
            "trajectory_candidate_microbatch_size": None,
            "cfg_guidance_weight": None,
            "trajectory_selection": _TRAJECTORY_SELECTION,
        },
        "inference": {
            "use_cameras": [
                "camera_front_wide_120fov",
                "camera_cross_left_120fov",
                "camera_cross_right_120fov",
                "camera_rear_left_70fov",
            ],
            "max_batch_size": 1,
            "subsample_factor": 1,
            "context_length": 1,
            "output_frequency_hz": 2,
        },
        "runtime": _LMDRIVE_RUNTIME,
    },
    "Qwen/Qwen-Drive-1.0-4B": {
        "model": {
            "model_type": "l2d_qwen_drive",
            "device": "cuda",
            "force_determinism": True,
            "image_decode_device": "cpu",
            "num_trajectory_samples": 1,
            "tokenizer_path": None,
            "trajectory_candidate_microbatch_size": None,
            "cfg_guidance_weight": None,
            "trajectory_selection": _TRAJECTORY_SELECTION,
        },
        "inference": {
            "use_cameras": _THREE_VIEW_CAMERAS,
            "max_batch_size": 1,
            "subsample_factor": 5,
            "context_length": 4,
            "output_frequency_hz": 10,
        },
    },
    "Zewei-Zhou/AutoVLA": {
        "model": {
            "model_type": "l2d_autovla",
            "device": "cuda",
            "force_determinism": True,
            "image_decode_device": "cpu",
            "num_trajectory_samples": 1,
            "trajectory_candidate_microbatch_size": None,
            "cfg_guidance_weight": None,
            "trajectory_selection": _TRAJECTORY_SELECTION,
        },
        "inference": {
            "use_cameras": _THREE_VIEW_CAMERAS,
            "max_batch_size": 1,
            "subsample_factor": 5,
            "context_length": 4,
            "output_frequency_hz": 2,
        },
    },
    "OpenDriveVLA/OpenDriveVLA-0.5B": {
        "model": {
            "model_type": "l2d_opendrivevla",
            "device": "cuda",
            "force_determinism": True,
            "image_decode_device": "cpu",
            "num_trajectory_samples": 1,
            "trajectory_candidate_microbatch_size": None,
            "cfg_guidance_weight": None,
            "trajectory_selection": _TRAJECTORY_SELECTION,
        },
        "inference": {
            "use_cameras": [
                "camera_front_wide_120fov",
                "camera_cross_right_120fov",
                "camera_cross_left_120fov",
                "camera_front_tele_30fov",
                "camera_rear_left_70fov",
                "camera_rear_right_70fov",
            ],
            "max_batch_size": 1,
            "subsample_factor": 1,
            "context_length": 1,
            "output_frequency_hz": 2,
        },
    },
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def model_protocol_configuration_errors(model_id, configuration):
    """Enforce newly audited adapters' exact, route-isolated input protocol."""

    protocol = STRICT_ADAPTER_PROTOCOLS.get(model_id)
    if protocol is None:
        return []
    errors = []
    if not isinstance(configuration, dict):
        return ["driver_configuration_protocol_mismatch"]
    model = configuration.get("model")
    inference = configuration.get("inference")
    if not isinstance(model, dict) or not isinstance(inference, dict):
        return ["driver_configuration_protocol_mismatch"]
    expected_model = protocol["model"]
    # Versioned AutoVLA adapter: retain every genuine official decoded pose.
    # Explicit config binding is mandatory; the historical strict-10 protocol
    # remains unchanged when this field is absent. Unknown policies still fail.
    if (model_id == "Zewei-Zhou/AutoVLA"
            and model.get("autovla_horizon_policy") == "official_native_length_v1"):
        expected_model = {**expected_model, "autovla_horizon_policy": "official_native_length_v1"}
    if (
        set(model) != set(expected_model) | {"checkpoint_path"}
        or {key: model.get(key) for key in expected_model} != expected_model
        or inference != protocol["inference"]
        or ("runtime" in protocol and configuration.get("runtime") != protocol["runtime"])
    ):
        errors.append("driver_configuration_protocol_mismatch")
    optimizer = configuration.get("trajectory_optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("enabled") is not False:
        errors.append("trajectory_optimizer_not_disabled")
    if configuration.get("rectification", "missing") is not None:
        errors.append("camera_rectification_not_disabled")
    route = configuration.get("route")
    if not isinstance(route, dict) or route.get("use_waypoint_commands") is not False:
        errors.append("target_derived_navigation_not_disabled")
    return errors


def build_single_seed_execution_plan(source, public, sidecar, campaign, qualification):
    """Project exact compiler outputs, not a relabelled 180-run denominator."""
    validate_source(source, require_exact_release=True)
    _require(campaign.get("schema_version") == "l2d.realsim.campaign_plan.v0.1",
             "invalid campaign schema")
    _require(_text(campaign.get("protocol_id")), "missing protocol identity")
    seeds = campaign.get("model_seeds")
    _require(isinstance(seeds, list) and len(seeds) == 1
             and type(seeds[0]) is int and seeds[0] >= 0, "exactly one integer seed required")
    _require(campaign.get("arms") == ["R0", "R1"], "invalid campaign arms")
    count = len(source["pcus"])
    for field, expected in (("pcu_count", count), ("pairs_per_model", count),
                            ("runs_per_model", count * 2)):
        _require(type(campaign.get(field)) is int and campaign[field] == expected,
                 "campaign denominator mismatch: " + field)
    layers = {layer: sum(p["primary_layer"] == layer for p in source["pcus"])
              for layer in ("A", "B", "C")}
    _require(campaign.get("layer_counts") == layers
             and all(type(v) is int for v in campaign["layer_counts"].values()),
             "campaign layer counts mismatch")
    models = campaign.get("models")
    _require(isinstance(models, list) and bool(models)
             and all(isinstance(m, dict) and _text(m.get("name")) for m in models),
             "invalid campaign model list")
    names = [m["name"] for m in models]
    _require(len(set(names)) == len(names), "duplicate campaign model")
    for field, expected in (("expected_model_count", len(names)),
                            ("expected_run_slots", count * 2 * len(names))):
        _require(type(campaign.get(field)) is int and campaign[field] == expected,
                 "campaign total mismatch: " + field)
    # Verify all private contracts/instructions and public input isolation.
    # Matching source_binding text alone cannot prove compiler fidelity.
    expected_public, expected_sidecar, leakage = compile_source(source)
    _require(public == expected_public and sidecar == expected_sidecar,
             "compiled artifacts differ from frozen source")
    _require(leakage.get("pass") is True, "compiled input leakage check failed")
    binding = expected_sidecar["source_binding"]
    rows = qualification.get("pcus")
    _require(qualification.get("schema_version") == "l2d.realsim.qualification.v0.1"
             and qualification.get("source_binding") == binding
             and qualification.get("pass") is True
             and qualification.get("all_global_gates_pass") is True
             and qualification.get("global_errors") == [], "oracle qualification not valid")
    _require(all(type(qualification.get(k)) is int and qualification[k] == count
                 for k in ("candidate_count", "eligible_count", "target_pcu_count")),
             "oracle qualification denominator mismatch")
    _require(isinstance(rows, list) and len(rows) == count
             and all(isinstance(r, dict) and _text(r.get("pcu_id"))
                     and r.get("eligible") is True for r in rows)
             and Counter(r["pcu_id"] for r in rows) == Counter(p["pcu_id"] for p in source["pcus"]),
             "exact oracle-qualified cohort missing")
    slots = single_seed_slots(source, sidecar, seeds[0])
    selected = {slot["session_uuid"] for slot in slots}
    full = build_run_plan(public, sidecar)
    pairs = {slot["pcu_id"] + ":" + str(seeds[0]) for slot in slots}
    return dict(schema_version="l2d.realsim.run_plan.v0.1", source_binding=binding,
                protocol_id=campaign["protocol_id"], model_seeds=deepcopy(seeds),
                run_count=len(selected), runs=[deepcopy(r) for r in full["runs"]
                                              if r["session_uuid"] in selected],
                runtime_paired_input_assertions={key: deepcopy(value) for key, value in
                    full["runtime_paired_input_assertions"].items() if key in pairs})


def partition_single_seed_plan(plan, groups):
    """Keep existing runtime geometry/timing groups; reject gaps or duplicates."""
    rows, seeds = plan.get("runs"), plan.get("model_seeds")
    _require(isinstance(seeds, list) and len(seeds) == 1 and type(seeds[0]) is int,
             "invalid single-seed partition plan")
    _require(isinstance(rows, list) and bool(rows)
             and type(plan.get("run_count")) is int and plan["run_count"] == len(rows)
             and all(isinstance(r, dict) and _text(r.get("session_uuid"))
                     and _text(r.get("pcu_id")) and r.get("arm_id") in ("R0", "R1")
                     and type(r.get("random_seed")) is int and r["random_seed"] == seeds[0]
                     for r in rows), "invalid partition run ledger")
    counts = Counter((r["pcu_id"], r["arm_id"]) for r in rows)
    expected = {r["pcu_id"] for r in rows}
    _require(len({r["session_uuid"] for r in rows}) == len(rows)
             and counts == Counter({(pid, arm): 1 for pid in expected for arm in ("R0", "R1")}),
             "partition plan has duplicate or incomplete arms")
    _require(set(plan.get("runtime_paired_input_assertions", {})) == {
        pid + ":" + str(seeds[0]) for pid in expected}, "partition pair assertions mismatch")
    seen, names, result = set(), set(), []
    for group in groups:
        name, ids = group.get("group_id"), group.get("pcu_ids")
        _require(_text(name) and name not in names, "invalid or duplicate group identity")
        _require(isinstance(ids, list) and bool(ids) and all(_text(pid) for pid in ids)
                 and len(ids) == len(set(ids)) and set(ids) <= expected
                 and not seen.intersection(ids), "invalid or duplicate group membership")
        names.add(name)
        seen.update(ids)
        runs = [deepcopy(row) for row in plan["runs"] if row["pcu_id"] in ids]
        keys = {r["pcu_id"] + ":" + str(r["random_seed"]) for r in runs}
        result.append(dict(group_id=name, pcu_ids=deepcopy(ids), plan={**deepcopy(plan),
            "run_count": len(runs), "runs": runs,
            "runtime_paired_input_assertions": {key: deepcopy(value) for key, value in
                plan["runtime_paired_input_assertions"].items() if key in keys}}))
    _require(seen == expected, "runtime groups do not cover full cohort")
    return result


def build_single_seed_manifest(source, public, sidecar, campaign, qualification, *,
                               model_name, model_identity, inference_configuration,
                               runtime_address, driver_address, dry_run=True):
    """Create a preflight manifest by default; not a launch or completion receipt."""
    _require(type(dry_run) is bool, "dry_run must be boolean")
    plan = build_single_seed_execution_plan(source, public, sidecar, campaign, qualification)
    _require(model_name in {m["name"] for m in campaign["models"]}, "model absent from campaign")
    _require(isinstance(inference_configuration, dict) and bool(inference_configuration),
             "explicit inference configuration required")
    _require(isinstance(model_identity, dict) and bool(model_identity), "model identity required")
    _require(_text(runtime_address) and _text(driver_address), "explicit service addresses required")
    manifest = create_run_manifest(plan, runtime_address, driver_address, dry_run,
        model_identity=deepcopy(model_identity), use_registered_driver=True)
    manifest.update(schema_version=SCHEMA, protocol_id=campaign["protocol_id"],
        model_seeds=deepcopy(campaign["model_seeds"]), model_name=model_name,
        campaign_binding=stable_hash(campaign), qualification_binding=stable_hash(qualification),
        sidecar_binding=stable_hash(sidecar), public_workload_binding=stable_hash(public),
        inference_configuration=deepcopy(inference_configuration),
        configuration_id=stable_hash(inference_configuration),
        invalid_sim_policy="preserve evidence and stop; no automatic selective rerun",
        formal_model_evaluation_complete=False)
    manifest["manifest_binding"] = stable_hash({k: v for k, v in manifest.items() if k != "manifest_binding"})
    return manifest


def single_seed_manifest_errors(source, public, sidecar, campaign, qualification,
                                manifest, inference_configuration, *, require_actual=True):
    """Check intended execution bindings; does not prove launch or run evidence."""
    errors = []
    try:
        expected = build_single_seed_manifest(source, public, sidecar, campaign, qualification,
            model_name=manifest["model_name"], model_identity=manifest["model_identity"],
            inference_configuration=inference_configuration,
            runtime_address=manifest["runtime_address"], driver_address=manifest["driver_address"],
            dry_run=manifest["dry_run"])
        if manifest != expected:
            errors.append("single_seed_manifest_or_configuration_mismatch")
        if require_actual and manifest["dry_run"] is not False:
            errors.append("manifest_not_actual_execution")
        identity = manifest["model_identity"]
        adapter = ADAPTERS.get(identity.get("model_id"))
        if (identity.get("model_id") not in MODEL_PREFIXES
                or not adapter or adapter[0] != manifest["model_name"]):
            errors.append("model_adapter_identity_not_supported")
        else:
            config_model = inference_configuration.get("model", {})
            route = inference_configuration.get("route", {})
            _require(isinstance(config_model, dict) and isinstance(route, dict),
                     "invalid driver configuration model or route")
            if (config_model.get("model_type") != adapter[1]
                    or not _text(config_model.get("checkpoint_path"))
                    or Path(config_model["checkpoint_path"]).resolve() != Path(identity["path"]).resolve()):
                errors.append("driver_configuration_model_identity_mismatch")
            if route.get("use_waypoint_commands") is not False:
                errors.append("target_derived_navigation_not_disabled")
            errors.extend(model_protocol_configuration_errors(identity["model_id"], inference_configuration))
            if (identity.get("complete") is not True or not identity.get("snapshot_revision")
                    or checkpoint_identity(Path(identity["path"]), identity["model_id"]) != identity):
                errors.append("checkpoint_identity_incomplete_or_changed")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        errors.append("invalid_single_seed_manifest_inputs:" + str(exc))
    return errors
