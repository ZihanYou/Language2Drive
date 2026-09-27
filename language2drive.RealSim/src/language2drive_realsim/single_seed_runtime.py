"""Read-only preparation of a fresh runtime namespace for audited single-seed reuse.

The old attempt is never a write destination. Configuration relocation changes
only output locations, not inference or simulation policy. Full configuration
identities remain distinct and the explicit relocation receipt links them.
Preparing these files does not launch services or attest complete legality.
"""

from copy import deepcopy
import os
from pathlib import Path
import re
import uuid

from .compat import strict_zip
from .io import load_json, stable_hash
from .single_seed_execution import (
    build_single_seed_execution_plan, build_single_seed_manifest,
    partition_single_seed_plan, single_seed_manifest_errors,
)
from .single_seed_reuse import reusable_sessions_from_cache


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def driver_relocation(original, relocated, old_attempt, new_attempt):
    """Allow exactly the driver output-dir rebase; retain both full identities.

    Upstream alpasim_driver/main.py uses output_dir for config/debug-file output,
    not model inputs, sampling, trajectory selection or optimization. No other
    key (including seemingly operational host/port/logging) is exempted here.
    """
    old, new = Path(old_attempt).resolve(), Path(new_attempt).resolve()
    _require(old != new and not new.is_relative_to(old) and not old.is_relative_to(new),
             "new runtime must be disjoint from the old attempt")
    _require(original.get("output_dir") == str(old / "model-driver"),
             "unexpected original driver output location")
    expected = {**deepcopy(original), "output_dir": str(new / "model-driver")}
    _require(relocated == expected, "driver relocation changed more than output_dir")
    policy = {k:v for k,v in original.items() if k != "output_dir"}
    result = dict(schema_version="l2d.realsim.driver_output_relocation.v0.1",
        allowed_changed_fields=["output_dir"], inference_policy_unchanged=True,
        origin_configuration_id=stable_hash(original),
        execution_configuration_id=stable_hash(relocated),
        inference_policy_id=stable_hash(policy),
        origin_output_dir=original["output_dir"], execution_output_dir=relocated["output_dir"],
        scope="explicit_output_relocation_not_execution_or_score_authority")
    result["relocation_binding"] = stable_hash(result)
    return result


def _build_runtime_namespace(root, campaign_path, bundle_path, attempt_name, *, existing):
    """Return exact JSON/YAML file contents in memory; never write or launch.

    A new launcher must validate/materialize these files exclusively, perform
    live service/GPU checks and record an actual manifest before submitting.
    The returned prepared manifest remains dry_run=true.
    """
    root = Path(root).resolve()
    _require(isinstance(attempt_name, str)
             and re.fullmatch(r"single-seed-[a-z0-9][a-z0-9-]{0,95}", attempt_name),
             "invalid single-seed runtime name")
    runtime_root = root / ".runtime"
    _require(runtime_root.is_dir() and not runtime_root.is_symlink(), "runtime root missing or symlinked")
    target = runtime_root / attempt_name
    if existing:
        _require(target.is_dir() and not target.is_symlink(), "runtime namespace missing or symlinked")
    else:
        _require(not os.path.lexists(target), "preserve existing runtime namespace")
    campaign = load_json(root / campaign_path)
    compiled = root / campaign["compiled_dir"]
    inputs = dict(source=load_json(root / campaign["source"]),
        public=load_json(compiled / "public_workload.json"),
        sidecar=load_json(compiled / "sidecar.private.json"), campaign=campaign,
        qualification=load_json(root / campaign["qualification_report"]))
    bundle = load_json(root / bundle_path)
    _require(bundle.get("schema_version") == "l2d.realsim.single_seed_preparation.v0.1"
             and bundle.get("ready_to_launch") is False
             and bundle.get("model_runs_submitted") == 0
             and bundle.get("formal_model_evaluation_complete") is False,
             "expected an unlaunched single-seed preparation")
    plan = build_single_seed_execution_plan(**inputs)
    _require(bundle["full_run_plan"] == plan, "prepared full plan changed")
    audit = load_json(bundle["reuse_audit_source"])
    reused = reusable_sessions_from_cache(audit, plan, bundle["manifest"])
    _require(bundle.get("reuse_audit_binding") == audit["audit_binding"]
             and bundle.get("reusable_sessions") == sorted(reused)
             and bundle.get("reusable_run_count") == len(reused)
             and bundle.get("pending_run_count") == plan["run_count"] - len(reused),
             "prepared reuse accounting differs from audit")
    old = Path(audit["old_attempt"]).resolve()
    _require(Path(bundle["historical_preparation_source"]).resolve() == old / "model-preparation.private.json"
             and Path(bundle["configuration_source"]).resolve() == old / "driver-config.yaml",
             "historical configuration sources differ from audited attempt")
    original = load_json(old / "driver-config.yaml")
    _require(not single_seed_manifest_errors(**inputs, manifest=bundle["manifest"],
             inference_configuration=original, require_actual=False), "original preparation manifest invalid")
    historical = load_json(old / "model-preparation.private.json")
    launch = load_json(old / "launch-context.private.json")
    bound_paths = {Path(r["path"]).resolve() for r in
        historical["references"] + launch["execution_file_references"]}
    required_sources = {old / "driver-config.yaml", old / "renderer-compose.yaml"}
    required_sources.update(Path(g["directory"]).resolve() / filename
        for g in historical["groups"]
        for filename in ("user-config.yaml", "network-config.yaml", "eval-config.yaml"))
    _require(required_sources <= bound_paths, "runtime source config not bound by old attempt")
    groups = partition_single_seed_plan(plan, [dict(group_id=Path(g["directory"]).name,
        pcu_ids=g["pcu_ids"]) for g in historical["groups"]])
    _require(len(groups) == len(bundle["groups"]) == len(bundle["pending_execution_groups"]),
             "prepared group count differs")
    files, runtime_groups = {}, []
    for group, before, supplied, supplied_pending in strict_zip(groups, historical["groups"],
            bundle["groups"], bundle["pending_execution_groups"]):
        origin_dir = Path(before["directory"]).resolve()
        _require(origin_dir.parent == old, "group configuration outside audited attempt")
        group["historical_config_directory_read_only"] = str(origin_dir)
        _require(group == supplied, "prepared full group differs")
        pending = deepcopy(group)
        rows = [r for r in group["plan"]["runs"] if r["session_uuid"] not in reused]
        ids = {r["pcu_id"] for r in rows}
        keys = {r["pcu_id"] + ":" + str(r["random_seed"]) for r in rows}
        pending["pcu_ids"] = [pid for pid in group["pcu_ids"] if pid in ids]
        pending["plan"].update(runs=rows, run_count=len(rows), runtime_paired_input_assertions={
            k:v for k,v in group["plan"]["runtime_paired_input_assertions"].items() if k in keys})
        pending["requires_execution"] = bool(rows)
        _require(pending == supplied_pending, "prepared pending group differs")
        name = group["group_id"]
        _require(re.fullmatch(r"group-[0-9]{2}", name), "invalid historical group name")
        user = load_json(origin_dir / "user-config.yaml")
        cache = user["simulation_config"]["force_gt_frame_cache"]
        _require(cache == dict(enabled=False, cache_dir=str(old / "frame-cache")),
                 "only disabled frame-cache location may be rebased")
        cache["cache_dir"] = str(target / "frame-cache")
        files[name + "/user-config.yaml"] = user
        for filename in ("network-config.yaml", "eval-config.yaml"):
            files[name + "/" + filename] = load_json(origin_dir / filename)
        files[name + "/run_plan.private.json"] = pending["plan"]
        # RuntimeEvaluator reads this only after a complete rollout. Unlike
        # Wizard deployment, the host launcher must prepare it explicitly.
        # This UUID labels an execution group, not a model session or source.
        files[name + "/run_metadata.yaml"] = dict(
            run_uuid=str(uuid.uuid5(uuid.NAMESPACE_URL,
                plan["protocol_id"] + ":" + str(target / name))),
            run_name=attempt_name + "/" + name)
        runtime_groups.append(dict(group_id=name, directory=str(target / name),
            origin_directory=str(origin_dir), pcu_ids=pending["pcu_ids"],
            run_count=len(rows), requires_execution=bool(rows)))
    driver = {**deepcopy(original), "output_dir": str(target / "model-driver")}
    relocation = driver_relocation(original, driver, old, target)
    manifest = build_single_seed_manifest(**inputs, model_name=bundle["manifest"]["model_name"],
        model_identity=bundle["manifest"]["model_identity"], inference_configuration=driver,
        runtime_address=bundle["manifest"]["runtime_address"],
        driver_address=bundle["manifest"]["driver_address"])
    _require(not single_seed_manifest_errors(**inputs, manifest=manifest,
        inference_configuration=driver, require_actual=False), "relocated preparation manifest invalid")
    compose = load_json(old / "renderer-compose.yaml")
    old_compose_name = compose["name"]
    compose["name"] = "l2d-" + attempt_name
    _require(compose["name"] != old_compose_name, "renderer project namespace not distinct")
    _require(all(s.get("pull_policy") == "never" for s in compose["services"].values()),
             "renderer must use existing images without downloads")
    files.update({"driver-config.yaml": driver, "renderer-compose.yaml": compose,
        "full_run_plan.private.json": plan, "prepared_manifest.private.json": manifest,
        "configuration-relocation.private.json": relocation})
    metadata = dict(schema_version="l2d.realsim.single_seed_runtime_preparation.v0.1",
        scope="unlaunched_runtime_configuration_not_model_evaluation", attempt_directory=str(target),
        source_binding=plan["source_binding"], protocol_id=plan["protocol_id"],
        model_name=manifest["model_name"], model_seeds=plan["model_seeds"],
        expected_runs=plan["run_count"], reused_runs=len(reused), pending_runs=plan["run_count"]-len(reused),
        campaign_path=str((root / campaign_path).resolve()), campaign_binding=stable_hash(campaign),
        preparation_path=str((root / bundle_path).resolve()), preparation_binding=stable_hash(bundle),
        reuse_audit_path=bundle["reuse_audit_source"], reuse_audit_binding=audit["audit_binding"],
        origin_prepared_manifest_binding=bundle["manifest"]["manifest_binding"],
        execution_prepared_manifest_binding=manifest["manifest_binding"],
        configuration_relocation_binding=relocation["relocation_binding"],
        inference_policy_id=relocation["inference_policy_id"],
        oracle_preflight=historical["oracle_preflight"], sidecar=str(compiled / "sidecar.private.json"),
        groups=runtime_groups, reusable_sessions=sorted(reused),
        operational_changes=["driver.output_dir", "disabled force_gt_frame_cache.cache_dir", "compose.name",
            "runtime_eval_group_identity.run_metadata"],
        ready_to_launch=False, model_runs_submitted=0, formal_model_evaluation_complete=False,
        file_content_bindings={name:stable_hash(value) for name,value in files.items()},
        required_next_gates=["exclusive_namespace_materialization", "live_resource_and_service_preflight",
            "actual_manifest_and_launch_binding", "new_execution_receipt_authority",
            "complete_legality_evidence_before_formal_scoring"],
        asl_hashes_computed=0, checkpoint_shard_hashes_computed=0, asset_hashes_computed=0, downloads=0)
    metadata["namespace_binding"] = stable_hash(metadata)
    files["namespace-preparation.private.json"] = metadata
    return dict(attempt_directory=str(target), files=files, summary=dict(
        model_name=manifest["model_name"], expected_runs=plan["run_count"], reused_runs=len(reused),
        pending_runs=metadata["pending_runs"], groups=len(groups),
        inference_policy_unchanged=True, ready_to_launch=False, model_runs_submitted=0,
        formal_model_evaluation_complete=False))


def build_runtime_namespace(root, campaign_path, bundle_path, attempt_name):
    """Prepare a non-existent destination in memory; never overwrite or launch."""
    return _build_runtime_namespace(root, campaign_path, bundle_path, attempt_name, existing=False)


def validate_runtime_namespace(root, campaign_path, bundle_path, attempt_name):
    """Rebuild from audited origins and compare every materialized config file.

    Small configuration hashes alone are not trusted: a changed namespace with
    freshly recomputed bindings must still exactly match its frozen origins.
    This is read-only and does not attest live readiness or actual execution.
    """
    expected = _build_runtime_namespace(root, campaign_path, bundle_path, attempt_name, existing=True)
    directory = Path(expected["attempt_directory"])
    for name, value in expected["files"].items():
        path = directory / name
        _require(path.resolve() == path and path.is_file(), "namespace config missing or symlinked: " + name)
        _require(load_json(path) == value, "namespace config differs from audited origin: " + name)
    return expected
