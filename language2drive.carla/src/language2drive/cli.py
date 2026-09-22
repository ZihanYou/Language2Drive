from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import carla_runtime_assets as carla_runtime_assets_module
from .generator import generate_core_sources
from .io import atomic_json
from .maps import build_registry, review_gaps
from .release_report import render
from .release_scoring import score_plan, stamp_scope
from .runner import (
    CARLA_RUNTIME_MANIFEST,
    CARLA_RUNTIME_MAP_WITNESS,
    CARLA_RUNTIME_PROVENANCE,
    CARLA_RUNTIME_ROOT,
    active_core_sidecar_path,
    calibrate_deadlines,
    default_ledger,
    evaluator_environment,
    execute_plan,
    freeze_deadlines,
    freeze_carla_runtime_manifest,
    make_fdq_plan,
    make_plan,
    resolve_fdq_inputs,
    resolve_plan_inputs,
    select_fdq_runs,
    select_runs,
    structural_qualification,
)
from .scoring import score_suite
from .suite import compile_suite, load_sources, validate_suite
from .supplementary import BASELINE_IDS, build_supplementary, validate_supplementary
from .visualize import build_review_page, visualize_suite

ROOT = Path(__file__).resolve().parents[2]


def _strict_cli_json(raw: str, label: str):
    def reject_constant(value: str):
        raise ValueError(f"{label}: non-finite JSON constant {value}")
    def pairs(values):
        output = {}
        for key, value in values:
            if key in output:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            output[key] = value
        return output
    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=pairs)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _seeds(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("seeds must contain non-negative integers")
    return list(dict.fromkeys(values))


def _route_entry_ids(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("route entry IDs must be comma-separated integers") from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("route entry IDs must contain non-negative integers")
    return list(dict.fromkeys(values))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="l2d",
        description="Language2Drive v1 fixed-layout build, qualification, execution, and reporting CLI.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build-maps", help="build offline CARLA lane graphs and migrate existing labels")
    commands.add_parser("review-assets", help="render the semantic-asset review page and print unresolved items")

    build = commands.add_parser("build-suite", help="regenerate 128 PCUs and atomically compile public/private suite artifacts")
    build.add_argument("--allow-incomplete", action="store_true", help="development only: relax frozen cardinality checks")
    build.add_argument("--reuse-sources", action="store_true", help="compile existing sources/core_pcus.json without regenerating it")

    validate = commands.add_parser("validate", help="validate schema, quota, split, map-family, and ladder constraints")
    validate.add_argument("--allow-incomplete", action="store_true", help="development only: relax frozen cardinality checks")

    commands.add_parser("visualize", help="render joint lane/route/target/entity PCU visualizations")
    commands.add_parser("qualify", help="run static oracle-path structural checks (not dynamic CARLA witnesses)")
    commands.add_parser("build-supplementary", help="build the independent 6-case Safety and six dev anti-gaming workloads")
    commands.add_parser("validate-supplementary", help="validate Safety/anti-gaming counts, semantics, and opaque identities")
    runtime_assets = commands.add_parser(
        "freeze-runtime-assets",
        help="freeze and fully revalidate the exact installed CARLA runtime asset inventory",
    )
    runtime_assets.add_argument(
        "--replace", action="store_true",
        help="replace an existing manifest after hashing and revalidating the full runtime",
    )
    runtime_assets.add_argument("--host", default="127.0.0.1")
    runtime_assets.add_argument("--port", type=int, default=2000)

    run = commands.add_parser("run", help="create a recoverable per-route evaluator plan; --execute requires a running CARLA server")
    run.add_argument("--agent", default=None, help="built-in agent name (default: simlingo)")
    run.add_argument("--agent-path", type=_path, help="custom Bench2Drive agent Python file")
    run.add_argument("--agent-config", type=_path, help="custom checkpoint/config path")
    run.add_argument("--track", choices=("standardized", "native", "oracle"), required=True)
    run.add_argument("--split", choices=("dev", "test"), required=True)
    run.add_argument(
        "--tier", choices=("small", "base", "large"), default="large",
        help="main benchmark PCU tier (default: large / all 119 currently loadable PCUs)",
    )
    run.add_argument(
        "--seeds", type=_seeds, default=_seeds("1"),
        help="comma-separated evaluator seeds (default: 1; pass 1,2,3 for a multi-seed run)",
    )
    run.add_argument("--pilot", action="store_true", help="use the frozen 72-run Standardized pilot ledger")
    run.add_argument("--ledger", type=_path, help="override the expected-run ledger")
    run.add_argument("--results", type=_path, default=ROOT / "results")
    run.add_argument("--plan-output", type=_path, default=ROOT / "generated/run_plan.json")
    run.add_argument("--max-runs", type=int, help="cap selected pending jobs (useful for a one-route smoke)")
    run.add_argument(
        "--route-entry-ids",
        type=_route_entry_ids,
        help="select exact comma-separated route entry IDs (for deterministic smoke matrices)",
    )
    run.add_argument("--retry-finalized", action="store_true", help="retry even when a final crash/failure/success verdict exists")
    run.add_argument("--execute", action="store_true", help="execute serially; otherwise only write a command plan")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=2000)
    run.add_argument("--traffic-manager-port", type=int, default=8000)
    run.add_argument("--gpu-rank", type=int, default=0)
    run.add_argument("--skip-server-check", action="store_true", help="expert/debug only: do not probe CARLA before execution")
    run.add_argument("--wall-timeout", type=float, help="per-job infrastructure wall timeout in seconds")
    run.add_argument(
        "--fast-session-validation", action="store_true",
        help="full authority scan once per batch; use lightweight per-route process/input guards",
    )
    fdq = commands.add_parser("run-fdq", help="create or execute the recoverable loadable-scenario FDQ plan")
    fdq.add_argument("--agent", default=None, help="built-in agent name (default: simlingo)")
    fdq.add_argument("--agent-path", type=_path, help="custom Bench2Drive agent Python file")
    fdq.add_argument("--agent-config", type=_path, help="custom checkpoint/config path")
    fdq.add_argument(
        "--seeds", type=_seeds, default=_seeds("1"),
        help="comma-separated evaluator seeds (default: 1; pass 1,2,3 for the full FDQ matrix)",
    )
    fdq.add_argument("--results", type=_path, default=ROOT / "results")
    fdq.add_argument("--plan-output", type=_path, default=ROOT / "generated/fdq_run_plan.json")
    fdq.add_argument("--max-runs", type=int, help="cap selected pending jobs")
    fdq.add_argument("--retry-finalized", action="store_true")
    fdq.add_argument("--execute", action="store_true")
    fdq.add_argument("--host", default="127.0.0.1")
    fdq.add_argument("--port", type=int, default=2000)
    fdq.add_argument("--traffic-manager-port", type=int, default=8000)
    fdq.add_argument("--gpu-rank", type=int, default=0)
    fdq.add_argument("--skip-server-check", action="store_true")
    fdq.add_argument("--wall-timeout", type=float, help="per-job infrastructure wall timeout in seconds")
    fdq.add_argument("--fast-session-validation", action="store_true")

    safety = commands.add_parser("run-safety", help="create or execute the independent 6-case Safety Priority workload")
    safety.add_argument("--agent", default=None, help="built-in agent name (default: simlingo)")
    safety.add_argument("--agent-path", type=_path)
    safety.add_argument("--agent-config", type=_path)
    safety.add_argument("--results", type=_path, default=ROOT / "results")
    safety.add_argument("--plan-output", type=_path, default=ROOT / "generated/safety_run_plan.json")
    safety.add_argument("--max-runs", type=int)
    safety.add_argument("--retry-finalized", action="store_true")
    safety.add_argument("--execute", action="store_true")
    safety.add_argument("--host", default="127.0.0.1")
    safety.add_argument("--port", type=int, default=2000)
    safety.add_argument("--traffic-manager-port", type=int, default=8000)
    safety.add_argument("--gpu-rank", type=int, default=0)
    safety.add_argument("--skip-server-check", action="store_true")
    safety.add_argument("--wall-timeout", type=float)
    safety.add_argument("--fast-session-validation", action="store_true")

    baseline = commands.add_parser("run-baseline", help="create or execute one frozen dev anti-gaming control")
    baseline.add_argument("--baseline", choices=BASELINE_IDS, required=True)
    baseline.add_argument("--agent", default="simlingo", help="model for language-control baselines")
    baseline.add_argument("--agent-config", type=_path, help="override the frozen baseline agent config")
    baseline.add_argument(
        "--seeds", type=_seeds, default=_seeds("1"),
        help="comma-separated evaluator seeds (default: 1; pass 1,2,3 for the full baseline matrix)",
    )
    baseline.add_argument("--results", type=_path, default=ROOT / "results")
    baseline.add_argument("--plan-output", type=_path, default=ROOT / "generated/baseline_run_plan.json")
    baseline.add_argument("--max-runs", type=int)
    baseline.add_argument("--retry-finalized", action="store_true")
    baseline.add_argument("--execute", action="store_true")
    baseline.add_argument("--host", default="127.0.0.1")
    baseline.add_argument("--port", type=int, default=2000)
    baseline.add_argument("--traffic-manager-port", type=int, default=8000)
    baseline.add_argument("--gpu-rank", type=int, default=0)
    baseline.add_argument("--skip-server-check", action="store_true")
    baseline.add_argument("--wall-timeout", type=float)

    calibrate = commands.add_parser("calibrate-deadlines", help="fail-closed oracle deadline calibration for the next freeze")
    calibrate.add_argument("--results", type=_path, default=ROOT / "results")
    calibrate.add_argument("--output", type=_path, default=ROOT / "generated/deadlines.calibrated.json")
    freeze = commands.add_parser(
        "freeze-deadlines",
        help="atomically apply a complete, current 768-witness deadline calibration",
    )
    freeze.add_argument(
        "--calibration", type=_path,
        default=ROOT / "generated/deadlines.calibrated.json",
    )
    freeze.add_argument(
        "--replace", action="store_true",
        help="replace an existing valid freeze after fully revalidating the candidate",
    )


    score = commands.add_parser("score", help="recompute contract verdicts from traces and aggregate the complete declared plan scope")
    score.add_argument("--plan", type=_path, required=True, help="saved execution plan with complete evaluation scope")
    score.add_argument("--results", type=_path, default=ROOT / "results")
    score.add_argument("--output", type=_path, default=ROOT / "generated/score.json")

    report = commands.add_parser("report", help="render benchmark metrics as Markdown and JSON")
    report.add_argument("--score", type=_path, default=ROOT / "generated/score.json")
    report.add_argument("--output", type=_path, default=ROOT / "generated/REPORT.md")
    report.add_argument("--json-output", type=_path, default=ROOT / "generated/REPORT.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if hasattr(args, "agent_path") and args.agent is None:
        args.agent = "custom" if args.agent_path is not None else "simlingo"
    build, generated = ROOT / "build/suite", ROOT / "generated"
    try:
        if args.command == "build-maps":
            value = build_registry(ROOT / "config/maps.json", generated / "maps/registry.json")
            print(json.dumps(value, indent=2))
            return 1 if value["errors"] else 0
        if args.command == "review-assets":
            registry = generated / "maps/registry.json"
            if not registry.exists():
                build_registry(ROOT / "config/maps.json", registry)
            build_review_page(registry, ROOT / "assets/review", generated / "review/index.html")
            print(json.dumps(review_gaps(ROOT / "assets/review"), indent=2))
            return 0
        if args.command == "build-suite":
            generated_source = None
            if not args.reuse_sources:
                generated_source = generate_core_sources(ROOT, ROOT / "sources/core_pcus.json")
            value = compile_suite(
                ROOT / "sources",
                build,
                ROOT / "config/defaults.json",
                not args.allow_incomplete,
            )
            supplementary = build_supplementary(ROOT) if not args.allow_incomplete else None
            print(json.dumps({
                "source_generation": generated_source, "suite": value,
                "supplementary": supplementary,
            }, indent=2))
            return 0
        if args.command == "validate":
            defaults = json.loads((ROOT / "config/defaults.json").read_text(encoding="utf-8"))
            value = validate_suite(load_sources(ROOT / "sources"), defaults, not args.allow_incomplete)
            print(json.dumps(value, indent=2))
            return 0 if value["valid"] else 1
        if args.command == "visualize":
            print(f"rendered {visualize_suite(build / 'manifest.json', generated / 'visualizations')} PCUs")
            return 0
        if args.command == "qualify":
            value = structural_qualification(ROOT, generated / "structural_qualification.json")
            print(json.dumps({key: value[key] for key in (
                "qualification_name", "status", "is_dynamic_carla_rollout", "pcu_count", "arm_count",
                "arms_passed", "arms_failed", "geometric_targets_checked", "geometric_targets_passed",
                "opposite_geometric_targets_checked", "opposite_geometric_targets_clear",
                "dynamic_oracle_status", "errors",
            )}, indent=2))
            return 0 if value["status"] == "passed" else 1
        if args.command == "build-supplementary":
            value = build_supplementary(ROOT)
            print(json.dumps(value, indent=2))
            return 0
        if args.command == "validate-supplementary":
            value = validate_supplementary(ROOT)
            print(json.dumps(value, indent=2))
            return 0 if value["valid"] else 1
        if args.command == "freeze-runtime-assets":
            if CARLA_RUNTIME_MANIFEST.exists() and not args.replace:
                raise ValueError(
                    "CARLA runtime manifest already exists; pass --replace to refresh it"
                )
            witness_payload = _strict_cli_json(subprocess.run(
                [
                    sys.executable, "-c",
                    "import json,sys,carla; c=carla.Client(sys.argv[1],int(sys.argv[2])); c.set_timeout(15); "
                    "print(json.dumps({'client_version':str(c.get_client_version()),'server_version':str(c.get_server_version()),'available_maps':c.get_available_maps()}))",
                    args.host, str(args.port),
                ],
                env=evaluator_environment(),
                text=True,
                capture_output=True,
                timeout=25,
                check=True,
            ).stdout, "CARLA available-maps witness")
            class _WitnessClient:
                def get_available_maps(self): return witness_payload["available_maps"]
                def get_client_version(self): return witness_payload["client_version"]
                def get_server_version(self): return witness_payload["server_version"]
            witness = carla_runtime_assets_module.build_available_maps_witness(
                CARLA_RUNTIME_ROOT, _WitnessClient(),
            )
            atomic_json(CARLA_RUNTIME_MAP_WITNESS, witness)
            manifest, provenance = freeze_carla_runtime_manifest(
                CARLA_RUNTIME_ROOT,
                CARLA_RUNTIME_MANIFEST,
                CARLA_RUNTIME_PROVENANCE,
                generator_path=Path(carla_runtime_assets_module.__file__),
                available_maps_witness_path=CARLA_RUNTIME_MAP_WITNESS,
            )
            print(json.dumps({
                "manifest": str(CARLA_RUNTIME_MANIFEST),
                "provenance": str(CARLA_RUNTIME_PROVENANCE),
                "available_maps_witness": str(CARLA_RUNTIME_MAP_WITNESS),
                "resolved_carla_root": manifest["resolved_carla_root"],
                "file_count": manifest["file_count"],
                "total_size_bytes": manifest["total_size_bytes"],
                "inventory_sha256": manifest["inventory_sha256"],
                "manifest_sha256": provenance["manifest_sha256"],
                "server_version": witness["server_version"],
                "available_map_count": len(witness["available_maps"]),
            }, indent=2))
            return 0
        if args.command == "calibrate-deadlines":
            value = calibrate_deadlines(ROOT, args.results, args.output)
            print(json.dumps({
                "output": str(args.output) if value["status"] == "passed" else None,
                "output_written": value["status"] == "passed",
                **{key: value[key] for key in (
                    "status", "required_runs", "valid_runs", "expected_arms",
                    "calibrated_arms", "formula",
                )},
            }, indent=2))
            return 0 if value["status"] == "passed" else 2
        if args.command == "freeze-deadlines":
            value = freeze_deadlines(ROOT, args.calibration, replace=args.replace)
            print(json.dumps({
                key: value.get(key) for key in (
                    "status", "valid", "changed", "freeze_id", "base_sidecar_sha256",
                    "derived_sidecar_sha256", "overlay_sha256", "errors",
                )
            }, indent=2))
            return 0 if value.get("valid") else 2
        if args.command == "run":
            if args.max_runs is not None and args.max_runs <= 0:
                raise ValueError("--max-runs must be positive")
            if args.tier != "large" and args.track != "standardized":
                raise ValueError("--tier is only valid for the standardized main benchmark")
            if args.ledger is not None and args.tier != "large":
                raise ValueError("--tier cannot be combined with a custom --ledger")
            ledger = args.ledger or default_ledger(build, args.track, pilot=args.pilot, tier=args.tier)
            if args.pilot and args.track != "standardized":
                raise ValueError("--pilot is only valid for --track standardized")
            if args.track == "oracle" and args.agent != "oracle" and args.agent_path is None:
                raise ValueError("--track oracle requires --agent oracle or an explicit privileged --agent-path")
            resolved_inputs = resolve_plan_inputs(
                project_root=ROOT,
                build_dir=build,
                track=args.track,
                agent=args.agent,
                agent_path=args.agent_path,
                agent_config=args.agent_config,
                host=args.host,
                port=args.port,
                gpu_rank=args.gpu_rank,
            )
            runs, selection = select_runs(
                ledger,
                args.results,
                track=args.track,
                split=args.split,
                seeds=args.seeds,
                route_entry_ids=args.route_entry_ids,
                sidecar_path=resolved_inputs["sidecar"],
                expected_input_hashes=resolved_inputs["input_hashes"],
                retry_finalized=args.retry_finalized,
                max_runs=args.max_runs,
            )
            plan = make_plan(
                runs,
                project_root=ROOT,
                build_dir=build,
                results_dir=args.results,
                track=args.track,
                agent=args.agent,
                agent_path=args.agent_path,
                agent_config=args.agent_config,
                host=args.host,
                port=args.port,
                traffic_manager_port=args.traffic_manager_port,
                gpu_rank=args.gpu_rank,
                resolved_inputs=resolved_inputs,
            )
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            stamp_scope(plan, args, ledger, ROOT)
            atomic_json(args.plan_output, plan)
            summary: dict[str, object] = {
                "mode": "execute" if args.execute else "plan_only",
                "ledger": str(ledger),
                "plan": str(args.plan_output),
                "plan_sha256": plan["plan_sha256"],
                "carla_server_prerequisite": plan["carla_server_prerequisite"],
                **selection,
            }
            if args.execute:
                summary["execution"] = execute_plan(
                    plan,
                    results_dir=args.results,
                    require_server=not args.skip_server_check,
                    wall_timeout_seconds=args.wall_timeout,
                    validation_mode=(
                        "session" if args.fast_session_validation else "strict"
                    ),
                )
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "run-fdq":
            if args.max_runs is not None and args.max_runs <= 0:
                raise ValueError("--max-runs must be positive")
            ledger = default_ledger(build, "fdq")
            resolved_inputs = resolve_fdq_inputs(
                agent=args.agent,
                agent_path=args.agent_path,
                agent_config=args.agent_config,
                host=args.host,
                port=args.port,
                gpu_rank=args.gpu_rank,
            )
            runs, selection = select_fdq_runs(
                ledger,
                args.results,
                seeds=args.seeds,
                expected_input_hashes=resolved_inputs["input_hashes"],
                retry_finalized=args.retry_finalized,
                max_runs=args.max_runs,
            )
            plan = make_fdq_plan(
                runs,
                results_dir=args.results,
                agent=args.agent,
                agent_path=args.agent_path,
                agent_config=args.agent_config,
                host=args.host,
                port=args.port,
                traffic_manager_port=args.traffic_manager_port,
                gpu_rank=args.gpu_rank,
                resolved_inputs=resolved_inputs,
            )
            args.plan_output.parent.mkdir(parents=True, exist_ok=True)
            stamp_scope(plan, args, ledger, ROOT)
            atomic_json(args.plan_output, plan)
            summary: dict[str, object] = {
                "mode": "execute" if args.execute else "plan_only",
                "ledger": str(ledger),
                "plan": str(args.plan_output),
                "plan_sha256": plan["plan_sha256"],
                "carla_server_prerequisite": plan["carla_server_prerequisite"],
                **selection,
            }
            if args.execute:
                summary["execution"] = execute_plan(
                    plan,
                    results_dir=args.results,
                    require_server=not args.skip_server_check,
                    wall_timeout_seconds=args.wall_timeout,
                    validation_mode=(
                        "session" if args.fast_session_validation else "strict"
                    ),
                )
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "run-safety":
            if args.max_runs is not None and args.max_runs <= 0:
                raise ValueError("--max-runs must be positive")
            safety_dir = ROOT / "build/safety"
            ledger = safety_dir / "expected_runs.json"
            resolved_inputs = resolve_plan_inputs(
                project_root=ROOT, build_dir=safety_dir, track="standardized",
                agent=args.agent, agent_path=args.agent_path,
                agent_config=args.agent_config, evaluator_track="standardized",
                route_xml=safety_dir / "routes_standardized.xml",
                sidecar=safety_dir / "sidecar.private.json", workload="safety_priority",
                host=args.host, port=args.port, gpu_rank=args.gpu_rank,
            )
            runs, selection = select_runs(
                ledger, args.results, track="standardized", split="dev", seeds=[1],
                sidecar_path=resolved_inputs["sidecar"],
                expected_input_hashes=resolved_inputs["input_hashes"],
                retry_finalized=args.retry_finalized, max_runs=args.max_runs,
            )
            plan = make_plan(
                runs, project_root=ROOT, build_dir=safety_dir, results_dir=args.results,
                track="standardized", agent=args.agent, agent_path=args.agent_path,
                agent_config=args.agent_config, host=args.host, port=args.port,
                traffic_manager_port=args.traffic_manager_port, gpu_rank=args.gpu_rank,
                evaluator_track="standardized", route_xml=safety_dir / "routes_standardized.xml",
                sidecar=safety_dir / "sidecar.private.json", workload="safety_priority",
                resolved_inputs=resolved_inputs,
            )
            stamp_scope(plan, args, ledger, ROOT)
            atomic_json(args.plan_output, plan)
            summary = {"mode": "execute" if args.execute else "plan_only", "ledger": str(ledger),
                       "plan": str(args.plan_output), "plan_sha256": plan["plan_sha256"], **selection}
            if args.execute:
                summary["execution"] = execute_plan(
                    plan, results_dir=args.results, require_server=not args.skip_server_check,
                    wall_timeout_seconds=args.wall_timeout,
                    validation_mode=(
                        "session" if args.fast_session_validation else "strict"
                    ),
                )
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "run-baseline":
            if args.max_runs is not None and args.max_runs <= 0:
                raise ValueError("--max-runs must be positive")
            baseline_dir = ROOT / "build/baselines"
            manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
            spec = manifest["baselines"][args.baseline]
            ledger = ROOT / spec["ledger"]
            fixed = spec["agent"] == "fixed_policy"
            agent_path = ROOT / "tools/l2d_fixed_policy_agent.py" if fixed else None
            agent_config = args.agent_config or (ROOT / spec["agent_config"] if fixed else None)
            configured_sidecar = ROOT / spec["private_sidecar"]
            core_base_sidecar = ROOT / "build/suite/sidecar.private.json"
            actual_sidecar = (
                active_core_sidecar_path(ROOT)
                if configured_sidecar.resolve() == core_base_sidecar.resolve()
                else configured_sidecar
            )
            resolved_inputs = resolve_plan_inputs(
                project_root=ROOT, build_dir=baseline_dir, track="baseline",
                agent="custom" if fixed else args.agent, agent_path=agent_path,
                agent_config=agent_config, evaluator_track=spec["evaluator_track"],
                route_xml=ROOT / spec["route_xml"], sidecar=actual_sidecar,
                workload=f"baseline:{args.baseline}",
                host=args.host, port=args.port, gpu_rank=args.gpu_rank,
            )
            runs, selection = select_runs(
                ledger, args.results, track="baseline", split="dev", seeds=args.seeds,
                sidecar_path=resolved_inputs["sidecar"],
                expected_input_hashes=resolved_inputs["input_hashes"],
                retry_finalized=args.retry_finalized, max_runs=args.max_runs,
            )
            plan = make_plan(
                runs, project_root=ROOT, build_dir=baseline_dir, results_dir=args.results,
                track="baseline", agent="custom" if fixed else args.agent,
                agent_path=agent_path, agent_config=agent_config,
                host=args.host, port=args.port, traffic_manager_port=args.traffic_manager_port,
                gpu_rank=args.gpu_rank, evaluator_track=spec["evaluator_track"],
                route_xml=ROOT / spec["route_xml"], sidecar=actual_sidecar,
                workload=f"baseline:{args.baseline}",
                resolved_inputs=resolved_inputs,
            )
            stamp_scope(plan, args, ledger, ROOT)
            atomic_json(args.plan_output, plan)
            summary = {"mode": "execute" if args.execute else "plan_only", "baseline": args.baseline,
                       "ledger": str(ledger), "plan": str(args.plan_output),
                       "plan_sha256": plan["plan_sha256"], **selection}
            if args.execute:
                summary["execution"] = execute_plan(
                    plan, results_dir=args.results, require_server=not args.skip_server_check,
                    wall_timeout_seconds=args.wall_timeout,
                )
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "score":
            value = score_plan(ROOT, args.plan, args.results, args.output)
            print(json.dumps({"output": str(args.output), "model": value["model"],
                              "workload": value["workload"],
                              "expected_runs": len(value["scope"]["run_ids"])}, indent=2))
            return 0
        value = render(
            args.score,
            args.output,
            project_root=ROOT,
            json_output=args.json_output,
        )
        print(json.dumps({
            "report": str(args.output),
            "report_json": str(args.json_output),
            "model": value["model"],
            "workload": value["workload"],
        }, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"l2d: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
