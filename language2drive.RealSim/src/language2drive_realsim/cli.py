"""Language2Drive RealSim AlpaSim command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from .acquisition import acquire_shortlist
from .audit import inspect_environment
from .compiler import compile_to_directory
from .curation_review import build_curation_review
from .curation_review_server import serve_curation_review
from .discovery import discover_candidates
from .evaluation import evaluate_rollouts
from .finalize import final_report
from .feasibility import prepare_feasibility, prepare_additional_feasibility
from .io import atomic_json, load_json
from .local_runtime import prepare_local_runtime
from .provenance import build_provenance
from .preview_mining import build_preview_review
from .qualification import qualify_source
from .runtime_client import run as run_runtime
from .schema import SchemaError, validate_source
from .topology_mining import mine_shortlist_topology
from .verifier_selftest import run_verifier_selftests


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _ranked_mining_scene_ids(
    shortlist: Any, preview_selection: Any, limit: int
) -> List[str]:
    """Keep the acquisition shortlist, then fill from reviewed preview rank."""

    scene_ids: List[str] = []
    seen = set()
    for group in shortlist["groups"].values():
        for scene_id in group:
            if scene_id not in seen:
                seen.add(scene_id)
                scene_ids.append(scene_id)
    for row in preview_selection.get("scenes", []):
        scene_id = row["clip_id"]
        if scene_id not in seen:
            seen.add(scene_id)
            scene_ids.append(scene_id)
        if limit > 0 and len(scene_ids) >= limit:
            break
    return scene_ids[:limit] if limit > 0 else scene_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l2d-realsim")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("inspect")
    acquire = commands.add_parser("acquire-scenes")
    acquire.add_argument("--shortlist", type=Path)
    acquire.add_argument("--scene-cache", type=Path)
    acquire.add_argument("--workers", type=int, default=4)
    acquire.add_argument("--inventory-only", action="store_true")
    acquire.add_argument(
        "--minimum-free-gib",
        type=float,
        default=20.0,
        help="stop scheduling new USDZ downloads below this shared-disk reserve",
    )
    discover = commands.add_parser("discover")
    discover.add_argument("--limit", type=int, default=120)
    discover.add_argument("--suite", default="public_2601")
    discover.add_argument("--annotations", type=Path)
    discover.add_argument("--scene-cache", type=Path)
    discover.add_argument("--shortlist", type=Path)
    discover.add_argument("--catalog-wide", action="store_true")

    topology = commands.add_parser("mine-topology")
    topology.add_argument("--shortlist", type=Path)
    topology.add_argument("--scene-cache", type=Path)

    curation = commands.add_parser("prepare-curation")
    curation.add_argument("--topology-report", type=Path)
    curation.add_argument("--preview-inventory", type=Path)
    curation.add_argument("--preview-root", type=Path)
    curation.add_argument("--output-dir", type=Path)
    curation.add_argument("--no-render", action="store_true")

    feasibility = commands.add_parser("prepare-feasibility")
    feasibility.add_argument("--inventory", type=Path)
    feasibility.add_argument("--decisions", type=Path)
    feasibility.add_argument("--review-batch", help="registered independent review/acquisition batch")

    local = commands.add_parser("prepare-local-runtime")
    local.add_argument("--agent-candidate", type=Path,
                       help="explicit user-delegated offline candidate; no human approval is synthesized")
    local.add_argument("--wizard-dir", type=Path, required=True)
    local.add_argument("--output-dir", type=Path, required=True)
    local.add_argument("--gpu", type=int, required=True)
    local.add_argument("--proposal-key")
    local.add_argument("--review-batch", help="use only this registered batch's saved human decisions")
    local.add_argument("--runtime-port", type=int, default=6008)
    local.add_argument("--driver-port", type=int, default=50052)
    local.add_argument("--metrics-port", type=int, default=6000)

    serve_curation = commands.add_parser("serve-curation")
    serve_curation.add_argument("--bind", default="127.0.0.1")
    serve_curation.add_argument("--port", type=int, default=8765)
    serve_curation.add_argument("--inventory", type=Path)
    serve_curation.add_argument("--decisions", type=Path)
    serve_curation.add_argument("--output-dir", type=Path)
    serve_curation.add_argument("--agent-review-only", action="store_true",
                                help="Only accept delegated-pilot video review writes; keep initial review service separate")

    source_review = commands.add_parser("prepare-source-review")
    source_review.add_argument("--index", type=Path, required=True)
    source_review.add_argument("--output", type=Path, required=True)
    source_server = commands.add_parser("serve-source-review")
    source_server.add_argument("--inventory", type=Path, required=True)
    source_server.add_argument("--decisions", type=Path, required=True)
    source_server.add_argument("--bind", default="127.0.0.1")
    source_server.add_argument("--port", type=int, default=8767)
    reviewed_source = commands.add_parser("materialize-reviewed-source")
    reviewed_source.add_argument("--inventory", type=Path, required=True)
    reviewed_source.add_argument("--decisions", type=Path, required=True)
    reviewed_source.add_argument("--output", type=Path, required=True)
    staged_source = commands.add_parser("stage-reviewed-source",
        help="CPU preflight source containing current human approvals only; not a reduced release")
    staged_source.add_argument("--inventory", type=Path, required=True)
    staged_source.add_argument("--decisions", type=Path, required=True)
    staged_source.add_argument("--output", type=Path, required=True)
    migration = commands.add_parser("migrate-unchanged-source-reviews",
        help="Preserve exact unchanged source decisions in a new log; never approve changed/new targets")
    migration.add_argument("--previous-inventory", type=Path, required=True)
    migration.add_argument("--previous-decisions", type=Path, required=True)
    migration.add_argument("--inventory", type=Path, required=True)
    migration.add_argument("--output", type=Path, required=True)

    review = commands.add_parser("review-previews")
    review.add_argument("--selection", type=Path)
    review.add_argument("--preview-root", type=Path)
    review.add_argument("--output-dir", type=Path)
    review.add_argument("--workers", type=int, default=8)

    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--source", type=Path)
    compile_parser.add_argument("--output-dir", type=Path)

    validate = commands.add_parser("validate")
    validate.add_argument("--source", type=Path)
    validate.add_argument("--release", action="store_true")
    validate.add_argument("--report-dir", type=Path)

    qualify = commands.add_parser("qualify")
    qualify.add_argument("--source", type=Path)
    qualify.add_argument("--compiled-dir", type=Path)
    qualify.add_argument("--evidence-dir", type=Path)
    qualify.add_argument("--report-dir", type=Path)

    oracle = commands.add_parser("oracle")
    oracle.add_argument("--submit", action="store_true")
    oracle.add_argument("--dry-run", action="store_true")
    oracle.add_argument("--compiled-dir", type=Path)
    oracle.add_argument("--output-dir", type=Path)
    oracle.add_argument("--runtime-address", default="127.0.0.1:50050")
    oracle.add_argument("--driver-address", default="127.0.0.1:50052")
    oracle.add_argument("--concurrency", type=int, default=1)
    oracle.add_argument("--registered-driver", action="store_true")
    oracle.add_argument("--witness-index", type=Path)

    run = commands.add_parser("run")
    run.add_argument("--compiled-dir", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--runtime-address", default="127.0.0.1:50050")
    run.add_argument("--driver-address", default="127.0.0.1:50051")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--registered-driver", action="store_true")
    run.add_argument(
        "--model-id",
        choices=("nvidia/Alpamayo-1.5-10B", "nvidia/Alpamayo2-Super"),
        default="nvidia/Alpamayo2-Super",
    )
    run.add_argument("--model-checkpoint", type=Path)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--source", type=Path)
    evaluate.add_argument("--compiled-dir", type=Path)
    evaluate.add_argument("--evidence-dir", type=Path)
    evaluate.add_argument("--report-dir", type=Path)
    evaluate.add_argument("--run-manifest", type=Path, help="full-cohort actual model manifest; enables formal authority checks")
    evaluate.add_argument("--qualification-report", type=Path, help="matching successful full-cohort oracle qualification")

    finalize = commands.add_parser("finalize", help="check PCU reports; does not certify the full model campaign")
    finalize.add_argument("--source", type=Path)
    finalize.add_argument("--compiled-dir", type=Path)
    finalize.add_argument("--report-dir", type=Path)
    finalize.add_argument("--inspect-report", type=Path, help="historical inspection for comparison; live inspection remains authoritative")
    return parser


def _paths(root: Path) -> Any:
    generated = root / "generated"
    return {
        "source": root / "data" / "private" / "pcu_source.json",
        "annotations": root / "data" / "private" / "scene_annotations.json",
        "compiled": generated / "compiled",
        "evidence": root / ".runtime" / "oracle_evidence",
        "thresholds": root / "config" / "threshold_profile.json",
        "generated": generated,
    }


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Accept the requested long form as well as the compact installed command.
    if raw[:2] == ["realsim", "alpasim"]:
        raw = raw[2:]
    args = _parser().parse_args(raw)
    root = args.project_root.resolve()
    paths = _paths(root)
    paths["generated"].mkdir(parents=True, exist_ok=True)

    # Versioned runs must not silently overwrite a previous report. Legacy
    # invocations retain their existing default paths and overwrite behavior.
    report_dir = getattr(args, "report_dir", None) or paths["generated"]
    report_outputs = {
        "validate": ("verifier_test_report.json",),
        "qualify": ("verifier_test_report.json", "qualification_report.json", "oracle_witness_index.json"),
        "evaluate": ("evaluation_report.json",),
        "finalize": ("provenance.json", "final_report.json"),
    }
    if getattr(args, "report_dir", None) is not None:
        collisions = [str(report_dir / name) for name in report_outputs[args.command]
                      if (report_dir / name).exists() or (report_dir / name).is_symlink()]
        if collisions:
            _print({"pass": False, "error": "refusing to overwrite versioned reports", "paths": collisions})
            return 2

    if args.command == "inspect":
        report = inspect_environment(root)
        atomic_json(paths["generated"] / "inspect_report.json", report)
        _print(report)
        return 0

    if args.command == "prepare-local-runtime":
        report = prepare_local_runtime(root, args.wizard_dir, args.output_dir, args.gpu, args.proposal_key,
                                       args.runtime_port, args.driver_port, args.metrics_port,
                                       review_batch=args.review_batch, agent_candidate=args.agent_candidate)
        _print({"output_dir": str(args.output_dir.resolve()), "proposal_key": report["candidate"]["proposal_key"],
                "not_started": report["not_started"], "scope": report["scope"]})
        return 0

    if args.command == "acquire-scenes":
        scene_cache = (
            args.scene_cache
            or root / "vendor" / "alpasim" / "data" / "nre-artifacts" / "all-usdzs"
        )
        report = acquire_shortlist(
            args.shortlist or root / "config" / "mining_shortlist_v0.1.json",
            root / "vendor" / "alpasim" / "data" / "scenes" / "sim_scenes.csv",
            root / "vendor" / "alpasim" / "data" / "scenes" / "sim_suites.csv",
            scene_cache,
            args.workers,
            args.inventory_only,
            args.minimum_free_gib,
        )
        atomic_json(paths["generated"] / "acquisition_report.json", report)
        _print(
            {
                key: report[key]
                for key in (
                    "selected_scene_count",
                    "catalog_resolved_count",
                    "downloaded_artifact_count",
                    "verified_artifact_count",
                    "complete",
                    "errors",
                )
            }
        )
        return 0 if report["complete"] else 2

    if args.command == "discover":
        selected_scene_ids = None
        if not args.catalog_wide:
            shortlist = load_json(
                args.shortlist or root / "config" / "mining_shortlist_v0.1.json"
            )
            preview_selection_path = root / "generated" / "preview_selection.json"
            preview_selection = (
                load_json(preview_selection_path)
                if preview_selection_path.is_file()
                else {"scenes": []}
            )
            selected_scene_ids = _ranked_mining_scene_ids(
                shortlist, preview_selection, args.limit
            )
        report = discover_candidates(
            root / "vendor" / "alpasim" / "data" / "scenes" / "sim_scenes.csv",
            root / "vendor" / "alpasim" / "data" / "scenes" / "sim_suites.csv",
            args.suite,
            args.limit,
            args.annotations or paths["annotations"],
            args.scene_cache
            or root / "vendor" / "alpasim" / "data" / "nre-artifacts" / "all-usdzs",
            selected_scene_ids,
        )
        atomic_json(paths["generated"] / "candidate_inventory.json", report)
        _print(
            {
                key: report[key]
                for key in (
                    "candidate_count",
                    "downloaded_artifact_count",
                    "verified_artifact_count",
                    "prefilter_pass_count",
                    "suite",
                )
            }
        )
        return 0

    if args.command == "mine-topology":
        report = mine_shortlist_topology(
            args.shortlist or root / "config" / "mining_shortlist_v0.1.json",
            root / "vendor" / "alpasim" / "data" / "scenes" / "sim_scenes.csv",
            args.scene_cache
            or root / "vendor" / "alpasim" / "data" / "nre-artifacts" / "all-usdzs",
        )
        atomic_json(paths["generated"] / "topology_mining_report.json", report)
        _print(
            {
                key: report[key]
                for key in (
                    "selected_scene_count",
                    "verified_artifact_count",
                    "xodr_mined_count",
                    "route_aligned_mined_count",
                    "route_branch_hypothesis_count",
                    "lane_choice_hypothesis_count",
                    "route_aligned_opportunity_count",
                    "scene_error_count",
                    "qualification_warning",
                )
            }
        )
        return 0 if report["xodr_mined_count"] == report["selected_scene_count"] else 2

    if args.command == "prepare-curation":
        report = build_curation_review(
            args.topology_report
            or paths["generated"] / "topology_mining_report.json",
            args.preview_inventory
            or paths["generated"] / "preview_review_inventory.json",
            args.preview_root or root / "data" / "mining" / "nurec_26.01",
            args.output_dir or root / ".runtime" / "curation_review",
            root,
            render_evidence=not args.no_render,
        )
        atomic_json(
            paths["generated"] / "curation_review_inventory.private.json", report
        )
        _print(
            {
                key: report[key]
                for key in (
                    "review_scene_count",
                    "proposal_count",
                    "rendered_evidence_count",
                    "evidence_error_count",
                    "rendered_map_count",
                    "map_error_count",
                    "timing_prefilter_pass_count",
                    "timing_prefilter_reject_count",
                    "timing_prefilter_survivor_count",
                    "review_index",
                    "review_index_error",
                    "review_index_survivor_count",
                    "pending_curator_review_count",
                    "auto_qualified_count",
                )
            }
        )
        return 0 if (
            args.no_render
            or report["rendered_evidence_count"] == report["proposal_count"]
        ) else 2

    if args.command == "prepare-feasibility":
        if args.review_batch is not None:
            if args.inventory is not None or args.decisions is not None:
                raise ValueError("review-batch cannot be mixed with explicit review input paths")
            report = prepare_additional_feasibility(root, args.review_batch)
            queue_path = paths["generated"] / ("feasibility_queue_" + args.review_batch + ".private.json")
        else:
            report = prepare_feasibility(root, args.inventory, args.decisions)
            queue_path = paths["generated"] / "feasibility_queue.private.json"
        atomic_json(queue_path, report)
        _print(report["summary"])
        return 2 if report["invalid_reviews"] else 0

    if args.command == "prepare-source-review":
        from .source_review import prepare_source_review
        _print(prepare_source_review(root, args.index, args.output))
        return 0

    if args.command == "serve-source-review":
        from .source_review_server import serve_source_review
        serve_source_review(root, args.inventory, args.decisions, args.bind, args.port)
        return 0

    if args.command == "migrate-unchanged-source-reviews":
        from .source_review import migrate_unchanged_reviews
        _print(migrate_unchanged_reviews(root, args.previous_inventory, args.previous_decisions,
                                        args.inventory, args.output))
        return 0

    if args.command in {"materialize-reviewed-source", "stage-reviewed-source"}:
        from .source_review import SourceReviewStore
        store = SourceReviewStore(root, args.inventory, args.decisions)
        operation = store.stage_accepted if args.command == "stage-reviewed-source" else store.materialize
        _print(operation(args.output))
        return 0

    if args.command == "serve-curation":
        serve_curation_review(
            root,
            bind=args.bind,
            port=args.port,
            inventory_path=args.inventory,
            decisions_path=args.decisions,
            static_root=args.output_dir,
            agent_review_only=args.agent_review_only,
        )
        return 0

    if args.command == "review-previews":
        report = build_preview_review(
            args.selection or paths["generated"] / "preview_selection.json",
            args.preview_root or root / "data" / "mining" / "nurec_26.01",
            args.output_dir or root / "data" / "mining" / "contact_sheets",
            paths["generated"] / "preview_review_inventory.json",
            args.workers,
        )
        _print(
            {
                key: report[key]
                for key in (
                    "candidate_count",
                    "downloaded_preview_count",
                    "valid_preview_count",
                    "ocr_available_count",
                    "ocr_error_count",
                    "ocr_nonempty_count",
                )
            }
        )
        return 0 if report["valid_preview_count"] == report["candidate_count"] else 2

    if args.command == "compile":
        source_path = args.source or paths["source"]
        output_dir = args.output_dir or paths["compiled"]
        outputs = compile_to_directory(load_json(source_path), output_dir)
        _print({key: str(value) for key, value in outputs.items()})
        return 0

    if args.command == "validate":
        source_path = args.source or paths["source"]
        try:
            validate_source(load_json(source_path), require_exact_release=args.release)
            structural = {"pass": True, "error": None, "release_mode": args.release}
        except (SchemaError, OSError, ValueError) as error:
            structural = {"pass": False, "error": str(error), "release_mode": args.release}
        tests = run_verifier_selftests()
        atomic_json(report_dir / "verifier_test_report.json", tests)
        report = {"structural": structural, "verifier_tests": {"pass": tests["pass"]}}
        _print(report)
        return 0 if structural["pass"] and tests["pass"] else 2

    if args.command == "qualify":
        source_path = args.source or paths["source"]
        compiled = args.compiled_dir or paths["compiled"]
        evidence = args.evidence_dir or paths["evidence"]
        tests = run_verifier_selftests()
        atomic_json(report_dir / "verifier_test_report.json", tests)
        report, witnesses = qualify_source(
            load_json(source_path),
            load_json(compiled / "sidecar.private.json"),
            load_json(compiled / "leakage_audit.json"),
            evidence,
            load_json(paths["thresholds"]),
            tests,
        )
        atomic_json(report_dir / "qualification_report.json", report)
        atomic_json(report_dir / "oracle_witness_index.json", witnesses)
        _print({"pass": report["pass"], "eligible_count": report["eligible_count"], "global_errors": report["global_errors"]})
        return 0 if report["pass"] else 2

    if args.command == "oracle":
        if args.submit and args.witness_index is not None:
            _print({"pass": False, "error": "--witness-index cannot be used with --submit"})
            return 2
        if args.submit:
            compiled = args.compiled_dir or paths["compiled"]
            output_dir = args.output_dir or (root / ".runtime" / "oracle_run")
            manifest = run_runtime(
                compiled / "public_workload.json",
                compiled / "sidecar.private.json",
                output_dir,
                args.runtime_address,
                args.driver_address,
                args.concurrency,
                args.dry_run,
                track="privileged_oracle",
                use_registered_driver=args.registered_driver,
            )
            _print(manifest)
            return 0
        witness_path = args.witness_index or paths["generated"] / "oracle_witness_index.json"
        if not witness_path.exists():
            _print({"pass": False, "error": "run qualify first"})
            return 2
        witness = load_json(witness_path)
        _print({key: witness[key] for key in ("coverage_numerator", "coverage_denominator", "coverage_complete")})
        return 0 if witness["coverage_complete"] else 2

    if args.command == "run":
        compiled = args.compiled_dir or paths["compiled"]
        output_dir = args.output_dir or (root / ".runtime" / "run")
        model_checkpoint = args.model_checkpoint
        if model_checkpoint is None and args.model_id == "nvidia/Alpamayo2-Super":
            model_checkpoint = Path(
                os.environ.get(
                    "L2D_ALPAMAYO2_CHECKPOINT",
                    root / "model_zoo" / "alpamayo2-super",
                )
            )
        if model_checkpoint is None:
            configured = os.environ.get("L2D_ALPAMAYO_CHECKPOINT")
            if not configured:
                raise ValueError(
                    "Alpamayo 1.5 run requires --model-checkpoint or "
                    "L2D_ALPAMAYO_CHECKPOINT"
                )
            model_checkpoint = Path(configured)
        manifest = run_runtime(
            compiled / "public_workload.json",
            compiled / "sidecar.private.json",
            output_dir,
            args.runtime_address,
            args.driver_address,
            args.concurrency,
            args.dry_run,
            model_id=args.model_id,
            model_checkpoint=model_checkpoint,
            use_registered_driver=args.registered_driver,
        )
        _print(manifest)
        return 0

    if args.command == "evaluate":
        source_path = args.source or paths["source"]
        compiled = args.compiled_dir or paths["compiled"]
        evidence = args.evidence_dir or (root / ".runtime" / "model_evidence")
        if bool(args.run_manifest) != bool(args.qualification_report):
            raise ValueError("formal model evaluation requires both --run-manifest and --qualification-report")
        if args.run_manifest:
            from .model_evaluation import evaluate_model_rollouts
            report = evaluate_model_rollouts(
                load_json(source_path), load_json(compiled / "public_workload.json"),
                load_json(compiled / "sidecar.private.json"), evidence,
                load_json(paths["thresholds"]), load_json(args.run_manifest),
                load_json(args.qualification_report),
            )
        else:
            report = evaluate_rollouts(
                load_json(source_path),
                load_json(compiled / "sidecar.private.json"),
                evidence,
                load_json(paths["thresholds"]),
            )
        atomic_json(report_dir / "evaluation_report.json", report)
        _print(
            {
                "expected_rollouts": report["expected_rollouts"],
                "observed_rollouts": report["observed_rollouts"],
                "denominator_complete": report["denominator_complete"],
                "formal_model_evaluation_complete": report.get("formal_model_evaluation_complete", False),
                "metrics": report["metrics"],
            }
        )
        complete = report.get("formal_model_evaluation_complete") if args.run_manifest else report["denominator_complete"]
        return 0 if complete else 2

    if args.command == "finalize":
        source_path = args.source or paths["source"]
        compiled = args.compiled_dir or paths["compiled"]
        required = {
            "source": source_path,
            "qualification": report_dir / "qualification_report.json",
            "witness": report_dir / "oracle_witness_index.json",
            "verifier": report_dir / "verifier_test_report.json",
            "inspect": args.inspect_report or report_dir / "inspect_report.json",
            "leakage": compiled / "leakage_audit.json",
            "sidecar": compiled / "sidecar.private.json",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            _print({"status": "NOT RELEASE-READY", "blockers": ["missing evidence: " + item for item in missing]})
            return 2
        # Guard small report inputs across loading and provenance construction.
        # This catches observed concurrent changes, not malicious stat spoofing.
        optional = {
            "candidate": paths["generated"] / "candidate_inventory.json",
            "curation": paths["generated"] / "curation_review_inventory.private.json",
        }
        def stamps():
            result = {}
            for key, path in {**required, **optional}.items():
                if not path.exists():
                    result[key] = None
                else:
                    stat = path.stat()
                    result[key] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            return result
        before = stamps()
        inputs = {key: load_json(path) for key, path in required.items()}
        inventories = {key: load_json(path) if path.exists() else {} for key, path in optional.items()}
        # A cached JSON with release_execution_ready=true cannot authorize the
        # current environment. This inspection is read-only and starts no jobs.
        live_inspect = inspect_environment(root)
        try:
            provenance = build_provenance(
                root, source_path, required["sidecar"], required["qualification"]
            )
        except (OSError, ValueError) as error:
            _print({"status": "NOT RELEASE-READY", "release_ready": False,
                    "blockers": ["provenance snapshot failed: " + str(error)]})
            return 2
        report = final_report(
            inputs["source"],
            inputs["qualification"],
            inputs["leakage"],
            inputs["witness"],
            inputs["verifier"],
            provenance,
            live_inspect,
            inventories["candidate"],
            inventories["curation"],
        )
        if before != stamps():
            _print({"status": "NOT RELEASE-READY", "release_ready": False,
                    "blockers": ["evidence inputs changed during finalization; no reports written"]})
            return 2
        report["environment_check_origin"] = "live_read_only_inspection"
        report["cached_inspect_readiness_matches_live"] = all(
            inputs["inspect"].get(key) == live_inspect.get(key)
            for key in ("schema_version", "alpasim_git_sha", "oracle_execution_ready", "release_execution_ready")
        )
        report["provenance_scope"] = "current_workspace_snapshot_not_execution_time_attestation"
        # Publish the final report last. Versioned outputs are no-clobber even
        # if another writer wins after preflight. These two files are NOT a
        # transaction: retain any partial evidence and report failure explicitly.
        written = []
        try:
            for name, payload in (("provenance.json", provenance), ("final_report.json", report)):
                atomic_json(report_dir / name, payload, overwrite=args.report_dir is None)
                written.append(str(report_dir / name))
        except OSError as error:
            _print({"status": "NOT RELEASE-READY", "release_ready": False,
                    "blockers": ["report publication failed: " + str(error)],
                    "partial_outputs_retained": written})
            return 2
        _print(report)
        return 0 if report["pcu_report_checks_passed"] else 2
    return 2
