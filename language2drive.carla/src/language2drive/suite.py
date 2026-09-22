from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from .frozen_paths import ProvenancePath as Path
from typing import Any

from .io import atomic_json, sha256_file, stable_hash, topology_spawn_token
from .schema import ArmIdentity, SchemaError, validate_pcu
from .tiers import TIER_PCU_COUNTS, tier_pcu_ids
from .fdq import FDQ_ROUTE_BINDING_ROWS


CATEGORY_QUOTAS = {
    "A": {"junction": 12, "lane": 8, "speed_start": 6, "waiting_parking": 6},
    "B": {"speed_constraint": 8, "lane_constraint": 8, "object_relation": 8, "passing_order": 8},
    "C": {"road_name": 8, "port_target": 8, "parking_region": 8, "exact_parking": 8},
    "D": {"strict_order": 8, "conditional": 8, "until_invariant": 8, "memory_reference": 8},
}


def load_sources(source_dir: Path) -> list[dict[str, Any]]:
    pcus = []
    for path in sorted(source_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        batch = value if isinstance(value, list) else value.get("pcus", [])
        source_geometry = {} if isinstance(value, list) else value.get("geometry_registry", {})
        source_parking_registry = (
            {} if isinstance(value, list) else value.get("parking_manoeuvre_registry", {})
        )
        source_files = {} if isinstance(value, list) else value.get("source_files", {})
        if not isinstance(source_geometry, dict):
            raise SchemaError(f"{path}: geometry_registry must be an object")
        if not isinstance(source_parking_registry, dict):
            raise SchemaError(f"{path}: parking_manoeuvre_registry must be an object")
        if not isinstance(source_files, dict):
            raise SchemaError(f"{path}: source_files must be an object")
        for raw in batch:
            pcu = dict(raw)
            pcu["_source"] = str(path)
            pcu["_source_geometry_registry"] = source_geometry
            pcu["_source_parking_registry"] = source_parking_registry
            pcu["_source_files"] = source_files
            validate_pcu(pcu)
            pcus.append(pcu)
    return pcus


def _finite_oracle_path(points: Any) -> bool:
    """Validate the private, executable trajectory representation.

    Oracle paths are consumed directly by the evaluator, so every point must
    carry a complete transform.  Keep this deliberately stricter than the
    public Standardized coarse envelope below.
    """
    return isinstance(points, list) and bool(points) and all(
        isinstance(point, dict)
        and all(axis in point and math.isfinite(float(point[axis])) for axis in ("x", "y", "z", "yaw_deg"))
        for point in points
    )


def _finite_coarse_envelope(points: Any) -> bool:
    """Validate public route hints without inventing an arm-specific yaw.

    The public XML contract only requires finite x/y coordinates (and an
    optional finite z).  Requiring a yaw here would either reject legitimate
    shared envelopes or force the compiler to publish a direction that may
    distinguish counterfactual arms.
    """
    return isinstance(points, list) and bool(points) and all(
        isinstance(point, dict)
        and all(axis in point and math.isfinite(float(point[axis])) for axis in ("x", "y"))
        and ("z" not in point or math.isfinite(float(point["z"])))
        for point in points
    )


def _semantic_entity_tokens(pcu: dict[str, Any]) -> set[str]:
    """Return stable semantic-asset identities referenced by a PCU.

    ``group_id`` alone cannot catch an entity reused by differently named
    programs.  Extract the actual gate/region/slot/name references from the
    target and contracts so dev/test separation is verified on the payload.
    """

    tokens: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            kind = str(value.get("type", ""))
            identifier = value.get("id")
            if kind in {"gate", "region", "parking_slot"} and isinstance(identifier, str):
                tokens.add(f"geometry:{identifier.casefold()}")
            gate = value.get("enter_gate")
            if isinstance(gate, dict) and isinstance(gate.get("id"), str):
                tokens.add(f"geometry:{gate['id'].casefold()}")
            binding = value.get("bind_reference")
            if isinstance(binding, dict) and isinstance(binding.get("value"), str):
                tokens.add(f"geometry:{binding['value'].casefold()}")
            for key in ("physical_space_id", "source_entity_id", "slot_number"):
                if isinstance(value.get(key), str) and value[key].strip():
                    tokens.add(f"{key}:{value[key].strip().casefold()}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for arm in pcu.get("arms", []):
        visit(arm.get("target", {}))
        visit(arm.get("contract", {}))
        visit(arm.get("opposite_contract", {}))
    # A street name is text shared by many signs.  The benchmark entity is the
    # physical sign/edge-to-exit binding, which has a unique gate and source
    # edge.  Treating the lexical string itself as an entity would falsely
    # merge independent intersections along one road.
    for evidence in pcu.get("metadata", {}).get("fixed_semantic_mapping", {}).values():
        if isinstance(evidence, dict) and isinstance(evidence.get("gate"), str):
            tokens.add("road_binding:" + stable_hash({
                "gate": evidence["gate"], "edge_token": evidence.get("edge_token")
            }))
    return tokens


def _topology_tokens(pcu: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    arms = pcu.get("arms", [])
    if arms and _finite_oracle_path(arms[0].get("oracle_path")):
        start = arms[0]["oracle_path"][0]
        # One-metre cells catch a reused physical approach while tolerating
        # sub-centimetre serialization differences.
        tokens.add(topology_spawn_token(pcu["town"], start["x"], start["y"]))

    def add_lane(identifier: str) -> None:
        tokens.add(f"lane:{identifier}")
        parts = identifier.rsplit(":", 3)
        if len(parts) == 4:
            try:
                int(parts[1]); int(parts[2]); int(parts[3])
            except ValueError:
                return
            tokens.add(f"road:{parts[0]}:{parts[1]}")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            kind = str(value.get("type", ""))
            identifier = value.get("id")
            if kind == "lane" and isinstance(identifier, str):
                add_lane(identifier)
            relation = value.get("lane_relation")
            if isinstance(relation, dict) and isinstance(relation.get("value"), str):
                add_lane(relation["value"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for arm in arms:
        visit(arm.get("target", {})); visit(arm.get("contract", {}))
        speed_proof = arm.get("signal_free_corridor_proof")
        if isinstance(speed_proof, dict):
            lane_id = speed_proof.get("lane_id")
            if isinstance(lane_id, str) and lane_id:
                # Continuous-speed programs are topology-bound to the exact
                # audited lane corridor even though their public target is a
                # speed band/gate rather than a lane predicate.  Record both
                # lane and physical road identities so opposite-direction
                # lanes on a reused road cannot leak across dev/test.
                add_lane(lane_id)
    return tokens


def _actor_program_tokens(pcu: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for arm in pcu.get("arms", []):
        setup = arm.get("actor_setup", {})
        for actor in setup.get("actors", []) if isinstance(setup, dict) else []:
            signature = {key: actor.get(key) for key in (
                "blueprint", "program", "color", "relative_spawn_m", "slot"
            ) if actor.get(key) is not None}
            tokens.add("actor_program:" + stable_hash(signature))
        if isinstance(setup, dict) and setup.get("condition_cell") is not None:
            tokens.add("condition_program:" + stable_hash({
                "condition_cell": setup.get("condition_cell"),
                "slot_occupancy": setup.get("slot_occupancy", {}),
            }))
    return tokens


def validate_suite(pcus: list[dict[str, Any]], defaults: dict[str, Any], require_complete: bool = True) -> dict[str, Any]:
    ids, errors, warnings = [pcu["id"] for pcu in pcus], [], []
    if len(ids) != len(set(ids)):
        errors.append("duplicate PCU ids")
    counts = Counter(pcu["layer"] for pcu in pcus)
    splits = Counter(pcu["split"] for pcu in pcus)
    layer_splits = Counter((pcu["layer"], pcu["split"]) for pcu in pcus)
    categories = Counter((pcu["layer"], pcu.get("category")) for pcu in pcus)
    if require_complete:
        for layer, expected in defaults["quota"].items():
            if counts[layer] != expected:
                errors.append(f"layer {layer}: expected {expected} PCUs, found {counts[layer]}")
            if layer_splits[(layer, "dev")] != defaults["dev_per_layer"]:
                errors.append(f"layer {layer}: expected {defaults['dev_per_layer']} dev PCUs, found {layer_splits[(layer, 'dev')]}")
            if layer_splits[(layer, "test")] != expected-defaults["dev_per_layer"]:
                errors.append(f"layer {layer}: expected {expected-defaults['dev_per_layer']} test PCUs, found {layer_splits[(layer, 'test')]}")
        for layer, quota in CATEGORY_QUOTAS.items():
            for category, expected in quota.items():
                if categories[(layer, category)] != expected:
                    errors.append(f"{layer}/{category}: expected {expected}, found {categories[(layer, category)]}")

    group_splits: dict[str, set[str]] = defaultdict(set)
    split_keys: dict[str, dict[str, set[str]]] = {
        kind: defaultdict(set) for kind in ("semantic_entity", "topology", "actor_program")
    }
    for pcu in pcus:
        group_splits[pcu["group_id"]].add(pcu["split"])
        for token in _semantic_entity_tokens(pcu): split_keys["semantic_entity"][token].add(pcu["split"])
        for token in _topology_tokens(pcu): split_keys["topology"][token].add(pcu["split"])
        for token in _actor_program_tokens(pcu): split_keys["actor_program"][token].add(pcu["split"])
        if not _finite_coarse_envelope(pcu.get("coarse_envelope")) or len(pcu["coarse_envelope"]) < 2:
            errors.append(f"{pcu['id']}: invalid coarse envelope")
        for arm in pcu["arms"]:
            if not _finite_oracle_path(arm.get("oracle_path")):
                errors.append(f"{pcu['id']}/{arm['id']}: missing finite oracle path")
            if arm.get("deadline_seconds") is None or float(arm["deadline_seconds"]) <= 0:
                errors.append(f"{pcu['id']}/{arm['id']}: invalid deadline")
            ratio = arm.get("route_length_ratio")
            if ratio is not None and not 0.8 <= float(ratio) <= 1.25:
                errors.append(f"{pcu['id']}/{arm['id']}: route-length ratio {ratio} outside [0.8, 1.25]")
            intended_ratio = arm.get("intended_to_legal_ratio")
            if intended_ratio is not None and not 0.8 <= float(intended_ratio) <= 1.25:
                errors.append(
                    f"{pcu['id']}/{arm['id']}: intended-to-legal ratio {intended_ratio} outside [0.8, 1.25]"
                )
            if pcu["layer"] == "C" and pcu.get("category") == "exact_parking":
                slot = str(arm.get("target", {}).get("slot_number", ""))
                if not (len(slot) == 4 and slot[0] in "ABCDEF" and slot[1:].isdigit()):
                    errors.append(f"{pcu['id']}/{arm['id']}: pseudo parking target {slot!r}")
        paired_paths = [arm.get("oracle_path") for arm in pcu["arms"]]
        if all(_finite_oracle_path(path) for path in paired_paths):
            first_start, second_start = paired_paths[0][0], paired_paths[1][0]
            position_error = math.dist(
                tuple(float(first_start[axis]) for axis in ("x", "y", "z")),
                tuple(float(second_start[axis]) for axis in ("x", "y", "z")),
            )
            yaw_error = abs((float(first_start["yaw_deg"]) - float(second_start["yaw_deg"]) + 180.0) % 360.0 - 180.0)
            if position_error > 1e-3 or yaw_error > 1e-3:
                errors.append(
                    f"{pcu['id']}: counterfactual arms do not share an exact start transform "
                    f"(position={position_error:.6f}m, yaw={yaw_error:.6f}deg)"
                )
    leaked = sorted(group for group, values in group_splits.items() if len(values) > 1)
    if leaked:
        errors.append(f"split leakage in topology/entity/program groups: {leaked}")
    split_leaks: dict[str, list[str]] = {}
    for kind, token_splits in split_keys.items():
        values = sorted(token for token, token_split in token_splits.items() if len(token_split) > 1)
        split_leaks[kind] = values
        if values:
            errors.append(f"dev/test {kind} leakage: {values}")

    families: dict[str, set[str]] = defaultdict(set)
    for pcu in pcus:
        families[pcu["layer"]].add(pcu.get("map_family", ""))
    if require_complete:
        if len(families["A"] - {"CONTROL"}) < 4: errors.append("layer A must cover at least four map families")
        if len(families["B"] - {"CONTROL"}) < 3: errors.append("layer B must cover at least three map families")
        if not {"M-X", "M-P", "M-G"}.issubset(families["C"]): errors.append("layer C must cover M-X/M-P/M-G")
        if not {"M-U", "M-P", "M-G"}.issubset(families["D"]): errors.append("layer D must cover M-U/M-P/M-G")
        ladders = defaultdict(set)
        for pcu in pcus:
            if pcu.get("ladder_id"): ladders[pcu["ladder_id"]].add(pcu["layer"])
        matched = {key: sorted(value) for key, value in ladders.items() if len(value) >= 3}
        if len(matched) < 8: errors.append(f"expected at least 8 matched ladders spanning >=3 layers, found {len(matched)}")
    else:
        matched = {}

    entries = len(pcus) * 2 * int(defaults["paraphrases_per_arm"])
    return {"valid": not errors, "errors": errors, "warnings": warnings, "pcu_count": len(pcus),
            "arm_count": sum(len(pcu["arms"]) for pcu in pcus), "xml_entry_count": entries,
            "layer_counts": dict(counts), "split_counts": dict(splits),
            "layer_split_counts": {f"{layer}/{split}": value for (layer, split), value in sorted(layer_splits.items())},
            "category_counts": {f"{layer}/{category}": value for (layer, category), value in sorted(categories.items())},
            "map_families": {layer: sorted(value) for layer, value in families.items()}, "matched_ladders": matched,
            "split_audit": {kind: {"unique_keys": len(values), "leaks": split_leaks[kind]}
                            for kind, values in split_keys.items()}}


def _weather(parent: ET.Element) -> None:
    values = {"route_percentage": "0", "cloudiness": "20", "precipitation": "0", "precipitation_deposits": "0",
              "wetness": "0", "wind_intensity": "10", "sun_azimuth_angle": "90", "sun_altitude_angle": "60", "fog_density": "0"}
    weathers = ET.SubElement(parent, "weathers")
    ET.SubElement(weathers, "weather", **values)


def _route_entries(pcus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries, route_id = [], 0
    for pcu in sorted(pcus, key=lambda value: value["id"]):
        for arm in pcu["arms"]:
            for paraphrase, instruction in enumerate(arm["instructions"]):
                entries.append({"route_entry_id": route_id, "pcu": pcu, "arm": arm,
                                "paraphrase": paraphrase, "instruction": instruction})
                route_id += 1
    return entries


def _route_xml(entries: list[dict[str, Any]], track: str) -> ET.ElementTree:
    root = ET.Element("routes")
    for entry in entries:
        pcu, arm = entry["pcu"], entry["arm"]
        route = ET.SubElement(root, "route", id=str(entry["route_entry_id"]), town=pcu["town"])
        # This element is delivered to the agent process.  Keep only the
        # instruction and track-wide controls: PCU/arm/paraphrase identities
        # live exclusively in sidecar.private.json.  In particular, do not
        # add a deterministic l2d_ref that an agent could reverse by
        # enumerating the small fixed suite.
        ET.SubElement(route, "simlingo_metadata", benchmark_type="language2drive_v1", track=track,
                      instruction=entry["instruction"], start_speed_kmh="10" if pcu["map_family"] == "M-G" else "20",
                      end_speed_kmh="0")
        ET.SubElement(route, "instruction").text = entry["instruction"]
        _weather(route)
        waypoints = ET.SubElement(route, "waypoints")
        points = pcu["coarse_envelope"] if track == "standardized" else arm["oracle_path"]
        for point in points:
            ET.SubElement(waypoints, "position", **{axis: str(point.get(axis, 0)) for axis in ("x", "y", "z")})
        ET.SubElement(route, "scenarios")
    return ET.ElementTree(root)


def _write_xml(path: Path, tree: ET.ElementTree) -> None:
    if hasattr(ET, "indent"): ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _run_record(entry: dict[str, Any], seed: int, track: str, paraphrase: int | None = None) -> dict[str, Any]:
    pcu, arm = entry["pcu"], entry["arm"]
    para = entry["paraphrase"] if paraphrase is None else paraphrase
    if track == "oracle":
        identity_values = {"pcu_id": pcu["id"], "arm_id": arm["id"], "paraphrase": para, "seed": seed, "track": track}
        run_id = f"oracle__{pcu['id']}__{arm['id']}__p{para}__s{seed}"
    else:
        identity = ArmIdentity(pcu["id"], arm["id"], para, seed, track)
        identity_values, run_id = identity.__dict__, identity.run_id
    return {**identity_values, "run_id": run_id, "route_entry_id": entry["route_entry_id"],
            "layer": pcu["layer"], "split": pcu["split"], "status": "expected"}


def _replace_directory_atomically(temporary: Path, output_dir: Path) -> None:
    backup = output_dir.with_name(f".{output_dir.name}.previous")
    if backup.exists(): shutil.rmtree(backup)
    if output_dir.exists(): os.replace(output_dir, backup)
    try:
        os.replace(temporary, output_dir)
    except BaseException:
        if backup.exists() and not output_dir.exists(): os.replace(backup, output_dir)
        raise
    if backup.exists(): shutil.rmtree(backup)


def compile_suite(source_dir: Path, output_dir: Path, defaults_path: Path, require_complete: bool = True) -> dict[str, Any]:
    defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    pcus = load_sources(source_dir)
    validation = validate_suite(pcus, defaults, require_complete=require_complete)
    if not validation["valid"]:
        raise SchemaError("suite qualification failed:\n- " + "\n- ".join(validation["errors"]))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="l2d-suite-", dir=output_dir.parent))
    try:
        entries = _route_entries(pcus)
        _write_xml(temporary / "routes_standardized.xml", _route_xml(entries, "standardized"))
        _write_xml(temporary / "routes_native.xml", _route_xml(entries, "native"))
        shutil.copy2(temporary / "routes_standardized.xml", temporary / "routes.xml")
        private_pcus, public_routes, private_routes = [], [], []
        source_hashes: dict[str, str] = {}
        for pcu in pcus:
            source = Path(pcu["_source"])
            source_hashes[str(source)] = sha256_file(source)
            private_pcus.append({key: value for key, value in pcu.items() if key not in {
                "_source", "_source_geometry_registry", "_source_parking_registry", "_source_files",
            }})
        # Close every private geometry/proof dependency before serializing the
        # public manifest and provenance.  Updating ``source_hashes`` after
        # manifest creation would leave otherwise valid parking proofs with a
        # stale, incomplete release provenance record.
        geometry_registry: dict[str, dict[str, Any]] = {}
        parking_registry: dict[str, Any] = {}
        registered_source_files: dict[str, str] = {}
        for pcu in pcus:
            for identifier, geometry in pcu.get("_source_geometry_registry", {}).items():
                if identifier in geometry_registry and geometry_registry[identifier] != geometry:
                    raise SchemaError(f"conflicting source geometry for {identifier}")
                geometry_registry[identifier] = geometry
            incoming_registry = pcu.get("_source_parking_registry", {})
            if incoming_registry:
                if parking_registry and incoming_registry != parking_registry:
                    raise SchemaError("conflicting parking manoeuvre registries")
                parking_registry = incoming_registry
            for path, digest in pcu.get("_source_files", {}).items():
                if path in registered_source_files and registered_source_files[path] != digest:
                    raise SchemaError(f"conflicting source file hash for {path}")
                source_path = Path(path)
                if not source_path.is_file() or sha256_file(source_path) != digest:
                    raise SchemaError(f"source file hash is stale or missing for {path}")
                registered_source_files[path] = digest
        source_hashes.update(registered_source_files)
        for entry in entries:
            pcu, arm = entry["pcu"], entry["arm"]
            public_routes.append({"route_entry_id": entry["route_entry_id"],
                                  "instruction": entry["instruction"], "track": "standardized"})
            private_routes.append({"route_entry_id": entry["route_entry_id"], "pcu_id": pcu["id"], "arm_id": arm["id"],
                                   "paraphrase": entry["paraphrase"], "instruction": entry["instruction"],
                                   "town": pcu["town"], "split": pcu["split"], "layer": pcu["layer"],
                                   "contract": arm["contract"], "opposite_contract": arm["opposite_contract"],
                                   "target": arm.get("target"), "actor_setup": arm.get("actor_setup", {}),
                                   "deadline_seconds": arm["deadline_seconds"], "oracle_path": arm["oracle_path"],
                                   **({"legal_speed_mps": arm["legal_speed_mps"],
                                      "legal_speed_proof": arm["legal_speed_proof"],
                                      "speed_feasibility_cap_proof": arm["speed_feasibility_cap_proof"]}
                                     if "legal_speed_mps" in arm else {}),
                                   **({"signal_free_corridor_proof": arm["signal_free_corridor_proof"]}
                                      if "signal_free_corridor_proof" in arm else {}),
                                   **({"intended_path_length_m": arm["intended_path_length_m"],
                                      "parking_v5_route_proof": arm["parking_v5_route_proof"]}
                                      if "parking_v5_route_proof" in arm else {})})
        semantic_blockers = [pcu["id"] for pcu in pcus if pcu.get("category") == "road_name"
                             and pcu.get("metadata", {}).get("semantic_review") != "confirmed"]
        manifest = {"schema_version": defaults["schema_version"], "release_status": "engineering/pre-release",
                    "layout": "fixed", "routes": public_routes, "validation": validation,
                    "config_sha256": stable_hash(defaults), "source_hashes": source_hashes,
                    "qualification": {"static_suite": "passed", "dynamic_oracle": "pending",
                                      "semantic_human_review": "pending" if semantic_blockers else "passed",
                                      "blocking_semantic_pcus": semantic_blockers,
                                      "minimal_human_review": "generated/review/MINIMAL_HUMAN_REVIEW.csv"}}
        atomic_json(temporary / "manifest.json", manifest)
        atomic_json(temporary / "public_context.json", {"routes": public_routes})
        atomic_json(temporary / "sidecar.private.json", {
            "schema_version": "language2drive.private_sidecar.v1",
            "pcus": private_pcus, "routes": private_routes,
            "geometry_registry": dict(sorted(geometry_registry.items())),
            "parking_manoeuvre_registry": parking_registry,
            "source_files": dict(sorted(registered_source_files.items())),
        })

        full_runs = [_run_record(entry, seed, "standardized") for entry in entries for seed in defaults["seeds"]]
        canonical = [entry for entry in entries if entry["paraphrase"] == 0]
        oracle_runs = [_run_record(entry, seed, "oracle", paraphrase=0) for entry in canonical for seed in defaults["seeds"]]
        pilot_pcus = {next(pcu["id"] for pcu in pcus if pcu["layer"] == layer and pcu["split"] == "dev") for layer in "ABCD"}
        pilot_entries = [entry for entry in entries if entry["pcu"]["id"] in pilot_pcus]
        pilot_runs = [_run_record(entry, seed, "standardized") for entry in pilot_entries for seed in defaults["seeds"]]
        smoke_ids = [f"{layer}-{number:03d}" for layer in "ABCD" for number in (1, 2)]
        smoke_entries = [next(entry for entry in canonical if entry["pcu"]["id"] == identifier)
                         for identifier in smoke_ids]
        native_runs = [_run_record(entry, 1, "native", paraphrase=0) for entry in smoke_entries]
        fdq_runs = [{"run_id": f"fdq__{identifier}__s{seed}", "ability": identifier, "seed": seed, "status": "expected"}
                     for identifier, _, _, _ in FDQ_ROUTE_BINDING_ROWS for seed in defaults["seeds"]]
        safety_categories = ("junction", "lane", "speed_constraint", "object_relation",
                             "passing_order", "exact_parking", "conditional", "until_invariant")
        safety_entries = []
        for category in safety_categories:
            candidate = next(entry for entry in canonical if entry["pcu"].get("category") == category)
            safety_entries.append(candidate)
        safety_runs = [_run_record(entry, 1, "standardized", paraphrase=0) for entry in safety_entries]
        tier_manifest = {}
        for tier, pcu_count in TIER_PCU_COUNTS.items():
            identifiers = tier_pcu_ids(pcus, tier)
            tier_runs = [run for run in full_runs if run["pcu_id"] in identifiers]
            expected_runs = pcu_count * 2 * defaults["paraphrases_per_arm"] * len(defaults["seeds"])
            if len(tier_runs) != expected_runs:
                raise SchemaError(f"{tier} tier run cardinality differs")
            tier_manifest[tier] = {
                "pcu_count": pcu_count,
                "pcu_ids": list(identifiers),
                "route_entries": pcu_count * 2 * defaults["paraphrases_per_arm"],
                "runs_per_seed": pcu_count * 2 * defaults["paraphrases_per_arm"],
                "full_seed_runs": expected_runs,
            }
            atomic_json(temporary / f"{tier}_expected_runs.json", {
                "expected_count": len(tier_runs), "runs": tier_runs,
            })
        atomic_json(temporary / "benchmark_tiers.json", {
            "schema_version": "language2drive.benchmark_tiers.v1",
            "tiers": tier_manifest,
        })
        for name, runs in (("expected_runs.json", full_runs), ("oracle_expected_runs.json", oracle_runs),
                           ("pilot_expected_runs.json", pilot_runs), ("native_smoke_expected_runs.json", native_runs),
                           ("safety_priority_expected_runs.json", safety_runs), ("fdq_expected_runs.json", fdq_runs)):
            atomic_json(temporary / name, {"expected_count": len(runs), "runs": runs})
        atomic_json(temporary / "provenance.json", {"generator_version": "0.2.0", "config_sha256": stable_hash(defaults),
                                                    "source_hashes": source_hashes, "fixed_layout_only": True,
                                                    "train_overlap": "unknown"})
        _replace_directory_atomically(temporary, output_dir)
    except BaseException:
        if temporary.exists(): shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**validation, "expected_run_count": len(full_runs), "oracle_witness_expected": len(oracle_runs),
            "pilot_expected": len(pilot_runs), "native_smoke_expected": len(native_runs),
            "safety_priority_expected": len(safety_runs), "fdq_expected": len(fdq_runs)}
