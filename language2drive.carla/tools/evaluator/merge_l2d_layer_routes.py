#!/usr/bin/env python3
"""
Merge per-route Language2Drive XML files into a single routes file (Plan A).

Each source XML is expected to contain exactly one <route>. Route ids inside the
source files may collide (e.g. weather_code reused across files). This script
assigns sequential unique ids (0, 1, 2, ...) and writes a manifest mapping merged
ids back to the original files for downstream metrics and analysis.
"""

from __future__ import annotations

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _collect_route_xmls(input_dir: Path) -> List[Path]:
    return sorted(input_dir.rglob("*.xml"))


def _route_metadata(route_elem: ET.Element, source_rel: str, merged_id: int) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "merged_id": str(merged_id),
        "route_name": f"RouteScenario_{merged_id}",
        "source_xml": source_rel,
        "original_id": route_elem.get("id"),
        "town": route_elem.get("town"),
    }

    simlingo_meta = route_elem.find("simlingo_metadata")
    if simlingo_meta is not None:
        meta["simlingo_metadata"] = dict(simlingo_meta.attrib)

    instruction = route_elem.find("instruction")
    if instruction is not None and instruction.text:
        meta["instruction"] = instruction.text.strip()

    return meta


def merge_layer_routes(
    input_dir: Path,
    output_xml: Path,
    manifest_path: Path,
    layer_name: str = "C",
) -> Dict[str, Any]:
    xml_files = _collect_route_xmls(input_dir)
    if not xml_files:
        raise FileNotFoundError(f"No XML files found under {input_dir}")

    root = ET.Element("routes")
    manifest_routes: List[Dict[str, Any]] = []

    for merged_id, xml_path in enumerate(xml_files):
        rel_path = str(xml_path.relative_to(input_dir))
        tree = ET.parse(xml_path)
        routes = list(tree.getroot().findall("route"))
        if len(routes) != 1:
            raise ValueError(
                f"Expected exactly one <route> in {rel_path}, found {len(routes)}"
            )

        route = copy.deepcopy(routes[0])
        route.set("id", str(merged_id))
        route.set("merged_source", rel_path)
        root.append(route)

        manifest_routes.append(_route_metadata(route, rel_path, merged_id))

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    merged_tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(merged_tree, space="   ")
    merged_tree.write(output_xml, encoding="utf-8", xml_declaration=True)

    manifest = {
        "layer": layer_name,
        "input_dir": str(input_dir.resolve()),
        "merged_xml": str(output_xml.resolve()),
        "total_routes": len(manifest_routes),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routes": manifest_routes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Merge Language2Drive layer route XMLs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data_l2d" / "layer_C",
        help="Directory containing per-route XML files",
    )
    parser.add_argument(
        "--output-xml",
        type=Path,
        default=repo_root / "data_l2d" / "layer_C_merged.xml",
        help="Merged routes XML output path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "data_l2d" / "layer_C_manifest.json",
        help="Manifest JSON mapping merged ids to source XML files",
    )
    parser.add_argument(
        "--layer-name",
        type=str,
        default="C",
        help="Layer label stored in the manifest",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing merged XML and manifest",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    if (
        not args.force
        and args.output_xml.exists()
        and args.manifest.exists()
    ):
        print(f"Merged routes already exist:\n  {args.output_xml}\n  {args.manifest}")
        print("Use --force to regenerate.")
        return

    manifest = merge_layer_routes(
        input_dir=args.input_dir,
        output_xml=args.output_xml,
        manifest_path=args.manifest,
        layer_name=args.layer_name,
    )


if __name__ == "__main__":
    main()
