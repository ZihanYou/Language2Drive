from __future__ import annotations

import html
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .io import sha256_file


def build_review_page(registry: Path, review_dir: Path, output: Path) -> None:
    maps = json.loads(registry.read_text(encoding="utf-8"))["maps"]
    rows = "".join(
        f"<tr><td>{html.escape(m['town'])}</td><td>{m['family']}</td><td>{m['role']}</td>"
        f"<td>{m['lane_count']}</td><td>{m['lane_sample_count']}</td>"
        f"<td>{m['parking_space_objects']} / {m['parking_space_unique']}</td>"
        f"<td>{'fallback' if m['topology_fallback'] else 'CARLA topology'}</td></tr>" for m in maps
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""<!doctype html><meta charset=utf-8><title>L2D asset review</title>
<style>body{{font:14px system-ui;margin:2rem;color:#20242a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbc;padding:.45rem;text-align:left}}th{{background:#edf2f7}}.warn{{background:#fff4df;padding:1rem;border-left:4px solid #d97706}}</style>
<h1>Language2Drive semantic asset review</h1>
<p class=warn>Existing labels were projected automatically. Non-blocking geometry fixes are the rows marked <code>needs_fix</code> in asset_review.csv (port V13/V14, excluded from core). Publication evidence review is listed separately in <code>MINIMAL_HUMAN_REVIEW.md</code>.</p>
<table><tr><th>Town</th><th>Family</th><th>Role</th><th>Lanes</th><th>Samples</th><th>Parking raw/unique</th><th>Connectivity</th></tr>{rows}</table>
<p>Editable review files: {html.escape(str(review_dir))}</p>""", encoding="utf-8")


def _iter_xy(value: Any) -> Iterable[tuple[float, float]]:
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            yield float(value["x"]), float(value["y"])
        for child in value.values(): yield from _iter_xy(child)
    elif isinstance(value, list):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            yield float(value[0]), float(value[1])
        else:
            for child in value: yield from _iter_xy(child)


def _plot_polygon(axis: Any, polygon: Any, color: str, label: str) -> None:
    if not isinstance(polygon, list) or len(polygon) < 3: return
    xs, ys = zip(*[(float(point[0]), float(point[1])) for point in polygon])
    axis.fill(xs, ys, color=color, alpha=.18, label=label)
    axis.plot(xs, ys, color=color, linewidth=1.6)


def _target(axis: Any, target: dict[str, Any], color: str, label: str) -> None:
    if not isinstance(target, dict): return
    if target.get("polygon"): _plot_polygon(axis, target["polygon"], color, label)
    center = target.get("center")
    if isinstance(center, dict) and "x" in center:
        radius = float(target.get("radius_m", 3))
        from matplotlib.patches import Circle
        axis.add_patch(Circle((center["x"], center["y"]), radius, edgecolor=color, facecolor=color, alpha=.18, label=label))
    if target.get("heading_deg") is not None:
        points = list(_iter_xy(target.get("polygon", [])))
        if points:
            cx, cy = sum(x for x, _ in points)/len(points), sum(y for _, y in points)/len(points)
            yaw = math.radians(float(target["heading_deg"]))
            axis.arrow(cx, cy, 3*math.cos(yaw), 3*math.sin(yaw), color=color, width=.08, head_width=.8)
    if target.get("type") == "if_else":
        _target(axis, target.get("then", {}), color, label+" then")
        _target(axis, target.get("else", {}), color, label+" else")


def _render(path: Path, pcu: dict[str, Any], arms: list[dict[str, Any]], lanes: list[list[list[float]]],
            entities: list[dict[str, Any]], parking_spaces: list[dict[str, Any]] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#d9485f", "#168b6a")
    focus = list(_iter_xy(pcu["coarse_envelope"]))
    for arm in arms:
        focus.extend(_iter_xy(arm.get("oracle_path", [])))
        focus.extend(_iter_xy(arm.get("target", {})))
    if not focus: return
    low_x, high_x = min(x for x, _ in focus), max(x for x, _ in focus)
    low_y, high_y = min(y for _, y in focus), max(y for _, y in focus)
    span = max(high_x-low_x, high_y-low_y, 20); margin = max(12, span*.12)
    low_x, high_x, low_y, high_y = low_x-margin, high_x+margin, low_y-margin, high_y+margin

    fig, axis = plt.subplots(figsize=(7.2, 7.2), dpi=120)
    for coordinates in lanes:
        if not coordinates: continue
        xs, ys = zip(*[(p[0], p[1]) for p in coordinates])
        if max(xs) < low_x or min(xs) > high_x or max(ys) < low_y or min(ys) > high_y: continue
        axis.plot(xs, ys, color="#bcc5ce", linewidth=.55, alpha=.8, zorder=1)
    for feature in entities:
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if len(coordinates) < 2 or not (low_x <= coordinates[0] <= high_x and low_y <= coordinates[1] <= high_y): continue
        status = feature.get("properties", {}).get("status")
        axis.scatter([coordinates[0]], [coordinates[1]], s=12, marker="x", color="#e59f00" if status == "needs_fix" else "#56616c", zorder=2)
    for feature in parking_spaces or []:
        polygon = feature.get("geometry", {}).get("coordinates", [[]])[0]
        if not polygon: continue
        xs,ys=zip(*[(float(point[0]),float(point[1])) for point in polygon])
        if max(xs)<low_x or min(xs)>high_x or max(ys)<low_y or min(ys)>high_y:continue
        axis.plot(xs,ys,color="#9aa7b5",linewidth=.45,alpha=.65,zorder=1)
    envelope = list(_iter_xy(pcu["coarse_envelope"]))
    axis.plot([x for x, _ in envelope], [y for _, y in envelope], color="#2463eb", linewidth=2.2,
              linestyle="--", label="shared coarse envelope", zorder=4)
    for arm, color in zip(arms, colors):
        route = list(_iter_xy(arm.get("oracle_path", [])))
        if route:
            axis.plot([x for x, _ in route], [y for _, y in route], color=color, linewidth=1.7,
                      label=f"{arm['id']} legal oracle", zorder=3)
        _target(axis, arm.get("target", {}), color, f"{arm['id']} target")
    axis.set(xlim=(low_x, high_x), ylim=(low_y, high_y), aspect="equal", xlabel="CARLA x (m)", ylabel="CARLA y (m)")
    axis.grid(alpha=.15); axis.legend(fontsize=7, loc="best")
    source = ", ".join(pcu.get("source_refs", [])[:2])
    axis.set_title(f"{pcu['id']} · Layer {pcu['layer']} · {pcu['town']}\n{source}", fontsize=9)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path); plt.close(fig)


def visualize_suite(manifest: Path, output: Path) -> int:
    project = manifest.parents[2]
    sidecar_path = manifest.parent / "sidecar.private.json"
    entity_path = project / "assets/review/semantic_entities.geojson"
    parking_path = project / "generated/maps/parking_spaces.geojson"
    private = json.loads(sidecar_path.read_text(encoding="utf-8"))["pcus"]
    entity_collection = json.loads(entity_path.read_text(encoding="utf-8"))
    entities_by_town: dict[str, list[dict[str, Any]]] = {}
    for feature in entity_collection["features"]:
        entities_by_town.setdefault(feature["properties"]["town"], []).append(feature)
    parking_by_town: dict[str,list[dict[str,Any]]]={}
    if parking_path.is_file():
        for feature in json.loads(parking_path.read_text(encoding="utf-8"))["features"]:
            parking_by_town.setdefault(feature["properties"]["town"],[]).append(feature)
    lanes_by_town: dict[str, list[list[list[float]]]] = {}
    lane_paths: dict[str, Path] = {}
    for town in {pcu["town"] for pcu in private}:
        lane_path = project / f"generated/maps/lane_graphs/{town}.lanes.geojson"
        lane_paths[town] = lane_path
        geojson = json.loads(lane_path.read_text(encoding="utf-8"))
        lanes_by_town[town] = [feature["geometry"]["coordinates"] for feature in geojson["features"]]

    # Render into a sibling staging directory and swap only after every
    # expected image exists.  In-place rendering used to leave arm PNGs from
    # older suite snapshots behind, which made a visual audit ambiguous.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    backup = output.parent / f".{output.name}.previous"
    try:
        links = []
        artifacts: list[dict[str, Any]] = []
        for pcu in private:
            if len(pcu.get("arms", [])) != 2:
                raise ValueError(f"{pcu.get('id')}: visualization requires exactly two arms")
            directory = staging / pcu["id"]
            composite = directory / "composite.png"
            _render(composite, pcu, pcu["arms"], lanes_by_town[pcu["town"]], entities_by_town.get(pcu["town"], []),parking_by_town.get(pcu["town"],[]))
            expected = [composite]
            for arm in pcu["arms"]:
                arm_path = directory / f"arm_{arm['id']}.png"
                _render(arm_path, pcu, [arm], lanes_by_town[pcu["town"]], entities_by_town.get(pcu["town"], []),parking_by_town.get(pcu["town"],[]))
                expected.append(arm_path)
            missing = [str(path.relative_to(staging)) for path in expected if not path.is_file()]
            if missing:
                raise RuntimeError(f"{pcu['id']}: visualization outputs missing: {missing}")
            artifacts.append({
                "pcu_id": pcu["id"],
                "arm_ids": [arm["id"] for arm in pcu["arms"]],
                "files": [
                    {
                        "path": str(path.relative_to(staging)),
                        "sha256": sha256_file(path),
                    }
                    for path in expected
                ],
            })
            links.append(f'<article><h2>{pcu["id"]} · {pcu["category"]}</h2><img loading="lazy" src="{pcu["id"]}/composite.png"><p>{html.escape(pcu["arms"][0]["instructions"][0])} ↔ {html.escape(pcu["arms"][1]["instructions"][0])}</p></article>')
        index = staging / "index.html"
        index.write_text("<!doctype html><meta charset=utf-8><title>L2D visual audit</title><style>body{font:14px system-ui;max-width:1100px;margin:auto}article{border-bottom:1px solid #ddd;padding:1rem}img{max-width:720px;width:100%}</style><h1>Language2Drive v1 joint map audit</h1>"+"".join(links), encoding="utf-8")
        source_hashes = {
            "build/suite/manifest.json": sha256_file(manifest),
            "build/suite/sidecar.private.json": sha256_file(sidecar_path),
            "assets/review/semantic_entities.geojson": sha256_file(entity_path),
            **(
                {"generated/maps/parking_spaces.geojson": sha256_file(parking_path)}
                if parking_path.is_file() else {}
            ),
            **{
                str(path.relative_to(project)): sha256_file(path)
                for _, path in sorted(lane_paths.items())
            },
        }
        visualization_manifest = {
            "schema_version": "language2drive.visualizations.v1",
            "pcu_count": len(private),
            "arm_count": sum(len(pcu["arms"]) for pcu in private),
            "png_count": sum(len(item["files"]) for item in artifacts),
            "source_hashes": source_hashes,
            "artifacts": artifacts,
            "index": {"path": "index.html", "sha256": sha256_file(index)},
        }
        (staging / "MANIFEST.json").write_text(
            json.dumps(visualization_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

        if backup.exists():
            shutil.rmtree(backup)
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            if backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return len(private)
