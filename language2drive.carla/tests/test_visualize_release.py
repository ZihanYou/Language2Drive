from __future__ import annotations

import hashlib
import json
from pathlib import Path

from language2drive.visualize import visualize_suite


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_visualize_replaces_stale_snapshot_and_records_current_sources(
    tmp_path: Path, monkeypatch,
) -> None:
    project = tmp_path / "project"
    suite = project / "build/suite"
    manifest = suite / "manifest.json"
    _write_json(manifest, {"schema_version": "test"})
    pcu = {
        "id": "A-001", "layer": "A", "town": "Town01", "category": "junction",
        "coarse_envelope": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
        "source_refs": ["unit-test"],
        "arms": [
            {"id": "left", "instructions": ["Turn left."], "oracle_path": []},
            {"id": "right", "instructions": ["Turn right."], "oracle_path": []},
        ],
    }
    sidecar = suite / "sidecar.private.json"
    _write_json(sidecar, {"pcus": [pcu]})
    _write_json(project / "assets/review/semantic_entities.geojson", {"features": []})
    _write_json(project / "generated/maps/parking_spaces.geojson", {"features": []})
    _write_json(
        project / "generated/maps/lane_graphs/Town01.lanes.geojson",
        {"features": [{"geometry": {"coordinates": [[0, 0], [1, 0]]}}]},
    )
    output = project / "generated/visualizations"
    stale = output / "A-001/arm_stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    def fake_render(path: Path, *_args, **_kwargs) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"rendered:{path.name}".encode())

    monkeypatch.setattr("language2drive.visualize._render", fake_render)
    assert visualize_suite(manifest, output) == 1

    actual = {path.relative_to(output) for path in output.rglob("*") if path.is_file()}
    assert actual == {
        Path("A-001/composite.png"), Path("A-001/arm_left.png"),
        Path("A-001/arm_right.png"), Path("index.html"), Path("MANIFEST.json"),
    }
    visual_manifest = json.loads((output / "MANIFEST.json").read_text())
    assert visual_manifest["pcu_count"] == 1
    assert visual_manifest["arm_count"] == 2
    assert visual_manifest["png_count"] == 3
    assert visual_manifest["source_hashes"]["build/suite/sidecar.private.json"] == (
        hashlib.sha256(sidecar.read_bytes()).hexdigest()
    )

