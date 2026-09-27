from copy import deepcopy

import pytest

from language2drive_realsim.alpasim_bridge import normalize_simulation_result
from language2drive_realsim.events import validate_trace
from language2drive_realsim.render_validity import NATIVE_MODE, native_render_protocol
from language2drive_realsim.schema import SchemaError, validate_pcu
from language2drive_realsim.verifier import _unsafe_result
from .factories import make_pcu
from .test_alpasim_bridge import Result, Trajectory, Metric


def annotations():
    return {"reconstruction_support_polygon": None, "render_validity_protocol": {
        "mode": NATIVE_MODE, "native_metric": "img_is_black", "maximum_black_samples": 0,
        "frozen_before_model_evaluation": True, "pilot_evidence_refs": ["pilot/seed1701"]}}


def native_rows(black=(0, 0, 1)):
    result = Result()
    result.ego_recorded_ground_truth_trajectory = Trajectory()
    metrics = [Metric(name, Result.timestamps_us, list(black) if name == "img_is_black" else [0, 0, 0], [True]*3)
               for name in ("collision_any", "offroad", "img_is_black")]
    return normalize_simulation_result(result, metrics, annotations())


def test_explicit_native_protocol_not_an_invented_support_polygon():
    rows = native_rows()
    validate_trace(rows)
    assert [r["render_valid"] for r in rows] == [True, True, False]
    assert all("reconstruction_support_valid" not in r for r in rows)
    assert all(r["reconstruction_deviation_m"] is None for r in rows)
    assert all(r["recorded_path_deviation_m"] == 0 for r in rows)
    failure = _unsafe_result(rows, 1.7)
    assert failure.failure == "F3_UNSAFE_FOLLOW" and failure.secondary_tags == ["native_black_image"]
    assert _unsafe_result(native_rows((1, 0, 0)), 1.7).failure == "F0_PREINTERVENTION"


def test_native_mode_keeps_all_private_review_gates():
    pcu = make_pcu(1, "A5_lane_change_keep")
    pcu["private_annotations"].update(annotations())
    validate_pcu(pcu)
    for field in ("render_reviewed", "legality_reviewed", "balance_reviewed"):
        changed = deepcopy(pcu)
        changed["private_annotations"][field] = False
        with pytest.raises(SchemaError, match="reviews must all pass"):
            validate_pcu(changed)
    del pcu["private_annotations"]["render_validity_protocol"]
    with pytest.raises(SchemaError):
        validate_pcu(pcu)


@pytest.mark.parametrize("field,value", [("mode", "unknown"), ("maximum_black_samples", 1),
    ("maximum_black_samples", False), ("frozen_before_model_evaluation", False),
    ("native_metric", "invented"), ("pilot_evidence_refs", []), ("pilot_evidence_refs", ["x", "x"])])
def test_native_protocol_cannot_relax_threshold_or_drop_pilot_basis(field, value):
    value_annotations = annotations()
    value_annotations["render_validity_protocol"][field] = value
    with pytest.raises(ValueError):
        native_render_protocol(value_annotations)


def test_native_mode_cannot_override_known_polygon_or_omit_deviation():
    value = annotations()
    value["reconstruction_support_polygon"] = [[0, 0], [1, 0], [1, 1]]
    with pytest.raises(ValueError, match="override"):
        native_render_protocol(value)
    metrics = [Metric(name, Result.timestamps_us, [0]*3, [True]*3)
               for name in ("collision_any", "offroad", "img_is_black")]
    with pytest.raises(ValueError, match="recorded-path deviation"):
        normalize_simulation_result(Result(), metrics, annotations())


@pytest.mark.parametrize("field,value", [("image_black", None), ("render_valid", True),
    ("reconstruction_support_valid", True), ("recorded_path_deviation_m", None)])
def test_native_trace_rejects_fabricated_or_missing_signals(field, value):
    rows = native_rows()
    rows[-1][field] = value
    with pytest.raises(ValueError):
        validate_trace(rows)
