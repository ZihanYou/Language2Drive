from copy import deepcopy

import pytest
from shapely.geometry import Point, shape

from language2drive_realsim.events import point_in_polygon, point_to_polygon_distance, derive_events
from language2drive_realsim.schema import SchemaError, _validate_polygon, validate_source
from language2drive_realsim.compiler import compile_source
from .factories import make_release_source, basic_trace


def ring(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def geometry():
    return {"type": "MultiPolygon", "coordinates": [
        [ring(0, 0, 4, 4), ring(1, 1, 2, 2)], [ring(4.01, 0, 6, 4)]]}


@pytest.mark.parametrize("point,inside,distance", [
    ([.5, .5], True, 0), ([1.5, 1.5], False, .5),
    ([1, 1.5], True, 0), ([4.005, 2], False, .005),
    ([4, 2], True, 0), ([5, 2], True, 0), ([7, 2], False, 1),
])
def test_holes_gaps_and_boundaries_preserved(point, inside, distance):
    g = geometry()
    assert point_in_polygon(point, g) is inside
    assert point_to_polygon_distance(point, g) == pytest.approx(distance)


def test_native_membership_and_distance_match_independent_geometry_engine():
    g = geometry()
    expected = shape(g)
    for x in [-1, 0, .5, 1, 1.5, 2, 3, 4, 4.001, 4.005, 4.009, 4.01, 5, 6, 7]:
        for y in [-1, 0, .5, 1, 1.5, 2, 3, 4, 5]:
            p = Point(x, y)
            assert point_in_polygon([x, y], g) == expected.covers(p)
            assert point_to_polygon_distance([x, y], g) == pytest.approx(expected.distance(p))


def test_polygon_with_hole_and_legacy_ring_share_boundary_convention():
    g = {"type": "Polygon", "coordinates": geometry()["coordinates"][0]}
    _validate_polygon(g, "test")
    assert not point_in_polygon([1.5, 1.5], g)
    assert point_in_polygon([1.5, 1.5], g["coordinates"][0])
    assert point_to_polygon_distance([1.5, 1.5], g) == .5


@pytest.mark.parametrize("mutation", ["empty", "kind", "extra", "open", "nan", "bool", "duplicate", "zero_area",
                                    "crossing", "hole_outside", "hole_touch", "holes_overlap", "ring_object"])
def test_schema_rejects_malformed_or_invalid_polygon_rings(mutation):
    g = geometry()
    if mutation == "empty": g["coordinates"] = []
    elif mutation == "kind": g["type"] = "LineString"
    elif mutation == "extra": g["buffer"] = 1
    elif mutation == "open": g["coordinates"][0][0].pop()
    elif mutation == "nan": g["coordinates"][0][0][1][0] = float("nan")
    elif mutation == "bool": g["coordinates"][0][0][1][0] = True
    elif mutation == "duplicate": g["coordinates"][0][0].insert(1, [0, 0])
    elif mutation == "zero_area": g["coordinates"][0][0] = [[0, 0], [1, 0], [2, 0], [0, 0]]
    elif mutation == "crossing": g["coordinates"][0][0] = [[0, 0], [4, 4], [0, 4], [3, 0], [0, 0]]
    elif mutation == "hole_outside": g["coordinates"][0][1] = ring(5, 5, 6, 6)
    elif mutation == "hole_touch": g["coordinates"][0][1] = ring(0, 1, 2, 2)
    elif mutation == "holes_overlap": g["coordinates"][0].append(ring(1.5, 1.5, 3, 3))
    elif mutation == "ring_object": g["coordinates"][0][0] = {"type": "Polygon", "coordinates": [ring(0, 0, 4, 4)]}
    with pytest.raises(SchemaError):
        _validate_polygon(g, "test")


def test_source_compile_keeps_multi_geometry_private_and_unchanged():
    source = make_release_source()
    pcu = source["pcus"][0]
    for name in ("scopes", "regions", "lanes"):
        pcu["private_annotations"][name]["geometry-test"] = geometry()
    pcu["private_annotations"]["reconstruction_support_polygon"] = geometry()
    before = deepcopy(source)
    validate_source(source, require_exact_release=True)
    public, sidecar, audit = compile_source(source)
    assert source == before and audit["pass"]
    assert "MultiPolygon" not in str(public)
    assert any(s["annotations"]["reconstruction_support_polygon"] == geometry() for s in sidecar["sessions"].values())


def test_region_event_does_not_enter_a_hole():
    rows = basic_trace()[:3]
    for r, p in zip(rows, ([.5, .5], [1.5, 1.5], [5, 2])):
        r["position"] = list(p)
    events = derive_events(rows, {"regions": {"region-a": geometry()}})
    assert [(e.kind, e.time_s) for e in events if e.subject == "region-a"] == [
        ("EnterRegion", 0), ("ExitRegion", 1), ("EnterRegion", 2)]


@pytest.mark.parametrize("kind", ["gap", "hole"])
def test_bridge_scores_unsupported_gap_or_hole_without_infrastructure_retry(kind):
    from language2drive_realsim.alpasim_bridge import normalize_simulation_result
    from language2drive_realsim.verifier import verify_trace
    from .test_alpasim_bridge import Result, Metric
    from .factories import make_pcu

    if kind == "gap":
        support = {"type": "MultiPolygon", "coordinates": [
            [ring(-1, -1, .8, 1)], [ring(1.2, -1, 3, 1)]]}
    else:
        support = {"type": "Polygon", "coordinates": [
            ring(-1, -1, 3, 1), ring(.8, -.5, 1.2, .5)]}
    metrics = [Metric(n, Result.timestamps_us, [0] * 3, [True] * 3)
               for n in ("collision_any", "offroad", "img_is_black")]
    annotations = {"reconstruction_support_polygon": support,
                   "scopes": {"scope-test": support}}
    rows = normalize_simulation_result(Result(), metrics, annotations)
    assert [r["reconstruction_support_valid"] for r in rows] == [True, False, True]
    assert rows[1]["reconstruction_deviation_m"] == pytest.approx(.2)
    assert rows[1]["scopes"] == [] and rows[1]["image_black"] is False
    result = verify_trace(make_pcu(1, "A1_intersection_left_right"), "R0", rows)
    assert result.failure == "F3_UNSAFE_FOLLOW"
    assert result.secondary_tags == ["outside_reconstruction_support"]
