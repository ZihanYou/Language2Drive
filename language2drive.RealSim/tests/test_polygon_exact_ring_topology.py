"""Source topology must not inherit a physical epsilon from gate events."""
from copy import deepcopy
import pytest
from language2drive_realsim.schema import SchemaError, _validate_polygon, _ring_edges_intersect


def test_retained_ba590_original_intersection_short_edge_is_not_a_crossing():
    ring=[[96.12580871582031,10.231142044067383],[99.8680419921875,10.985823631286621],
          [102.30911683956552,11.488890982352848],[102.93256381290692,8.435976440039335],
          [100.48544311523438,7.9566826820373535],[96.6859130859375,7.237374782562256],
          [91.08052825927734,6.208524227142334],[85.03394317626953,5.154303550720215],
          [85.03394317626932,5.154303550721681],[84.5999263962489,8.2244363998794],
          [90.57141876220703,9.222240447998047],[96.12580871582031,10.231142044067383]]
    geometry=dict(type="Polygon",coordinates=[ring]);before=deepcopy(geometry)
    _validate_polygon(geometry,"source")
    assert geometry==before and len(geometry['coordinates'][0])==12


@pytest.mark.parametrize("scale",[1.,1e-3,1e-6])
def test_true_crossings_reject_even_below_runtime_gate_epsilon(scale):
    ring=[[0,0],[2*scale,2*scale],[0,2*scale],[scale,0],[0,0]]
    with pytest.raises(SchemaError,match="self-intersects"):
        _validate_polygon(dict(type="Polygon",coordinates=[ring]),"source")


@pytest.mark.parametrize("a,b,c,d,expected",[
    ([0,0],[1,0],[1+1e-12,0],[2,0],False),
    ([0,0],[1,0],[1,0],[2,0],True),
    ([0,0],[1,0],[.5,0],[2,0],True),
    ([0,0],[1,1],[0,1],[1,0],True),
    ([0,0],[1,1],[0,1e-12],[1,1+1e-12],False),
])
def test_exact_predicate_handles_separation_touching_overlap_and_crossing(a,b,c,d,expected):
    assert _ring_edges_intersect(a,b,c,d) is expected
    assert _ring_edges_intersect(d,c,b,a) is expected
