import pytest

from language2drive_realsim.io import atomic_json, canonical_json, load_json


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_json_never_overwrites_previous_evidence(tmp_path, value):
    path = tmp_path / "evidence.json"
    atomic_json(path, {"previous": True})
    with pytest.raises(ValueError):
        canonical_json({"metric": value})
    with pytest.raises(ValueError):
        atomic_json(path, {"metric": value})
    assert load_json(path) == {"previous": True}
    assert list(tmp_path.iterdir()) == [path]
