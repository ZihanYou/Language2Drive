"""No-clobber publication tests; no external filesystem or runtime required."""
import os

import pytest

from language2drive_realsim.io import atomic_json, load_json


def test_exclusive_publish_is_complete(tmp_path):
    target = tmp_path / "report.json"
    atomic_json(target, {"ready": False}, overwrite=False)
    assert load_json(target) == {"ready": False}
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("kind", ["file", "dangling_symlink"])
def test_exclusive_publish_preserves_existing_path(tmp_path, kind):
    target = tmp_path / "report.json"
    if kind == "file":
        atomic_json(target, {"original": True})
    else:
        target.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        atomic_json(target, {"replacement": True}, overwrite=False)
    if kind == "file":
        assert load_json(target) == {"original": True}
    else:
        assert target.is_symlink()
    assert list(tmp_path.iterdir()) == [target]


def test_concurrent_winner_is_not_replaced(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    link = os.link

    def concurrent_link(source, destination):
        atomic_json(target, {"other_writer": True})
        return link(source, destination)

    monkeypatch.setattr(os, "link", concurrent_link)
    with pytest.raises(FileExistsError):
        atomic_json(target, {"ours": True}, overwrite=False)
    assert load_json(target) == {"other_writer": True}
    assert list(tmp_path.iterdir()) == [target]
