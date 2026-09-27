import pytest

from language2drive_realsim.compat import strict_zip


def test_strict_zip_equal_lengths():
    assert list(strict_zip((1, 2), ("a", "b"))) == [(1, "a"), (2, "b")]


@pytest.mark.parametrize("values", [((1,), (2, 3)), ((1, 2), (3,))])
def test_strict_zip_rejects_length_mismatch(values):
    with pytest.raises(ValueError, match="different lengths"):
        list(strict_zip(*values))


def test_strict_zip_is_lazy():
    values = strict_zip(iter((1, 2)), iter((3,)))
    assert next(values) == (1, 3)
    with pytest.raises(ValueError, match="different lengths"):
        next(values)


def test_strict_zip_accepts_no_iterables():
    assert list(strict_zip()) == []
