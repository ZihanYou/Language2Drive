"""Small runtime compatibility helpers for the declared Python >=3.8 floor."""

from itertools import zip_longest


_MISSING = object()


def strict_zip(*iterables):
    """Provide ``zip(..., strict=True)`` semantics on Python 3.8 and newer.

    The check remains lazy and fail-closed, matching the behavior relied on by
    the geometry and evidence builders without requiring Python 3.10 syntax.
    """

    for values in zip_longest(*iterables, fillvalue=_MISSING):
        if any(value is _MISSING for value in values):
            raise ValueError("strict_zip arguments have different lengths")
        yield values
