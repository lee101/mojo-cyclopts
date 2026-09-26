"""Parity tests for the slice kernels.

``selects_nothing`` is checked against
``cyclopts.validators._slice._selects_nothing`` over an exhaustive grid of
concrete bounds, because the interesting behaviour is entirely in the corner
cases: unbounded ends, mixed sign frames, negative steps. ``slice_indices`` is
checked against CPython's ``slice.indices`` over the same grid plus a range of
sequence lengths.
"""

import itertools

import pytest
from cyclopts.validators import Slice as UpstreamSlice
from cyclopts.validators._slice import _selects_nothing as upstream_selects_nothing

from mojo_cyclopts import Slice, selects_nothing, slice_indices

BOUNDS = [None, -7, -3, -1, 0, 1, 3, 7, 12]
STEPS = [None, -3, -1, 1, 2, 5]


def _grid():
    for start, stop, step in itertools.product(BOUNDS, BOUNDS, STEPS):
        yield slice(start, stop, step)


def test_selects_nothing_exhaustive():
    n = 0
    for s in _grid():
        if s.step == 0:
            continue
        assert selects_nothing(s) == upstream_selects_nothing(s), s
        n += 1
    assert n == 486  # the full grid, so a silently skipped case shows up


def test_slice_validator_parity():
    theirs = UpstreamSlice(allow_empty=False)
    mine = Slice(allow_empty=False)
    for s in _grid():
        if s.step == 0:
            continue

        def run(fn):
            try:
                fn(slice, s)
            except ValueError as exc:
                return str(exc)
            return None

        assert run(mine) == run(theirs), s


def test_slice_validator_allow_empty_default():
    """With allow_empty=True nothing is ever rejected, matching upstream."""
    theirs, mine = UpstreamSlice(), Slice()
    for s in _grid():
        if s.step == 0:
            continue
        mine(slice, s)
        theirs(slice, s)


def test_slice_validator_ignores_non_slices():
    theirs, mine = UpstreamSlice(allow_empty=False), Slice(allow_empty=False)
    for v in [1, None, 3.5, [1, 2]]:
        mine(type(v), v)
        theirs(type(v), v)


def test_slice_validator_on_containers():
    theirs, mine = UpstreamSlice(allow_empty=False), Slice(allow_empty=False)
    good = [slice(0, 3), slice(1, None)]
    bad = [slice(0, 3), slice(2, 1)]
    mine(slice, good)
    theirs(slice, good)
    with pytest.raises(ValueError):
        theirs(slice, bad)
    with pytest.raises(ValueError):
        mine(slice, bad)


def test_zero_step_is_not_reported_as_empty():
    """A zero step raises on application, which is not this validator's
    business, so it is never reported as selecting nothing."""
    assert selects_nothing(slice(0, 0, 0)) is False
    assert upstream_selects_nothing(slice(0, 0, 0)) is False


def test_unbounded_negative_step_cannot_be_proved_empty():
    assert selects_nothing(slice(None, None, -1)) is False
    assert selects_nothing(slice(None, 3, -1)) is False
    assert selects_nothing(slice(3, None, -1)) is False


def test_slice_indices_exhaustive():
    checked = 0
    for length in (0, 1, 2, 5, 8, 13):
        for s in _grid():
            if s.step == 0:
                continue
            got = slice_indices(s, length)
            start, stop, step = s.indices(length)
            assert got == (start, stop, step, len(range(start, stop, step))), (
                length, s, got
            )
            checked += 1
    assert checked == 486 * 6  # the full grid at every length


@pytest.mark.parametrize(
    "length,s",
    [
        (0, slice(None)),
        (0, slice(None, None, -1)),
        (5, slice(-100, 100)),
        (5, slice(-100, 100, -1)),
        (5, slice(3, 1, -1)),
        (5, slice(4, 0, -2)),
        (7, slice(None, None, 3)),
        (7, slice(-1, -5, -1)),
        (1, slice(0, 100, 7)),
        (10**18, slice(-1, None, 1)),
    ],
)
def test_slice_indices_large_magnitudes(length, s):
    start, stop, step = s.indices(length)
    assert slice_indices(s, length) == (start, stop, step, len(range(start, stop, step)))


def test_slice_indices_delegates_outside_int64():
    """A length beyond int64 cannot be represented in the kernel's arithmetic,
    so the shim uses the interpreter, which is the reference by construction."""
    s = slice(2, 9)
    huge = 2**64 + 5
    assert slice_indices(s, huge) == (*s.indices(huge), len(range(2, 9, 1)))
