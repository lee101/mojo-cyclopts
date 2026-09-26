"""Parity tests for ``mojo_cyclopts.validators.Number``.

Every case is checked against the real ``cyclopts.validators.Number`` on the
same input, including the exact ``ValueError`` message. A plausible bug -- a
comparison negated the wrong way, the wrong rule reported, an off-by-one in
the batch index mapping, NaN sneaking through a bound -- changes the message or
the index, so it fails here.
"""

import math

import pytest

from cyclopts.validators import Number as UpstreamNumber

from mojo_cyclopts import Number


def _both(**kw):
    return UpstreamNumber(**kw), Number(**kw)


def _call(fn, value):
    try:
        fn(int, value)
    except ValueError as exc:
        return ("raise", str(exc))
    return ("ok", None)


@pytest.mark.parametrize(
    "kw,values",
    [
        ({}, [0, 1, -1, 2.5, -2.5, 0.0, 1e300]),
        ({"lt": 10}, [-1e9, -1, 0, 9, 9.999, 10, 10.001, 1e9]),
        ({"lte": 10}, [-1e9, 9, 10, 10.0001, 1e9]),
        ({"gt": 10}, [-1e9, 9.999, 10, 10.0001, 1e9]),
        ({"gte": 10}, [-1e9, 9.999, 10, 1e9]),
        ({"lt": 0, "gte": -10}, [-11, -10, -1, 0, 1]),
        ({"modulo": 5}, [0, 5, 10, 15, 1, -5, -10, 2.5, 7.5]),
        ({"modulo": 3}, [0, 3, 9, 6, 1, 2, 10, -3, -6]),
        ({"lt": 100, "lte": 100, "gt": 0, "gte": 0, "modulo": 7},
         [-1, 0, 1, 6, 7, 8, 98, 99, 100, 101, 105, 112]),
    ],
)
def test_scalar_parity(kw, values):
    theirs, mine = _both(**kw)
    for v in values:
        assert _call(mine, v) == _call(theirs, v), (kw, v)


@pytest.mark.parametrize(
    "kw,values",
    [
        ({"gte": 0, "lte": 150}, [0, 1, 149, 150, 151, -1]),
        ({"modulo": 4}, [0, 4, 8, 3, 9, -4]),
        ({"lt": 1.0}, [0.999999, 1.0, 1.000001]),
    ],
)
def test_container_parity(kw, values):
    """A container is validated element by element; upstream stops at the first
    failure, so the reported constraint must be the first failing leaf's."""
    theirs, mine = _both(**kw)
    assert _call(mine, list(values)) == _call(theirs, list(values))
    assert _call(mine, tuple(values)) == _call(theirs, tuple(values))
    assert _call(mine, {k: v for k, v in enumerate(values)}) == _call(
        theirs, {k: v for k, v in enumerate(values)}
    )


def test_non_numeric_leaves_are_ignored():
    """Upstream returns silently for anything that is not int|float."""
    theirs, mine = _both(gte=0, lte=10)
    for value in [None, object(), 1 + 2j, [], {}]:
        assert _call(mine, value) == _call(theirs, value) == ("ok", None)
    assert _call(mine, [1, None, 2, object(), 11]) == _call(theirs, [1, None, 2, object(), 11])
    # bytes is a Sequence, so upstream recurses into it as a run of ints and
    # so must the shim.
    assert _call(mine, b"xyz") == _call(theirs, b"xyz")


def test_str_container_raises_type_error():
    """A str is a Sequence but never a container of values, so upstream's
    iter_container_elements rejects it -- including as an element of a list,
    because the validator recurses into every leaf."""
    theirs, mine = _both(gte=0)
    for value in ("abc", ["a", "b"]):
        with pytest.raises(TypeError):
            theirs(int, value)
        with pytest.raises(TypeError):
            mine(int, value)


def test_nan_cannot_bypass_a_bound():
    """The upstream validator negates each comparison precisely so that NaN,
    which compares false against everything, is rejected. A kernel that
    rewrote the rule as `v >= lt` would let NaN through."""
    theirs, mine = _both(lt=1, lte=1, gt=0, gte=0)
    for v in (math.nan, math.inf, -math.inf):
        assert _call(mine, v) == _call(theirs, v)
    assert _call(mine, math.nan) != ("ok", None)
    # A validator with no bounds accepts NaN, exactly as upstream does.
    plain_theirs, plain_mine = _both()
    assert _call(plain_mine, math.nan) == _call(plain_theirs, math.nan) == ("ok", None)


def test_batch_reports_the_first_failing_leaf():
    """The kernel returns the first violating index in its own buffer and the
    shim maps that back to the caller's element order. A wrong mapping shows
    up here because the reported constraint differs leaf by leaf. The ladder
    order is lt, lte, gt, gte, modulo -- so a value that breaks both lte and
    gte is reported as the lte failure, exactly as upstream reports it."""
    assert _call(Number(lte=1, gte=0), [0, 1, 2]) == ("raise", "Must be <= 1.")
    assert _call(Number(gte=0, lte=1), [5, -5]) == ("raise", "Must be <= 1.")
    # -5 clears lte but breaks gte, so the reported rule is the one that
    # actually fired, not the first bound in the constructor call.
    assert _call(Number(gte=0, lte=1), [-5, 5]) == ("raise", "Must be >= 0.")
    assert _call(Number(lt=0, gte=-10), [-5, -20]) == ("raise", "Must be >= -10.")
    assert _call(Number(lt=0, gte=-10), [-20, -5]) == ("raise", "Must be >= -10.")
    # A leaf that breaks only the later rule still reports that rule.
    assert _call(Number(gte=0), [-5, 5]) == ("raise", "Must be >= 0.")
    assert _call(Number(lte=1, gte=0), [0, 1, 0, 1, 9]) == ("raise", "Must be <= 1.")


def test_empty_and_all_valid_containers_pass():
    theirs, mine = _both(gte=0)
    assert _call(mine, []) == ("ok", None)
    assert _call(mine, [1, 2, 3]) == ("ok", None) == _call(theirs, [1, 2, 3])


def test_ints_beyond_float64_range_fall_back_to_python():
    """An int larger than 2**53 is not exact in float64, so the shim checks it
    with Python's own arbitrary-precision comparisons instead of the kernel.
    Upstream's answer is the reference."""
    theirs, mine = _both(lt=2**53, modulo=3)
    big = 2**60 + 1
    assert _call(mine, big) == _call(theirs, big)
    assert _call(mine, [big, 1]) == _call(theirs, [big, 1])
    # A wide value and a failing narrow value: the earlier failure wins.
    assert _call(mine, [2**70, 99]) == _call(theirs, [2**70, 99])
    assert _call(mine, [99, 2**70]) == _call(theirs, [99, 2**70])


def test_modulo_zero_raises_zero_division():
    theirs, mine = _both(modulo=0)
    with pytest.raises(ZeroDivisionError):
        theirs(int, 5)
    with pytest.raises(ZeroDivisionError):
        mine(int, 5)


def test_equality_and_repr():
    assert Number(gte=0, lte=5) == Number(gte=0, lte=5)
    assert Number(gte=0) != Number(gte=1)
    assert "gte=0" in repr(Number(gte=0))
