"""Parity tests for the ASCII literal parsers.

``to_int`` is compared against ``cyclopts._convert._int`` token for token.
``to_float`` is compared against CPython's ``float``, and on the kernel's exact
path the comparison is *bit-for-bit*, not a tolerance: the kernel only claims a
result when the single multiply/divide it performs is provably correctly
rounded, so a one-ULP error is a real failure. ``round_half_even`` is compared
against Python's ``round`` exactly, because it is exact integer arithmetic.
"""

import math

import pytest
from cyclopts._convert import _int as upstream_int

from mojo_cyclopts import round_half_even, to_float, to_int

INT_TOKENS = [
    "0", "1", "9", "10", "255", "4096", "-1", "-42", "+7", "+0", "-0",
    "007", "0x0", "0xff", "0xFF", "0XdeadBEEF", "-0xff", "+0X10",
    "0o0", "0o17", "0O777", "-0o7",
    "0b0", "0b1011", "0B1111", "-0b1",
    "1_000", "1_0_0", "0x_10" if False else "0x10_0f", "0b1010_1010",
    "  42", "42  ", "\t42\n", "  -17  ", "0b_1", "0x_1", "-0x_10",
    "9007199254740991", "-9007199254740991", "9223372036854775807",
    "-9223372036854775808",
    # routed to upstream by the shim, must still agree
    "9223372036854775808", "123456789012345678901234567890",
    "-123456789012345678901234567890",
    "30.0", "-30.4", "2.5", "0.0", "-0.0", "  12.6  ",
]

BAD_INT_TOKENS = ["", " ", "abc", "0x", "--1", "1.2.3", "12a", "1_", "_1",
                   "1e5", "0b12", "0o18", "0xg", "1__0"]


@pytest.mark.parametrize("token", INT_TOKENS)
def test_to_int_matches_upstream(token):
    assert to_int(token) == upstream_int(token)


@pytest.mark.parametrize("token", BAD_INT_TOKENS)
def test_to_int_rejects_like_upstream(token):
    with pytest.raises(ValueError) as theirs:
        upstream_int(token)
    with pytest.raises(ValueError) as mine:
        to_int(token)
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize(
    "token",
    ["10", "10.0", "1_0", "1_0.5", "007", "0", "-0", "-0.0", "+5", "5.",
     ".5", "-.25", "1e5", "1E5", "1e+5", "1e-5", "-1.5e-3", "1.5E+3",
     "1e22", "1e-22", "123456789012345678", "0.1", "0.2", "0.3",
     "3.141592653589793", "2.718281828459045", "1e-300", "1e300",
     "  2.5  ", "1_000.000_1", "inf", "-inf", "+inf", "INF", "Infinity",
     "-INFINITY", "nan", "NaN", "-nan", "0e999", "-0e-999", "1e-323",
     "9007199254740991", "9007199254740992", "0.1000000000000000055511"],
)
def test_to_float_matches_cpython_exactly(token):
    expect = float(token)
    got = to_float(token)
    if math.isnan(expect):
        assert math.isnan(got)
    else:
        # Bit-for-bit: the kernel's fast path is correctly rounded, so an
        # inexact result here means the kernel claimed a value it could not
        # guarantee.
        assert got == expect and math.copysign(1.0, got) == math.copysign(1.0, expect), token


@pytest.mark.parametrize("token", ["", "abc", "1.2.3", "1e", "1e+", "1..2", "--1", "1_", "_1", "1e5x", "."])
def test_to_float_rejects_like_cpython(token):
    with pytest.raises(ValueError) as theirs:
        float(token)
    with pytest.raises(ValueError) as mine:
        to_float(token)
    assert str(mine.value) == str(theirs.value)


def test_float_out_of_envelope_is_still_correct():
    """Tokens past the exact envelope (long digit strings, big exponents) are
    delegated to CPython, so they stay correct even though the kernel declines
    them. Getting these wrong would show up as a ULP error."""
    for token in ["1.7976931348623157e308", "2.2250738585072014e-308",
                  "1.0000000000000000000001e10", "0.1234567890123456789012345",
                  "1e400", "-1e400", "5e-324", "2e-324"]:
        got = to_float(token)
        expect = float(token)
        assert got == expect, token


def test_round_half_even_matches_python():
    vals = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.0, 1.0, 2.0, 3.0,
            0.49999999999999994, 2.5000000000000004, 123456.5, 123457.5,
            -123456.5, 4.5, -4.5, 1e16 + 0.5, 1e18,
            2**53 + 0.5, 0.4999999999999999]
    got = round_half_even(vals)
    for v, g in zip(vals, got):
        assert int(g) == round(v), v


def test_round_half_even_refuses_beyond_int64():
    """The batch form is int64, so a value whose rounded result overflows it is
    reported rather than silently truncated."""
    assert list(round_half_even([5.0, 6.0])) == [5, 6]
    for vals in ([1e19], [-1e19], [5.0, 1e300]):
        with pytest.raises(OverflowError):
            round_half_even(vals)


def test_round_half_even_empty():
    assert round_half_even([]).size == 0
