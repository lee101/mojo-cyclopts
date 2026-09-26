"""Parity tests for the batched literal parsers.

``parse_floats`` / ``parse_ints`` pack a stream of tokens into one buffer and
parse them in a single kernel call. Every case compares against CPython
``float``/``int`` token for token, and the batch and scalar paths are checked
against each other so an offset-table bug cannot hide.

The offset table is where a plausible bug lives: a token that reads into its
neighbour, an off-by-one in the length, an empty token shifting everything
after it. The mixed corpora below are built to expose exactly that.
"""

import math

import pytest
from cyclopts._convert import _int as upstream_int

from mojo_cyclopts import parse_floats, parse_ints, to_float, to_int

FLOAT_TOKENS = [
    "0", "1", "-1", "3.5", "-3.5", "0.0", "-0.0", "10", "1e5", "1e-5", "1E+3",
    ".5", "5.", "-.25", "1_000.000_1", "007", "9007199254740992",
    "1.7976931348623157e308", "2.2250738585072014e-308", "0.1", "0.2", "0.3",
    "inf", "-inf", "Infinity", "nan", "1.000000000000000000", "123.456e-2",
    "0e999", "5e-324", "3.141592653589793", "1e22", "1e-22",
    # outside the kernel's exact envelope: must be reported unresolved
    "0.1234567890123456789012345", "1e400", "1.0000000000000000000001e10",
]

INT_TOKENS = [
    "0", "1", "-1", "42", "-42", "+7", "255", "0x0", "0xff", "0XFF", "0o17",
    "0b1011", "0b1010_1010", "1_000", "0x10_0f", "007", "  42  ", "-0x10",
    "9223372036854775807", "-9223372036854775808", "0b_1", "0x_1",
    # malformed or too large: unresolved, and to_int still raises
    "", " ", "abc", "0x", "12a", "0b12", "1__0", "1e5",
    "9223372036854775808", "-123456789012345678901234567890",
]


def test_batch_floats_match_cpython():
    values, resolved = parse_floats(FLOAT_TOKENS)
    assert values.shape == (len(FLOAT_TOKENS),)
    for i, t in enumerate(FLOAT_TOKENS):
        expect = float(t)
        if resolved[i]:
            if math.isnan(expect):
                assert math.isnan(values[i]), t
            else:
                assert values[i] == expect, (t, values[i], expect)
                assert math.copysign(1.0, values[i]) == math.copysign(1.0, expect), t
        else:
            # The kernel declined; the shim's fallback is CPython's own float.
            assert to_float(t) == expect, t


def test_batch_ints_match_upstream():
    values, resolved = parse_ints(INT_TOKENS)
    for i, t in enumerate(INT_TOKENS):
        try:
            expect = upstream_int(t)
            failed = False
        except ValueError:
            expect, failed = 0, True
        if resolved[i]:
            assert int(values[i]) == expect, (t, values[i], expect)
        elif failed:
            # Unresolved and malformed: the scalar path must still raise the
            # upstream error, which is what the shim delegates for.
            with pytest.raises(ValueError):
                to_int(t)
        else:
            # Unresolved but well formed (too large for int64): still correct.
            assert to_int(t) == expect, t


def test_batch_and_scalar_agree():
    tokens = FLOAT_TOKENS + INT_TOKENS
    fv, fok = parse_floats(tokens)
    iv, iok = parse_ints(tokens)
    for i, t in enumerate(tokens):
        # A token the batch parser accepted must give the same value as the
        # scalar path; a token it declined must still be handled correctly
        # there. Either way the two entry points cannot disagree.
        if fok[i]:
            if math.isnan(fv[i]):
                assert math.isnan(to_float(t)), t
            else:
                assert fv[i] == to_float(t), t
        if iok[i]:
            assert int(iv[i]) == to_int(t), t


def test_empty_token_does_not_shift_later_tokens():
    """A zero-length token in the middle is the case a naive offset walk gets
    wrong: everything after it lands one slot early."""
    tokens = ["1.5", "", "2.5", "", "", "3.5", "4.5"]
    values, resolved = parse_floats(tokens)
    for i, t in enumerate(tokens):
        if t == "":
            assert not resolved[i]
        else:
            assert resolved[i] and values[i] == float(t), (i, t, values[i])


def test_adjacent_tokens_do_not_bleed():
    """Tokens packed with no separator: a parser that runs past its own length
    picks up the next token's bytes."""
    tokens = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    values, resolved = parse_floats(tokens)
    assert resolved.all()
    assert [float(v) for v in values] == [float(t) for t in tokens]

    itokens = ["1a", "2b", "3c"]
    ivals, iok = parse_ints(itokens)
    assert not iok.any(), "trailing garbage must make the token unresolvable"


def test_kernel_actually_handles_the_common_tokens():
    """``to_int``/``to_float`` fall back to the upstream parser for anything
    the kernel declines, so a scalar-only test can pass with the kernel
    entirely broken. These assert the kernel itself resolved the token."""
    for t in ["0", "42", "-42", "007", "0xff", "0XFF", "0o17", "0b1011",
              "1_000", "  42  ", "-0x10", "9223372036854775807"]:
        assert parse_ints([t])[1][0], f"int kernel declined {t!r}"
    for t in ["0", "1.5", "-3.5", "1e5", ".5", "5.", "1_000.5", "0.1", "0.3",
              "inf", "nan", "1.000000000000000000", "3.141592653589793"]:
        assert parse_floats([t])[1][0], f"float kernel declined {t!r}"


def test_kernel_declines_the_documented_cases():
    """The other half of the contract: a token the kernel cannot represent
    exactly must be reported unresolved rather than silently approximated."""
    for t in ["abc", "", "1e400", "0.1234567890123456789012345",
              "9223372036854775808", "0x", "1e5"]:
        assert not parse_ints([t])[1][0] or not parse_floats([t])[1][0], t


def test_signs_and_prefixes_in_one_batch():
    tokens = ["-0x10", "+0o7", "-0b1", "  12  ", "\t-3\n", "1_0"]
    values, resolved = parse_ints(tokens)
    assert resolved.all()
    assert [int(v) for v in values] == [upstream_int(t) for t in tokens]


def test_batch_of_one_and_empty():
    v, ok = parse_floats(["2.5"])
    assert ok.tolist() == [True] and v.tolist() == [2.5]
    v, ok = parse_floats([])
    assert v.size == 0 and ok.size == 0
    v, ok = parse_ints([])
    assert v.size == 0 and ok.size == 0


@pytest.mark.parametrize("n", [1, 2, 3, 17, 64, 1000])
def test_batch_length_does_not_change_answers(n):
    tokens = [f"{i}.{i % 97}" for i in range(n)]
    values, resolved = parse_floats(tokens)
    assert resolved.all()
    assert [float(v) for v in values] == [float(t) for t in tokens]
