"""Mojo kernels for the numeric surface of cyclopts.

cyclopts is a command-line argument parser. Its bulk is plumbing: signature
binding, help rendering, shell completion, string conversion and error
formatting. The genuinely numeric part is small but real, and this is what
lives here:

  * ``validators.Number`` -- range / modulo validation of numeric CLI values.
    The Python side hands the kernel a flat run of numbers and the kernel
    classifies them, which is the same comparison ladder the pure-Python
    validator walks one element at a time.
  * ``validators.Slice._selects_nothing`` -- integer range-frame analysis that
    decides whether a ``slice`` provably selects nothing.
  * ``slice.indices`` -- CPython's ``slice_adjust_indices`` plus the stepped
    range-length formula, reproduced exactly on int64.
  * ``_convert._int`` / ``_convert._float`` -- the ASCII-to-number literal
    parser, including base prefixes, signs, underscores and decimal exponents.
    Both a scalar and a batched entry point, because a stream of tokens packs
    into one buffer and amortises the call boundary.

Every exported symbol takes buffer addresses as plain ``Int`` values and
rebuilds the pointer inside the body, because ``@export`` rejects parametric
functions and an inferred pointer origin would make the symbol parametric.
"""

from std.math import floor as _floor
from std.math import inf
from std.utils import StaticTuple

comptime U8P = Pointer[UInt8, AnyOrigin[mut=True]]
comptime I64P = Pointer[Int64, AnyOrigin[mut=True]]
comptime I32P = Pointer[Int32, AnyOrigin[mut=True]]
comptime F64P = Pointer[Float64, AnyOrigin[mut=True]]

comptime RULE_NONE: Int32 = 0
comptime RULE_LT: Int32 = 1
comptime RULE_LTE: Int32 = 2
comptime RULE_GT: Int32 = 3
comptime RULE_GTE: Int32 = 4
comptime RULE_MOD: Int32 = 5

# Exact powers 10**0 .. 10**22. Every one of these is exactly representable as
# a float64, which is what makes the single multiply/divide in the float
# literal parser correctly rounded.
comptime POW10 = StaticTuple[Float64, 23](
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
)


# ---------------------------------------------------------------------------
# validators.Number
# ---------------------------------------------------------------------------


@export("co_number_check")
def co_number_check(
    vals_addr: Int,
    n: Int,
    has_lt: Int,
    lt: Float64,
    has_lte: Int,
    lte: Float64,
    has_gt: Int,
    gt: Float64,
    has_gte: Int,
    gte: Float64,
    has_mod: Int,
    modulo: Float64,
    first_bad_addr: Int,
    first_rule_addr: Int,
) abi("C") -> Int64:
    """Classify ``n`` values against the ``Number`` constraint ladder.

    Returns the number of violations and stores the first violating index in
    ``first_bad[0]`` and the constraint that rejected it (a RULE_* code) in
    ``first_rule[0]``. ``first_bad`` is -1 when nothing violates.

    The comparisons are negated exactly the way cyclopts negates them --
    ``not (v < lt)`` rather than ``v >= lt``. The two agree for ordinary
    values, but for a NaN the negated form is true, so NaN is rejected. That
    is the behaviour the upstream validator documents.
    """
    var vals = F64P(unsafe_from_address=vals_addr)
    var first_bad = I64P(unsafe_from_address=first_bad_addr)
    var first_rule = I32P(unsafe_from_address=first_rule_addr)
    var bad = Int64(0)
    first_bad[unsafe_offset=0] = Int64(-1)
    first_rule[unsafe_offset=0] = RULE_NONE
    for i in range(n):
        var v = vals[unsafe_offset=i]
        var rule = RULE_NONE
        if has_lt != 0 and not (v < lt):
            rule = RULE_LT
        elif has_lte != 0 and not (v <= lte):
            rule = RULE_LTE
        elif has_gt != 0 and not (v > gt):
            rule = RULE_GT
        elif has_gte != 0 and not (v >= gte):
            rule = RULE_GTE
        elif has_mod != 0 and (v % modulo) != Float64(0.0):
            rule = RULE_MOD
        if rule != RULE_NONE:
            bad += Int64(1)
            if first_bad[unsafe_offset=0] == Int64(-1):
                first_bad[unsafe_offset=0] = Int64(i)
                first_rule[unsafe_offset=0] = rule
    return bad


# ---------------------------------------------------------------------------
# validators.Slice._selects_nothing
# ---------------------------------------------------------------------------


@export("co_slice_selects_nothing")
def co_slice_selects_nothing(
    has_start: Int, start: Int64, has_stop: Int, stop: Int64,
    has_step: Int, step_in: Int64,
) abi("C") -> Int32:
    """Port of ``cyclopts.validators._slice._selects_nothing``.

    Returns 1 when the slice provably selects zero elements for *any* sequence
    length, 0 when it does not -- including the "cannot prove it" cases, which
    upstream also allows. A zero step means applying the slice raises, so it is
    not reported as empty.
    """
    var step = Int64(1)
    if has_step != 0:
        step = step_in
    if step == Int64(0):
        return Int32(0)

    var start_c = start if has_start != 0 else Int64(0)
    var stop_c = stop
    if step > Int64(0):
        if has_stop == 0:
            # Runs to the end of the sequence: never provably empty.
            return Int32(0)
    else:
        if has_start == 0 or has_stop == 0:
            return Int32(0)

    # Both bounds are concrete, but they are only comparable without knowing
    # the sequence length when they share a sign frame.
    if (start_c < Int64(0)) != (stop_c < Int64(0)):
        return Int32(0)

    if step > Int64(0):
        return Int32(1) if start_c >= stop_c else Int32(0)
    return Int32(1) if start_c <= stop_c else Int32(0)


# ---------------------------------------------------------------------------
# slice.indices
# ---------------------------------------------------------------------------


@export("co_slice_resolve")
def co_slice_resolve(
    has_start: Int, start_in: Int64, has_stop: Int, stop_in: Int64,
    has_step: Int, step_in: Int64, length: Int64, res_addr: Int,
) abi("C") -> Int64:
    """CPython ``slice.indices`` on int64.

    ``res[0..2]`` receive (start, stop, step) and the return value is
    ``len(range(start, stop, step))``. Bounds are clipped exactly the way
    ``slice_adjust_indices`` clips them: a negative bound is offset by the
    length and then clamped to -1 (negative step) or 0, and a non-negative
    bound is clamped to ``length - 1`` (negative step) or ``length``.
    """
    var res = I64P(unsafe_from_address=res_addr)
    var step = Int64(1)
    if has_step != 0:
        step = step_in

    var lower = Int64(0)
    var upper = length
    if step < Int64(0):
        lower = Int64(-1)
        upper = length - Int64(1)

    var start = start_in
    var stop = stop_in
    if has_start == 0:
        start = upper if step < Int64(0) else lower
    elif start < Int64(0):
        start += length
        if start < Int64(0):
            start = lower
    elif start >= length:
        start = upper
    if has_stop == 0:
        stop = lower if step < Int64(0) else upper
    elif stop < Int64(0):
        stop += length
        if stop < Int64(0):
            stop = lower
    elif stop >= length:
        stop = upper

    res[unsafe_offset=0] = start
    res[unsafe_offset=1] = stop
    res[unsafe_offset=2] = step

    var count = Int64(0)
    if step > Int64(0):
        if stop > start:
            count = (stop - start + step - Int64(1)) // step
    else:
        if start > stop:
            count = (start - stop + (-step) - Int64(1)) // (-step)
    return count


# ---------------------------------------------------------------------------
# ASCII literal parsing
# ---------------------------------------------------------------------------


@inline(.always)
def _is_space(c: UInt8) -> Bool:
    return c == UInt8(32) or c == UInt8(9) or c == UInt8(10) or c == UInt8(11) or c == UInt8(12) or c == UInt8(13)


@inline(.always)
def _digit_value(c: UInt8) -> Int32:
    if c >= UInt8(48) and c <= UInt8(57):
        return Int32(c - UInt8(48))
    if c >= UInt8(97) and c <= UInt8(102):
        return Int32(c - UInt8(97)) + Int32(10)
    if c >= UInt8(65) and c <= UInt8(70):
        return Int32(c - UInt8(65)) + Int32(10)
    return Int32(-1)


@inline(.always)
def _lower(c: UInt8) -> UInt8:
    if c >= UInt8(65) and c <= UInt8(90):
        return c + UInt8(32)
    return c


@inline(.always)
def _eq_ci(p: U8P, i: Int, j: Int, c: UInt8) -> Bool:
    return _lower(p[unsafe_offset=i + j]) == c


@inline(.always)
def _is_inf_word(p: U8P, lo: Int, rest: Int) -> Bool:
    if rest == 3:
        return _eq_ci(p, lo, 0, UInt8(105)) and _eq_ci(p, lo, 1, UInt8(110)) and _eq_ci(p, lo, 2, UInt8(102))
    if rest == 8:
        return (
            _eq_ci(p, lo, 0, UInt8(105))
            and _eq_ci(p, lo, 1, UInt8(110))
            and _eq_ci(p, lo, 2, UInt8(102))
            and _eq_ci(p, lo, 3, UInt8(105))
            and _eq_ci(p, lo, 4, UInt8(110))
            and _eq_ci(p, lo, 5, UInt8(105))
            and _eq_ci(p, lo, 6, UInt8(116))
            and _eq_ci(p, lo, 7, UInt8(121))
        )
    return False


def _parse_int_at(base_addr: Int, n: Int) -> Tuple[Int64, Bool]:
    """Integer literal over ``p[0:n]``.

    Handles a leading sign, ``0x``/``0o``/``0b`` base prefixes, ASCII
    whitespace around the token, and ``_`` digit separators. Returns the value
    and True, or (0, False) when the token is malformed or the value does not
    fit in an int64 -- the caller then defers to the real parser so the raised
    error text stays the upstream one.
    """
    var p = U8P(unsafe_from_address=base_addr)
    var lo = Int(0)
    var hi = n
    while lo < hi and _is_space(p[unsafe_offset=lo]):
        lo += 1
    while hi > lo and _is_space(p[unsafe_offset=hi - 1]):
        hi -= 1
    if lo >= hi:
        return (Int64(0), False)

    var neg = False
    if p[unsafe_offset=lo] == UInt8(45):
        neg = True
        lo += 1
    elif p[unsafe_offset=lo] == UInt8(43):
        lo += 1
    if lo >= hi:
        return (Int64(0), False)

    # Base prefix. Only the first letter after the leading 0 distinguishes the
    # bases: 0x, 0o, 0b. Anything else after the 0 is an ordinary decimal zero
    # (so "007" is 7), and a token that is just "0x" is malformed.
    var base = Int64(10)
    if hi - lo >= 2 and p[unsafe_offset=lo] == UInt8(48):
        var c1 = _lower(p[unsafe_offset=lo + 1])
        if c1 == UInt8(120):
            base = Int64(16)
            lo += 2
        elif c1 == UInt8(111):
            base = Int64(8)
            lo += 2
        elif c1 == UInt8(98):
            base = Int64(2)
            lo += 2

    var acc = Int64(0)
    var seen = Int32(0)
    var prev_digit = False
    var overflow = False
    var i = lo
    while i < hi:
        var c = p[unsafe_offset=i]
        if c == UInt8(95):
            # Underscores are only legal between digits.
            if not prev_digit or i + 1 >= hi:
                return (Int64(0), False)
            i += 1
            prev_digit = False
            continue
        var d = _digit_value(c)
        if d < Int32(0) or Int64(d) >= base:
            return (Int64(0), False)
        var next = acc * base + Int64(d)
        if next < acc:
            overflow = True
        acc = next
        seen += Int32(1)
        prev_digit = True
        i += 1
    if seen == Int32(0) or overflow:
        return (Int64(0), False)
    return (-acc if neg else acc, True)


def _parse_float_at(base_addr: Int, n: Int) -> Tuple[Float64, Bool]:
    """Floating-point literal over ``p[0:n]``, on the exact fast path.

    Trailing zeros are stripped from the digit string and the value is formed
    as ``mantissa * 10**k`` (or ``/ 10**k``). When the mantissa is at most
    2**53 and ``|k| <= 22`` both operands are exactly representable and the
    single multiply or divide is correctly rounded, so the result is
    bit-identical to a full correctly-rounded ``strtod``. Outside that
    envelope -- very long literals, or exponents past the exact power-of-ten
    table -- the second element is False and the caller hands the token to the
    real parser.

    ``inf``, ``infinity`` and ``nan`` (any case, optional sign) are handled
    directly, and an all-zero literal is 0.0 regardless of exponent.
    """
    var p = U8P(unsafe_from_address=base_addr)
    var lo = Int(0)
    var hi = n
    while lo < hi and _is_space(p[unsafe_offset=lo]):
        lo += 1
    while hi > lo and _is_space(p[unsafe_offset=hi - 1]):
        hi -= 1
    if lo >= hi:
        return (Float64(0.0), False)

    var neg = False
    if p[unsafe_offset=lo] == UInt8(45):
        neg = True
        lo += 1
    elif p[unsafe_offset=lo] == UInt8(43):
        lo += 1
    if lo >= hi:
        return (Float64(0.0), False)

    var rest = hi - lo
    if _is_inf_word(p, lo, rest):
        return (-inf[DType.float64]() if neg else inf[DType.float64](), True)
    if (
        rest == 3
        and _eq_ci(p, lo, 0, UInt8(110))
        and _eq_ci(p, lo, 1, UInt8(97))
        and _eq_ci(p, lo, 2, UInt8(110))
    ):
        return (Float64(0.0) / Float64(0.0), True)

    # The digit string is accumulated in uint64 so that up to 19 digits fit
    # even when the leading ones already exceed 2**53. The 2**53 test happens
    # later, after trailing zeros have been stripped, so a literal such as
    # "1.000000000000000000" -- nineteen digits, mantissa 10**18 -- is still
    # recognised as the single digit it really is.
    var mant = UInt64(0)
    var ndigits = Int32(0)
    var too_many = False
    var exp10 = Int64(0)
    var seen_digit = False
    var prev_digit = False
    var after_point = False
    var i = lo
    while i < hi:
        var c = p[unsafe_offset=i]
        if c == UInt8(46) and not after_point:
            after_point = True
            i += 1
            continue
        if c == UInt8(95):
            if not prev_digit or i + 1 >= hi:
                return (Float64(0.0), False)
            i += 1
            prev_digit = False
            continue
        var d = _digit_value(c)
        if d < Int32(0) or d > Int32(9):
            break
        if not too_many:
            if ndigits >= Int32(19):
                too_many = True
            else:
                mant = mant * UInt64(10) + UInt64(d)
        ndigits += Int32(1)
        seen_digit = True
        prev_digit = True
        if after_point:
            exp10 -= Int64(1)
        i += 1
    if not seen_digit:
        return (Float64(0.0), False)

    # Exponent part, if any.
    if i < hi and (p[unsafe_offset=i] == UInt8(101) or p[unsafe_offset=i] == UInt8(69)):
        i += 1
        var eneg = False
        if i < hi and (p[unsafe_offset=i] == UInt8(45) or p[unsafe_offset=i] == UInt8(43)):
            eneg = p[unsafe_offset=i] == UInt8(45)
            i += 1
        var e = Int64(0)
        var eover = False
        var edigits = False
        while i < hi:
            var c = p[unsafe_offset=i]
            if c == UInt8(95):
                if not prev_digit or i + 1 >= hi:
                    return (Float64(0.0), False)
                i += 1
                prev_digit = False
                continue
            var d = _digit_value(c)
            if d < Int32(0) or d > Int32(9):
                return (Float64(0.0), False)
            e = e * Int64(10) + Int64(d)
            if e > Int64(1000000):
                eover = True
                e = Int64(1000000)
            edigits = True
            prev_digit = True
            i += 1
        if not edigits:
            return (Float64(0.0), False)
        exp10 = exp10 - e if eneg else exp10 + e
        if eover:
            exp10 = Int64(1000000)
    if i != hi:
        return (Float64(0.0), False)

    if mant == UInt64(0):
        return (Float64(-0.0) if neg else Float64(0.0), True)
    if too_many:
        return (Float64(0.0), False)

    # Dropping a trailing zero divides the mantissa by 10, so the exponent has
    # to move the other way to leave the value alone. It buys coverage: a
    # literal like "1.000000000000000000" is 10**18 in digit form but a single
    # digit once stripped.
    while mant % UInt64(10) == UInt64(0):
        mant = mant // UInt64(10)
        exp10 += Int64(1)

    if mant > UInt64(9007199254740992):
        return (Float64(0.0), False)
    if exp10 > Int64(22) or exp10 < Int64(-22):
        return (Float64(0.0), False)

    # A positive exponent multiplies and a negative one divides, by an
    # exactly-representable power of ten. Either way the single operation has
    # one rounding, so the result is the correctly-rounded value of the
    # literal -- what a full strtod produces. Multiplying by a rounded 1e-1
    # instead would not be, which is why 10**-1 is not in the table.
    var value: Float64
    if exp10 > Int64(0):
        value = Float64(mant) * POW10[exp10]
    elif exp10 < Int64(0):
        value = Float64(mant) / POW10[-exp10]
    else:
        value = Float64(mant)
    return (-value if neg else value, True)


@export("co_parse_int")
def co_parse_int(s_addr: Int, n: Int, ok_addr: Int) abi("C") -> Int64:
    """Scalar integer literal parse. See ``_parse_int_at``."""
    var ok = I32P(unsafe_from_address=ok_addr)
    var r = _parse_int_at(s_addr, n)
    if r[1]:
        ok[unsafe_offset=0] = Int32(1)
        return r[0]
    ok[unsafe_offset=0] = Int32(0)
    return Int64(0)


@export("co_parse_float")
def co_parse_float(s_addr: Int, n: Int, ok_addr: Int) abi("C") -> Float64:
    """Scalar floating-point literal parse. See ``_parse_float_at``."""
    var ok = I32P(unsafe_from_address=ok_addr)
    var r = _parse_float_at(s_addr, n)
    if r[1]:
        ok[unsafe_offset=0] = Int32(1)
        return r[0]
    ok[unsafe_offset=0] = Int32(0)
    return Float64(0.0)


@export("co_parse_int_batch")
def co_parse_int_batch(
    blob_addr: Int, count: Int, offs_addr: Int, lens_addr: Int,
    res_addr: Int, ok_addr: Int,
) abi("C") -> Int64:
    """Parse ``count`` integer tokens packed back to back in one buffer.

    ``offs[i]`` is where token ``i`` starts and ``lens[i]`` how long it is;
    ``res[i]`` receives the value and ``ok[i]`` 1 or 0. Returns the number of
    tokens the kernel parsed. A token's own bytes bound its scan, so a
    neighbouring token can never bleed in.
    """
    var offs = I32P(unsafe_from_address=offs_addr)
    var lens = I32P(unsafe_from_address=lens_addr)
    var res = I64P(unsafe_from_address=res_addr)
    var ok = I32P(unsafe_from_address=ok_addr)
    var done = Int64(0)
    for i in range(count):
        var start = Int(offs[unsafe_offset=i])
        var n = Int(lens[unsafe_offset=i])
        var r = _parse_int_at(blob_addr + start, n)
        if r[1]:
            done += Int64(1)
            res[unsafe_offset=i] = r[0]
            ok[unsafe_offset=i] = Int32(1)
        else:
            res[unsafe_offset=i] = Int64(0)
            ok[unsafe_offset=i] = Int32(0)
    return done


@export("co_parse_float_batch")
def co_parse_float_batch(
    blob_addr: Int, count: Int, offs_addr: Int, lens_addr: Int,
    res_addr: Int, ok_addr: Int,
) abi("C") -> Int64:
    """Parse ``count`` float tokens packed back to back in one buffer.

    Same layout as ``co_parse_int_batch``. Returns the number of tokens the
    kernel parsed, which is the number the shim did not have to hand to
    CPython's ``float``.
    """
    var offs = I32P(unsafe_from_address=offs_addr)
    var lens = I32P(unsafe_from_address=lens_addr)
    var res = F64P(unsafe_from_address=res_addr)
    var ok = I32P(unsafe_from_address=ok_addr)
    var done = Int64(0)
    for i in range(count):
        var start = Int(offs[unsafe_offset=i])
        var n = Int(lens[unsafe_offset=i])
        var r = _parse_float_at(blob_addr + start, n)
        res[unsafe_offset=i] = r[0]
        if r[1]:
            done += Int64(1)
            ok[unsafe_offset=i] = Int32(1)
        else:
            ok[unsafe_offset=i] = Int32(0)
    return done


@export("co_round_half_even")
def co_round_half_even(
    x_addr: Int, n: Int, res_addr: Int, mask_addr: Int
) abi("C") -> Int64:
    """Python ``round(float)`` semantics on a run of values.

    Ties go to even. An input whose rounded result does not fit in int64 is
    flagged: ``mask[i]`` is set to 1 and ``res[i]`` is left at 0. The return
    value is the number of flagged elements, so the shim knows exactly which
    ones cannot be represented.
    """
    var xs = F64P(unsafe_from_address=x_addr)
    var res = I64P(unsafe_from_address=res_addr)
    var mask = I32P(unsafe_from_address=mask_addr)
    var flagged = Int64(0)
    for i in range(n):
        var x = xs[unsafe_offset=i]
        var lo = _floor(x)
        var diff = x - lo
        var r = lo
        if diff > Float64(0.5):
            r = lo + Float64(1.0)
        elif diff == Float64(0.5):
            if r % Float64(2.0) != Float64(0.0):
                r = lo + Float64(1.0)
        if r >= Float64(9223372036854775808.0) or r < Float64(-9223372036854775808.0):
            res[unsafe_offset=i] = Int64(0)
            mask[unsafe_offset=i] = Int32(1)
            flagged += Int64(1)
        else:
            res[unsafe_offset=i] = Int64(r)
            mask[unsafe_offset=i] = Int32(0)
    return flagged
