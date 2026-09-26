"""Port of ``cyclopts._convert``'s numeric literal coercion onto Mojo.

``cyclopts`` turns a command-line token into a number before anything else
looks at it. ``_int`` and ``_float`` are the two functions that do the actual
arithmetic on the token, and both are byte loops: walk the ASCII, accumulate
digits, apply a sign, a base prefix or a decimal exponent. That is the work
``src/kernels.mojo`` does.

Both entry points keep the upstream contract exactly, including the raised
error type and message for a malformed token, because the shim forwards the
cases the kernel declines to the real converters. The kernel declines a token
it cannot represent exactly: an integer outside int64, or a float literal
outside the exact envelope described in ``kernels.mojo``.
"""

from __future__ import annotations

import ctypes

import numpy as np

from . import _lib

__all__ = [
    "to_int",
    "to_float",
    "parse_ints",
    "parse_floats",
    "round_half_even",
]

# A per-call np.frombuffer would allocate a fresh array object for every token,
# which costs more than the parse itself, so scalar tokens go through one
# reusable scratch buffer. The kernel is synchronous and never retains the
# pointer, so a single buffer is safe.
_SCRATCH = bytearray(4096)
_SCRATCH_ADDR = ctypes.addressof(ctypes.c_char.from_buffer(_SCRATCH))


def _token_addr(s: str) -> tuple[int, int]:
    raw = s.encode("utf-8", "surrogatepass")
    if len(raw) > len(_SCRATCH):
        raise ValueError(f"token longer than {len(_SCRATCH)} bytes")
    ctypes.memmove(_SCRATCH_ADDR, raw, len(raw))
    return _SCRATCH_ADDR, len(raw)


def _ok_slot() -> np.ndarray:
    return np.empty(1, dtype=np.int32)


def to_int(s: str) -> int:
    """``cyclopts._convert._int``.

    The kernel handles the common case: a sign, a ``0x``/``0o``/``0b`` prefix,
    surrounding ASCII whitespace, ``_`` separators, and a decimal point (which
    upstream turns into ``int(round(float(s)))``). Malformed tokens and values
    outside int64 are handed to the upstream converter, so the ``ValueError``
    text is upstream's.
    """
    from cyclopts._convert import _int as upstream_int

    addr, n = _token_addr(s)
    ok = _ok_slot()
    value = _lib.lib.co_parse_int(addr, n, ok.ctypes.data)
    if int(ok[0]) == 1:
        return int(value)
    return upstream_int(s)


def to_float(s: str) -> float:
    """``float`` coercion, as cyclopts performs it for float annotations.

    On the kernel's exact path the result is bit-identical to CPython's
    correctly-rounded ``strtod``. Outside the exact envelope -- a significant
    digit string wider than 2**53, or a decimal exponent past the exact
    power-of-ten table -- the token is handed to CPython's ``float``.
    """
    addr, n = _token_addr(s)
    ok = _ok_slot()
    value = _lib.lib.co_parse_float(addr, n, ok.ctypes.data)
    if int(ok[0]) == 1:
        return float(value)
    return float(s)


def _pack(tokens: list[str]):
    """Pack tokens back to back into one uint8 buffer with an offset table.

    A token's own length bounds its scan, so a token can never read into its
    neighbour; the kernel receives the offsets and lengths alongside the blob.
    """
    raws = [t.encode("utf-8", "surrogatepass") for t in tokens]
    offsets = np.empty(len(raws), dtype=np.int32)
    lengths = np.empty(len(raws), dtype=np.int32)
    pos = 0
    for i, r in enumerate(raws):
        offsets[i] = pos
        lengths[i] = len(r)
        pos += len(r)
    blob = np.frombuffer(b"".join(raws), dtype=np.uint8)
    return blob, offsets, lengths


def parse_floats(tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Parse a stream of float tokens in one kernel call.

    cyclopts converts one token at a time, so this batch form is an addition
    rather than a port: it is the same parser with the tokens packed into a
    single buffer, which is what you want when a whole column of values has to
    be converted at once.

    Returns ``(values, resolved)``. ``resolved[i]`` is True where the kernel
    produced the value and False where the token fell outside its exact
    envelope, in which case ``values[i]`` is 0.0 and the caller should use
    CPython's ``float`` for that token.
    """
    blob, offsets, lengths = _pack(tokens)
    values = np.zeros(len(tokens), dtype=np.float64)
    ok = np.zeros(len(tokens), dtype=np.int32)
    _lib.lib.co_parse_float_batch(
        blob.ctypes.data, len(tokens), offsets.ctypes.data,
        lengths.ctypes.data, values.ctypes.data, ok.ctypes.data,
    )
    return values, ok.astype(bool)


def parse_ints(tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``parse_floats`` for integer tokens. Returns (values, resolved)."""
    blob, offsets, lengths = _pack(tokens)
    values = np.zeros(len(tokens), dtype=np.int64)
    ok = np.zeros(len(tokens), dtype=np.int32)
    _lib.lib.co_parse_int_batch(
        blob.ctypes.data, len(tokens), offsets.ctypes.data,
        lengths.ctypes.data, values.ctypes.data, ok.ctypes.data,
    )
    return values, ok.astype(bool)


def round_half_even(values) -> np.ndarray:
    """``round(float)`` for a run of values, via the Mojo kernel.

    Ties go to even, matching Python. This is exact integer arithmetic on an
    exact float64 input, so the answer is bit-for-bit Python's, not merely
    close. The kernel flags any input whose rounded result does not fit in
    int64; this raises ``OverflowError`` for those rather than returning a
    silently wrong value. Returns an int64 array.
    """
    xs = np.ascontiguousarray(values, dtype=np.float64)
    res = np.empty(xs.size, dtype=np.int64)
    mask = np.empty(xs.size, dtype=np.int32)
    _lib.lib.co_round_half_even(
        xs.ctypes.data, xs.size, res.ctypes.data, mask.ctypes.data
    )
    flagged = np.nonzero(mask)[0]
    if flagged.size:
        i = int(flagged[0])
        raise OverflowError(
            f"round({float(xs[i])!r}) does not fit in int64 (index {i})"
        )
    return res
