"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory: every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay ``c_int64`` for addresses. ``c_int``
truncates them and segfaults.
"""

import ctypes
import pathlib

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-cyclopts.so"

I64 = ctypes.c_int64
F64 = ctypes.c_double
I32 = ctypes.c_int32


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.co_number_check.restype = I64
    lib.co_number_check.argtypes = (
        [I64, I64]
        + [I64, F64] * 5
        + [I64, I64]
    )

    lib.co_slice_selects_nothing.restype = I32
    lib.co_slice_selects_nothing.argtypes = [I64, I64] * 3

    lib.co_slice_resolve.restype = I64
    lib.co_slice_resolve.argtypes = [I64, I64] * 3 + [I64, I64]

    lib.co_parse_int.restype = I64
    lib.co_parse_int.argtypes = [I64, I64, I64]

    lib.co_parse_float.restype = F64
    lib.co_parse_float.argtypes = [I64, I64, I64]

    lib.co_round_half_even.restype = I64
    lib.co_round_half_even.argtypes = [I64, I64, I64, I64]

    lib.co_parse_int_batch.restype = I64
    lib.co_parse_int_batch.argtypes = [I64, I64, I64, I64, I64, I64]

    lib.co_parse_float_batch.restype = I64
    lib.co_parse_float_batch.argtypes = [I64, I64, I64, I64, I64, I64]
    return lib


lib = _load()


# Rule codes the kernel reports for a rejected value; kept in step with the
# RULE_* constants in src/kernels.mojo.
RULE_NAMES = {0: None, 1: "lt", 2: "lte", 3: "gt", 4: "gte", 5: "modulo"}

# Upstream error text, keyed by the same rule code. Note the modulo rule has
# its own wording: "Must be a multiple of N."
RULE_MESSAGES = {
    1: "Must be < {bound!r}.",
    2: "Must be <= {bound!r}.",
    3: "Must be > {bound!r}.",
    4: "Must be >= {bound!r}.",
    5: "Must be a multiple of {bound!r}.",
}
