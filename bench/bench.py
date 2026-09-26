"""Correctness-gated benchmark for mojo-cyclopts.

Every case verifies agreement with the real ``cyclopts`` (or with CPython's
``float``, which is what cyclopts defers to) before timing, so a regression in
a kernel shows up as a correctness failure rather than a suspiciously good
number.

Baselines are the fastest reasonable alternative: a vectorised NumPy
expression for the validation ladder, and CPython's own C-level ``float`` /
``int`` for the literal parser. The real ``validators.Number`` applied element
by element is reported on its own line, labelled for what it is -- an
interpreter-overhead measurement, not a competitor.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_cyclopts as mcp  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _outcome(fn):
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    return None


def bench_number_batch(n: int = 1 << 22):
    """One batched constraint check against a vectorised NumPy expression.

    The NumPy side is the fastest fair equivalent: the same three comparisons
    as temporaries over the same buffer. Both are memory-bound, so the honest
    expectation is parity -- but the kernel reads the input once and stops at
    no particular point, where NumPy materialises three boolean temporaries.
    """
    rng = np.random.default_rng(0)
    vals = rng.integers(-1000, 1001, size=n).astype(np.float64)
    validator = mcp.Number(gte=0, lte=1000, modulo=5)

    def numpy_side():
        return (vals < 0) | (vals > 1000) | (vals % 5 != 0)

    def mojo_side():
        return validator.first_violation(vals)

    first = int(np.flatnonzero(numpy_side())[0])
    hit = mojo_side()
    assert hit is not None, "test data should contain violations"
    assert hit[0] == first, (hit, first)
    assert 0 <= hit[1] <= 5
    return f"Number batch n={n}", _time(numpy_side, 3), _time(mojo_side, 3)


def bench_number_python_loop(n: int = 20000):
    """The real cyclopts validator applied element by element, against the
    shim's single batched call. This is the shape cyclopts actually uses for a
    container-annotated parameter, and it is dominated by interpreter
    overhead, so the large ratio says nothing about the kernel -- it is a
    separate line for exactly that reason."""
    from cyclopts.validators import Number as UpstreamNumber

    values = [float((i % 201) * 5) for i in range(n)]
    theirs = UpstreamNumber(gte=0, lte=1000, modulo=5)
    mine = mcp.Number(gte=0, lte=1000, modulo=5)

    def run(fn):
        try:
            fn(int, values)
        except ValueError:
            pass

    assert _outcome(lambda: run(theirs)) == _outcome(lambda: run(mine))
    return f"Number loop (py overhead) n={n}", _time(lambda: run(theirs), 3), _time(
        lambda: run(mine), 3
    )


def bench_float_tokens(count: int = 200000):
    """Scalar token parsing against CPython's C-level ``float``.

    The baseline is ``float`` itself, a mature correctly-rounded strtod. One
    token is a few dozen byte operations, so the ctypes call boundary is the
    dominant cost on this side and the kernel loses. Reported as measured.
    """
    tokens = [
        f"{i % 1000}.{(i * 7919) % 1000000:06d}e{(i % 13) - 6:+d}" for i in range(count)
    ]
    for t in tokens[:1000]:
        assert mcp.to_float(t) == float(t), t

    def cpython():
        return [float(t) for t in tokens]

    def mojo():
        return [mcp.to_float(t) for t in tokens]

    return f"float tokens (scalar) x{count}", _time(cpython, 3), _time(mojo, 3)


def bench_float_batch(count: int = 200000):
    """The same tokens through the batched entry point, where the call
    boundary is paid once instead of once per token. This is the case the
    kernel is actually for."""
    tokens = [
        f"{i % 1000}.{(i * 7919) % 1000000:06d}e{(i % 13) - 6:+d}" for i in range(count)
    ]
    values, resolved = mcp.parse_floats(tokens)
    assert resolved.all(), "every benchmark token is inside the kernel envelope"
    expect = [float(t) for t in tokens]
    assert list(values) == expect, "batch parse must match CPython exactly"

    def cpython():
        return [float(t) for t in tokens]

    def mojo():
        return mcp.parse_floats(tokens)

    return f"float tokens (batch) x{count}", _time(cpython, 3), _time(mojo, 3)


def bench_int_batch(count: int = 200000):
    tokens = [str((i * 2654435761) % (1 << 40)) for i in range(count)]
    values, resolved = mcp.parse_ints(tokens)
    assert resolved.all()
    assert [int(v) for v in values] == [int(t) for t in tokens]

    def cpython():
        return [int(t) for t in tokens]

    def mojo():
        return mcp.parse_ints(tokens)

    return f"int tokens (batch) x{count}", _time(cpython, 3), _time(mojo, 3)


def bench_slice_indices(count: int = 200000):
    """``slice.indices`` over a stream of triples, against the interpreter's own
    implementation -- the same C routine Python reaches internally."""
    rng = np.random.default_rng(3)
    a = rng.integers(-20, 20, size=count)
    b = rng.integers(-20, 20, size=count)
    c = rng.integers(-5, 5, size=count)
    keep = c != 0
    a, b, c = a[keep], b[keep], c[keep]
    n = a.size

    def expect(s):
        start, stop, step = s.indices(20)
        return start, stop, step, len(range(start, stop, step))

    for i in range(1000):
        s = slice(int(a[i]), int(b[i]), int(c[i]))
        assert mcp.slice_indices(s, 20) == expect(s), s

    def cpython():
        out = []
        for i in range(n):
            s = slice(int(a[i]), int(b[i]), int(c[i]))
            start, stop, step = s.indices(20)
            out.append((start, stop, step, len(range(start, stop, step))))
        return out

    def mojo():
        return [
            mcp.slice_indices(slice(int(a[i]), int(b[i]), int(c[i])), 20)
            for i in range(n)
        ]

    return f"slice.indices x{n}", _time(cpython, 3), _time(mojo, 3)


CASES = (
    ("number batch vs numpy", bench_number_batch),
    ("number loop (py overhead)", bench_number_python_loop),
    ("float tokens scalar vs float()", bench_float_tokens),
    ("float tokens batch vs float()", bench_float_batch),
    ("int tokens batch vs int()", bench_int_batch),
    ("slice.indices vs python", bench_slice_indices),
)


def main():
    print(f"{'case':<36}{'reference':>12}{'mojo-cyclopts':>16}{'ratio':>9}")
    print("-" * 73)
    for _, fn in CASES:
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<36}{ref * 1e3:>10.2f}ms{got * 1e3:>14.2f}ms{ratio:>8.2f}x")


if __name__ == "__main__":
    main()
