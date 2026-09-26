# mojo-cyclopts

The compute-oriented subset of [cyclopts](https://cyclopts.readthedocs.io/), a
command-line argument parser, with the arithmetic moved into Mojo and callable
from Python.

**cyclopts has no array-processing core.** It is a parser: it binds a Python
signature to `argv`, converts tokens to annotated types, renders help, and
raises nicely formatted errors. Almost none of that is a loop over numbers.
What *is* arithmetic is small and specific, and this package ports exactly
that rather than inventing work to look complete:

- the **numeric constraint ladder** in `validators.Number` — a five-rule
  comparison ladder (`<`, `<=`, `>`, `>=`, modulo) that has to reject NaN,
- the **integer range-frame analysis** in `validators.Slice` that decides
  whether a `slice` provably selects nothing for *any* sequence length,
- **CPython's `slice.indices`**, reproduced exactly on int64,
- the **ASCII-to-number literal parser** (`_convert._int`, `float`) — a
  byte-at-a-time walk over the token: sign, `0x`/`0o`/`0b` prefix, `_`
  separators, decimal point, exponent.

The Python package is `mojo_cyclopts`, so it installs alongside the real
`cyclopts` and the tests import both and compare them directly.

```python
import mojo_cyclopts as mcp

mcp.to_int("0xdeadbeef")            # 3735928559
mcp.to_float("1.000000000000000000")  # 1.0
mcp.selects_nothing(slice(5, 2))    # True
mcp.slice_indices(slice(-9, 9, 2), 10)  # (0, 9, 2, 5)

v = mcp.Number(gte=0, lte=150, modulo=5)
v.first_violation([1, 2, 3, 4, 5, 6])  # (5, 5) -> 6 fails "multiple of 5"
v(int, [1, 2, 3])                    # ok
v(int, [1, 2, 99])                   # ValueError: Must be <= 150.
```

## Covered subset

| area | ported API | what the kernel does |
| --- | --- | --- |
| Numeric validation | `Number` (`lt`, `lte`, `gt`, `gte`, `modulo`), `Number.first_violation` | `co_number_check` walks a contiguous run of values once, applies the five comparisons with the same negation upstream uses, and returns the first violating index plus the rule that rejected it |
| Slice emptiness | `Slice(allow_empty=...)`, `selects_nothing` | `co_slice_selects_nothing` reproduces the sign-frame / unbounded-end / zero-step analysis on int64 |
| Slice resolution | `slice_indices` | `co_slice_resolve` is CPython's `slice_adjust_indices` plus the stepped range-length formula |
| Literal parsing | `to_int`, `to_float`, `parse_ints`, `parse_floats` | `co_parse_int`, `co_parse_float` and their batched forms scan the token bytes and accumulate the value |
| Rounding | `round_half_even` | `co_round_half_even` applies Python's `round(float)` (ties to even) over a run |

Not implemented, and not ported on purpose: `App`/`run` and command binding,
help and usage rendering, shell completion, config-file and environment
variable sources, `Parameter`/`Group`/`Argument` metadata and `bind()`, the
`LimitedChoice` group validator, the `Path` validator, the `bool`/`None`/
`datetime`/`timedelta`/`Enum`/`Literal`/`Union`/`Path`/`bytes` converters, and
the `CoercionError`/`ValidationError` exception hierarchy. All of that is
plumbing, control flow and string handling. Use the real `cyclopts` for it —
this package does not shadow or replace it.

### Deliberate delegations

Three things the shim hands back to the real implementation rather than
guessing at, because a wrong guess would be silently wrong rather than loudly
wrong:

- an **integer literal outside int64** and any **malformed token** go to
  `cyclopts._convert._int`, so the `ValueError` text is upstream's;
- a **float literal outside the exact envelope** goes to CPython `float`. The
  envelope is: at most 19 significant digits, mantissa at most 2**53 after
  trailing zeros are stripped, and `|decimal exponent| <= 22`. Inside it the
  single multiply or divide by an exactly-representable power of ten is
  correctly rounded, so the result is *bit-identical* to a full `strtod`.
  Outside it the kernel reports "unresolved" instead of approximating;
- an **int wider than 2**53** in a `Number` batch is checked with Python's
  arbitrary-precision comparisons, because the kernel works in float64.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` compiles `src/kernels.mojo` into
`dist/libmojo-cyclopts.so`. Set `PYTHONPATH=python` when using the package
outside a Pixi task. `pixi run bench` runs the benchmark below.

## Performance

Best-of-N wall clock in one process. Every case checks agreement with the real
`cyclopts` or with CPython first. Numbers are from one run on a shared
36-core box, so expect tens of percent of run-to-run noise.

| case | reference | mojo-cyclopts | result |
| --- | ---: | ---: | ---: |
| `Number` batch, 4.2M values vs vectorised NumPy | 177.83 ms | 55.16 ms | 3.22x faster |
| `Number` element-by-element, 20k, vs real cyclopts | 63.72 ms | 11.12 ms | 5.73x faster |
| float tokens, scalar, 200k, vs `float()` | 71.65 ms | 3467.05 ms | 0.02x — 48x slower |
| float tokens, batched, 200k, vs `float()` | 67.82 ms | 288.34 ms | 0.24x — 4.3x slower |
| int tokens, batched, 200k, vs `int()` | 58.68 ms | 193.82 ms | 0.30x — 3.3x slower |
| `slice.indices`, 180k, vs CPython | 584.65 ms | 2029.56 ms | 0.29x — 3.5x slower |

Read these honestly:

- The **`Number` batch** is the one real win. The kernel reads the input once
  and keeps a running first-violation, where NumPy materialises three boolean
  temporaries of the same size.
- The **`Number` loop** line measures interpreter overhead, not kernel speed.
  Upstream walks five Python-level comparisons per element; the shim flattens
  the container once and makes one call. It says nothing about the kernel.
- **The literal parsers lose to CPython, and they are expected to.** A token is
  a few dozen byte operations, and a ctypes call boundary costs about 8 us on
  this box, so a single token can never win. Batching the tokens into one
  buffer is what makes the kernel usable at all (0.02x -> 0.24x), and it still
  loses to `strtod`, which is a mature, table-driven, correctly-rounded
  implementation. The value of the port here is behavioural, not throughput:
  exact `slice.indices` semantics, NaN-rejecting validation, and a parse whose
  result is provably the correctly-rounded one inside its envelope.
- **`slice.indices` loses for the same reason**: 0.7 us inside CPython against
  an 8 us ctypes round trip. The kernel's value is that it is a faithful
  reimplementation, not that it is faster.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-cyclopts.so`.

The `python/mojo_cyclopts` layer owns every array. It normalises inputs to
contiguous `float64` (or a packed `uint8` blob for tokens), then makes one call
into the kernel. Buffers cross the C ABI as 64-bit addresses and are
reconstructed in Mojo as `Pointer[T, AnyOrigin[mut=True]]`, which keeps the
exported symbols non-parametric — `@export` rejects a parametric function, and
an inferred pointer origin would make it one.

The `co_number_check` kernel returns the first violating index *and* the rule
that rejected it, in the order upstream checks them (`lt`, `lte`, `gt`, `gte`,
`modulo`). The shim maps that index back to the caller's element order, so a
container whose leaves are a mix of narrow floats and wide ints still reports
the first failure in the caller's order. The rule code also selects the error
message, so the text is upstream's verbatim, including `Must be a multiple of
N.` for the modulo rule.

Mojo emits FMA, so in general a result matches NumPy only to a tolerance. That
does not apply to the kernels here: none of them is a fused dot product, and
the float parser's correctness rests on single correctly-rounded operations,
not on accumulation order. The parity tests therefore assert **exact**
equality — `values[i] == float(token)`, including the sign of zero — and that
is the claim the implementation actually makes.

## Tests

```
178 passed
```

- `tests/test_number.py` — the constraint ladder against
  `cyclopts.validators.Number`, scalar and container, including the exact
  `ValueError` text, NaN rejection, the rule-reporting order, and ints beyond
  float64.
- `tests/test_slice.py` — `selects_nothing` against
  `cyclopts.validators._slice._selects_nothing` over an exhaustive 486-case
  grid, and `slice_indices` against CPython's `slice.indices` over that grid at
  six sequence lengths (2916 cases).
- `tests/test_convert.py` — the scalar literal parsers against
  `cyclopts._convert._int` and CPython `float`, bit-for-bit.
- `tests/test_batch_parse.py` — the batched parsers, plus assertions that the
  kernel itself (not the fallback) resolved the common tokens, so a completely
  broken kernel cannot pass behind the delegation.

## License

MIT
