"""Port of ``cyclopts.validators`` numeric validators onto the Mojo kernels.

The validators here keep cyclopts' call signature -- ``__call__(type_, value)``
raising ``ValueError`` with the same message -- but the comparison ladder that
decides which constraint a run of values violates, and the integer range-frame
analysis that backs ``Slice``, run in ``src/kernels.mojo``.

The public field names, defaults and semantics match
``cyclopts.validators.Number`` and ``cyclopts.validators.Slice``, so a caller
can swap one for the other.
"""

from __future__ import annotations

import array
import collections.abc
from typing import Any

import numpy as np

from . import _lib

__all__ = ["Number", "Slice", "selects_nothing", "slice_indices"]

# A three-slot scratch buffer for slice_resolve, so the common per-call path
# does not allocate. An array.array of int64 rather than a NumPy array because
# reading a NumPy scalar element costs several microseconds, which would
# dominate the call it is trying to measure. The kernel writes all three slots
# before returning and the values are copied out immediately.
_RESOLVE_BUF = array.array("q", [0, 0, 0])
_RESOLVE_ADDR = _RESOLVE_BUF.buffer_info()[0]

# Ints beyond this are not exactly representable as float64, so the kernel's
# comparisons would stop matching Python's arbitrary-precision arithmetic and
# those elements are checked in Python instead.
_F64_EXACT_INT = 2**53


def _flatten(value: Any, out: list) -> None:
    """Append the scalar leaves of ``value`` to ``out``.

    Mirrors upstream's recursive descent through ``iter_container_elements``:
    mappings descend into their values, other containers into their items, and
    a ``str`` anywhere in the tree raises ``TypeError`` because it is a Sequence
    but never a container of values.
    """
    cls = type(value)
    if cls is list or cls is tuple:
        # Fast path: a list or tuple of numbers is the overwhelmingly common
        # shape, and collections.abc isinstance checks are far more expensive
        # per element than a concrete type comparison.
        for v in value:
            vt = type(v)
            if vt is int or vt is float:
                out.append(v)
            else:
                _flatten(v, out)
        return
    if isinstance(value, collections.abc.Mapping):
        for v in value.values():
            _flatten(v, out)
        return
    if isinstance(value, collections.abc.Sequence | collections.abc.Set):
        if isinstance(value, str):
            raise TypeError("str is not a container of values")
        for v in value:
            _flatten(v, out)
        return
    out.append(value)


class Number:
    """Limit input number to a value range.

    Same contract as ``cyclopts.validators.Number``: every leaf of a
    container-annotated parameter is validated individually (and a mapping's
    *values*, not its keys), a non-numeric leaf is silently accepted, and NaN
    cannot slip past a bound because the comparison is negated exactly the way
    upstream negates it.
    """

    __slots__ = ("lt", "lte", "gt", "gte", "modulo")

    def __init__(
        self,
        *,
        lt: int | float | None = None,
        lte: int | float | None = None,
        gt: int | float | None = None,
        gte: int | float | None = None,
        modulo: int | float | None = None,
    ) -> None:
        self.lt = lt
        self.lte = lte
        self.gt = gt
        self.gte = gte
        self.modulo = modulo

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{k}={v!r}"
            for k, v in (
                ("lt", self.lt),
                ("lte", self.lte),
                ("gt", self.gt),
                ("gte", self.gte),
                ("modulo", self.modulo),
            )
            if v is not None
        )
        return f"Number({parts})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Number):
            return NotImplemented
        return all(getattr(self, f) == getattr(other, f) for f in self.__slots__)

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, f) for f in self.__slots__))

    def __call__(self, type_: Any, value: Any) -> None:
        leaves: list = []
        _flatten(value, leaves)
        # A single batch for the whole container: a wide list becomes one pass
        # over a contiguous buffer instead of a Python loop.
        self._check_many(leaves)

    # -- batch API ---------------------------------------------------------

    def first_violation(self, values) -> tuple[int, int] | None:
        """Validate a 1-D run of numbers in a single kernel call.

        This is the entry point for bulk validation: hand it a NumPy array (or
        any sequence of int/float) and it returns ``(index, rule)`` for the
        first value that violates a bound, or ``None`` when the whole run is
        valid. ``rule`` is the same code the kernel reports, mapping to the
        field names in :data:`mojo_cyclopts._lib.RULE_NAMES`.

        Unlike ``__call__`` this does not recurse into containers and does not
        raise; it answers the question directly, which is what a caller with
        megabytes of values actually wants.
        """
        if self.modulo == 0:
            raise ZeroDivisionError("integer modulo by zero")
        arr = np.ascontiguousarray(values, dtype=np.float64)
        first_bad = np.empty(1, dtype=np.int64)
        first_rule = np.empty(1, dtype=np.int32)
        _lib.lib.co_number_check(
            arr.ctypes.data,
            arr.size,
            1 if self.lt is not None else 0,
            float("inf") if self.lt is None else float(self.lt),
            1 if self.lte is not None else 0,
            float("inf") if self.lte is None else float(self.lte),
            1 if self.gt is not None else 0,
            float("inf") if self.gt is None else float(self.gt),
            1 if self.gte is not None else 0,
            float("inf") if self.gte is None else float(self.gte),
            1 if self.modulo is not None else 0,
            float("inf") if self.modulo is None else float(self.modulo),
            first_bad.ctypes.data,
            first_rule.ctypes.data,
        )
        k = int(first_bad[0])
        return None if k < 0 else (k, int(first_rule[0]))

    # -- internals ---------------------------------------------------------

    def _check_many(self, values: list) -> None:
        if self.modulo == 0:
            # Upstream reaches `value % 0` and raises ZeroDivisionError.
            raise ZeroDivisionError("integer modulo by zero")

        numeric: list[float] = []
        numeric_pos: list[int] = []
        wide: list[tuple[int, int]] = []
        for i, v in enumerate(values):
            if isinstance(v, int):
                if -_F64_EXACT_INT <= v <= _F64_EXACT_INT:
                    numeric.append(float(v))
                    numeric_pos.append(i)
                else:
                    wide.append((i, v))
            elif isinstance(v, float):
                numeric.append(v)
                numeric_pos.append(i)
        # Anything else is not int|float, so upstream returns silently.

        arr = np.ascontiguousarray(numeric, dtype=np.float64)
        first_bad = np.empty(1, dtype=np.int64)
        first_rule = np.empty(1, dtype=np.int32)
        _lib.lib.co_number_check(
            arr.ctypes.data,
            arr.size,
            1 if self.lt is not None else 0,
            float("inf") if self.lt is None else float(self.lt),
            1 if self.lte is not None else 0,
            float("inf") if self.lte is None else float(self.lte),
            1 if self.gt is not None else 0,
            float("inf") if self.gt is None else float(self.gt),
            1 if self.gte is not None else 0,
            float("inf") if self.gte is None else float(self.gte),
            1 if self.modulo is not None else 0,
            float("inf") if self.modulo is None else float(self.modulo),
            first_bad.ctypes.data,
            first_rule.ctypes.data,
        )
        hit: tuple[int, int] | None = None
        k = int(first_bad[0])
        if k >= 0:
            hit = (numeric_pos[k], int(first_rule[0]))
        for pos, v in wide:
            rule = self._rule_for(v)
            if rule and (hit is None or pos < hit[0]):
                hit = (pos, rule)
        if hit is not None:
            field = _lib.RULE_NAMES[hit[1]]
            raise ValueError(
                _lib.RULE_MESSAGES[hit[1]].format(bound=getattr(self, field))
            )

    def _rule_for(self, value: Any) -> int:
        """Python mirror of the kernel ladder, for values the kernel declines.

        Only used for ints beyond the exact float64 range, where Python's
        arbitrary-precision comparison and modulo are the correct reference.
        """
        if self.lt is not None and not value < self.lt:
            return 1
        if self.lte is not None and not value <= self.lte:
            return 2
        if self.gt is not None and not value > self.gt:
            return 3
        if self.gte is not None and not value >= self.gte:
            return 4
        if self.modulo is not None and value % self.modulo:
            return 5
        return 0


class Slice:
    """Assertions on properties of a :class:`slice`.

    Same contract as ``cyclopts.validators.Slice``: with ``allow_empty=False``
    a slice that provably selects nothing is rejected. The "provably" analysis
    -- sign frames, unbounded ends, the zero step -- is the Mojo kernel's
    ``co_slice_selects_nothing``.
    """

    __slots__ = ("allow_empty",)

    def __init__(self, *, allow_empty: bool = True) -> None:
        self.allow_empty = allow_empty

    def __repr__(self) -> str:
        return f"Slice(allow_empty={self.allow_empty!r})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Slice):
            return NotImplemented
        return self.allow_empty == other.allow_empty

    def __hash__(self) -> int:
        return hash((self.allow_empty,))

    def __call__(self, type_: Any, value: Any) -> None:
        leaves: list = []
        _flatten(value, leaves)
        for v in leaves:
            if not isinstance(v, slice):
                continue
            if not self.allow_empty and selects_nothing(v):
                raise ValueError("Slice must select a non-empty range.")


def selects_nothing(s: slice) -> bool:
    """``cyclopts.validators._slice._selects_nothing``, via the Mojo kernel."""
    has_start = 0 if s.start is None else 1
    has_stop = 0 if s.stop is None else 1
    has_step = 0 if s.step is None else 1
    return bool(
        _lib.lib.co_slice_selects_nothing(
            has_start, int(s.start) if s.start is not None else 0,
            has_stop, int(s.stop) if s.stop is not None else 0,
            has_step, int(s.step) if s.step is not None else 1,
        )
    )


def slice_indices(s: slice, length: int) -> tuple[int, int, int, int]:
    """``slice.indices(length)`` plus the number of elements it selects.

    Reproduces CPython's ``slice_adjust_indices`` exactly over the int64 range;
    a length or bound outside that range is delegated to the interpreter.
    """
    lo, hi = -(2**63), 2**63 - 1
    for v in (length, s.start, s.stop, s.step):
        if v is not None and not (lo <= v <= hi):
            start, stop, step = s.indices(length)
            return start, stop, step, len(range(start, stop, step))
    count = _lib.lib.co_slice_resolve(
        0 if s.start is None else 1, 0 if s.start is None else int(s.start),
        0 if s.stop is None else 1, 0 if s.stop is None else int(s.stop),
        0 if s.step is None else 1, 1 if s.step is None else int(s.step),
        length, _RESOLVE_ADDR,
    )
    return (_RESOLVE_BUF[0], _RESOLVE_BUF[1], _RESOLVE_BUF[2], int(count))
