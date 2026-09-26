"""mojo-cyclopts: the numeric surface of cyclopts, with Mojo kernels.

cyclopts itself is a command-line argument parser. This package ports the part
of it that is arithmetic rather than plumbing -- numeric constraint validation,
slice range analysis, and the ASCII-to-number literal conversion every token
goes through -- into ``src/kernels.mojo``, and keeps cyclopts' own signatures
and error messages.

It installs alongside the real ``cyclopts`` package, which the parity tests
import to compare against. Argument binding, help rendering, shell
completion, config files, env-var plumbing and every other non-numeric part of
cyclopts are not ported; use the real package for those.
"""

from .convert import (
    parse_floats,
    parse_ints,
    round_half_even,
    to_float,
    to_int,
)
from .validators import Number, Slice, selects_nothing, slice_indices

__all__ = [
    "Number",
    "Slice",
    "parse_floats",
    "parse_ints",
    "round_half_even",
    "selects_nothing",
    "slice_indices",
    "to_float",
    "to_int",
]
__version__ = "0.1.0"
