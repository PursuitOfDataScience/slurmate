from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    # The INSTALLED package metadata, which is the single source of truth for an
    # installed slurmate and therefore cannot drift from PyPI.
    #
    # It can, however, differ from the source tree it is imported from: this
    # reads the dist-info of whatever is installed, not this checkout's
    # `pyproject.toml`. Measured on this cluster -- a 0.7.0 tree run with
    # `python -m slurmate` against a 0.5.2 install reports **0.5.2**. That is
    # standard `importlib.metadata` behaviour and the number is true of the
    # metadata; it is simply not a claim about which code is executing. Said
    # plainly here because the previous wording ("can never drift") promised
    # something this mechanism does not provide, and a version string is exactly
    # the field a reader trusts without checking.
    __version__ = version("slurmate")
except PackageNotFoundError:
    # Not installed at all. Note this is the ONLY case that reaches the sentinel:
    # an installed-but-different version raises nothing and returns confidently.
    __version__ = "0.0.0+unknown"
