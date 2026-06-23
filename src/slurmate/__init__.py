from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    # The installed package metadata (pyproject's `version`) is the single
    # source of truth, so `slurmate --version` can never drift from PyPI.
    __version__ = version("slurmate")
except PackageNotFoundError:
    # Running from a source checkout that was never installed (e.g. some CI or
    # editable-build edge cases). Fall back to a sentinel rather than crashing.
    __version__ = "0.0.0+unknown"
