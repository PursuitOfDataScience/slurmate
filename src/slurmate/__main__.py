"""Allow ``python -m slurmate`` alongside the ``slurmate`` console script.

Without this module the package could only be started through the entry point
installed by pip; ``python -m slurmate`` failed with "No module named
slurmate.__main__", which is the natural way to run a tool from a source
checkout or a venv whose bin dir isn't on PATH.
"""
from __future__ import annotations

from .main import main

if __name__ == "__main__":
    main()
