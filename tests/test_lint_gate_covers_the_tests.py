"""The lint gate checked `src/` only, and four sibling repos check their tests.

Measured across the family: `nodetop` runs `ruff check src tests`, and `rapidu`,
`slurmpast` and `slurmwatch` all run `ruff check .`, which includes `tests/`.
This repo ran `ruff check src/`. So twelve violations had accumulated in test
files where nothing would ever see them -- four pre-existing, eight in files
added by recent work -- and a fix that reformats a test could introduce more
without the gate noticing.

All twelve are now fixed and the gate covers `tests/`. This test pins the scope,
because the drift it is guarding against is silent: narrowing the workflow back
to `src/` breaks nothing that anyone would notice until violations pile up again.

**The type gate is deliberately NOT widened here.** Measured: `mypy src/ tests/`
reports **2360 errors in 25 files** on this suite, because the tests are largely
unannotated. Three siblings do check their tests with mypy, so the gap is real --
but closing it is a project of its own, not a lint fix, and pretending otherwise
by adding `--ignore-errors` or a blanket ignore would make the gate lie.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys


def _workflow() -> str:
    root = pathlib.Path(__file__).resolve().parent.parent
    return (root / ".github" / "workflows" / "ci.yml").read_text()


class TestTheGateSeesTheTests:
    def test_the_workflow_lints_the_tests_directory(self) -> None:
        line = next(ln for ln in _workflow().splitlines() if re.search(r"run:\s*ruff check", ln))
        assert "tests" in line, line

    def test_the_tests_actually_pass_that_gate(self) -> None:
        """Not just configured -- clean. A red gate is worse than a narrow one."""
        root = pathlib.Path(__file__).resolve().parent.parent
        done = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "src/", "tests/"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stdout + done.stderr


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_the_workflow_still_lints_src(self) -> None:
        line = next(ln for ln in _workflow().splitlines() if re.search(r"run:\s*ruff check", ln))
        assert "src" in line, line

    def test_the_type_gate_is_still_scoped_to_src(self) -> None:
        # Deliberate, and measured: `mypy src/ tests/` reports 2360 errors here.
        line = next(ln for ln in _workflow().splitlines() if re.search(r"run:\s*mypy", ln))
        assert "src" in line and "tests" not in line, line

    def test_src_is_still_clean_on_its_own(self) -> None:
        root = pathlib.Path(__file__).resolve().parent.parent
        done = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "src/"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stdout
