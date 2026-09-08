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

**There are two gates, and the first pass widened only one of them.** GitHub
cannot express `needs:` across workflow files, so `release.yml` carries its own
copy of the lint/type/test gate rather than depending on `ci.yml` -- and it is
the copy that stands in front of an irreversible PyPI upload. `ci.yml` was
widened and `release.yml` was left at `ruff check src/`, which means the stricter
scope applied to every push *except* the one that ships. The parametrisation
below reads both files, so narrowing either one reddens a test; the earlier
version of this module hardcoded `ci.yml` and so stayed green through exactly
the omission it was written to prevent.

**The type gate is deliberately NOT widened here.** Measured: `mypy src/ tests/`
reports **2360 errors in 25 files** on this suite, because the tests are largely
unannotated. Three siblings do check their tests with mypy, so the gap is real --
but closing it is a project of its own, not a lint fix, and pretending otherwise
by adding `--ignore-errors` or a blanket ignore would make the gate lie. The
control below pins that decision in both workflows.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every workflow that runs the gate. Both, or the pin only covers half of it.
WORKFLOWS = ["ci.yml", "release.yml"]


def _gate_lines(workflow: str, tool: str) -> list[str]:
    """The `run:` lines in `workflow` that invoke `tool`, which must exist."""
    text = (ROOT / ".github" / "workflows" / workflow).read_text()
    lines = [ln for ln in text.splitlines() if re.search(rf"run:\s*{tool}\b", ln)]
    assert lines, f"{workflow} runs no `{tool}` step at all"
    return lines


class TestTheGateSeesTheTests:
    @pytest.mark.parametrize("workflow", WORKFLOWS)
    def test_the_workflow_lints_the_tests_directory(self, workflow: str) -> None:
        for line in _gate_lines(workflow, "ruff check"):
            assert "tests" in line, f"{workflow}: {line}"

    def test_the_tests_actually_pass_that_gate(self) -> None:
        """Not just configured -- clean. A red gate is worse than a narrow one."""
        done = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "src/", "tests/"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stdout + done.stderr


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    @pytest.mark.parametrize("workflow", WORKFLOWS)
    def test_the_workflow_still_lints_src(self, workflow: str) -> None:
        for line in _gate_lines(workflow, "ruff check"):
            assert "src" in line, f"{workflow}: {line}"

    @pytest.mark.parametrize("workflow", WORKFLOWS)
    def test_the_type_gate_is_still_scoped_to_src(self, workflow: str) -> None:
        # Deliberate, and measured: `mypy src/ tests/` reports 2360 errors here.
        for line in _gate_lines(workflow, "mypy"):
            assert "src" in line and "tests" not in line, f"{workflow}: {line}"

    def test_src_is_still_clean_on_its_own(self) -> None:
        done = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "src/"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stdout
