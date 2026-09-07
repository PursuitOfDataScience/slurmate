"""The interactive stack is not paid for by a run that never draws a prompt.

`slurmate --print` in a job script, `--dry-run` in CI, `--yes` from cron and
`--version` from a wrapper all imported `prompt_toolkit`, because `main` pulled
`slurmate.tui` in at module scope.  Measured with ``-X importtime``:

    prompt_toolkit.key_binding   269 ms   of slurmate.main's 364 ms
    heavy modules loaded         176

Startup, median of three: **0.46 s -> 0.16 s**.  The suite's own wall time went
from ~61 s to ~35 s with it, for the same reason.

`questionary` and `theme.questionary_style` were already deferred on exactly this
argument; the wizard class and one `KeyBindings` type annotation were the two
that were not.  Nothing about the interactive path changes -- it imports what it
needs at the moment it needs it.
"""

from __future__ import annotations

import subprocess
import sys

#: Packages that exist only to draw an interactive terminal UI.
#:
#: `rich` is deliberately absent: it renders the summary panel and the script
#: preview, which every mode prints, so it is not interactive-only.
INTERACTIVE_ONLY = ("prompt_toolkit", "questionary")


def _modules_after(argv: list[str]) -> set[str]:
    """Top-level package names loaded by running slurmate with ``argv``."""
    code = (
        f"import sys, json;"
        f"sys.argv = ['slurmate'] + {argv!r};"
        f"import slurmate.main;"
        f"print(json.dumps(sorted({{m.split('.')[0] for m in sys.modules}})))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    import json

    return set(json.loads(done.stdout))


class TestTheNonInteractivePathStaysLight:
    def test_importing_main_does_not_load_the_interactive_stack(self):
        loaded = _modules_after(["--version"])
        found = sorted(p for p in INTERACTIVE_ONLY if p in loaded)
        assert not found, (
            f"importing slurmate.main loaded {found}; the wizard's dependencies "
            f"belong behind the branch that needs them"
        )

    def test_the_names_are_still_reachable_where_they_are_used(self):
        # The control: deferring an import must not make it unreachable. Both of
        # these are imported inside the branch that needs them.
        from slurmate.tui import Wizard, _parse_custom_flags

        assert Wizard.__name__ == "Wizard"
        assert _parse_custom_flags("--exclusive") == ["--exclusive"]

    def test_main_does_not_import_prompt_toolkit_at_module_scope(self):
        """Read from the source, so a future edit that re-adds it fails here.

        A runtime check alone would pass as long as *some* other module got
        there first, which is how this regressed the first time.
        """
        import ast
        import pathlib

        main_py = pathlib.Path(__file__).resolve().parent.parent / "src" / "slurmate" / "main.py"
        tree = ast.parse(main_py.read_text())
        offenders = []
        for node in tree.body:  # module scope only, not inside functions
            if isinstance(node, ast.Import):
                offenders += [
                    a.name for a in node.names if a.name.split(".")[0] in INTERACTIVE_ONLY
                ]
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in INTERACTIVE_ONLY:
                    offenders.append(node.module or "")
                # `.tui` transitively imports prompt_toolkit, so it counts too.
                if node.level and (node.module or "") == "tui":
                    offenders.append(".tui")
        assert not offenders, f"module-scope interactive imports in main.py: {offenders}"

    def test_the_wizard_path_still_gets_its_dependency(self):
        # Not a measurement of speed -- a check that the deferral is a deferral
        # and not a removal.
        loaded = _modules_after(["--version"])
        assert "prompt_toolkit" not in loaded
        code = "import sys;import slurmate.tui;print('prompt_toolkit' in sys.modules)"
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert done.stdout.strip() == "True", done.stderr
