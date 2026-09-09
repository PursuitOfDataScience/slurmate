"""No ``console.print`` may hand rich a raw ANSI escape.

H4 found seven `Error:` lines built with `theme.c.RED`/`c.RESET` -- raw ANSI --
and handed to `rich`. Rich does not read ANSI in a `print` argument:
`\x1b[38;2;255;0;0m` is not a markup tag, so rich left the bracketed part as
text, its repr highlighter styled the digits inside it, and the terminal drew a
lone `ESC` followed by the visible characters `[38;2;255;0;0m` -- while the
message itself was never coloured.

`issues.md` closed those seven and named the rest as the next sweep, with the
count and the reason: *"the other ten `c.*`-into-`console.print` sites listed at
the end of L11 have the identical defect. Same severity, out of scope for a round
that was asked for three groups"*, and *"the scan that finds them is three lines
of `ast.walk`"*. All ten are now routed through `_print_issue` (the five that are
`Error:` lines) or `_print_indented` with rich markup (the two `Warning:` lines
and the three dim detail lines), and that scan is the first test below, so an
eleventh site cannot be added quietly.

Two things came free with it. Four of the ten hardcoded `✗` where the other
six use `theme.g.ERR`, which is what `--ascii` acts on -- exactly the defect
`_print_issue`'s docstring records for the sites that hardcoded `⚠`. And
every interpolated value now goes through `rich.markup.escape`, so a field name
or a `--custom-sbatch` value containing `[...]` is shown rather than parsed as
markup.

Invisible to the rest of the suite, which runs under `NO_COLOR=1` or a pipe where
`theme.C.__getattribute__` returns `""` and the line is plain. The interactive
case -- the only one a human sees -- is the broken one, so the behavioural tests
here force colour on.
"""

from __future__ import annotations

import argparse
import ast
import io
import pathlib
import re
import sys

import pytest
from rich.console import Console

from slurmate import main as main_module
from tests.test_literal_indent_sweep import (
    _BATCH_DEFAULTS,
    _as_a_terminal_would_show,
    _batch_stderr,
    _columns,
)

#: The attributes of `theme.c` that resolve to a raw SGR escape.
_ANSI_ATTRS = ("RED", "GREEN", "YELLOW", "BLUE", "CYAN", "GRAY", "DIM", "BOLD", "RESET")

_MAIN = pathlib.Path(main_module.__file__)

_NTASKS_LINE = "  ✗ Error: --ntasks-per-node must be a positive integer (got 'zero')"


def _print_calls_with_raw_ansi(path: pathlib.Path) -> list[tuple[int, str]]:
    """``(line, source)`` for every ``*.print`` call interpolating a raw escape."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        if not ast.unparse(node.func).endswith(".print"):
            continue
        src = ast.unparse(node)
        if any(f"c.{attr}" in src for attr in _ANSI_ATTRS):
            found.append((node.lineno, src))
    return found


def _mentions_raw_ansi(node: ast.AST) -> bool:
    """Does this expression name one of `theme.c`'s raw SGR attributes?

    Includes `getattr(c, "RED")`, because the attribute form is what the scan
    reads and an indirection through `getattr` is the same escape reaching the
    same place.
    """
    text = ast.unparse(node)
    if any(f"c.{attr}" in text for attr in _ANSI_ATTRS):
        return True
    return any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == "getattr"
        and len(sub.args) >= 2
        and isinstance(sub.args[1], ast.Constant)
        and sub.args[1].value in _ANSI_ATTRS
        for sub in ast.walk(node)
    )


def _raw_ansi_reaching_print_via_a_variable(path: pathlib.Path) -> list[tuple[int, str]]:
    """`(line, name)` where a local holding a raw escape is handed to `*.print`.

    The direct scan above reads the print call's own source, so it sees
    `console.print(f"{c.RED}...")` and misses the one refactor a developer
    actually performs on a line that has grown too long::

        msg = f"  {c.RED}Error: ...{c.RESET}"
        err_console.print(msg)

    Tabulated against the direct scan, four forms escaped it: a variable, a
    `%`-formatted variable, `print(*parts)` where `parts` is a list holding
    `c.RED`, and `getattr(c, "RED")`. The first three are all this one hop, and
    the last is handled by `_mentions_raw_ansi`.

    **One hop, deliberately, and the limit is stated rather than implied.** A
    chain (`a = c.RED; b = a; print(b)`) is not followed: that is dataflow
    analysis, and the point here is to catch the ordinary refactor, not to be a
    type checker. Scoped per function, so a module-level constant named the same
    as a local cannot taint an unrelated call.
    """
    found: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text())
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        tainted: dict[str, int] = {}
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and _mentions_raw_ansi(node.value)
            ):
                tainted.setdefault(node.targets[0].id, node.lineno)
            elif (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                and _mentions_raw_ansi(node.value)
            ):
                tainted.setdefault(node.target.id, node.lineno)
        if not tainted:
            continue
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func).endswith(".print")):
                continue
            used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            found.extend((node.lineno, name) for name in sorted(used & set(tainted)))
    return found


def _painted(width: int, **flags: object) -> str:
    """`_run_batch`'s stderr with colour forced on."""
    args = argparse.Namespace(**{**_BATCH_DEFAULTS, **flags})
    buffer = io.StringIO()
    stashed, sys.stderr = sys.stderr, buffer
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.delenv("NO_COLOR", raising=False)
            patch.setenv("FORCE_COLOR", "1")
            # `theme.c` caches the decision on first access.
            from slurmate import theme

            theme.c.__dict__.pop("_use_color", None)
            with pytest.raises(SystemExit), _columns(width):
                main_module._run_batch(args, Console(file=io.StringIO(), width=width), {})
            theme.c.__dict__.pop("_use_color", None)
    finally:
        sys.stderr = stashed
    return buffer.getvalue()


class TestNoRawEscapeIsHandedToRich:
    def test_the_scan_finds_no_print_call_building_its_own_ansi(self) -> None:
        offenders = _print_calls_with_raw_ansi(_MAIN)
        assert offenders == [], [
            f"main.py:{line} {src[:70]}" for line, src in offenders
        ]

    def test_a_rewritten_line_shows_no_escape_as_text(self) -> None:
        visible = _as_a_terminal_would_show(_painted(80, ntasks_per_node="zero"))
        assert "[38;2;" not in visible, visible
        assert "[0m" not in visible, visible
        assert visible.splitlines()[0] == _NTASKS_LINE, visible.splitlines()[0]

    def test_a_rewritten_line_is_actually_red(self) -> None:
        # "carries some SGR" is not enough, which is the trap the H4 test
        # recorded: before the fix the line carried rich's own highlighter
        # colours around the digits it found inside the escape, while the
        # sentence stayed uncoloured. The sentence has to be red.
        first = _painted(80, ntasks_per_node="zero").splitlines()[0]
        assert "\x1b[31m" in re.findall(r"\x1b\[[\d;]*m", first), repr(first)


class TestControls:
    """These hold whether or not the ten sites were rewritten."""

    def test_the_plain_message_text_is_unchanged(self) -> None:
        # The rewrite had to fix the escape without touching what the message
        # says. Under `NO_COLOR` the old code emitted this exact line too,
        # because `c.*` resolves to "" there.
        assert _batch_stderr(80, ntasks_per_node="zero")[0] == _NTASKS_LINE

    def test_the_scan_is_not_a_blanket_ban_on_the_colour_helper(self) -> None:
        """`theme.c` is fine; handing its output to rich unparsed is not.

        Without this, the rule above could be "satisfied" by deleting every use
        of `theme.c`, which is not the fix -- the summary panel and the wizard
        write their own escapes to a plain stream on purpose.
        """
        text = _MAIN.read_text()
        total = len(re.findall(r"\bc\.[A-Z_]+", text))
        in_prints = sum(
            len(re.findall(r"\bc\.[A-Z_]+", src))
            for _, src in _print_calls_with_raw_ansi(_MAIN)
        )
        assert total - in_prints > 0, (total, in_prints)

    def test_the_group_h4_already_fixed_still_renders_clean(self) -> None:
        # Neither the scan nor the rewrite may disturb the seven lines that were
        # routed through `_print_issue` first.
        visible = _as_a_terminal_would_show(_painted(80, memory="not-a-memory-value"))
        assert "[38;2;" not in visible, visible
        assert visible.splitlines()[0] == (
            "  ✗ Error: Invalid memory value: not-a-memory-value"
        ), visible.splitlines()[0]


class TestTheScanSeesTheIndirectRoutesToo:
    """The direct scan reads the print call's own source, so one hop escapes it.

    Tabulated by feeding crafted snippets to the direct scan, which is how the
    holes were found rather than by reading it:

    ==================================  ==========
    form                                direct scan
    ==================================  ==========
    ``print(f"{c.RED}x")``              catches
    ``print(c.RED + "x")``              catches
    ``print(fmt(f"{c.RED}x"))``         catches
    ``msg = f"{c.RED}x"; print(msg)``   MISSES
    ``line = "%s" % c.RED; print(line)``MISSES
    ``parts = [c.RED]; print(*parts)``  MISSES
    ``print(getattr(c, "RED"))``        MISSES
    ==================================  ==========

    The first of the four is the one that matters: extracting a long f-string
    into a local is the ordinary refactor, and it silently reintroduces H4 —
    rich renders the raw escape as visible text and leaves the line uncoloured.
    """

    def _scan(self, body: str, tmp_path: pathlib.Path) -> list[tuple[int, str]]:
        path = tmp_path / "probe.py"
        path.write_text("def f():\n" + "".join(f"    {ln}\n" for ln in body.splitlines()))
        return _raw_ansi_reaching_print_via_a_variable(path)

    ESCAPED = {
        "a variable": 'msg = f"{c.RED}boom{c.RESET}"\nconsole.print(msg)',
        "a percent-formatted variable": 'line = "%s boom" % c.RED\nerr_console.print(line)',
        "star-args from a list": 'parts = [c.RED, "boom"]\nconsole.print(*parts)',
        "getattr": 'msg = getattr(c, "RED") + "boom"\nconsole.print(msg)',
    }

    @pytest.mark.parametrize("form", sorted(ESCAPED))
    def test_a_form_that_escaped_the_direct_scan_is_caught(
        self, form: str, tmp_path: pathlib.Path
    ) -> None:
        assert self._scan(self.ESCAPED[form], tmp_path), form

    @pytest.mark.parametrize(
        "body",
        [
            'msg = "[red]boom[/]"\nconsole.print(msg)',  # markup, which is correct
            'msg = f"{g.ERR} boom"\nconsole.print(msg)',  # a theme glyph, not an escape
            'msg = f"{c.RED}boom"\nlog.write(msg)',  # not a rich print at all
            'msg = f"{c.RED}boom"\nreturn msg',  # never printed here
        ],
    )
    def test_it_does_not_cry_wolf(self, body: str, tmp_path: pathlib.Path) -> None:
        assert self._scan(body, tmp_path) == [], body

    def test_the_taint_does_not_leak_between_functions(self, tmp_path: pathlib.Path) -> None:
        # Scoped per function, so a name that holds an escape in one function
        # cannot implicate a same-named local somewhere else.
        path = tmp_path / "two.py"
        path.write_text(
            'def a():\n    msg = f"{c.RED}boom"\n    return msg\n\n'
            'def b():\n    msg = "[red]fine[/]"\n    console.print(msg)\n'
        )
        assert _raw_ansi_reaching_print_via_a_variable(path) == []

    def test_the_real_source_has_no_indirect_route_either(self) -> None:
        # The direct scan is asserted empty elsewhere in this file; this is the
        # same claim for the hop it could not see.
        offenders = [
            f"{_MAIN.name}:{line} via {name}"
            for line, name in _raw_ansi_reaching_print_via_a_variable(_MAIN)
        ]
        assert offenders == [], offenders

    def test_the_direct_scan_still_works(self) -> None:
        # The control on the widening: the original scan must not have been
        # replaced, only supplemented.
        probe = _MAIN.parent / "main.py"
        assert probe.exists()
        assert _print_calls_with_raw_ansi(_MAIN) == []
