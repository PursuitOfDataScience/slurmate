"""The rest of `main.py`'s literal-indent `console.print` calls.

`issues.md`'s **L11** fixed two sites and then recorded a deferral: three more
groups carry their indent as literal spaces inside the string, but "none of them
interpolates an unbounded list the way this message does, so none has been
observed to wrap". That premise was tested at six widths a real terminal is
(60, 70, 80, 90, 100, 120), by driving each group's own code path, and it is
**false for all three**::

    site                                     wraps at (columns)
    905  --gpus inferred-format hint         60 70 80 90 100 120   (147 cells)
    515  --cpus rejection                    -- (52 cells, + a user number)
    518  --nodes rejection                   -- (52 cells, + a user number)
    521  --gpus rejection                    -- (56 cells, + a user number)
    524  --ntasks-per-node rejection         60                    (63 cells)
    530  Invalid memory value: <user>        any (200-char --mem -> 4 lines @80)
    531  Give MEMORY_FORMS.                  60 70 80 90 100 120   (195 cells)
    541  Invalid --mem-per-cpu: <user>       any
    544  Give MEMORY_FORMS.                  60 70 80 90 100 120
    556  Invalid time limit value: <user>    any
    557  Give time_forms().                  60 70 80 90           (99 cells)
    2451 Slurm rejects the edited script     60 70 80 90 100 120   (+ sbatch's text)
    2453 Choose "Open in editor" ...         60 70                 (80 cells)
    2470 Slurm would not take this job ...   60 80 120             (+ sbatch's text)
    2478 each _hard_errors message           60 80                 (85 cells)
    2479 This job has errors ...             60 70 80              (90 cells)

The `Give ...` lines are the sharpest counter-example: `MEMORY_FORMS` is 195
cells of **fixed** text, so that line wraps at every width in the table --
unconditionally, with nothing user-supplied in it at all. Measured before the
fix, `--mem not-a-memory-value` at 60 columns::

    '  ✗ Error: Invalid memory value: not-a-memory-value'
    '  Give a number of megabytes (`4096`), or a whole number '   <- trailing space
    'with a unit (`16G`, `512M`, `1T`; lower case works, and a '  <- column 0
    'trailing B is fine: `16GB`). `0` means "all the memory on '  <- column 0
    'the node".'                                                 <- column 0

3 lines with trailing whitespace and 3 at column 0; after, 0 and 0 at all six
widths.

**A third defect fell out of the same measurement.** Seven of these sites carried
their colour as a raw `c.RED`/`c.RESET` ANSI escape handed to `rich`.
`[38;2;255;0;0m` is not a markup tag, so rich left it as text, its repr
highlighter styled the digits *inside* it, and the sequence reached the terminal
broken. Interpreting the real output as a terminal would, at `FORCE_COLOR=1` and
80 columns::

    before: '  [38;2;255;0;0m✗ Error: Invalid memory value: ...[0m'   68 cells, not red
    after:  '  ✗ Error: Invalid memory value: ...'                    51 cells, \x1b[31m

So the fix is: every site in the three groups now goes through
`_print_indented`, and the seven `Error:` lines through `_print_issue` (which
composes the same sentence as markup and hands it to `_print_indented`). Two
things come along for free and are pinned below -- the interpolated user value is
escaped (a `--mem '[red]x'` used to inject markup), and the glyph comes from
`theme.g`, so `--ascii` reaches these lines for the first time.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
import types
from collections.abc import Iterator

import pytest
from rich.console import Console

from slurmate import main as main_module

#: The widths L11 measured at, plus 100, which is where `time_forms()` stops
#: wrapping and `MEMORY_FORMS` does not.
WIDTHS = (60, 70, 80, 90, 100, 120)

#: A real permanent refusal, in slurmate's own wording for one.
PERMANENT_REFUSAL = (
    "sbatch rejected an option in the script: Batch job submission failed: "
    "Invalid account or account/partition combination specified"
)
#: A real transient one.
TRANSIENT_REFUSAL = (
    "sbatch rejected an option in the script: Batch job submission failed: "
    "Requested node configuration is not available"
)
#: Two real `_hard_errors` messages, as `validate_job_config` words them.
HARD_ERRORS = [
    "Time limit 3-00:00:00 exceeds the maximum for partition 'cpu-shared' "
    "(1-00:00:00)",
    "Memory 900G exceeds the largest node in 'cpu-shared' (192G)",
]


# --------------------------------------------------------------------------- #
# Drivers. Each one runs the real code path; only the cluster and the human are
# stubbed, so what is measured is what the tool prints.
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _plain_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock cluster, no colour -- so the lines are the lines."""
    monkeypatch.setenv("SLURMATE_MOCK", "1")
    monkeypatch.setenv("NO_COLOR", "1")


#: A job the batch path accepts, so each test can spoil exactly one field.
_BATCH_DEFAULTS: dict[str, object] = {
    "partition": "cpu-shared", "cpus": 4, "time": "1:00:00",
    "job_name": "x", "command": "true",
}


def _batch_stderr(width: int, **flags: object) -> list[str]:
    """`_run_batch`'s stderr at ``width``.

    The real batch path. Every rejection in the 515-557 group fires before the
    first cluster query, so this needs no Slurm and no stubbing beyond the width.
    """
    args = argparse.Namespace(**{**_BATCH_DEFAULTS, **flags})
    buffer = io.StringIO()
    stashed, sys.stderr = sys.stderr, buffer
    try:
        with pytest.raises(SystemExit) as exit_info, _columns(width):
            main_module._run_batch(
                args, Console(file=io.StringIO(), width=width), {}
            )
    finally:
        sys.stderr = stashed
    assert exit_info.value.code == 1
    return buffer.getvalue().splitlines()


@contextlib.contextmanager
def _columns(width: int) -> Iterator[None]:
    """`COLUMNS`, which is what a bare `Console(stderr=True)` sizes itself from."""
    stashed = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = str(width)
    try:
        yield
    finally:
        if stashed is None:
            del os.environ["COLUMNS"]
        else:
            os.environ["COLUMNS"] = stashed


def _gpu_hint_lines(width: int, *, select: str = "select/linear") -> list[str]:
    """The inferred-format hint (`main.py:905`), from `_check_gpu_format` itself.

    `select_type` is the one thing stubbed: all three clusters slurmate has been
    measured on run `select/cons_tres`, which is why this branch has no live
    cluster to exercise it -- the same reason
    `test_cluster_portability.py`'s `FAKE_SELECT_HARNESS` exists.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, width=width, force_terminal=False, no_color=True)
    replacement = None
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(main_module, "fetch_select_type", lambda: select)
        replacement = main_module._check_gpu_format(
            "gpus", gpus_requested=True, force=False, err_console=console,
            inferred=True,
        )
    # Pinned here rather than in a control of its own: the hint is only correct
    # if the substitution it announces actually happened.
    assert replacement == "gres_type"
    return buffer.getvalue().splitlines()


def _menu_lines(
    width: int, *, refusal: str | None = None, permanent: bool = True,
    transient: bool = False, hard_errors: list[str] | None = None,
    actions: tuple[str, ...] = ("Submit to Slurm", "Quit without submitting"),
) -> list[str]:
    """The real action menu in `_main`, scripted.

    `questionary` is imported inside `_main`, so the prompt is replaced on the
    module object; the scheduler verdict and the human's choices are the only
    other stubs. Everything between -- argparse, the wizard bypass,
    `build_and_show`, the loop -- is the tool.
    """
    chosen = iter(actions)

    class _Prompt:
        def __init__(self) -> None:
            # Not a `KeyBindings`, so `_main` leaves the Esc binding alone.
            self.application = types.SimpleNamespace(key_bindings=object())

        def ask(self) -> str | None:
            return next(chosen, None)

    import questionary

    buffer = io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(questionary, "select", lambda *a, **k: _Prompt())
        patch.setattr(questionary, "confirm", lambda *a, **k: _Prompt())
        patch.setattr(main_module, "_edit_script_in_editor", lambda script: script)
        patch.setattr(main_module, "_submit_and_report",
                      lambda *a, **k: print("  <submitted>"))
        if refusal is not None:
            patch.setattr(main_module, "check_script_with_scheduler",
                          lambda script: refusal)
            patch.setattr(main_module, "refusal_is_permanent", lambda r: permanent)
            patch.setattr(main_module, "refusal_is_transient", lambda r: transient)
        if hard_errors is not None:
            patch.setattr(main_module, "_hard_errors", lambda answers: hard_errors)
        patch.setattr(sys, "argv", [
            "slurmate", "--job-name", "x", "--partition", "cpu-shared",
            "--cpus", "1", "--memory", "1G", "--time", "00:05:00",
            "--command", "true",
        ])
        stashed, sys.stdout = sys.stdout, buffer
        try:
            with _columns(width), contextlib.ExitStack() as stack:
                main_module._main(stack)
        finally:
            sys.stdout = stashed
    return buffer.getvalue().splitlines()


def _block(lines: list[str], start: str, stop: str | None = None) -> list[str]:
    """The paragraph(s) beginning at ``start``, up to but excluding ``stop``.

    Scoped deliberately: `main.py` has further literal-indent prints outside
    these three groups (see `issues.md`), and a blanket assertion over all of
    stdout would be measuring them too.
    """
    heads = [i for i, line in enumerate(lines) if start in line]
    assert heads, f"{start!r} never printed; got {lines[-6:]}"
    begin = heads[0]
    end = len(lines)
    if stop is not None:
        for i in range(begin, len(lines)):
            if stop in lines[i]:
                end = i
                break
    return lines[begin:end]


def _trailing(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if line != line.rstrip()]


def _unindented(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if line and not line.startswith(" ")]


def _overwidth(lines: list[str], width: int) -> list[int]:
    return [i for i, line in enumerate(lines) if len(line) > width]


# --------------------------------------------------------------------------- #
# Group 515-557: the batch-mode validation rejections.
# --------------------------------------------------------------------------- #

class TestTheBatchRejectionsAreIndentedThroughout:
    """`main.py:515-557`. Before: 3 trailing-space lines and 3 at column 0 at 60."""

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_memory_rejection_keeps_its_indent_at_every_width(
        self, width: int
    ) -> None:
        lines = _batch_stderr(width, memory="not-a-memory-value")
        assert _trailing(lines) == [], (width, lines)
        assert _unindented(lines) == [], (width, lines)
        assert _overwidth(lines, width) == [], (width, lines)

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_time_rejection_keeps_its_indent_at_every_width(
        self, width: int
    ) -> None:
        lines = _batch_stderr(width, time="1:0:0:0")
        assert _trailing(lines) == [], (width, lines)
        assert _unindented(lines) == [], (width, lines)

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_mem_per_cpu_rejection_keeps_its_indent_at_every_width(
        self, width: int
    ) -> None:
        lines = _batch_stderr(width, mem_per_cpu="not-a-memory-value")
        assert _trailing(lines) == [], (width, lines)
        assert _unindented(lines) == [], (width, lines)

    def test_a_user_supplied_value_long_enough_to_wrap_the_error_line(self) -> None:
        # `{memory_val}` is whatever was typed, so the `Error:` line has no
        # length bound of its own -- 200 characters wrapped it into four lines at
        # 80 columns, three of them at column 0.
        lines = _batch_stderr(80, memory="g" * 200)
        assert len(lines) > 4
        assert _trailing(lines) == [], lines
        assert _unindented(lines) == [], lines

    def test_the_ntasks_rejection_keeps_its_indent_where_it_wraps(self) -> None:
        # 63 cells, so 60 is the width that wraps it -- the one member of the
        # four numeric rejections that wraps on its fixed text alone.
        lines = _batch_stderr(60, ntasks_per_node=0)
        assert len(lines) == 2, lines
        assert _trailing(lines) == [], lines
        assert _unindented(lines) == [], lines

    @pytest.mark.parametrize("flag,value", [
        ("cpus", 0), ("nodes", 0), ("gpus", -1), ("ntasks_per_node", 0),
    ])
    def test_every_numeric_rejection_goes_through_the_helper(
        self, flag: str, value: int
    ) -> None:
        # All four interpolate a number the user chose, so none is bounded; the
        # helper is what makes the length stop mattering.
        lines = _batch_stderr(60, **{flag: value})
        assert _trailing(lines) == [], (flag, lines)
        assert _unindented(lines) == [], (flag, lines)

    def test_the_interpolated_value_is_escaped_not_interpreted(self) -> None:
        # Came with routing through `_print_issue`. Before, rich read the value
        # as markup and `[red]` vanished from the message.
        lines = _batch_stderr(200, memory="[red]x")
        assert "Invalid memory value: [red]x" in lines[0], lines

    def test_the_glyph_now_follows_ascii_mode(self) -> None:
        # These sites hardcoded "✗", so `--ascii` did not reach them.
        # `_print_issue` takes the glyph from `theme.g`, which does.
        from slurmate import theme
        theme.set_ascii(True)
        try:
            lines = _batch_stderr(200, memory="bad")
        finally:
            theme.set_ascii(False)
        assert lines[0].startswith("  x Error:"), lines


class TestTheColourTerminalGetsColourNotEscapeBytes:
    """The third defect the same measurement turned up.

    Seven of these sites built their colour from `c.RED`/`c.RESET` -- raw ANSI --
    and handed the string to `rich`. Rich does not read ANSI in a `print`
    argument: it treated `[38;2;255;0;0m` as text, highlighted the numbers inside
    it, and emitted a broken sequence, so the escape appeared **on screen** and
    the line was not coloured at all.
    """

    def _painted(self, width: int, **flags: object) -> str:
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
                    main_module._run_batch(
                        args, Console(file=io.StringIO(), width=width), {}
                    )
                theme.c.__dict__.pop("_use_color", None)
        finally:
            sys.stderr = stashed
        return buffer.getvalue()

    def test_no_escape_sequence_reaches_the_screen_as_text(self) -> None:
        painted = self._painted(80, memory="not-a-memory-value")
        visible = _as_a_terminal_would_show(painted)
        assert "[38;2;" not in visible, visible
        assert "[0m" not in visible, visible
        assert visible.splitlines()[0] == (
            "  ✗ Error: Invalid memory value: not-a-memory-value"
        ), visible.splitlines()[0]

    def test_the_error_line_is_actually_red(self) -> None:
        # "carries some SGR" is not enough: before, the line carried rich's own
        # highlighter colours (`\x1b[1;36m` around the digits it found inside the
        # escape sequence) while the message itself was uncoloured. What has to
        # be true is that the sentence is red.
        painted = self._painted(80, memory="not-a-memory-value")
        first = painted.splitlines()[0]
        codes = re.findall(r"\x1b\[[\d;]*m", first)
        assert "\x1b[31m" in codes, (codes, repr(first))


def _as_a_terminal_would_show(painted: str) -> str:
    """Consume well-formed CSI sequences; keep what the terminal would draw."""
    out: list[str] = []
    i = 0
    while i < len(painted):
        char = painted[i]
        if char == "\x1b":
            j = i + 1
            if j < len(painted) and painted[j] == "[":
                j += 1
                while j < len(painted) and painted[j] in "0123456789;:<=>?":
                    j += 1
                if j < len(painted) and "\x40" <= painted[j] <= "\x7e":
                    i = j + 1          # a complete CSI: the terminal eats it
                    continue
            i += 1                     # a lone ESC: only the ESC is eaten
            continue
        out.append(char)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# Group 905: the inferred --gpus format hint.
# --------------------------------------------------------------------------- #

class TestTheInferredGpuHintIsIndentedThroughout:
    """`main.py:905`. 147 cells, so there is no terminal width where it fits."""

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_hint_keeps_its_indent_at_every_width(self, width: int) -> None:
        lines = _gpu_hint_lines(width)
        assert len(lines) > 1, (width, lines)      # the premise, at every width
        assert _trailing(lines) == [], (width, lines)
        assert _unindented(lines) == [], (width, lines)
        assert _overwidth(lines, width) == [], (width, lines)

    def test_the_hint_sits_at_the_depth_its_neighbours_use(self) -> None:
        # 2, the same as the `_print_issue` refusal and the `--force` hint this
        # function prints on its other two branches.
        for line in _gpu_hint_lines(60):
            assert line.startswith("  "), repr(line)
            assert not line.startswith("   "), repr(line)


# --------------------------------------------------------------------------- #
# Group 2451-2480: the interactive submit menu's rejections.
# --------------------------------------------------------------------------- #

class TestTheSubmitMenuRejectionsAreIndentedThroughout:
    """`main.py:2451-2480`, driven through `_main`'s real action loop."""

    @pytest.mark.parametrize("width", WIDTHS)
    def test_a_permanent_refusal_of_a_hand_edit_keeps_its_indent(
        self, width: int
    ) -> None:
        lines = _menu_lines(
            width, refusal=PERMANENT_REFUSAL, permanent=True,
            actions=("Open script in vim", "Submit to Slurm",
                     "Quit without submitting"),
        )
        block = _block(lines, "Slurm rejects the edited", stop="Not submitted")
        assert len(block) > 2, (width, block)
        assert _trailing(block) == [], (width, block)
        assert _unindented(block) == [], (width, block)
        assert _overwidth(block, width) == [], (width, block)

    @pytest.mark.parametrize("width", (60, 80, 120))
    def test_a_transient_refusal_keeps_its_indent(self, width: int) -> None:
        lines = _menu_lines(
            width, refusal=TRANSIENT_REFUSAL, permanent=False, transient=True,
            actions=("Open script in vim", "Submit to Slurm",
                     "Quit without submitting"),
        )
        block = _block(lines, "would not take this job", stop="<submitted>")
        assert len(block) > 1, (width, block)
        assert _trailing(block) == [], (width, block)
        assert _unindented(block) == [], (width, block)

    @pytest.mark.parametrize("width", (60, 70, 80, 120))
    def test_the_hard_error_list_keeps_its_indent(self, width: int) -> None:
        lines = _menu_lines(width, hard_errors=HARD_ERRORS)
        block = _block(lines, "exceeds the maximum", stop="Not submitted")
        assert _trailing(block) == [], (width, block)
        assert _unindented(block) == [], (width, block)
        assert _overwidth(block, width) == [], (width, block)


# --------------------------------------------------------------------------- #
# Controls. Every one of these passes with the change reverted as well.
# --------------------------------------------------------------------------- #

class TestControls:
    """What must not change. Verified green against the pre-fix code too."""

    @pytest.mark.parametrize("width", WIDTHS)
    def test_control_the_memory_rejection_loses_no_text_to_the_wrap(
        self, width: int
    ) -> None:
        # Compared against the same report at 200 columns rather than a literal,
        # so this is about the wrap and not about the wording.
        joined = " ".join(line.strip() for line in _batch_stderr(width, memory="bad"))
        assert joined == " ".join(
            line.strip() for line in _batch_stderr(200, memory="bad")
        ), (width, joined)

    def test_control_the_memory_rejection_still_says_what_would_have_worked(
        self,
    ) -> None:
        text = " ".join(_batch_stderr(60, memory="16 gigs"))
        assert "Invalid memory value: 16 gigs" in text, text
        assert "16G" in text, text

    def test_control_the_time_rejection_still_lists_the_forms(self) -> None:
        text = " ".join(line.strip() for line in _batch_stderr(60, time="1:0:0:0"))
        assert "Invalid time limit value: 1:0:0:0" in text, text
        assert "HH:MM:SS" in text, text

    @pytest.mark.parametrize("flag,value,phrase", [
        ("cpus", 0, "--cpus must be a positive integer (got 0)"),
        ("nodes", 0, "--nodes must be a positive integer (got 0)"),
        ("gpus", -1, "--gpus must be a non-negative integer (got -1)"),
        ("ntasks_per_node", 0,
         "--ntasks-per-node must be a positive integer (got 0)"),
        ("mem_per_cpu", "zz", "Invalid --mem-per-cpu value: zz"),
    ])
    def test_control_every_rejection_still_says_the_same_thing(
        self, flag: str, value: object, phrase: str
    ) -> None:
        text = " ".join(line.strip() for line in _batch_stderr(200, **{flag: value}))
        assert phrase in text, text

    @pytest.mark.parametrize("width", WIDTHS)
    def test_control_the_rejections_still_fit_the_terminal(
        self, width: int
    ) -> None:
        # True before as well -- rich wrapped to the width either way. It is
        # where the wrapped line *starts* that changed.
        for line in _batch_stderr(width, memory="not-a-memory-value"):
            assert len(line) <= width, (width, line)

    def test_control_a_valid_value_is_still_accepted(self) -> None:
        # The teeth above all run the refusal path; this is the other branch, so
        # a stricter check cannot pass by rejecting everything.
        args = argparse.Namespace(**{**_BATCH_DEFAULTS, "memory": "16G"})
        answers = main_module.run_batch(
            args, Console(file=io.StringIO(), width=80), {}
        )
        assert answers["memory"] == "16G"

    @pytest.mark.parametrize("width", WIDTHS)
    def test_control_the_gpu_hint_loses_no_text_to_the_wrap(
        self, width: int
    ) -> None:
        joined = " ".join(line.strip() for line in _gpu_hint_lines(width))
        assert joined == (
            "--gpus '<type>:count' reads as gpu_format 'gpus', which this "
            "cluster cannot parse; using 'gres_type' instead "
            "(pass --gpu-format to choose)."
        ), (width, joined)

    def test_control_a_cons_tres_cluster_still_gets_no_hint_at_all(self) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, width=60, force_terminal=False, no_color=True)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(main_module, "fetch_select_type",
                          lambda: "select/cons_tres")
            assert main_module._check_gpu_format(
                "gpus", gpus_requested=True, force=False, err_console=console,
                inferred=True,
            ) is None
        assert buffer.getvalue() == ""

    def test_control_an_explicit_format_is_still_refused_not_downgraded(
        self,
    ) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, width=80, force_terminal=False, no_color=True)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(main_module, "fetch_select_type", lambda: "select/linear")
            with pytest.raises(SystemExit) as exit_info:
                main_module._check_gpu_format(
                    "gpus", gpus_requested=True, force=False,
                    err_console=console, inferred=False,
                )
        assert exit_info.value.code == 1
        assert "needs select/cons_tres" in buffer.getvalue()

    def test_control_a_permanent_refusal_still_blocks_the_submit(self) -> None:
        lines = _menu_lines(
            80, refusal=PERMANENT_REFUSAL, permanent=True,
            actions=("Open script in vim", "Submit to Slurm",
                     "Quit without submitting"),
        )
        assert "<submitted>" not in " ".join(lines)
        assert "Not submitted." in " ".join(lines)

    def test_control_a_transient_refusal_still_submits_anyway(self) -> None:
        lines = _menu_lines(
            80, refusal=TRANSIENT_REFUSAL, permanent=False, transient=True,
            actions=("Open script in vim", "Submit to Slurm",
                     "Quit without submitting"),
        )
        assert "<submitted>" in " ".join(lines)
        assert "the script is valid; submitting anyway" in " ".join(
            line.strip() for line in lines
        )

    def test_control_hard_errors_still_block_the_submit_and_name_both(
        self,
    ) -> None:
        joined = " ".join(
            line.strip() for line in _menu_lines(200, hard_errors=HARD_ERRORS)
        )
        assert "<submitted>" not in joined
        assert HARD_ERRORS[0] in joined, joined
        assert HARD_ERRORS[1] in joined, joined
        assert "This job has errors Slurm will reject." in joined

    def test_control_a_clean_job_still_reaches_the_submit(self) -> None:
        # The other branch of the menu, so the checks above cannot pass by
        # refusing everything.
        lines = _menu_lines(80)
        assert "<submitted>" in " ".join(lines)

    def test_control_each_hard_error_is_still_a_paragraph_of_its_own(self) -> None:
        # A loop over `errs`, so each message stays its own indented paragraph
        # rather than the list being run together. True before as well -- at 120
        # neither message wraps -- which is why it is a control.
        block = _block(_menu_lines(120, hard_errors=HARD_ERRORS),
                       "exceeds the maximum", stop="Not submitted")
        assert block[0].startswith("  ✗ Time limit"), block
        assert block[1].startswith("  ✗ Memory 900G"), block

    @pytest.mark.parametrize("width", (60, 80, 120))
    def test_control_the_menu_rejections_still_fit_the_terminal(
        self, width: int
    ) -> None:
        block = _block(
            _menu_lines(width, refusal=PERMANENT_REFUSAL, permanent=True,
                        actions=("Open script in vim", "Submit to Slurm",
                                 "Quit without submitting")),
            "Slurm rejects the edited", stop="Not submitted",
        )
        for line in block:
            assert len(line) <= width, (width, line)
