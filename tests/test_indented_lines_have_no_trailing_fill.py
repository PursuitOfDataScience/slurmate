"""Every indented status line was padded out to the terminal with spaces.

`_print_indented` exists so that "a wrapped line keeps the indent" -- a `rich`
`Padding` indents every line of a wrapped paragraph, where two literal spaces
inside the markup indent only the first. That part works. What came with it is
that a `Padding` renders as a **full-width block**: `expand` defaults to `True`,
so every call site filled the rest of the row with spaces.

Measured on the real surface at 90 columns::

    $ COLUMNS=90 slurmate --dry-run ... | grep -cP ' $'
    1
    '  \x1b[31m<glyph> Slurm refuses this job: ...(Access/permission denied)\x1b[0m           \r'

79 cells of text and 11 trailing spaces, *outside* the style reset, filling the
row exactly. Nothing is misdrawn, which is why it survived: the cost is that the
line copies with junk on the end, a captured log flags on any
`grep -nP ' $'`, and a line that exactly fills the terminal leaves the cursor at
the right margin. `expand=False` sizes the block to the text instead.

**Then the same thing one layer in.** `expand=False` sizes the *block*, but
`Padding` renders its child with `pad=True`, so a paragraph long enough to WRAP
was measured at the full available width and every line of it was filled to the
wrap column again. That was pinned here as a residual limitation. It is gone:
`_print_indented` now asks `rich` where to wrap and hands each resulting line to
the single-line path, where the block is measured to that line's own length.
`rich` still decides the wrap points -- see `TestControls`, which pins that a long
line still wraps, still respects the width, and loses nothing to the wrap.

**And the two sites that were not going through the helper at all.** Recorded as
**L11** in `issues.md`: `main.py` printed the continuation lines of a multi-line
site-check issue, and the ``--force`` hint, with their indent as **literal
spaces inside the string** -- the arrangement this helper exists to replace. Both
now go through it, at ``indent=4`` and ``indent=2``. Measured on a real
``--dry-run`` against an unknown partition on an 85-partition cluster, which is
the message here that always wraps because it interpolates the partition list::

    COLUMNS=60, before:  3 lines with a trailing space, 3 lines at column 0
      "    This cluster's partitions: caslake, aaz, aettinger-gpu, "
      'ai4s-hackathon, amd, amd-hm, andrewferguson, '     <- indent lost
    COLUMNS=60, after:   0 and 0
      "    This cluster's partitions: caslake, aaz, aettinger-gpu,"
      '    ai4s-hackathon, amd, amd-hm, andrewferguson,'

0 and 0 at 60, 70, 80, 90 and 120 columns. `_FORCE_HINT` carried its own two-space
prefix and is printed from three sites, so the prefix moved out of the constant
rather than being indented twice -- all three hand it to the helper now.

`TestTheSiteCheckReportIsIndentedThroughout` measures that on the real report, and
`TestTheReportItselfIsUnchanged` pins what must not change there in either state:
the wrap still happens and the text survives it, the report still fits the width,
the head line and the ``--force`` downgrade are untouched, and the check still
exits 1.
"""

from __future__ import annotations

import io
from unittest import mock

import pytest
from rich.console import Console

from slurmate import main as main_module
from slurmate.main import _print_indented

#: Long enough to wrap several times at the width below.
LONG = (
    "Slurm refuses this job: Account is not specified (Access/permission denied) "
    "and the partition does not accept this QoS from this account either"
)
WIDTH = 60


def _render(markup: str, width: int = WIDTH, indent: int = 2) -> list[str]:
    """The lines a real Console produces, with no colour to obscure them."""
    buffer = io.StringIO()
    console = Console(file=buffer, width=width, force_terminal=False, no_color=True)
    _print_indented(console, markup, indent)
    return buffer.getvalue().splitlines()


class TestNoLineIsPaddedToTheTerminal:
    def test_a_short_line_ends_at_its_text(self) -> None:
        lines = _render("done")
        assert lines == ["  done"], lines

    def test_a_line_that_fits_carries_no_fill_at_any_width(self) -> None:
        for width in (40, 60, 90, 120):
            lines = _render("Slurm refuses this job: no account", width=width)
            assert lines == ["  Slurm refuses this job: no account"], (width, lines)

    def test_the_rendered_line_is_exactly_indent_plus_text(self) -> None:
        assert _render("refused") == ["  refused"]
        assert _render("refused", indent=6) == ["      refused"]
        assert _render("[red]refused[/]") == ["  refused"]

    def test_a_styled_line_is_not_padded_after_its_reset(self) -> None:
        # The reported line's shape: markup, then the fill sat outside the reset.
        buffer = io.StringIO()
        console = Console(file=buffer, width=WIDTH, force_terminal=True)
        _print_indented(console, "[red]refused[/]")
        painted = buffer.getvalue().rstrip("\n")
        assert not painted.endswith(" "), repr(painted)
        assert "refused" in painted

    def test_the_line_no_longer_fills_the_row(self) -> None:
        # A row filled to the last cell is what leaves the cursor at the margin.
        assert all(len(line) < WIDTH for line in _render("refused"))

    def test_a_paragraph_that_wraps_carries_no_fill_on_any_line(self) -> None:
        """The residual this module used to pin, now closed.

        `expand=False` drops the block's own right-hand fill; it does not stop
        `Padding` rendering its child with `pad=True`, so a wrapping paragraph was
        measured at the full width and each of its lines filled to the wrap column
        -- at 60 columns, three lines of 60 cells where the text was 50, 55 and 41.
        The wrap now happens first and each line is padded as the single line it
        is. This is the case the two newly-routed site-check callers hit on every
        invocation, which is why it stopped being an acceptable limit.
        """
        lines = _render(LONG)
        assert len(lines) > 1
        assert [line for line in lines if line != line.rstrip()] == [], lines


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_a_long_line_still_wraps(self) -> None:
        assert len(_render(LONG)) > 1

    def test_every_wrapped_line_still_starts_at_the_indent(self) -> None:
        # The whole reason `_print_indented` exists: two literal spaces in the
        # markup indent the first line only.
        for line in _render(LONG):
            assert line.startswith("  "), repr(line)
            assert not line.startswith("   "), repr(line)

    def test_the_wrap_still_respects_the_width(self) -> None:
        assert all(len(line) <= WIDTH for line in _render(LONG))

    def test_nothing_is_lost_to_the_wrap(self) -> None:
        joined = " ".join(line.strip() for line in _render(LONG))
        assert joined == LONG, joined

    def test_the_indent_is_still_configurable(self) -> None:
        lines = _render("x", indent=6)
        assert len(lines) == 1
        assert lines[0].startswith("      x"), repr(lines[0])

    def test_markup_is_still_interpreted(self) -> None:
        # `Text.from_markup`, not a literal: the tag must not reach the screen.
        line = _render("[red]refused[/]")[0]
        assert line.strip() == "refused", repr(line)
        assert "[red]" not in line

    def test_a_bracketed_value_still_survives_as_markup_would(self) -> None:
        # Callers escape their interpolations; an unknown tag is left alone by
        # rich rather than swallowing the line.
        line = _render("(JobArrayTaskLimit)")[0]
        assert line.strip() == "(JobArrayTaskLimit)", repr(line)


#: Ten names, so the "This cluster's partitions:" line is long enough to wrap at
#: every width below. Real names from the cluster this was measured on.
FIXTURE_PARTITIONS = [
    "caslake", "aaz", "aettinger-gpu", "ai4s-hackathon", "amd", "amd-hm",
    "andrewferguson", "andrewferguson-gpu", "astroplasmas", "beagle3",
]


def _target_check_lines(
    width: int, *, force: bool = False, catch_exit: bool = True
) -> list[str]:
    """The real site-check report, rendered at ``width`` with the cluster stubbed.

    `_check_cluster_targets` is the one path that prints through BOTH the fixed
    helper (its `_print_issue` head line) and the two bare prints below it, which
    is what makes it the surface to measure the scope boundary on. Stubbed rather
    than run against Slurm so the wrap points are fixed.

    ``force=True`` downgrades the issue to a warning and skips the ``--force``
    hint, which is how the two sibling sites are told apart.

    ``catch_exit=False`` lets the ``SystemExit`` out, so a control can assert the
    status this path exits with rather than only what it printed on the way.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, width=width, force_terminal=False, no_color=True)
    with mock.patch.object(
        main_module, "fetch_all_partition_names", return_value=FIXTURE_PARTITIONS
    ), mock.patch.object(
        main_module, "fetch_user_accounts", return_value=[]
    ), mock.patch.object(
        main_module, "fetch_known_qos", return_value=[]
    ), mock.patch.object(
        main_module, "fetch_node_features", return_value=None
    ):
        try:
            main_module._check_cluster_targets(
                "nosuchpartition", None, [], force=force, err_console=console
            )
        except SystemExit:
            # The non-force path exits 1 after printing, which is its job.
            if not catch_exit:
                raise
    return buffer.getvalue().splitlines()


def _trailing(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if line != line.rstrip()]


def _unindented(lines: list[str]) -> list[int]:
    """Lines that come back at column 0 -- the indent the wrap dropped."""
    return [i for i, line in enumerate(lines) if line and not line.startswith(" ")]


#: The widths L11 was measured at. 60 and 70 wrap the partition list into three
#: lines and the hint into two; 80 and above wrap the list only.
WIDTHS = (60, 70, 80, 90, 120)


class TestTheSiteCheckReportIsIndentedThroughout:
    """L11, on the real report: two sites printed their indent as literal text.

    `main.py` carried the continuation lines of a multi-line issue at four literal
    spaces and the ``--force`` hint at two, so `rich` indented the first line of
    each and wrapped the remainder to column 0 -- with the space it broke on left
    on the end. Both now go through `_print_indented`, which pads every line of
    the paragraph and rstrips each one.
    """

    @pytest.mark.parametrize("width", WIDTHS)
    def test_no_line_of_the_report_has_trailing_whitespace(self, width: int) -> None:
        # On this fixture, with the literal prefixes: 3 at 60, 2 at 70, 1 at
        # 80/90/120. Routing the two sites through the helper alone would have
        # made it WORSE -- 5 at 60 -- because `Padding` filled every line of a
        # wrapping paragraph to the wrap column, which is the second half of the
        # change this pins.
        lines = _target_check_lines(width)
        assert _trailing(lines) == [], (width, lines)

    @pytest.mark.parametrize("width", WIDTHS)
    def test_no_continuation_line_comes_back_at_column_zero(self, width: int) -> None:
        lines = _target_check_lines(width)
        assert _unindented(lines) == [], (width, lines)

    def test_the_issue_continuations_keep_their_deeper_indent(self) -> None:
        # `indent=4`, which is what a continuation line means: it is indented
        # under the `Error:` line, not level with it. With `--force` the hint is
        # not printed, so what is left is this site alone.
        lines = _target_check_lines(60, force=True)
        assert lines[1].startswith("    This cluster's partitions:"), lines
        for index in range(1, len(lines)):
            assert lines[index].startswith("    "), (index, lines)
            assert not lines[index].startswith("     "), (index, lines)

    def test_the_force_hint_keeps_the_two_space_indent(self) -> None:
        # The prefix moved out of `_FORCE_HINT` (three call sites) and became the
        # helper's default indent, so the wrapped remainder is indented too.
        lines = _target_check_lines(60)
        assert lines[4].startswith("  Pass --force"), lines
        assert lines[5].startswith("  another cluster)."), lines
        assert not lines[5].startswith("   "), lines

    def test_the_helper_line_in_the_real_report_carries_no_fill(self) -> None:
        """The head line, which went through the helper all along.

        Red with `expand=True` restored, and kept because it is the one line of
        this report whose indent never depended on the two fixed sites.
        """
        for width in WIDTHS:
            lines = _target_check_lines(width)
            assert lines[0] == lines[0].rstrip(), (width, lines[0])
            assert 0 not in _trailing(lines), (width, lines)


class TestTheReportItselfIsUnchanged:
    """Controls on the surface the fix moved. Each passes in BOTH states."""

    def test_control_the_head_line_still_names_the_partition(self) -> None:
        lines = _target_check_lines(60)
        assert lines[0].strip().endswith(
            "Error: no partition 'nosuchpartition' on this cluster."
        ), lines

    @pytest.mark.parametrize("width", WIDTHS)
    def test_control_no_text_is_lost_to_the_wrap(self, width: int) -> None:
        # The one thing a wrap change could silently break. Compared against the
        # 120-column rendering of the same report rather than a literal, so this
        # is about the wrap and not about the fixture's partition names.
        joined = " ".join(line.strip() for line in _target_check_lines(width))
        assert joined == " ".join(
            line.strip() for line in _target_check_lines(120)
        ), (width, joined)

    @pytest.mark.parametrize("width", WIDTHS)
    def test_control_the_report_still_fits_the_width(self, width: int) -> None:
        for line in _target_check_lines(width):
            assert len(line) <= width, (width, line)

    def test_control_the_partition_list_still_wraps_at_sixty(self) -> None:
        # The premise of every measurement above: this message is long enough to
        # reach a wrap point at 60 columns.
        assert len(_target_check_lines(60, force=True)) > 2

    def test_control_the_first_continuation_line_is_still_at_column_four(self) -> None:
        # True of a literal four-space prefix too -- rich indents the first line.
        # It is the SECOND line that told the two states apart.
        assert _target_check_lines(60)[1].startswith("    This cluster's"), (
            _target_check_lines(60)
        )

    def test_control_the_force_hint_is_still_offered_and_still_says_why(self) -> None:
        lines = _target_check_lines(60)
        assert lines[4].strip().startswith("Pass --force"), lines
        assert "another cluster" in " ".join(line.strip() for line in lines), lines

    def test_control_force_still_downgrades_and_drops_the_hint(self) -> None:
        lines = _target_check_lines(60, force=True)
        assert "Warning:" in lines[0], lines
        assert "Pass --force" not in " ".join(lines), lines

    def test_control_the_check_still_exits_nonzero(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            _target_check_lines(60, catch_exit=False)
        assert exit_info.value.code == 1
