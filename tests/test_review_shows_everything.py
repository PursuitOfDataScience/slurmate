"""The review screen clipped the job it was asking you to confirm.

Found by painting the real wizard's screen -- `write_to_screen`, the same call
prompt_toolkit makes -- instead of reading the layout code. On a 120-column
terminal with an ordinary job, the last screen before submission read:

    │ Output directory /home/youzhi/slu│ │  #SBATCH --output=/home/youzhi/slurmwatch/logs/spec│
    │ Modules          python/anaconda-│ │  #SBATCH --error=/home/youzhi/slurmwatch/logs/specc│
    │ Command          python train.py │ │  python train.py --config configs/anneal_a35.yaml -│

Three answers and four directives cut mid-string, with nothing on screen saying
so. `wrap_lines=False` was set explicitly on the summary window and left at its
default (also off) on the script window, so prompt_toolkit clipped. At 90
columns the summary column is 22 cells wide and five of twelve values were cut.

Both panels wrap now. Two details are deliberate:

* **The summary wraps under its own value column** when the column is wide enough
  to be worth using, because `_render_review_config` computes `label_w` from the
  real labels precisely so values line up -- its comment says a fixed width left
  "multi-line continuations lined up with nothing". `render_info` is what knows
  the width, and it does not exist on the first frame, so `wrap_lines=True` is
  the net that catches that one frame (it wraps to column 0: no value lost, just
  not aligned). Below ~5 cells of value column, column-0 wrapping is left to do
  the job, because a four-cell ribbon under an eighteen-cell indent is worse than
  a full-width wrap.
* **`_review_max_scroll` counts wrapped ROWS.** `total_lines - visible` assumed
  one row per line; with wrapping, the tail of the script became unreachable --
  the same "you cannot see what you are submitting" fault by another route.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition

from slurmate.tui import STEPS, Wizard

#: An ordinary job, with the long values a real one has (a path, a module list,
#: a training command).
ANSWERS: dict[str, Any] = {
    "job_name": "speccurve-anneal",
    "partition": "amd",
    "account": "pi-chenhao",
    "cpus": 8,
    "memory": "32G",
    "time": "12:00:00",
    "nodes": 1,
    "output_dir": "/home/youzhi/slurmwatch/logs",
    "modules": "python/anaconda-2023.09, cuda/12.4",
    "env_type": "conda",
    "env_name": "AI",
    "command": "python train.py --config configs/anneal_a35.yaml --resume checkpoints/last.pt",
}

#: Values that MUST be readable in full on the summary panel.
LONG_VALUES = [
    "speccurve-anneal",
    "/home/youzhi/slurmwatch/logs",
    "python/anaconda-2023.09, cuda/12.4",
    "python train.py --config configs/anneal_a35.yaml --resume checkpoints/last.pt",
]

SIZES = [(90, 30), (100, 30), (120, 40)]


def _wizard() -> Wizard:
    w = Wizard()
    w.answers.update(ANSWERS)
    w.idx = len(STEPS) - 1
    with set_app(w.app):
        w._on_enter_step()
        # `_invalidate()` gates on `app.is_running`; the content pane is composed
        # once per layout, so a step change needs the rebuild done by hand here.
        w.app.layout = w._build_layout()
    return w


def _paint(w: Wizard, cols: int, rows: int) -> list[str]:
    """The painted screen, as prompt_toolkit itself would draw it."""
    screen = Screen(initial_width=cols, initial_height=rows)
    screen.width, screen.height = cols, rows
    # Without a running app the render counter never moves and every Window
    # returns its CACHED first frame -- which is how a draft of this file
    # "painted" step 23 and got step 1 back.
    w.app.render_counter += 1
    with set_app(w.app):
        w.app.layout.container.write_to_screen(
            screen, MouseHandlers(), WritePosition(0, 0, cols, rows), "", False, None
        )
    return [
        "".join(screen.data_buffer[y][x].char for x in range(cols)).rstrip() for y in range(rows)
    ]


def _panel_raw(lines: list[str], title: str) -> list[str]:
    """`_panel`, but keeping the interior spaces -- alignment is the point here."""
    head = next(ln for ln in lines if f"─ {title} " in ln)
    start = head.index(f"─ {title} ") - 2
    end = head.index("╮", start) + 1
    out = []
    for ln in lines[lines.index(head) + 1 :]:
        cell = ln[start:end]
        if "╰" in cell:
            break
        # Between the card's own two borders, interior padding intact.
        inner = cell[cell.index("│") + 1 : cell.rindex("│")] if cell.count("│") > 1 else ""
        out.append(inner.rstrip())
    return out


def _panel(lines: list[str], title: str) -> list[str]:
    """The interior rows of the card titled `title`, as painted."""
    head = next(ln for ln in lines if f"─ {title} " in ln)
    start = head.index(f"─ {title} ") - 2
    end = head.index("╮", start) + 1
    out = []
    for ln in lines[lines.index(head) + 1 :]:
        cell = ln[start:end]
        if "╰" in cell:
            break
        out.append(cell.strip("│ ").rstrip())
    return out


def _squashed(rows: list[str]) -> str:
    """The panel's text with ALL whitespace removed.

    Wrapping breaks mid-word (`spec` / `curve-anneal`), so rejoining with a space
    invents one -- an earlier draft compared `spec curve-anneal` against
    `speccurve-anneal` and failed for that reason alone. Dropping whitespace on
    both sides asks the question that matters: are the value's characters all on
    screen, in order?
    """
    return re.sub(r"\s+", "", "".join(rows))


def _twice(cols: int, rows: int) -> tuple[Wizard, list[str]]:
    """Paint twice: the first frame is what populates `render_info`."""
    w = _wizard()
    _paint(w, cols, rows)
    return w, _paint(w, cols, rows)


class TestNothingOnTheReviewScreenIsCutOff:
    @pytest.mark.parametrize(("cols", "rows"), SIZES)
    @pytest.mark.parametrize("value", LONG_VALUES)
    def test_every_answer_is_readable_in_full(self, cols: int, rows: int, value: str) -> None:
        if cols < 120 and value == ANSWERS["command"]:
            pytest.skip("the card runs out of ROWS here -- see the class below")

        async def run() -> str:
            _w, painted = _twice(cols, rows)
            return _squashed(_panel(painted, "Job Configuration"))

        text = asyncio.run(run())
        assert re.sub(r"\s+", "", value) in text, (cols, value, text)


class TestTheRemainingLimitIsVerticalNotHorizontal:
    """What is still cut at 90 and 100 columns, measured rather than implied.

    Wrapping trades horizontal loss for height, and the summary card is a fixed
    height. Measured at 90x30: a 12-answer summary wraps to 20 rows in a card
    with 20, so the last piece of the longest value (`checkpoints/last.pt`) is
    below the card's own fold. Two things make that a much smaller fault than the
    clipping it replaced, and both are asserted here: the cut is at the END of the
    LAST value rather than in the middle of five of them, and the full command is
    on screen in the `Final Script` panel next to it -- which scrolls.
    """

    def test_the_cut_is_the_tail_of_the_last_row(self) -> None:
        async def run() -> list[str]:
            _w, painted = _twice(90, 30)
            return _panel(painted, "Job Configuration")

        rows = asyncio.run(run())
        # Every answer except the command's tail is present.
        text = _squashed(rows)
        for value in LONG_VALUES[:-1]:
            assert re.sub(r"\s+", "", value) in text, (value, text)
        assert rows[-1].strip(), "the card's last row is blank, so height is not the limit"

    def test_the_full_command_is_readable_in_the_script_panel(self) -> None:
        async def run() -> str:
            w, painted = _twice(90, 30)
            w._review_scroll = w._review_max_scroll()
            return _squashed(_panel(_paint(w, 90, 30), "Final Script"))

        text = asyncio.run(run())
        assert re.sub(r"\s+", "", ANSWERS["command"]) in text, text

    @pytest.mark.parametrize(("cols", "rows"), SIZES)
    def test_the_script_directives_are_readable_in_full(self, cols: int, rows: int) -> None:
        async def run() -> str:
            _w, painted = _twice(cols, rows)
            return _squashed(_panel(painted, "Final Script"))

        text = asyncio.run(run())
        assert "#SBATCH--job-name=speccurve-anneal" in text, text
        assert "/home/youzhi/slurmwatch/logs/speccurve-anneal-%j.out" in text, text

    @pytest.mark.parametrize(("cols", "rows"), SIZES)
    def test_the_last_script_line_can_be_scrolled_to(self, cols: int, rows: int) -> None:
        """The scroll bound, which wrapping would otherwise strand short."""

        async def run() -> str:
            w, _painted = _twice(cols, rows)
            w._review_scroll = w._review_max_scroll()
            return _squashed(_panel(_paint(w, cols, rows), "Final Script"))

        text = asyncio.run(run())
        assert "train.py" in text, text  # the command is the script's last line

    def test_a_wrapped_value_hangs_under_its_own_column(self) -> None:
        """Where there is room, a continuation lines up with the value above it.

        Asserted as a COLUMN, not as text: `wrap_lines` alone puts the
        continuation at column 0, which is the misalignment
        `_render_review_config`'s own comment says the label width exists to
        prevent. Only the pre-wrap gives the hanging indent, so only a column
        assertion can tell the two apart.
        """

        async def run() -> list[str]:
            _w, painted = _twice(120, 40)
            return _panel_raw(painted, "Job Configuration")

        rows = asyncio.run(run())
        head = next(ln for ln in rows if ln.lstrip().startswith("Output directory"))
        value_col = rows.index(head)
        column = rows[value_col + 1]
        assert column.lstrip() == "rmwatch/logs", column
        # It starts under the value, i.e. one past the label column.
        assert len(column) - len(column.lstrip()) == head.index("/home"), (head, column)


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_a_short_value_still_sits_beside_its_label_on_one_line(self) -> None:
        async def run() -> list[str]:
            _w, painted = _twice(120, 40)
            return _panel(painted, "Job Configuration")

        rows = asyncio.run(run())
        assert any(re.fullmatch(r"Partition\s+amd", ln) for ln in rows), rows
        assert any(re.fullmatch(r"Memory\s+32G", ln) for ln in rows), rows

    def test_the_label_column_is_still_the_longest_label(self) -> None:
        async def run() -> list[str]:
            _w, painted = _twice(120, 40)
            return _panel(painted, "Job Configuration")

        rows = asyncio.run(run())
        starts = {
            m.end() for ln in rows if (m := re.match(r"(?:Partition|Memory|Nodes|CPUs)\s+", ln))
        }
        assert len(starts) == 1, (starts, rows)
        assert starts == {len("Output directory") + 1}, (starts, rows)

    def test_the_scroll_bound_matches_the_old_arithmetic_when_nothing_wraps(self) -> None:
        """With every line inside the panel, rows == lines and so does the bound."""

        async def run() -> tuple[int, int]:
            w, _painted = _twice(120, 40)
            info = w._review_script_window.render_info
            visible = max(1, info.window_height if info else 0)
            w._review_line_widths = [1] * w._review_total_lines  # nothing wraps
            return w._review_max_scroll(), max(0, w._review_total_lines - visible)

        got, expected = asyncio.run(run())
        assert got == expected, (got, expected)

    def test_entering_the_review_still_starts_at_the_top(self) -> None:
        async def run() -> int:
            w = _wizard()
            with set_app(w.app):
                w._setup_review()
            return w._review_scroll

        assert asyncio.run(run()) == 0

    def test_the_script_is_still_the_one_the_builder_produces(self) -> None:
        from slurmate.builder import build_from_answers

        async def run() -> tuple[int, int]:
            w = _wizard()
            with set_app(w.app):
                lines = w._build_script_lines()
            return len(lines), len(build_from_answers(w.answers).split("\n"))

        got, expected = asyncio.run(run())
        assert got == expected, (got, expected)

    def test_the_first_frame_already_shows_the_values(self) -> None:
        """`render_info` is None there, so `wrap_lines` is what has to carry it."""

        async def run() -> str:
            w = _wizard()
            return _squashed(_panel(_paint(w, 120, 40), "Job Configuration"))

        text = asyncio.run(run())
        assert "/home/youzhi/slurmwatch/logs" in text, text
