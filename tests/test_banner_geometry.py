"""The banner is four parallel lists, and nothing checked that they agree.

`print_banner` walks the art with `enumerate(lines)` and indexes a *different*
list by that same `i`:

    for i, line in enumerate(lines):            # static, coloured
        print(f"{BANNER_GRADIENT[i]}...{line}...")
    ...
    for i, line in enumerate(lines):            # animated
        r, g, b = _brighten(BASE_RGB[i], intensity)

so `BANNER_LINES`, `BANNER_LINES_ASCII`, `BANNER_GRADIENT` and `BASE_RGB` are a
parallel-array contract. All four happen to be 6 long, and nothing said they had
to be. Measured: appending one line to the art raises

    IndexError: list index out of range

on the **coloured** static path -- i.e. on an ordinary interactive terminal,
which is the only place the banner is ever seen, while every non-colour test
stays green because that branch prints `line` alone.

The existing coverage does not reach this. `TestClusterPortability`'s banner
tests pin the ASCII *fold* and one of them asserts `len(art) == 6`, but it runs
a subprocess under `PYTHONIOENCODING=ascii`, so it reads `BANNER_LINES_ASCII`
and says nothing about the Unicode variant or about the gradient at all. Branch
coverage put the whole animated block (`theme.py:347-370`) among the unexecuted
lines of the package's lowest-covered real module.

One trap worth recording: `_to_rgb` falls back to `(255, 0, 128)` when its regex
does not match, and **G1's real value is also `(255, 0, 128)`** -- so the
sentinel cannot be used to detect a parse failure. The check here re-reads the
digits out of the escape sequence instead.
"""

import io
import re
import sys
from unittest import mock

import pytest

from slurmate import theme

#: The lists that `print_banner` indexes by the same loop variable.
PARALLEL = ("BANNER_LINES", "BANNER_LINES_ASCII", "BANNER_GRADIENT", "BASE_RGB")


def _render(colour, ascii_flag=False, encoding="utf-8"):
    """Return the banner's stdout, forcing the colour decision either way."""
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    theme.set_ascii(ascii_flag)
    try:
        with mock.patch.object(theme, "_should_use_color", return_value=colour), \
             mock.patch.object(theme, "output_encoding", return_value=encoding), \
             mock.patch.dict("os.environ", {}, clear=False) as env:
            for name in ("SLURMATE_NO_BANNER", "SLURMATE_ASCII",
                         "SLURMATE_BANNER_ANIMATE"):
                env.pop(name, None)
            theme.print_banner(interactive=False)
    finally:
        sys.stdout = saved
        theme.set_ascii(False)
    return buf.getvalue()


class TestTheParallelListsAgree:
    def test_all_four_are_the_same_length(self):
        lengths = {name: len(getattr(theme, name)) for name in PARALLEL}
        assert len(set(lengths.values())) == 1, lengths

    @pytest.mark.parametrize("variant", ["BANNER_LINES", "BANNER_LINES_ASCII"])
    def test_a_gradient_code_exists_for_every_line(self, variant):
        # The IndexError contract, stated per variant because `use_ascii()`
        # decides which one the loop walks at call time.
        assert len(getattr(theme, variant)) <= len(theme.BANNER_GRADIENT)
        assert len(getattr(theme, variant)) <= len(theme.BASE_RGB)

    @pytest.mark.parametrize("variant", ["BANNER_LINES", "BANNER_LINES_ASCII"])
    def test_each_variant_is_rectangular(self, variant):
        widths = {len(line) for line in getattr(theme, variant)}
        assert len(widths) == 1, (variant, sorted(widths))

    def test_every_gradient_code_really_parses(self):
        """Not via the sentinel: G1's true value equals the fallback."""
        for code in theme.BANNER_GRADIENT:
            match = re.fullmatch(r"\033\[38;2;(\d+);(\d+);(\d+)m", code)
            assert match, repr(code)
            expected = tuple(int(g) for g in match.groups())
            assert theme._to_rgb(code) == expected, code


class TestTheColouredPathWalksEveryLine:
    @pytest.mark.parametrize("ascii_flag", [False, True])
    def test_the_coloured_static_banner_prints_all_its_lines(self, ascii_flag):
        # Drives the branch that would raise: `BANNER_GRADIENT[i]`.
        out = _render(colour=True, ascii_flag=ascii_flag,
                      encoding="ascii" if ascii_flag else "utf-8")
        lines = theme.BANNER_LINES_ASCII if ascii_flag else theme.BANNER_LINES
        for line in lines:
            assert line in out, (line, out)

    def test_each_art_line_carries_its_own_gradient_code(self):
        out = _render(colour=True)
        for code, line in zip(theme.BANNER_GRADIENT, theme.BANNER_LINES):
            assert f"{code}{theme.c.BOLD}\033[3m{line}" in out, (code, line)

    def test_the_gradient_codes_are_distinct_so_i_is_load_bearing(self):
        # If they were all equal, indexing by `i` would be untestable and the
        # length contract above would not matter.
        assert len(set(theme.BANNER_GRADIENT)) == len(theme.BANNER_GRADIENT)


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_both_variants_have_the_same_line_count(self):
        """Currently guaranteed by construction -- passes in both states.

        `BANNER_LINES_ASCII` is `[line.translate(_ASCII_BANNER_MAP) for line in
        BANNER_LINES]`, so the counts cannot diverge today. Pinned anyway
        because the animation moves the cursor up `len(lines) + 1` and restores
        to a saved position: if the two variants ever differed -- someone
        hand-writing the ASCII art instead of deriving it -- the region would
        be right on one terminal and wrong on the other.
        """
        assert len(theme.BANNER_LINES) == len(theme.BANNER_LINES_ASCII)

    def test_to_rgb_still_falls_back_on_an_unparsable_code(self):
        # The documented degenerate branch: a 256-colour or reset code has no
        # RGB triple to read, and the animation must still get *a* colour.
        assert theme._to_rgb("\033[38;5;201m") == (255, 0, 128)
        assert theme._to_rgb("") == (255, 0, 128)

    def test_banner_lines_still_switches_on_use_ascii(self):
        theme.set_ascii(True)
        try:
            assert theme.banner_lines() == theme.BANNER_LINES_ASCII
        finally:
            theme.set_ascii(False)
        assert theme.banner_lines() == theme.BANNER_LINES

    def test_the_uncoloured_banner_emits_no_escape_codes(self):
        out = _render(colour=False)
        assert "\033[38;2;" not in out
        for line in theme.BANNER_LINES:
            assert line in out

    def test_the_ascii_variant_carries_no_block_art(self):
        # The property the existing fold tests are about, restated locally so
        # this file's own use of the two lists cannot drift from theirs.
        assert not any("█" in line for line in theme.BANNER_LINES_ASCII)
        assert any("█" in line for line in theme.BANNER_LINES)

    def test_the_no_banner_env_var_still_wins(self):
        with mock.patch.dict("os.environ", {"SLURMATE_NO_BANNER": "1"}):
            buf, saved = io.StringIO(), sys.stdout
            sys.stdout = buf
            try:
                theme.print_banner(interactive=False)
            finally:
                sys.stdout = saved
        assert buf.getvalue() == ""
