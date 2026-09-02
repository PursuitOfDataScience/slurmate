"""``FORCE_COLOR=""`` is not the same as unset, and rich decides that.

``_should_use_color`` exists so slurmate's own ``c.*`` codes and rich's panels
agree about colour -- its docstring records the split that prompted it, where
``FORCE_COLOR=1`` while piping left "rich's panels kept their colour while every
``c.*``-prefixed status line lost it". It claimed to match rich "down to the
edges", including that ``FORCE_COLOR=""`` "counts as unset".

It does not. Re-measured against the installed rich (15.0.0) on a tty, with the
bare ``Console()`` this package constructs, an empty value is the single input
rich reads as **not a terminal** (``is_terminal=False``, ``color_system=None``);
every other value, including ``"0"``, ``" "`` and ``"no"``, leaves colour on.

So the split came back inverted: on ``--dry-run`` over a tty with
``FORCE_COLOR=``, rich's styling collapsed from 253 SGR sequences to 1 while the
``c.*``-coloured "Dry run — not submitted." line kept its grey -- one coloured
line on an otherwise plain screen.

Only the empty case changed. The other five values agreed with rich before and
still do, on a tty and piped.
"""

import pytest

from slurmate.theme import _should_use_color


@pytest.fixture
def tty(monkeypatch):
    """A stdout that claims to be a terminal, so the env is the only variable.

    ``isatty`` is patched on whatever object is currently installed rather than
    by replacing ``sys.stdout``: under pytest's default fd-level capture the
    replacement is swapped back out and every assertion here inverted, passing
    only under ``-s``. Patching the attribute survives either capture mode.
    """
    import sys

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    return None


def test_a_present_but_empty_force_color_turns_colour_off(tty, monkeypatch):
    """The one value rich reads as "not a terminal"."""
    monkeypatch.setenv("FORCE_COLOR", "")
    assert _should_use_color() is False


def test_control_unset_still_follows_the_terminal(tty):
    """CONTROL, passing with the change present or absent.

    Unset is the common case and must keep deferring to ``isatty()`` -- the fix
    is about a value being PRESENT and empty, not about absence.
    """
    assert _should_use_color() is True


def test_control_every_other_value_still_forces_colour(tty, monkeypatch):
    """CONTROL, in both states, and the reason the fix is one branch wide.

    ``"0"`` is the one that looks like it should mean off and does not, because
    rich treats it as on. Whitespace stays on for the same reason: rich only
    singles out the empty string.
    """
    for value in ("1", "0", "no", " ", "true"):
        monkeypatch.setenv("FORCE_COLOR", value)
        assert _should_use_color() is True, value


def test_control_no_color_still_wins(tty, monkeypatch):
    """CONTROL, in both states. The precedence the docstring promises.

    Measured on the CLI too: ``NO_COLOR=1 FORCE_COLOR=1`` on a tty emitted 0
    colour codes where ``FORCE_COLOR=1`` alone emitted 60.
    """
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    assert _should_use_color() is False


def test_control_term_dumb_still_wins(tty, monkeypatch):
    """CONTROL, in both states. ``TERM=dumb`` sits beside ``NO_COLOR``."""
    monkeypatch.setenv("TERM", "dumb")
    assert _should_use_color() is False
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _should_use_color() is False, "dumb outranks an explicit force"


def test_it_agrees_with_the_installed_rich_at_every_edge(tty, monkeypatch):
    """The claim the docstring makes, asserted against rich rather than restated.

    Skipped if rich is absent; this package needs it, but the assertion is about
    agreement and there is nothing to agree with.

    The Console is built the way this package builds it -- bare, over a file that
    reports a tty -- and NOT with ``force_terminal=True``: forcing it overrides
    the ``is_terminal`` detection that is exactly where rich reads an empty
    ``FORCE_COLOR``, so the comparison would be blind to the case under test.
    """
    Console = pytest.importorskip("rich.console").Console

    class _TtyFile:
        encoding = "utf-8"

        def isatty(self):
            return True

        def write(self, _s):
            return 0

        def flush(self):
            pass

    for value in (None, "", " ", "0", "1", "no"):
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        if value is not None:
            monkeypatch.setenv("FORCE_COLOR", value)
        console = Console(file=_TtyFile())
        rich_on = console.is_terminal and console.color_system is not None
        assert _should_use_color() is rich_on, (value, rich_on)
