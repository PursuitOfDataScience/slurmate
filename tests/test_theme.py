"""Tests for theme colors and the startup banner."""

from slurmate import theme
from slurmate.theme import C, print_banner


class TestWarningColorUnified:
    def test_yellow_is_amber_not_pure_yellow(self, monkeypatch):
        # D4: CLI warning/SU color standardized on amber, matching the TUI.
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.delenv("NO_COLOR", raising=False)
        c = C()
        c.__dict__["_use_color"] = True  # force color on for the assertion
        assert "255;170;0" in c.YELLOW  # amber
        assert "255;255;0" not in c.YELLOW  # not pure yellow


class TestColorDecisionCached:
    def test_decision_computed_once_per_instance(self, monkeypatch):
        # P3-6: the color decision is cached, not recomputed on every access.
        calls = {"n": 0}
        real = theme._should_use_color

        def counting() -> bool:
            calls["n"] += 1
            return real()

        monkeypatch.setattr(theme, "_should_use_color", counting)
        c = C()
        _ = (c.PINK, c.CYAN, c.GREEN, c.RED, c.RESET)
        assert calls["n"] == 1

    def test_no_color_still_respected(self, monkeypatch):
        # Regression guard: caching must not break NO_COLOR.
        monkeypatch.setenv("NO_COLOR", "1")
        c = C()
        assert c.PINK == ""
        assert c.RESET == ""


class TestBanner:
    def test_batch_mode_hides_esc_hint(self, capsys, monkeypatch):
        # D8: the "ESC to go back" hint is meaningless in batch mode.
        monkeypatch.delenv("SLURMATE_NO_BANNER", raising=False)
        monkeypatch.delenv("SLURMATE_BANNER_ANIMATE", raising=False)
        print_banner(interactive=False)
        out = capsys.readouterr().out
        assert "ESC to go back" not in out

    def test_interactive_shows_esc_hint(self, capsys, monkeypatch):
        monkeypatch.delenv("SLURMATE_NO_BANNER", raising=False)
        monkeypatch.delenv("SLURMATE_BANNER_ANIMATE", raising=False)
        print_banner(interactive=True)
        out = capsys.readouterr().out
        assert "ESC to go back" in out

    def test_no_banner_env(self, capsys, monkeypatch):
        monkeypatch.setenv("SLURMATE_NO_BANNER", "1")
        print_banner()
        assert capsys.readouterr().out == ""

    def test_no_banner_falsey_value_still_shows(self, capsys, monkeypatch):
        # Bare truthiness meant SLURMATE_NO_BANNER=0 wrongly suppressed the
        # banner; only affirmative values (1/true/yes/on) should hide it.
        monkeypatch.setenv("SLURMATE_NO_BANNER", "0")
        monkeypatch.delenv("SLURMATE_BANNER_ANIMATE", raising=False)
        print_banner(interactive=False)
        assert "Slurmate" in capsys.readouterr().out


class TestForceColor:
    """L8: rich honours FORCE_COLOR, so the plain-ANSI path must too."""

    def test_force_color_enables_color_off_a_tty(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert theme._should_use_color() is True
        c = C()
        assert "38;2;" in c.RED

    def test_no_color_still_wins(self, monkeypatch):
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.setenv("NO_COLOR", "1")
        assert theme._should_use_color() is False

    def test_dumb_terminal_still_wins(self, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.setenv("TERM", "dumb")
        assert theme._should_use_color() is False

    def test_clicolor_force_is_not_honored(self, monkeypatch):
        # rich ignores CLICOLOR_FORCE (measured), so honouring it here would
        # recreate the very half-coloured output this alignment removes.
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        monkeypatch.setenv("CLICOLOR_FORCE", "1")
        import sys
        monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
        assert theme._should_use_color() is False

    def test_animation_still_requires_a_real_tty(self, capsys, monkeypatch):
        # FORCE_COLOR can now enable colour on a non-TTY, but cursor-control
        # animation into a pipe would just emit escape soup.
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.setenv("SLURMATE_BANNER_ANIMATE", "1")
        print_banner(interactive=False)
        out = capsys.readouterr().out
        assert "\033[s" not in out and "\033[u" not in out   # no save/restore
        assert "Slurmate" in out

    def test_matches_rich_on_edge_values(self, monkeypatch):
        # Parity is the whole point of honouring it, so pin the edges against the
        # installed rich: an empty value is NOT a force, "0" is (rich agrees).
        from rich.console import Console
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.delenv("TERM", raising=False)
        for value in ("", "0", "1", "true"):
            monkeypatch.setenv("FORCE_COLOR", value)
            assert theme._should_use_color() is Console().is_terminal, value
