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
