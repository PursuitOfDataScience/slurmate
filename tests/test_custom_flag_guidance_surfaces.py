"""Both places a custom flag is typed show BOTH value forms.

`issues.md` P6 declines to guess at an unquoted spaced value, and says what shipped
instead: "the guidance where the value is typed. The wizard's custom-flags subtitle
and the `--custom-sbatch` help now both say a value may use `=` **or** a space, and
that a value containing a space must be quoted."

The `--custom-sbatch` help did say it. The wizard subtitle did not: it showed
`--exclusive --reservation=abc` and the quoting rule, and never mentioned that a value
may follow its option after a space. The wizard is the surface where most values are
typed, and the space form is genuinely supported -- `_parse_custom_flags("-C bigmem")`
returns one flag, and `-C bigmem` reaches the script as `#SBATCH -C bigmem`. So the
one form a user could not discover was the one the parser had gained support for.

The subtitle is also reused verbatim as the validation error (`tui.py`: `f"Invalid
input: {s.subtitle}"`), which is why the fix adds `-C bigmem` to the EXAMPLE rather
than a sentence of prose: that line has to stay readable as a rejection message.
"""

import pytest

from slurmate.builder import _quote_custom_flag
from slurmate.tui import STEPS, _parse_custom_flags

#: The two forms a value may take, as they appear in guidance.
EQUALS_EXAMPLE = "=abc"
SPACE_EXAMPLE = "-C bigmem"


def _subtitle() -> str:
    return next(s for s in STEPS if s.key == "custom_sbatch").subtitle


def _cli_help(capsys: pytest.CaptureFixture[str]) -> str:
    from slurmate.main import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--help"])
    return " ".join(capsys.readouterr().out.split())


class TestBothSurfacesShowBothForms:
    def test_the_wizard_subtitle_shows_the_space_form(self) -> None:
        assert SPACE_EXAMPLE in _subtitle(), _subtitle()

    def test_the_wizard_subtitle_still_shows_the_equals_form(self) -> None:
        assert EQUALS_EXAMPLE in _subtitle(), _subtitle()

    def test_the_wizard_subtitle_still_says_to_quote_a_spaced_value(self) -> None:
        assert "quote" in _subtitle().lower(), _subtitle()

    def test_the_cli_help_shows_both_forms_and_the_quoting_rule(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        help_text = _cli_help(capsys)
        assert "or a space" in help_text, help_text
        assert "quote" in help_text.lower()

    def test_the_subtitle_stays_short_enough_to_read_as_a_rejection(self) -> None:
        """It doubles as `Invalid input: <subtitle>`, so prose is expensive here."""
        assert len(_subtitle()) <= 130, len(_subtitle())


class TestTheGuidanceMatchesTheParser:
    """Guidance that promises a form the parser rejects would be worse than silence."""

    def test_the_space_form_really_yields_one_flag(self) -> None:
        assert _parse_custom_flags("-C bigmem") == ["-C bigmem"]

    def test_the_space_form_reaches_the_script_intact(self) -> None:
        flags = _parse_custom_flags("-C bigmem")
        assert [_quote_custom_flag(f) for f in flags] == ["-C bigmem"]

    def test_the_quoting_rule_is_real(self) -> None:
        # What the guidance tells the user to do, end to end.
        flags = _parse_custom_flags('--comment="my run"')
        assert [_quote_custom_flag(f) for f in flags] == ['--comment="my run"']


class TestControls:
    """None of these reads the guidance text, so all hold with the fix in or out."""

    def test_a_valueless_flag_is_untouched(self) -> None:
        assert _parse_custom_flags("--exclusive") == ["--exclusive"]

    def test_the_dashless_form_still_gains_its_dashes(self) -> None:
        assert _parse_custom_flags("exclusive") == ["--exclusive"]

    def test_a_comma_inside_a_value_is_still_kept(self) -> None:
        assert _parse_custom_flags("--exclude=node1,node2") == ["--exclude=node1,node2"]

    def test_a_comma_between_flags_still_separates_them(self) -> None:
        assert _parse_custom_flags("-C bigmem,--exclusive") == ["-C bigmem", "--exclusive"]

    def test_every_step_still_has_a_subtitle_string(self) -> None:
        assert all(isinstance(s.subtitle, str) for s in STEPS)
