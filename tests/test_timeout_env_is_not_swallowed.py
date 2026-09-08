"""`$SLURMATE_TIMEOUT` swallowed every value it could not use.

`total_timeout` reads the one environment variable that changes how long slurmate
will wait on a degraded controller. Three forms are unusable, and all three fell
through to the default behind a `logger.debug` -- which nothing shows by default.
Measured before this change:

    SLURMATE_TIMEOUT='garbage' -> 45.0     silently
    SLURMATE_TIMEOUT='0'       -> 45.0     silently
    SLURMATE_TIMEOUT='-5'      -> 45.0     silently

i.e. each was indistinguishable from leaving the variable unset, which is the
state a reader who set it believes they are *not* in.

The sibling packages settled this and named the principle. slurmpast's `_timeout`
says it in as many words -- "it used to accept anything: `garbage` was
indistinguishable from leaving it unset, and `0` -- which a reader naturally
writes meaning *no timeout* -- silently became 300 seconds with nothing said. A
setting that does not do what it says and does not complain is worse than one
that is not offered." -- and slurmwatch rejects the same class by name. Zero gets
its own message here for that reason: writing it is a request to remove the
budget, and quietly getting 45s is the opposite answer.

**The returned numbers are unchanged.** This moves the report from silent to
warned, not the behaviour; `TestControls` pins every value, including the two
valid ones, so a reader's configured budget cannot have shifted under the fix.

Warning rather than raising is deliberate and argued in the docstring: this
variable is read by `slurm_deadline` on the interactive wizard path, where an
exception would replace a working session with a traceback over a typo.
`logger.warning` reaches stderr through logging's last-resort handler, measured.
"""

from __future__ import annotations

import logging
import os
from unittest import mock

import pytest

from slurmate import system_utils as su

#: Values the function cannot use, all of which used to be silent.
UNUSABLE = ["garbage", "0", "-5", "0.0", "1e-9x", "abc123"]


def _read(monkeypatch_value: str | None):
    env = {} if monkeypatch_value is None else {su._TOTAL_TIMEOUT_ENV: monkeypatch_value}
    ctx = mock.patch.dict(os.environ, env, clear=monkeypatch_value is None)
    with ctx:
        return su.total_timeout()


class TestAnUnusableSettingIsSaidOutLoud:
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_warns_rather_than_debug_logging(self, value, caplog) -> None:
        caplog.set_level(logging.WARNING, logger=su.logger.name)
        with mock.patch.dict(os.environ, {su._TOTAL_TIMEOUT_ENV: value}):
            su.total_timeout()
        assert caplog.records, f"{value!r} was swallowed"
        assert all(r.levelno == logging.WARNING for r in caplog.records)

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_message_names_the_variable_the_value_and_the_budget(
        self, value, caplog
    ) -> None:
        caplog.set_level(logging.WARNING, logger=su.logger.name)
        with mock.patch.dict(os.environ, {su._TOTAL_TIMEOUT_ENV: value}):
            su.total_timeout()
        text = " ".join(r.getMessage() for r in caplog.records)
        assert su._TOTAL_TIMEOUT_ENV in text, text
        assert repr(value) in text or value in text, text
        assert "45s" in text, text  # the budget that actually applied

    def test_zero_gets_its_own_answer(self, caplog) -> None:
        """Writing 0 is a request to remove the budget, not a typo."""
        caplog.set_level(logging.WARNING, logger=su.logger.name)
        with mock.patch.dict(os.environ, {su._TOTAL_TIMEOUT_ENV: "0"}):
            su.total_timeout()
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "no setting that removes the budget" in text, text
        assert "not a number" not in text, "zero is a number; the wrong branch fired"

    def test_a_non_number_is_told_it_is_not_a_number(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger=su.logger.name)
        with mock.patch.dict(os.environ, {su._TOTAL_TIMEOUT_ENV: "garbage"}):
            su.total_timeout()
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "is not a number" in text, text
        assert "removes the budget" not in text, "the non-positive branch fired"

    def test_the_two_causes_do_not_share_one_message(self, caplog) -> None:
        """They want different actions, so they must not read the same."""
        messages = {}
        for value in ("garbage", "-5"):
            caplog.clear()
            caplog.set_level(logging.WARNING, logger=su.logger.name)
            with mock.patch.dict(os.environ, {su._TOTAL_TIMEOUT_ENV: value}):
                su.total_timeout()
            messages[value] = " ".join(r.getMessage() for r in caplog.records)
        assert messages["garbage"] != messages["-5"], messages


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("garbage", 45.0),
            ("0", 45.0),
            ("-5", 45.0),
            ("60", 60.0),
            ("  30  ", 30.0),
            ("", 45.0),
            (None, 45.0),
        ],
    )
    def test_every_returned_budget_is_unchanged(self, value, expected) -> None:
        # The whole point of the fix being a REPORT and not a behaviour change:
        # nobody's configured budget moved.
        assert _read(value) == expected

    def test_a_valid_setting_says_nothing(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger=su.logger.name)
        with mock.patch.dict(os.environ, {su._TOTAL_TIMEOUT_ENV: "90"}):
            assert su.total_timeout() == 90.0
        assert not caplog.records, [r.getMessage() for r in caplog.records]

    def test_an_unset_variable_says_nothing(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger=su.logger.name)
        with mock.patch.dict(os.environ, {}, clear=True):
            assert su.total_timeout() == su.DEFAULT_TOTAL_TIMEOUT
        assert not caplog.records, [r.getMessage() for r in caplog.records]

    def test_the_default_and_the_variable_name_are_unchanged(self) -> None:
        assert su.DEFAULT_TOTAL_TIMEOUT == 45.0
        assert su._TOTAL_TIMEOUT_ENV == "SLURMATE_TIMEOUT"

    def test_the_deadline_still_opens_with_that_budget(self) -> None:
        """`slurm_deadline` yields the DEADLINE INSTANT, not the budget.

        An earlier draft of this test asserted the budget and read back
        `11072501.95` -- a `time.monotonic()` value. `_PHASE_BUDGET` is where the
        figure lands, which is also what the "budget exceeded" message quotes.
        """
        with (
            mock.patch.dict(os.environ, {su._TOTAL_TIMEOUT_ENV: "12"}),
            su.slurm_deadline(),
        ):
            assert su._PHASE_BUDGET == 12.0
        # ...and an unusable value still opens the phase, on the default.
        with (
            mock.patch.dict(os.environ, {su._TOTAL_TIMEOUT_ENV: "garbage"}),
            su.slurm_deadline(),
        ):
            assert su._PHASE_BUDGET == su.DEFAULT_TOTAL_TIMEOUT

    def test_an_explicit_argument_still_beats_the_environment(self) -> None:
        with (
            mock.patch.dict(os.environ, {su._TOTAL_TIMEOUT_ENV: "99"}),
            su.slurm_deadline(total=7.0),
        ):
            assert su._PHASE_BUDGET == 7.0

    def test_a_nested_phase_still_keeps_the_outer_deadline(self) -> None:
        # The documented rule: "a phase cannot extend its own budget by opening
        # another one."
        with (
            mock.patch.dict(os.environ, {su._TOTAL_TIMEOUT_ENV: "20"}),
            su.slurm_deadline() as outer,
        ):
            # This inner `with` stays nested on purpose: nesting is what the
            # test is about, so merging it away would delete the subject.
            with su.slurm_deadline(total=999.0) as inner:
                assert inner == outer
            assert su._PHASE_BUDGET == 20.0
