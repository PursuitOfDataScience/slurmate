"""SM-29 and SM-30, the two findings still open after the 0.7.0 round.

* **SM-29** -- the per-call timeout was the only one there was, so against an
  unresponsive ``slurmctld`` the worst case was the *sum* of the calls.  A sibling
  package in this suite (``rapidu.quota._budget``) contains both the fix and a
  docstring describing this precise failure mode.  Measured here: **141 s** with
  no total bound.
* **SM-30** -- ``--custom-sbatch='--comment=my run'`` emitted ``#SBATCH
  --comment=my`` followed by ``#SBATCH --run``, which sbatch refuses outright, and
  ``--dry-run`` -- the mode whose entire purpose is "would this work" -- reported
  that script as ``ETA: now``.
"""

import sys
import time

import pytest

from slurmate import system_utils as SU
from slurmate.builder import _normalize_custom_flags, unquoted_custom_values
from slurmate.system_utils import _read_test_only_output, refusal_is_permanent


@pytest.fixture(autouse=True)
def _no_leftover_deadline(monkeypatch):
    """No test inherits or leaves a deadline: it is module state."""
    monkeypatch.setattr(SU, "_DEADLINE", None)
    monkeypatch.delenv("SLURMATE_TIMEOUT", raising=False)


# --------------------------------------------------------------------------
# SM-29 -- a bound on the total, not only on each call
# --------------------------------------------------------------------------
class TestTheTotalSlurmTimeIsBounded:
    """One ``--dry-run`` makes four Slurm invocations, each granted the full 30 s.

    The per-call handler is clean -- no traceback, a usable message -- and that is
    exactly what made this easy to miss: nothing was broken, the process was
    merely silent for two minutes.  Every fix that teaches slurmate to consult
    more of the cluster (the SelectType read, the association check, the QOS
    MaxWall) adds another 30 s to the worst case.
    """

    def test_a_call_is_capped_by_what_is_left(self):
        with SU.slurm_deadline(10.0):
            assert SU._budget(30) == pytest.approx(10.0, abs=0.5)

    def test_a_call_past_the_deadline_is_not_run_at_all(self, monkeypatch):
        ran = []
        monkeypatch.setattr(
            SU.subprocess,
            "run",
            lambda *a, **k: ran.append(a) or (_ for _ in ()).throw(AssertionError),
        )
        with SU.slurm_deadline(0.01):
            time.sleep(0.05)
            out, err, rc = SU._run_command(["sinfo", "-h"])
        assert ran == [], "a fork and an exec spent to learn nothing"
        assert rc == -1
        assert "total budget" in err
        assert "SLURMATE_TIMEOUT" in err, "and it says how to change it"

    def test_the_sum_of_many_calls_cannot_exceed_the_total(self, monkeypatch):
        """The whole finding, in one assertion.

        Each call asks for the full per-call timeout and each one is capped by
        what remains, so ten calls against a wedged controller cost the budget
        rather than ten times the per-call bound.
        """
        asked: list[float] = []

        def _slow(*_a, **kwargs):
            asked.append(kwargs["timeout"])
            time.sleep(kwargs["timeout"])
            raise SU.subprocess.TimeoutExpired(cmd="sinfo", timeout=kwargs["timeout"])

        monkeypatch.setattr(SU.subprocess, "run", _slow)
        started = time.monotonic()
        with SU.slurm_deadline(0.6):
            for _ in range(10):
                SU._run_command(["sinfo", "-h"], timeout=30)
        elapsed = time.monotonic() - started
        assert elapsed < 3.0, f"{elapsed:.1f}s for a 0.6s budget"
        assert sum(asked) <= 1.2, asked
        assert max(asked) <= 0.6, "no single call outlived the budget"

    def test_no_deadline_leaves_each_call_its_own_timeout(self):
        # The right behaviour for a direct API caller and for a long-lived
        # interactive session, where a process-wide deadline would expire while
        # the user was reading the screen.
        assert SU._budget(30) == 30

    def test_a_nested_block_cannot_extend_the_budget(self):
        with SU.slurm_deadline(0.2):
            outer = SU._DEADLINE
            with SU.slurm_deadline(600.0):
                assert outer == SU._DEADLINE
            assert outer == SU._DEADLINE

    def test_the_deadline_is_released_on_the_way_out(self):
        with SU.slurm_deadline(1.0):
            assert SU._DEADLINE is not None
        assert SU._DEADLINE is None

    def test_the_deadline_is_released_after_an_exception(self):
        with pytest.raises(RuntimeError), SU.slurm_deadline(1.0):
            raise RuntimeError("boom")
        assert SU._DEADLINE is None

    @pytest.mark.parametrize("raw,expected", [("12", 12.0), ("0.5", 0.5)])
    def test_the_env_override_is_honoured(self, monkeypatch, raw, expected):
        monkeypatch.setenv("SLURMATE_TIMEOUT", raw)
        assert SU.total_timeout() == expected

    @pytest.mark.parametrize("raw", ["", "abc", "0", "-5"])
    def test_a_nonsense_override_falls_back_to_the_default(self, monkeypatch, raw):
        # A site's stale `export` must not disable the bound or set it to zero,
        # which would skip every query and report the cluster as unreachable.
        monkeypatch.setenv("SLURMATE_TIMEOUT", raw)
        assert SU.total_timeout() == SU.DEFAULT_TOTAL_TIMEOUT

    def test_the_batch_path_shares_one_deadline_across_both_phases(
        self,
        monkeypatch,
        capsys,
    ):
        """`run_batch` and `build_and_show` are two phases of one decision.

        Each opens its own deadline -- they have to, since on the interactive
        path they are minutes apart -- so with no outer one the batch path's
        bound was the sum of the two. Measured against a controller that never
        answers: 141 s with no bound at all, 2x the budget with one per phase.
        """
        from slurmate import main as M

        seen: list[float | None] = []

        def _fake_run_batch(args, console, config):
            seen.append(SU._DEADLINE)
            return {
                "job_name": "p",
                "partition": "build",
                "account": "a",
                "cpus": 1,
                "memory": "1G",
                "time_limit": "00:05:00",
                "command": "echo hi",
                "nodes": 1,
            }

        def _fake_build_and_show(answers, console, **_kw):
            seen.append(SU._DEADLINE)
            return "#!/bin/bash\n", {}

        monkeypatch.setattr(M, "run_batch", _fake_run_batch)
        monkeypatch.setattr(M, "build_and_show", _fake_build_and_show)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "slurmate",
                "-p",
                "build",
                "-A",
                "a",
                "-c",
                "1",
                "--memory=1G",
                "-t",
                "00:05:00",
                "-J",
                "p",
                "--command",
                "echo hi",
                "--dry-run",
            ],
        )
        M.main()
        assert len(seen) == 2, seen
        assert seen[0] is not None, "the batch path opens one before any query"
        assert seen[0] == seen[1], "and the second phase reuses it"
        # And it does not outlive the run.
        assert SU._DEADLINE is None

    def test_the_wizard_path_opens_none_up_front(self, monkeypatch):
        """A user takes minutes to answer.

        A deadline opened before the first question would expire while they read
        the screen and then report the cluster as unreachable, so the interactive
        path gets its bound per build instead.
        """
        from slurmate import main as M

        seen: list[float | None] = []

        class _Wizard:
            def run(self):
                seen.append(SU._DEADLINE)
                return None

        # Patched where it is DEFINED, not on one importer's copy: `main` now
        # imports the wizard inside the branch that needs it, so that
        # `prompt_toolkit` is not paid for by a non-interactive run.
        import slurmate.tui as tuimod

        monkeypatch.setattr(tuimod, "Wizard", _Wizard)
        monkeypatch.setattr(M, "_require_terminal_for_wizard", lambda: None)
        monkeypatch.setattr(M, "_is_batch_mode", lambda *a, **k: False)
        monkeypatch.setattr(sys, "argv", ["slurmate"])
        # A cancelled wizard returns rather than exiting; either is fine here.
        M.main()
        assert seen == [None]

    def test_the_deadline_is_closed_before_the_action_menu(self, monkeypatch):
        """A deadline must not span a human's time in ``$EDITOR``.

        Batch mode WITHOUT ``--yes``/``--print``/``--dry-run`` falls through to
        the action menu, and the batch branch's deadline was still open there. A
        user who picks "Open script in $EDITOR" is gone for as long as it takes
        them to edit, so the budget expires and every submit-time gate then gets
        ``rc=-1`` from `_run_command`'s "skipped" branch -- which reads as *no
        problem*. Measured: inside an expired deadline a script containing
        ``#SBATCH --nosuchopt`` gives ``check_script_with_scheduler(...) == ''``,
        where outside it gives the refusal.
        """
        import types

        import questionary

        from slurmate import main as M

        seen: list[tuple[str, float | None]] = []
        actions = iter(["Open script in vi", "Submit to Slurm"])

        class _Prompt:
            def __init__(self) -> None:
                self.application = types.SimpleNamespace(key_bindings=None)

            def ask(self) -> str:
                return next(actions)

        monkeypatch.setattr(questionary, "select", lambda *a, **k: _Prompt())
        monkeypatch.setattr(
            M,
            "run_batch",
            lambda args, console, config: {
                "job_name": "p",
                "partition": "build",
                "account": "a",
                "cpus": 1,
                "memory": "1G",
                "time_limit": "00:05:00",
                "command": "echo hi",
                "nodes": 1,
            },
        )
        monkeypatch.setattr(
            M,
            "build_and_show",
            lambda answers, console, **_kw: ("#!/bin/bash\n", {}),
        )
        monkeypatch.setattr(
            M,
            "_edit_script_in_editor",
            lambda script: (seen.append(("editor", SU._DEADLINE)), script)[1],
        )
        monkeypatch.setattr(
            M,
            "check_script_with_scheduler",
            lambda script: (seen.append(("gate", SU._DEADLINE)), "")[1],
        )
        monkeypatch.setattr(
            M,
            "_submit_and_report",
            lambda *a, **k: seen.append(("submit", SU._DEADLINE)),
        )
        monkeypatch.setattr(M, "_show_script_and_summary", lambda *a, **k: None)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "slurmate",
                "-p",
                "build",
                "-A",
                "a",
                "-c",
                "1",
                "--memory=1G",
                "-t",
                "00:05:00",
                "-J",
                "p",
                "--command",
                "echo hi",
            ],
        )
        M.main()
        assert [name for name, _d in seen] == ["editor", "gate", "submit"], seen
        assert all(d is None for _n, d in seen), seen

    def test_an_expired_deadline_would_have_silenced_the_gate(self, monkeypatch):
        """Why the test above is worth having: what the gate answers when the
        budget is gone.

        ``''`` is the same value a clean script produces, so an expired deadline
        does not degrade the gate -- it inverts it.
        """
        monkeypatch.setattr(
            SU.subprocess,
            "run",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run past the deadline")),
        )
        script = "#!/bin/bash\n#SBATCH --nosuchopt\necho hi\n"
        with SU.slurm_deadline(0.01):
            time.sleep(0.05)
            assert SU.check_script_with_scheduler(script) == ""


# --------------------------------------------------------------------------
# SM-30 (1) -- an unquoted value became an option nobody typed
# --------------------------------------------------------------------------
class TestAnUnquotedValueDoesNotBecomeADirective:
    """``--help`` does say to quote such a value, and the quoted forms are
    correct -- this is not a request to support unquoted spaces.  The defect is
    that the third row was emitted *silently*, from a tool whose output is a
    script somebody submits.

    ============================== ==================================
    ``--custom-sbatch=``           emitted
    ============================== ==================================
    ``--comment="my run"``         ``#SBATCH --comment="my run"``
    ``--comment='my run'``         ``#SBATCH --comment="my run"``
    ``--comment=my run``           ``--comment=my`` + ``--run``
    ============================== ==================================
    """

    def test_the_invalid_directive_is_gone(self):
        flags = _normalize_custom_flags("--comment=my run")
        assert flags == ["--comment=my run"]
        assert not any(f.startswith("--run") for f in flags)

    @pytest.mark.parametrize(
        "value",
        ['--comment="my run"', "--comment='my run'", "--comment=my run"],
    )
    def test_all_three_spellings_agree(self, value):
        assert _normalize_custom_flags(value) == ["--comment=my run"]

    def test_the_script_carries_a_directive_sbatch_accepts(self):
        from slurmate.builder import build_sbatch_script

        script = build_sbatch_script(
            job_name="p",
            partition="build",
            account="a",
            cpus=1,
            memory="1G",
            time_limit="00:05:00",
            command="echo hi",
            custom_sbatch=["--comment=my run"],
        )
        assert '#SBATCH --comment="my run"' in script
        assert "#SBATCH --run" not in script

    def test_the_fold_is_reported(self):
        assert unquoted_custom_values("--comment=my run") == [("--comment", "my run")]

    def test_a_quoted_value_is_not_reported(self):
        # Nothing to say: the user wrote what they meant.
        assert unquoted_custom_values('--comment="my run"') == []

    @pytest.mark.parametrize(
        "value,expected",
        [
            # The documented behaviour that must not regress: a bare word after a
            # VALUELESS option is an option the user wrote without dashes.
            ("--exclusive hold", ["--exclusive", "--hold"]),
            # A known value-taking flag still takes its value.
            ("-C bigmem", ["-C bigmem"]),
            # And a word that cannot be an option name is still a value.
            ("-o /logs/%j.out", ["-o /logs/%j.out"]),
        ],
    )
    def test_the_existing_forms_are_untouched(self, value, expected):
        assert _normalize_custom_flags(value) == expected
        assert unquoted_custom_values(value) == []

    def _run_batch(self, custom):
        from rich.console import Console

        from slurmate.main import parse_args, run_batch

        args = parse_args(
            [
                "-p",
                "build",
                "-A",
                "a",
                "-c",
                "1",
                "--memory=1G",
                "-t",
                "00:05:00",
                "-J",
                "p",
                "--command",
                "echo hi",
                "--custom-sbatch=" + custom,
                "--print",
                "--force",
            ]
        )
        code = None
        try:
            run_batch(args, Console(), {})
        except SystemExit as exc:
            code = exc.code
        return code

    def test_the_batch_path_refuses_rather_than_warning(self, capsys):
        """It warned, and a warning is not enough.

        Whichever reading is taken, the other one's information is gone:
        `--custom-sbatch='--array=1-10 hold'` is either an array spec of
        `1-10 hold` or `--array=1-10 --hold`, and a tool that guesses either
        writes a script the user did not ask for. So it exits 1 and names both
        readings -- the same choice SM-15 settled for a custom flag that
        duplicates a managed directive.
        """
        code = self._run_batch("--comment=my run")
        # Whitespace-collapsed: rich hard-wraps to the terminal width, so the
        # sentence being asserted is split across lines at a column that depends
        # on the machine running the test.
        err = " ".join(capsys.readouterr().err.split())
        assert code == 1, "a warning let the run continue"
        assert "Error" in err
        assert "cannot be an sbatch option" in err
        # Both readings, so the user picks rather than the tool.
        assert '--comment="my run"' in err
        assert "--comment=my --run" in err

    @pytest.mark.parametrize(
        "custom",
        ['--comment="my run"', "--comment='my run'", "--exclusive"],
    )
    def test_a_correctly_quoted_value_produces_no_error(self, capsys, custom):
        # The control: the refusal must not fire on the forms `--help` invites.
        code = self._run_batch(custom)
        err = " ".join(capsys.readouterr().err.split())
        assert code is None, err
        assert "cannot be an sbatch option" not in err

    def test_the_wizard_half_is_wired_as_an_error(self):
        """`site_check_issues` classifies it "error", not "warning".

        Asserted through the reporter rather than through `site_check_issues`,
        because the wizard hands it an already-split LIST (`tui._parse_custom_flags`
        returns `['--comment=my run']`) and a list element has no fold to report --
        `_normalize_custom_flags` leaves it exactly as given. So the level is what
        this can honestly pin here.
        """
        import inspect

        from slurmate.main import site_check_issues

        src = inspect.getsource(site_check_issues)
        where = src.index("unquoted_custom_values")
        assert '"error"' in src[where - 200 : where + 400]


# --------------------------------------------------------------------------
# SM-30 (2) -- a rejected script reported as submittable
# --------------------------------------------------------------------------
class TestSbatchRejectingItsOwnDirectivesIsARefusal:
    """The deeper half, and true of any malformed directive rather than only of
    the split above.

    ``_test_only_refusal`` requires positive evidence -- ``allocation failure:``
    or ``Reason:`` -- and that rule is right: a non-zero exit alone also means
    "the controller is unreachable", and reading that as "your job can never run"
    trades one confident wrong answer for another.  sbatch rejecting an option in
    the file is a *third* case, it carries neither marker, and it fell into the
    "no verdict" branch -- so ``--dry-run`` fell through to the free-capacity
    estimate and answered ``ETA: now`` about a script sbatch had just refused.
    """

    @pytest.mark.parametrize(
        "stderr",
        [
            "sbatch: unrecognized option '--run'",
            "sbatch: error: unrecognized option '--nosuchflag'",
            "sbatch: invalid option -- 'Z'",
            "sbatch: option '--comment' requires an argument",
        ],
    )
    def test_it_is_read_as_a_refusal(self, stderr):
        start, refusal = _read_test_only_output("", stderr, 1)
        assert start is None
        assert refusal, f"{stderr!r} produced no verdict"
        # The wording the fix settled on. Pinned because it is also the phrase
        # `_PERMANENT_REFUSAL_MARKERS` keys on, so the two must not drift apart.
        assert "rejected an option in the script" in refusal

    @pytest.mark.parametrize(
        "stderr",
        [
            "sbatch: unrecognized option '--run'",
            "sbatch: error: unrecognized option '--nosuchflag'",
        ],
    )
    def test_it_is_permanent(self, stderr):
        # A directive sbatch cannot parse is not going to start parsing later, so
        # there is nothing to retry and the submit guard should block.
        _start, refusal = _read_test_only_output("", stderr, 1)
        assert refusal_is_permanent(refusal)

    def test_the_offending_token_is_named(self):
        _start, refusal = _read_test_only_output(
            "", "sbatch: unrecognized option '--nosuchflag'", 1
        )
        assert "--nosuchflag" in refusal

    @pytest.mark.parametrize(
        "stderr",
        [
            "sbatch: error: Batch job submission failed: Unable to contact slurm controller",
            "sbatch: error: slurm_receive_msg: Zero Bytes were transmitted",
            "",
        ],
    )
    def test_an_unreachable_controller_is_still_not_a_refusal(self, stderr):
        # The branch this fix must not widen: no positive evidence, no verdict.
        _start, refusal = _read_test_only_output("", stderr, 1)
        assert refusal == ""

    def test_a_policy_refusal_is_unchanged(self):
        _start, refusal = _read_test_only_output(
            "", "allocation failure: Invalid account or account/partition combination specified", 1
        )
        assert refusal == ("Invalid account or account/partition combination specified")

    def test_a_placement_line_still_wins(self):
        # An accepted job must not be turned into a refusal by anything here.
        from datetime import datetime, timedelta

        when = (datetime.now() + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")
        start, refusal = _read_test_only_output(
            "", f"sbatch: Job 1 to start at {when} using 1 processors", 0
        )
        assert refusal == ""
        assert start is not None and start > 0


class TestARejectionSaysWhatWouldHaveWorked:
    """Polish pass, 2026-08-28: three hard rejections taught the user nothing.

    `--memory`, `--mem-per-cpu` and `--time` each printed the offending value and
    exited 1. Every other refusal in this package names the remedy — an unknown
    partition lists the real ones, a rejected flag names `--force` — so these
    three were the odd ones out, and they are the errors a first-time user is
    most likely to hit ("16 gigs", "2 hours").

    The danger in fixing it is advertising a form the validator does not take, so
    what is pinned here is not the wording: every example the message offers is
    fed back through the validator that produced the rejection.
    """

    @staticmethod
    def _examples(text):
        # The forms are quoted in backticks; the word spellings are bare capitals.
        import re

        return re.findall(r"`([^`]+)`", text) + re.findall(r"\b([A-Z]{4,})\b", text)

    def test_every_memory_form_offered_is_accepted(self):
        from slurmate.system_utils import MEMORY_FORMS, validate_memory

        offered = self._examples(MEMORY_FORMS)
        assert offered, MEMORY_FORMS
        rejected = [v for v in offered if not validate_memory(v)]
        assert not rejected, f"the message offers {rejected}, which `validate_memory` refuses"

    def test_every_time_form_offered_is_accepted(self):
        from slurmate.system_utils import time_forms, validate_time

        # `D-HH` and friends are shapes, not values, so they are instantiated.
        shapes = {
            "MM:SS": "45:30",
            "HH:MM:SS": "02:30:00",
            "D-HH": "1-12",
            "D-HH:MM": "1-12:30",
            "D-HH:MM:SS": "1-12:30:45",
        }
        offered = [shapes.get(v, v) for v in self._examples(time_forms())]
        assert offered, time_forms()
        rejected = [v for v in offered if not validate_time(v)]
        assert not rejected, f"the message offers {rejected}, which `validate_time` refuses"

    def test_the_shape_list_covers_every_pattern(self):
        """The message must not describe fewer forms than the validator takes.

        A user reading it should not conclude a spelling is unsupported when it
        is — that is the same misdirection as the original silence, inverted.
        """
        from slurmate.system_utils import _TIME_PATTERNS, time_forms

        text = time_forms()
        # One offered example per pattern, counted by matching each back.
        import re

        shapes = ["90", "45:30", "02:30:00", "1-12", "1-12:30", "1-12:30:45"]
        for pattern in _TIME_PATTERNS:
            assert any(re.match(pattern, s) for s in shapes), (
                f"{pattern} is accepted but no example in the message matches it"
            )
        assert len(shapes) == len(_TIME_PATTERNS), (
            f"{len(_TIME_PATTERNS)} patterns but {len(shapes)} examples — the "
            f"message and the grammar have diverged"
        )
        assert "`90`" in text

    def test_a_rejection_actually_prints_the_guidance(self):
        """End to end, because the helper being right is not the same as it being used."""
        import os
        import pathlib
        import subprocess
        import sys

        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "slurmate",
                "--print",
                "--job-name",
                "x",
                "--partition",
                "p",
                "--cpus",
                "1",
                "--command",
                "true",
                "--memory",
                "16 gigs",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env={
                **os.environ,
                "PYTHONPATH": str(pathlib.Path(__file__).parent.parent / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_COLOR": "1",
                "COLUMNS": "200",
            },
        )
        assert done.returncode == 1
        combined = done.stdout + done.stderr
        assert "Invalid memory value: 16 gigs" in combined, combined
        assert "16G" in combined, "the rejection did not say what would have worked"


class TestCtrlCIsNotACrash:
    """Polish pass, 2026-08-28: SIGINT escaped as a six-line traceback.

    Measured against a `sinfo`/`sacctmgr` that never returns, interrupted after
    six seconds: a `KeyboardInterrupt` traceback out of
    `subprocess.communicate`'s selector poll. That wait is exactly where it
    happens, because it is the only part of a run long enough to interrupt -- and
    a traceback there reads as a crash in a tool the user was cancelling on
    purpose.

    Three of the five sibling tools already exited cleanly here; one of their
    comments names the same six-line traceback. 130 is the shell's convention for
    "terminated by SIGINT", so a wrapper can tell a cancellation from a failure.
    """

    @staticmethod
    def _interrupt_during(argv, wait=4.0):
        """Run slurmate against hanging Slurm clients and SIGINT it.

        Returns ``(returncode, stderr, pgid)``. The group is swept before
        returning and the caller can assert it is empty.
        """
        import contextlib
        import os
        import pathlib
        import signal
        import subprocess
        import sys
        import tempfile
        import time

        root = pathlib.Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            fake = pathlib.Path(tmp) / "bin"
            fake.mkdir()
            for name in ("sinfo", "squeue", "sacct", "scontrol", "sacctmgr", "sbatch"):
                stub = fake / name
                stub.write_text("#!/bin/bash\nsleep 300\n")
                stub.chmod(0o755)
            proc = subprocess.Popen(
                [sys.executable, "-m", "slurmate", *argv],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tmp,
                env={
                    "PATH": f"{fake}:/usr/bin:/bin",
                    "HOME": os.environ.get("HOME", tmp),
                    "PYTHONPATH": str(root / "src"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "NO_COLOR": "1",
                    "COLUMNS": "200",
                },
                # Its own session, so the stubs it spawns are a group of their own
                # and the sweep below can reach them. Without this they sit in
                # pytest's group, where there is nothing safe to sweep them with.
                start_new_session=True,
            )
            time.sleep(wait)
            proc.send_signal(signal.SIGINT)
            try:
                _out, err = proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                _out, err = proc.communicate()
                pytest.fail("SIGINT did not stop it")
            finally:
                # The `sleep 300` stub outlives the slurmate that ran it: the
                # `KeyboardInterrupt` unwinds through `subprocess.run`, which kills
                # the `bash` it started, and the `sleep` that bash forked is
                # orphaned onto init for the full five minutes. Measured: a stub
                # from this test still running minutes after the suite had
                # finished, one per run of the suite. `proc.pid` is the group's
                # leader, and nothing but this invocation is in it.
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
            return proc.returncode, err, proc.pid

    def test_it_exits_130_without_a_traceback(self):
        rc, err, _pgid = self._interrupt_during(
            [
                "--print",
                "--job-name",
                "x",
                "--partition",
                "p",
                "--cpus",
                "1",
                "--time",
                "00:05:00",
                "--command",
                "true",
            ]
        )
        assert "Traceback" not in err, err[-400:]
        assert "KeyboardInterrupt" not in err, err[-400:]
        assert rc == 130, f"rc={rc}, stderr={err[-200:]}"

    def test_no_stub_outlives_the_interrupt(self):
        """Cancelling a run must not leave a `sleep 300` behind for five minutes.

        `rc == 130` is what makes this test non-vacuous: it says slurmate was
        interrupted *while waiting on a stub*, so there was a stub to leak. Before
        the sweep, that `sleep` was reparented to init and ran out its five
        minutes -- one per run of the suite, sitting in the process table long
        after the run that made it, which is exactly the kind of residue a test
        has no business leaving on a shared login node.
        """
        import os
        import time

        rc, _err, pgid = self._interrupt_during(
            [
                "--print",
                "--job-name",
                "x",
                "--partition",
                "p",
                "--cpus",
                "1",
                "--time",
                "00:05:00",
                "--command",
                "true",
            ]
        )
        assert rc == 130, f"nothing was hanging, so nothing could leak: rc={rc}"
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                os.killpg(pgid, 0)      # signal 0: does the group still exist?
            except ProcessLookupError:
                return
            time.sleep(0.2)
        pytest.fail(f"process group {pgid} outlived the interrupt: a stub is still running")


class TestAFatalSignalPutsTheTerminalBack:
    """Polish pass, 2026-08-28: SIGTERM/SIGHUP left the wizard's terminal raw.

    Measured by driving the wizard in a real pty:

        typed ctrl-C  exit=0        termios restored          alt 1/1
        SIGTERM       killed by 15  termios -echo/-icanon     alt 1/0
        SIGHUP        killed by 1   termios -echo/-icanon     alt 1/0

    A terminal left that way has no echo and no line editing, so it does not crash
    anything — it hands the user a shell that appears dead, with no echo to tell
    them that typing `reset` is working. SIGHUP is the one that matters most on a
    login node: it is what a dropped ssh connection sends.

    `finally` cannot fix this, which is the point: a default-handled SIGTERM ends
    the process without raising, so no `finally` runs. The handler restores and
    then **re-raises with the default disposition**, so the exit status stays
    honest — a tool that swallows SIGTERM is worse than one that leaves a messy
    terminal. Both sibling packages reached the same conclusion.
    """

    #: What "the wizard is up" looks like on the wire: prompt_toolkit has switched
    #: to the alternate screen and composed the first frame. Measured here, that
    #: lands 0.94 s after `execv` -- and the banner alone lands at 0.36 s, which is
    #: why a byte count is not the same question. After the frame the pty goes
    #: quiet until something is typed.
    #:
    #: Signalling on a fixed sleep instead is what made this class fail in company
    #: and pass alone. The 5 s wait it used was a bet on interpreter startup, and
    #: startup is the one thing a loaded node stretches: slurmate imports ~200
    #: modules, `PYTHONDONTWRITEBYTECODE` recompiles them from a shared filesystem
    #: on every drive, CI adds `--cov` tracing on top, and the whole-suite run pays
    #: that while 2,200 other tests contend for the same cores. Both ways of losing
    #: the bet are bad, and one of them is silent:
    #:
    #:   * the signal lands **before the handler is installed** -- the default
    #:     disposition ends the process, and every assertion in the signal test then
    #:     passes for the wrong reason. Measured, signalling at 0.2 s: 0 bytes drawn,
    #:     `alt` (0, 0) -- which satisfies `alt[1] == alt[0]` -- `echo_canon`
    #:     (True, True) read off a master whose termios nobody ever touched, and
    #:     `signalled` True. A green that tested nothing.
    #:   * the signal lands **mid-startup** -- the 25 s exit budget below is spent
    #:     waiting for an interpreter that is still importing, and the failure is
    #:     then reported against the handler: "the wizard never exited after
    #:     sig:HUP".
    #:
    #: So wait for the frame and *then* signal: the budget measures what it claims
    #: to (0.01-0.20 s from signal to exit, measured at load 5-11), and a drive is
    #: faster than the 5 s it replaces. `TestWizardActuallyStarts._render` in
    #: `test_cluster_portability.py` waits for these same landmarks, for the same
    #: reason, and its comment says why a byte count was not enough.
    UP = (b"\x1b[?1049h", b"Steps", b"Job name", b"Step 1")

    @classmethod
    def _drive(cls, action, startup=30.0, argv=()):
        """Run the wizard in a pty, signal or key it, and report the aftermath."""
        import contextlib
        import os
        import pathlib
        import pty
        import select
        import signal as sig
        import sys
        import termios
        import time

        root = pathlib.Path(__file__).resolve().parent.parent
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover - the child execs immediately
            os.environ.update(
                {
                    "PYTHONPATH": str(root / "src"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TERM": "xterm-256color",
                    "COLUMNS": "120",
                    "LINES": "40",
                }
            )
            os.chdir(str(root))
            os.execv(sys.executable, [sys.executable, "-m", "slurmate", *argv])

        out = bytearray()

        def pump(seconds):
            end = time.time() + seconds
            while time.time() < end:
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    return
                if not chunk:
                    return
                out.extend(chunk)

        def lflags():
            try:
                bits = termios.tcgetattr(fd)[3]
            except Exception:
                return None
            return (bool(bits & termios.ECHO), bool(bits & termios.ICANON))

        def is_up():
            return all(mark in out for mark in cls.UP)

        def until_up(limit):
            """Read until the first frame is composed, or the wizard is gone."""
            end = time.time() + limit
            while not is_up() and time.time() < end:
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    return "gone"      # no slave left open: the wizard has exited
                if not chunk:
                    return "gone"
                out.extend(chunk)
            return "up" if is_up() else "timeout"

        try:
            state = until_up(startup)
            if state == "timeout":  # pragma: no cover - only on a stuck startup
                os.kill(pid, sig.SIGKILL)
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(pid, 0)
                pytest.fail(
                    f"the wizard never composed a frame in {startup:.0f}s "
                    f"({len(out)} bytes read) -- nothing was signalled, so this "
                    f"says nothing about the handler"
                )
            if state == "up":
                if action.startswith("sig:"):
                    os.kill(pid, getattr(sig, "SIG" + action.split(":", 1)[1]))
                else:
                    with contextlib.suppress(OSError):
                        os.write(fd, action.split(":", 1)[1].encode())

            signalled = code = None
            deadline = time.time() + 25
            while time.time() < deadline:
                pump(0.5)
                try:
                    done, status = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if done:
                    signalled = os.WIFSIGNALED(status)
                    code = None if signalled else os.waitstatus_to_exitcode(status)
                    break
            else:  # pragma: no cover - only on a hang
                os.kill(pid, sig.SIGKILL)
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(pid, 0)
                pytest.fail(f"the wizard never exited after {action}")

            text = out.decode("utf-8", "replace")
            return {
                "signalled": signalled,
                "code": code,
                "echo_canon": lflags(),
                "alt": (text.count("\x1b[?1049h"), text.count("\x1b[?1049l")),
                "traceback": "Traceback" in text,
            }
        finally:
            # Give the pty back. `os.forkpty()` opens the master without
            # `O_CLOEXEC`, so a leaked one is inherited by every later
            # `pty.fork()` child: measured before this close, the wizard under
            # test ran holding fds 11-13 -> `/dev/ptmx`, the masters of the three
            # drives that came before it, and the suite ended holding three ptys
            # it could never release.
            with contextlib.suppress(OSError):
                os.close(fd)

    @pytest.mark.parametrize("signame", ["TERM", "HUP"])
    def test_the_terminal_is_restored_and_the_signal_is_not_swallowed(self, signame):
        got = self._drive(f"sig:{signame}")
        # The signal has to land on a wizard that is *up*, or the default
        # disposition ends the process before the handler exists and every
        # assertion below is satisfied by a terminal nobody ever touched. See `UP`
        # for the measurement: this is the assertion that fails when the drive
        # goes back to signalling on a fixed sleep.
        assert got["alt"][0] == 1, f"the wizard was not up when {signame} landed: {got}"
        assert got["echo_canon"] == (True, True), f"terminal left raw: {got}"
        assert got["alt"][1] == got["alt"][0], f"alternate screen left open: {got}"
        assert not got["traceback"], got
        # Re-raised with the default disposition, so the status still says
        # "signalled" rather than a tidy exit code.
        assert got["signalled"] is True, got

    def test_a_typed_ctrl_c_is_still_a_clean_zero(self):
        """The control: 0x03 is a KEY in raw mode, not a signal.

        prompt_toolkit turns it into `KeyboardInterrupt` and unwinds properly, and
        that path was already correct — the fix must not turn it into a signalled
        exit.
        """
        got = self._drive("keys:\x03")
        assert got["echo_canon"] == (True, True), got
        assert got["signalled"] is False and got["code"] == 0, got

    def test_the_handlers_are_removed_afterwards(self):
        """The context manager must not leave the process with our handlers.

        Everything after the wizard — the summary, the submit, the action menu —
        should see whatever disposition it had before.
        """
        import signal as sig

        from slurmate.tui import restore_terminal_on_fatal_signal

        before = {s: sig.getsignal(getattr(sig, s)) for s in ("SIGTERM", "SIGHUP")}
        with restore_terminal_on_fatal_signal():
            inside = {s: sig.getsignal(getattr(sig, s)) for s in ("SIGTERM", "SIGHUP")}
        after = {s: sig.getsignal(getattr(sig, s)) for s in ("SIGTERM", "SIGHUP")}
        assert inside != before, "the handlers were never installed"
        assert after == before, "the handlers outlived the wizard"

    def test_a_drive_gives_the_pty_back(self):
        """`_drive` must not leak the master it opened.

        It leaked one per call, and `os.forkpty()` opens the master without
        `O_CLOEXEC`, so the leak was *inherited*: measured mid-suite, the wizard
        under test was running with fds 11, 12 and 13 pointing at `/dev/ptmx` --
        the masters of the three drives before it -- and the pytest process
        finished the suite holding three ptys nothing could release (6 open
        descriptors at the start of the suite, 9 at the end, all three of them
        this class's).

        POSIX hands `open()` the lowest free descriptor, so where a probe lands is
        the detector: it moves up if the drive kept one.
        """
        import os

        probe = os.open(os.devnull, os.O_RDONLY)
        os.close(probe)
        self._drive("sig:TERM")
        again = os.open(os.devnull, os.O_RDONLY)
        os.close(again)
        assert again == probe, (
            f"the drive leaked a descriptor: a probe landed on fd {probe} before "
            f"it and on fd {again} after"
        )

    def test_a_wizard_that_exits_before_it_draws_is_reported_as_exited(self):
        """CONTROL for the frame wait: it must not hold a dead wizard's hand.

        `--version` prints a line and exits without ever drawing, so the frame the
        wait is looking for never arrives. The wait has to notice that and report
        the exit it can see, rather than spend the whole startup budget and then
        blame the handler for a process that was never signalled -- which is the
        shape of the failure this class was reporting.

        Green in both states, which is the point of it: the fixed pump this
        replaced also returned early on EOF, so an honest early exit was never
        the thing that was broken and the frame wait must not become the thing
        that breaks it.
        """
        import time

        started = time.time()
        got = self._drive("sig:TERM", argv=("--version",))
        elapsed = time.time() - started
        assert elapsed < 10, f"the drive sat on a wizard that had already gone: {elapsed:.1f}s"
        assert got["signalled"] is False and got["code"] == 0, got
        assert got["alt"] == (0, 0), got
