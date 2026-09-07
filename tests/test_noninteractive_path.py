"""The non-interactive surface: the flags, and the config file behind them.

Every previous round worked on the wizard. ``main.py``'s flag path is the other
way in -- the one a Makefile, a CI job or a wrapper script uses -- and a defect
there is silent by construction, because nobody is watching the terminal.

Two findings from driving it are pinned here.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import pytest
from rich.console import Console

from slurmate.main import run_batch

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


# ── A config file is a way of SUPPLYING --custom-sbatch ──────────────────────


class TestConfigCustomSbatchReachesTheSameRefusals:
    """``custom_sbatch`` from a file must be judged like ``custom_sbatch`` typed.

    Measured on midway3, identical value, two surfaces:

        slurmate --print --custom-sbatch='--comment=my run' ...
            -> rc 1, "which cannot be an sbatch option", both readings named

        custom_sbatch = "--comment=my run"   # in ./.slurmate.toml
            -> rc 0, `#SBATCH --comment="my run"`, zero bytes on stderr

    ``_run_batch`` read ``args.custom_sbatch`` for both refusals, so a
    config-supplied value reached ``unquoted_custom_values(None)`` and
    ``managed_custom_flags(None)`` -- and the ambiguity is undetectable
    afterwards, because ``_parse_custom_flags`` has already picked a reading by
    the time the parsed list exists. (That is why ``site_check_issues``, which
    sees only the list, cannot catch this one either: the existing
    ``test_the_wizard_half_is_wired_as_an_error`` says so in as many words.)

    Two surfaces disagreeing about one fact is the failure this package works
    hardest to avoid, and the silent surface was the non-interactive one.
    """

    def _ns(self, **over: object) -> argparse.Namespace:
        base: dict[str, object] = {
            "job_name": "p", "account": None, "partition": "cpu-shared", "qos": None,
            "cpus": 1, "memory": "1G", "time": "00:05:00", "nodes": 1,
            "ntasks_per_node": None, "gpus": None, "gpu_type": None,
            "gpu_format": None, "constraint": None, "array": None, "modules": None,
            "env": None, "env_type": None, "output_dir": None, "output_file": None,
            "command": "echo hi", "custom_sbatch": None, "yes": False, "force": False,
        }
        base.update(over)
        return argparse.Namespace(**base)

    def _run(self, args: argparse.Namespace, config: dict[str, object]) -> int | None:
        try:
            run_batch(args, Console(), config)
        except SystemExit as exc:
            return int(exc.code or 0)
        return None

    # ── the defect ──────────────────────────────────────────────────────────

    def test_an_ambiguous_config_value_is_refused_like_the_flag(self, capsys):
        code = self._run(self._ns(), {"custom_sbatch": "--comment=my run"})
        err = " ".join(capsys.readouterr().err.split())
        assert code == 1, "a config file smuggled the value past the refusal"
        assert "cannot be an sbatch option" in err
        # Both readings, so the user picks rather than the tool -- the same
        # sentence the flag path prints for the same value.
        assert '--comment="my run"' in err
        assert "--comment=my --run" in err

    def test_a_managed_config_value_is_refused_like_the_flag(self, capsys):
        # This one WAS reported downstream by site_check_issues, but only in the
        # modes that consult it: `--dry-run` printed the error and still exited 0,
        # while the identical value as a flag exited 1 before a script existed.
        code = self._run(self._ns(), {"custom_sbatch": "--partition=other"})
        err = " ".join(capsys.readouterr().err.split())
        assert code == 1
        assert "--partition, which slurmate manages" in err

    # ── controls: these must hold with and without the fix ──────────────────

    def test_a_toml_list_has_only_one_reading_and_is_accepted(self, capsys):
        """CONTROL. ``custom_sbatch = ["--comment=my run"]`` is not ambiguous.

        A list element is a flag the *user* delimited, so there is no second
        reading to refuse -- and refusing it would break the documented way to
        write a value containing a space in a config file. The raw config value
        is therefore passed through as-is (str stays str, list stays list)
        rather than being pre-parsed.
        """
        code = self._run(self._ns(), {"custom_sbatch": ["--comment=my run"]})
        err = " ".join(capsys.readouterr().err.split())
        assert code is None, err
        assert "cannot be an sbatch option" not in err

    @pytest.mark.parametrize(
        "value",
        ['--comment="my run"', "--comment='my run'", "--exclusive"],
    )
    def test_the_forms_help_invites_are_untouched(self, capsys, value):
        """CONTROL. The refusal must not fire on a correctly quoted value."""
        code = self._run(self._ns(), {"custom_sbatch": value})
        err = " ".join(capsys.readouterr().err.split())
        assert code is None, err
        assert "cannot be an sbatch option" not in err

    def test_an_explicit_flag_still_beats_the_file(self, capsys):
        """CONTROL. Precedence is unchanged: the flag wins, so the *flag* is
        what gets judged. A config value the flag overrode must not be able to
        fail a run that supplied a perfectly good ``--custom-sbatch``."""
        code = self._run(
            self._ns(custom_sbatch="--exclusive"),
            {"custom_sbatch": "--comment=my run"},
        )
        err = " ".join(capsys.readouterr().err.split())
        assert code is None, err
        assert "cannot be an sbatch option" not in err

    def test_the_flag_path_itself_is_unchanged(self, capsys):
        """CONTROL. The typed form still refuses -- the fix widened the input,
        it did not move the check."""
        code = self._run(self._ns(custom_sbatch="--comment=my run"), {})
        assert code == 1
        assert "cannot be an sbatch option" in " ".join(capsys.readouterr().err.split())

    def test_no_custom_flags_at_all_is_quiet(self, capsys):
        """CONTROL. The commonest invocation must gain no output."""
        code = self._run(self._ns(), {})
        assert code is None
        assert "cannot be an sbatch option" not in capsys.readouterr().err


# ── The config search order, as documented vs. as measured ──────────────────


def _config_section() -> str:
    """The README's "Configuration file" section."""
    text = README.read_text()
    start = text.index("## ⚙️ Configuration file")
    end = text.index("\n## ", start + 1)
    return text[start:end]


class TestTheReadmeDescribesTheRealConfigSearchOrder:
    """The README said "first match wins"; the loader merges per key.

    Measured by setting the same key in three places and seeing which reached
    the script: **CLI flag > ./.slurmate.toml > ~/.config/slurmate/config.toml >
    built-in default**, and a key set in only the global file survives a project
    file that never mentions it.

    "First match wins" documents the behaviour ``load_config`` was changed *away*
    from -- and its own docstring says why: a one-line project file naming this
    cluster's partition used to discard the global account, memory, time limit
    and module list, "silently, and each with its own failure". A user who reads
    the README still expects that, so they duplicate the global file into every
    project to be safe. The same section already said "merged in that order"
    twenty lines further down, so the page contradicted itself.
    """

    def test_the_merge_is_what_actually_happens(self, tmp_path, monkeypatch):
        """The claim, checked against the loader rather than trusted."""
        from slurmate import system_utils as su

        home = tmp_path / "home"
        proj = tmp_path / "proj"
        (home / ".config" / "slurmate").mkdir(parents=True)
        proj.mkdir()
        (home / ".config" / "slurmate" / "config.toml").write_text(
            'account = "rcc-staff"\ncpus = 7\n'
        )
        (proj / ".slurmate.toml").write_text('cpus = 9\n')
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        monkeypatch.chdir(proj)
        su._reset_config_notices()
        try:
            cfg = su.load_config()
        finally:
            su._reset_config_notices()
        assert cfg["cpus"] == 9, "the more specific file must win the key"
        assert cfg["account"] == "rcc-staff", "and must not destroy the rest"

    def test_the_readme_does_not_claim_first_match_wins(self):
        section = _config_section()
        assert "first match wins" not in section.lower(), (
            "the README documents the first-file-wins behaviour load_config was "
            "changed away from; both files are read and merged per key"
        )

    def test_the_readme_says_the_files_are_merged(self):
        assert re.search(r"merged\s+per\s+key", _config_section(), re.I), (
            "the search order needs to say it is a merge, not a first match"
        )

    def test_the_two_paths_are_still_listed_most_specific_first(self):
        """CONTROL. The priority order itself was right and must not move."""
        section = _config_section()
        assert section.index(".slurmate.toml") < section.index(
            "~/.config/slurmate/config.toml"
        )

    def test_flags_still_documented_as_beating_the_file(self):
        """CONTROL. The top of the precedence chain is unchanged."""
        assert "flags always win" in _config_section()
