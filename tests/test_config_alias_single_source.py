"""One fact, written twice in opposite directions, and the two disagreed.

`system_utils.CONFIG_ALIASES` is "CLI flag spellings that differ from the config
key by more than a dash". `main.CONFIG_ARG_DESTS` is "config keys whose argparse
dest is spelled differently" — the same table inverted. Written out by hand, they
drifted:

    CONFIG_ALIASES   = {"time": "time_limit", "array": "array_spec"}
    CONFIG_ARG_DESTS = {"time_limit": "time", "array_spec": "array",
                        "env_name": "env"}

So `env = "myenv"` in a config file was reported as an unknown key and its value
DROPPED. That is exactly the harm `_normalize_config_keys` documents for `time`
("was dropped and the user got the 2-hour default for a 36-hour run"), and here it
is worse than a wrong default: the script then activates no environment at all, so
the job fails on its first import. Measured before the fix::

    config={'env': 'myenv'}  ->  activation=NONE

`--env` was the ONLY flag in that state — checked against the real parser, below.
`main` now derives its table by inverting `CONFIG_ALIASES`, so a fourth pair cannot
be added to one and forgotten in the other.
"""

import contextlib
import io

import pytest

from slurmate import system_utils as su
from slurmate.builder import build_from_answers
from slurmate.main import CONFIG_ARG_DESTS
from slurmate.system_utils import CONFIG_ALIASES, CONFIG_KEYS

#: Flags that are run-mode switches rather than job fields, so they have no
#: config spelling by design. Named rather than inferred: a rule that skips
#: "whatever is not in CONFIG_KEYS" would skip the defect this file is about.
#:
#: `--error` is `argparse.SUPPRESS` and documented as derived from `--output-file`
#: ("error derives .err"); `--gres`, `--gpus-per-node` and `--gpus-per-task` are
#: Slurm-syntax shorthands for a `gpu_format` value, which IS a config key.
#: `version` is absent on purpose: `--version` uses `action="version"` and never
#: becomes a dest, so listing it would break the "every excluded name is a real
#: dest" guard below.
_NOT_JOB_FIELDS = frozenset({
    "ascii", "demo", "dry_run", "force", "no_save_script", "print", "yes",
    "error", "gres", "gpus_per_node", "gpus_per_task",
})

BASE = {
    "job_name": "j",
    "partition": "p",
    "cpus": 4,
    "memory": "8G",
    "time_limit": "01:00:00",
    "command": "python train.py",
    "nodes": 1,
    "output_dir": "logs",
    "env_type": "conda",
}


def _normalised(config):
    su._reset_config_notices()
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        out = su._normalize_config_keys(dict(config), "/tmp/c.toml")
    return out, " ".join(buf.getvalue().split())


class TestTheEnvKeyReachesTheScript:
    def test_the_cli_spelling_is_accepted_in_a_config_file(self):
        kept, note = _normalised({"env": "myenv"})
        assert kept == {"env_name": "myenv"}, kept
        assert "unknown key" not in note, note

    def test_and_the_value_actually_activates_the_environment(self):
        """End to end, because a kept key that never reaches the script is the
        same defect one step later."""
        kept, _ = _normalised({"env": "myenv"})
        script = build_from_answers({**BASE, **kept})
        assert "conda activate myenv" in script, script

    def test_control_no_env_at_all_still_activates_nothing(self):
        """CONTROL. The pre-fix output for `env = "myenv"` was this — an absent
        activation — so it has to stay reachable for the case where it is right,
        or the test above would pass off a script that always activates."""
        script = build_from_answers(dict(BASE))
        assert "conda activate" not in script, script


class TestTheTwoTablesCannotDisagreeAgain:
    def test_one_is_the_inverse_of_the_other(self):
        assert {v: k for k, v in CONFIG_ALIASES.items()} == CONFIG_ARG_DESTS

    def test_the_inversion_is_lossless(self):
        """Guard on the derivation itself: two CLI spellings mapping to one config
        key would silently lose an entry when inverted, and the result would still
        look like a valid table."""
        assert len(set(CONFIG_ALIASES.values())) == len(CONFIG_ALIASES)
        assert len(CONFIG_ARG_DESTS) == len(CONFIG_ALIASES)

    def test_neither_table_is_written_out_by_hand_twice(self):
        """The pair that drifted, pinned as source. `main` must not respell it."""
        from pathlib import Path

        import slurmate.main as m

        source = Path(m.__file__).read_text()
        for pair in ('"time_limit": "time"', '"array_spec": "array"',
                     '"env_name": "env"'):
            assert pair not in source, f"main respells the alias: {pair}"
        assert "CONFIG_ALIASES.items()" in source

    def test_the_readme_names_every_alias(self):
        """The THIRD copy of this fact, and it had drifted the same way.

        `README.md` promises "a key copied from `--help` does what it looks like
        it does" and then listed only two pairs -- the promise `--env` broke. A
        prose list is a copy like any other, so it is checked rather than trusted:
        every alias the table accepts has to appear beside the key it lands on.
        """
        from pathlib import Path

        readme = Path(__file__).resolve().parent.parent / "README.md"
        text = readme.read_text()
        section = text.split("**CLI spellings work too.**", 1)
        assert len(section) == 2, "the README no longer has the alias paragraph"
        paragraph = section[1].split("\n\n", 1)[0]
        for cli_dest, config_key in CONFIG_ALIASES.items():
            assert f"`{cli_dest}`" in paragraph, (cli_dest, paragraph)
            assert f"`{config_key}`" in paragraph, (config_key, paragraph)

    def test_every_config_key_the_table_names_is_real(self):
        """An alias pointing at a key `CONFIG_KEYS` does not hold would be dropped
        by the very check the alias exists to get past."""
        for cli_dest, config_key in CONFIG_ALIASES.items():
            assert config_key in CONFIG_KEYS, (cli_dest, config_key)


class TestTheMembershipRuleHoldsAgainstTheRealParser:
    """The rule the comment states: "every CLI flag whose dest differs from its
    config key by more than a dash". Enforced against the parser rather than a
    list, so a new job-field flag with a differing dest fails here.
    """

    @staticmethod
    def _job_field_flags():
        """Every argparse DEST the CLI defines.

        Read off `parse_args([])` rather than by scanning the source or reaching
        into `_actions`: the dest is exactly the name `CONFIG_ARG_DESTS` has to
        match, argparse has already applied the dash-to-underscore rule that the
        alias table's comment says makes most entries unnecessary, and a Namespace
        cannot drift from the parser the way a hand-copied list can.
        """
        from slurmate.main import parse_args

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            namespace = parse_args([])
        assert buf.getvalue() == "", buf.getvalue()
        return dict(vars(namespace))

    def test_the_parser_is_readable_and_not_empty(self):
        """Vacuity guard: an empty flag set would make the sweep below pass."""
        flags = self._job_field_flags()
        assert len(flags) > 20, sorted(flags)
        assert "env" in flags and "time" in flags

    def test_no_job_field_flag_is_missing_a_config_spelling(self):
        missing = sorted(
            name
            for name in self._job_field_flags()
            if name not in CONFIG_KEYS
            and name not in CONFIG_ALIASES
            and name not in _NOT_JOB_FIELDS
        )
        assert missing == [], missing

    def test_the_excluded_set_is_not_hiding_a_real_field(self):
        """The exclusion list must name only flags that really are switches or
        documented shorthands -- otherwise the sweep above could be silenced by
        adding a name to it. Every excluded flag must NOT be a config key."""
        for name in _NOT_JOB_FIELDS:
            assert name not in CONFIG_KEYS, name
        assert set(self._job_field_flags()) >= _NOT_JOB_FIELDS, sorted(
            _NOT_JOB_FIELDS - set(self._job_field_flags())
        )


class TestControls:
    @pytest.mark.parametrize(
        ("written", "kept"),
        [("time", "time_limit"), ("array", "array_spec"),
         ("time_limit", "time_limit"), ("array_spec", "array_spec"),
         ("env_name", "env_name"), ("job-name", "job_name")],
    )
    def test_control_every_spelling_that_already_worked_still_lands(self, written, kept):
        """The `env` case is deliberately NOT here: it fails with the fix removed,
        so it is a finding case, not a control. Keeping it in this parametrisation
        made a test named `test_control_*` redden under the neuter -- which is how
        a control stops meaning anything."""
        got, note = _normalised({written: "X"})
        assert got == {kept: "X"}, (got, note)

    def test_control_a_genuine_typo_is_still_reported_and_dropped(self):
        """The other half of `_normalize_config_keys`: an outright typo must still
        warn rather than be accepted. Holds in both states."""
        kept, note = _normalised({"envv": "X"})
        assert kept == {}
        assert "unknown key 'envv'" in note, note

    def test_control_the_disclosure_still_maps_config_keys_to_dests(self):
        """`_config_keys_in_effect` reads `CONFIG_ARG_DESTS`, so the derivation has
        to answer for the two pairs that already worked."""
        import argparse

        from slurmate.main import _config_keys_in_effect

        args = argparse.Namespace(time=None, array=None, env=None, partition="p")
        in_effect = _config_keys_in_effect(
            args, {"time_limit": "1:00", "array_spec": "1-2", "env_name": "e",
                   "partition": "p"}
        )
        assert in_effect == ["time_limit", "array_spec", "env_name"]
