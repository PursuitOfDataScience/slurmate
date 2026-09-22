"""The machine-readable surface is a contract, so it is pinned like one.

`slurmate brief --json` and `slurmate check --json` exist to be parsed by a
program that is not in this repository and cannot be updated when a key is
renamed. Every human surface in this package is free to move between releases
(the README says so, and pre-1.0 means it), which is exactly why these two
cannot be: an agent that reads ``partitions[].nodes_up`` has no way to notice
that it became ``partitions[].up`` except by silently getting ``None`` and
filtering every partition away.

So the key set is written out here rather than derived from the code. A test
that walks the dict it is testing proves only that the dict is self-consistent.
Removing or renaming a field has to be a deliberate edit to this file, and the
diff is the place the decision gets made.

The sibling rule is `SCHEMA_VERSION`: additive keys do not bump it, so this
file asserts that the documented keys are *present*, never that nothing else
is. A consumer must ignore what it does not recognise.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from slurmate import agent

#: Every top-level key `cluster_brief` promises.
BRIEF_KEYS = {
    "schema_version", "generated_at", "mock", "cluster", "tools", "you",
    "partitions", "partition_filter", "partitions_elided", "node_features",
    "gpu_request_formats", "config_defaults", "errors",
}

#: Every key on a partition row. `fetch_partitions` produces all of these, and
#: `_partition_row` is the projection that keeps its private ones out.
PARTITION_KEYS = {
    "name", "is_default", "state", "nodes", "nodes_up", "cpus_per_node",
    "mem_per_node_mb", "heterogeneous", "timelimit", "has_gpu", "gpu_types",
    "gpus_per_node", "queue",
}

CLUSTER_KEYS = {"name", "slurm_version", "select_type", "max_array_size",
                "hostname", "qos"}

CHECK_KEYS = {"schema_version", "mock", "ok", "checked", "findings",
              "scheduler", "checked_script"}

#: The values `partition_filter` may take. Each means something different
#: about what an *absent* partition implies, so a consumer switches on it.
FILTERS = {"associations", "accounts", "public", "named", "none"}

ESC = "\x1b"

#: A partition `MOCK_PARTITIONS` actually carries. Spelled once: the suite runs
#: under SLURMATE_MOCK, so a real cluster's name here would make every check
#: below assert against "no such partition" instead of the thing it is testing.
MOCK_GPU_PARTITION = "gpu-shared"


def _cli(*args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the real console entry point in a child process.

    A child, not an in-process call: the thing under test is what reaches
    stdout, and that includes anything a module-level import decides to print.
    """
    import os

    env = dict(os.environ, SLURMATE_MOCK="1")
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "slurmate", *args],
        capture_output=True, text=True, check=False, env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
    )


class TestTheBriefSchema:
    def test_every_documented_key_is_present(self):
        brief = agent.cluster_brief()
        assert set(brief) >= BRIEF_KEYS, BRIEF_KEYS - set(brief)

    def test_the_cluster_block_names_the_site(self):
        assert set(agent.cluster_brief()["cluster"]) >= CLUSTER_KEYS

    def test_every_partition_row_carries_every_field(self):
        for row in agent.cluster_brief(all_partitions=True)["partitions"]:
            assert set(row) >= PARTITION_KEYS, PARTITION_KEYS - set(row)

    def test_no_private_key_escapes_into_the_document(self):
        """`fetch_partitions` and `_get_partition` both carry ``_``-prefixed
        bookkeeping. A key that leaks into a published schema becomes one
        somebody depends on, and then it is not private any more."""
        for row in agent.cluster_brief(all_partitions=True)["partitions"]:
            leaked = [k for k in row if k.startswith("_")]
            assert not leaked, leaked

    def test_the_filter_rule_is_named_and_known(self):
        assert agent.cluster_brief()["partition_filter"] in FILTERS

    def test_mock_mode_says_so_at_the_top_level(self):
        """The one field whose absence would make every other one a lie."""
        assert agent.cluster_brief()["mock"] is True

    def test_the_schema_version_is_reported_and_is_an_integer(self):
        assert agent.cluster_brief()["schema_version"] == agent.SCHEMA_VERSION
        assert isinstance(agent.SCHEMA_VERSION, int)

    def test_an_unscoped_association_list_reads_as_unknown_not_as_none_allowed(self):
        """``you.partitions`` is None, never [].

        `fetch_user_partitions` returns None for "no filtering is justified",
        which is a different claim from "you may use no partition". Collapsing
        the two would make a site that scopes access by account look like one
        that has locked the caller out of everything.
        """
        you = agent.cluster_brief()["you"]
        assert you["partitions"] is None or isinstance(you["partitions"], list)
        assert you["partitions"] != [] or agent.cluster_brief()["partitions"] == []

    def test_naming_a_partition_returns_only_that_one(self):
        every = agent.cluster_brief(all_partitions=True)["partitions"]
        wanted = every[0]["name"]
        brief = agent.cluster_brief(partition=wanted)
        assert [p["name"] for p in brief["partitions"]] == [wanted]
        assert brief["partition_filter"] == "named"
        assert brief["partitions_elided"] == len(every) - 1

    def test_elided_accounts_for_every_partition_left_out(self):
        """An omission nobody can count is an omission nobody notices."""
        brief = agent.cluster_brief()
        every = agent.cluster_brief(all_partitions=True)["partitions"]
        assert len(brief["partitions"]) + brief["partitions_elided"] == len(every)


class TestTheCheckSchema:
    def test_every_documented_key_is_present(self):
        result = agent.check_request({"partition": MOCK_GPU_PARTITION, "command": "true"})
        assert set(result) >= CHECK_KEYS, CHECK_KEYS - set(result)

    def test_a_finding_is_a_level_and_a_message(self):
        result = agent.check_request({"partition": "nope-not-here", "command": "true"})
        for finding in result["findings"]:
            assert set(finding) == {"level", "message"}
            assert finding["level"] in ("error", "warning")

    def test_ok_is_false_when_any_finding_is_an_error(self):
        result = agent.check_request({"partition": "nope-not-here", "command": "true"})
        assert any(f["level"] == "error" for f in result["findings"])
        assert result["ok"] is False

    def test_the_scheduler_verdict_is_one_of_three_words(self):
        result = agent.check_request({"partition": MOCK_GPU_PARTITION, "command": "true"})
        assert result["scheduler"]["verdict"] in ("accepted", "refused", "unavailable")

    def test_the_flag_form_is_judged_against_a_script_it_shows_you(self):
        """A refusal has to be readable against the directives that earned it."""
        result = agent.check_request({"partition": MOCK_GPU_PARTITION, "cpus": 4, "command": "true"})
        assert f"#SBATCH --partition={MOCK_GPU_PARTITION}" in result["checked_script"]


class TestSilenceIsNotAPass:
    """Found with `sbatch` off PATH entirely, and it is the worst answer this
    module can give.

    `check_script_with_scheduler` returns "" for both "the controller accepted
    it" and "there was no controller to ask". That is correct for its own
    caller, which hunts for refusals and whose docstring says outright that
    "could not ask" must never render as "cannot run"; reading the same ""
    as *acceptance* is the inverse of that mistake, and it produced
    `ok: true`, `verdict: accepted`, zero findings on a machine that had not
    read one byte of any cluster. An agent reads `ok` and hands the script on.
    """

    def test_no_scheduler_is_its_own_verdict(self, mocker):
        # _force_mock off as well as sbatch: the case under test is a REAL
        # machine without a controller, which is the one that produced the
        # false pass. Demo mode fabricates a verdict on purpose.
        mocker.patch("slurmate.system_utils._force_mock", return_value=False)
        mocker.patch("slurmate.system_utils.is_tool_available", return_value=False)
        from slurmate.system_utils import scheduler_verdict

        verdict, reason = scheduler_verdict("#!/bin/bash\ntrue\n")
        assert verdict == "unavailable" and "sbatch" in reason

    def test_demo_mode_fabricates_the_controller_like_everything_else(self):
        """`mock` is reported at the top of every document, so a simulated
        verdict is marked; a demo in which the scheduler alone refused to
        answer would be unusable."""
        from slurmate.system_utils import scheduler_verdict

        assert scheduler_verdict("#!/bin/bash\ntrue\n")[0] == "accepted"

    def test_an_empty_script_is_not_an_accepted_one(self):
        from slurmate.system_utils import scheduler_verdict

        assert scheduler_verdict("")[0] == "unavailable"

    def test_ok_means_checked_and_clean_not_merely_quiet(self, mocker):
        mocker.patch("slurmate.agent.scheduler_verdict",
                     return_value=("unavailable", "sbatch is not on PATH"))
        result = agent.check_request({"partition": MOCK_GPU_PARTITION, "command": "true"})
        assert result["ok"] is False
        assert result["checked"] is False
        assert any("NOT asked" in f["message"] for f in result["findings"])

    def test_a_clean_verified_job_is_still_ok(self, mocker):
        mocker.patch("slurmate.agent.scheduler_verdict", return_value=("accepted", ""))
        result = agent.check_request({"partition": MOCK_GPU_PARTITION, "command": "true"})
        assert result["ok"] is True and result["checked"] is True

    def test_a_brief_that_read_nothing_says_so(self, mocker):
        """Zero partitions, zero errors and every cluster field null reads as
        a real but empty cluster. `last_cluster_error` is silent here because
        `is_tool_available` short-circuits before any command runs."""
        mocker.patch("slurmate.agent.is_mock", return_value=False)
        mocker.patch("slurmate.agent.fetch_partitions", return_value=[])
        mocker.patch("slurmate.agent.is_tool_available", return_value=False)
        brief = agent.cluster_brief()
        assert brief["errors"], "an unreadable cluster reported no error"
        assert "not a Slurm login node" in brief["errors"][0]


class TestParsingAScriptBack:
    def test_the_common_directives_land_in_the_fields_the_validators_read(self):
        answers, findings = agent.parse_sbatch_script(
            "#!/bin/bash\n"
            "#SBATCH --job-name=probe\n"
            "#SBATCH --partition=gpu\n"
            "#SBATCH -A my_lab\n"
            "#SBATCH -c 8\n"
            "#SBATCH --mem=32G\n"
            "#SBATCH -t 04:00:00\n"
            "#SBATCH --nodes 2\n"
            "python train.py\n"
        )
        assert answers["job_name"] == "probe"
        assert answers["partition"] == "gpu"
        assert answers["account"] == "my_lab"
        assert answers["cpus"] == 8
        assert answers["memory"] == "32G"
        assert answers["time_limit"] == "04:00:00"
        assert answers["nodes"] == 2
        assert findings == []

    @pytest.mark.parametrize("flag,fmt", [
        ("--gres=gpu:a100:2", "gres_type"),
        ("--gpus=a100:2", "gpus"),
        ("--gpus-per-node=a100:2", "gpus_per_node"),
        ("--gpus-per-task=a100:2", "gpus_per_task"),
    ])
    def test_all_four_gpu_spellings_set_the_format_they_imply(self, flag, fmt):
        """Which spelling was used is itself a finding: `unsupported_gpu_format`
        can only object to a format it was told about, and on a cons_res site
        two of these four do not parse at all."""
        answers, _ = agent.parse_sbatch_script(f"#!/bin/bash\n#SBATCH {flag}\n")
        assert answers["gpus"] == 2
        assert answers["gpu_type"] == "a100"
        assert answers["gpu_format"] == fmt

    def test_a_directive_given_twice_is_an_error(self):
        """Slurm accepts it and silently honours the last one, so the script
        reads one way and runs another. Once the answers dict is built the
        earlier value is gone, which is why the parser has to catch it."""
        _, findings = agent.parse_sbatch_script(
            "#!/bin/bash\n#SBATCH -J first\n#SBATCH -J second\n"
        )
        assert any(f[0] == "error" and "more than once" in f[1] for f in findings)

    def test_two_spellings_of_one_field_are_also_a_duplicate(self):
        _, findings = agent.parse_sbatch_script(
            "#!/bin/bash\n#SBATCH -J first\n#SBATCH --job-name=second\n"
        )
        assert any("set twice" in f[1] for f in findings)

    def test_directives_after_the_first_command_are_ignored(self):
        """Slurm stops reading there too, so validating them would report on a
        directive the scheduler never sees."""
        answers, _ = agent.parse_sbatch_script(
            "#!/bin/bash\n#SBATCH -p gpu\necho hi\n#SBATCH -p bigmem\n"
        )
        assert answers["partition"] == "gpu"

    def test_an_unmodelled_directive_is_kept_rather_than_dropped(self):
        """Dropping it would make the script slurmate validated differ from the
        one the user holds."""
        answers, _ = agent.parse_sbatch_script(
            "#!/bin/bash\n#SBATCH --exclusive\n#SBATCH --dependency=afterok:42\n"
        )
        assert "--exclusive" in answers["custom_sbatch"]
        assert "--dependency afterok:42" in answers["custom_sbatch"]

    def test_a_non_numeric_count_is_reported_not_crashed_on(self):
        answers, findings = agent.parse_sbatch_script("#!/bin/bash\n#SBATCH -c eight\n")
        assert "cpus" not in answers
        assert any("not a whole number" in f[1] for f in findings)


class TestTheScriptBodyIsRead:
    """The half of a script that is not `#SBATCH`.

    `check_modules` and `check_conda_env` existed all along and catch the
    failure that costs the most time: the job queues, is scheduled, starts,
    and dies on `module load` an hour later. Nothing ever handed them a
    script, because they read ``answers["modules"]`` and only the wizard and
    the CLI flags filled that in. So `check --script` returned a clean bill on
    a script whose first command was `module load does/not/exist`, while
    SKILL.md told an agent that exact case was covered.
    """

    @pytest.mark.parametrize("line,expected", [
        ("module load cuda/12.1", ["cuda/12.1"]),
        ("module add gcc/9.3.0", ["gcc/9.3.0"]),
        ("ml python/3.11 cuda/12.1", ["python/3.11", "cuda/12.1"]),
        ("  module load  a  b  ", ["a", "b"]),
    ])
    def test_every_module_spelling_is_read(self, line, expected):
        assert agent.parse_script_body(f"#!/bin/bash\n{line}\n")["modules"] == expected

    @pytest.mark.parametrize("line", ["module purge", "module unload cuda/12.1"])
    def test_unloading_is_not_a_request(self, line):
        """Those name a module the script is getting rid of. Reporting one as
        missing is backwards."""
        assert "modules" not in agent.parse_script_body(f"#!/bin/bash\n{line}\n")

    def test_a_flag_or_a_shell_expansion_is_not_a_module_name(self):
        body = agent.parse_script_body("#!/bin/bash\nmodule load -f $TOOLCHAIN cuda/12.1\n")
        assert body["modules"] == ["cuda/12.1"]

    @pytest.mark.parametrize("line,etype,ename", [
        ("conda activate myenv", "conda", "myenv"),
        ("mamba activate myenv", "mamba", "myenv"),
        ("source activate myenv", "conda", "myenv"),
        ("source /opt/venvs/proj/bin/activate", "venv", "/opt/venvs/proj"),
        (". /opt/venvs/proj/bin/activate", "venv", "/opt/venvs/proj"),
    ])
    def test_every_activation_spelling_is_read(self, line, etype, ename):
        body = agent.parse_script_body(f"#!/bin/bash\n{line}\n")
        assert (body["env_type"], body["env_name"]) == (etype, ename)

    def test_a_commented_out_line_is_not_an_activation(self):
        assert agent.parse_script_body("#!/bin/bash\n# conda activate old\n") == {}

    def test_the_body_never_overwrites_a_directive(self):
        """A `#SBATCH` line is what the caller declared; the two disagree only
        in a script that is already confusing."""
        answers, _ = agent.parse_sbatch_script(
            "#!/bin/bash\n#SBATCH -p gpu-shared\nmodule load a\n"
        )
        assert answers["partition"] == "gpu-shared"
        assert answers["modules"] == ["a"]

    def test_a_bad_module_in_the_body_reaches_the_findings(self, mocker):
        """The wiring, end to end: a module named only in the commands is
        checked, which is the claim SKILL.md makes.

        `check_modules` is patched because it cannot answer under
        SLURMATE_MOCK: `_module_command()` returns None with no module system,
        and staying silent there is correct. What is under test is the path
        from a command line in the body to that function's argument, which is
        the link that did not exist. Verified live on midway3, where the real
        function answers: `module load cuda/99.9-does-not-exist` comes back
        naming the eleven `cuda` versions that do.
        """
        seen = mocker.patch("slurmate.main.check_modules",
                            return_value=[("error", "module 'no/such/module' not found")])
        script = "#!/bin/bash\n#SBATCH -p gpu-shared\nmodule load no/such/module\ntrue\n"
        answers, extra = agent.parse_sbatch_script(script)
        result = agent.check_request(answers, script=script, extra=extra)
        assert seen.call_args.args[0] == ["no/such/module"]
        assert any("no/such/module" in f["message"] for f in result["findings"])


class TestAModuleCheckThatCannotRunSaysSo:
    """Found on Booth's Mercury, and it was a false *pass*, not a gap.

    Over a non-login ssh there ``MODULEPATH`` is unset, but `shutil.which`
    still finds ``/usr/bin/modulecmd``, which then answers every query with
    ``ERROR: No module path defined`` and the shell fragment ``test 0 = 1;``
    on the same stream as its listing, exiting 0. `fetch_module_matches` read
    those two lines as two module names, so every name looked present and
    `check` approved a script naming a module that does not exist.
    """

    def test_no_module_path_means_no_module_system(self, mocker):
        from slurmate import system_utils as su

        mocker.patch.dict(su.os.environ, {"MODULEPATH": ""}, clear=False)
        assert su._module_command() is None

    def test_an_error_line_is_not_a_match(self, mocker):
        """None (cannot ask), never [] (asked, nothing matched): [] makes the
        caller say the module is missing, and a list holding the error text
        makes it say the module is there."""
        from slurmate import system_utils as su

        mocker.patch.dict(su.os.environ, {"MODULEPATH": "/opt/modules"}, clear=False)
        mocker.patch.object(su, "_module_command", return_value=["modulecmd", "bash"])
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(su, "_run_command",
                            return_value=("test 0 = 1;", "ERROR: No module path defined", 0))
        assert su.fetch_module_matches("anything") is None
        assert su.check_modules(["anything"]) == []

    def test_shell_fragments_never_read_as_module_names(self, mocker):
        from slurmate import system_utils as su

        mocker.patch.dict(su.os.environ, {"MODULEPATH": "/opt/modules"}, clear=False)
        mocker.patch.object(su, "_module_command", return_value=["modulecmd", "bash"])
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(su, "_run_command",
                            return_value=("test 0 = 1;", "cuda/12.1\ncuda/11.8\n", 0))
        assert su.fetch_module_matches("cuda") == ["cuda/12.1", "cuda/11.8"]

    def test_check_says_it_could_not_look_instead_of_passing(self, mocker):
        """"I found no problem" and "I could not look" reach a caller as the
        same clean bill, and an agent acts on a clean bill."""
        mocker.patch("slurmate.agent._module_command", return_value=None)
        result = agent.check_request(
            {"partition": MOCK_GPU_PARTITION, "modules": ["cuda/12.1"], "command": "true"}
        )
        said = [f for f in result["findings"] if "NOT checked" in f["message"]]
        assert said and said[0]["level"] == "warning"
        assert "cuda/12.1" in said[0]["message"]

    def test_it_stays_quiet_when_there_is_nothing_to_check(self, mocker):
        mocker.patch("slurmate.agent._module_command", return_value=None)
        result = agent.check_request({"partition": MOCK_GPU_PARTITION, "command": "true"})
        assert not any("NOT checked" in f["message"] for f in result["findings"])


class TestThePartitionFilterDoesNotHideWhatYouOwn:
    """`public` on its own was wrong, and it shipped for one revision.

    "Open to all accounts" and "open to me" are different sets. Measured on
    midway3: 6 public, 21 the caller's accounts reach, and ``beagle3`` (which
    they hold ``beagle3-users`` for) in the gap. With rule 3 of SKILL.md
    telling an agent never to name a partition the brief did not list, the
    weaker filter talks it out of a partition the user pays for.
    """

    def test_account_reachability_outranks_merely_public(self, mocker):
        mocker.patch("slurmate.agent.fetch_user_partitions", return_value=None)
        mocker.patch("slurmate.agent.fetch_reachable_partitions",
                     return_value={"gpu-shared", "cpu-shared"})
        public = mocker.patch("slurmate.agent.fetch_public_partitions")
        brief = agent.cluster_brief()
        assert brief["partition_filter"] == "accounts"
        assert {"gpu-shared", "cpu-shared"} <= {p["name"] for p in brief["partitions"]}
        public.assert_not_called()

    def test_public_is_only_reached_when_no_account_answers(self, mocker):
        mocker.patch("slurmate.agent.fetch_user_partitions", return_value=None)
        mocker.patch("slurmate.agent.fetch_reachable_partitions", return_value=set())
        brief = agent.cluster_brief()
        assert brief["partition_filter"] in ("public", "none")

    def test_nothing_established_filters_nothing(self, mocker):
        """Reporting zero partitions because no rule could answer would be the
        most useless possible brief."""
        mocker.patch("slurmate.agent.fetch_user_partitions", return_value=None)
        mocker.patch("slurmate.agent.fetch_reachable_partitions", return_value=None)
        mocker.patch("slurmate.agent.fetch_public_partitions", return_value=[])
        brief = agent.cluster_brief()
        assert brief["partition_filter"] == "none"
        assert brief["partitions_elided"] == 0
        assert brief["partitions"]


class TestQueueDepth:
    """A capacity table says what a partition is, never whether anything can
    get on it today. That is usually what the choice turns on."""

    def test_every_row_carries_a_queue(self):
        for row in agent.cluster_brief()["partitions"]:
            assert "queue" in row

    def test_an_unreadable_squeue_reads_as_unknown_not_as_idle(self, mocker):
        """The single most inviting thing a chooser can be told."""
        mocker.patch("slurmate.agent.fetch_queue_depth", return_value=None)
        for row in agent.cluster_brief()["partitions"]:
            assert row["queue"] is None

    def test_a_partition_with_no_jobs_is_zero_not_unknown(self, mocker):
        mocker.patch("slurmate.agent.fetch_queue_depth",
                     return_value={"gpu-shared": {"running": 3, "pending": 9}})
        rows = {p["name"]: p["queue"] for p in agent.cluster_brief(all_partitions=True)["partitions"]}
        assert rows["gpu-shared"] == {"running": 3, "pending": 9}
        assert all(q == {"running": 0, "pending": 0}
                   for name, q in rows.items() if name != "gpu-shared")

    def test_only_running_and_pending_are_counted(self, mocker):
        """A COMPLETING or CANCELLED job is leaving; counting it overstates
        the wait it implies."""
        from slurmate import system_utils as su

        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(
            "gpu|RUNNING\ngpu|PENDING\ngpu|COMPLETING\ngpu|CANCELLED\n", "", 0))
        assert su.fetch_queue_depth() == {"gpu": {"running": 1, "pending": 1}}

    def test_the_default_partition_marker_is_stripped_from_the_name(self, mocker):
        """`squeue` prints the site default as ``name*``, and a key that does
        not match `sinfo`'s would silently report every default as idle."""
        from slurmate import system_utils as su

        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command",
                            return_value=("caslake*|RUNNING\n", "", 0))
        assert su.fetch_queue_depth() == {"caslake": {"running": 1, "pending": 0}}


class TestTheEnvironmentListsAreGatedOnCost:
    def test_absent_by_default_rather_than_empty(self):
        """"Not looked up" must not be readable as "this cluster has no
        modules". Measured: the module list is 5.3 s and 526 names here."""
        assert "environment" not in agent.cluster_brief()

    def test_full_includes_them(self):
        env = agent.cluster_brief(full=True)["environment"]
        assert set(env) == {"conda_envs", "modules"}


class TestJsonIsMachineReadable:
    """Exactly one document on stdout, whatever the terminal is doing."""

    @pytest.mark.parametrize("argv", [
        ("brief", "--json"),
        ("check", "--json", "-p", MOCK_GPU_PARTITION, "--command", "true"),
    ])
    def test_stdout_parses_and_carries_no_escape_sequence(self, argv):
        done = _cli(*argv, env_extra={"FORCE_COLOR": "1"})
        assert ESC not in done.stdout, "an ANSI sequence reached a JSON stream"
        payload = json.loads(done.stdout)
        assert payload["schema_version"] == agent.SCHEMA_VERSION

    def test_the_banner_never_reaches_a_json_stream(self):
        """`--print` already keeps this discipline; so must these. A caller
        piping into a parser must not have to strip a banner off the front."""
        done = _cli("brief", "--json")
        assert "SLURMATE" not in done.stdout
        assert done.stdout.lstrip().startswith("{")

    def test_the_mock_warning_goes_to_stdout_as_data_not_as_a_banner(self):
        assert json.loads(_cli("brief", "--json").stdout)["mock"] is True


class TestExitCodes:
    def test_a_clean_request_exits_zero(self):
        done = _cli("check", "--json", "-p", MOCK_GPU_PARTITION, "--command", "true")
        assert done.returncode == 0, done.stdout + done.stderr

    def test_an_error_exits_one(self):
        done = _cli("check", "--json", "-p", "no-such-partition", "--command", "true")
        assert done.returncode == 1
        assert json.loads(done.stdout)["ok"] is False

    def test_a_warning_alone_does_not_fail(self):
        """`validate_job_config` documents warnings as "may be rejected"
        against figures that can undercount a heterogeneous partition. Exiting
        non-zero on one would make honest uncertainty look like a refusal, and
        a caller looping on the exit code could never converge."""
        result = agent.check_request({"partition": MOCK_GPU_PARTITION, "cpus": 100000,
                                      "command": "true"})
        assert any(f["level"] == "warning" for f in result["findings"])
        assert not any(f["level"] == "error" for f in result["findings"])
        assert result["ok"] is True

    def test_checking_nothing_is_a_usage_error(self):
        done = _cli("check", "--json")
        assert done.returncode == 2
        assert done.stdout == ""

    def test_a_script_cannot_be_combined_with_job_flags(self, tmp_path):
        """Two sources for one answer produced a verdict that contradicted
        itself: the flag cleared slurmate's finding while `sbatch --test-only`,
        which can only judge the text on disk, still refused it."""
        script = tmp_path / "job.sbatch"
        script.write_text(f"#!/bin/bash\n#SBATCH -p {MOCK_GPU_PARTITION}\ntrue\n")
        done = _cli("check", "--script", str(script), "-p", "bigmem")
        assert done.returncode == 2
        assert "cannot be combined" in done.stderr


class TestTheVerbsDoNotCollideWithTheExistingCli:
    def test_a_bare_slurmate_is_still_the_wizard_not_a_verb(self):
        assert not agent.is_verb([])
        assert not agent.is_verb(["--partition", "brief"])

    def test_only_the_first_token_is_read_as_a_verb(self):
        """A value that happens to read "brief" always arrives after its flag,
        so there is no existing invocation this can capture."""
        assert agent.is_verb(["brief"])
        assert agent.is_verb(["brief", "--json"])
        assert not agent.is_verb(["--json", "brief"])

    def test_the_verb_namespace_was_free_before_this_existed(self):
        """`parse_args` declares no positionals, which is the whole reason a
        verb can be added without breaking a command anybody types today."""
        from slurmate.main import parse_args

        with pytest.raises(SystemExit):
            parse_args(["brief"])


class TestTheBundledSkill:
    def test_it_ships_inside_the_package(self):
        assert "slurmate brief --json" in agent.skill_text()

    def test_it_names_the_traps_that_cannot_be_inferred(self):
        """The whole value of the file: an agent can read a partition table on
        its own, but not that a "+" means a floor or that a feature-only GPU
        model cannot be asked for by GRES type."""
        text = agent.skill_text()
        for phrase in ("mock", "heterogeneous", "nodes_up",
                       "gpu_request_formats", "--test-only"):
            assert phrase in text, phrase

    def test_installing_writes_it_and_refuses_to_clobber(self, tmp_path):
        path, written = agent.install_skill(str(tmp_path))
        assert written and Path(path).is_file()
        again, written_again = agent.install_skill(str(tmp_path))
        assert again == path and written_again is False
        Path(path).write_text("edited by the user")
        _, forced = agent.install_skill(str(tmp_path), force=True)
        assert forced and "slurmate brief" in Path(path).read_text()

    def test_it_is_declared_as_package_data(self):
        """A wheel without it makes `slurmate skill --install` fail on exactly
        the installs that cannot fall back to a checkout."""
        pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
        assert re.search(r'slurmate\s*=\s*\[[^\]]*data/SKILL\.md', pyproject)


class TestTheAgentPathStaysLight:
    def test_neither_verb_loads_the_wizard(self):
        """An agent pays 0.16 s, not 0.46 s. `prompt_toolkit` alone was 269 ms
        of a 364 ms import; see test_startup_cost."""
        code = (
            "import sys, json;"
            "sys.argv = ['slurmate', 'brief'];"
            "import slurmate.agent;"
            "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
        )
        done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, check=True,
                              cwd=str(Path(__file__).resolve().parent.parent))
        loaded = set(json.loads(done.stdout))
        assert "prompt_toolkit" not in loaded
        assert "questionary" not in loaded
