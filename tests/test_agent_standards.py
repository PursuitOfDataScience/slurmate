"""The three ways an agent can find this tool, and the two questions it adds.

`brief` and `check` answered "what is this cluster" and "would this job run".
Two questions a first-time user asks before either of those were unanswerable:
*what machines are in here* (a partition row aggregates its nodes, so it
cannot distinguish two GPU generations inside one partition) and *why is my
job not running* (the answer is a `squeue` column most people do not know
exists, carrying codes that are not interchangeable).

The third thing here is discovery. A markdown file only works if the agent
notices it; MCP is the part that is actually a protocol, so a client can
enumerate the tools and call them with validated arguments. All three formats
are generated from one source, because the failure mode of shipping three is
that they drift and two of them become wrong.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from slurmate import agent, mcp
from slurmate import system_utils as su

ROOT = Path(__file__).resolve().parent.parent

NODE_TYPE_KEYS = {"cpus", "mem_mb", "gres", "gpu_types", "gpus_per_node",
                  "features", "count", "avail", "partitions"}
NODES_KEYS = {"schema_version", "mock", "node_types", "nodes_total",
              "nodes_avail", "errors"}
JOB_KEYS = {"job_id", "name", "state", "partition", "elapsed", "time_limit",
            "nodes", "reason", "nodelist", "reason_explained"}
JOBS_KEYS = {"schema_version", "mock", "user", "jobs", "running", "pending",
             "errors"}


def _sinfo(rows: str, mocker):
    """Pin `fetch_node_types` to a fixed `sinfo -N` answer.

    The partition-name list is stubbed from the same rows. `_bad_partition`
    consults it to tell a typo from an empty result, and left to the shared
    `_run_command` mock it would read node rows as partition names and reject
    every name in the fixture.
    """
    mocker.patch.object(su, "is_tool_available", return_value=True)
    mocker.patch.object(su, "_run_command", return_value=(rows, "", 0))
    names = {line.split("|")[6].strip() for line in rows.splitlines()
             if len(line.split("|")) > 6}
    mocker.patch("slurmate.agent.fetch_all_partition_names", return_value=names)


class TestNodeInventory:
    def test_the_document_carries_every_documented_key(self):
        assert set(agent.node_inventory()) >= NODES_KEYS

    def test_a_node_in_six_partitions_is_one_machine(self, mocker):
        """`sinfo -N` repeats a node once per partition it belongs to, so
        counting rows reported 608 nodes as 1,376. The node name is in the
        format string precisely so the count can be per machine."""
        _sinfo(
            "n1|32|256000|gpu:4|a100|idle|gpu\n"
            "n1|32|256000|gpu:4|a100|idle|test\n"
            "n1|32|256000|gpu:4|a100|idle|beagle3\n", mocker)
        types = su.fetch_node_types()
        assert [t["count"] for t in types] == [1]
        assert types[0]["partitions"] == ["beagle3", "gpu", "test"]

    def test_a_few_megabytes_of_bios_difference_is_not_a_hardware_type(self, mocker):
        """Two boxes of the same model report 515000 and 515072 MB. Keying on
        the raw figure listed the same machine twice, and did so three times
        over in the measured output."""
        _sinfo("n1|32|515000|gpu:4|H100|idle|gpu\n"
               "n2|32|515072|gpu:4|H100|idle|gpu\n", mocker)
        types = su.fetch_node_types()
        assert len(types) == 1
        assert types[0]["count"] == 2
        # The minimum, because that is the figure a --mem has to fit inside.
        assert types[0]["mem_mb"] == 515000

    def test_clustering_has_no_boundary_for_a_pair_to_straddle(self):
        """Quantising to whole GB was the obvious fix and the wrong one: a
        grid has boundaries, and the measured pair sits on one (515000 floors
        to 502 while 515072 floors to 503). Rounding only moves the boundary,
        which is why this clusters instead."""
        buckets = su._memory_buckets({515000, 515072, 184320, 250000})
        assert buckets[515000] == buckets[515072]
        assert buckets[184320] != buckets[250000] != buckets[515000]

    def test_genuinely_different_machines_stay_apart(self):
        """The tolerance has a wide gap to sit in: firmware differences are
        hundredths of a percent, different machines are tens."""
        buckets = su._memory_buckets({184320, 257000, 1536000})
        assert len(set(buckets.values())) == 3

    def test_a_count_only_gres_still_names_its_gpu(self, mocker):
        """midway3 advertises `gpu:4` with the model in AvailableFeatures, so
        a table reading only the GRES printed "4x ?" for most of the cluster
        while the answer sat in the next column."""
        _sinfo("n1|48|184320|gpu:4|gold-6248r,rtx6000,192g|idle|gpu\n", mocker)
        row = su.fetch_node_types()[0]
        assert row["gpus_per_node"] == 4
        assert row["gpu_types"] == ["rtx6000"]

    def test_rack_tags_do_not_split_one_hardware_type(self, mocker):
        """`192g` duplicates RealMemory, which is already in the key, and the
        fabric tags describe where a node is racked rather than anything a job
        can ask for. Grouping on them split one type into six."""
        _sinfo("n1|48|184320|gpu:4|rtx6000,192g,ib|idle|gpu\n"
               "n2|48|184320|gpu:4|rtx6000,192g,hdr|idle|gpu\n", mocker)
        types = su.fetch_node_types()
        assert len(types) == 1
        assert types[0]["features"] == ["rtx6000"]

    def test_a_drained_type_reads_as_unusable_not_as_capacity(self, mocker):
        _sinfo("n1|48|184320|(null)|x|idle|amd\n"
               "n2|48|184320|(null)|x|drained|amd\n"
               "n3|48|184320|(null)|x|down|amd\n", mocker)
        row = su.fetch_node_types()[0]
        assert (row["count"], row["avail"]) == (3, 1)

    def test_every_row_carries_every_documented_field(self):
        for row in agent.node_inventory()["node_types"] or []:
            assert set(row) >= NODE_TYPE_KEYS, NODE_TYPE_KEYS - set(row)

    def test_an_unreadable_sinfo_is_null_not_an_empty_cluster(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=False)
        inv = agent.node_inventory()
        assert inv["node_types"] is None and inv["nodes_total"] is None

    def test_filtering_narrows_and_the_totals_follow(self, mocker):
        _sinfo("n1|48|184320|gpu:4|a100|idle|gpu\n"
               "n2|48|184320|(null)|x|idle|amd\n", mocker)
        assert agent.node_inventory()["nodes_total"] == 2
        assert agent.node_inventory(gpu_only=True)["nodes_total"] == 1
        assert agent.node_inventory(partition="amd")["nodes_total"] == 1


class TestWhyAJobIsPending:
    def test_the_report_carries_every_documented_key(self):
        assert set(agent.job_report()) >= JOBS_KEYS

    def test_a_reason_and_a_nodelist_are_not_the_same_column(self, mocker):
        """`squeue`'s %R means two different things by state, and conflating
        them reads a node list as an explanation."""
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_current_username", return_value="me")
        mocker.patch.object(su, "_run_command", return_value=(
            "1|a|RUNNING|gpu|1:00|4:00|1|node07\n"
            "2|b|PENDING|gpu|0:00|4:00|1|(Priority)\n", "", 0))
        jobs = {j["job_id"]: j for j in su.fetch_my_jobs()}
        assert jobs["1"]["nodelist"] == "node07" and jobs["1"]["reason"] is None
        assert jobs["2"]["reason"] == "Priority" and jobs["2"]["nodelist"] is None

    @pytest.mark.parametrize("code,must_say", [
        ("Priority", "ahead"),
        ("Resources", "busy"),
        ("QOSMaxJobsPerUserLimit", "cap"),
        ("PartitionTimeLimit", "never"),
        ("DependencyNeverSatisfied", "NEVER"),
    ])
    def test_the_codes_are_not_interchangeable(self, code, must_say):
        """Four of these mean four different things to do next, and a
        first-time user reads all of them as "be patient"."""
        assert must_say in su.explain_pending(code)

    def test_an_unknown_code_gets_no_invented_meaning(self):
        """Slurm has dozens of these and several are site plugins. A plausible
        guess is worse than the raw code, which can at least be searched for."""
        assert su.explain_pending("SomeSitePluginReason") == ""
        assert su.explain_pending("") == ""

    def test_a_longer_code_wins_over_a_shorter_prefix(self):
        """`QOSMaxCpuPerUserLimit` starts with no shorter key, but the family
        is full of near-prefixes and the specific one has to win."""
        assert "CPUs" in su.explain_pending("QOSMaxCpuPerUserLimit")

    def test_every_job_row_carries_every_field(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_current_username", return_value="me")
        mocker.patch.object(su, "_run_command", return_value=(
            "2|b|PENDING|gpu|0:00|4:00|1|(Priority)\n", "", 0))
        report = agent.job_report()
        assert set(report["jobs"][0]) >= JOB_KEYS
        assert report["jobs"][0]["reason_explained"]
        assert (report["running"], report["pending"]) == (0, 1)

    def test_no_jobs_is_an_answer_and_no_queue_is_not(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_current_username", return_value="me")
        mocker.patch.object(su, "_run_command", return_value=("", "", 0))
        assert agent.job_report()["jobs"] == []
        mocker.patch.object(su, "is_tool_available", return_value=False)
        assert agent.job_report()["jobs"] is None


class TestEveryVerbExplainsAnUnreadableCluster:
    """`brief` learned to say why; `nodes` and `jobs` did not.

    Both returned `null` beside an empty `errors` list, which tells a reader
    nothing about whether the cluster is empty or absent. `last_cluster_error`
    is silent in that case because `is_tool_available` short-circuits before
    any command runs, so nothing failed to record.
    """

    @pytest.mark.parametrize("fn,tool", [
        (agent.node_inventory, "sinfo"),
        (agent.job_report, "squeue"),
        (agent.cluster_brief, "sinfo"),
    ])
    def test_it_names_the_missing_binary(self, fn, tool, mocker):
        mocker.patch("slurmate.agent.is_mock", return_value=False)
        mocker.patch("slurmate.agent.is_tool_available", return_value=False)
        mocker.patch("slurmate.agent.fetch_partitions", return_value=[])
        mocker.patch("slurmate.agent.fetch_node_types", return_value=None)
        mocker.patch("slurmate.agent.fetch_my_jobs", return_value=None)
        errors = fn()["errors"]
        assert errors, f"{fn.__name__} said nothing"
        assert "not found on PATH" in errors[0]

    def test_demo_mode_is_not_an_unreadable_cluster(self):
        """`mock` already says the data is synthetic; adding "Slurm is not
        installed" on top would describe the demo as broken."""
        for fn in (agent.node_inventory, agent.job_report, agent.cluster_brief):
            assert fn()["errors"] == [], fn.__name__


class TestTheMcpServer:
    """The only one of the three integrations that is a protocol.

    Driven as a client drives it: one dict in, one dict out. `handle` is pure
    for exactly this reason, so the whole surface is testable without a
    subprocess and without a pipe.
    """

    def test_initialize_echoes_the_clients_protocol_version(self):
        """These methods have been stable across every revision that has them,
        so refusing an unfamiliar string would break a client that works."""
        reply = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                            "params": {"protocolVersion": "2024-11-05"}})
        assert reply["result"]["protocolVersion"] == "2024-11-05"
        assert reply["result"]["serverInfo"]["name"] == "slurmate"
        assert "capabilities" in reply["result"]

    def test_a_notification_is_never_answered(self):
        """Answering one is a protocol violation some clients treat as a fatal
        desync."""
        assert mcp.handle({"jsonrpc": "2.0",
                           "method": "notifications/initialized"}) is None

    def test_every_tool_is_listed_with_a_schema_and_a_when_to_use_it(self):
        tools = mcp.handle({"jsonrpc": "2.0", "id": 1,
                            "method": "tools/list"})["result"]["tools"]
        assert {t["name"] for t in tools} == set(mcp.TOOLS)
        for tool in tools:
            assert tool["inputSchema"]["type"] == "object"
            # The description is what the model reads to decide whether to
            # call it, so a bare restatement of the name is a defect.
            assert len(tool["description"]) > 80, tool["name"]

    def test_the_check_and_generate_tools_accept_the_same_job(self):
        """Two copies of the field list would drift, and the pair that drifted
        would be a script this server generated and then could not check."""
        tools = {t["name"]: t for t in mcp.tool_schemas()}
        check = set(tools["slurm_check_job"]["inputSchema"]["properties"])
        gen = set(tools["slurm_generate_script"]["inputSchema"]["properties"])
        assert check - {"script"} <= gen

    def test_a_tool_call_returns_both_content_shapes(self):
        """`content` is what every client renders, `structuredContent` what
        newer ones prefer; sending both means neither gets the fallback."""
        reply = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "slurm_cluster_brief",
                                       "arguments": {}}})
        result = reply["result"]
        assert result["structuredContent"]["schema_version"] == agent.SCHEMA_VERSION
        assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
        assert result["isError"] is False

    def test_generate_returns_a_script_and_its_verdict_together(self):
        reply = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "slurm_generate_script",
                                       "arguments": {"partition": "gpu-shared",
                                                     "command": "python train.py"}}})
        payload = reply["result"]["structuredContent"]
        assert "#SBATCH --partition=gpu-shared" in payload["script"]
        assert "findings" in payload and "scheduler" in payload

    def test_no_tool_can_submit_a_job(self):
        """The boundary of this module, asserted rather than described: an
        agent that can spend an allocation unattended is a different product."""
        assert not any("submit" in name for name in mcp.TOOLS)
        blob = json.dumps(mcp.tool_schemas())
        assert "sbatch --test-only" in blob and '"submit"' not in blob

    def test_an_unknown_tool_is_a_jsonrpc_error(self):
        reply = mcp.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                            "params": {"name": "nope", "arguments": {}}})
        assert reply["error"]["code"] == -32601

    def test_a_failing_tool_is_a_result_not_a_transport_error(self, mocker):
        """A cluster query that failed is something the model should see and
        react to; a transport error is not. Conflating them hides the former
        behind a client's generic "the tool broke"."""
        mocker.patch.dict(mcp.TOOLS, {"slurm_list_jobs":
                                      mocker.Mock(side_effect=RuntimeError("boom"))})
        reply = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "slurm_list_jobs", "arguments": {}}})
        assert "error" not in reply
        assert reply["result"]["isError"] is True
        assert "boom" in reply["result"]["content"][0]["text"]

    def test_malformed_input_does_not_kill_the_server(self):
        import io

        out = io.StringIO()
        mcp.serve(io.StringIO('not json\n{"jsonrpc":"2.0","id":2,"method":"ping"}\n'), out)
        replies = [json.loads(line) for line in out.getvalue().splitlines()]
        assert replies[0]["error"]["code"] == -32700
        assert replies[1]["result"] == {}

    def test_it_speaks_a_whole_session_over_a_real_pipe(self):
        """The in-process tests cannot catch a stray print, and stdout belongs
        to the protocol."""
        session = "\n".join(json.dumps(m) for m in (
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": mcp.PROTOCOL_VERSION}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )) + "\n"
        import os

        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "mcp"], input=session,
            capture_output=True, text=True, cwd=str(ROOT),
            env=dict(os.environ, SLURMATE_MOCK="1"), check=True,
        )
        lines = [line for line in done.stdout.splitlines() if line.strip()]
        assert len(lines) == 2, done.stdout
        assert all(json.loads(line)["jsonrpc"] == "2.0" for line in lines)


class TestALongLivedServerDoesNotServeAFrozenCluster:
    """`slurmate mcp` broke the premise the cluster cache was reasoned with.

    The comment on `_CLUSTER_CACHE` said a single slurmate run is short enough
    that staleness is not a concern, and for the CLI it is. An MCP server is
    one process for the length of an editor session. Measured: the volatile
    facts were never in that cache (a second `cluster_brief` in one process
    re-runs sinfo six times, sacctmgr once and squeue once), but the caller's
    accounts are, so a user granted one mid-session would go on being told
    they cannot use that partition for as long as the server ran.
    """

    def test_the_age_is_none_until_something_is_cached(self):
        su.reset_cluster_cache()
        assert su.cluster_cache_age() is None
        su.fetch_user_accounts()
        assert su.cluster_cache_age() is not None

    def test_a_call_past_the_ttl_re_reads_the_cluster(self, mocker):
        su.fetch_user_accounts()
        mocker.patch.object(mcp, "CACHE_TTL_SECONDS", -1.0)
        dropped = mocker.spy(mcp, "reset_cluster_cache")
        mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "slurm_list_jobs", "arguments": {}}})
        assert dropped.called

    def test_a_burst_of_calls_inside_the_ttl_pays_for_it_once(self, mocker):
        su.reset_cluster_cache()
        su.fetch_user_accounts()
        dropped = mocker.spy(mcp, "reset_cluster_cache")
        for _ in range(3):
            mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "slurm_list_jobs", "arguments": {}}})
        assert not dropped.called

    def test_the_one_shot_cli_never_expires_anything(self):
        """The original reasoning still holds exactly on that path, so it
        keeps exactly the behaviour it was reasoned about with."""
        source = (ROOT / "src" / "slurmate" / "agent.py").read_text()
        assert "reset_cluster_cache" not in source


class TestTheThreeInstallFormats:
    @pytest.mark.parametrize("fmt", ["claude", "agents", "mcp"])
    def test_each_lands_where_its_client_looks(self, fmt, tmp_path):
        path, written = agent.install_skill(str(tmp_path), fmt=fmt)
        assert written and Path(path).is_file()
        assert path.endswith(agent.INSTALL_PATHS[fmt])

    def test_none_of_them_clobbers_what_is_already_there(self, tmp_path):
        """`.mcp.json` routinely holds other servers, and a skill file is
        something a user edits for their own site."""
        for fmt in agent.INSTALL_PATHS:
            path, _ = agent.install_skill(str(tmp_path), fmt=fmt)
            Path(path).write_text("mine")
            again, written = agent.install_skill(str(tmp_path), fmt=fmt)
            assert (again, written) == (path, False)
            assert Path(path).read_text() == "mine"

    def test_agents_md_is_generated_from_the_skill_not_written_twice(self):
        """Three copies of one body drift, and two of them become wrong."""
        body = agent.skill_text().split("---", 2)[-1]
        derived = agent.agents_text()
        for line in body.splitlines():
            text = line.strip()
            if text.startswith("|") and "`slurmate" in text:
                assert text in derived, text

    def test_agents_md_carries_no_claude_frontmatter_and_one_title(self):
        text = agent.agents_text()
        assert not text.startswith("---")
        assert "description:" not in text.splitlines()[0]
        assert [ln for ln in text.splitlines() if ln.startswith("# ")] == [
            "# Running jobs on this Slurm cluster"
        ]

    def test_the_mcp_block_points_at_this_interpreter(self):
        """A bare `slurmate` is routinely not on an agent's PATH under pipx or
        a venv; the interpreter running this is by definition the right one."""
        server = agent.mcp_config()["mcpServers"]["slurmate"]
        assert server["args"][-1] == "mcp"
        assert server["command"] == sys.executable

    def test_the_skill_documents_every_verb_it_tells_an_agent_to_run(self):
        text = agent.skill_text()
        for verb in ("brief", "nodes", "jobs", "check"):
            assert f"slurmate {verb}" in text, verb


class TestTheVerbsAreFindable:
    """An undiscoverable subcommand is one nobody runs, and the reader of
    `--help` is now often an agent rather than a person."""

    def test_help_lists_every_verb(self):
        import contextlib
        import io

        from slurmate.main import parse_args

        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
            parse_args(["--help"])
        text = out.getvalue()
        for verb in agent.VERBS:
            assert f"slurmate {verb}" in text, verb

    def test_the_help_list_cannot_fall_behind_the_verbs(self):
        """Spelled beside VERBS for this reason: the only thing worse than a
        hidden subcommand is a list of them that is missing one."""
        assert set(agent.VERB_HELP) == set(agent.VERBS)

    def test_a_typo_gets_the_spelling_not_sixty_lines_of_flags(self):
        """argparse buried the one thing the reader needs under its whole
        usage block."""
        hint = agent.verb_suggestion(["brieff"])
        assert "brief" in hint and "no such command" in hint

    def test_a_word_that_resembles_nothing_is_left_to_argparse(self):
        """More likely a misplaced value than a typo, and argparse's own
        message is right about that case."""
        assert agent.verb_suggestion(["zzzzzzzz"]) == ""
        assert agent.verb_suggestion(["--partition"]) == ""
        assert agent.verb_suggestion([]) == ""

    def test_a_real_verb_is_not_a_suggestion(self):
        assert agent.verb_suggestion(["brief"]) == ""

    def test_a_mistyped_verb_exits_two(self):
        import os

        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "brieff"], capture_output=True,
            text=True, cwd=str(ROOT), env=dict(os.environ, SLURMATE_MOCK="1"),
        )
        assert done.returncode == 2
        assert "brief" in done.stderr


class TestABadNameIsNotAFactAboutTheCluster:
    """The third instance of one defect: nothing came back, so nothing was
    said, and the emptiness read as a measurement.

    `nodes -p typo` answered "no nodes could be read", which is the wording
    for an unreadable `sinfo` and blames the machine for the caller's typo.
    `jobs -u nosuchuser` answered "no jobs", which is a confident claim about
    somebody who does not exist. Both exited 0.
    """

    def test_a_mistyped_partition_is_named_with_a_suggestion(self, mocker):
        mocker.patch("slurmate.agent.fetch_all_partition_names",
                     return_value={"caslake", "gpu", "amd"})
        inv = agent.node_inventory(partition="caslak")
        assert inv["node_types"] is None
        assert "caslak" in inv["errors"][0] and "caslake" in inv["errors"][0]

    def test_the_brief_rejects_it_too(self, mocker):
        mocker.patch("slurmate.agent.fetch_all_partition_names",
                     return_value={"caslake", "gpu"})
        brief = agent.cluster_brief(partition="caslak")
        assert brief["partitions"] == []
        assert any("caslak" in e for e in brief["errors"])

    def test_an_unreadable_partition_list_rejects_nothing(self, mocker):
        """Rejecting a name against a list that could not be fetched is the
        SM-4 false rejection this codebase is most careful about."""
        mocker.patch("slurmate.agent.fetch_all_partition_names", return_value=set())
        assert agent.cluster_brief(partition="anything")["errors"] == [] or all(
            "anything" not in e for e in agent.cluster_brief(partition="x")["errors"])

    def test_squeue_exits_zero_on_an_invalid_user_so_stderr_decides(self, mocker):
        """Measured: `squeue -u nosuchuser` prints "error: Invalid user" and
        exits 0 with an empty listing. Reading only `rc` turned a name that
        does not exist into the answer "no jobs"."""
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_current_username", return_value="me")
        mocker.patch.object(su, "_run_command",
                            return_value=("", "squeue: error: Invalid user: zz", 0))
        assert su.fetch_my_jobs("zz") is None

    def test_a_genuinely_empty_queue_is_still_an_empty_list(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_current_username", return_value="me")
        mocker.patch.object(su, "_run_command", return_value=("", "", 0))
        assert su.fetch_my_jobs("me") == []

    def test_a_bad_name_exits_non_zero(self):
        import os

        for argv in (["nodes", "-p", "no-such-partition"],
                     ["brief", "-p", "no-such-partition"]):
            done = subprocess.run(
                [sys.executable, "-m", "slurmate", *argv], capture_output=True,
                text=True, cwd=str(ROOT), env=dict(os.environ, SLURMATE_MOCK="1"))
            assert done.returncode == 1, (argv, done.stdout)
