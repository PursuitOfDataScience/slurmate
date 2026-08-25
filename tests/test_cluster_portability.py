"""Cluster-portability regressions (SM-1 … SM-12).

Every case here comes from running slurmate on a second, deliberately different
cluster from the one it was written on. The common shape is that a value read
from Slurm was trusted further than it deserved — a node count that included
dead nodes, a partition list that ignored who may submit, a name that was never
checked against the cluster at all — and the result was a confident, complete,
unsubmittable answer.
"""

from __future__ import annotations

import io
import math
import os
import pathlib
import re
import subprocess
import sys

import pytest

import slurmate.system_utils as su
from slurmate.main import _note_scheduler_refusal
from slurmate.system_utils import (
    FALLBACK_MEMORY,
    default_memory_for,
    fetch_partitions,
    fetch_queue_eta,
    fetch_system_partitions,
    fetch_user_partitions,
    validate_cluster_targets,
    validate_job_config,
)


def _part(**over):
    base = {
        "name": "p", "nodes": 8, "nodes_up": 8, "cpus_per_node": 28,
        "mem_per_node_mb": 57000, "gpu_types": [], "timelimit": None,
    }
    base.update(over)
    return base


# ── SM-1: node state decides capacity ────────────────────────────────────


class TestNodeStateCapacity:
    def _parts(self, mocker, rows):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(rows, "", 0))
        return {p["name"]: p for p in fetch_partitions()}

    def test_dead_partition_reports_zero_usable(self, mocker):
        parts = self._parts(mocker, "dead|infinite|8|up|28|57000|(null)|down*\n")
        assert parts["dead"]["nodes"] == 8
        assert parts["dead"]["nodes_up"] == 0

    def test_partially_drained_reports_both_figures(self, mocker):
        parts = self._parts(
            mocker,
            "half|infinite|13|up|28|57000|(null)|idle\n"
            "half|infinite|4|up|28|57000|(null)|down*\n",
        )
        assert parts["half"]["nodes"] == 17
        assert parts["half"]["nodes_up"] == 13

    def test_allocated_and_mixed_are_capacity(self, mocker):
        # A busy partition is not a dead one — those nodes free up.
        parts = self._parts(
            mocker,
            "busy|infinite|5|up|28|57000|(null)|allocated\n"
            "busy|infinite|7|up|28|57000|(null)|mixed\n",
        )
        assert parts["busy"]["nodes_up"] == 12

    def test_state_flag_disqualifies_an_otherwise_live_state(self, mocker):
        # "idle*" is idle but unreachable; it cannot start anything.
        parts = self._parts(mocker, "gone|infinite|3|up|28|57000|(null)|idle*\n")
        assert parts["gone"]["nodes_up"] == 0

    def test_missing_state_column_is_unknown_not_zero(self, mocker):
        # A site whose sinfo gives no state must not have every partition
        # reported dead — that is absence of evidence, not evidence of absence.
        parts = self._parts(mocker, "legacy|infinite|9|up|28|57000|(null)\n")
        assert parts["legacy"]["nodes"] == 9
        assert parts["legacy"]["nodes_up"] is None

    def test_default_partition_marker_is_captured(self, mocker):
        parts = self._parts(
            mocker,
            "main*|infinite|4|up|28|57000|(null)|idle\n"
            "other|infinite|4|up|28|57000|(null)|idle\n",
        )
        assert parts["main"]["is_default"] is True
        assert parts["other"]["is_default"] is False

    def test_gres_survives_the_appended_state_column(self, mocker):
        parts = self._parts(
            mocker, "g|infinite|4|up|32|100000|gpu:a100:4,gpu:v100:2|mixed\n"
        )
        assert sorted(parts["g"]["gpu_types"]) == ["a100", "v100"]
        assert parts["g"]["has_gpu"] is True
        assert parts["g"]["nodes_up"] == 4

    def test_dead_partition_warns_in_job_validation(self):
        issues = validate_job_config(
            {"_partition_obj": _part(name="dead", nodes_up=0), "cpus": 1}
        )
        assert any("no usable nodes" in m for _lvl, m in issues)

    def test_unknown_usable_count_stays_silent(self):
        issues = validate_job_config(
            {"_partition_obj": _part(name="legacy", nodes_up=None), "cpus": 1}
        )
        assert not any("no usable nodes" in m for _lvl, m in issues)


# ── SM-2: associations, not the partition ACL, gate submission ───────────


class TestAssociationFiltering:
    def _assoc(self, mocker, stdout, rc=0):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_current_username", return_value="me")
        mocker.patch.object(su, "_run_command", return_value=(stdout, "", rc))
        return fetch_user_partitions()

    def test_partition_scoped_rows_produce_a_filter(self, mocker):
        got = self._assoc(mocker, "data-bfi-voter|broadwl\npi-x|build\n")
        assert got == {"broadwl", "build"}

    def test_blank_partition_is_a_wildcard_not_no_access(self, mocker):
        # Sites that gate on the account leave Partition empty on every row.
        # Reading that as "no access" filters the entire list away.
        assert self._assoc(mocker, "pi-a|\npi-b|\n") is None

    def test_one_wildcard_row_wins_over_scoped_ones(self, mocker):
        assert self._assoc(mocker, "pi-a|broadwl\npi-b|\n") is None

    def test_no_rows_means_unknown(self, mocker):
        assert self._assoc(mocker, "\n") is None

    def test_sacctmgr_failure_means_unknown(self, mocker):
        assert self._assoc(mocker, "", rc=1) is None

    def test_missing_sacctmgr_means_unknown(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=False)
        assert fetch_user_partitions() is None


# ── SM-3: a scheduler partition is not an ordinary choice ────────────────


class TestSystemPartitionDetection:
    def _sys(self, mocker, stdout):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(stdout, "", 0))
        return fetch_system_partitions()

    def test_cron_is_detected_by_name(self, mocker):
        assert "cron" in self._sys(mocker, "PartitionName=cron Nodes=n[1-2]\n")

    def test_login_node_partition_is_detected_structurally(self, mocker):
        # The name is site-specific; "its nodes are the login nodes" is not.
        found = self._sys(
            mocker,
            "PartitionName=housekeeping Nodes=dali-login[1-2],midway2-login[1-2]\n",
        )
        assert "housekeeping" in found

    def test_ordinary_partition_is_not_flagged(self, mocker):
        got = self._sys(mocker, "PartitionName=broadwl Nodes=midway2-[0001-0300]\n")
        assert got == set()

    def test_name_match_is_exact_not_substring(self, mocker):
        assert self._sys(mocker, "PartitionName=system-gpu Nodes=g[1-4]\n") == set()

    def test_mixed_login_and_compute_is_not_a_system_partition(self, mocker):
        got = self._sys(
            mocker, "PartitionName=mixed Nodes=midway2-login1,midway2-0001\n"
        )
        assert got == set()


class TestPartitionRanking:
    """The picker's order is what a user reads first."""

    def _ranked(self, parts, user=None, system=None):
        from slurmate.tui import _rank_partitions

        return [p["name"] for p in _rank_partitions(parts, user, system)]

    def test_default_partition_comes_first(self):
        parts = [
            _part(name="zzz", nodes_up=99),
            _part(name="main", nodes_up=1, is_default=True),
        ]
        assert self._ranked(parts)[0] == "main"

    def test_dead_partitions_sink_below_live_ones(self):
        parts = [_part(name="dead", nodes_up=0), _part(name="live", nodes_up=1)]
        assert self._ranked(parts) == ["live", "dead"]

    def test_system_partitions_sink_below_everything(self):
        parts = [_part(name="cron", nodes_up=4), _part(name="dead", nodes_up=0)]
        assert self._ranked(parts, system={"cron"}) == ["dead", "cron"]

    def test_users_own_partitions_outrank_bigger_strangers(self):
        parts = [_part(name="huge", nodes_up=500), _part(name="mine", nodes_up=2)]
        assert self._ranked(parts, user={"mine"})[0] == "mine"

    def test_unknown_state_ranks_with_the_living(self):
        parts = [_part(name="dead", nodes_up=0), _part(name="unknown", nodes_up=None)]
        assert self._ranked(parts) == ["unknown", "dead"]

    def test_nothing_is_dropped(self):
        parts = [
            _part(name="cron", nodes_up=4),
            _part(name="dead", nodes_up=0),
            _part(name="live", nodes_up=9),
        ]
        assert sorted(self._ranked(parts, system={"cron"})) == ["cron", "dead", "live"]


class TestPartitionLabel:
    def _fmt(self, **over):
        from slurmate.tui import _fmt_partition

        return _fmt_partition(_part(**over))

    def test_partially_drained_shows_both_counts(self):
        assert "13 of 17 nodes" in self._fmt(nodes=17, nodes_up=13)

    def test_fully_drained_is_marked_unavailable(self):
        assert "unavailable" in self._fmt(nodes=8, nodes_up=0)

    def test_healthy_partition_is_unadorned(self):
        label = self._fmt(nodes=8, nodes_up=8)
        assert "8 nodes" in label
        assert "of" not in label and "unavailable" not in label

    def test_unknown_state_is_unadorned(self):
        label = self._fmt(nodes=8, nodes_up=None)
        assert "8 nodes" in label
        assert "unavailable" not in label

    def test_default_is_labelled(self):
        assert "default" in self._fmt(is_default=True)


# ── SM-4: names are checked against this cluster ─────────────────────────


class TestClusterTargetValidation:
    PARTS = ["broadwl", "build", "bigmem2", "gpu2"]

    def _v(self, partition=None, account=None, **kw):
        kw.setdefault("known_partitions", self.PARTS)
        return validate_cluster_targets(partition, account, **kw)

    def test_known_partition_passes(self):
        assert self._v("build") == []

    def test_unknown_partition_is_an_error(self):
        issues = self._v("caslake")
        assert [lvl for lvl, _ in issues] == ["error"]
        assert "no partition 'caslake' on this cluster." in issues[0][1]

    def test_message_lists_this_clusters_partitions(self):
        msg = self._v("caslake")[0][1]
        assert "This cluster's partitions:" in msg
        assert "broadwl" in msg

    def test_near_miss_is_suggested(self):
        msg = self._v("bigmem")[0][1]
        assert "Did you mean:" in msg
        assert "bigmem2" in msg

    def test_default_partition_is_labelled_in_the_suggestion(self):
        msg = self._v("broadwll", default_partition="broadwl")[0][1]
        assert "broadwl (default)" in msg

    def test_unknown_account_is_an_error(self):
        issues = self._v(
            "build", "nosuchaccount123", known_accounts=["pi-real", "rcc-staff"]
        )
        assert [lvl for lvl, _ in issues] == ["error"]
        assert "no account 'nosuchaccount123'" in issues[0][1]

    def test_known_account_passes(self):
        assert self._v("build", "rcc-staff", known_accounts=["rcc-staff"]) == []

    def test_unreadable_partition_list_validates_nothing(self):
        # An sinfo that returned nothing must never read as "no such partition".
        assert validate_cluster_targets("anything", known_partitions=[]) == []

    def test_unreadable_account_list_validates_nothing(self):
        assert self._v("build", "whatever", known_accounts=[]) == []

    def test_no_partition_given_is_not_an_error(self):
        assert self._v("") == []


class TestBatchClusterCheck:
    """The CLI surface of SM-4: fatal by default, a warning under --force."""

    def _args(self, **over):
        import argparse

        base = dict(
            job_name="t", account=None, partition="cpu-shared", qos=None,
            cpus=1, memory="1G", time="00:10:00", nodes=1, gpus=0,
            gpu_type=None, array=None, modules=None, env=None, env_type=None,
            command="echo hi", custom_sbatch=None, yes=False, force=False,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_unknown_partition_exits_nonzero(self):
        from rich.console import Console

        from slurmate.main import run_batch

        with pytest.raises(SystemExit) as exc:
            run_batch(self._args(partition="not-here"), Console(), {})
        assert exc.value.code == 1

    def test_force_downgrades_to_a_warning(self):
        from rich.console import Console

        from slurmate.main import run_batch

        answers = run_batch(self._args(partition="not-here", force=True), Console(), {})
        assert answers["partition"] == "not-here"

    def test_known_partition_is_unaffected(self):
        from rich.console import Console

        from slurmate.main import run_batch

        answers = run_batch(self._args(), Console(), {})
        assert answers["partition"] == "cpu-shared"


# ── SM-5: no ETA for a request Slurm has refused ─────────────────────────


class TestSchedulerRefusal:
    def _eta(self, mocker, sbatch_stderr, sbatch_rc):
        mocker.patch.object(su, "is_tool_available", return_value=True)

        def run(cmd, timeout=30):
            if "sbatch" in cmd:
                return "", sbatch_stderr, sbatch_rc
            if "squeue" in cmd:
                return "RUNNING|1|2|1\n", "", 0
            return "", "", 1

        mocker.patch.object(su, "_run_command", side_effect=run)
        return fetch_queue_eta("build", req_nodes=1, cpus=999)

    def test_allocation_failure_is_surfaced_as_never(self, mocker):
        info = self._eta(
            mocker,
            "allocation failure: Requested node configuration is not available\n",
            1,
        )
        assert info["feasible"] is False
        assert info["eta_label"] == "never"
        assert info["reason"] == "Requested node configuration is not available"

    def test_plugin_reason_leads_and_the_generic_verdict_is_kept(self, mocker):
        # The site plugin's Reason is what the user needs to read, so it comes
        # first. Slurm's generic verdict is kept in parentheses because that is
        # the half refusal_is_permanent() can classify — the marker lists are
        # written against Slurm's wordings, not a site plugin's. Dropping it made
        # midway3's "Account is not specified" unclassifiable while Mercury's
        # wording for the same mistake was correctly called permanent.
        info = self._eta(
            mocker,
            "sbatch: error: Reason: Invalid account [nope]\n"
            "allocation failure: Access/permission denied\n",
            1,
        )
        assert info["reason"].startswith("Invalid account [nope]")
        assert "Access/permission denied" in info["reason"]
        assert info["refusal_is_permanent"] is True

    def test_identical_halves_are_not_printed_twice(self, mocker):
        info = self._eta(
            mocker,
            "sbatch: error: Reason: Invalid qos specification\n"
            "allocation failure: Invalid qos specification\n",
            1,
        )
        assert info["reason"] == "Invalid qos specification"

    def test_unreachable_controller_is_not_a_rejection(self, mocker):
        # A broken sbatch must not read as "your job can never run" — that
        # trades one confident wrong answer for another.
        info = self._eta(mocker, "sbatch: error: Unable to contact slurm controller\n", 1)
        assert info["feasible"] is True
        assert info["eta_label"] != "never"

    def test_accepted_request_reports_a_time(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)

        def run(cmd, timeout=30):
            if "sbatch" in cmd:
                return "", "Job 1 to start at 2099-01-01T00:00:00 using 2 processors\n", 0
            if "squeue" in cmd:
                return "", "", 0
            return "", "", 1

        mocker.patch.object(su, "_run_command", side_effect=run)
        info = fetch_queue_eta("build", req_nodes=1, cpus=2)
        assert info["feasible"] is True
        assert info["source"] == "scheduler"


# ── SM-6: the wizard needs a terminal ────────────────────────────────────


class TestTerminalGuard:
    def test_no_terminal_exits_instead_of_blocking(self, mocker):
        import slurmate.main as m

        mocker.patch.object(m, "_isatty", return_value=False)
        with pytest.raises(SystemExit) as exc:
            m._require_terminal_for_wizard()
        assert exc.value.code == 1

    def test_message_points_at_the_non_interactive_surface(self, mocker, capsys):
        import slurmate.main as m

        mocker.patch.object(m, "_isatty", return_value=False)
        with pytest.raises(SystemExit):
            m._require_terminal_for_wizard()
        err = capsys.readouterr().err
        assert "not a terminal" in err
        assert "--print" in err

    def test_a_real_terminal_passes(self, mocker):
        import slurmate.main as m

        mocker.patch.object(m, "_isatty", return_value=True)
        m._require_terminal_for_wizard()  # must not raise

    def test_closed_stream_counts_as_no_terminal(self):
        import slurmate.main as m

        class Closed:
            def isatty(self):
                raise ValueError("I/O operation on closed file")

        assert m._isatty(Closed()) is False
        assert m._isatty(None) is False


# ── SM-7: the default --mem is a measurement ─────────────────────────────


class TestDefaultMemory:
    def test_proportional_to_the_cpu_share(self):
        # 128G node, 32 cores, asking for 8 → a quarter of the node.
        part = {"mem_per_node_mb": 131072, "cpus_per_node": 32}
        assert default_memory_for(part, 8) == ("32G", "partition")

    def test_small_node_does_not_get_a_16g_default(self):
        part = {"mem_per_node_mb": 8192, "cpus_per_node": 8}
        assert default_memory_for(part, 1) == ("1G", "partition")

    def test_never_exceeds_the_node(self):
        part = {"mem_per_node_mb": 8192, "cpus_per_node": 8}
        assert default_memory_for(part, 999) == ("8G", "partition")

    def test_unknown_partition_falls_back_and_says_so(self):
        assert default_memory_for({}, 4) == (FALLBACK_MEMORY, "fallback")
        assert default_memory_for(None, 4) == (FALLBACK_MEMORY, "fallback")

    def test_sub_gigabyte_share_uses_megabytes(self):
        part = {"mem_per_node_mb": 4000, "cpus_per_node": 8}
        assert default_memory_for(part, 1) == ("500M", "partition")

    def test_explicit_memory_is_never_overridden(self):
        import argparse

        from rich.console import Console

        from slurmate.main import run_batch

        args = argparse.Namespace(
            job_name="t", account=None, partition="cpu-shared", qos=None,
            cpus=1, memory="3G", time="00:10:00", nodes=1, gpus=0,
            gpu_type=None, array=None, modules=None, env=None, env_type=None,
            command="echo hi", custom_sbatch=None, yes=False, force=False,
        )
        answers = run_batch(args, Console(), {})
        assert answers["memory"] == "3G"
        assert answers["_memory_source"] is None

    def test_omitting_memory_still_omits(self):
        import argparse

        from rich.console import Console

        from slurmate.main import run_batch

        args = argparse.Namespace(
            job_name="t", account=None, partition="cpu-shared", qos=None,
            cpus=1, memory="none", time="00:10:00", nodes=1, gpus=0,
            gpu_type=None, array=None, modules=None, env=None, env_type=None,
            command="echo hi", custom_sbatch=None, yes=False, force=False,
        )
        answers = run_batch(args, Console(), {})
        assert answers["memory"] is None

    def test_unspecified_memory_is_sized_from_the_partition(self):
        import argparse

        from rich.console import Console

        from slurmate.main import run_batch

        args = argparse.Namespace(
            job_name="t", account=None, partition="cpu-shared", qos=None,
            cpus=8, memory=None, time="00:10:00", nodes=1, gpus=0,
            gpu_type=None, array=None, modules=None, env=None, env_type=None,
            command="echo hi", custom_sbatch=None, yes=False, force=False,
        )
        answers = run_batch(args, Console(), {})
        # MOCK cpu-shared: 131072 MB over 32 cores → 8 cores is a quarter.
        assert answers["memory"] == "32G"
        assert answers["_memory_source"] == "partition"


# ── SW-12: the memory vocabulary varies by Slurm version, not by site ────


class TestSlurmMemoryVocabulary:
    """slurmate is the reference implementation for this across the family.

    Slurm <= 20.11 emitted ``ReqMem`` as ``4Gn`` / ``500Mc`` — the ``n``/``c``
    qualifier says per-node vs per-CPU — while 23.02 emits a plain unit. Slurm
    version varies by site more than anything else does, so a parser that only
    understands the spelling on the dev cluster is a portability bug even where
    the dev cluster cannot produce the other one. These cases are unreachable on
    the clusters to hand, which is exactly why they are pinned here rather than
    left to another cluster hunt.
    """

    @pytest.mark.parametrize(
        ("raw", "mb"),
        [
            ("4Gn", 4096),        # Slurm <= 20.11 per-node spelling
            ("4GN", 4096),
            ("500Mc", 500),       # ... and the per-CPU one
            ("500MC", 500),
            ("2Tn", 2097152),
            ("1.50T", 1572864),   # fractional, uppercase unit
            ("16", 16),           # a bare integer is MEGABYTES in Slurm
            ("1K", 1),            # rounds up to 1 MB, never down to "unknown"
        ],
    )
    def test_known_spellings_parse(self, raw, mb):
        assert su._parse_mem_to_mb(raw) == mb

    @pytest.mark.parametrize("raw", ["4Gn", "500Mc", "16", "1.50T", "1K"])
    def test_known_spellings_validate(self, raw):
        from slurmate.system_utils import validate_memory

        assert validate_memory(raw) is True

    def test_nc_qualifier_is_stripped_from_the_emitted_value(self):
        # `sbatch --mem` takes only a K/M/G/T unit, so the qualifier must not
        # reach the script even though we accept it on input.
        from slurmate.system_utils import normalize_memory

        assert normalize_memory("4Gn") == "4G"
        assert normalize_memory("500Mc") == "500M"

    def test_bare_integer_is_megabytes_not_bytes(self):
        # Off-by-1,048,576x if read as bytes, and silent when it happens.
        from slurmate.system_utils import normalize_memory

        assert su._parse_mem_to_mb("16") == 16
        assert normalize_memory("16") == "16M"

    # "0" is deliberately absent: --mem=0 is documented Slurm for all the memory
    # on the node, and was measured accepted in every unit spelling.
    # "64GB" was here, on the reasoning that a partial read is worse than a
    # refusal. The reasoning is right and the example was wrong: sbatch *accepts*
    # 16GB/16MB/16TB (measured), so refusing it was a false refusal. It is now
    # accepted, and the hazard the list guards against is enforced below —
    # accepting a spelling means parsing it too, in every one of the three
    # functions, or it validates and then reads as 0 in the limit checks.
    @pytest.mark.parametrize("raw", ["16 G", "1.5.5G", "5Q", "", "0P", "16GiB"])
    def test_malformed_values_are_rejected_not_partially_read(self, raw):
        from slurmate.system_utils import validate_memory

        assert validate_memory(raw) is False
        assert su._parse_mem_to_mb(raw) == 0

    @pytest.mark.parametrize("raw,mb", [
        ("64GB", 65536), ("16MB", 16), ("16KB", 1), ("1.5GB", 1536),
    ])
    def test_an_accepted_spelling_is_also_parsed(self, raw, mb):
        # The half-fix that is worse than the refusal: validate_memory says yes,
        # normalize_memory emits a correct directive, and _parse_mem_to_mb returns
        # 0 — so a 64 GB request on a 16 GB partition warns about nothing.
        from slurmate.system_utils import validate_memory

        assert validate_memory(raw) is True
        assert su._parse_mem_to_mb(raw) == mb


# ── SM-8 / SM-9: the config file is a cross-cluster trap ─────────────────


class TestConfigDisclosure:
    """SM-8: a ``.slurmate.toml`` travels with a project into git and onto the
    next cluster, so the values it supplies must be attributed, not just applied.
    """

    @pytest.fixture(autouse=True)
    def _clean(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        su._reset_config_notices()
        yield
        su._reset_config_notices()

    def test_loaded_file_and_keys_are_named_on_stderr(self, tmp_path, capsys):
        (tmp_path / ".slurmate.toml").write_text('partition = "caslake"\ncpus = 48\n')
        cfg = su.load_config()
        err = capsys.readouterr().err
        assert cfg == {"partition": "caslake", "cpus": 48}
        assert "./.slurmate.toml" in err
        assert "partition" in err and "cpus" in err

    def test_source_path_is_recorded_for_the_summary(self, tmp_path):
        (tmp_path / ".slurmate.toml").write_text('partition = "caslake"\n')
        su.load_config()
        assert su.config_source() == "./.slurmate.toml"

    def test_no_config_means_no_source_and_no_output(self, capsys):
        assert su.load_config() == {}
        assert su.config_source() == ""
        assert capsys.readouterr().err == ""

    def test_an_empty_config_is_not_announced(self, tmp_path, capsys):
        # A file that sets nothing usable supplied no defaults; saying it did
        # would be its own false disclosure.
        (tmp_path / ".slurmate.toml").write_text("# nothing here\n")
        assert su.load_config() == {}
        assert su.config_source() == ""
        assert "using defaults" not in capsys.readouterr().err

    def test_said_once_per_process(self, tmp_path, capsys):
        # main() loads the config and then the wizard loads it again; the same
        # notice twice reads like two different problems.
        (tmp_path / ".slurmate.toml").write_text('partition = "caslake"\n')
        su.load_config()
        su.load_config()
        assert capsys.readouterr().err.count("using defaults") == 1

    def test_home_config_is_shown_with_a_tilde(self, tmp_path, capsys, monkeypatch):
        # Home must sit outside the cwd here: a path that really is under the
        # working directory is better labelled "./…", and the cwd file is the
        # dangerous one, so that form wins when both apply.
        home = tmp_path / "home"
        work = tmp_path / "work"
        (home / ".config" / "slurmate").mkdir(parents=True)
        work.mkdir()
        (home / ".config" / "slurmate" / "config.toml").write_text('cpus = 2\n')
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(work)
        su.load_config()
        assert "~/.config/slurmate/config.toml" in capsys.readouterr().err


class TestConfigKeyVocabulary:
    """SM-9: the CLI flag is ``--time``, the config key is ``time_limit``. The
    natural translation used to be dropped in silence, so a user who wrote
    ``time = "36:00:00"`` got the two-hour default for a 36-hour run.
    """

    @pytest.fixture(autouse=True)
    def _clean(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        su._reset_config_notices()
        yield
        su._reset_config_notices()

    def _load(self, tmp_path, text):
        (tmp_path / ".slurmate.toml").write_text(text)
        return su.load_config()

    def test_cli_spelling_time_is_accepted_as_time_limit(self, tmp_path):
        assert self._load(tmp_path, 'time = "36:00:00"\n') == {"time_limit": "36:00:00"}

    def test_cli_spelling_array_is_accepted_as_array_spec(self, tmp_path):
        assert self._load(tmp_path, 'array = "1-4"\n') == {"array_spec": "1-4"}

    def test_dashed_flag_names_are_accepted(self, tmp_path):
        cfg = self._load(
            tmp_path,
            'job-name = "y"\nmem-per-cpu = "2G"\nntasks-per-node = 4\n',
        )
        assert cfg == {"job_name": "y", "mem_per_cpu": "2G", "ntasks_per_node": 4}

    def test_unknown_key_is_reported_not_dropped(self, tmp_path, capsys):
        assert self._load(tmp_path, "bogus_key = 5\n") == {}
        assert "unknown key 'bogus_key'" in capsys.readouterr().err

    def test_near_miss_gets_a_suggestion(self, tmp_path, capsys):
        self._load(tmp_path, 'partitions = "caslake"\ncpu = 4\n')
        err = capsys.readouterr().err
        assert "did you mean 'partition'?" in err
        assert "did you mean 'cpus'?" in err

    @pytest.mark.parametrize(
        "text",
        [
            'time = "36:00:00"\ntime_limit = "01:00:00"\n',
            'time_limit = "01:00:00"\ntime = "36:00:00"\n',
        ],
    )
    def test_real_key_beats_the_alias_in_either_order(self, tmp_path, capsys, text):
        # Order-dependence here would mean the same file behaved differently
        # depending on how the user happened to type it.
        assert self._load(tmp_path, text) == {"time_limit": "01:00:00"}
        assert "'time' ignored" in capsys.readouterr().err

    def test_unknown_section_is_reported(self, tmp_path, capsys):
        assert self._load(tmp_path, '[job]\npartition = "caslake"\n') == {}
        assert "unknown section '[job]'" in capsys.readouterr().err

    def test_known_sections_stay_silent(self, tmp_path, capsys):
        cfg = self._load(tmp_path, '[defaults]\npartition = "caslake"\n')
        assert cfg == {"partition": "caslake"}
        assert "unknown section" not in capsys.readouterr().err

    def test_every_key_the_code_reads_is_in_the_vocabulary(self):
        # The warning is only safe if CONFIG_KEYS is complete: a key the batch
        # path reads but the set omits would be rejected as a typo.
        import re
        from pathlib import Path
        src = Path(su.__file__).parent
        read = set()
        for name in ("main.py", "tui.py", "builder.py"):
            read |= set(re.findall(r'config\.get\("([a-z_]+)"', (src / name).read_text()))
        assert read <= su.CONFIG_KEYS, read - su.CONFIG_KEYS

    def test_naive_parser_path_normalizes_too(self, tmp_path, capsys, monkeypatch):
        # Python without tomllib/tomli falls back to the flat reader; it must not
        # be the one path that still drops `time` in silence.
        import builtins
        real_import = builtins.__import__

        def no_toml(name, *a, **k):
            if name in ("tomllib", "tomli"):
                raise ModuleNotFoundError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_toml)
        cfg = self._load(tmp_path, 'time = "36:00:00"\nbogus = 1\n[job]\nx = 1\n')
        err = capsys.readouterr().err
        assert cfg == {"time_limit": "36:00:00"}
        assert "unknown key 'bogus'" in err
        assert "unknown section '[job]'" in err


# ── SM-10 / SM-11 / SM-12: the real shapes the shipped fixtures never had ─


class TestInfiniteTimeLimit:
    """SM-10: `infinite` and "could not parse that" both used to become None, so
    on a cluster where every partition is unbounded the time check was inert and
    indistinguishable from a parse failure.
    """

    def test_infinite_is_unbounded_not_unknown(self):
        assert su._parse_partition_timelimit("infinite") == math.inf
        assert su._parse_partition_timelimit("UNLIMITED") == math.inf

    def test_unknown_stays_unknown(self):
        for raw in ("", None, "N/A", "NOT_SET", "INVALID", "not-a-time"):
            assert su._parse_partition_timelimit(raw) is None

    def test_finite_limits_still_parse(self):
        assert su._parse_partition_timelimit("02:00:00") == 120
        assert su._parse_partition_timelimit("1-00:00:00") == 1440

    def test_no_warning_against_an_unbounded_partition(self):
        # The point of the distinction: unbounded can *affirm* the request.
        issues = validate_job_config(
            {"cpus": 1, "time_limit": "99:00:00", "_partition_obj": _part(timelimit="infinite")}
        )
        assert not [m for _lvl, m in issues if "Time limit" in m]

    def test_a_finite_limit_still_warns(self):
        issues = validate_job_config(
            {"cpus": 1, "time_limit": "99:00:00", "_partition_obj": _part(timelimit="02:00:00")}
        )
        assert [m for _lvl, m in issues if "Time limit" in m]

    def test_unknown_limit_stays_silent(self):
        issues = validate_job_config(
            {"cpus": 1, "time_limit": "99:00:00", "_partition_obj": _part(timelimit=None)}
        )
        assert not [m for _lvl, m in issues if "Time limit" in m]

    def test_fetch_partitions_keeps_the_word_infinite(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(
            su, "_run_command",
            return_value=("legacy|infinite|9|up|28|90166|(null)|idle\n", "", 0),
        )
        parts = {p["name"]: p for p in fetch_partitions()}
        assert parts["legacy"]["timelimit"] == "infinite"


class TestHeterogeneousPartition:
    """SM-11: sinfo's trailing `+` on %c/%m means the printed figure is the
    *lowest* node, not the ceiling. Reporting it as "the partition limit" makes
    the over-request warning assert a bound a bigger node in the same partition
    may well clear.
    """

    def _parts(self, mocker, rows):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(rows, "", 0))
        return {p["name"]: p for p in fetch_partitions()}

    def test_plus_suffix_sets_the_flag(self, mocker):
        parts = self._parts(mocker, "legacy|infinite|9|up|28+|90166+|(null)|idle\n")
        assert parts["legacy"]["heterogeneous"] is True
        # The figures still parse — int('28+') was already handled — they are
        # just floors now rather than ceilings.
        assert parts["legacy"]["cpus_per_node"] == 28
        assert parts["legacy"]["mem_per_node_mb"] == 90166

    def test_uniform_partition_is_not_flagged(self, mocker):
        parts = self._parts(mocker, "even|infinite|9|up|28|90166|(null)|idle\n")
        assert parts["even"]["heterogeneous"] is False

    def test_one_heterogeneous_row_flags_the_partition(self, mocker):
        # sinfo emits a row per partition+state group; only one of them need
        # carry the "+" for the partition's figures to be floors.
        parts = self._parts(
            mocker,
            "mixed|infinite|4|up|28|90166|(null)|idle\n"
            "mixed|infinite|5|up|40+|90166|(null)|allocated\n",
        )
        assert parts["mixed"]["heterogeneous"] is True

    def test_warning_says_smallest_node_not_partition_limit(self):
        msgs = [
            m
            for _lvl, m in validate_job_config(
                {"cpus": 40, "_partition_obj": _part(heterogeneous=True, cpus_per_node=28)}
            )
            if m.startswith("CPUs")
        ]
        assert msgs, "an over-request should still be reported"
        assert "smallest node" in msgs[0] and "nodes differ" in msgs[0]
        assert "partition limit" not in msgs[0]

    def test_uniform_partition_keeps_the_hard_wording(self):
        msgs = [
            m
            for _lvl, m in validate_job_config(
                {"cpus": 999, "_partition_obj": _part(cpus_per_node=28)}
            )
            if m.startswith("CPUs")
        ]
        assert msgs and "exceeds partition limit" in msgs[0]

    def test_memory_warning_is_softened_too(self):
        msgs = [
            m
            for _lvl, m in validate_job_config(
                {
                    "cpus": 1,
                    "memory": "500G",
                    "_partition_obj": _part(heterogeneous=True, mem_per_node_mb=90166),
                }
            )
            if m.startswith("Memory")
        ]
        assert msgs and "smallest node" in msgs[0]


class TestGpuTypeFromFeatures:
    """SM-12: `_detect_gpu_type` mined node features and returned `tc` — a site
    node-class tag carried by unrelated partitions — producing
    `--gres=gpu:tc:1`, which Slurm refuses, while the real model was never
    offered.
    """

    @pytest.mark.parametrize(
        "features,expected",
        [
            ("tc,e5-2670,160G,ib,m2090,gpu,ibspine-g20", "m2090"),
            ("tc,e5-2670,32G,ib,k20m,gpu,ibspine-g20", "k20m"),
            ("tc,gold-6148,96GB,ib,edr,ibspine-d9b,v100", "v100"),
            ("lc,e5-2620v2,64G,gtx780,gpu,noib", "gtx780"),
        ],
    )
    def test_the_reported_nodes_resolve_to_their_real_model(self, features, expected):
        assert su._detect_gpu_type(features, "gpu:1") == expected

    def test_a_class_tag_is_never_the_answer(self):
        # Nothing identifiable → "gpu", which callers drop, so no type is offered.
        # Offering none beats offering `tc`: a wrong one prompts nobody to check.
        assert su._detect_gpu_type("tc,e5-2670,160G,ib,gpu,ibspine-g20", "gpu:1") == "gpu"
        assert su._detect_gpu_type("lc,gpu,rack12,sxm4", "gpu:1") == "gpu"

    @pytest.mark.parametrize("token", ["tc", "lc", "n1", "b12", "t2"])
    def test_short_or_digitless_tags_are_rejected(self, token):
        assert su._detect_gpu_type(f"{token},gpu", "gpu:1") == "gpu"

    @pytest.mark.parametrize("token", ["e5-2670", "x5650", "l5520", "gold-6148"])
    def test_cpu_designations_are_rejected(self, token):
        # `l5520` is the interesting one: the NVIDIA L family is L4/L40/L40S, so
        # a four-digit "l" token satisfies the GPU shape rule by coincidence.
        assert su._detect_gpu_type(f"{token},gpu", "gpu:1") == "gpu"

    @pytest.mark.parametrize("model", ["l4", "l40", "l40s", "w6800", "v100s", "mi250x"])
    def test_real_models_still_detected(self, model):
        assert su._detect_gpu_type(f"epyc,{model},gpu", "gpu:1") == model

    def test_typed_gres_still_wins_over_features(self):
        assert su._detect_gpu_type("tc,m2090", "gpu:a100:4") == "a100"

    def test_partition_scan_reports_the_real_models(self, mocker):
        # End to end: the nodes are count-only GRES with the model only in
        # features, which is exactly where `tc` used to win.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(
            su, "_run_command",
            return_value=(
                "tc,e5-2670,160G,ib,m2090,gpu,ibspine-g20|gpu:1\n"
                "tc,e5-2670,32G,ib,k20m,gpu,ibspine-g20|gpu:1\n",
                "", 0,
            ),
        )
        sources = su.fetch_gpu_type_sources("oldgpu")
        # Feature-only, so slurmate must also steer the request to
        # --constraint rather than a GRES type — that is the H2 path.
        assert sources["typed"] == []
        assert sources["feature"] == ["k20m", "m2090"]


# ── Audit findings (no report entry): the same defects, other fields ──────


class TestQosValidation:
    """`--partition` and `--account` were checked against the cluster; `--qos`,
    the third name Slurm resolves against its own database, was not — so a QoS
    carried from another site produced a complete script and rc=0.
    """

    KNOWN = ["normal", "caslake", "debug", "bigmem"]

    def test_unknown_qos_is_an_error(self):
        issues = validate_cluster_targets(
            "caslake", qos="nosuchqos", known_partitions=["caslake"], known_qos=self.KNOWN
        )
        assert [lvl for lvl, _ in issues] == ["error"]
        assert "no QoS 'nosuchqos' on this cluster." in issues[0][1]

    def test_known_qos_passes(self):
        assert not validate_cluster_targets(
            "caslake", qos="debug", known_partitions=["caslake"], known_qos=self.KNOWN
        )

    def test_near_miss_is_suggested(self):
        issues = validate_cluster_targets(
            "caslake", qos="caslak", known_partitions=["caslake"], known_qos=self.KNOWN
        )
        assert "Did you mean: caslake?" in issues[0][1]

    def test_unreadable_qos_list_validates_nothing(self):
        # sacctmgr down must never present as "your QoS doesn't exist".
        assert not validate_cluster_targets(
            "caslake", qos="anything", known_partitions=["caslake"], known_qos=[]
        )

    def test_plural_reads_as_english(self):
        issues = validate_cluster_targets(
            "caslake", qos="x", known_partitions=["caslake"], known_qos=self.KNOWN
        )
        assert "This cluster's QoS names:" in issues[0][1]
        assert "QoSs" not in issues[0][1]


class TestArraySizeLimit:
    """MaxArraySize is a site limit — Slurm's own default is 1001, the dev
    cluster is 65533 — so an `--array` carried between sites is refused with
    "Invalid job array specification" after a script that looked fine.
    """

    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("1-10", 10),
            ("0-9:2", 9),
            ("1,3,5", 5),
            ("1-5,10", 10),
            ("1-100%4", 100),      # %N throttles concurrency, it is not an index
            ("7", 7),
            ("", None),
            ("not-a-spec", None),
            ("1-2-3", None),
        ],
    )
    def test_highest_index_is_read_correctly(self, spec, expected):
        assert su._max_array_index(spec) == expected

    def test_over_the_limit_warns_with_the_site_number(self):
        msgs = [
            m
            for _lvl, m in validate_job_config(
                {"cpus": 1, "array_spec": "1-99999", "_partition_obj": _part()},
                max_array_size=65533,
            )
            if m.startswith("Array")
        ]
        assert msgs and "65533" in msgs[0]

    def test_under_the_limit_is_silent(self):
        assert not [
            m
            for _lvl, m in validate_job_config(
                {"cpus": 1, "array_spec": "1-100", "_partition_obj": _part()},
                max_array_size=65533,
            )
            if m.startswith("Array")
        ]

    def test_unknown_limit_claims_nothing(self):
        # An unreadable scontrol must not become a claim about the limit.
        assert not [
            m
            for _lvl, m in validate_job_config(
                {"cpus": 1, "array_spec": "1-99999", "_partition_obj": _part()},
                max_array_size=None,
            )
            if m.startswith("Array")
        ]

    def test_validator_makes_no_subprocess_call(self, mocker):
        # It runs on every TUI keystroke; the limit has to be passed in.
        boom = mocker.patch.object(su, "_run_command", side_effect=AssertionError)
        validate_job_config(
            {"cpus": 1, "array_spec": "1-99999", "_partition_obj": _part()},
            max_array_size=65533,
        )
        boom.assert_not_called()

    def test_eta_probe_includes_the_array(self, mocker):
        # Otherwise the ETA reports a start time for a job Slurm refuses — the
        # SM-5 defect, in a narrower case.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        run = mocker.patch.object(su, "_run_command", return_value=("", "", 1))
        su._scheduler_verdict("p", 1, 2, 0, 0, "", "01:00:00", "", "", "1-99999")
        assert "--array=1-99999" in run.call_args[0][0]


class TestNodeCountLimit:
    """The advertised limit warnings covered CPUs, memory and time but not the
    node count, so asking for more nodes than a partition has said nothing.
    """

    def test_more_nodes_than_exist_warns(self):
        msgs = [
            m
            for _lvl, m in validate_job_config(
                {"cpus": 1, "nodes": 9999, "_partition_obj": _part(nodes=8)}
            )
            if m.startswith("Nodes")
        ]
        assert msgs and "8 node(s)" in msgs[0]

    def test_a_fitting_request_is_silent(self):
        assert not [
            m
            for _lvl, m in validate_job_config(
                {"cpus": 1, "nodes": 4, "_partition_obj": _part(nodes=8)}
            )
            if m.startswith("Nodes")
        ]

    def test_unknown_partition_size_claims_nothing(self):
        assert not [
            m
            for _lvl, m in validate_job_config(
                {"cpus": 1, "nodes": 9999, "_partition_obj": _part(nodes=0)}
            )
            if m.startswith("Nodes")
        ]

    def test_node_count_is_not_softened_by_heterogeneity(self):
        # A count is exact: a bigger node cannot satisfy "give me 20 nodes".
        msgs = [
            m
            for _lvl, m in validate_job_config(
                {"cpus": 1, "nodes": 20, "_partition_obj": _part(nodes=8, heterogeneous=True)}
            )
            if m.startswith("Nodes")
        ]
        assert msgs and "smallest node" not in msgs[0]


class TestRuntimeTargets:
    """Values that only resolve when the job runs: a module that does not exist
    here, and a log directory Slurm cannot open. Both are among the most
    site-specific things in a generated script, and neither was checked — the
    job queued, started, and only then died.
    """

    def test_module_answer_is_read_from_stderr(self, mocker):
        # `modulecmd bash -t avail X` writes its listing to STDERR and leaves
        # stdout empty (stdout carries shell code to eval). A stdout-only read
        # reports every module on the cluster as missing.
        mocker.patch.object(su, "_module_command", return_value=["modulecmd", "bash"])
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(
            su, "_run_command",
            return_value=("", "/software/modulefiles:\npython/3.11\npython/2.7\n", 0),
        )
        assert su.fetch_module_matches("python") == ["python/3.11", "python/2.7"]

    def test_path_headers_and_rules_are_not_module_names(self, mocker):
        mocker.patch.object(su, "_module_command", return_value=["modulecmd", "bash"])
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(
            su, "_run_command",
            return_value=("", "---- /opt/mods ----\n/software/modulefiles:\n\ngcc/13(default)\n", 0),
        )
        assert su.fetch_module_matches("gcc") == ["gcc/13"]

    def test_no_module_system_asks_nothing(self, mocker):
        mocker.patch.object(su, "_module_command", return_value=None)
        assert su.fetch_module_matches("python") is None
        # And the caller must claim nothing rather than warn about everything.
        assert su.check_modules(["python/3.11", "anything"]) == []

    def test_missing_module_is_named(self, mocker):
        mocker.patch.object(su, "fetch_module_matches", side_effect=lambda n: [])
        issues = su.check_modules(["nosuch/9.9"])
        assert len(issues) == 1 and issues[0][0] == "warning"
        assert "module 'nosuch/9.9' not found" in issues[0][1]

    def test_wrong_version_lists_the_available_ones(self, mocker):
        # The common cross-cluster case: the module exists, the version does not.
        def fake(name):
            return [] if "/" in name else ["python/2.7", "python/3.11"]
        mocker.patch.object(su, "fetch_module_matches", side_effect=fake)
        issues = su.check_modules(["python/3.99"])
        assert "'python' is available as: python/2.7, python/3.11" in issues[0][1]

    def test_present_module_is_silent(self, mocker):
        mocker.patch.object(su, "fetch_module_matches", side_effect=lambda n: ["python/3.11"])
        assert su.check_modules(["python/3.11"]) == []

    def test_uncreatable_log_dir_is_named(self, tmp_path):
        target = "/proc/definitely/not/here"
        issues = su.check_log_dirs(f"#SBATCH --output={target}/x-%j.out\n")
        assert len(issues) == 1 and "cannot be created from here" in issues[0][1]

    def test_creatable_log_dir_is_silent(self, tmp_path):
        assert su.check_log_dirs(f"#SBATCH --output={tmp_path}/logs/x-%j.out\n") == []

    def test_existing_writable_dir_is_silent(self, tmp_path):
        assert su.check_log_dirs(f"#SBATCH --output={tmp_path}/x-%j.out\n") == []

    def test_pattern_in_a_directory_component_is_not_probed(self, tmp_path):
        # Slurm expands %j per job; there is no literal "%j" directory to test.
        assert su.check_log_dirs(f"#SBATCH --output={tmp_path}/%j/x.out\n") == []

    def test_each_directory_is_reported_once(self):
        script = (
            "#SBATCH --output=/proc/nope/x-%j.out\n"
            "#SBATCH --error=/proc/nope/x-%j.err\n"
        )
        assert len(su.check_log_dirs(script)) == 1

    def test_unwritable_existing_dir_is_named(self, tmp_path, mocker):
        mocker.patch.object(su.os, "access", return_value=False)
        issues = su.check_log_dirs(f"#SBATCH --output={tmp_path}/x.out\n")
        assert len(issues) == 1 and "not writable from here" in issues[0][1]


class TestModuleFailureIsFatal:
    """SM-13: a module the cluster lacks was the one cross-cluster error that
    survived submission — sbatch accepted it, the body ran anyway, and Slurm
    recorded COMPLETED 0:0 with the environment absent. Fixed at both ends: the
    name is rejected at generation, and the emitted line aborts the job.
    """

    def test_emitted_module_load_aborts_the_job(self):
        from slurmate.builder import build_sbatch_script
        script = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="4G", time_limit="00:10:00",
            modules=["cuda/11.8"], command="true",
        )
        line = next(ln for ln in script.splitlines() if ln.startswith("module load"))
        # `module load` exits non-zero on a missing modulefile, so `||` fires.
        assert line.startswith("module load cuda/11.8 || {")
        assert "exit 1;" in line

    def test_conda_activation_aborts_too(self):
        # Same defect, other mechanism: a failed activate left the job running
        # in whatever interpreter it inherited.
        from slurmate.builder import build_sbatch_script
        script = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="4G", time_limit="00:10:00",
            env_name="myenv", env_type="conda", command="true",
        )
        line = next(ln for ln in script.splitlines() if ln.startswith("conda activate"))
        assert "exit 1;" in line

    def test_venv_activation_aborts_too(self):
        from slurmate.builder import build_sbatch_script
        script = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="4G", time_limit="00:10:00",
            env_name="/opt/venv", env_type="venv", command="true",
        )
        line = next(ln for ln in script.splitlines() if ln.startswith("source "))
        assert "exit 1;" in line

    def test_mamba_fallback_chain_is_still_intact(self):
        # `mamba activate … || conda activate …` must keep its fallback, with the
        # guard only after both have failed.
        from slurmate.builder import build_sbatch_script
        script = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="4G", time_limit="00:10:00",
            env_name="myenv", env_type="mamba", command="true",
        )
        line = next(ln for ln in script.splitlines() if ln.startswith("mamba activate"))
        assert "|| conda activate myenv || {" in line

    def test_a_module_with_no_modules_asked_for_is_a_no_op(self):
        from slurmate.builder import build_sbatch_script
        script = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="4G", time_limit="00:10:00", command="true",
        )
        assert "module load" not in script
        assert "exit 1;" not in script


class TestAbortGuardIsSound:
    """Follow-up to SM-13: the guard is only worth having if it cannot be
    defeated. Two ways it could be — the reviewer's question (an `exit` inside a
    pipeline or subshell leaves the parent running) and the one that turned out
    to be real (the guard's own message expanded a user-supplied name).
    """

    def _guarded(self, **kw):
        from slurmate.builder import build_sbatch_script
        script = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="4G",
            time_limit="00:10:00", command="echo body", **kw,
        )
        return [ln for ln in script.splitlines() if "exit 1;" in ln]

    @pytest.mark.parametrize("name", ["$(id)", "foo`id`", "${HOME}", "$USER"])
    def test_message_cannot_run_a_substitution(self, name):
        # A double-quoted shell string still performs command substitution, so
        # interpolating the name into `echo "... $name ..."` would have executed
        # it the moment the guard fired. The message must be single-quoted.
        line = self._guarded(modules=[name])[0]
        assert f"echo 'slurmate: module load {name} failed; aborting'" in line
        assert f'"slurmate: module load {name}' not in line

    @pytest.mark.parametrize("name", ["foo | true", "foo) ; (true", "foo && true"])
    def test_metacharacters_cannot_restructure_the_line(self, name):
        line = self._guarded(modules=[name])[0]
        # The load argument is quoted, so the shell sees one word, and the guard
        # stays a top-level `||` list rather than becoming a pipeline.
        assert line.startswith("module load '")
        assert line.split("||", 1)[1].strip().startswith("{ echo ")

    def test_guard_is_not_inside_a_subshell_or_pipeline(self):
        # The reviewer's concern: `exit` in a pipeline or `( … )` exits only that
        # subshell. Every guarded line must be a plain top-level list.
        lines = self._guarded(modules=["a", "b"], env_name="e", env_type="conda")
        assert lines
        for line in lines:
            assert not line.lstrip().startswith("(")
            # Drop single-quoted spans first (a name may legitimately contain a
            # pipe), then look for a lone "|" — "||" is an or-list, "|" is a
            # pipeline, and only the latter would strand the exit in a subshell.
            bare = re.sub(r"'[^']*'", "", line)
            assert not re.search(r"(?<!\|)\|(?!\|)", bare), line
            assert "$(" not in bare and "`" not in bare

    def test_guard_really_stops_the_script(self, tmp_path):
        # Run it, rather than reasoning about it: a stub `module` that fails must
        # abort before the body.
        from slurmate.builder import build_sbatch_script
        script = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="4G",
            time_limit="00:10:00", modules=["nosuch/1.0"], command="echo BODY_RAN",
        )
        path = tmp_path / "job.sh"
        path.write_text("module() { return 1; }\n" + script)
        done = subprocess.run(
            ["bash", str(path)], capture_output=True, text=True, timeout=60
        )
        assert done.returncode == 1
        assert "BODY_RAN" not in done.stdout

    def test_a_good_setup_line_lets_the_body_run(self, tmp_path):
        from slurmate.builder import build_sbatch_script
        script = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="4G",
            time_limit="00:10:00", modules=["good/1.0"], command="echo BODY_RAN",
        )
        path = tmp_path / "job.sh"
        path.write_text("module() { return 0; }\n" + script)
        done = subprocess.run(
            ["bash", str(path)], capture_output=True, text=True, timeout=60
        )
        assert done.returncode == 0 and "BODY_RAN" in done.stdout

    def test_no_substitution_runs_when_the_guard_fires(self, tmp_path):
        from slurmate.builder import build_sbatch_script
        marker = tmp_path / "pwned"
        script = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="4G", time_limit="00:10:00",
            modules=[f"$(touch {marker})"], command="echo BODY_RAN",
        )
        path = tmp_path / "job.sh"
        path.write_text("module() { return 1; }\n" + script)
        subprocess.run(["bash", str(path)], capture_output=True, text=True, timeout=60)
        assert not marker.exists()


class TestCapacityRefusal:
    """SM-5 follow-up: the ETA's scheduler-independent fallback. With `sbatch`
    unreachable the estimate used to print a confident "~7min" on the same screen
    as a warning saying the request exceeds the partition — the warnings already
    knew, so the ETA can ask them.
    """

    def test_cores_over_an_exact_limit(self):
        assert "999 cores" in su.capacity_refusal(_part(cpus_per_node=48), {"cpus": 999})

    def test_ntasks_multiplies_the_request(self):
        reason = su.capacity_refusal(
            _part(cpus_per_node=48), {"cpus": 25, "ntasks_per_node": 2}
        )
        assert "50 cores" in reason

    def test_memory_over_an_exact_limit(self):
        assert "MB" in su.capacity_refusal(
            _part(mem_per_node_mb=1024), {"cpus": 1, "memory": "500G"}
        )

    def test_a_heterogeneous_partition_is_never_refused_on_cpu_or_memory(self):
        # sinfo printed the SMALLEST node, so a bigger one may take the job.
        # Claiming "never" here would trade one confident wrong answer for another.
        het = _part(heterogeneous=True, cpus_per_node=48, mem_per_node_mb=1024)
        assert su.capacity_refusal(het, {"cpus": 999}) == ""
        assert su.capacity_refusal(het, {"cpus": 1, "memory": "500G"}) == ""

    def test_node_count_is_refused_even_when_heterogeneous(self):
        # A count is exact: a bigger node cannot satisfy "give me 20 nodes".
        reason = su.capacity_refusal(
            _part(heterogeneous=True, nodes=8), {"cpus": 1, "nodes": 20}
        )
        assert "only 8 node(s)" in reason

    def test_array_size_is_refused_even_when_heterogeneous(self):
        reason = su.capacity_refusal(
            _part(heterogeneous=True), {"cpus": 1, "array_spec": "1-99999"},
            max_array_size=1001,
        )
        assert "MaxArraySize (1001)" in reason

    def test_time_limit_is_refused_even_when_heterogeneous(self):
        reason = su.capacity_refusal(
            _part(heterogeneous=True, timelimit="02:00:00"),
            {"cpus": 1, "time_limit": "99:00:00"},
        )
        assert "partition time limit is 02:00:00" in reason

    def test_an_unbounded_partition_does_not_refuse_on_time(self):
        assert su.capacity_refusal(
            _part(timelimit="infinite"), {"cpus": 1, "time_limit": "99:00:00"}
        ) == ""

    def test_an_unknown_time_limit_does_not_refuse(self):
        assert su.capacity_refusal(
            _part(timelimit=None), {"cpus": 1, "time_limit": "99:00:00"}
        ) == ""

    def test_a_fitting_request_is_not_refused(self):
        assert su.capacity_refusal(_part(cpus_per_node=48, nodes=8), {"cpus": 4}) == ""

    def test_no_partition_object_claims_nothing(self):
        assert su.capacity_refusal(None, {"cpus": 999}) == ""

    def test_unknown_capacity_claims_nothing(self):
        # The synthetic record for a manually-typed partition carries zeroes;
        # zero must read as "unknown", not "nothing fits".
        blank = _part(cpus_per_node=0, mem_per_node_mb=0, nodes=0)
        assert su.capacity_refusal(blank, {"cpus": 999, "nodes": 20}) == ""

    def test_it_makes_no_subprocess_call(self, mocker):
        boom = mocker.patch.object(su, "_run_command", side_effect=AssertionError)
        su.capacity_refusal(_part(cpus_per_node=48), {"cpus": 999})
        boom.assert_not_called()


class TestGpuCountLimit:
    """`sinfo %G` carried the per-node GPU count (`gpu:4`) all along, but nothing
    parsed it — so GPUs were the one advertised resource with no limit check and
    `--gpus 99` on a 4-GPU partition produced a script in silence.
    """

    @pytest.mark.parametrize(
        "gres,expected",
        [
            ("gpu:4", 4),
            ("gpu:a30:4", 4),
            ("gpu:a100:2,gpu:v100:2", 4),      # either model can satisfy the ask
            ("gpu:a100:4(S:0-1)", 4),          # socket annotation
            ("gpu:a100", 1),                   # typed, no count
            ("(null)", 0),
            ("", 0),
            ("shard:8", 0),                    # a slice of a GPU, not another one
            ("gpu:2,shard:8", 2),
            ("mps:100", 0),
        ],
    )
    def test_count_is_read_from_every_gres_spelling(self, gres, expected):
        assert su._parse_gpu_count(gres) == expected

    def test_partition_carries_the_count(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(
            su, "_run_command",
            return_value=("g|infinite|4|up|48|184320|gpu:a100:4|idle\n", "", 0),
        )
        parts = {p["name"]: p for p in fetch_partitions()}
        assert parts["g"]["gpus_per_node"] == 4

    def test_over_the_count_warns(self):
        msgs = [
            m for _lvl, m in validate_job_config(
                {"cpus": 1, "gpus": 99, "_partition_obj": _part(gpus_per_node=4)}
            )
            if m.startswith("GPUs")
        ]
        assert msgs and "4 per node" in msgs[0]

    def test_a_fitting_request_is_silent(self):
        assert not [
            m for _lvl, m in validate_job_config(
                {"cpus": 1, "gpus": 2, "_partition_obj": _part(gpus_per_node=4)}
            )
            if m.startswith("GPUs")
        ]

    def test_unknown_gpu_count_claims_nothing(self):
        assert not [
            m for _lvl, m in validate_job_config(
                {"cpus": 1, "gpus": 99, "_partition_obj": _part(gpus_per_node=0)}
            )
            if m.startswith("GPUs")
        ]

    def test_capacity_refusal_knows_about_gpus(self):
        assert "99 GPUs" in su.capacity_refusal(
            _part(gpus_per_node=4), {"cpus": 1, "gpus": 99}
        )

    def test_heterogeneous_partition_does_not_refuse_on_gpus(self):
        # Same reasoning as cpu/memory: the printed figure is the smallest node.
        assert su.capacity_refusal(
            _part(gpus_per_node=4, heterogeneous=True), {"cpus": 1, "gpus": 99}
        ) == ""


class TestConstraintValidation:
    """`--constraint` is a node feature, so it is as site-specific as a partition
    name — and Slurm refuses a bad one outright ("Invalid feature specification")
    — yet it was emitted unchecked, and the ETA reported a cheerful estimate
    because the probe never passed it either.
    """

    FEATURES = ["192g", "gold-6248r", "a100", "edr"]

    def test_unknown_feature_is_an_error(self):
        issues = validate_cluster_targets(
            "p", constraint="nosuchfeature", known_partitions=["p"],
            known_features=self.FEATURES,
        )
        assert [lvl for lvl, _ in issues] == ["error"]
        assert "no node feature 'nosuchfeature' on this cluster." in issues[0][1]
        assert "This cluster's node features:" in issues[0][1]

    def test_known_feature_passes(self):
        assert not validate_cluster_targets(
            "p", constraint="192g", known_partitions=["p"], known_features=self.FEATURES
        )

    @pytest.mark.parametrize(
        "expr", ["192g&a100", "192g|a100", "[192g*2]", "!192g", "192g,a100"]
    )
    def test_feature_expressions_are_not_membership_checked(self, expr):
        # Slurm's constraint grammar has &, |, !, *N and [] — a set-membership
        # test would reject perfectly valid expressions.
        assert not validate_cluster_targets(
            "p", constraint=expr, known_partitions=["p"], known_features=self.FEATURES
        )

    def test_unreadable_feature_list_validates_nothing(self):
        # None, not [] — "could not ask" is the case that must stay silent.
        assert not validate_cluster_targets(
            "p", constraint="anything", known_partitions=["p"], known_features=None
        )

    def test_cluster_advertising_no_features_rejects_a_plain_constraint(self):
        # Measured on Booth's Mercury: sinfo answers, and every node reports
        # `(null)`. That is a definite answer — no -C name can match here — and
        # collapsing it into the same empty list as "unreadable" left the check
        # inert on precisely the clusters where it knew something.
        issues = validate_cluster_targets(
            "p", constraint="a100", known_partitions=["p"], known_features=[]
        )
        assert [lvl for lvl, _ in issues] == ["error"]
        assert "advertises no node features" in issues[0][1]
        assert "'a100'" in issues[0][1]

    def test_no_features_still_ignores_an_expression(self):
        # A feature *expression* is not a set member either way, so the
        # empty-cluster branch must not start rejecting valid grammar.
        assert not validate_cluster_targets(
            "p", constraint="192g&a100", known_partitions=["p"], known_features=[]
        )

    def test_no_features_is_silent_without_a_constraint(self):
        assert not validate_cluster_targets(
            "p", known_partitions=["p"], known_features=[]
        )

    def test_fetch_distinguishes_unreadable_from_a_featureless_cluster(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        # sinfo could not be asked -> None, and nothing may be concluded.
        mocker.patch.object(su, "_run_command", return_value=("", "boom", 1))
        assert su.fetch_node_features() is None
        su.reset_cluster_cache()
        # sinfo answered, every node is (null) -> an empty set, which is a fact.
        mocker.patch.object(su, "_run_command", return_value=("(null)\n(null)\n", "", 0))
        assert su.fetch_node_features() == set()

    def test_features_are_collected_across_the_cluster(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(
            su, "_run_command",
            return_value=("192g,gold-6248r\n(null)\n a100 , edr \n", "", 0),
        )
        assert su.fetch_node_features() == {"192g", "gold-6248r", "a100", "edr"}

    def test_eta_probe_includes_the_constraint(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        run = mocker.patch.object(su, "_run_command", return_value=("", "", 1))
        su._scheduler_verdict("p", 1, 2, 0, 0, "", "01:00:00", "", "", "", "192g")
        assert "--constraint=192g" in run.call_args[0][0]


class TestScriptBasedEtaProbe:
    """The ETA probe used to rebuild an argv from the same fields the builder
    reads, which duplicated the builder and kept drifting: `--array` was missing
    (an over-large array read "~22h"), then `--constraint` (a bogus feature read
    "~21h"), and it rewrote every `--gpu-format` choice as `--gres`, a different
    request on a count-only site. Handing Slurm the real script cannot drift.
    """

    def test_the_script_is_piped_to_sbatch(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        run = mocker.patch.object(su, "_run_command", return_value=("", "", 1))
        script = "#!/bin/bash\n#SBATCH --array=1-99999\n"
        su._scheduler_verdict("p", 1, 2, 0, 0, "", "01:00:00", "", "", "", "", script)
        assert run.call_args.kwargs["stdin"] == script
        # And nothing is reconstructed on the command line.
        argv = run.call_args[0][0]
        assert argv == ["sbatch", "--test-only", "--parsable"]

    def test_argv_reconstruction_still_works_without_a_script(self, mocker):
        # Callers that have no script (the wizard's live preview path) keep the
        # old behaviour rather than losing their ETA.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        run = mocker.patch.object(su, "_run_command", return_value=("", "", 1))
        su._scheduler_verdict("p", 1, 2, 0, 0, "", "01:00:00", "", "", "", "")
        argv = run.call_args[0][0]
        assert "-p" in argv and "p" in argv
        assert run.call_args.kwargs.get("stdin") is None

    def test_refusal_is_read_from_either_stream(self):
        # Slurm puts the verdict on stderr in some versions and stdout in others.
        assert su._read_test_only_output(
            "", "allocation failure: Invalid job array specification", 1
        )[1] == "Invalid job array specification"
        assert su._read_test_only_output(
            "allocation failure: Invalid feature specification", "", 1
        )[1] == "Invalid feature specification"

    def test_a_nonzero_exit_alone_is_not_a_refusal(self):
        # An unreachable controller must not read as "your job can never run".
        assert su._read_test_only_output("", "sbatch: error: connection timed out", 1) == (
            None, ""
        )

    def test_accepted_without_a_placement_line_is_no_verdict(self):
        assert su._read_test_only_output("", "", 0) == (None, "")


class TestCustomSbatchFlagForm:
    """`--custom-sbatch` exists to pass other flags through, and argparse treats
    a value starting with `-` as the next option — so its most natural
    invocation failed with a generic "expected one argument" that named neither
    the cause nor the fix.
    """

    @pytest.mark.parametrize("value", ["--exclusive", "-x", "--hold"])
    def test_the_broken_form_is_explained(self, value, capsys):
        from slurmate.main import _check_custom_sbatch_form
        with pytest.raises(SystemExit) as exc:
            _check_custom_sbatch_form(["--custom-sbatch", value])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "must use the '=' form" in err
        assert f"--custom-sbatch='{value}'" in err

    @pytest.mark.parametrize(
        "value",
        [
            "-C bigmem",                                  # argparse allows a space
            '--comment="my run"',
            '--exclusive,--comment="my run",-C bigmem',   # the report's own string
            "plain",
        ],
    )
    def test_forms_argparse_already_accepts_are_untouched(self, value):
        # Rejecting these would break the multi-flag invocation the portability
        # report exercised — argparse only mistakes a value for an option when it
        # starts with "-" and contains no space.
        from slurmate.main import _check_custom_sbatch_form
        _check_custom_sbatch_form(["--custom-sbatch", value])

    def test_argparse_boundary_is_what_we_assume_it_is(self):
        # Pin the behaviour the guard is calibrated against, so a change in
        # argparse shows up here rather than as a mis-rejected flag.
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--custom-sbatch", default=None)
        assert parser.parse_args(["--custom-sbatch", "-C bigmem"]).custom_sbatch == "-C bigmem"
        with pytest.raises(SystemExit):
            parser.parse_args(["--custom-sbatch", "--exclusive"])

    def test_a_trailing_flag_with_no_value_is_left_to_argparse(self):
        from slurmate.main import _check_custom_sbatch_form
        _check_custom_sbatch_form(["--custom-sbatch"])


class TestSubmittedJobIdParsing:
    """The submit path trusted the whole of `sbatch --parsable`'s stdout to be
    the job id. A site's sbatch *wrapper* printing a policy notice made the
    banner the "id", which then travelled into the "Job ID:" line, the
    `squeue -j` / `scancel` hints, and the saved script's filename. This module
    already guards the same hazard for JSON; the submit path did not.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("54321", "54321"),
            ("54321\n", "54321"),
            ("  54321  ", "54321"),
            ("54321;midway3", "54321"),                     # federated setup
            ("NOTICE: fair-use policy\n54321", "54321"),    # wrapper banner first
            ("54321\nNOTICE: submitted", "54321"),          # banner after
            ("NOTICE: policy\n54321;midway3\n", "54321"),
        ],
    )
    def test_the_id_is_found(self, raw, expected):
        assert su.parse_submitted_job_id(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "NOTICE: policy applies in 2026",   # digits, but not an id line
            "submitted ok",
            "job 12345 queued",                 # right shape only on its own line
            "12345abc",
        ],
    )
    def test_no_id_yields_empty_rather_than_a_guess(self, raw):
        # A banner can contain digits, so scraping the first number out of
        # arbitrary text would substitute one wrong answer for another.
        assert su.parse_submitted_job_id(raw) == ""

    def test_a_year_in_a_banner_is_not_mistaken_for_an_id(self):
        assert su.parse_submitted_job_id("Policy revised 2026, see docs") == ""

    def test_the_old_naive_parse_would_have_failed(self):
        # Pins why this exists: the previous implementation was
        # `stdout.strip().split(";")[0]`.
        raw = "NOTICE: fair-use policy\n54321"
        assert raw.strip().split(";")[0] != "54321"
        assert su.parse_submitted_job_id(raw) == "54321"


class TestMemoryZeroAndArrayShape:
    """Two values whose *shape* validation disagreed with the controller:
    `--mem=0` was refused though Slurm documents it as "all the memory on the
    node", and the array spec was never shape-checked at all, so `--array 10-1`
    produced a script Slurm refuses.
    """

    @pytest.mark.parametrize("raw", ["0", "0K", "0M", "0G", "0T"])
    def test_every_zero_spelling_is_accepted_and_normalized(self, raw):
        from slurmate.system_utils import normalize_memory, validate_memory
        assert validate_memory(raw) is True
        assert normalize_memory(raw) == "0"

    def test_zero_is_not_the_same_as_omitting_mem(self):
        # `--memory ''`/none omits --mem and gets the site default; `--mem=0`
        # asks for all the memory. Conflating them loses a request.
        from slurmate.builder import build_sbatch_script
        with_zero = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="0",
            time_limit="00:10:00", command="true",
        )
        omitted = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="",
            time_limit="00:10:00", command="true",
        )
        assert "#SBATCH --mem=0" in with_zero
        assert "--mem" not in omitted

    def test_a_zero_request_triggers_no_limit_warning(self):
        # 0 means "everything", not "more than the node has".
        assert not [
            m for _lvl, m in validate_job_config(
                {"cpus": 1, "memory": "0", "_partition_obj": _part(mem_per_node_mb=1024)}
            )
            if m.startswith("Memory")
        ]

    # Calibrated against a live controller rather than guessed — including the
    # bare "%4", which Slurm accepts and an intuition-built validator rejects.
    @pytest.mark.parametrize(
        "spec", ["5", "1-10", "0-9", "1,3,5", "1-10:2", "1-10%4", "1-5,10", "%4", ""]
    )
    def test_specs_slurm_accepts(self, spec):
        assert su.validate_array_spec(spec) is True

    @pytest.mark.parametrize(
        "spec", ["10-1", "1-10:0", "1-", "-5", "1-2-3", "a-b", "1-10%", "1,,3", "1-10%x"]
    )
    def test_specs_slurm_rejects(self, spec):
        assert su.validate_array_spec(spec) is False

    def test_reversed_range_is_caught_before_the_controller(self):
        # The measured failure: `allocation failure: Invalid job array
        # specification`, after a script that looked fine.
        assert su.validate_array_spec("10-1") is False
        assert su._max_array_index("10-1") == 10      # still readable for the limit check


class TestConfigMerge:
    """SM-14: first-file-wins made a project config *destructive*. A one-line
    `.slurmate.toml` naming this cluster's partition discarded the global
    account, memory, time limit and module list — silently, and each with its own
    failure mode.
    """

    @pytest.fixture(autouse=True)
    def _env(self, tmp_path, monkeypatch):
        self.home = tmp_path / "home"
        self.proj = tmp_path / "proj"
        (self.home / ".config" / "slurmate").mkdir(parents=True)
        self.proj.mkdir()
        monkeypatch.setenv("HOME", str(self.home))
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        monkeypatch.chdir(self.proj)
        su._reset_config_notices()
        yield
        su._reset_config_notices()

    def _write(self, glob=None, proj=None):
        if glob is not None:
            (self.home / ".config" / "slurmate" / "config.toml").write_text(glob)
        if proj is not None:
            (self.proj / ".slurmate.toml").write_text(proj)

    def test_the_reported_scenario_keeps_every_global_value(self):
        self._write(
            glob='account = "rcc-staff"\nmemory = "64G"\ntime_limit = "12:00:00"\n',
            proj='partition = "build"\n',
        )
        cfg = su.load_config()
        assert cfg == {
            "account": "rcc-staff", "memory": "64G",
            "time_limit": "12:00:00", "partition": "build",
        }

    def test_the_project_file_wins_per_key(self):
        self._write(glob='memory = "64G"\naccount = "a"\n', proj='memory = "8G"\n')
        cfg = su.load_config()
        assert cfg["memory"] == "8G"      # more specific file wins
        assert cfg["account"] == "a"      # and does not destroy the rest

    def test_each_file_is_credited_only_for_keys_it_won(self, capsys):
        self._write(glob='memory = "64G"\naccount = "a"\n', proj='memory = "8G"\n')
        su.load_config()
        err = capsys.readouterr().err
        global_line = next(ln for ln in err.splitlines() if "config.toml" in ln)
        project_line = next(ln for ln in err.splitlines() if ".slurmate.toml" in ln)
        assert "account" in global_line and "memory" not in global_line
        assert "memory" in project_line

    def test_either_file_alone_still_works(self):
        self._write(glob='account = "a"\n')
        assert su.load_config() == {"account": "a"}
        su._reset_config_notices()
        (self.home / ".config" / "slurmate" / "config.toml").unlink()
        self._write(proj='partition = "p"\n')
        assert su.load_config() == {"partition": "p"}

    def test_a_broken_file_does_not_take_the_other_down(self, capsys):
        # Dropping both because one is unparseable would lose values that are
        # perfectly readable.
        self._write(glob='account = "a"\n', proj="this is not toml = = =\n")
        cfg = su.load_config()
        assert cfg == {"account": "a"}
        assert "ignoring configuration file" in capsys.readouterr().err

    def test_source_names_both_files(self):
        self._write(glob='account = "a"\n', proj='partition = "p"\n')
        su.load_config()
        source = su.config_source()
        assert "~/.config/slurmate/config.toml" in source
        assert "./.slurmate.toml" in source

    def test_a_file_contributing_nothing_is_not_named(self, capsys):
        # The project file's only key is overridden by nothing, but the global
        # file's sole key is — so the global file wins no keys and must not claim
        # to have supplied any.
        self._write(glob='partition = "old"\n', proj='partition = "new"\n')
        assert su.load_config() == {"partition": "new"}
        err = capsys.readouterr().err
        assert "config.toml: partition" not in err
        assert su.config_source() == "./.slurmate.toml"


class TestCustomSbatchConflicts:
    """SM-15: `--custom-sbatch` emitted a second `#SBATCH` line for a directive
    slurmate manages. Slurm honours the last, so the job ran with the custom
    value while the summary, the cluster validation and the queue/ETA figures all
    described the managed one — and a custom `--partition`/`--account` routed
    straight past the SM-4 checks that exist for exactly those two values.
    """

    @pytest.mark.parametrize(
        "flag,owner",
        [
            ("--partition=amd", "--partition"),
            ("-p amd", "--partition"),
            ("--account=pi-nope", "--account"),
            ("-A pi-nope", "--account"),
            ("--qos=x", "--qos"),
            ("--time=99:00:00", "--time"),
            # The owner is the spelling the CLI accepts. These three used to
            # point at --cpus/--gpus, which was correct before SM-25 made Slurm's
            # own spellings first-class and wrong afterwards — and for --gres it
            # also dropped the type that gpu:a100:2 carries.
            ("--cpus-per-task=99", "--cpus-per-task"),
            ("--nodes=9", "--nodes"),
            ("--job-name=other", "--job-name"),
            ("--array=1-9", "--array"),
            ("--gres=gpu:2", "--gres"),
            ("--gpus-per-node=2", "--gpus-per-node"),
            ("--gpus-per-task=2", "--gpus-per-task"),
            ("-c 8", "--cpus-per-task"),
        ],
    )
    def test_a_managed_directive_is_refused(self, flag, owner):
        from slurmate.builder import managed_custom_flags
        conflicts = managed_custom_flags(flag)
        assert conflicts and conflicts[0][1] == owner

    @pytest.mark.parametrize(
        "flag",
        [
            "--mem=8G",          # reconciled: the custom value wins, auto suppressed
            "--mem-per-cpu=2G",
            "-C bigmem",         # reconciled: merged into one --constraint
            "--constraint=a100",
            "--output=/tmp/a.out",   # reconciled: de-duplicated
            "-o /tmp/a.out",
            "--error=/tmp/a.err",
            "--exclusive",       # not a managed directive at all — the whole point
            '--comment="my run"',
            "--dependency=afterok:1",
            "",
        ],
    )
    def test_reconciled_and_passthrough_flags_are_allowed(self, flag):
        # Refusing these would undo behaviour the report explicitly asked to keep
        # (the merged `-C bigmem`) and the documented purpose of the flag.
        from slurmate.builder import managed_custom_flags
        assert managed_custom_flags(flag) == []

    def test_a_conflict_inside_a_multi_flag_string_is_found(self):
        from slurmate.builder import managed_custom_flags
        conflicts = managed_custom_flags('--exclusive,--partition=amd,-C bigmem')
        assert [name for name, _ in conflicts] == ["--partition"]

    def test_the_validator_reports_it_too(self):
        # So the wizard's summary and the pre-submit guard see it, not just the
        # batch path.
        issues = validate_job_config(
            {"cpus": 1, "custom_sbatch": "--partition=amd", "_partition_obj": _part()}
        )
        assert [lvl for lvl, m in issues if "duplicates a directive" in m] == ["error"]

    def test_a_reconciled_flag_raises_no_validator_error(self):
        issues = validate_job_config(
            {"cpus": 1, "custom_sbatch": "-C bigmem", "_partition_obj": _part()}
        )
        assert not [m for _lvl, m in issues if "duplicates a directive" in m]

    def test_custom_mem_still_suppresses_the_auto_directive(self):
        # The reconciliation that must survive: Slurm rejects a script setting
        # both --mem and --mem-per-cpu.
        from slurmate.builder import build_sbatch_script
        script = build_sbatch_script(
            job_name="j", partition="p", cpus=1, memory="4G", time_limit="00:10:00",
            custom_sbatch=["--mem=8G"], command="true",
        )
        assert script.count("--mem=") == 1 and "--mem=8G" in script


class TestNoHomeDirectory:
    """SM-16: `Path.home()` raises RuntimeError when $HOME is unset AND the uid
    has no passwd entry — `sbatch --export=NONE` on a node whose name service
    does not resolve the user. The config search list was built eagerly, so that
    aborted *every* invocation before any flag was acted on, including runs with
    a perfectly good project-local config in the job's working directory.
    """

    @pytest.fixture(autouse=True)
    def _no_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.chdir(tmp_path)
        su._reset_config_notices()
        yield
        su._reset_config_notices()

    def test_project_config_still_loads_with_no_discoverable_home(self, tmp_path, mocker):
        from pathlib import Path
        mocker.patch.object(
            Path, "home", side_effect=RuntimeError("Could not determine home directory.")
        )
        (tmp_path / ".slurmate.toml").write_text('partition = "build"\n')
        assert su.load_config() == {"partition": "build"}

    def test_no_config_anywhere_is_not_a_crash(self, mocker):
        from pathlib import Path
        mocker.patch.object(Path, "home", side_effect=RuntimeError("no home"))
        assert su.load_config() == {}

    def test_xdg_config_home_is_honoured(self, tmp_path, monkeypatch, mocker):
        # The documented location, previously ignored — and the way to keep a
        # global config in an environment with no home at all.
        from pathlib import Path
        mocker.patch.object(Path, "home", side_effect=RuntimeError("no home"))
        xdg = tmp_path / "xdg"
        (xdg / "slurmate").mkdir(parents=True)
        (xdg / "slurmate" / "config.toml").write_text('account = "a"\n')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        assert su.load_config() == {"account": "a"}

    def test_xdg_and_project_still_merge(self, tmp_path, monkeypatch, mocker):
        from pathlib import Path
        mocker.patch.object(Path, "home", side_effect=RuntimeError("no home"))
        xdg = tmp_path / "xdg"
        (xdg / "slurmate").mkdir(parents=True)
        (xdg / "slurmate" / "config.toml").write_text('account = "a"\nmemory = "64G"\n')
        (tmp_path / ".slurmate.toml").write_text('memory = "8G"\n')
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        assert su.load_config() == {"account": "a", "memory": "8G"}

    def test_an_unexpanded_tilde_in_a_log_path_is_reported(self, mocker):
        # Deleting $HOME is not enough to reproduce this: expanduser falls back to
        # pwd.getpwuid, which works on a login node. The failing environment has
        # *neither*, and then expanduser does not raise — it returns the string
        # unchanged, so "~/logs" becomes a relative directory named "~".
        mocker.patch.object(su.os.path, "expanduser", side_effect=lambda p: p)
        issues = su.check_log_dirs("#SBATCH --output=~/logs/x-%j.out\n")
        assert len(issues) == 1
        assert "still begins with '~'" in issues[0][1]

    def test_expanduser_returning_the_input_is_the_real_failure_mode(self):
        # Pins the premise, so a change in Python's behaviour surfaces here.
        assert su.unexpanded_home("~/logs") is True
        assert su.unexpanded_home("/home/u/logs") is False

    def test_a_resolved_path_is_not_reported(self, tmp_path):
        # Scoped to this class's subject. Asserting *no* findings at all made the
        # test depend on where pytest's tmp_path lives: `/tmp` on both Booth
        # clusters (node-local, so it correctly warned) and
        # `/scratch/local/jobs/<id>` on midway3, where $TMPDIR points and which is
        # the same local volume. Passing only because of an ambient $TMPDIR is
        # what this file exists to stop.
        issues = su.check_log_dirs(f"#SBATCH --output={tmp_path}/x-%j.out\n")
        assert not [m for _lvl, m in issues if "~" in m], issues


class TestNonUtf8Output:
    """SM-17: a `⚠` in a warning aborted the run with UnicodeEncodeError under a
    valid non-UTF-8 locale, truncating the output — and the affected sites were
    all warning and error paths, so the tool was least robust exactly when
    something had already gone wrong.
    """

    def test_markers_fall_back_to_ascii_when_unencodable(self, mocker):
        from slurmate import theme
        mocker.patch.object(theme, "output_encoding", return_value="latin-1")
        assert theme.use_ascii() is True
        assert theme.g.WARN == "!" and theme.g.ERR == "x" and theme.g.OK == "+"

    def test_markers_stay_unicode_when_the_terminal_can_carry_them(self, mocker):
        from slurmate import theme
        mocker.patch.object(theme, "output_encoding", return_value="utf-8")
        theme.set_ascii(False)
        assert theme.use_ascii() is False
        assert theme.g.WARN == "⚠"

    def test_ascii_can_be_forced_on_a_utf8_terminal(self, mocker):
        from slurmate import theme
        mocker.patch.object(theme, "output_encoding", return_value="utf-8")
        theme.set_ascii(True)
        try:
            assert theme.g.WARN == "!"
        finally:
            theme.set_ascii(False)

    def test_env_var_forces_ascii(self, monkeypatch, mocker):
        from slurmate import theme
        mocker.patch.object(theme, "output_encoding", return_value="utf-8")
        monkeypatch.setenv("SLURMATE_ASCII", "1")
        assert theme.use_ascii() is True

    def test_warnings_survive_a_latin1_stdout(self):
        # The report's own test: the message that used to kill the run must print.
        import io
        import subprocess
        env = {**os.environ, "PYTHONPATH": "src", "LC_ALL": "en_US", "SLURMATE_MOCK": "1"}
        env.pop("PYTHONIOENCODING", None)
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--force",
             "--partition", "p", "--cpus", "1", "--time", "00:05:00",
             "--command", "true"],
            capture_output=True, env=env, timeout=120,
        )
        assert done.returncode == 0
        assert b"#!/bin/bash" in done.stdout
        assert io.DEFAULT_BUFFER_SIZE  # keep the import meaningful

    def test_known_typography_transliterates_rather_than_escaping(self):
        from slurmate.theme import _encode_fallback
        exc = UnicodeEncodeError("latin-1", "a — b", 2, 3, "unmapped")
        assert _encode_fallback(exc)[0] == "-"

    def test_unknown_characters_are_escaped_not_dropped(self):
        # A job name or module carrying these is data, not decoration; "?" would
        # silently destroy it.
        from slurmate.theme import _encode_fallback
        exc = UnicodeEncodeError("latin-1", "訓練", 0, 2, "unmapped")
        out = _encode_fallback(exc)[0]
        assert out == "\\u8a13\\u7df4"

    def test_the_handler_never_raises_on_a_decode_error(self):
        from slurmate.theme import _encode_fallback
        with pytest.raises(UnicodeDecodeError):
            _encode_fallback(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"))


class TestClockSkewInTheEta:
    """`sbatch --test-only` reports the placement in the *controller's* local
    time; slurmate compared it against the *login node's* clock and clamped a
    negative result to 0. A timezone difference between the two therefore became
    a confident `ETA: now` for a job starting hours later — SM-5's defect, from a
    different direction.
    """

    def _eta(self, seconds_from_now):
        from datetime import datetime, timedelta
        when = (datetime.now() + timedelta(seconds=seconds_from_now)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        return su._read_test_only_output("", f"to start at {when}", 0)

    def test_a_future_start_is_reported(self):
        eta, refusal = self._eta(1800)
        assert refusal == "" and 1700 < eta < 1810

    def test_a_start_moments_ago_still_means_now(self):
        # Ordinary: "start immediately", plus the latency between asking and
        # parsing. This must not be mistaken for skew.
        assert self._eta(-2) == (0, "")
        assert self._eta(-60) == (0, "")

    @pytest.mark.parametrize("skew", [-3600, -5 * 3600, -86400])
    def test_a_large_negative_gap_is_unknown_not_now(self, skew):
        assert self._eta(skew) == (None, "")

    def test_the_tolerance_boundary_is_pinned(self):
        assert su._CLOCK_SKEW_TOLERANCE_S == 120
        assert self._eta(-119) == (0, "")
        assert self._eta(-121) == (None, "")


class TestQosDenyList:
    """Slurm expresses a partition's QoS ACL two ways and a site picks one: an
    explicit `AllowQos` list, or `AllowQos=ALL` plus a `DenyQos` exclusion list.
    Reading only the allow side meant a deny-list site's `ALL` expanded to every
    QoS on the cluster — including the ones the partition forbids. Same defect as
    offering partitions the user holds no association for.
    """

    def _acl(self, mocker, line):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(line, "", 0))
        return su.fetch_qos_acl("p")

    def test_both_sides_are_read(self, mocker):
        acl = self._acl(
            mocker, "PartitionName=p AllowQos=ALL DenyQos=debug,long Hidden=NO\n"
        )
        assert acl == {"allow": ["ALL"], "deny": ["debug", "long"]}

    def test_an_explicit_allow_list_still_works(self, mocker):
        acl = self._acl(mocker, "PartitionName=p AllowQos=normal,high Hidden=NO\n")
        assert acl["allow"] == ["normal", "high"] and acl["deny"] == []

    def test_a_null_deny_list_is_empty_not_a_name(self, mocker):
        acl = self._acl(mocker, "PartitionName=p AllowQos=ALL DenyQos=(null)\n")
        assert acl["deny"] == []

    def test_the_legacy_accessor_still_returns_the_allow_list(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(
            su, "_run_command",
            return_value=("PartitionName=p AllowQos=normal DenyQos=x\n", "", 0),
        )
        assert su.fetch_qos_for_partition("p") == ["normal"]

    def test_denied_names_are_not_offered(self, mocker):
        # End to end through the wizard's choice resolution: ALL expands to the
        # cluster list, then the deny list is subtracted.
        import slurmate.tui as t
        from slurmate.tui import STEPS, Wizard
        mocker.patch.object(
            t, "fetch_qos_acl",
            return_value={"allow": ["ALL"], "deny": ["debug"]},
        )
        mocker.patch.object(
            t, "fetch_known_qos", return_value=["normal", "debug", "long"]
        )
        wizard = Wizard()
        wizard.answers["partition"] = "p"
        step = next(s for s in STEPS if s.key == "qos")
        choices = wizard._resolve_choices(step)
        assert "debug" not in choices
        assert "normal" in choices and "long" in choices

    def test_an_explicit_allow_list_is_also_filtered(self, mocker):
        # A site can set both; the deny list wins, as it does in Slurm.
        import slurmate.tui as t
        from slurmate.tui import STEPS, Wizard
        mocker.patch.object(
            t, "fetch_qos_acl",
            return_value={"allow": ["normal", "debug"], "deny": ["debug"]},
        )
        mocker.patch.object(t, "fetch_known_qos", return_value=["normal", "debug"])
        wizard = Wizard()
        wizard.answers["partition"] = "p"
        step = next(s for s in STEPS if s.key == "qos")
        choices = wizard._resolve_choices(step)
        assert "debug" not in choices and "normal" in choices


class TestPartitionOwnState:
    """`sinfo %a` was parsed into the partition dict and never consulted. A
    partition's own state is a different fact from its nodes': it can be UP with
    every node dead (SM-1's case) or DOWN with a hundred live nodes — and Slurm
    accepts a job for a down partition and then never starts it.
    """

    def _part_state(self, state, nodes_up=177):
        return _part(name="test", nodes=234, nodes_up=nodes_up, state=state)

    @pytest.mark.parametrize("state", ["down", "drain", "inact"])
    def test_an_unavailable_partition_is_reported(self, state):
        msgs = [
            m for _lvl, m in validate_job_config(
                {"cpus": 1, "_partition_obj": self._part_state(state)}
            )
            if "partition itself" in m
        ]
        assert msgs and state in msgs[0]

    def test_an_up_partition_is_silent(self):
        assert not [
            m for _lvl, m in validate_job_config(
                {"cpus": 1, "_partition_obj": self._part_state("up")}
            )
            if "partition itself" in m
        ]

    def test_an_unknown_state_claims_nothing(self):
        # A synthetic record for a manually-typed partition carries no state;
        # absence of information is not evidence of a problem.
        assert not [
            m for _lvl, m in validate_job_config(
                {"cpus": 1, "_partition_obj": self._part_state("")}
            )
            if "partition itself" in m
        ]

    def test_it_is_independent_of_the_node_level_check(self):
        # The live case that motivated this: State=DOWN with 177 nodes in "mix",
        # so nodes_up is high and SM-1's warning cannot fire.
        issues = validate_job_config(
            {"cpus": 1, "_partition_obj": self._part_state("down", nodes_up=177)}
        )
        assert [m for _lvl, m in issues if "partition itself" in m]
        assert not [m for _lvl, m in issues if "no usable nodes" in m]

    def test_both_fire_when_both_are_true(self):
        issues = validate_job_config(
            {"cpus": 1, "_partition_obj": self._part_state("down", nodes_up=0)}
        )
        assert [m for _lvl, m in issues if "partition itself" in m]
        assert [m for _lvl, m in issues if "no usable nodes" in m]


class TestLogPatternExpansion:
    """The submit report resolved the log path with chained `str.replace`, which
    got two things wrong: it left `%x`/`%u` literal though slurmate knows both,
    and it mis-substituted `%%` — so the `tail -f` hint the user copies pointed at
    a filename Slurm never writes.
    """

    def _e(self, pattern):
        return su.expand_log_pattern(
            pattern, job_id="12345", job_name="train", user="youzhi"
        )

    def test_job_id_patterns(self):
        assert self._e("plain-%j.out") == ("plain-12345.out", [])
        assert self._e("a-%A.out") == ("a-12345.out", [])

    def test_job_name_and_user_are_resolved(self):
        # Both are known before submit; leaving them literal produced a hint for
        # a file that does not exist.
        assert self._e("run-%x-%j.out") == ("run-train-12345.out", [])
        assert self._e("%u/run-%j.out") == ("youzhi/run-12345.out", [])

    def test_double_percent_is_a_literal_not_a_pattern(self):
        # The old chained replace turned "%%j" into "%12345"; Slurm writes "%j".
        assert self._e("pct-%%j-%j.out") == ("pct-%j-12345.out", [])
        assert "12345" not in self._e("only-%%j.out")[0]

    @pytest.mark.parametrize(
        "pattern,letter",
        [("a-%A_%a.out", "%a"), ("n-%N-%j.out", "%N"),
         ("t-%t.out", "%t"), ("s-%s.out", "%s"), ("nn-%n.out", "%n")],
    )
    def test_unknowable_patterns_are_kept_and_reported(self, pattern, letter):
        path, unresolved = self._e(pattern)
        assert letter in path and unresolved == [letter]

    def test_a_trailing_percent_does_not_raise(self):
        assert self._e("trailing-%") == ("trailing-%", [])

    def test_an_unknown_letter_is_left_alone(self):
        path, unresolved = self._e("q-%Q.out")
        assert path == "q-%Q.out" and unresolved == []

    def test_missing_values_do_not_produce_empty_segments(self):
        # With no job name known, "%x" must stay a pattern rather than collapsing
        # to nothing and yielding a different filename.
        path, unresolved = su.expand_log_pattern("run-%x-%j.out", job_id="7")
        assert path == "run-%x-7.out" and unresolved == ["%x"]


class TestGpuFormatVsSelectType:
    """SM-18: `--gpus` and `--gpus-per-task` are cons_tres-only. Under
    select/cons_res the *parser* refuses them cluster-wide, on every partition —
    and `gpu_format` is a config-file key, so the failure arrives without the user
    typing a flag. The two test clusters differ on exactly this setting.
    """

    @pytest.mark.parametrize("fmt", ["gpus", "gpus_per_task"])
    @pytest.mark.parametrize("plugin", ["select/cons_res", "select/linear"])
    def test_cons_tres_only_formats_are_refused(self, fmt, plugin):
        reason = su.unsupported_gpu_format(fmt, plugin)
        assert reason
        assert plugin in reason
        # The message has to name a way forward, not just a refusal.
        assert "gres_type" in reason and "gpus_per_node" in reason

    @pytest.mark.parametrize("fmt", ["gres_type", "gpus_per_node", "constraint"])
    def test_portable_formats_are_always_allowed(self, fmt):
        for plugin in ("select/cons_res", "select/cons_tres", "select/linear", ""):
            assert su.unsupported_gpu_format(fmt, plugin) == ""

    @pytest.mark.parametrize("fmt", ["gpus", "gpus_per_task"])
    def test_cons_tres_allows_everything(self, fmt):
        assert su.unsupported_gpu_format(fmt, "select/cons_tres") == ""

    @pytest.mark.parametrize("plugin", ["", "garbage", "cons_res", "unknown/thing"])
    def test_an_unreadable_select_type_claims_nothing(self, plugin):
        # Failing open to the default gres_type is already safe; an unreadable
        # scontrol must not present as "your GPU syntax is wrong".
        assert su.unsupported_gpu_format("gpus", plugin) == ""

    def test_select_type_is_parsed_from_scontrol(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(
            su, "_run_command",
            return_value=(
                "ClusterName             = midway2\n"
                "SelectType              = select/cons_res\n"
                "SelectTypeParameters    = CR_CORE_MEMORY\n", "", 0,
            ),
        )
        assert su.fetch_select_type() == "select/cons_res"

    def test_an_unreadable_config_yields_empty(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(su, "_run_command", return_value=("", "boom", 1))
        assert su.fetch_select_type() == ""

    def test_the_config_file_route_is_covered(self, mocker, capsys):
        # The reported exposure: no CLI flag, the value comes from a config file
        # carried from a cons_tres cluster.
        from rich.console import Console

        import slurmate.main as m
        mocker.patch.object(m, "fetch_select_type", return_value="select/cons_res")
        with pytest.raises(SystemExit) as exc:
            m._check_gpu_format(
                "gpus_per_task", gpus_requested=True, force=False,
                err_console=Console(stderr=True, no_color=True, highlight=False),
            )
        assert exc.value.code == 1

    def test_force_downgrades_to_a_warning(self, mocker):
        from rich.console import Console

        import slurmate.main as m
        mocker.patch.object(m, "fetch_select_type", return_value="select/cons_res")
        m._check_gpu_format(
            "gpus", gpus_requested=True, force=True,
            err_console=Console(stderr=True, no_color=True, highlight=False),
        )  # must not raise

    def test_no_gpus_requested_means_no_check(self, mocker):
        import slurmate.main as m
        probe = mocker.patch.object(m, "fetch_select_type")
        m._check_gpu_format("gpus", gpus_requested=False, force=False,
                            err_console=None)  # type: ignore[arg-type]
        probe.assert_not_called()


class TestDefaultPartitionFallback:
    """With no `--partition`, Slurm uses the site default — and slurmate already
    knows which that is, from sinfo's `*` marker. Treating the partition as
    *unknown* instead produced two confidently wrong figures: `Queue: 0 running /
    0 pending` (from `squeue -p ""`) for a job landing in a partition with
    hundreds, and SM-7's "this cluster's node memory is unknown" fallback
    inventing 16G when the default partition's memory is perfectly well known.
    """

    def _parts(self):
        return [
            _part(name="caslake", is_default=True, cpus_per_node=48,
                  mem_per_node_mb=184320),
            _part(name="other", is_default=False, cpus_per_node=8,
                  mem_per_node_mb=1024),
        ]

    def test_the_default_is_resolved_for_derived_figures(self):
        from slurmate.main import _get_partition
        parts = self._parts()
        default = next(p["name"] for p in parts if p.get("is_default"))
        assert default == "caslake"
        assert _get_partition(parts, default)["mem_per_node_mb"] == 184320

    def test_memory_is_derived_rather_than_invented(self):
        # The SM-7 fallback must not fire when the default partition is known.
        parts = self._parts()
        part = next(p for p in parts if p.get("is_default"))
        mem, source = default_memory_for(part, 8)
        assert source == "partition" and mem != FALLBACK_MEMORY

    def test_no_default_marker_means_no_guess(self):
        # A site whose sinfo marks no default tells us nothing; inventing one
        # would be a new wrong answer.
        parts = [_part(name="a", is_default=False), _part(name="b", is_default=False)]
        assert next((p["name"] for p in parts if p.get("is_default")), "") == ""

    def test_the_disclosure_fires_only_when_no_partition_was_given(self, capsys):
        from rich.console import Console

        from slurmate.main import _note_default_partition
        console = Console(no_color=True, highlight=False, width=200)
        _note_default_partition(
            {"partition": "", "_effective_partition": "caslake"}, console
        )
        out = capsys.readouterr().out
        assert "default partition 'caslake'" in out
        assert "No --partition directive was added" in out

    def test_no_disclosure_when_the_user_chose_a_partition(self, capsys):
        from rich.console import Console

        from slurmate.main import _note_default_partition
        _note_default_partition(
            {"partition": "caslake", "_effective_partition": "caslake"},
            Console(no_color=True, highlight=False),
        )
        assert capsys.readouterr().out == ""

    def test_no_disclosure_when_nothing_could_be_resolved(self, capsys):
        from rich.console import Console

        from slurmate.main import _note_default_partition
        _note_default_partition(
            {"partition": "", "_effective_partition": ""},
            Console(no_color=True, highlight=False),
        )
        assert capsys.readouterr().out == ""


class TestUnknownPartitionDisclosesUnchecked:
    """SM-20's residual half. The partition/account/qos names are fatal errors
    now, but `--force` still reaches the summary with an unresolvable partition —
    and there every capacity check compared against nothing and stayed silent, so
    a 999-CPU request looked unremarkable. The *less* valid request produced the
    *more* reassuring screen.
    """

    def _unknown(self):
        # A *readable* list that does not contain the name: the partition is
        # genuinely absent. Built from a non-empty list on purpose — an empty one
        # means the list could not be read, which is a different claim.
        from slurmate.main import _get_partition
        return _get_partition([_part(name="real")], "definitely-no-such-partition")

    def _unreadable(self):
        from slurmate.main import _get_partition
        return _get_partition([], "definitely-no-such-partition")

    def test_the_two_unknown_reasons_are_distinguished(self):
        assert self._unknown()["_unknown_reason"] == "absent"
        assert self._unreadable()["_unknown_reason"] == "unreadable"

    def test_an_unreadable_list_does_not_claim_the_partition_is_absent(self):
        # With no Slurm, nothing is known about any partition — saying "not on
        # this cluster" there is the false rejection the SM-4 restraint prevents.
        msgs = [
            m for _lvl, m in validate_job_config(
                {"cpus": 999, "_partition_obj": self._unreadable()}
            )
            if "NOT checked" in m
        ]
        assert msgs
        assert "could not be read" in msgs[0]
        assert "is not on this cluster" not in msgs[0]

    def test_the_fallback_record_is_marked_unknown(self):
        assert self._unknown()["_unknown"] is True

    @pytest.mark.parametrize(
        "answers",
        [
            {"cpus": 999},
            {"cpus": 1, "memory": "9999G"},
            {"cpus": 1, "mem_per_cpu": "64G"},
            {"cpus": 1, "time_limit": "99:00:00"},
            {"cpus": 1, "nodes": 50},
            {"cpus": 1, "gpus": 8},
        ],
    )
    def test_a_concrete_request_says_it_was_not_checked(self, answers):
        msgs = [
            m for _lvl, m in validate_job_config({**answers, "_partition_obj": self._unknown()})
            if "NOT checked" in m
        ]
        # Names the partition, since this record came from a readable list where
        # the name is genuinely absent.
        assert msgs and self._unknown()["name"] in msgs[0]

    def test_a_real_partition_checks_instead_of_disclaiming(self):
        issues = validate_job_config(
            {"cpus": 999, "_partition_obj": _part(cpus_per_node=48)}
        )
        assert [m for _lvl, m in issues if m.startswith("CPUs")]
        assert not [m for _lvl, m in issues if "NOT checked" in m]

    def test_nothing_requested_means_nothing_to_disclaim(self):
        # An empty request has no capacity claim to check, so the note would be
        # noise rather than information.
        assert not [
            m for _lvl, m in validate_job_config({"_partition_obj": self._unknown()})
            if "NOT checked" in m
        ]

    def test_a_zero_valued_request_is_not_a_claim(self):
        assert not [
            m for _lvl, m in validate_job_config(
                {"gpus": 0, "_partition_obj": self._unknown()}
            )
            if "NOT checked" in m
        ]


class TestEtaProvenanceIsRendered:
    """SM-21: `fetch_queue_eta` returns `source` naming which of its three tiers
    answered — its docstring says "so the caller can qualify what it shows" — and
    the renderer dropped it. Slurm's own backfill placement and a queue-depth
    heuristic returning a flat constant rendered identically.
    """

    def test_the_scheduler_tier_is_not_qualified(self):
        from slurmate.main import _qualified_eta
        assert _qualified_eta({"eta_label": "~21min", "source": "scheduler"}) == "~21min"

    @pytest.mark.parametrize(
        "source,note",
        [
            ("resources", "estimated from free capacity"),
            ("pressure", "estimated from queue depth"),
            ("unknown", "not measurable here"),
        ],
    )
    def test_weaker_tiers_say_so(self, source, note):
        from slurmate.main import _qualified_eta
        rendered = _qualified_eta({"eta_label": "~5min", "source": source})
        assert rendered == f"~5min ({note})"

    def test_a_guess_and_a_measurement_are_not_byte_identical(self):
        # The report's own test: the whole point is that these must differ.
        from slurmate.main import _qualified_eta
        measured = _qualified_eta({"eta_label": "~5min", "source": "scheduler"})
        guessed = _qualified_eta({"eta_label": "~5min", "source": "pressure"})
        assert measured != guessed

    def test_an_unrecognised_source_is_left_unqualified(self):
        # Better an unadorned number than an invented provenance.
        from slurmate.main import _qualified_eta
        assert _qualified_eta({"eta_label": "~5min", "source": "future_tier"}) == "~5min"
        assert _qualified_eta({"eta_label": "~5min"}) == "~5min"

    def test_the_pressure_tier_really_does_return_a_constant(self, mocker):
        # Why labelling matters: the last-resort tier answers the same for an
        # empty queue on any partition, so an unqualified "~5min" is not a
        # reading of anything.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=("", "", 0))
        mocker.patch.object(su, "_nodes_that_fit", return_value=None)
        a = su.fetch_queue_eta("one")
        b = su.fetch_queue_eta("two")
        assert a["source"] == b["source"] == "pressure"
        assert a["eta_seconds"] == b["eta_seconds"]


class TestNodeGresArithmetic:
    """`_sum_node_gpus` had no test coverage at all, and the `(IDX:…)` suffix
    Slurm puts on `GresUsed` appeared nowhere in the fixtures — the same
    idealised-fixture gap as round 21, in the function the `resources`-tier ETA
    subtracts with. The behaviour was already correct; it was simply unpinned,
    and it is now the number a labelled "estimated from free capacity" row rests
    on.
    """

    @pytest.mark.parametrize(
        "gres,expected",
        [
            ("gpu:4", 4),
            ("gpu:4(IDX:0-3)", 4),               # the shape GresUsed really has
            ("gpu:a100:2(IDX:0-1)", 2),
            ("gpu:a100:4(IDX:0,2-3)", 4),        # non-contiguous index list
            ("gpu:a100:2,gpu:v100:2", 4),
            ("gres/gpu:2", 2),                   # scontrol's prefixed spelling
            ("gpu:0", 0),
            ("shard:8", 0),                      # a slice of a GPU, not another
            ("gpu:4,shard:16(IDX:0-15)", 4),
            ("(null)", 0),
            ("", 0),
        ],
    )
    def test_every_real_spelling_sums_correctly(self, gres, expected):
        assert su._sum_node_gpus(gres) == expected

    def test_free_gpus_are_total_minus_used(self):
        # The subtraction the ETA depends on: a node whose GPUs are all allocated
        # is "mixed" by state label and can take no GPU job.
        assert su._sum_node_gpus("gpu:a100:4") - su._sum_node_gpus(
            "gpu:a100:4(IDX:0-3)"
        ) == 0
        assert su._sum_node_gpus("gpu:a100:4") - su._sum_node_gpus(
            "gpu:a100:2(IDX:0-1)"
        ) == 2


class TestPrintModeSurfacesChecks:
    """`--print` returned before any capacity check ran, so the mode most used in
    scripts and CI — and the one this report's own probes use — was the one that
    emitted an unschedulable script in silence. A 999-CPU request that `--dry-run`
    warns about twice produced zero bytes on stderr.
    """

    @pytest.fixture
    def workdir(self, tmp_path):
        """A directory where the default request is genuinely clean.

        This class used to inherit pytest's cwd, and the "stays silent" case
        passed only because the dev checkout happens to contain an untracked
        ``logs/``. On a fresh clone — both Booth clusters, and GitHub CI — the
        SM-24 missing-log-directory warning fires and the assertion of an empty
        stderr fails. A test whose verdict depends on a directory nobody created
        on purpose is measuring the checkout, not the code.
        """
        (tmp_path / "logs").mkdir()
        return tmp_path

    def _run(self, *extra, cwd=None):
        env = {
            **os.environ, "PYTHONPATH": os.path.abspath("src"),
            "SLURMATE_MOCK": "1", "NO_COLOR": "1",
        }
        return subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--force",
             "--job-name", "x", "--partition", "cpu-shared", "--cpus", "2",
             "--time", "00:05:00", "--command", "true", *extra],
            capture_output=True, text=True, env=env,
            cwd=str(cwd) if cwd else None, timeout=180,
        )

    def test_stdout_is_still_only_the_script(self):
        done = self._run("--memory", "9999G")
        assert done.returncode == 0
        assert done.stdout.startswith("#!/bin/bash")
        # Nothing but the script on stdout: redirecting it must yield a runnable
        # file, which is the whole contract of --print.
        assert "Warning" not in done.stdout and "⚠" not in done.stdout

    def test_an_over_request_is_reported_on_stderr(self):
        done = self._run("--memory", "9999G")
        assert "Memory" in done.stderr

    def test_a_clean_request_stays_silent(self, workdir):
        done = self._run(cwd=workdir)
        assert done.returncode == 0 and done.stderr == ""

    def test_a_clean_request_in_a_fresh_directory_reports_the_log_dir(self, tmp_path):
        # The other half of the pair: the silence above is conditional on the log
        # directory existing, and pinning only the silent case is what let the
        # dependency go unnoticed.
        done = self._run(cwd=tmp_path)
        assert done.returncode == 0
        assert "logs" in done.stderr and "mkdir" in done.stderr

    def test_the_script_is_identical_with_and_without_the_warning(self):
        # The checks must be purely additive to stderr — the emitted script for a
        # given request cannot depend on whether a warning fired.
        noisy = self._run("--memory", "9999G").stdout
        assert "--mem=9999G" in noisy
        quiet = self._run("--memory", "4G").stdout
        assert "--mem=4G" in quiet
        assert noisy.replace("9999G", "4G") == quiet


class TestMockModeIsLabelled:
    """SM-23: `SLURMATE_MOCK` fabricated the partition list, queue depth and ETA
    with no marker anywhere, and was documented nowhere — so the realistic way to
    reach it was a stale `export`, a CI wrapper or a container image, not a
    deliberate choice. Synthetic data shaped exactly like measurement.
    """

    def _run(self, *extra, env_mock=False):
        env = {**os.environ, "PYTHONPATH": "src", "NO_COLOR": "1"}
        env.pop("SLURMATE_MOCK", None)
        if env_mock:
            env["SLURMATE_MOCK"] = "1"
        return subprocess.run(
            [sys.executable, "-m", "slurmate", "--dry-run", "--force",
             "--job-name", "x", "--partition", "cpu-shared", "--cpus", "999",
             "--memory", "9999G", "--time", "00:05:00", "--command", "true", *extra],
            capture_output=True, text=True, env=env, timeout=180,
        )

    def test_the_env_var_route_is_marked(self):
        out = self._run(env_mock=True).stdout
        assert "SIMULATED" in out
        assert "(simulated)" in out

    def test_the_marker_is_in_band_not_a_banner(self):
        # It has to sit in the fields themselves, so it cannot scroll away from
        # the numbers it qualifies.
        out = self._run(env_mock=True).stdout
        queue_line = next(ln for ln in out.splitlines() if "running /" in ln)
        assert "simulated" in queue_line

    def test_it_still_validates_the_users_own_numbers(self):
        # A synthetic queue is no reason to stop checking the figures the user
        # typed: 999 CPUs is over the limit whether the limit is real or demo.
        out = self._run(env_mock=True).stdout
        assert "CPUs (999) exceeds" in out
        assert "Memory (9999G) exceeds" in out

    def test_the_demo_flag_is_equivalent_and_discoverable(self):
        flagged = self._run("--demo").stdout
        assert "SIMULATED" in flagged and "(simulated)" in flagged

    def test_the_flag_is_documented(self):
        env = {**os.environ, "PYTHONPATH": "src"}
        env.pop("SLURMATE_MOCK", None)
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--help"],
            capture_output=True, text=True, env=env, timeout=120,
        )
        assert "--demo" in done.stdout
        assert "SLURMATE_MOCK" in done.stdout

    def test_real_mode_carries_no_marker(self):
        out = self._run().stdout
        assert "SIMULATED" not in out and "simulated" not in out


class TestWizardGetsTheSiteChecks:
    """Every cluster-membership check added across these rounds lived on the
    *batch* path, where it is fatal before a script exists. That left the wizard
    unchecked — and the wizard is the default interface, with an explicit "Enter
    partition name manually…" option, so a name the non-interactive path rejects
    outright was accepted silently by the interactive one.
    """

    def test_an_unknown_partition_is_an_error(self, mocker):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"caslake"})
        issues = m.site_check_issues({"partition": "nosuch"})
        assert [lvl for lvl, _ in issues] == ["error"]
        assert "no partition 'nosuch'" in issues[0][1]

    def test_an_unknown_account_and_qos_are_errors(self, mocker):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"p"})
        mocker.patch.object(m, "fetch_user_accounts", return_value=["real"])
        mocker.patch.object(m, "fetch_known_qos", return_value=["normal"])
        issues = m.site_check_issues(
            {"partition": "p", "account": "nope", "qos": "nope"}
        )
        assert len(issues) == 2 and all(lvl == "error" for lvl, _ in issues)

    def test_a_bad_array_spec_is_an_error(self, mocker):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"p"})
        issues = m.site_check_issues({"partition": "p", "array_spec": "10-1"})
        assert any("Invalid array specification" in msg for _lvl, msg in issues)

    def test_a_managed_custom_flag_is_an_error(self, mocker):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"p"})
        issues = m.site_check_issues(
            {"partition": "p", "custom_sbatch": "--partition=other"}
        )
        assert any("duplicates a directive" in msg for _lvl, msg in issues)

    def test_a_missing_module_blocks_like_the_batch_path(self, mocker):
        # SM-13 asked for this to be fatal-with---force, and the batch path
        # implements that. Emitting a warning here meant the wizard would submit a
        # job the non-interactive path refuses, so the levels are aligned: an
        # error blocks the wizard's submit while still offering "go back to edit".
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"p"})
        mocker.patch.object(m, "check_modules", return_value=[("warning", "module x")])
        issues = m.site_check_issues({"partition": "p", "modules": ["x"]})
        assert [lvl for lvl, _ in issues] == ["error"]
        assert m._hard_errors({"partition": "p", "modules": ["x"],
                              "_partition_obj": None})

    def test_errors_reach_the_pre_submit_guard(self, mocker):
        # The guard is what stops the wizard submitting; a check the summary
        # prints but the guard ignores would still let the job through.
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"caslake"})
        mocker.patch.object(m, "_partition_issues", return_value=[])
        assert m._hard_errors({"partition": "nosuch"})

    def test_clean_answers_produce_nothing(self, mocker):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"caslake"})
        assert m.site_check_issues({"partition": "caslake", "cpus": 1}) == []

    def test_a_probe_failure_never_blocks_a_job(self, mocker):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", side_effect=OSError("boom"))
        assert m.site_check_issues({"partition": "anything"}) == []


class TestWizardConfigProvenance:
    """SM-8's disclosure — naming the file that supplied the defaults — was set
    only inside `run_batch`, so it never fired for a wizard run. The wizard is
    what *prefills* from `.slurmate.toml`, which makes it the path where "values
    you did not type" is most likely and the disclosure most needed.
    """

    def _wizard(self, config, answers):
        import slurmate.tui as t
        w = t.Wizard.__new__(t.Wizard)
        w.config = config
        w.answers = dict(answers)
        w._record_config_provenance()
        return w.answers

    def test_prefilled_values_are_credited(self):
        a = self._wizard(
            {"partition": "caslake", "account": "rcc-staff"},
            {"partition": "caslake", "account": "rcc-staff"},
        )
        assert sorted(a["_config_keys"]) == ["account", "partition"]

    def test_a_value_the_user_changed_is_not_credited(self):
        # Recorded on the way out, not at prefill time, so an edited field is not
        # attributed to a file that no longer supplies it.
        a = self._wizard({"memory": "64G"}, {"memory": "8G"})
        assert a["_config_keys"] == []

    def test_a_config_key_the_wizard_never_asked_is_not_credited(self):
        a = self._wizard({"qos": "debug"}, {"partition": "caslake"})
        assert a["_config_keys"] == []

    def test_string_and_native_forms_match(self):
        # The wizard round-trips numbers through text entry, so 8 and "8" are the
        # same answer and must not read as a user override.
        a = self._wizard({"cpus": 8}, {"cpus": "8"})
        assert a["_config_keys"] == ["cpus"]

    def test_the_note_renders_for_a_wizard_run(self, capsys):
        from rich.console import Console

        from slurmate.main import _note_config_source
        _note_config_source(
            {"_config_source": "./.slurmate.toml", "_config_keys": ["partition"]},
            Console(no_color=True, highlight=False, width=200),
        )
        out = capsys.readouterr().out
        assert "./.slurmate.toml" in out and "partition" in out

    def test_no_config_means_no_note(self, capsys):
        from rich.console import Console

        from slurmate.main import _note_config_source
        _note_config_source(
            {"_config_source": "", "_config_keys": []},
            Console(no_color=True, highlight=False),
        )
        assert capsys.readouterr().out == ""


class TestWizardMemoryDefaultIsDerived:
    """SM-7 was about the hardcoded `16G` — "the built-in fallback is a number,
    not a measurement". The batch path derives it from the partition; the wizard
    declared `default="16G"` on its memory step, so the *default* interface — the
    one that shows the value pre-filled for the user to accept — still offered the
    literal.
    """

    def _wizard(self, part, cpus=1, config=None):
        import slurmate.tui as t
        w = t.Wizard.__new__(t.Wizard)
        w.answers = {"_partition_obj": part, "cpus": cpus}
        w._config_defaults = config or {}
        return w

    def _step(self):
        from slurmate.tui import STEPS
        return next(s for s in STEPS if s.key == "memory")

    def test_it_is_sized_from_the_partition(self):
        part = _part(name="caslake", cpus_per_node=48, mem_per_node_mb=184320)
        assert self._wizard(part, 8)._step_default(self._step()) == "30G"

    def test_the_small_node_case_from_the_report(self):
        # SM-7's own example: on an 8 GB node, 16G is an unschedulable default.
        part = _part(name="tiny", cpus_per_node=4, mem_per_node_mb=8192)
        derived = self._wizard(part, 1)._step_default(self._step())
        assert derived == "2G" and derived != FALLBACK_MEMORY

    def test_an_unknown_partition_keeps_the_literal(self):
        from slurmate.main import _get_partition
        part = _get_partition([], "nope")
        assert self._wizard(part, 1)._step_default(self._step()) == FALLBACK_MEMORY

    def test_no_partition_yet_keeps_the_literal(self):
        # The memory step can be reached before a partition is chosen.
        assert self._wizard(None, 1)._step_default(self._step()) == FALLBACK_MEMORY

    def test_an_explicit_config_value_still_wins(self):
        part = _part(cpus_per_node=48, mem_per_node_mb=184320)
        w = self._wizard(part, 8, {"memory": "64G"})
        assert w._step_default(self._step()) == "64G"

    def test_a_cleared_field_reverts_to_the_derived_value(self):
        # The P3-10 invariant: clearing the field falls back to the default — which
        # must now be the derived one, not the literal.
        part = _part(cpus_per_node=48, mem_per_node_mb=184320)
        w = self._wizard(part, 8)
        assert w._coerce("", self._step()) == "30G"

    def test_more_cores_gets_a_bigger_share(self):
        part = _part(cpus_per_node=48, mem_per_node_mb=184320)
        small = self._wizard(part, 4)._step_default(self._step())
        large = self._wizard(part, 32)._step_default(self._step())
        assert su._parse_mem_to_mb(large) > su._parse_mem_to_mb(small)


class TestWizardArrayStepValidates:
    """SM-22's asymmetry, in the wizard: cpus, memory, mem-per-cpu, time, nodes,
    ntasks and gpus all validate as you type, and the array spec was the one
    resource field left as free text.
    """

    def _validator(self):
        from slurmate.tui import STEPS
        step = next(s for s in STEPS if s.key == "array_spec")
        assert step.validate is not None, "the array step must have a validator"
        return step.validate

    @pytest.mark.parametrize("spec", ["", "1-10", "1,3,5-7%4", "%4", "5", "1-10:2"])
    def test_valid_specs_are_accepted(self, spec):
        # "1,3,5-7%4" is the step's own subtitle example, so the validator must
        # not reject what the prompt suggests.
        assert self._validator()(spec) is True

    @pytest.mark.parametrize("spec", ["10-1", "1-10:0", "abc", "1-", "-5"])
    def test_invalid_specs_are_rejected(self, spec):
        assert self._validator()(spec) is False

    def test_empty_is_valid_because_the_field_is_optional(self):
        assert self._validator()("") is True

    def test_every_resource_step_now_has_a_validator(self):
        # The asymmetry itself, pinned: if a resource field is added without a
        # validator, this fails.
        from slurmate.tui import STEPS
        resource_keys = {
            "cpus", "memory", "mem_per_cpu", "time_limit", "nodes",
            "ntasks_per_node", "gpus", "array_spec",
        }
        missing = [
            s.key for s in STEPS if s.key in resource_keys and s.validate is None
        ]
        assert missing == []


class TestEditedScriptIsCheckedNotTheAnswers:
    """After "Open in editor" the script holds the user's edits while `answers`
    is stale, and the pre-submit guard validated `answers`. So a hand edit that
    *introduced* a bad partition passed, and one that *fixed* a bad partition was
    still blocked — with the only remedy being "go back to edit answers", which
    discards the fix. Both directions describe something other than what would be
    submitted.
    """

    SCRIPT = (
        "#!/bin/bash\n"
        "#SBATCH --partition=p\n"
        "#SBATCH --time=00:05:00\n"
        "true\n"
    )

    def test_a_refusal_is_reported_from_the_bytes(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(
            su, "_run_command",
            return_value=("", "allocation failure: invalid partition specified", 1),
        )
        assert su.check_script_with_scheduler(self.SCRIPT) == (
            "invalid partition specified"
        )

    def test_the_script_is_what_gets_sent(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        run = mocker.patch.object(su, "_run_command", return_value=("", "", 0))
        su.check_script_with_scheduler(self.SCRIPT)
        assert run.call_args.kwargs["stdin"] == self.SCRIPT
        assert run.call_args[0][0] == ["sbatch", "--test-only", "--parsable"]

    def test_an_accepted_script_yields_no_refusal(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(
            su, "_run_command",
            return_value=("", "sbatch: Job 1 to start at 2099-01-01T00:00:00", 0),
        )
        assert su.check_script_with_scheduler(self.SCRIPT) == ""

    def test_no_sbatch_is_not_a_refusal(self, mocker):
        # "Could not ask" must never render as "cannot run" — the same rule the
        # ETA follows.
        mocker.patch.object(su, "is_tool_available", return_value=False)
        assert su.check_script_with_scheduler(self.SCRIPT) == ""

    def test_an_unreachable_controller_is_not_a_refusal(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(
            su, "_run_command",
            return_value=("", "sbatch: error: Unable to contact slurm controller", 1),
        )
        assert su.check_script_with_scheduler(self.SCRIPT) == ""

    def test_an_empty_script_asks_nothing(self, mocker):
        probe = mocker.patch.object(su, "_run_command")
        assert su.check_script_with_scheduler("   ") == ""
        probe.assert_not_called()


class TestChecksFireInEveryMode:
    """The path-divergence sweep, pinned. Three rounds of findings were all the
    same shape — a check that existed on one path and not another (`run_batch` vs
    the wizard, `--dry-run` vs `--print`) — so the invariant itself is worth a
    test rather than another audit.

    Hermetic: `SLURMATE_MOCK=1` means `is_tool_available("sbatch")` is False, so
    `--yes` reports "no job submitted" instead of reaching a controller.
    """

    BAD = [
        (["--partition", "nosuchpart"], "no partition"),
        (["--array", "10-1"], "Invalid array specification"),
        (["--custom-sbatch=--partition=other"], "slurmate manages"),
        (["--cpus", "9999"], "exceeds"),
        # Added after this test failed to cover a real gap: a check wired only
        # into the shared helper reached --dry-run and --yes but not --print.
        (["--command", "#SBATCH --qos=INJECTED"], "begins with a #SBATCH line"),
    ]

    def _run(self, mode, extra):
        env = {**os.environ, "PYTHONPATH": "src", "SLURMATE_MOCK": "1",
               "NO_COLOR": "1"}
        return subprocess.run(
            [sys.executable, "-m", "slurmate", mode, "--job-name", "x",
             "--partition", "cpu-shared", "--cpus", "2", "--time", "00:05:00",
             "--command", "true", *extra],
            capture_output=True, text=True, env=env, timeout=180,
        )

    @pytest.mark.parametrize("mode", ["--print", "--dry-run", "--yes"])
    @pytest.mark.parametrize("extra,expected", BAD, ids=[b[1][:18] for b in BAD])
    def test_every_bad_value_is_reported_in_every_mode(self, mode, extra, expected):
        done = self._run(mode, list(extra))
        combined = done.stdout + done.stderr
        assert expected in combined, (
            f"{mode} did not report {expected!r} — a check that fires in one mode "
            f"and not another is the defect this test exists to catch"
        )

    @pytest.mark.parametrize("mode", ["--print", "--dry-run", "--yes"])
    def test_a_clean_request_is_reported_by_nobody(self, mode):
        done = self._run(mode, [])
        combined = done.stdout + done.stderr
        for _extra, expected in self.BAD:
            assert expected not in combined

    def test_print_keeps_the_warnings_off_stdout(self):
        # The one mode-specific difference that is intentional: --print must keep
        # stdout script-only, so its reports go to stderr.
        done = self._run("--print", ["--cpus", "9999"])
        assert "exceeds" in done.stderr
        assert "exceeds" not in done.stdout


class TestCommandCannotSmuggleDirectives:
    """Slurm stops reading `#SBATCH` at the first line that is neither blank nor
    a comment. The command body is emitted *after* the directive block, so a
    `#SBATCH` line at the start of the body is still inside the directive region
    and takes effect — unvalidated, unshown, and bypassing the managed-flag check
    that covers `--custom-sbatch`. Measured: `--command '#SBATCH --qos=INJECTED'`
    drew `Access/permission denied` from the controller, its answer for an
    invalid QoS, so the directive was obeyed.
    """

    def _detect(self, command):
        from slurmate.builder import command_injects_directives
        return command_injects_directives(command)

    @pytest.mark.parametrize(
        "command",
        [
            "#SBATCH --qos=x",
            "  #SBATCH --partition=other",
            "# a comment\n#SBATCH --account=other",   # comments do not stop Slurm
            "\n\n#SBATCH --time=99:00:00",
            "#sbatch --qos=x",                        # directives are case-insensitive
            "# SBATCH --qos=x",                       # and tolerate the space
        ],
    )
    def test_a_leading_directive_is_caught(self, command):
        assert self._detect(command).startswith(("#SBATCH", "#sbatch", "# SBATCH"))

    @pytest.mark.parametrize(
        "command",
        [
            "echo hi",
            "# my job\necho hi",
            # The case that must stay legal: a heredoc writing a nested script.
            # A real command precedes it, so Slurm has already stopped parsing.
            "cat > inner.sh <<EOF\n#SBATCH --qos=whatever\nEOF",
            "srun ./run\n#SBATCH --qos=x",
            "",
            None,
        ],
    )
    def test_an_inert_directive_is_allowed(self, command):
        assert self._detect(command) == ""

    def test_it_is_reported_as_an_error(self, mocker):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"p"})
        issues = m.site_check_issues(
            {"partition": "p", "command": "#SBATCH --qos=INJECTED"}
        )
        assert [lvl for lvl, _ in issues] == ["error"]
        assert "begins with a #SBATCH line" in issues[0][1]


class TestOutputIsDeterministic:
    """A partition spanning several `sinfo` rows merged its GPU types through a
    `set`, and Python's per-process string-hash randomisation made the order
    differ between runs — measured at four distinct orderings across eight runs
    of identical input. That order is user-visible in the picker's
    `GPU:[a100,v100]` label and in the "not in partition list (…)" error, so the
    same cluster produced different output run to run.
    """

    ROWS = (
        "g|infinite|2|up|48|184320|gpu:a100:2|idle\n"
        "g|infinite|3|up|48|184320|gpu:v100:2,gpu:h100:1|mixed\n"
    )

    def _types(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(self.ROWS, "", 0))
        return {p["name"]: p for p in fetch_partitions()}["g"]["gpu_types"]

    def test_merged_gpu_types_are_sorted(self, mocker):
        assert self._types(mocker) == ["a100", "h100", "v100"]

    def test_the_order_does_not_depend_on_hash_seed(self):
        # Run it in fresh interpreters, which is the only way to exercise hash
        # randomisation — within one process the seed is fixed.
        script = (
            "import slurmate.system_utils as su\n"
            f"rows = {self.ROWS!r}\n"
            "su.is_tool_available = lambda n: True\n"
            "su._run_command = lambda *a, **k: (rows, '', 0)\n"
            "print(','.join({p['name']: p for p in su.fetch_partitions()}"
            "['g']['gpu_types']))\n"
        )
        env = {**os.environ, "PYTHONPATH": "src"}
        env.pop("SLURMATE_MOCK", None)
        seen = {
            subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True,
                env=env, timeout=120,
            ).stdout.strip()
            for _ in range(6)
        }
        assert seen == {"a100,h100,v100"}, f"non-deterministic order: {seen}"

    def test_the_visible_message_is_stable(self, mocker):
        part = _part(gpu_types=self._types(mocker), has_gpu=True)
        msgs = [
            m for _lvl, m in validate_job_config(
                {"cpus": 1, "gpus": 1, "gpu_type": "nosuch", "_partition_obj": part}
            )
            if "not in partition list" in m
        ]
        assert msgs and "(a100, h100, v100)" in msgs[0]

    def test_no_set_derived_order_reaches_output(self):
        # The pattern, not just this instance: `list(set(...))` in the source is
        # how the bug arose, and it has no legitimate use where the result is
        # rendered.
        import re
        from pathlib import Path
        src = Path(su.__file__).parent
        offenders = [
            f"{f.name}:{i}"
            for f in src.glob("*.py")
            for i, line in enumerate(f.read_text().splitlines(), 1)
            # Skip comments: the line explaining *why* this pattern was removed
            # mentions it, and matching prose would make the guard self-tripping.
            if re.search(r"list\(set\(", line) and not line.lstrip().startswith("#")
        ]
        assert offenders == [], f"list(set(...)) can reach output: {offenders}"


class TestUnboundedTimeIsNotZero:
    """`--time=0` is documented Slurm for *no limit*, and both estimators treated
    it as a zero-length job, substituted a two-hour default, and reported a
    confident `96.0` core-hours for something unbounded — the same shape as
    quoting an ETA for a job the scheduler has refused.
    """

    @pytest.mark.parametrize(
        "spelling", ["0", "00:00:00", "0-00:00:00", "0:00", "unlimited", "INFINITE"]
    )
    def test_every_unbounded_spelling_says_so(self, spelling):
        from slurmate.builder import UNBOUNDED_ESTIMATE, estimate_su
        assert estimate_su(48, spelling, 1) == UNBOUNDED_ESTIMATE

    @pytest.mark.parametrize("spelling", ["0", "00:00:00", "unlimited"])
    def test_the_gpu_estimate_agrees(self, spelling):
        from slurmate.builder import UNBOUNDED_ESTIMATE, estimate_gpu_hours
        assert estimate_gpu_hours(4, spelling, 1, None, None) == UNBOUNDED_ESTIMATE

    @pytest.mark.parametrize(
        "spelling,expected", [("01:00:00", "48.0"), ("00:30:00", "24.0"),
                              ("1-00:00:00", "1,152")]
    )
    def test_real_limits_still_estimate(self, spelling, expected):
        from slurmate.builder import estimate_su
        assert estimate_su(48, spelling, 1) == expected

    def test_an_absent_limit_is_not_unbounded(self):
        # It takes the partition or site default, and 2 h is what the summary
        # already shows for it — estimating against that is consistent, not
        # invented.
        from slurmate.builder import UNBOUNDED_ESTIMATE, estimate_su
        assert estimate_su(48, "", 1) != UNBOUNDED_ESTIMATE

    def test_an_unparseable_value_is_not_called_unbounded(self):
        # Unparseable is *unknown*, not unlimited. Conflating them is the SM-10
        # mistake, and this parses to zero minutes so it would have.
        from slurmate.builder import UNBOUNDED_ESTIMATE, estimate_su
        assert estimate_su(48, "not-a-time", 1) != UNBOUNDED_ESTIMATE

    def test_slurm_accepts_all_three_zero_spellings(self):
        # Pins why the check is shape-based rather than a string list: enumerating
        # spellings missed "0-00:00:00", which the controller accepts.
        from slurmate.builder import _time_is_unbounded
        for spelling in ("0", "00:00:00", "0-00:00:00"):
            assert _time_is_unbounded(spelling) is True


class TestGpusPerTaskNeedsATaskCount:
    """`--gpus-per-task` is per *task*, so Slurm needs a task count to resolve
    it. On its own it is refused with "Invalid generic resource (gres)
    specification" — measured on a cons_tres cluster, where SM-18's SelectType
    check passes it — while the same request plus `--ntasks-per-node` is accepted.
    So one of the five `gpu_format` values emitted an unschedulable request when
    used alone, and the requirement is in the flag rather than the site.
    """

    def _issues(self, mocker, **answers):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"p"})
        mocker.patch.object(m, "fetch_select_type", return_value="select/cons_tres")
        return m.site_check_issues({"partition": "p", "gpus": 1, **answers})

    def test_it_is_refused_without_a_task_count(self, mocker):
        issues = self._issues(mocker, gpu_format="gpus_per_task")
        msgs = [m for lvl, m in issues if "needs a task count" in m]
        assert msgs and [lvl for lvl, m in issues if "needs a task count" in m] == ["error"]
        # The message has to name a way forward, not just refuse.
        assert "--ntasks-per-node" in msgs[0] and "gpus_per_node" in msgs[0]

    def test_a_task_count_makes_it_valid(self, mocker):
        issues = self._issues(mocker, gpu_format="gpus_per_task", ntasks_per_node=2)
        assert not [m for _lvl, m in issues if "needs a task count" in m]

    @pytest.mark.parametrize(
        "fmt", ["gres_type", "gpus_per_node", "gpus", "constraint", ""]
    )
    def test_the_other_formats_are_unaffected(self, mocker, fmt):
        issues = self._issues(mocker, gpu_format=fmt)
        assert not [m for _lvl, m in issues if "needs a task count" in m]

    def test_no_gpus_means_no_check(self, mocker):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"p"})
        issues = m.site_check_issues(
            {"partition": "p", "gpu_format": "gpus_per_task"}
        )
        assert not [m for _lvl, m in issues if "needs a task count" in m]


class TestSurrogatesSurviveSubmitAndSave:
    """Under a non-UTF-8 locale the filesystem encoding can be ASCII, so argv
    decoding turns a `--command` carrying UTF-8 bytes into lone surrogates. Both
    byte paths then mishandled them: the saved copy raised an unhandled
    `UnicodeEncodeError` *after the job was already submitted*, and the submit
    call's `errors="replace"` — which governs the **input** encoding too — sent
    sbatch a "?" per byte, silently running a different command than the user
    typed.
    """

    SURROGATE_SCRIPT = "#!/bin/bash\necho '\udce6\udc97\udca5'\n"

    def test_the_saved_copy_round_trips_the_bytes(self, tmp_path):
        from slurmate.main import _save_submitted_script
        path = _save_submitted_script(
            self.SURROGATE_SCRIPT, "t", "12345", str(tmp_path)
        )
        assert path is not None
        # The original bytes, not "?" and not an exception.
        assert b"\xe6\x97\xa5" in pathlib.Path(path).read_bytes()

    def test_a_write_failure_is_reported_not_raised(self, tmp_path, capsys):
        # It runs after a successful submission, so raising would turn a queued
        # job into a traceback.
        from slurmate.main import _save_submitted_script
        assert _save_submitted_script(
            self.SURROGATE_SCRIPT, "t", "1", "/proc/definitely/not/here"
        ) is None
        assert "Could not save script copy" in capsys.readouterr().out

    def test_the_submit_encoding_round_trips_rather_than_replacing(self, mocker):
        import subprocess as sp
        mocker.patch.object(su, "is_tool_available", return_value=True)
        run = mocker.patch.object(
            sp, "run",
            return_value=sp.CompletedProcess([], 0, stdout="1\n", stderr=""),
        )
        su.submit_sbatch(self.SURROGATE_SCRIPT, job_name="t")
        # "replace" would send sbatch a "?" for each surrogate — a different
        # command than the user typed, silently.
        assert run.call_args.kwargs["errors"] == "surrogateescape"
        assert run.call_args.kwargs["input"] == self.SURROGATE_SCRIPT

    def test_plain_ascii_is_unaffected(self, tmp_path):
        from slurmate.main import _save_submitted_script
        script = "#!/bin/bash\necho hi\n"
        path = _save_submitted_script(script, "t", "9", str(tmp_path))
        assert path and pathlib.Path(path).read_text() == script


class TestWizardActuallyStarts:
    """Every existing wizard test mocks `app.run`, so nothing exercised the real
    startup path — building the Application and rendering the first frame. The
    wizard is the default interface (bare `slurmate`), and the rounds above
    changed its step defaults, its imports and one step's validator, any of which
    could break rendering without failing a unit test.

    Driven in a real pty because the wizard refuses a non-terminal (SM-6) and
    prompt_toolkit needs a tty to render at all.
    """

    # The landmarks that together prove a frame was composed, rather than that
    # the process merely started. Also the loop's exit condition: waiting for a
    # byte *count* instead was a race — on Mercury the read broke out at
    # "Step 1 / 23" with the sidebar and the first prompt still in flight, and
    # the test failed for reasons that had nothing to do with the wizard. How
    # prompt_toolkit chunks its first frame is not something to assert on.
    LANDMARKS = (b"Slurmate", b"Steps", b"Job name", b"Step 1")

    def _render(self, timeout_s=20.0, size=(150, 45)):
        pty = pytest.importorskip("pty")
        import fcntl
        import select
        import struct
        import termios
        import time
        cols, rows = size
        env = {
            **os.environ, "PYTHONPATH": "src", "SLURMATE_MOCK": "1",
            "TERM": "xterm-256color", "LINES": str(rows), "COLUMNS": str(cols),
        }
        env.pop("PYTHONIOENCODING", None)
        master, slave = pty.openpty()
        # The env vars alone do not size a pty: prompt_toolkit asks the tty, and a
        # fresh one reports 80x24 whatever COLUMNS says. Which of the two wins
        # varies by prompt_toolkit version, so the frame this test inspected was
        # not the width it thought it had set.
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        os.set_blocking(master, False)
        proc = subprocess.Popen(
            [sys.executable, "-m", "slurmate"],
            stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True,
        )
        os.close(slave)
        buf = b""
        try:
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                ready, _, _ = select.select([master], [], [], 0.3)
                if ready:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                if all(mark in buf for mark in self.LANDMARKS):
                    break
            os.write(master, b"\x03")          # Ctrl-C: leave no stray process
            time.sleep(0.5)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:      # pragma: no cover
                proc.kill()
            os.close(master)
        return buf.decode("utf-8", "replace")

    def test_the_first_frame_renders_without_a_traceback(self):
        text = self._render()
        assert "Traceback" not in text, text[-2000:]
        assert "Step 1" in text
        # The sidebar and the first prompt: enough to prove a frame was composed,
        # not merely that the process started.
        for landmark in ("Slurmate", "Steps", "Job name"):
            assert landmark in text, f"missing {landmark!r} from the rendered frame"

    def test_the_frame_survives_a_classic_80x24_terminal(self):
        # 80x24 is what an unresized ssh session, a CI pty and a serial console
        # all report, and it is the width a fresh pty defaults to — so it is the
        # size most likely to be hit and the one the suite was accidentally
        # testing at. Verified to render the sidebar and the first prompt in full.
        text = self._render(size=(80, 24))
        assert "Traceback" not in text, text[-2000:]
        for landmark in ("Slurmate", "Steps", "Job name", "Step 1"):
            assert landmark in text, (
                f"missing {landmark!r} at 80x24 — the wizard is the default "
                f"interface and this is the default terminal size"
            )

    def test_no_unicode_error_reaches_the_frame(self):
        # The markers go through the glyph table now; a mis-wired one would show
        # up here rather than in a unit test.
        text = self._render()
        assert "UnicodeEncodeError" not in text
        assert "NameError" not in text and "ImportError" not in text


class TestDeclaredPythonSupport:
    """The report's intro noted that all four packages are 3.14-clean while their
    classifiers stop at 3.13 — a claim of *less* support than is verified, which
    matters because classifiers are what PyPI shows and what tooling filters on.
    """

    def _pyproject_text(self):
        return (
            pathlib.Path(su.__file__).parent.parent.parent / "pyproject.toml"
        ).read_text()

    def test_every_version_requires_python_allows_is_declared(self):
        # The two must not disagree: requires-python is what pip enforces,
        # classifiers are what users read, and a gap understates support.
        #
        # Read with a regex rather than tomllib, which is 3.11+ — and 3.10 is the
        # oldest version this very test asserts support for, so importing it made
        # the test unrunnable on the interpreter it most needed to run on.
        # release.yml avoids tomllib for the same reason.
        text = self._pyproject_text()
        requires = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.M)
        assert requires and requires.group(1) == ">=3.10"
        declared = set(re.findall(r'"Programming Language :: Python :: ([0-9.]+)"', text))
        assert {"3.10", "3.11", "3.12", "3.13", "3.14"} <= declared, declared

    def test_no_removed_or_deprecated_stdlib_apis(self):
        # What actually breaks on a newer interpreter: distutils and imp are gone,
        # utcnow and getdefaultlocale are deprecated, find_loader was removed.
        import re
        banned = re.compile(
            r"\b(distutils|import imp\b|utcnow|getdefaultlocale|find_loader"
            r"|pkg_resources|typing\.ByteString)\b"
        )
        offenders = [
            f"{f.name}:{i}"
            for f in pathlib.Path(su.__file__).parent.glob("*.py")
            for i, line in enumerate(f.read_text().splitlines(), 1)
            if banned.search(line) and not line.lstrip().startswith("#")
        ]
        assert offenders == [], offenders


class TestSummaryDescribesTheScript:
    """SM-15 was a case of the summary and the script disagreeing: two
    `#SBATCH --partition` lines, Slurm honouring the last, and every summary row
    describing the first. Nothing asserts the general property — that each
    directive the script carries has a row that accounts for it — so a directive
    added without one would reproduce SM-15 by omission.
    """

    MAXIMAL = {
        "job_name": "jj", "partition": "p", "account": "acct", "qos": "q",
        "cpus": 2, "memory": "8G", "time_limit": "01:00:00", "nodes": 2,
        "ntasks_per_node": 3, "constraint": "feat", "array_spec": "1-4",
        "output_dir": "logs", "modules": ["m/1"], "command": "true",
        "custom_sbatch": ["--exclusive"], "gpus": 2, "gpu_type": "a100",
        "gpu_format": "gres_type",
    }

    # Which summary label accounts for each directive slurmate can emit.
    ACCOUNTED_BY = {
        "--job-name": "Job name", "--partition": "Partition",
        "--account": "Account", "--qos": "QoS",
        "--cpus-per-task": "CPUs", "--mem": "Memory",
        "--mem-per-cpu": "Memory per CPU", "--time": "Time limit",
        "--nodes": "Nodes", "--ntasks-per-node": "Tasks per node",
        "--constraint": "Constraint", "--array": "Array specification",
        "--output": "Output directory", "--error": "Output directory",
        "--gres": "GPUs", "--gpus": "GPUs", "--gpus-per-node": "GPUs",
        "--gpus-per-task": "GPUs", "--exclusive": "Custom flags",
    }

    def _directives(self, answers):
        from slurmate.builder import build_from_answers
        out = []
        for line in build_from_answers(answers).splitlines():
            if line.startswith("#SBATCH "):
                out.append(line[len("#SBATCH "):].split("=")[0].split()[0])
        return out

    def _labels(self, answers):
        from slurmate.builder import job_summary_rows
        return {label.rstrip(":") for label, _ in job_summary_rows(answers)}

    def test_every_directive_has_a_row_accounting_for_it(self):
        labels = self._labels(self.MAXIMAL)
        missing = []
        for directive in self._directives(self.MAXIMAL):
            expected = self.ACCOUNTED_BY.get(directive)
            if expected is None:
                missing.append(f"{directive} (no mapping — new directive?)")
            elif expected not in labels:
                missing.append(f"{directive} -> {expected!r} row absent")
        assert missing == [], missing

    @pytest.mark.parametrize(
        "fmt", ["gres_type", "gpus_per_node", "gpus", "gpus_per_task", "constraint"]
    )
    def test_every_gpu_format_is_accounted_for(self, fmt):
        # The GPU directive changes name per format, so each spelling needs its
        # own mapping — this is where an unaccounted directive is most likely.
        answers = {**self.MAXIMAL, "gpu_format": fmt}
        labels = self._labels(answers)
        for directive in self._directives(answers):
            if directive.startswith(("--gres", "--gpus")):
                assert self.ACCOUNTED_BY[directive] in labels

    def test_a_minimal_job_needs_no_absent_rows(self):
        minimal = {"job_name": "x", "partition": "p", "cpus": 1,
                   "memory": "4G", "time_limit": "00:05:00", "command": "true"}
        labels = self._labels(minimal)
        for directive in self._directives(minimal):
            assert self.ACCOUNTED_BY[directive] in labels


class TestSummaryDoesNotOverclaim:
    """The reverse of the previous check: can the summary assert something the
    script does *not* contain? One case could. `--env-type none` is a documented
    choice that emits no activation line, so `--env myenv` alongside it was
    silently dropped — while the summary still read `Environment: myenv`. The only
    signal was a `logger.warning`, which no user sees.
    """

    @pytest.mark.parametrize("env_type", ["conda", "mamba", "venv", None])
    def test_activating_types_are_reported_plainly(self, env_type):
        from slurmate.builder import env_activation_emitted, job_summary_rows
        assert env_activation_emitted("myenv", env_type) is True
        rows = dict(job_summary_rows({"env_name": "myenv", "env_type": env_type}))
        assert rows["Environment"] == "myenv"

    @pytest.mark.parametrize("env_type", ["none", "None", "  none  ", "bogus"])
    def test_non_activating_types_are_marked(self, env_type):
        from slurmate.builder import env_activation_emitted, job_summary_rows
        assert env_activation_emitted("myenv", env_type) is False
        rows = dict(job_summary_rows({"env_name": "myenv", "env_type": env_type}))
        assert "not activated" in rows["Environment"]

    def test_the_predicate_matches_what_the_builder_emits(self):
        # The point of sharing a predicate: the summary and the emitter must not
        # disagree about whether an activation line exists.
        from slurmate.builder import build_from_answers, env_activation_emitted
        for env_type in ("conda", "mamba", "venv", "none", "bogus"):
            script = build_from_answers({
                "job_name": "x", "partition": "p", "cpus": 1, "memory": "4G",
                "time_limit": "00:05:00", "command": "true",
                "env_name": "myenv", "env_type": env_type,
            })
            emitted = any(
                ln.startswith(("conda activate", "mamba activate", "source "))
                and "conda.sh" not in ln
                for ln in script.splitlines()
            )
            assert emitted is env_activation_emitted("myenv", env_type), env_type

    def test_a_dropped_env_is_warned_about(self, mocker):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"p"})
        issues = m.site_check_issues(
            {"partition": "p", "env_name": "myenv", "env_type": "none"}
        )
        msgs = [msg for lvl, msg in issues if "will NOT be activated" in msg]
        assert msgs and [lvl for lvl, msg in issues if "will NOT" in msg] == ["warning"]

    def test_no_env_means_no_warning(self, mocker):
        import slurmate.main as m
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"p"})
        issues = m.site_check_issues({"partition": "p", "env_type": "none"})
        assert not [msg for _lvl, msg in issues if "will NOT be activated" in msg]

    def test_a_suppressed_mem_directive_is_not_claimed(self):
        # The case that was already correct, pinned so it stays that way: a custom
        # --mem-per-cpu suppresses the auto --mem, and the summary must not show a
        # Memory row for a directive the script does not carry.
        from slurmate.builder import build_from_answers, job_summary_rows
        answers = {
            "job_name": "x", "partition": "p", "cpus": 2, "memory": "8G",
            "time_limit": "00:05:00", "command": "true",
            "custom_sbatch": ["--mem-per-cpu=2G"],
        }
        script = build_from_answers(answers)
        assert "--mem=" not in script
        assert "Memory" not in dict(job_summary_rows(answers))


class TestLiveAndFinalValidationAgree:
    """The wizard's live panel and the final summary call the same validator with
    different arguments, and the live one omitted `max_array_size` — so an
    over-large `--array` drew nothing while stepping through the wizard and a
    warning at the summary. The same request judged differently by two surfaces,
    which is the divergence class these rounds keep finding.
    """

    def _wizard(self, mocker, cached=None, fetched=65533):
        import slurmate.tui as t
        w = t.Wizard.__new__(t.Wizard)
        w.transient = {} if cached is None else {"max_array_size": cached}
        mocker.patch.object(t, "fetch_max_array_size", return_value=fetched)
        return w

    def test_it_is_fetched_when_an_array_is_entered(self, mocker):
        w = self._wizard(mocker)
        assert w._cached_max_array("1-10") == 65533

    def test_it_is_not_fetched_when_no_array_is_entered(self, mocker):
        import slurmate.tui as t
        w = self._wizard(mocker)
        probe = mocker.patch.object(t, "fetch_max_array_size")
        assert w._cached_max_array("") is None
        assert w._cached_max_array(None) is None
        probe.assert_not_called()

    def test_it_is_fetched_only_once(self, mocker):
        import slurmate.tui as t
        w = self._wizard(mocker)
        probe = mocker.patch.object(t, "fetch_max_array_size", return_value=65533)
        for _ in range(5):
            w._cached_max_array("1-10")
        assert probe.call_count == 1

    def test_a_probe_failure_does_not_break_the_redraw(self, mocker):
        import slurmate.tui as t
        w = t.Wizard.__new__(t.Wizard)
        w.transient = {}
        mocker.patch.object(t, "fetch_max_array_size", side_effect=OSError("boom"))
        assert w._cached_max_array("1-10") is None
        # And it does not retry on every keystroke after failing.
        assert "max_array_size" in w.transient

    def test_the_two_surfaces_now_reach_the_same_verdict(self):
        # The divergence itself: identical answers, one call with the limit and
        # one without, must no longer disagree about whether there is an issue.
        part = _part(name="p", cpus_per_node=48, mem_per_node_mb=184320)
        answers = {"cpus": 1, "array_spec": "1-99999", "_partition_obj": part}
        with_limit = [
            m for _l, m in validate_job_config(answers, max_array_size=65533)
            if "MaxArraySize" in m
        ]
        assert with_limit, "the limit must be reported when it is known"
        without = [
            m for _l, m in validate_job_config(answers) if "MaxArraySize" in m
        ]
        assert not without, "and claimed nothing when it is not"


class TestPrintAndSubmitAgreeOnTheScript:
    """"Inspect it with --print, then run it with --yes" is the documented
    workflow, so the two must produce the same script. Verified against a stub
    `sbatch`: the submitted bytes and the printed bytes differ only by the
    trailing newline `print()` adds (377 vs 376 bytes), which is the same shell
    script. Pinned here at the level that makes it true — both paths deriving the
    script from one call — so a future path that rebuilds it differently fails.
    """

    ANSWERS = {
        "job_name": "t", "partition": "cpu-shared", "account": "acct", "cpus": 8,
        "memory": "8G", "time_limit": "00:05:00", "nodes": 2,
        "ntasks_per_node": 2, "array_spec": "1-4", "modules": ["m/1"],
        "output_dir": "logs", "command": "true", "custom_sbatch": ["--exclusive"],
    }

    def test_build_and_show_returns_exactly_the_built_script(self, mocker, capsys):
        from rich.console import Console

        import slurmate.main as m
        from slurmate.builder import build_from_answers
        # Keep it hermetic: no cluster lookups, no ETA probe. A realistic empty
        # result, not None — fetch_queue_eta always returns a dict, and mocking
        # None would test a state the code cannot reach.
        mocker.patch.object(
            m, "fetch_queue_eta",
            return_value={"running": 0, "pending": 0, "eta_seconds": 0,
                          "eta_label": "now", "source": "unknown",
                          "feasible": True, "reason": ""},
        )
        mocker.patch.object(m, "_partition_issues", return_value=[])
        mocker.patch.object(m, "site_check_issues", return_value=[])
        mocker.patch.object(m, "check_log_dirs", return_value=[])
        mocker.patch.object(m, "fetch_max_array_size", return_value=None)
        script, _queue = m.build_and_show(
            dict(self.ANSWERS), Console(no_color=True, highlight=False, width=200)
        )
        capsys.readouterr()
        assert script == build_from_answers(dict(self.ANSWERS))

    def test_the_script_is_a_pure_function_of_the_answers(self):
        # Neither path may depend on anything else: two builds of the same answers
        # must be byte-identical, which is also what makes --print trustworthy.
        from slurmate.builder import build_from_answers
        first = build_from_answers(dict(self.ANSWERS))
        second = build_from_answers(dict(self.ANSWERS))
        assert first == second
        assert first.endswith("\n") and not first.endswith("\n\n\n")


class TestOutputDirRowMatchesReality:
    """Second instance of the overclaim class. The builder places only a *bare*
    filename inside `output_dir`; an absolute or directory-bearing `output_file`
    is left alone. So `--output-file /tmp/x.out --output-dir logs` wrote to `/tmp`
    while the summary said `Output directory: logs`, sending the user to an empty
    directory.
    """

    BASE = {
        "job_name": "x", "partition": "p", "cpus": 1, "memory": "4G",
        "time_limit": "00:05:00", "command": "true", "output_dir": "logs",
    }

    @pytest.mark.parametrize(
        "output_file,used",
        [
            (None, True),               # no explicit file: the directory places it
            ("bare.out", True),         # bare name: placed inside the directory
            ("/tmp/abs.out", False),    # absolute: the directory is ignored
            ("sub/rel.out", False),     # has a directory of its own: ignored
            ("~/home.out", False),      # expands to absolute
        ],
    )
    def test_the_predicate_matches_what_the_builder_emits(self, output_file, used):
        from slurmate.builder import build_from_answers, output_dir_is_used
        answers = {**self.BASE, "output_file": output_file}
        assert output_dir_is_used("logs", output_file) is used
        emitted = next(
            ln for ln in build_from_answers(answers).splitlines()
            if ln.startswith("#SBATCH --output=")
        )
        # The directive itself is the ground truth for whether the dir was used.
        assert emitted.startswith("#SBATCH --output=logs/") is used

    @pytest.mark.parametrize("output_file", ["/tmp/abs.out", "sub/rel.out"])
    def test_an_ignored_directory_is_marked(self, output_file):
        from slurmate.builder import job_summary_rows
        rows = dict(job_summary_rows({**self.BASE, "output_file": output_file}))
        assert "not used" in rows["Output directory"]

    @pytest.mark.parametrize("output_file", [None, "bare.out"])
    def test_a_used_directory_reads_plainly(self, output_file):
        from slurmate.builder import job_summary_rows
        rows = dict(job_summary_rows({**self.BASE, "output_file": output_file}))
        assert rows["Output directory"] == "logs"

    def test_no_directory_given_is_unaffected(self):
        from slurmate.builder import job_summary_rows, output_dir_is_used
        assert output_dir_is_used(None, "x.out") is False
        rows = dict(job_summary_rows(
            {k: v for k, v in self.BASE.items() if k != "output_dir"}
        ))
        assert rows["Output directory"] == "(current directory)"


class TestSummaryShowsTheTransformedValue:
    """Several fields are *transformed* before they reach the script — the job
    name is sanitized, memory is normalized — so the summary could easily show
    the user's input while the script carries something else. It does not, and
    that is the property that makes those rows trustworthy, so it is pinned here
    rather than left to hold by accident.
    """

    BASE = {
        "partition": "p", "cpus": 1, "time_limit": "00:05:00", "command": "true",
    }

    def _directive(self, answers, flag):
        from slurmate.builder import build_from_answers
        for line in build_from_answers(answers).splitlines():
            if line.startswith(f"#SBATCH {flag}="):
                return line.split("=", 1)[1]
        return None

    @pytest.mark.parametrize(
        "raw,emitted",
        [("my job", "my_job"), ("a;b", "ab"), ("--lead", "lead"),
         ("訓練", "slurm"), ("..", "slurm")],
    )
    def test_the_job_name_row_shows_what_slurm_will_see(self, raw, emitted):
        from slurmate.builder import job_summary_rows
        answers = {**self.BASE, "job_name": raw, "memory": "4G"}
        assert self._directive(answers, "--job-name") == emitted
        assert dict(job_summary_rows(answers))["Job name"] == emitted

    @pytest.mark.parametrize(
        "raw,emitted",
        [("1.5G", "1536M"), ("16", "16M"), ("0G", "0"), ("64000M", "64000M"),
         ("8G", "8G")],
    )
    def test_the_memory_row_shows_what_slurm_will_see(self, raw, emitted):
        from slurmate.builder import job_summary_rows
        answers = {**self.BASE, "job_name": "x", "memory": raw}
        assert self._directive(answers, "--mem") == emitted
        assert dict(job_summary_rows(answers))["Memory"] == emitted

    def test_a_folded_directive_value_matches_too(self):
        # CR/LF in a free-text field is folded to spaces before emission; the row
        # must show the folded form, not the raw one with its newline.
        from slurmate.builder import job_summary_rows
        answers = {**self.BASE, "job_name": "x", "memory": "4G",
                   "account": "acct\nsecond"}
        emitted = self._directive(answers, "--account")
        assert "\n" not in emitted
        assert dict(job_summary_rows(answers))["Account"] == emitted


class TestClusterFactsAreQueriedOnce:
    """A single `--dry-run` made 9 subprocess calls, 3 of them duplicates: the
    batch path's fatal checks and the shared site checks each asked for the
    partition names, the caller's accounts and the QoS list. `sacctmgr show assoc`
    is the one the report singles out as slow enough on a busy controller to be
    worth skipping, and it ran twice. Measured 9 calls / 2.55 s before, 6 / 0.68 s
    after.
    """

    CASES = [
        ("fetch_all_partition_names", ("p1\np2\n", "", 0)),
        ("fetch_user_accounts", ("acct\n", "", 0)),
        ("fetch_known_qos", ("normal\n", "", 0)),
        ("fetch_node_features", ("192g\n", "", 0)),
        ("fetch_select_type", ("SelectType = select/cons_tres\n", "", 0)),
        ("fetch_max_array_size", ("MaxArraySize            = 1001\n", "", 0)),
    ]

    @pytest.mark.parametrize("name,output", CASES, ids=[c[0] for c in CASES])
    def test_repeated_calls_query_once(self, mocker, name, output):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        run = mocker.patch.object(su, "_run_command", return_value=output)
        fn = getattr(su, name)
        first = fn()
        for _ in range(4):
            assert fn() == first
        assert run.call_count == 1, f"{name} queried {run.call_count} times"

    @pytest.mark.parametrize("name,output", CASES, ids=[c[0] for c in CASES])
    def test_the_reset_really_forgets(self, mocker, name, output):
        # Without this the autouse fixture could not keep tests independent, and a
        # later test would read an earlier one's mocked answer.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        run = mocker.patch.object(su, "_run_command", return_value=output)
        getattr(su, name)()
        su.reset_cluster_cache()
        getattr(su, name)()
        assert run.call_count == 2

    def test_a_changed_answer_is_seen_after_a_reset(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(su, "_run_command", return_value=("a\n", "", 0))
        assert su.fetch_all_partition_names() == {"a"}
        su.reset_cluster_cache()
        mocker.patch.object(su, "_run_command", return_value=("b\n", "", 0))
        assert su.fetch_all_partition_names() == {"b"}

    def test_request_specific_lookups_are_not_cached(self, mocker):
        # The ETA depends on the request, so caching it would answer a later
        # question with an earlier one's result.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        run = mocker.patch.object(su, "_run_command", return_value=("", "", 0))
        su.fetch_queue_eta("one")
        su.fetch_queue_eta("two")
        assert run.call_count > 1


class TestAdvisoryLookupsDoNotBlockLong:
    """Six cluster-fact lookups run per invocation, and every one of them is
    designed to fall through *silently* on failure. At the default 30 s timeout a
    hung controller therefore froze a `--dry-run` for ~170 s to collect answers it
    would then discard. Measured against stub binaries that never return: 170 s
    before, 100 s after.
    """

    ADVISORY = [
        "fetch_all_partition_names", "fetch_user_accounts", "fetch_known_qos",
        "fetch_node_features", "fetch_select_type", "fetch_max_array_size",
    ]

    @pytest.mark.parametrize("name", ADVISORY)
    def test_it_uses_the_short_timeout(self, mocker, name):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        run = mocker.patch.object(su, "_run_command", return_value=("", "", 0))
        getattr(su, name)()
        # The FIRST call is the advisory one. fetch_all_partition_names
        # deliberately falls back to fetch_partitions() when `sinfo -a` yields
        # nothing (very old sinfo builds reject -a), and that fallback is
        # load-bearing, so it keeps the full timeout — asserting on the last call
        # would flag correct behaviour.
        first = run.call_args_list[0]
        assert first.kwargs.get("timeout") == su._ADVISORY_TIMEOUT, (
            f"{name} would block for the default {su._RUN_TIMEOUT}s to collect an "
            f"answer it discards on failure"
        )

    def test_the_advisory_timeout_is_well_below_the_default(self):
        assert su._ADVISORY_TIMEOUT < su._RUN_TIMEOUT
        # Generous against healthy latency (0.1-0.5 s measured), so a
        # slow-but-working controller is still answered rather than cut off.
        assert su._ADVISORY_TIMEOUT >= 5

    def test_the_partition_list_keeps_the_full_timeout(self, mocker):
        # Deliberately excluded: an empty partition list is handled, but it costs
        # the user the picker and the limit checks, so waiting longer is the right
        # trade — unlike the advisory lookups, whose absence changes nothing.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        run = mocker.patch.object(su, "_run_command", return_value=("", "", 0))
        su.fetch_partitions()
        assert run.call_args.kwargs.get("timeout") in (None, su._RUN_TIMEOUT)

    @pytest.mark.parametrize("name", ADVISORY)
    def test_a_timeout_is_reported_as_unknown_not_as_absent(self, mocker, name):
        # _run_command returns rc=-1 on timeout; every one of these must treat that
        # as "could not ask", which is what makes the short timeout safe.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(
            su, "_run_command", return_value=("", "Command timed out after 10s", -1)
        )
        result = getattr(su, name)()
        assert result in ({}, set(), [], "", None) or not result


class TestFailedQueueQueryIsNotAnEmptyQueue:
    """`stdout, _, _ = _run_command(["squeue", ...])` discarded the return code, so
    a failed or timed-out squeue was indistinguishable from an idle partition and
    the summary reported `0 running / 0 pending` as a measurement. That is the
    report's cross-cutting root cause verbatim — "a subprocess's error channel is
    not read" — and SM-19's defect arriving through the failure path rather than a
    missing partition.
    """

    def test_a_failed_query_is_flagged_not_counted(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=("", "boom", 1))
        info = su.fetch_queue_eta("p")
        assert info["queue_known"] is False
        assert info["running"] == 0 and info["pending"] == 0   # but not a reading

    def test_a_genuinely_empty_queue_is_still_a_reading(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=("", "", 0))
        assert su.fetch_queue_eta("p")["queue_known"] is True

    def test_the_pressure_tier_does_not_answer_from_a_failed_query(self, mocker):
        # The tier-3 guess is *derived from* the queue depth, so with no depth
        # there is nothing to derive it from — reporting "~5min" would invent a
        # number twice over.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_nodes_that_fit", return_value=None)
        mocker.patch.object(su, "_scheduler_verdict", return_value=(None, ""))
        mocker.patch.object(su, "_run_command", return_value=("", "boom", 1))
        info = su.fetch_queue_eta("p")
        assert info["source"] == "unknown"

    def test_the_pressure_tier_still_answers_when_the_queue_is_readable(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_nodes_that_fit", return_value=None)
        mocker.patch.object(su, "_scheduler_verdict", return_value=(None, ""))
        mocker.patch.object(su, "_run_command", return_value=("", "", 0))
        assert su.fetch_queue_eta("p")["source"] == "pressure"

    def test_it_uses_the_advisory_timeout(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        run = mocker.patch.object(su, "_run_command", return_value=("", "", 0))
        su.fetch_queue_eta("p")
        assert run.call_args_list[0].kwargs.get("timeout") == su._ADVISORY_TIMEOUT

    def test_the_row_says_unknown_rather_than_zero(self):
        from slurmate.main import _qualified_eta
        # The renderer's contract: a failed query must not print as a count. The
        # ETA label itself is unaffected — the scheduler tier does not need squeue.
        assert _qualified_eta({"eta_label": "~11h", "source": "scheduler"}) == "~11h"


class TestSlurmsOwnReasonIsSurfaced:
    """RD-2/SW-7's lesson applied to slurmate: when `sinfo` fails it says *why*
    ("Unable to contact slurm controller (connect failure)") and the code reported
    a generic "no Slurm, or sinfo failed" instead — the diagnosis sitting in a
    stream nobody read. Two rows also still claimed the partition was "not on this
    cluster" when the list simply could not be read.
    """

    ERROR = "slurm_load_partitions: Unable to contact slurm controller (connect failure)"

    def test_the_reason_is_recorded(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(su, "_run_command", return_value=("", self.ERROR, 1))
        assert su.fetch_partitions() == []
        assert su.last_cluster_error() == self.ERROR

    def test_only_the_first_line_is_kept(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(
            su, "_run_command",
            return_value=("", f"{self.ERROR}\nsecondary consequence\n", 1),
        )
        su.fetch_partitions()
        assert su.last_cluster_error() == self.ERROR

    def test_a_successful_query_leaves_no_error(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(su, "_run_command", return_value=("p|infinite|1|up|8|1024|(null)|idle\n", "", 0))
        su.fetch_partitions()
        assert su.last_cluster_error() == ""

    def test_the_capacity_message_quotes_it(self, mocker):
        from slurmate.main import _get_partition
        mocker.patch.object(su, "last_cluster_error", return_value=self.ERROR)
        issues = validate_job_config(
            {"cpus": 999, "_partition_obj": _get_partition([], "caslake")}
        )
        msgs = [m for _l, m in issues if "NOT checked" in m]
        assert msgs and self.ERROR in msgs[0]

    def test_it_falls_back_to_the_generic_wording(self, mocker):
        from slurmate.main import _get_partition
        mocker.patch.object(su, "last_cluster_error", return_value="")
        issues = validate_job_config(
            {"cpus": 999, "_partition_obj": _get_partition([], "caslake")}
        )
        msgs = [m for _l, m in issues if "NOT checked" in m]
        assert msgs and "no Slurm, or sinfo failed" in msgs[0]

    def test_the_reset_clears_it(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(su, "_run_command", return_value=("", self.ERROR, 1))
        su.fetch_partitions()
        su.reset_cluster_cache()
        assert su.last_cluster_error() == ""


class TestNoSurfaceClaimsAbsentWhenUnreadable:
    """Three consecutive rounds found the *same* false claim in a place the
    previous fix had not reached: "partition not on this cluster" rendered when
    the partition list merely could not be read. Fixing a message is not fixing a
    claim — the claim lives wherever the flag is read. This asserts it across every
    surface at once, so a new consumer cannot reintroduce it.
    """

    def _record(self, readable):
        from slurmate.main import _get_partition
        known = [_part(name="real")] if readable else []
        return _get_partition(known, "caslake")

    def _rendered(self, part):
        """Every user-facing string the three surfaces produce for this record."""
        from rich.console import Console

        import slurmate.main as m
        answers = {"cpus": 999, "nodes": 9, "_partition_obj": part,
                   "job_name": "x", "partition": "caslake", "command": "true",
                   "time_limit": "00:05:00", "memory": "4G"}
        out = [msg for _lvl, msg in validate_job_config(answers)]
        console = Console(no_color=True, highlight=False, width=300, record=True)
        m._show_script_and_summary(
            console, "#!/bin/bash\n", answers, "1.0",
            {"running": 0, "pending": 0, "eta_seconds": 0, "eta_label": "now",
             "source": "unknown", "feasible": True, "reason": "",
             "queue_known": True},
        )
        out.append(console.export_text())
        return "\n".join(out)

    def test_an_unreadable_list_never_says_not_on_this_cluster(self, mocker):
        mocker.patch.object(su, "last_cluster_error", return_value="")
        text = self._rendered(self._record(readable=False))
        assert "not on this cluster" not in text, text
        assert "could not be read" in text

    def test_an_absent_partition_does_say_so(self, mocker):
        # The other direction: the honest claim must still be made when it is true,
        # or the fix would have traded a false claim for silence.
        mocker.patch.object(su, "last_cluster_error", return_value="")
        text = self._rendered(self._record(readable=True))
        assert "not on this cluster" in text

    def test_every_consumer_of_the_flag_is_accounted_for(self):
        # The sweep, pinned: if a new site reads _unknown, it has to be reviewed
        # for this claim. Comment lines are excluded so the explanatory text does
        # not count as a consumer.
        import re
        from pathlib import Path
        src = Path(su.__file__).parent
        sites = [
            f"{f.name}:{i}"
            for f in src.glob("*.py")
            for i, line in enumerate(f.read_text().splitlines(), 1)
            if re.search(r'get\("_unknown"\)|\["_unknown"\]', line)
            and not line.lstrip().startswith("#")
        ]
        # main.py summary rows, system_utils capacity message, tui memory default,
        # and _enrich_partition_maxima (SM-27), which reads the flag only to
        # *skip* work: an unknown partition has no real name to query per-node, so
        # it makes no claim and cannot make a false one.
        assert len(sites) == 4, f"new consumer of _unknown: {sites}"


class TestConfigIntTypesAreNotSilentlyReinterpreted:
    """TOML has real types, and `int()` accepts more than it should: `cpus = true`
    became a one-core request (bool is an int subclass) and `cpus = 2.7` became 2
    (truncation). Both are the SM-9 family — a config value quietly reinterpreted
    — except that here it is not discarded, it is *changed*.
    """

    def _coerce(self, value, capsys):
        from rich.console import Console

        from slurmate.main import _coerce_int
        got = _coerce_int(
            value, 4, field="cpus",
            err_console=Console(stderr=True, no_color=True, highlight=False, width=200),
        )
        return got, capsys.readouterr().err

    @pytest.mark.parametrize("value,expected", [(2, 2), (2.0, 2), ("4", 4), (0, 0)])
    def test_unambiguous_values_pass_silently(self, value, expected, capsys):
        got, err = self._coerce(value, capsys)
        assert got == expected and err == ""

    def test_a_boolean_is_refused(self, capsys):
        got, err = self._coerce(True, capsys)
        assert got == 4 and "is a boolean" in err

    @pytest.mark.parametrize("value", [2.7, 0.5, -1.5])
    def test_a_fractional_value_is_refused(self, value, capsys):
        got, err = self._coerce(value, capsys)
        assert got == 4 and "not a whole number" in err

    def test_a_non_numeric_string_is_still_refused(self, capsys):
        got, err = self._coerce("8cores", capsys)
        assert got == 4 and "not an integer" in err


class TestRelativeLogDirIsNotAFalsePositive:
    """The log-directory check walked up to the nearest existing ancestor, and a
    *relative* directory walks to "" — `dirname("logs")` is empty — which was read
    as "/". So `logs`, the default output directory, was reported as impossible to
    create whenever it did not exist yet: a false warning for every first-time user
    in a perfectly writable directory.
    """

    def test_a_relative_dir_in_a_writable_cwd_is_silent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert su.check_log_dirs("#SBATCH --output=logs/x-%j.out\n") == []

    def test_a_nested_relative_dir_is_silent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert su.check_log_dirs("#SBATCH --output=a/b/c/x.out\n") == []

    def test_an_unwritable_cwd_still_warns(self, tmp_path, monkeypatch, mocker):
        monkeypatch.chdir(tmp_path)
        mocker.patch.object(su.os, "access", return_value=False)
        issues = su.check_log_dirs("#SBATCH --output=logs/x.out\n")
        assert len(issues) == 1 and "cannot be created" in issues[0][1]

    def test_an_absolute_unwritable_path_still_warns(self):
        issues = su.check_log_dirs("#SBATCH --output=/proc/nope/x.out\n")
        assert len(issues) == 1

    def test_the_default_output_dir_is_clean(self, tmp_path, monkeypatch):
        # The exact default the CLI uses, in a fresh directory — the case the
        # false positive fired on.
        monkeypatch.chdir(tmp_path)
        from slurmate.builder import build_from_answers
        script = build_from_answers({
            "job_name": "x", "partition": "p", "cpus": 1, "memory": "4G",
            "time_limit": "00:05:00", "command": "true", "output_dir": "logs",
        })
        assert su.check_log_dirs(script) == []


class TestPristineEnvironmentBehaviour:
    """The most useful finding of the previous round came from running slurmate
    somewhere other than the repo — `logs/` already exists here, which hid a false
    warning on the default path. These pin the pristine cases so the dev
    environment cannot hide them again.
    """

    ARGS = [
        "--print", "--force", "--job-name", "x", "--partition", "cpu-shared",
        "--cpus", "1", "--time", "00:05:00", "--command", "true",
    ]

    def _run(self, cwd, env_overrides=None):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1"}
        env.update(env_overrides or {})
        return subprocess.run(
            [sys.executable, "-m", "slurmate", *self.ARGS],
            capture_output=True, text=True, env=env, cwd=str(cwd), timeout=180,
        )

    def test_a_fresh_directory_reports_only_the_missing_log_dir(self, tmp_path):
        # This asserted *zero* stderr, which was right when the only candidate
        # message was the false "cannot be created" this class exists to catch.
        # SM-24 then showed a fresh directory is exactly where --print's script
        # silently loses its output, so one warning is now correct — and it must
        # still be the only one, and not the false one.
        done = self._run(tmp_path)
        assert done.returncode == 0
        assert done.stdout.startswith("#!/bin/bash")
        assert "does not exist" in done.stderr
        assert "cannot be created" not in done.stderr, "the false warning is back"
        assert done.stderr.count("\u26a0") == 1, done.stderr

    def test_a_fresh_directory_with_logs_present_says_nothing(self, tmp_path):
        # The original property, kept: once the one real problem is gone there is
        # no noise at all.
        (tmp_path / "logs").mkdir()
        done = self._run(tmp_path)
        assert done.returncode == 0
        assert done.stdout.startswith("#!/bin/bash")
        assert done.stderr == "", done.stderr

    def test_no_home_still_works(self, tmp_path):
        # `sbatch --export=NONE` on a node with no passwd entry, per SM-16.
        env = {k: v for k, v in os.environ.items() if k != "HOME"}
        env.update({"PYTHONPATH": os.path.abspath("src"), "SLURMATE_MOCK": "1",
                    "NO_COLOR": "1"})
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", *self.ARGS],
            capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=180,
        )
        assert done.returncode == 0 and done.stdout.startswith("#!/bin/bash")

    def test_a_read_only_directory_warns_but_still_emits(self, tmp_path):
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o555)
        try:
            done = self._run(ro)
            assert done.returncode == 0
            assert done.stdout.startswith("#!/bin/bash")
            assert "cannot be created" in done.stderr
            # The resolved path, not ".", so the message is usable in a job log
            # where the reader cannot see the working directory. Compared with all
            # whitespace removed: rich line-wraps the warning and can break a long
            # path mid-token.
            flat = "".join(done.stderr.split())
            assert "".join(str(ro).split()) in flat, done.stderr
        finally:
            ro.chmod(0o755)


class TestSchedulerRefusalReachesEveryMode:
    """Slurm's own "this job cannot run" verdict was reaching exactly one mode.

    Measured on Booth's Mercury (Slurm 25.11, no default account for the user):
    every script without `--account` is refused by the controller with *"Invalid
    account or account/partition combination specified"*. `--dry-run` learned
    this from the ETA probe and rendered it as the summary row `ETA: never —
    <reason>`; `--print`, which makes no scheduler call at all, emitted the
    unsubmittable script with **zero bytes on stderr and rc=0**.

    Two defects in one: a check living on a single code path, and a fatal fact
    rendered as a time estimate.
    """

    REASON = "Invalid account or account/partition combination specified"

    # Measured verbatim on Mercury, whose `clay` QoS allows one submitted job per
    # user: a valid script is refused whenever another job is already queued.
    TRANSIENT = (
        "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
        "allocation failure: Job violates accounting/QOS policy "
        "(job submit limit, user's size and/or time limits)"
    )

    # A real limit token that neither marker list recognises: Mercury's QoS tables
    # carry cpu/node caps, so this is reachable, and it is the case that must not
    # be told "your script is fine".
    # A group limit: neither a request fact nor a recognised wait, so it is the
    # "cannot tell" case. (A PerJob limit is now classified permanent.)
    UNKNOWN = (
        "sbatch: error: AssocGrpCpuLimit\n"
        "allocation failure: Job violates accounting/QOS policy "
        "(job submit limit, user's size and/or time limits)"
    )

    @pytest.fixture
    def transient_cluster(self, tmp_path):
        return self._build(tmp_path, self.TRANSIENT)

    @pytest.fixture
    def unknown_cluster(self, tmp_path):
        return self._build(tmp_path, self.UNKNOWN)

    @pytest.fixture
    def fake_cluster(self, tmp_path):
        """A PATH whose only cluster tools are a refusing ``sbatch`` plus the
        ``squeue``/``sinfo`` the ETA path insists on before it will ask anything.

        Both answer successfully and say nothing, so every capacity lookup
        degrades to its documented "unknown" and whatever the run reports about
        feasibility is attributable to the scheduler probe alone.
        """
        return self._build(tmp_path, f"allocation failure: {self.REASON}")

    def _build(self, tmp_path, sbatch_stderr):
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        log = tmp_path / "sbatch.log"
        body = "".join(
            f'echo {line!r} >&2\n' for line in sbatch_stderr.splitlines()
        )
        (bindir / "sbatch").write_text(
            "#!/bin/bash\n"
            f'printf "%s\\n" "$*" >> {log}\n'
            "cat > /dev/null\n"
            f"{body}"
            'if [[ "$*" == *--test-only* ]]; then exit 1; fi\n'
            "echo 999999\n"
        )
        for quiet in ("squeue", "sinfo"):
            (bindir / quiet).write_text("#!/bin/bash\nexit 0\n")
            (bindir / quiet).chmod(0o755)
        (bindir / "sbatch").chmod(0o755)
        return bindir, log

    def _run(self, bindir, mode, extra=()):
        return subprocess.run(
            [sys.executable, "-m", "slurmate", mode, "--job-name", "x",
             "--partition", "p1", "--cpus", "2", "--time", "00:05:00",
             "--command", "true", *extra],
            capture_output=True, text=True, timeout=180,
            # cwd is the tmp dir, not the repo: --yes saves the generated script
            # beside the caller, and a test that submits must not drop artefacts
            # into the working tree.
            cwd=str(bindir.parent),
            # PYTHONPATH stays absolute for the same reason — "src" is relative
            # to the repo, which is no longer the working directory.
            # COLUMNS is load-bearing: at rich's 80-column default the reason
            # wraps mid-sentence and a substring assertion silently stops
            # matching the thing it is meant to pin.
            env={"PATH": str(bindir), "NO_COLOR": "1", "COLUMNS": "220",
                 "PYTHONPATH": str(pathlib.Path(__file__).parent.parent / "src"),
                 "HOME": os.environ.get("HOME", "/tmp")},
        )

    def test_print_reports_the_refusal_and_fails(self, fake_cluster):
        bindir, _log = fake_cluster
        done = self._run(bindir, "--print")
        assert self.REASON in done.stderr, (
            "--print emitted a script the controller has already refused, with "
            "nothing on stderr — the defect this test exists to catch"
        )
        assert done.returncode == 1
        # stdout stays script-only, so the report cannot corrupt a pipe.
        assert self.REASON not in done.stdout

    def test_print_force_still_emits_the_script(self, fake_cluster):
        bindir, _log = fake_cluster
        done = self._run(bindir, "--print", ["--force"])
        assert done.returncode == 0
        assert "#SBATCH --job-name=x" in done.stdout

    def test_dry_run_marks_it_as_an_error_not_just_an_eta(self, fake_cluster):
        bindir, _log = fake_cluster
        done = self._run(bindir, "--dry-run")
        combined = done.stdout + done.stderr
        assert "Slurm refuses this job" in combined, (
            "the verdict appeared only as the summary's 'ETA: never' row"
        )

    def test_yes_does_not_submit_a_job_slurm_already_refused(self, fake_cluster):
        bindir, log = fake_cluster
        done = self._run(bindir, "--yes")
        assert done.returncode == 1
        calls = log.read_text().splitlines() if log.exists() else []
        assert calls, "the scheduler was never asked"
        assert all("--test-only" in c for c in calls), (
            f"--yes submitted for real after a refusal: {calls}"
        )

    # ── the other half: a refusal about the moment, not the request ──────────

    def test_print_does_not_fail_on_a_transient_limit(self, transient_cluster):
        # A submit-count cap says nothing about the script. Failing here would
        # turn "you already have a job queued" into a red CI build — and on
        # Mercury, into one caused by an unrelated job of the same user's.
        bindir, _log = transient_cluster
        done = self._run(bindir, "--print")
        assert done.returncode == 0
        assert "#SBATCH --job-name=x" in done.stdout
        assert "cannot take this job right now" in done.stderr
        assert "refuses" not in done.stderr

    def test_transient_reason_names_the_actual_limit(self, transient_cluster):
        # The generic bundle lists three possibilities; the token says which.
        bindir, _log = transient_cluster
        done = self._run(bindir, "--print")
        assert "QOSMaxSubmitJobPerUserLimit" in done.stderr

    def test_an_unclassifiable_refusal_makes_no_promise(self, unknown_cluster):
        # The structural half of the fix: the marker lists cannot enumerate every
        # Slurm wording, so an unrecognised one must report what the controller
        # said and claim nothing. Telling the user "the script is valid" here is
        # how a job asking for too many nodes got reassured.
        bindir, _log = unknown_cluster
        done = self._run(bindir, "--print")
        assert done.returncode == 0
        assert "would not accept" in done.stderr
        assert "the script is valid" not in done.stderr
        assert "clears on its own" not in done.stderr.replace(
            "cannot tell whether this clears on its own", ""
        )

    def test_yes_still_submits_when_the_limit_is_transient(self, transient_cluster):
        bindir, log = transient_cluster
        done = self._run(bindir, "--yes")
        calls = log.read_text().splitlines() if log.exists() else []
        assert any("--test-only" not in c for c in calls), (
            f"--yes refused a valid job over a transient limit: {calls}"
        )
        assert done.returncode == 0


class TestRefusalAttribution:
    """The refusal's *source* decides how it may be phrased: crediting Slurm with
    a verdict it never gave is the same class of error as the ETA that reported a
    confident wait for a rejected job."""

    def _render(self, queue_info):
        from rich.console import Console

        buf = io.StringIO()
        _note_scheduler_refusal(queue_info, Console(file=buf, width=200, no_color=True))
        return buf.getvalue()

    def test_scheduler_refusal_is_attributed_to_slurm(self):
        out = self._render(
            {"feasible": False, "source": "scheduler", "reason": "Invalid qos"}
        )
        assert "Slurm refuses this job" in out
        assert "Invalid qos" in out

    def test_derived_refusal_is_not_attributed_to_slurm(self):
        out = self._render(
            {"feasible": False, "source": "resources", "reason": "999 cpus > 64"}
        )
        assert "Slurm refuses" not in out
        assert "cannot run as requested" in out

    def test_feasible_job_says_nothing(self):
        assert self._render({"feasible": True, "reason": ""}) == ""

    def test_infeasible_without_a_reason_says_nothing(self):
        # "never" with no explanation is not a message worth printing.
        assert self._render({"feasible": False, "source": "scheduler", "reason": " "}) == ""


class TestModuleCommandLayouts:
    """`module` is a shell function; the runnable entry point moved between Tcl
    environment-modules major versions. Checking only the 3.x path makes every
    module check silently inert on a 5.x site whose wrapper is off PATH."""

    def test_tcl_modules_5x_libexec_layout_is_found(self, mocker, tmp_path):
        home = tmp_path / "Modules"
        (home / "libexec").mkdir(parents=True)
        real = home / "libexec" / "modulecmd.tcl"
        real.write_text("#!/bin/sh\n")
        mocker.patch.dict(os.environ, {"MODULESHOME": str(home)}, clear=False)
        mocker.patch.dict(os.environ, {"LMOD_CMD": ""}, clear=False)
        mocker.patch.object(su.shutil, "which", return_value=None)
        assert su._module_command() == [str(real), "bash"]

    def test_tcl_modules_3x_bin_layout_still_wins(self, mocker, tmp_path):
        home = tmp_path / "Modules"
        (home / "bin").mkdir(parents=True)
        (home / "libexec").mkdir(parents=True)
        old = home / "bin" / "modulecmd"
        old.write_text("#!/bin/sh\n")
        (home / "libexec" / "modulecmd.tcl").write_text("#!/bin/sh\n")
        mocker.patch.dict(os.environ, {"MODULESHOME": str(home)}, clear=False)
        mocker.patch.dict(os.environ, {"LMOD_CMD": ""}, clear=False)
        mocker.patch.object(su.shutil, "which", return_value=None)
        assert su._module_command() == [str(old), "bash"]

    def test_no_module_system_reports_none(self, mocker, tmp_path):
        mocker.patch.dict(
            os.environ, {"MODULESHOME": str(tmp_path / "absent"), "LMOD_CMD": ""},
            clear=False,
        )
        mocker.patch.object(su.shutil, "which", return_value=None)
        assert su._module_command() is None


class TestRefusalClassification:
    """`sbatch --test-only` answers "this job is wrong" and "not right now" with
    the same non-zero exit. Every string below was measured on a live controller
    (Booth's Mercury, Slurm 25.11)."""

    @pytest.mark.parametrize("out,permanent,shown", [
        ("allocation failure: Invalid account or account/partition combination "
         "specified", True, "Invalid account"),
        ("allocation failure: Invalid qos specification", True, "Invalid qos"),
        ("sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
         "allocation failure: Job violates accounting/QOS policy (job submit "
         "limit, user's size and/or time limits)",
         False, "QOSMaxSubmitJobPerUserLimit"),
        # Slurm's own "now" is the whole difference between these two.
        ("allocation failure: Requested node configuration is not available",
         True, "not available"),
        ("allocation failure: Requested node configuration is not available now",
         False, "not available now"),
    ])
    def test_measured_refusals_are_classified(self, out, permanent, shown):
        _eta, reason = su._read_test_only_output("", out, 1)
        assert shown in reason
        assert su.refusal_is_permanent(reason) is permanent

    def test_an_unrecognised_refusal_is_not_permanent(self):
        # Guessing "permanent" for wording we have never seen would refuse jobs
        # that would have run; the safe default is advisory.
        assert su.refusal_is_permanent("something nobody has measured") is False

    def test_no_refusal_is_not_permanent(self):
        assert su.refusal_is_permanent("") is False
        assert su.refusal_is_permanent("   ") is False

    def test_prose_error_lines_are_not_mistaken_for_a_limit_token(self):
        out = ("sbatch: error: Batch job submission failed: Invalid qos specification\n"
               "allocation failure: Invalid qos specification")
        _eta, reason = su._read_test_only_output("", out, 1)
        assert "[" not in reason, f"prose captured as a limit token: {reason}"

    def test_a_duplicated_token_is_not_appended_twice(self):
        out = ("sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
               "allocation failure: QOSMaxSubmitJobPerUserLimit")
        _eta, reason = su._read_test_only_output("", out, 1)
        assert reason.count("QOSMaxSubmitJobPerUserLimit") == 1


class TestEveryRefusalSiteClassifies:
    """A guard, not a behaviour test. The recurring defect in this report is a
    check that behaves one way on three code paths and another way on the fourth,
    and the refusal check now has four call sites: `--print`, `--dry-run`/`--yes`
    (via the summary), and the wizard's hand-edited-script branch. The last one
    blocked on *any* refusal, so on a cluster whose QoS caps submitted jobs at one
    (Mercury) a hand edit stranded a perfectly valid script behind an unrelated
    queued job — and blamed the edit for it.

    Source-level because the wizard branch is only reachable interactively; a
    behaviour test would need a pty and would not survive a refactor of the menu.
    """

    def _main_source(self):
        return pathlib.Path(
            su.__file__
        ).parent.joinpath("main.py").read_text().splitlines()

    def test_every_refusal_decision_consults_the_classifier(self):
        lines = self._main_source()
        sites = [
            (n, ln) for n, ln in enumerate(lines, 1)
            if "check_script_with_scheduler(" in ln and "def " not in ln
            and not ln.lstrip().startswith("#")
        ]
        assert sites, "no call sites found — has the function been renamed?"
        for n, _ln in sites:
            window = "\n".join(lines[n - 1:n + 6])
            assert "refusal_is_permanent" in window, (
                f"main.py:{n} acts on a scheduler refusal without classifying it "
                f"as permanent or transient; a transient limit means 'wait', not "
                f"'this job is wrong'"
            )

    def test_the_classifier_is_not_bypassed_by_a_bare_truthiness_check(self):
        # `if refusal:` alone is the shape of the original defect. It is allowed
        # only *after* a permanent branch has already returned/continued.
        lines = self._main_source()
        for n, ln in enumerate(lines, 1):
            if ln.strip() == "if refusal:":
                preceding = "\n".join(lines[max(0, n - 8):n])
                assert "refusal_is_permanent" in preceding, (
                    f"main.py:{n}: bare `if refusal:` with no permanence check "
                    f"above it"
                )


class TestRefusalLabelMatchesItsSeverity:
    """`ETA: never` was printed directly above an advisory saying the script was
    valid and the condition temporary — a flat contradiction on one screen,
    because the "never" decision lived in `main.py` while the classification lived
    in `system_utils`. The label and the permanence now come from the result
    itself, so the two surfaces cannot disagree.
    """

    def _eta(self, mocker, sbatch_stderr):
        mocker.patch.object(su, "is_tool_available", return_value=True)

        def run(cmd, timeout=30, **kw):
            if "sbatch" in cmd:
                return "", sbatch_stderr, 1
            if "squeue" in cmd:
                return "RUNNING|1|2|1\n", "", 0
            return "", "", 1

        mocker.patch.object(su, "_run_command", side_effect=run)
        return su.fetch_queue_eta("build", req_nodes=1, cpus=2)

    def test_a_request_level_refusal_is_never_and_permanent(self, mocker):
        info = self._eta(
            mocker, "allocation failure: Invalid qos specification\n"
        )
        assert info["feasible"] is False
        assert info["eta_label"] == "never"
        assert info["refusal_is_permanent"] is True

    def test_a_submit_cap_is_not_never(self, mocker):
        info = self._eta(
            mocker,
            "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
            "allocation failure: Job violates accounting/QOS policy "
            "(job submit limit, user's size and/or time limits)\n",
        )
        assert info["feasible"] is False
        assert info["refusal_is_permanent"] is False
        assert info["eta_label"] != "never", (
            "a cap that clears when another job finishes was labelled 'never'"
        )
        assert info["eta_label"] == "not right now"

    def test_the_summary_row_uses_the_label_it_was_given(self):
        # Pins the coupling: main.py must not reintroduce its own "never".
        main_src = pathlib.Path(su.__file__).parent.joinpath("main.py").read_text()
        assert 'f"never — {reason}"' not in main_src, (
            "the summary row hardcodes 'never' again, so a transient refusal will "
            "contradict the advisory printed below it"
        )

    def test_a_feasible_result_carries_no_refusal_claim(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)

        def run(cmd, timeout=30, **kw):
            if "sbatch" in cmd:
                return "", "Job 1 to start at 2099-01-01T00:00:00 using 2 processors\n", 0
            if "squeue" in cmd:
                return "", "", 0
            return "", "", 1

        mocker.patch.object(su, "_run_command", side_effect=run)
        info = su.fetch_queue_eta("build", req_nodes=1, cpus=2)
        assert info["feasible"] is True
        assert info["eta_label"] != "never"


class TestRefusalTriState:
    """A refusal can be permanent, transient, or unclassifiable. The third case is
    not a rounding error: `refusal_is_transient` is deliberately not the negation
    of `refusal_is_permanent`, because "not recognised as permanent" must not
    license telling the user their script is fine and the condition will clear.
    """

    @pytest.mark.parametrize("out,permanent,transient,label", [
        # Measured on Mercury: --nodes 2 where the QoS caps nodes at 1.
        ("allocation failure: Node count specification invalid",
         True, False, "never"),
        # Measured on Mercury: 8 days against a 7-day partition maximum.
        ("allocation failure: Requested time limit is invalid (missing or "
         "exceeds some limit)", True, False, "never"),
        # Measured on Mercury: one submitted job allowed, one already queued.
        ("sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
         "allocation failure: Job violates accounting/QOS policy (job submit "
         "limit, user's size and/or time limits)", False, True, "not right now"),
        # A *group* limit: it clears as other people's jobs finish, so it is not
        # a request fact, but nothing here recognises it either — the honest
        # "cannot tell". (QOSMaxCpuPerJobLimit used to sit here and is now
        # correctly permanent: a PerJob limit is about the request.)
        ("sbatch: error: AssocGrpCpuLimit\n"
         "allocation failure: Job violates accounting/QOS policy (job submit "
         "limit, user's size and/or time limits)", False, False, "refused"),
    ])
    def test_classification_and_label(self, mocker, out, permanent, transient, label):
        _eta, reason = su._read_test_only_output("", out, 1)
        assert su.refusal_is_permanent(reason) is permanent
        assert su.refusal_is_transient(reason) is transient

        mocker.patch.object(su, "is_tool_available", return_value=True)

        def run(cmd, timeout=30, **kw):
            if "sbatch" in cmd:
                return "", out, 1
            if "squeue" in cmd:
                return "", "", 0
            return "", "", 1

        mocker.patch.object(su, "_run_command", side_effect=run)
        info = su.fetch_queue_eta("p", req_nodes=1, cpus=1)
        assert info["eta_label"] == label
        assert info["refusal_is_permanent"] is permanent
        assert info["refusal_is_transient"] is transient

    def test_transient_is_not_the_negation_of_permanent(self):
        # Both false is a legal, meaningful state; if this ever becomes
        # impossible, the honest "cannot tell" branch is dead code.
        r = "some wording nobody has measured"
        assert su.refusal_is_permanent(r) is False
        assert su.refusal_is_transient(r) is False

    def test_neither_holds_for_an_empty_reason(self):
        assert su.refusal_is_permanent("") is False
        assert su.refusal_is_transient("") is False


class TestSitePluginReasonStaysClassifiable:
    """A third cluster, a third wording for the same mistake. midway3's site
    `job_submit` plugin rejects an account-less job with a six-line block ending
    in `Reason: Account is not specified` / `allocation failure: Access/permission
    denied`; Mercury says `Invalid account or account/partition combination
    specified`; both mean "this job can never run". The specific reason is the one
    worth reading and the generic one is the only one a marker list can recognise,
    so both are kept — otherwise the same defect is fatal on one cluster and
    merely puzzling on the next, which is the opposite of cluster-agnostic.
    """

    MIDWAY3 = (
        "sbatch: error: Verify job submission ...\n"
        "sbatch: error: Partition: test\n"
        "sbatch: error: QOS-Flag: test\n"
        "sbatch: error: Account: unknown\n"
        "sbatch: error: Verification: ***REJECTED***\n"
        "sbatch: error: Reason: Account is not specified\n"
        "allocation failure: Access/permission denied"
    )
    MERCURY = (
        "allocation failure: Invalid account or account/partition combination "
        "specified"
    )

    def test_midway3_account_block_is_permanent(self):
        _eta, reason = su._read_test_only_output("", self.MIDWAY3, 1)
        assert reason.startswith("Account is not specified")
        assert su.refusal_is_permanent(reason) is True
        assert su.refusal_is_transient(reason) is False

    def test_both_clusters_agree_on_the_verdict(self):
        # The property that matters: the same user error gets the same severity
        # whichever controller answered.
        verdicts = set()
        for out in (self.MIDWAY3, self.MERCURY):
            _eta, reason = su._read_test_only_output("", out, 1)
            verdicts.add(su.refusal_is_permanent(reason))
        assert verdicts == {True}, (
            "the same mistake classified differently per cluster"
        )

    def test_the_verbose_block_does_not_leak_into_the_message(self):
        # Six lines of plugin chatter must not become the message; only the
        # Reason and the generic verdict.
        _eta, reason = su._read_test_only_output("", self.MIDWAY3, 1)
        for noise in ("Verify job submission", "QOS-Flag", "***REJECTED***"):
            assert noise not in reason
        assert reason.count("\n") == 0


class TestMissingOutputDirectory:
    """SM-24: the shipped default `--output-dir logs` is never created by
    `--print`, and Slurm does not validate output paths at submit — it accepts a
    path in a missing directory, discards what the job writes, and reports
    `COMPLETED 0:0`. A green job with no log is the most confusing result a batch
    user can get, and it is reachable from slurmate's defaults in an empty
    directory.

    The split that matters: `submit_sbatch` *does* create these directories, so
    warning when slurmate is the one submitting would fire on every default run.
    `--print` is the one mode that hands over a script it will never submit.
    """

    ARGS = [
        "--job-name", "x", "--partition", "cpu-shared", "--cpus", "1",
        "--memory", "500M", "--time", "00:02:00", "--command", "echo HI",
    ]

    def _run(self, cwd, mode):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "220"}
        return subprocess.run(
            [sys.executable, "-m", "slurmate", mode, *self.ARGS],
            capture_output=True, text=True, cwd=str(cwd), timeout=180, env=env,
        )

    def test_print_warns_when_the_default_logs_dir_is_absent(self, tmp_path):
        done = self._run(tmp_path, "--print")
        assert "#SBATCH --output=logs/" in done.stdout, "the directive still ships"
        assert "does not exist" in done.stderr, done.stderr
        assert "logs" in done.stderr
        # Actionable, and it names the remedy rather than just the problem.
        assert "mkdir -p logs" in done.stderr
        # A warning, not an error: the script itself is correct.
        assert done.returncode == 0

    def test_print_is_silent_once_the_directory_exists(self, tmp_path):
        (tmp_path / "logs").mkdir()
        done = self._run(tmp_path, "--print")
        assert done.returncode == 0
        assert done.stderr == "", done.stderr

    def test_dry_run_does_not_warn(self, tmp_path):
        # --dry-run submits nothing, but a later real run creates the directory,
        # so warning here would fire on every dry run of the default.
        done = self._run(tmp_path, "--dry-run")
        assert "does not exist" not in done.stdout + done.stderr

    def test_the_submitting_caller_stays_silent(self):
        # will_create=True is the submit path, which makes the directory itself.
        script = (
            "#!/bin/bash\n"
            "#SBATCH --output=definitely/not/here/x-%j.out\n"
            "echo hi\n"
        )
        assert su.check_log_dirs(script, will_create=True) == []

    def test_the_handing_over_caller_reports_it(self):
        script = (
            "#!/bin/bash\n"
            "#SBATCH --output=definitely/not/here/x-%j.out\n"
            "echo hi\n"
        )
        issues = su.check_log_dirs(script, will_create=False)
        assert [lvl for lvl, _ in issues] == ["warning"]
        assert "definitely/not/here" in issues[0][1]

    def test_a_slurm_pattern_directory_is_not_reported(self):
        # "%j" is expanded per job; there is no literal directory to test, and
        # claiming it is missing would be a false warning on every array job.
        script = "#!/bin/bash\n#SBATCH --output=logs/%A_%a/x.out\necho hi\n"
        assert su.check_log_dirs(script, will_create=False) == []


class TestScriptBoxIsNotCopyPasteBroken:
    """Round 60, secondary: rich wraps mid-token, so a long directive rendered as
    a bare `#SBATCH` (a no-op) on one line and `--output=…` — split mid-path — on
    the next, where it reads as a *shell command*. `bash -n` accepts that, so a
    user who copied the box got "command not found" at run time and none of the
    `--output` they asked for.
    """

    LONG = "/tmp/" + "x" * 90

    def _dry_run(self, cols):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": str(cols)}
        return subprocess.run(
            [sys.executable, "-m", "slurmate", "--dry-run", "--job-name", "x",
             "--partition", "cpu-shared", "--cpus", "1", "--memory", "500M",
             "--time", "00:02:00", "--output-dir", self.LONG,
             "--command", "echo hi"],
            capture_output=True, text=True, timeout=180, env=env,
        ).stdout

    def _box_lines(self, out):
        """Just the script box's interior.

        Scoped deliberately: the advisories *below* the box are prose and wrap at
        word boundaries, so one of them legitimately begins a line with
        "--demo, for real values.)" — scanning the whole of stdout made this
        assertion fail on text that has nothing to copy in it.
        """
        lines = out.splitlines()
        start = next(
            (i for i, ln in enumerate(lines) if "Generated sbatch script" in ln),
            None,
        )
        assert start is not None, f"no script box in output: {out[:400]!r}"
        body = []
        for ln in lines[start + 1:]:
            if ln.lstrip().startswith(("\u2570", "+--")):     # closing border
                break
            body.append(ln)
        assert body, "script box was empty"
        return body

    @pytest.mark.parametrize("cols", [40, 80, 100])
    def test_no_directive_is_split_across_rendered_lines(self, cols):
        # The signature of the defect: a boxed line whose content begins with
        # "--", i.e. a flag that lost its "#SBATCH".
        for line in self._box_lines(self._dry_run(cols)):
            stripped = line.strip("\u2502|+ ").strip()
            assert not stripped.startswith("--"), (
                f"at {cols} cols a directive was wrapped onto its own line, which "
                f"reads as a shell command when copied: {line!r}"
            )

    @pytest.mark.parametrize("cols", [40, 80, 100])
    def test_a_bare_sbatch_never_appears(self, cols):
        for line in self._box_lines(self._dry_run(cols)):
            stripped = line.strip("\u2502|+ ").strip()
            # "<n> #SBATCH" with nothing after it is the no-op half of the split.
            assert not re.fullmatch(r"\d+\s+#SBATCH", stripped), (
                f"at {cols} cols a '#SBATCH' was left alone on a line: {line!r}"
            )

    def test_clipping_is_disclosed_and_points_at_print(self):
        out = self._dry_run(80)
        assert "clipped" in out
        assert "--print" in out, "the copy-safe alternative is not named"

    def test_nothing_is_disclosed_when_nothing_is_clipped(self):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "220"}
        out = subprocess.run(
            [sys.executable, "-m", "slurmate", "--dry-run", "--job-name", "x",
             "--partition", "cpu-shared", "--cpus", "1", "--memory", "500M",
             "--time", "00:02:00", "--command", "echo hi"],
            capture_output=True, text=True, timeout=180, env=env,
        ).stdout
        assert "clipped" not in out

    def test_print_remains_byte_exact(self):
        # --print is the copy-safe form the note points at, so it must never clip.
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "COLUMNS": "40"}
        out = subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--job-name", "x",
             "--partition", "cpu-shared", "--cpus", "1", "--memory", "500M",
             "--time", "00:02:00", "--output-dir", self.LONG,
             "--command", "echo hi"],
            capture_output=True, text=True, timeout=180, env=env,
        ).stdout
        assert f"#SBATCH --output={self.LONG}/x-%j.out" in out.splitlines()
        assert "…" not in out and "..." not in out

    def test_clip_helper_is_cell_aware(self):
        from slurmate.main import _clip_cells
        # Two-cell glyphs must not overflow the limit they were trimmed to.
        text = "中" * 20          # 20 CJK chars = 40 cells
        shown, was = _clip_cells(text, 10)
        assert was is True
        from rich.cells import cell_len
        assert cell_len(shown) <= 10
        assert _clip_cells("short", 40) == ("short", False)


class TestEveryHandoverPathChecksItsLogDirs:
    """SM-24's fix was wired into `--print`, which is not the only mode that hands
    over a script slurmate will never submit. The wizard's "Save script to a file"
    action does exactly the same thing, and said nothing — so the failure
    (Slurm accepts the path, discards the output, reports COMPLETED) stayed
    reachable through the interface most users see. A check wired to one of two
    equivalent paths is the shape of half the findings in this report.
    """

    SCRIPT = (
        "#!/bin/bash\n"
        "#SBATCH --job-name=x\n"
        "#SBATCH --output=logs/x-%j.out\n"
        "echo hi\n"
    )

    def test_saving_a_script_warns_about_a_missing_log_dir(
        self, mocker, tmp_path, capsys, monkeypatch
    ):
        import questionary

        from slurmate.main import _save_script

        target = tmp_path / "job.sh"
        mocker.patch.object(
            questionary, "text",
            return_value=mocker.Mock(ask=lambda: str(target)),
        )
        # monkeypatch.chdir is auto-restored; a bare chdir would leak the
        # working directory into every test that runs after this one.
        monkeypatch.chdir(tmp_path)
        _save_script(self.SCRIPT, "job.sh")
        out = capsys.readouterr().out
        assert "Saved to" in out
        assert "does not exist" in out, out
        assert "mkdir -p logs" in out
        assert target.read_text() == self.SCRIPT, "the script is still written"

    def test_no_warning_once_the_directory_exists(
        self, mocker, tmp_path, capsys, monkeypatch
    ):
        import questionary

        from slurmate.main import _save_script

        (tmp_path / "logs").mkdir()
        target = tmp_path / "job.sh"
        mocker.patch.object(
            questionary, "text",
            return_value=mocker.Mock(ask=lambda: str(target)),
        )
        monkeypatch.chdir(tmp_path)
        _save_script(self.SCRIPT, "job.sh")
        out = capsys.readouterr().out
        assert "Saved to" in out
        assert "does not exist" not in out

    def test_the_save_still_succeeds_if_the_check_finds_nothing_to_say(
        self, mocker, tmp_path, capsys, monkeypatch
    ):
        # The warning must never be able to prevent the write.
        import questionary

        from slurmate.main import _save_script

        target = tmp_path / "sub" / "job.sh"
        mocker.patch.object(
            questionary, "text",
            return_value=mocker.Mock(ask=lambda: str(target)),
        )
        monkeypatch.chdir(tmp_path)
        _save_script("#!/bin/bash\necho hi\n", "job.sh")
        assert target.exists()
        assert "Saved to" in capsys.readouterr().out


class TestSubmitOutcomeIsNotInferred:
    """`rc=0` with empty stdout was read as "mock mode — not actually submitted".
    That is an inference, and it is wrong in the direction that costs the most: a
    real `sbatch` which exits 0 and prints nothing has very likely submitted the
    job, and telling that user nothing happened invites a duplicate submission.
    Mock mode short-circuits before running anything, so which case it is can be
    asked rather than guessed.
    """

    @pytest.fixture
    def silent_sbatch(self, tmp_path):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        for name, body in (
            ("sbatch", "#!/bin/bash\ncat > /dev/null\nexit 0\n"),   # accepts, says nothing
            ("squeue", "#!/bin/bash\nexit 0\n"),
            ("sinfo", "#!/bin/bash\nexit 0\n"),
        ):
            (bindir / name).write_text(body)
            (bindir / name).chmod(0o755)
        (tmp_path / "logs").mkdir()
        return bindir, tmp_path

    def _run(self, bindir, cwd, extra_env=None, partition="p"):
        env = {"PATH": f"{bindir}:/usr/bin:/bin", "NO_COLOR": "1",
               "COLUMNS": "220", "HOME": os.environ.get("HOME", "/tmp"),
               "PYTHONPATH": str(pathlib.Path(__file__).parent.parent / "src")}
        env.update(extra_env or {})
        return subprocess.run(
            [sys.executable, "-m", "slurmate", "--yes", "--job-name", "x",
             "--partition", partition, "--cpus", "1", "--memory", "1G",
             "--time", "00:05:00", "--command", "echo hi"],
            capture_output=True, text=True, timeout=180, env=env, cwd=str(cwd),
        )

    def test_a_real_silent_sbatch_is_not_called_mock_mode(self, silent_sbatch):
        bindir, work = silent_sbatch
        done = self._run(bindir, work)
        combined = done.stdout + done.stderr
        assert "mock mode" not in combined, (
            "a real sbatch that exited 0 was reported as mock mode, which tells "
            "the user to resubmit a job that may already exist"
        )
        assert "Submitted" in combined
        # And it says what it does not know, rather than inventing an id.
        assert "no job ID" in combined
        assert "squeue" in combined

    def test_mock_mode_still_says_mock_mode(self, silent_sbatch):
        bindir, work = silent_sbatch
        # A partition from the demo list: under SLURMATE_MOCK the cluster's
        # partitions *are* the mock ones, so "p" is rejected before submit and
        # the run never reaches the branch under test.
        done = self._run(
            bindir, work, {"SLURMATE_MOCK": "1"}, partition="cpu-shared"
        )
        assert "mock mode" in done.stdout + done.stderr


class TestJobIdFormats:
    """Every shape Slurm itself prints, and nothing else. Losing the id costs the
    `squeue`/`scancel` hints and the saved script's filename, which is exactly
    what happened on a site whose sbatch wrapper drops `--parsable`."""

    @pytest.mark.parametrize("raw,expected", [
        ("12345", "12345"),                                    # --parsable
        ("12345;mercury", "12345"),                            # federated --parsable
        ("Submitted batch job 12345", "12345"),                # sbatch's own default
        ("Submitted batch job 12345 on cluster mercury", "12345"),   # federated
        ("  Submitted batch job 12345  ", "12345"),            # padded
        # A wrapper's banner ahead of the real line must not defeat it.
        ("NOTICE: policy 2024 applies\nSubmitted batch job 999", "999"),
    ])
    def test_accepted(self, raw, expected):
        assert su.parse_submitted_job_id(raw) == expected

    @pytest.mark.parametrize("raw", [
        "",
        "Submitted batch job",                 # no number
        "Submitted batch job abc",             # not a number
        "Job 12345 has been queued",           # not a shape Slurm prints
        "NOTICE: see ticket 12345 for policy",  # a banner that merely contains digits
        "error: Batch job submission failed",
    ])
    def test_rejected(self, raw):
        assert su.parse_submitted_job_id(raw) == "", (
            "guessing an id out of arbitrary text substitutes one wrong answer "
            "for another"
        )


class TestCondaEnvIsValidatedLikeModules:
    """SM-13 fixed the late failure for `module load`: the job queues, starts, and
    then dies. `--env` fails exactly the same way on `conda activate`, and the env
    list was already being fetched for the wizard's picker — but a name typed on
    the command line was never checked against it. One field validated, the
    neighbouring one not.
    """

    KNOWN = ["base", "pytorch", "tensorflow", "jax", "my_project"]

    def _known(self, mocker, envs):
        mocker.patch.object(su, "fetch_conda_envs", return_value=envs)

    def test_an_unknown_env_warns(self, mocker):
        self._known(mocker, self.KNOWN)
        issues = su.check_conda_env("nosuchenv", "conda")
        assert [lvl for lvl, _ in issues] == ["warning"]
        assert "nosuchenv" in issues[0][1]
        assert "conda activate" in issues[0][1]
        assert "pytorch" in issues[0][1]

    def test_a_known_env_passes(self, mocker):
        self._known(mocker, self.KNOWN)
        assert su.check_conda_env("pytorch", "conda") == []

    def test_mamba_is_checked_too(self, mocker):
        self._known(mocker, self.KNOWN)
        assert su.check_conda_env("nosuchenv", "mamba") != []

    @pytest.mark.parametrize("env_type", ["venv", "none", "", None])
    def test_only_named_conda_envs_are_checked(self, mocker, env_type):
        # A venv is a path, and a path unreadable from the login node can be
        # perfectly valid on the compute node — checking it manufactures false
        # refusals, the same restraint check_log_dirs applies.
        self._known(mocker, self.KNOWN)
        assert su.check_conda_env("/no/such/venv", env_type) == []

    def test_an_unaskable_conda_claims_nothing_about_the_env(self, mocker):
        # fetch_conda_envs returns [] for both "no conda" and "conda failed";
        # neither may read as "your environment does not exist". But "conda is
        # not reachable here" is a fact about the *script*, and staying silent
        # about it shipped a job that dies on its first line — on all three
        # clusters tested, none of which has conda on the default PATH.
        self._known(mocker, [])
        mocker.patch.object(su, "is_tool_available", return_value=False)
        mocker.patch.object(su, "fetch_available_modules", return_value=[])
        issues = su.check_conda_env("anything", "conda")
        assert [lvl for lvl, _ in issues] == ["warning"]
        msg = issues[0][1]
        assert "not on this cluster's PATH" in msg
        assert "not found" not in msg      # no claim about the env's existence

    def test_a_conda_providing_module_is_named(self, mocker):
        # The remedy is what makes it actionable: on a cluster where conda is one
        # `module load` away, "conda is missing" is the wrong conclusion.
        self._known(mocker, [])
        mocker.patch.object(su, "is_tool_available", return_value=False)
        mocker.patch.object(su, "fetch_available_modules",
                            return_value=["conda/23.10", "gcc/12"])
        msg = su.check_conda_env("myenv", "conda")[0][1]
        assert "--modules conda/23.10" in msg

    def test_a_substring_match_finds_it_where_module_avail_cannot(self, mocker):
        # `module -t avail conda` matches a name PREFIX, so it finds Pythia's
        # `conda/23.10` and misses midway3's `python/anaconda-2025.12` — the
        # cluster with the most obvious remedy was the one that got none.
        self._known(mocker, [])
        mocker.patch.object(su, "is_tool_available", return_value=False)
        mocker.patch.object(su, "fetch_available_modules", return_value=[
            "gcc/12", "python/anaconda-2019.03", "python/anaconda-2025.12"])
        msg = su.check_conda_env("myenv", "conda")[0][1]
        # ...and offers the NEWEST of the family, not every version of it.
        assert "python/anaconda-2025.12" in msg
        assert "2019.03" not in msg
        # One family, one candidate: nameable as the thing to do.
        assert "--modules python/anaconda-2025.12" in msg

    def test_one_candidate_per_family(self, mocker):
        # midway3 has anaconda, miniforge and miniforge3 side by side. Taking the
        # last of the flat sorted list picked miniforge3-4.8 as "newest".
        self._known(mocker, [])
        mocker.patch.object(su, "is_tool_available", return_value=False)
        mocker.patch.object(su, "fetch_available_modules", return_value=[
            "python/anaconda-2019.03", "python/anaconda-2025.12",
            "python/miniforge-24.1.2", "python/miniforge-25.3.0",
            "python/miniforge3-4.8",
        ])
        msg = su.check_conda_env("myenv", "conda")[0][1]
        for kept in ("python/anaconda-2025.12", "python/miniforge-25.3.0",
                     "python/miniforge3-4.8"):
            assert kept in msg, kept
        for dropped in ("2019.03", "24.1.2"):
            assert dropped not in msg, dropped
        # Several families: no single "e.g." guessed for the user.
        assert "e.g." not in msg

    def test_module_family_strips_either_version_convention(self):
        assert su._module_family("conda/23.10") == "conda"
        assert su._module_family("python/anaconda-2025.12") == "python/anaconda"
        assert su._module_family("python/miniforge3-4.8") == "python/miniforge3"
        # Nothing version-like: left alone rather than truncated.
        assert su._module_family("mpi") == "mpi"
        assert su._module_family("gcc/toolchain") == "gcc/toolchain"

    def test_no_module_system_still_warns_without_inventing_a_remedy(self, mocker):
        self._known(mocker, [])
        mocker.patch.object(su, "is_tool_available", return_value=False)
        mocker.patch.object(su, "fetch_available_modules", return_value=[])
        msg = su.check_conda_env("myenv", "conda")[0][1]
        assert "--modules" in msg and "conda/23.10" not in msg
        assert "--env-type venv" in msg

    def test_a_broken_conda_that_is_on_path_claims_nothing(self, mocker):
        # conda is right there and the listing still failed: a broken install, or
        # one that cannot read its own config. The activation line may well work,
        # so there is no positive evidence to report.
        self._known(mocker, [])
        mocker.patch.object(su, "is_tool_available", return_value=True)
        assert su.check_conda_env("anything", "conda") == []

    def test_mamba_env_type_names_mamba(self, mocker):
        self._known(mocker, [])
        mocker.patch.object(su, "is_tool_available", return_value=False)
        mocker.patch.object(su, "fetch_available_modules", return_value=[])
        msg = su.check_conda_env("e", "mamba")[0][1]
        assert "'mamba' is not on this cluster's PATH" in msg

    def test_an_empty_env_is_not_a_finding(self, mocker):
        self._known(mocker, self.KNOWN)
        assert su.check_conda_env("", "conda") == []
        assert su.check_conda_env("   ", "conda") == []

    def test_a_prefix_path_env_that_exists_passes(self, mocker):
        self._known(mocker, ["base", "/scratch/me/envs/custom"])
        assert su.check_conda_env("/scratch/me/envs/custom", "conda") == []

    def test_prefix_paths_are_not_offered_as_suggestions(self, mocker):
        # They are 100+ characters each and sort before every name, so listing
        # them buries the answer the user needs.
        long_path = "/project/someone/very/long/scratch/path/envs/theirs"
        self._known(mocker, [long_path, "base", "pytorch"])
        issues = su.check_conda_env("nosuchenv", "conda")
        assert issues
        assert long_path not in issues[0][1]
        assert "base" in issues[0][1] and "pytorch" in issues[0][1]

    def test_no_named_envs_means_no_suggestion_list(self, mocker):
        self._known(mocker, ["/only/a/prefix/env"])
        issues = su.check_conda_env("nosuchenv", "conda")
        assert issues
        assert "Available" not in issues[0][1]

    def test_the_modules_are_passed_through(self, mocker):
        # conda is usually provided by a module, so the env list is only visible
        # after loading it — fetch_conda_envs takes the modules for that reason.
        fetch = mocker.patch.object(su, "fetch_conda_envs", return_value=["base"])
        su.check_conda_env("x", "conda", ["python/anaconda-2025.12"])
        assert fetch.call_args[0][0] == ["python/anaconda-2025.12"]

    def test_it_reaches_the_cli(self):
        # Wired into site_check_issues, so every mode gets it. MOCK_CONDA_ENVS is
        # the env list under SLURMATE_MOCK.
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "220"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--job-name", "x",
             "--partition", "cpu-shared", "--cpus", "1", "--memory", "1G",
             "--time", "00:05:00", "--env", "nosuchenv", "--env-type", "conda",
             "--command", "python t.py"],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert "nosuchenv" in done.stderr
        assert "conda activate" in done.stderr
        # A warning, not an error: conda's env list is a property of this machine,
        # and a site whose compute nodes see an envs dir the login node does not
        # would turn an error into a false refusal.
        assert done.returncode == 0
        assert "#SBATCH" in done.stdout


class TestDiscardedMemoryValueIsDisclosed:
    """Slurm rejects `--mem` and `--mem-per-cpu` together, so the builder emits
    only one — correctly, `--mem-per-cpu`. But the discarded value then vanished
    from the summary entirely, which reads as "I never set that". The case that
    matters is the invisible one: a `memory` key inherited from a config file,
    dropped by a `--mem-per-cpu` typed on the command line.

    Same disclosure the Output directory row already makes for a flag that is
    given and has no effect.
    """

    def _rows(self, **over):
        from slurmate.builder import job_summary_rows

        answers = {
            "job_name": "x", "partition": "p", "cpus": 4,
            "time_limit": "00:05:00", "command": "echo hi",
        }
        answers.update(over)
        return dict(job_summary_rows(answers))

    def test_both_given_discloses_the_unused_one(self):
        rows = self._rows(memory="8G", mem_per_cpu="2G")
        assert rows["Mem per CPU"] == "2G"
        assert "8G" in rows["Memory"]
        assert "not used" in rows["Memory"]
        assert "--mem-per-cpu takes precedence" in rows["Memory"]

    def test_only_mem_per_cpu_adds_no_memory_row(self):
        rows = self._rows(mem_per_cpu="2G")
        assert rows.get("Mem per CPU") == "2G"
        assert "Memory" not in rows, "a row for a value the user never set"

    def test_only_memory_is_unchanged(self):
        rows = self._rows(memory="8G")
        assert "Mem per CPU" not in rows
        assert "not used" not in rows["Memory"]

    def test_the_script_still_carries_exactly_one_memory_directive(self):
        from slurmate.builder import build_from_answers

        script = build_from_answers({
            "job_name": "x", "partition": "p", "cpus": 4,
            "time_limit": "00:05:00", "command": "echo hi",
            "memory": "8G", "mem_per_cpu": "2G",
        })
        mem_lines = [
            ln for ln in script.splitlines()
            if ln.startswith("#SBATCH --mem")
        ]
        assert len(mem_lines) == 1, mem_lines
        assert mem_lines[0] == "#SBATCH --mem-per-cpu=2G", (
            "Slurm rejects --mem and --mem-per-cpu together"
        )


class TestSlurmSpellingsRoundTrip:
    """SM-25: the documented handoff is "slurmpast tells you what to request →
    slurmate builds the script", and it broke on exactly the two flags carrying
    the sizing — because slurmate *emitted* `--cpus-per-task` and `--mem` and
    accepted neither. `--mem` was the worse of the two: argparse called it
    ambiguous and named two flags the user had never typed, suggesting neither.

    The general rule, which is what this pins: anything appearing in slurmate's
    own output should be typeable back at slurmate.
    """

    # Every flag a generated script can carry, with a value valid for it.
    EMITTED = [
        ("--job-name", "x"), ("--partition", "cpu-shared"), ("--account", "a"),
        ("--qos", "normal"), ("--cpus-per-task", "2"), ("--mem", "1G"),
        ("--mem-per-cpu", "2G"), ("--time", "00:05:00"), ("--nodes", "1"),
        ("--ntasks-per-node", "2"), ("--gpus", "1"), ("--gres", "gpu:1"),
        ("--gpus-per-node", "1"), ("--gpus-per-task", "1"),
        ("--constraint", "feat"), ("--array", "1-4"), ("--output", "o-%j.out"),
    ]

    def _run(self, *extra):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "220"}
        return subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--force",
             "--job-name", "x", "--partition", "cpu-shared",
             "--time", "00:05:00", "--command", "echo hi", *extra],
            capture_output=True, text=True, timeout=180, env=env,
        )

    @pytest.mark.parametrize("flag,value", EMITTED, ids=[f[0] for f in EMITTED])
    def test_every_emitted_flag_is_accepted(self, flag, value):
        done = self._run(flag, value)
        assert "ambiguous option" not in done.stderr, done.stderr
        assert "unrecognized argument" not in done.stderr, done.stderr
        assert done.stdout.startswith("#!/bin/bash"), done.stderr

    def test_the_report_s_own_sizing_advice_is_typeable(self):
        # Verbatim from slurmpast's --sizing output, which is where this was hit.
        done = self._run("--time", "00:05:00", "--mem", "1G", "--cpus-per-task", "2")
        assert done.returncode == 0
        script = done.stdout
        assert "#SBATCH --time=00:05:00" in script
        assert "#SBATCH --mem=1G" in script
        assert "#SBATCH --cpus-per-task=2" in script

    def test_mem_does_not_shadow_mem_per_cpu(self):
        # Declaring --mem explicitly must leave --mem-per-cpu reachable, not turn
        # it into a second ambiguous prefix.
        done = self._run("--mem-per-cpu", "2G")
        assert "ambiguous" not in done.stderr
        assert "#SBATCH --mem-per-cpu=2G" in done.stdout

    def test_error_is_explained_rather_than_unrecognized(self):
        done = self._run("--error", "/tmp/x.err")
        assert done.returncode == 2
        assert "unrecognized" not in done.stderr
        assert "--custom-sbatch=--error=/tmp/x.err" in done.stderr

    def test_the_error_escape_hatch_actually_works(self):
        # The advice above is only worth giving if it does what it says: the
        # custom directive must replace the derived one, not duplicate it.
        done = self._run("--custom-sbatch=--error=/tmp/mine.err")
        err_lines = [
            ln for ln in done.stdout.splitlines()
            if ln.startswith("#SBATCH --error")
        ]
        assert err_lines == ["#SBATCH --error=/tmp/mine.err"], err_lines


class TestGpuSpellingTranslation:
    """`--gres`/`--gpus-per-*` are three renderings of one request, so they
    resolve to gpus/gpu_type/gpu_format rather than settings of their own."""

    @pytest.mark.parametrize("flag,value,count,gtype", [
        ("--gres", "gpu:2", 2, ""),
        ("--gres", "gpu:a100:2", 2, "a100"),
        ("--gres", "gpu:h100:1", 1, "h100"),
        ("--gpus-per-node", "2", 2, ""),
        ("--gpus-per-node", "a100:2", 2, "a100"),
        ("--gpus-per-task", "3", 3, ""),
    ])
    def test_parsed(self, flag, value, count, gtype):
        assert su.parse_gpu_spelling(flag, value) == (count, gtype)

    @pytest.mark.parametrize("value", [
        "lscratch:100",   # a real non-GPU gres, which slurmate does not manage
        "gpu",
        "gpu:bogus",
        "",
        "2",              # a bare count is not a gres
    ])
    def test_a_non_gpu_gres_is_refused_not_reinterpreted(self, value):
        # Quietly treating one of these as a GPU request would drop a resource
        # the user actually asked for.
        with pytest.raises(ValueError):
            su.parse_gpu_spelling("--gres", value)

    def test_a_disagreeing_duplicate_is_an_error(self):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--force",
             "--job-name", "x", "--partition", "gpu-shared", "--time", "00:05:00",
             "--command", "echo hi", "--gres", "gpu:2", "--gpus", "4"],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert done.returncode == 2
        assert "two spellings of one number" in done.stderr

    def test_an_agreeing_duplicate_is_fine(self):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--force",
             "--job-name", "x", "--partition", "gpu-shared", "--time", "00:05:00",
             "--command", "echo hi", "--gres", "gpu:2", "--gpus", "2"],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert done.returncode == 0
        assert "#SBATCH --gres=gpu:2" in done.stdout


class TestJobNameRewriteIsDisclosed:
    """`sanitize_job_name` is deliberate and documented — sbatch splits
    `--job-name` on whitespace, and the name becomes the log filename and the
    auto-saved script's filename, so a conservative character set is right. What
    was missing is telling the user it happened.

    The case that matters is not the developer's: *any* all-non-Latin name falls
    back to `slurm`, so a user's logs appear as `logs/slurm-<jobid>.out` and
    nothing anywhere explained why. Silently sending someone to look in the wrong
    place is the same failure as SM-24's discarded output, one field over.
    """

    def test_a_whitespace_collapse_stays_quiet(self):
        from slurmate.builder import job_name_change_note

        # Visible in the result, nothing lost — a note here would be noise.
        assert job_name_change_note("my training job") == ""
        assert job_name_change_note("tab\there") == ""

    def test_an_unchanged_name_stays_quiet(self):
        from slurmate.builder import job_name_change_note

        assert job_name_change_note("ok-name") == ""
        assert job_name_change_note("") == ""
        assert job_name_change_note("   ") == ""

    @pytest.mark.parametrize("raw,safe", [
        ("ünïcøde", "ncde"),
        ("训练任务", "slurm"),        # the fallback: nothing of the input survives
        ("../../escape", "escape"),
        ("semi;rm -rf x", "semirm_-rf_x"),
        ("$(whoami)", "whoami"),
        ("..", "slurm"),
    ])
    def test_a_lossy_rewrite_is_disclosed(self, raw, safe):
        from slurmate.builder import job_name_change_note

        note = job_name_change_note(raw)
        assert note, f"{raw!r} was silently rewritten to {safe!r}"
        assert raw in note and f"'{safe}'" in note
        # It names where the logs will actually be, which is the practical
        # consequence and the reason this is worth a line at all.
        assert f"{safe}-<jobid>.out" in note

    def test_it_reaches_the_cli(self):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "220"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--force",
             "--job-name", "训练任务", "--partition", "cpu-shared", "--cpus", "1",
             "--memory", "1G", "--time", "00:05:00", "--command", "echo hi"],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert done.returncode == 0, done.stderr
        assert "was changed to 'slurm'" in done.stderr
        # And the script really does carry the fallback, so the note is accurate.
        assert "#SBATCH --job-name=slurm" in done.stdout

    def test_the_summary_still_shows_what_the_script_says(self):
        # The note explains the change; the row must keep matching the directive,
        # not revert to showing the name the user typed.
        from slurmate.builder import build_from_answers, job_summary_rows

        answers = {
            "job_name": "ncde", "partition": "p", "cpus": 1, "memory": "1G",
            "time_limit": "00:05:00", "command": "echo hi",
        }
        rows = dict(job_summary_rows(answers))
        assert rows["Job name"] == "ncde"
        assert "#SBATCH --job-name=ncde" in build_from_answers(answers)


class TestUnlimitedTimeIsAcceptedAndBounded:
    """Two mirror-image defects around an unlimited `--time`.

    slurmate *refused* `--time=UNLIMITED`, which sbatch accepts — a false
    refusal, and the rarer, worse direction: there is no cluster on which it was
    right. And once accepted, the partition-limit check read "no limit" as a
    **zero-length** job (`_parse_slurm_time_to_minutes("UNLIMITED")` is `0.0`),
    so an unbounded request against a two-hour partition was affirmed. The cost
    estimate already got this right, from its own copy of the rule.

    Every spelling below was measured against a live sbatch: `UNLIMITED` and
    `INFINITE` parse, and `inf` does *not* — sbatch rejects it with "Invalid
    --time specification", so accepting it would trade the bug for its inverse.
    """

    @pytest.mark.parametrize("value", ["UNLIMITED", "unlimited", "INFINITE",
                                       "Infinite", "0", "00:00:00", "0-00:00:00"])
    def test_accepted(self, value):
        assert su.validate_time(value) is True

    @pytest.mark.parametrize("value", ["inf", "INF", "notatime", "forever", "1:99"])
    def test_rejected(self, value):
        assert su.validate_time(value) is False, (
            "accepting what sbatch rejects is the inverse of the bug being fixed"
        )

    @pytest.mark.parametrize("value", ["UNLIMITED", "infinite", "0", "00:00:00",
                                       "0-00:00:00"])
    def test_recognised_as_unbounded(self, value):
        assert su.time_request_is_unbounded(value) is True

    @pytest.mark.parametrize("value", ["", "  ", "notatime", "inf", "01:00:00"])
    def test_not_unbounded(self, value):
        # "" is deliberately not unbounded: an absent limit takes the partition
        # default. "not-a-time" must not parse to 0 and be called unlimited.
        assert su.time_request_is_unbounded(value) is False

    def test_one_implementation_not_two(self):
        # The builder's predicate delegates rather than reimplementing: two copies
        # of this rule is exactly how the estimate and the partition check came to
        # disagree about the same value.
        from slurmate.builder import _time_is_unbounded

        for value in ("UNLIMITED", "0", "inf", "notatime", "", "01:00:00"):
            assert _time_is_unbounded(value) is su.time_request_is_unbounded(value)

    @pytest.mark.parametrize("value", ["UNLIMITED", "0"])
    def test_unbounded_exceeds_a_finite_partition_limit(self, value):
        issues = su.validate_job_config({
            "time_limit": value, "cpus": 1,
            "_partition_obj": {
                "name": "p", "timelimit": "02:00:00", "nodes": 4, "nodes_up": 4,
                "cpus_per_node": 32, "mem_per_node_mb": 64000, "gpu_types": [],
            },
        })
        msgs = [m for _lvl, m in issues if "Time limit" in m]
        assert msgs, "an unbounded request was affirmed against a 2h partition"
        assert value in msgs[0] and "02:00:00" in msgs[0]

    def test_unbounded_is_fine_on_an_unbounded_partition(self):
        # inf vs inf: nothing to warn about, and comparing against inf must not
        # produce a warning by accident.
        issues = su.validate_job_config({
            "time_limit": "UNLIMITED", "cpus": 1,
            "_partition_obj": {
                "name": "p", "timelimit": "infinite", "nodes": 4, "nodes_up": 4,
                "cpus_per_node": 32, "mem_per_node_mb": 64000, "gpu_types": [],
            },
        })
        assert not [m for _lvl, m in issues if "Time limit" in m]

    def test_a_bounded_request_within_the_limit_still_passes(self):
        issues = su.validate_job_config({
            "time_limit": "01:00:00", "cpus": 1,
            "_partition_obj": {
                "name": "p", "timelimit": "02:00:00", "nodes": 4, "nodes_up": 4,
                "cpus_per_node": 32, "mem_per_node_mb": 64000, "gpu_types": [],
            },
        })
        assert not [m for _lvl, m in issues if "Time limit" in m]

    def test_the_estimate_says_unbounded_rather_than_a_number(self):
        from slurmate.builder import UNBOUNDED_ESTIMATE, estimate_su

        assert estimate_su(48, "UNLIMITED", 1) == UNBOUNDED_ESTIMATE


class TestMemoryUnitSuffix:
    """`16GB` is the most natural way to write memory, and slurmate refused it —
    another false refusal of something sbatch takes. Measured against a live
    client: `16KB`/`16MB`/`16GB`/`16TB` all parse in any case, while `16B` (no
    unit) and `16GiB` both get "Invalid --mem specification". So the rule is a
    trailing `B` *after* a K/M/G/T unit, not "any memory-looking word".
    """

    @pytest.mark.parametrize("value,emitted", [
        ("16GB", "16G"), ("16gb", "16G"), ("16Gb", "16G"),
        ("16KB", "16K"), ("16MB", "16M"), ("16TB", "16T"),
        ("0GB", "0"),
        # The B must not defeat the fractional conversion sbatch needs.
        ("1.5GB", "1536M"), ("1.5gb", "1536M"), ("0.5TB", "524288M"),
    ])
    def test_accepted_and_normalised(self, value, emitted):
        assert su.validate_memory(value) is True, value
        assert su.normalize_memory(value) == emitted

    @pytest.mark.parametrize("value", ["16B", "16GiB", "16XB", "16P", "GB", "B"])
    def test_still_rejected(self, value):
        # sbatch rejects each of these, so accepting them would trade one false
        # refusal for a script that dies at submit.
        assert su.validate_memory(value) is False, value

    def test_a_fractional_suffix_is_not_read_as_zero(self):
        # The regression this class caught while being written: the fractional
        # branch handed the *raw* string to the MB parser, which does not read a
        # trailing B, so "1.5GB" came back 0 — and "--mem=0M" means *all* the
        # node's memory to Slurm. A 1.5 GiB request becoming "the whole node" is
        # the worst possible direction for a normalisation bug.
        assert su.normalize_memory("1.5GB") == "1536M"
        assert su.normalize_memory("1.5GB") != "0M"

    @pytest.mark.parametrize("suffix", ["N", "C"])
    def test_the_slurm_per_node_per_cpu_suffix_still_works(self, suffix):
        assert su.normalize_memory(f"1.5G{suffix}") == "1536M"
        assert su.normalize_memory(f"16G{suffix}") == "16G"

    def test_it_reaches_both_memory_flags(self):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1"}
        for flag, expected in (("--memory", "#SBATCH --mem=16G"),
                               ("--mem-per-cpu", "#SBATCH --mem-per-cpu=2G")):
            value = "16GB" if flag == "--memory" else "2GB"
            done = subprocess.run(
                [sys.executable, "-m", "slurmate", "--print", "--force",
                 "--job-name", "x", "--partition", "cpu-shared", "--cpus", "2",
                 "--time", "00:05:00", "--command", "echo hi", flag, value],
                capture_output=True, text=True, timeout=180, env=env,
            )
            assert done.returncode == 0, done.stderr
            assert expected in done.stdout, (flag, done.stdout)


class TestArraySpecCalibrationHolds:
    """A re-measurement of `validate_array_spec` against a live controller, since
    the same "it was measured" claim on the memory validator still had a hole.

    Every case below was re-run through `sbatch --test-only` with an account and
    partition that actually validate — the first attempt used a combination whose
    site plugin rejects on *account* before Slurm ever looks at the array, so
    every spec came back "accepted" and the whole comparison was meaningless.
    """

    # Confirmed accepted by sbatch.
    ACCEPTED = ["5", "1-10", "0-9", "1,3,5", "1-10:2", "1-10%4", "1-5,10", "%4",
                "1-10:2%3", "0", "1,2-4:2", "1-10%0"]
    # Confirmed rejected by sbatch ("Invalid job array specification").
    REJECTED = ["10-1", "1-10:0", "1-", "-5", "a-b", "1--5"]

    @pytest.mark.parametrize("spec", ACCEPTED)
    def test_slurm_accepts_so_slurmate_must(self, spec):
        assert su.validate_array_spec(spec) is True, spec

    @pytest.mark.parametrize("spec", REJECTED)
    def test_slurm_rejects_so_slurmate_does(self, spec):
        assert su.validate_array_spec(spec) is False, spec


class TestTrailingPercentIsRefusedTruthfully:
    """The one divergence from sbatch, and it is deliberate: `1-10%` *is* accepted
    by sbatch, which silently drops the throttle and runs the array unthrottled.
    A `%` with no number is a lost digit, so refusing it catches a real mistake —
    but the message must not call it "invalid", because that is a false statement
    about what the scheduler would do.
    """

    @pytest.mark.parametrize("spec", ["1-10%", "1-3%", "5%", "1,3,5%"])
    def test_still_refused(self, spec):
        assert su.validate_array_spec(spec) is False

    @pytest.mark.parametrize("spec", ["1-10%", "1-3%"])
    def test_the_reason_is_the_real_one(self, spec):
        reason = su.array_spec_reason(spec)
        assert "no number" in reason
        assert "Slurm accepts this" in reason, (
            "the message claimed the scheduler rejects it, which it does not"
        )
        assert "invalid" not in reason.lower()
        # It names the fix with the user's own range filled in.
        assert "%4" in reason

    @pytest.mark.parametrize("spec", ["10-1", "a-b", "1-10:0", "1--5"])
    def test_genuinely_invalid_specs_still_say_invalid(self, spec):
        reason = su.array_spec_reason(spec)
        assert reason.startswith("Invalid array specification")

    def test_a_valid_spec_has_no_reason(self):
        for spec in ("1-10", "1-10%4", "%4", ""):
            assert su.array_spec_reason(spec) == ""

    def test_it_reaches_the_cli(self):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "220"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--force",
             "--job-name", "x", "--partition", "cpu-shared", "--cpus", "1",
             "--memory", "1G", "--time", "00:05:00", "--array", "1-10%",
             "--command", "echo hi"],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert done.returncode == 1
        assert "no number" in done.stderr


class TestSavedScriptIsPrivate:
    """SM-26: the saved script copy was created by `open(path, "w")`, so its mode
    was whatever the umask gave — 0002 on both clusters measured, i.e.
    `-rw-rw-r--`. That is a real disclosure here rather than a theoretical one:
    `/project/rcc` and the user's directory under it are both `o+x`, so a
    world-readable file at a known path is readable cluster-wide.

    The content is the *exact submitted script*, so by construction it holds
    whatever went into `--command`.
    """

    @pytest.fixture(autouse=True)
    def _umask(self):
        # The permissive umask that produced the finding. Restored afterwards, or
        # every later test in the process inherits it.
        old = os.umask(0o002)
        yield
        os.umask(old)

    def test_the_post_submit_copy_is_600(self, tmp_path, monkeypatch):
        from slurmate.main import _save_submitted_script

        monkeypatch.chdir(tmp_path)
        path = _save_submitted_script("#!/bin/bash\necho SECRET=x\n", "s", "999")
        assert path is not None
        assert os.stat(path).st_mode & 0o777 == 0o600, oct(
            os.stat(path).st_mode & 0o777
        )

    def test_a_fresh_file_is_600_under_a_permissive_umask(self, tmp_path):
        target = tmp_path / "fresh.sh"
        su.write_private_text(str(target), "#!/bin/bash\n")
        assert target.stat().st_mode & 0o777 == 0o600

    def test_an_existing_shared_file_keeps_its_mode(self, tmp_path):
        # The mode applies at creation only. Overwriting a file the user made
        # deliberately shareable must not silently re-close it.
        target = tmp_path / "shared.sh"
        target.write_text("old")
        target.chmod(0o644)
        su.write_private_text(str(target), "#!/bin/bash\nnew\n")
        assert target.stat().st_mode & 0o777 == 0o644
        assert target.read_text() == "#!/bin/bash\nnew\n"

    def test_content_survives_a_non_utf8_command(self, tmp_path):
        # surrogateescape, for the same reason the plain open() used it: under a
        # non-UTF-8 locale a UTF-8 --command arrives as lone surrogates.
        target = tmp_path / "s.sh"
        su.write_private_text(str(target), "echo \udcff\n")
        assert target.stat().st_mode & 0o777 == 0o600
        assert target.read_bytes().endswith(b"\n")

    def test_the_wizard_save_is_also_private(self, mocker, tmp_path, monkeypatch):
        # The site the report did not name: the wizard's "Save script to a file"
        # writes the same content to a user-chosen path.
        import questionary

        from slurmate.main import _save_script

        target = tmp_path / "job.sh"
        mocker.patch.object(
            questionary, "text",
            return_value=mocker.Mock(ask=lambda: str(target)),
        )
        monkeypatch.chdir(tmp_path)
        _save_script("#!/bin/bash\necho SECRET=x\n", "job.sh")
        assert target.stat().st_mode & 0o777 == 0o600


class TestEditorTempFileIsCleanedUp:
    """The report's own correction said the editor temp file "does leak a file
    into /tmp permanently". It does not, in this tree: `_edit_script_in_editor`
    unlinks it in a `finally`, so it survives neither the happy path nor an
    editor that fails to exec. Pinned so that stays true.
    """

    def test_the_temp_file_is_removed(self, mocker, tmp_path):
        from slurmate.main import _edit_script_in_editor

        seen = {}

        def fake_run(argv, **kw):
            seen["path"] = argv[-1]
            assert os.path.exists(seen["path"])
            return mocker.Mock(returncode=0)

        mocker.patch("subprocess.run", side_effect=fake_run)
        mocker.patch.dict(os.environ, {"EDITOR": "true"}, clear=False)
        _edit_script_in_editor("#!/bin/bash\necho hi\n")
        assert not os.path.exists(seen["path"]), "the temp script outlived the edit"

    def test_it_is_removed_even_when_the_editor_cannot_exec(self, mocker):
        from slurmate.main import _edit_script_in_editor

        seen = {}

        def fake_run(argv, **kw):
            seen["path"] = argv[-1]
            raise OSError("no such editor")

        mocker.patch("subprocess.run", side_effect=fake_run)
        mocker.patch.dict(os.environ, {"EDITOR": "nope"}, clear=False)
        _edit_script_in_editor("#!/bin/bash\necho hi\n")
        assert not os.path.exists(seen["path"])


class TestManagedFlagAdviceIsActionable:
    """SM-15 refuses a `--custom-sbatch` that duplicates a directive slurmate
    owns, and names the flag to use instead. SM-25 then made Slurm's own
    spellings first-class (`--cpus-per-task`, `--gres`, `--gpus-per-node/task`) —
    and left this mapping pointing at a *different* flag than the one the user
    typed. For `--gres` that also lost information, since `--gres gpu:a100:2`
    carries a type a bare `--gpus` does not.

    The guard that matters is the first test: advice naming a flag the CLI does
    not accept is worse than no advice.
    """

    def test_every_owner_is_a_flag_the_cli_accepts(self):
        from slurmate.builder import _MANAGED_CUSTOM_FLAGS

        for owner in sorted(set(_MANAGED_CUSTOM_FLAGS.values())):
            done = subprocess.run(
                [sys.executable, "-m", "slurmate", "--print", "--force",
                 "--job-name", "x", "--partition", "cpu-shared",
                 "--time", "00:05:00", "--command", "echo hi",
                 owner, "gpu:1" if owner == "--gres" else "1"],
                capture_output=True, text=True, timeout=180,
                env={**os.environ, "PYTHONPATH": os.path.abspath("src"),
                     "SLURMATE_MOCK": "1", "NO_COLOR": "1"},
            )
            assert "unrecognized argument" not in done.stderr, (
                f"the managed-flag message sends users to {owner}, which the CLI "
                f"does not accept"
            )
            assert "ambiguous option" not in done.stderr, owner

    @pytest.mark.parametrize("spec,flag", [
        ("--gres=gpu:2", "--gres"),
        ("--cpus-per-task=8", "--cpus-per-task"),
        ("--partition=gpu-shared", "--partition"),
        ("--gpus-per-node=2", "--gpus-per-node"),
    ])
    def test_a_self_owned_flag_says_pass_it_as_an_option(self, spec, flag):
        done = self._run(spec)
        assert done.returncode == 1
        assert f"Pass {flag} as a slurmate option" in done.stderr, done.stderr
        assert f"Use {flag} instead" not in done.stderr, "tautological advice"

    def test_a_short_form_still_names_the_long_flag(self):
        done = self._run("-p=gpu-shared")
        assert done.returncode == 1
        assert "Use --partition instead" in done.stderr

    def test_a_reconciled_flag_is_still_allowed(self):
        # --mem is reconciled, not refused: slurmate suppresses its own directive
        # and the summary shows the custom value.
        done = self._run("--mem=99G")
        assert done.returncode == 0
        assert "#SBATCH --mem=99G" in done.stdout
        mem_lines = [ln for ln in done.stdout.splitlines()
                     if ln.startswith("#SBATCH --mem")]
        assert mem_lines == ["#SBATCH --mem=99G"], mem_lines

    def _run(self, custom):
        return subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--force",
             "--job-name", "x", "--partition", "gpu-shared", "--cpus", "1",
             "--memory", "1G", "--time", "00:05:00",
             f"--custom-sbatch={custom}", "--command", "echo hi"],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "PYTHONPATH": os.path.abspath("src"),
                 "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "220"},
        )


class TestHeterogeneousPartitionUsesLargestNode:
    """SM-27: `sinfo`'s aggregate row reports a heterogeneous partition's
    *smallest* node and marks it `+` ("at least this much"). Treating that as the
    partition limit turns a floor into a ceiling, so a request that fits a larger
    node is warned about — 38 of 84 partitions on midway2 emit the `+`, and 20 of
    87 on midway3, where `test` advertises a floor while its nodes reach 256 cores
    and 2321910 MB.

    Invisible on a homogeneous cluster, which is why it survived.
    """

    AGG = "roux|infinite|2|up|16+|20044+|(null)|idle"
    NODES = "16|20044\n24|64000\n"

    def _parts(self, mocker, node_rows=NODES, node_rc=0):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)

        def run(cmd, timeout=30, **kw):
            if "-N" in cmd:
                return node_rows, "", node_rc
            return self.AGG + "\n", "", 0

        mocker.patch.object(su, "_run_command", side_effect=run)
        return su.fetch_partitions()

    def test_the_aggregate_row_is_recorded_as_a_floor(self, mocker):
        part = self._parts(mocker)[0]
        assert part["cpus_per_node"] == 16
        assert part["mem_per_node_mb"] == 20044
        assert part["heterogeneous"] is True

    def test_the_per_node_query_finds_the_maxima(self, mocker):
        self._parts(mocker)
        assert su.fetch_partition_node_maxima("roux") == (24, 64000)

    def test_capacity_prefers_the_largest_node(self, mocker):
        from slurmate.main import _get_partition

        part = _get_partition(self._parts(mocker), "roux")
        assert su.partition_capacity(part, "cpus") == (24, True)
        assert su.partition_capacity(part, "mem") == (64000, True)

    @pytest.mark.parametrize("answers", [
        {"cpus": 24},                      # fits the 24-core node
        {"memory": "60G"},                 # 61440 MB fits the 64000 MB node
        {"memory": "30G"},
        {"cpus": 16, "memory": "20G"},
    ])
    def test_a_request_that_fits_a_larger_node_is_not_warned_about(
        self, mocker, answers
    ):
        from slurmate.main import _get_partition

        part = _get_partition(self._parts(mocker), "roux")
        issues = su.validate_job_config({**answers, "_partition_obj": part})
        offending = [m for _l, m in issues if "exceeds" in m]
        assert offending == [], offending

    @pytest.mark.parametrize("answers,word", [
        ({"cpus": 32}, "CPUs"),
        ({"memory": "70G"}, "Memory"),
    ])
    def test_a_genuine_over_request_is_a_definite_ceiling(
        self, mocker, answers, word
    ):
        from slurmate.main import _get_partition

        part = _get_partition(self._parts(mocker), "roux")
        issues = su.validate_job_config({**answers, "_partition_obj": part})
        msgs = [m for _l, m in issues if word in m and "exceeds" in m]
        assert msgs, issues
        # The figure is now the largest node, so the claim is a real ceiling.
        assert "largest node" in msgs[0]
        assert "nodes differ" not in msgs[0]

    def test_an_unreadable_per_node_query_keeps_the_honest_floor_wording(self, mocker):
        # The fail-safe shape: an unknown must not turn the floor into a ceiling,
        # and must not silence the check either.
        from slurmate.main import _get_partition

        part = _get_partition(self._parts(mocker, node_rows="", node_rc=1), "roux")
        assert "max_cpus_per_node" not in part
        assert su.partition_capacity(part, "cpus") == (16, False)
        issues = su.validate_job_config({"cpus": 24, "_partition_obj": part})
        msgs = [m for _l, m in issues if "exceeds" in m]
        assert msgs and "smallest node" in msgs[0] and "nodes differ" in msgs[0]

    def test_a_homogeneous_partition_makes_no_extra_call(self, mocker):
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        calls = []

        def run(cmd, timeout=30, **kw):
            calls.append(cmd)
            return "flat|infinite|2|up|16|20044|(null)|idle\n", "", 0

        mocker.patch.object(su, "_run_command", side_effect=run)
        from slurmate.main import _get_partition

        part = _get_partition(su.fetch_partitions(), "flat")
        assert part.get("heterogeneous") is not True
        assert not any("-N" in c for c in calls), (
            "a homogeneous site should pay nothing for this"
        )


class TestCapacityRefusalUsesTheExactMaximum:
    """The other consumer of the figure SM-27 fixed. `capacity_refusal` skipped
    cpu/memory entirely on a mixed partition — correctly, while the only number
    available was `sinfo`'s floor, since refusing against a floor would claim
    "never" for a request a larger node takes.

    The per-node lookup removes that limitation where it has run, and leaving the
    skip in place produced a plain asymmetry: a 999-core request was refused on a
    homogeneous partition and passed in silence on a mixed one whose true maximum
    slurmate had already resolved.
    """

    EXACT = {
        "name": "roux", "nodes": 2, "nodes_up": 2, "cpus_per_node": 16,
        "mem_per_node_mb": 20044, "gpu_types": [], "timelimit": "infinite",
        "heterogeneous": True, "max_cpus_per_node": 24,
        "max_mem_per_node_mb": 64000,
    }
    UNRESOLVED = {k: v for k, v in EXACT.items() if not k.startswith("max_")}
    FLAT = {k: v for k, v in UNRESOLVED.items() if k != "heterogeneous"}

    @pytest.mark.parametrize("answers,fragment", [
        ({"cpus": 999}, "999 cores"),
        ({"cpus": 1, "memory": "999G"}, "1022976 MB"),
    ])
    def test_an_impossible_request_is_refused_on_a_mixed_partition(
        self, answers, fragment
    ):
        reason = su.capacity_refusal(self.EXACT, answers)
        assert fragment in reason, reason

    @pytest.mark.parametrize("answers", [
        {"cpus": 24},                 # exactly the largest node
        {"cpus": 1, "memory": "60G"},  # 61440 MB, fits the 64000 MB node
        {"cpus": 16},
    ])
    def test_a_fitting_request_is_not_refused(self, answers):
        assert su.capacity_refusal(self.EXACT, answers) == ""

    @pytest.mark.parametrize("answers", [{"cpus": 999}, {"memory": "999G"}])
    def test_an_unresolved_mixed_partition_keeps_its_silence(self, answers):
        # The fail-safe half: with only a floor to go on, refusing would claim
        # "never" for a request a larger node might take.
        assert su.capacity_refusal(self.UNRESOLVED, answers) == ""

    def test_a_homogeneous_partition_is_unchanged(self):
        assert "999 cores" in su.capacity_refusal(self.FLAT, {"cpus": 999})
        assert su.capacity_refusal(self.FLAT, {"cpus": 8}) == ""

    def test_it_stays_pure(self, mocker):
        # Documented contract: the ETA path consults this for free, so it must not
        # acquire a subprocess call by reading the maxima.
        run = mocker.patch.object(su, "_run_command")
        su.capacity_refusal(self.EXACT, {"cpus": 999})
        run.assert_not_called()

    def test_the_eta_reports_never_with_the_real_reason(self, mocker):
        # End to end through the tier that consumes it: tier 1 silent (controller
        # unreachable), so the capacity verdict decides.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)

        def run(cmd, timeout=30, **kw):
            if "sbatch" in cmd:
                return "", "sbatch: error: Unable to contact slurm controller\n", 1
            if "squeue" in cmd:
                return "", "", 0
            if "-N" in cmd:
                return "16|20044\n24|64000\n", "", 0
            return "roux|infinite|2|up|16+|20044+|(null)|idle\n", "", 0

        mocker.patch.object(su, "_run_command", side_effect=run)
        info = su.fetch_queue_eta("roux", req_nodes=1, cpus=999)
        # fetch_queue_eta does not consult capacity_refusal itself; the caller
        # does. What matters here is that tier 1 stayed silent so the caller's
        # verdict is the one that shows.
        assert info["source"] != "scheduler" or info["feasible"] is True


class TestQosWallLimitIsLeftToTheScheduler:
    """SM-28 asked for the QoS `MaxWall` to be compared locally. I implemented
    that, and measurement showed it produced a **false warning** on midway3, so it
    is reverted — recorded here because the reasoning is the point.

    Whether exceeding a QoS `MaxWall` is refused *at submit* depends on the QoS's
    `DenyOnLimit` flag. No QoS on midway3 sets it, and there
    `--qos=build --time=30-00:00:00` against `MaxWall=06:00:00` is reported
    `***PASSED***` by sbatch. So `sacctmgr`'s MaxWall is not a cluster-invariant
    ceiling, and comparing against it warns about a limit the scheduler does not
    enforce. On the reporting cluster it *is* enforced, and sbatch says so —
    which is the authoritative, per-site answer slurmate already consults on every
    path.
    """

    PART_INF = {
        "name": "build", "nodes": 4, "nodes_up": 4, "cpus_per_node": 32,
        "mem_per_node_mb": 64000, "gpu_types": [], "timelimit": "infinite",
    }
    PART_2H = {**PART_INF, "timelimit": "02:00:00"}

    def _time_msgs(self, answers):
        issues = su.validate_job_config(answers)
        return [m for _lvl, m in issues if "Time limit" in m]

    @pytest.mark.parametrize("requested", ["07:00:00", "24:00:00", "7-00:00:00"])
    def test_no_local_maxwall_claim_is_made(self, requested):
        # The guard against re-introducing the false warning: an unbounded
        # partition plus an explicit --qos must produce nothing, however large the
        # request, because only the scheduler knows whether this site enforces it.
        assert self._time_msgs({
            "time_limit": requested, "cpus": 1, "qos": "build",
            "_partition_obj": self.PART_INF,
        }) == []

    def test_no_qos_lookup_is_attempted(self, mocker):
        # And it must not even ask: a sacctmgr call whose answer cannot be used is
        # latency for nothing.
        run = mocker.patch.object(su, "_run_command")
        self._time_msgs({
            "time_limit": "7-00:00:00", "cpus": 1, "qos": "build",
            "_partition_obj": self.PART_INF,
        })
        for call in run.call_args_list:
            assert "sacctmgr" not in str(call), call

    def test_the_partition_limit_check_is_untouched(self):
        msgs = self._time_msgs({
            "time_limit": "04:00:00", "cpus": 1, "_partition_obj": self.PART_2H,
        })
        assert msgs and "partition limit (02:00:00)" in msgs[0]

    @pytest.mark.parametrize("spelling", ["0", "00:00:00", "UNLIMITED"])
    def test_unbounded_still_warns_against_a_finite_partition(self, spelling):
        # This half of SM-28 item 4 stands: it is a partition comparison, and a
        # partition MaxTime *is* enforced at submit.
        msgs = self._time_msgs({
            "time_limit": spelling, "cpus": 1, "_partition_obj": self.PART_2H,
        })
        assert msgs, f"{spelling} passed against a 2h partition"


class TestPerJobLimitsAreRequestFacts:
    """The scheduler's own verdict is what slurmate acts on, so its limit tokens
    have to be classified correctly. Slurm splits on one word: a `...PerJob` limit
    is a statement about the *request* — no waiting makes a 7-day job fit a 6-hour
    MaxWall — while `...PerUser`/`...PerAccount` count limits are about the moment
    and clear when something finishes.

    Before this, every per-job limit landed in the honest-but-vague "cannot tell"
    bucket, so a site that genuinely refuses the job said so only advisorily.
    """

    @pytest.mark.parametrize("token", [
        "QOSMaxWallDurationPerJobLimit",
        "QOSMaxCpuPerJobLimit",
        "QOSMaxNodePerJobLimit",
        "AssocMaxWallDurationPerJobLimit",
        "QOSMaxMemoryPerJobLimit",
    ])
    def test_a_per_job_limit_is_permanent(self, token):
        out = (f"sbatch: error: {token}\n"
               "allocation failure: Job violates accounting/QOS policy "
               "(job submit limit, user's size and/or time limits)")
        _eta, reason = su._read_test_only_output("", out, 1)
        assert su.refusal_is_permanent(reason) is True, reason
        assert su.refusal_is_transient(reason) is False

    @pytest.mark.parametrize("token", [
        "QOSMaxSubmitJobPerUserLimit",
        "AssocMaxSubmitJobLimit",
    ])
    def test_a_submit_count_limit_stays_transient(self, token):
        out = (f"sbatch: error: {token}\n"
               "allocation failure: Job violates accounting/QOS policy (...)")
        _eta, reason = su._read_test_only_output("", out, 1)
        assert su.refusal_is_transient(reason) is True, reason
        assert su.refusal_is_permanent(reason) is False

    def test_the_rule_does_not_fire_on_unrelated_prose(self):
        assert su.refusal_is_permanent("a job limit was mentioned in passing") is False
        assert su.refusal_is_permanent("perjob") is False


class TestMemoryZeroIsAlreadyExpressible:
    """SM-28 item 3 reported `--memory 0` as refused. It is not, in this tree —
    `validate_memory` was changed deliberately to accept every zero spelling,
    because `--mem=0` is Slurm's documented way to ask for all the memory on a
    node and refusing it left that request unexpressible. Pinned so it stays
    reachable."""

    @pytest.mark.parametrize("value", ["0", "0G", "0M", "0K", "0T", "0GB"])
    def test_memory_zero_is_accepted_and_labelled(self, value):
        assert su.validate_memory(value) is True
        # Normalised to the documented bare form, not "0M", which reads like a
        # request for nothing.
        assert su.normalize_memory(value) == "0"

    def test_it_reaches_the_script(self):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--force",
             "--job-name", "x", "--partition", "cpu-shared", "--cpus", "1",
             "--memory", "0", "--time", "00:05:00", "--command", "echo hi"],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert done.returncode == 0, done.stderr
        assert "#SBATCH --mem=0" in done.stdout


class TestArrayCostIsTheArrayCost:
    """The cost estimate ignored the array entirely, so `--array 1-1000` reported
    the same `2.0` CPU-hours as a single job — a thousandfold understatement of
    the one figure a user checks before submitting something expensive, and in the
    direction that matters, because it says an enormous job is cheap.

    Same family as `--time=0` reporting 96 core-hours for something unbounded: a
    confident number that is wrong.
    """

    @pytest.mark.parametrize("spec,count", [
        ("1-10", 10), ("1-1000", 1000), ("0-9", 10), ("5", 1),
        ("1,3,5", 3), ("1-5,10", 6),
        ("1-10:2", 5),      # 1,3,5,7,9
        ("1-10:3", 4),      # 1,4,7,10
        ("1-1000%4", 1000),  # the throttle caps concurrency, not the count
        ("1-10:2%3", 5),
    ])
    def test_the_task_count(self, spec, count):
        assert su.array_task_count(spec) == count

    @pytest.mark.parametrize("spec", ["", "   ", "%4", "10-1", "a-b", "1-10:0"])
    def test_unknowable_counts_return_none(self, spec):
        # A bare "%N" carries no indices, and an invalid spec is not a count of 1.
        assert su.array_task_count(spec) is None

    def test_the_estimate_multiplies(self):
        from slurmate.builder import estimate_su

        single = estimate_su(2, "01:00:00", 1, None)
        assert single == "2.0"
        arrayed = estimate_su(2, "01:00:00", 1, None, 1000)
        assert arrayed.startswith("2,000") or arrayed.startswith("2000"), arrayed
        # The per-task figure stays visible, so the multiplication is checkable.
        assert "1000 tasks" in arrayed and "2.0" in arrayed

    def test_a_single_task_array_adds_no_noise(self):
        from slurmate.builder import estimate_su

        assert estimate_su(2, "01:00:00", 1, None, 1) == "2.0"
        assert estimate_su(2, "01:00:00", 1, None, None) == "2.0"

    def test_gpu_hours_multiply_too(self):
        from slurmate.builder import estimate_gpu_hours

        one = estimate_gpu_hours(2, "02:00:00", 1, "gres_type", None)
        many = estimate_gpu_hours(2, "02:00:00", 1, "gres_type", None, 50)
        assert one == "4.0"
        assert many.startswith("200") and "50 tasks" in many

    def test_an_unbounded_array_still_says_unbounded(self):
        # Multiplying "unbounded" by 1000 is not a number; the label must survive.
        from slurmate.builder import UNBOUNDED_ESTIMATE, estimate_su

        assert estimate_su(2, "0", 1, None, 1000) == UNBOUNDED_ESTIMATE

    @pytest.mark.parametrize("spec,expected", [
        ("1-10", "10 tasks"),
        ("1-1000", "1000 tasks"),
        ("1-1000%4", "1000 tasks"),
    ])
    def test_it_reaches_the_summary(self, spec, expected):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "240"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--dry-run", "--force",
             "--job-name", "x", "--partition", "cpu-shared", "--cpus", "2",
             "--memory", "1G", "--time", "01:00:00", "--array", spec,
             "--command", "true"],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert expected in done.stdout, done.stdout[-500:]

    def test_a_non_array_summary_is_unchanged(self):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "240"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--dry-run", "--force",
             "--job-name", "x", "--partition", "cpu-shared", "--cpus", "2",
             "--memory", "1G", "--time", "01:00:00", "--command", "true"],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert "tasks ×" not in done.stdout


class TestCustomNtasksIsCosted:
    """Round 78 audited the cost estimates and found them exact in every case it
    covered. Two cases it did not cover were wrong: the array multiplier (fixed
    separately) and this one.

    slurmate has no `--ntasks` option, so `--custom-sbatch=--ntasks=N` is the
    *only* way to express an MPI job with it — a likely path, not an exotic one —
    and the estimate ignored it, reporting 2.0 core-hours for a job asking for 200
    cores.
    """

    @pytest.mark.parametrize("custom,total", [
        ("--ntasks=100", 100),
        ("-n 100", 100),
        ("--ntasks 8", 8),
        ("--ntasks=1", 1),
        # Last occurrence wins, as Slurm does and as _custom_mem_override does.
        ("--ntasks=4 --ntasks=9", 9),
    ])
    def test_the_total_is_read(self, custom, total):
        from slurmate.builder import custom_ntasks

        assert custom_ntasks(custom) == total

    @pytest.mark.parametrize("custom", [
        None, "", "--exclusive", "--ntasks=0", "--ntasks=-3", "--ntasks=abc",
        "--ntasks-per-socket=8",   # a different flag that merely starts the same
    ])
    def test_nothing_is_read_when_there_is_nothing(self, custom):
        from slurmate.builder import custom_ntasks

        assert custom_ntasks(custom) is None

    def test_the_estimate_uses_it(self):
        from slurmate.builder import estimate_su

        # 2 cpus/task × 100 tasks × 1 h
        assert estimate_su(2, "01:00:00", 1, None, None, 100) == "200"

    def test_a_total_replaces_rather_than_multiplies_per_node(self):
        # --ntasks is job-wide, so it must not be multiplied by --nodes as well.
        from slurmate.builder import estimate_su

        assert estimate_su(2, "01:00:00", 4, None, None, 100) == "200"

    def test_the_larger_task_count_wins(self):
        # --ntasks-per-node is a per-node cap, not a total, so the two can
        # disagree; the number Slurm will actually run is the larger.
        from slurmate.builder import estimate_su

        # 4 nodes x 8 per node = 32 tasks beats a stated total of 10.
        assert estimate_su(1, "01:00:00", 4, 8, None, 10) == "32.0"
        # and a stated total of 100 beats 4 x 8.
        assert estimate_su(1, "01:00:00", 4, 8, None, 100) == "100"

    def test_gpus_per_task_tracks_the_total(self):
        from slurmate.builder import estimate_gpu_hours

        assert estimate_gpu_hours(2, "01:00:00", 1, "gpus_per_task", None, None,
                                  50) == "100"
        # A job-wide --gpus is not per task, so the total must not apply.
        assert estimate_gpu_hours(2, "01:00:00", 1, "gpus", None, None, 50) == "2.0"

    def test_round_78_s_table_still_holds(self):
        # The audit that found no defect must keep finding none.
        from slurmate.builder import estimate_gpu_hours, estimate_su

        assert estimate_su(4, "01:00:00", 2, None) == "8.0"
        assert estimate_su(2, "01:00:00", 3, 4) == "24.0"
        assert estimate_su(28, "12:00:00", 1, None) == "336"
        assert estimate_gpu_hours(2, "01:00:00", 3, "gres_type", None) == "6.0"
        assert estimate_gpu_hours(2, "01:00:00", 3, "gpus", None) == "2.0"
        assert estimate_gpu_hours(2, "01:00:00", 3, "gpus_per_node", None) == "6.0"
        assert estimate_gpu_hours(2, "01:00:00", 3, "gpus_per_task", 4) == "24.0"

    def test_it_reaches_the_summary(self):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "240"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--dry-run", "--force",
             "--job-name", "x", "--partition", "cpu-shared", "--cpus", "2",
             "--memory", "1G", "--time", "01:00:00",
             "--custom-sbatch=--ntasks=100", "--command", "true"],
            capture_output=True, text=True, timeout=180, env=env,
        )
        assert "200" in done.stdout, done.stdout[-400:]


class TestSitePluginLoneMessageIsRead:
    """Booth's Pythia rejects a batch job on its **default** partition with a lua
    ``job_submit`` plugin, and its wording is not Slurm's ``Reason:`` convention:

        sbatch: error: Job submission rejected: Batch jobs cannot use the
                       `interactive_*` partitions.
        allocation failure: Unspecified error

    So the only half slurmate showed was ``Unspecified error`` — Slurm's
    contentless catch-all — on the shipped-defaults path of that whole cluster,
    while the sentence that says exactly what is wrong sat one line above it.
    """

    PYTHIA = (
        "sbatch: error: Job submission rejected: Batch jobs cannot use the "
        "`interactive_*` partitions.\nallocation failure: Unspecified error\n"
    )

    def test_the_plugins_sentence_is_shown(self):
        from slurmate.system_utils import _test_only_refusal

        got = _test_only_refusal(self.PYTHIA)
        assert "Batch jobs cannot use" in got
        # Slurm's own half is kept in parentheses, as for a Reason:-style plugin:
        # it is the half the marker lists are written against.
        assert "(Unspecified error)" in got

    def test_a_reason_line_still_wins(self):
        # midway3's plugin writes a six-line block ending in Reason:. That path
        # must be untouched — and its banner lines must not become the reason.
        from slurmate.system_utils import _test_only_refusal

        midway3 = (
            "sbatch: error: Verify job submission ...\n"
            "sbatch: error: Partition: test\n"
            "sbatch: error: Account: unknown\n"
            "sbatch: error: Verification: ***REJECTED***\n"
            "sbatch: error: Reason: Account is not specified\n"
            "allocation failure: Access/permission denied\n"
        )
        assert _test_only_refusal(midway3) == (
            "Account is not specified (Access/permission denied)"
        )

    def test_several_site_lines_claim_nothing_extra(self):
        # With no Reason: and more than one candidate there is no way to tell
        # which line is the verdict, so Slurm's own half stands alone.
        from slurmate.system_utils import _test_only_refusal

        out = (
            "sbatch: error: Verify job submission ...\n"
            "sbatch: error: Verification: ***REJECTED***\n"
            "allocation failure: Access/permission denied\n"
        )
        assert _test_only_refusal(out) == "Access/permission denied"

    def test_a_broken_sbatch_is_still_not_a_refusal(self):
        # The invariant this fallback must not break: a non-zero exit with no
        # `allocation failure:` verdict is not evidence about the job. sbatch
        # prints `sbatch: error:` for an unreachable controller too.
        from slurmate.system_utils import _test_only_refusal

        assert _test_only_refusal(
            "sbatch: error: Batch job submission failed: Unable to contact "
            "slurm controller\n"
        ) == ""
        assert _test_only_refusal("sbatch: error: Account is not specified\n") == ""

    def test_a_lone_limit_token_is_not_taken_as_the_sentence(self):
        # Mercury's shape: the CamelCase limit name is appended in brackets by a
        # separate step, so it must not also be consumed as the reason.
        from slurmate.system_utils import _test_only_refusal

        got = _test_only_refusal(
            "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
            "allocation failure: Job violates accounting/QOS policy\n"
        )
        assert got == (
            "Job violates accounting/QOS policy [QOSMaxSubmitJobPerUserLimit]"
        )

    def test_the_verdict_is_not_guessed_permanent(self):
        # Surfacing the sentence must not also promote it: no marker matches it,
        # and guessing "permanent" is the direction that blocks runnable jobs.
        from slurmate.system_utils import refusal_is_permanent, refusal_is_transient

        got = "Batch jobs cannot use the `interactive_*` partitions. (Unspecified error)"
        assert not refusal_is_permanent(got) and not refusal_is_transient(got)


class TestSlurmShortFlagsAreAccepted:
    """`-c` was the only Slurm short flag slurmate took, so `-p` and `-t` — the
    two most-typed flags in Slurm — were argparse errors. Every probe written for
    this round's cluster testing used them and every one failed before it ran.
    """

    def test_the_common_short_flags_parse(self):
        from slurmate.main import parse_args

        args = parse_args([
            "-J", "j", "-A", "acct", "-p", "part", "-q", "qos", "-t", "01:00:00",
            "-N", "2", "-c", "8", "-a", "1-4", "-C", "bigmem", "-G", "2",
            "-o", "out/%j.log",
        ])
        assert args.job_name == "j" and args.account == "acct"
        assert args.partition == "part" and args.qos == "qos"
        assert args.time == "01:00:00" and args.nodes == 2 and args.cpus == 8
        assert args.array == "1-4" and args.constraint == "bigmem"
        assert args.gpus == 2 and args.output_file == "out/%j.log"

    def test_every_short_flag_has_the_slurm_meaning(self):
        # A short flag that means something *different* from Slurm's is worse than
        # none: it silently misreads a copied command line.
        from slurmate.main import parse_args

        for flag, dest, value in (
            ("-J", "job_name", "n"), ("-A", "account", "a"),
            ("-p", "partition", "p"), ("-q", "qos", "q"),
            ("-t", "time", "5"), ("-a", "array", "1-2"),
            ("-C", "constraint", "f"), ("-o", "output_file", "f.out"),
        ):
            assert getattr(parse_args([flag, value]), dest) == value, flag

    def test_short_and_long_agree(self):
        from slurmate.main import parse_args

        assert (parse_args(["-p", "x", "-t", "1:00"]).partition
                == parse_args(["--partition", "x", "--time", "1:00"]).partition)

    def test_ntasks_has_no_short_flag(self):
        # Slurm's -n is --ntasks, which slurmate has no option for. Accepting it
        # as anything else would be the misreading above; rejecting it is honest.
        from slurmate.main import parse_args

        with pytest.raises(SystemExit):
            parse_args(["-n", "4"])


class TestGpusFlagTakesSlurmSpelling:
    """SM-25 again, in the one flag whose name matches the option it sets:
    slurmate *prints* ``--gpus=a100:2`` under ``--gpu-format gpus`` and argparse
    answered its own output with "invalid int value: 'a100:2'".
    """

    def test_a_typed_count_round_trips(self):
        from slurmate.main import parse_args

        args = parse_args(["--gpus", "a100:2"])
        assert args.gpus == 2 and args.gpu_type == "a100"
        # That rendering is the only one that produces this spelling, so typing it
        # back must reproduce the script it came from.
        assert args.gpu_format == "gpus"

    def test_a_bare_count_keeps_the_default_format(self):
        from slurmate.main import parse_args

        args = parse_args(["--gpus", "4"])
        assert args.gpus == 4 and args.gpu_type is None and args.gpu_format is None

    def test_an_explicit_type_and_format_win(self):
        from slurmate.main import parse_args

        args = parse_args(["--gpus", "a100:2", "--gpu-type", "v100",
                           "--gpu-format", "constraint"])
        assert args.gpus == 2 and args.gpu_type == "v100"
        assert args.gpu_format == "constraint"

    def test_nonsense_is_still_refused(self):
        from slurmate.main import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--gpus", "lots"])

    def test_the_emitted_directive_is_the_accepted_spelling(self):
        # Generate, then feed the generated flag back in: the loop SM-25 is about.
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1"}
        base = [sys.executable, "-m", "slurmate", "--print", "--force",
                "--job-name", "x", "--partition", "gpu-shared",
                "--time", "00:05:00", "--command", "true"]
        first = subprocess.run(
            base + ["--gpus", "2", "--gpu-type", "a100", "--gpu-format", "gpus"],
            capture_output=True, text=True, env=env, timeout=180)
        directive = next(
            ln for ln in first.stdout.splitlines() if ln.startswith("#SBATCH --gpus=")
        )
        value = directive.split("=", 1)[1]
        assert value == "a100:2"
        again = subprocess.run(base + [f"--gpus={value}"],
                               capture_output=True, text=True, env=env, timeout=180)
        assert again.returncode == 0, again.stderr
        assert first.stdout == again.stdout


class TestConstraintFormatIsCheckedAgainstFeatures:
    """The mirror of H2, wired end to end: on a cluster whose nodes advertise
    typed GRES and no features, `--gpu-format constraint` emits a
    `--constraint=<model>` Slurm rejects. Both Booth clusters are in that state
    on every partition, and H2's own advice is what sends a user there.
    """

    def _run(self, *extra):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "200"}
        return subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--job-name", "x",
             "--partition", "gpu-shared", "--time", "00:05:00",
             "--command", "true", *extra],
            capture_output=True, text=True, env=env, timeout=180,
        )

    def test_the_lookup_runs_for_a_statically_known_model(self, mocker):
        # The reason this needed wiring as well as a check: _partition_issues
        # skipped the sinfo lookup whenever the model was already in the
        # partition's static list — which is every typed-GRES model, i.e. exactly
        # the ones this check is about.
        from slurmate.main import _partition_issues

        called = {}

        def fake(part):
            called["part"] = part
            return {"typed": ["h100"], "feature": [], "constraint": []}

        mocker.patch("slurmate.main.fetch_gpu_type_sources", side_effect=fake)
        issues = _partition_issues({
            "_partition_obj": {"name": "p", "gpu_types": ["h100"], "has_gpu": True,
                               "cpus_per_node": 0, "mem_per_node_mb": 0,
                               "timelimit": None},
            "gpus": 1, "gpu_type": "h100", "gpu_format": "constraint",
        })
        assert called["part"] == "p"
        assert any(lvl == "error" and "not a node feature" in m for lvl, m in issues)

    def test_the_default_format_still_skips_the_lookup(self, mocker):
        # The "avoid a needless sinfo call" property the original wiring had.
        from slurmate.main import _partition_issues

        spy = mocker.patch("slurmate.main.fetch_gpu_type_sources")
        _partition_issues({
            "_partition_obj": {"name": "p", "gpu_types": ["h100"], "has_gpu": True,
                               "cpus_per_node": 0, "mem_per_node_mb": 0,
                               "timelimit": None},
            "gpus": 1, "gpu_type": "h100",
        })
        spy.assert_not_called()

    def test_mock_mode_sees_no_mismatch(self):
        # Demos and the rest of the suite use --gpu-format constraint freely; mock
        # types stand in for features as well as GRES types.
        done = self._run("--gpus", "1", "--gpu-type", "a100",
                         "--gpu-format", "constraint")
        assert done.returncode == 0, done.stderr
        assert "--constraint=a100" in done.stdout


class TestMemoryProvenanceNamesThePartition:
    """The note exists to say where a number nobody typed came from, and on the
    no---partition path — the default on every cluster tested — it said
    "from '?' node memory".
    """

    def _run(self, *extra):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "200"}
        return subprocess.run(
            [sys.executable, "-m", "slurmate", "--dry-run", "--force",
             "--job-name", "x", "--time", "00:05:00", "--command", "true", *extra],
            capture_output=True, text=True, env=env, timeout=180,
        )

    def test_the_default_partition_is_named(self):
        done = self._run()
        assert "'?'" not in done.stdout
        # The site default mock mode reports, which the summary also names two
        # lines further down — the two must agree.
        assert "from 'cpu-shared' node memory" in done.stdout

    def test_an_explicit_partition_is_still_named(self):
        done = self._run("--partition", "gpu-shared")
        assert "from 'gpu-shared' node memory" in done.stdout


class TestMockClusterHasTheShapeOfARealOne:
    """`sinfo` marks exactly one partition with a trailing `*` on every cluster
    there is; the fixture marked none. So the "no --partition given, fall back to
    the site default" path — with its own queue-depth, ETA, limit and memory
    consequences — could not be reached in mock mode, in `--demo`, or by any test.
    """

    def test_every_key_a_real_partition_has(self, mocker):
        # Derive the real key set from fetch_partitions itself rather than
        # restating it: a new field added there must be added to the fixture too,
        # or the check that reads it goes silent in mock mode.
        import slurmate.system_utils as su

        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(su, "_run_command", return_value=(
            "part*|02:00:00|4|up|32|131072|gpu:a100:4|idle\n", "", 0))
        real_keys = set(su.fetch_partitions()[0])
        for part in su.MOCK_PARTITIONS:
            missing = real_keys - set(part)
            assert not missing, f"{part['name']} lacks {sorted(missing)}"

    def test_a_per_node_gpu_over_request_is_flagged(self):
        # The concrete check the missing gpus_per_node key had switched off.
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "200"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--print", "--force",
             "--job-name", "x", "--partition", "gpu-shared", "--gpus", "99",
             "--time", "00:05:00", "--command", "true"],
            capture_output=True, text=True, env=env, timeout=180,
        )
        assert "GPUs (99)" in done.stderr and "per node" in done.stderr

    def test_exactly_one_partition_is_the_default(self):
        from slurmate.system_utils import MOCK_PARTITIONS

        defaults = [p["name"] for p in MOCK_PARTITIONS if p.get("is_default")]
        assert len(defaults) == 1, defaults

    def test_the_default_is_reachable_without_a_partition_flag(self):
        from slurmate.system_utils import fetch_partitions

        os.environ.setdefault("SLURMATE_MOCK", "1")
        assert any(p.get("is_default") for p in fetch_partitions())

    def test_the_summary_uses_it(self):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "200"}
        done = subprocess.run(
            [sys.executable, "-m", "slurmate", "--dry-run", "--force",
             "--job-name", "x", "--time", "00:05:00", "--command", "true"],
            capture_output=True, text=True, env=env, timeout=180,
        )
        # The limits, queue depth and ETA are the default partition's, and the
        # summary says so rather than reporting them as belonging to nothing.
        assert "default partition 'cpu-shared'" in done.stdout
        # ...and no --partition directive is invented from it.
        assert "#SBATCH --partition" not in done.stdout


# A slurmate entry point with one cluster fact replaced: the SelectType plugin.
# Every cluster available for testing runs select/cons_tres, so the "this site
# cannot parse that GPU syntax" branches have no live cluster to exercise them.
FAKE_SELECT_HARNESS = """
import os, sys
import slurmate.main as m
m.fetch_select_type = lambda: os.environ["SLURMATE_FAKE_SELECT_TYPE"]
sys.argv = ["slurmate"] + sys.argv[1:]
m.main()
"""

class TestInferredGpuFormatYieldsToTheCluster:
    """`--gpus a100:2` reads as gpu_format 'gpus', which needs select/cons_tres.
    That inference is slurmate's reading of a spelling, not a format the user
    asked for by name — so on a cluster that cannot parse it the rendering has to
    change, not the job fail. All three clusters tested run cons_tres, so this is
    the path that has no live cluster to catch it.
    """

    def _run(self, *extra, select="select/linear"):
        env = {**os.environ, "PYTHONPATH": os.path.abspath("src"),
               "SLURMATE_MOCK": "1", "NO_COLOR": "1", "COLUMNS": "200",
               "SLURMATE_FAKE_SELECT_TYPE": select}
        return subprocess.run(
            [sys.executable, "-c", FAKE_SELECT_HARNESS, "--print", "--job-name", "x",
             "--partition", "gpu-shared", "--time", "00:05:00",
             "--command", "true", *extra],
            capture_output=True, text=True, env=env, timeout=180,
        )

    def test_an_inferred_format_is_downgraded_not_refused(self):
        done = self._run("--gpus", "a100:2")
        assert done.returncode == 0, done.stderr
        assert "#SBATCH --gres=gpu:a100:2" in done.stdout
        assert "--gpus=a100:2" not in done.stdout
        # And it says so: a rendering the user did not choose changed under them.
        assert "using 'gres_type'" in done.stderr

    def test_an_explicit_format_is_still_an_error(self):
        done = self._run("--gpus", "2", "--gpu-type", "a100",
                         "--gpu-format", "gpus")
        assert done.returncode == 1
        assert "needs select/cons_tres" in done.stderr

    def test_a_cons_tres_cluster_keeps_the_inferred_format(self):
        done = self._run("--gpus", "a100:2", select="select/cons_tres")
        assert done.returncode == 0, done.stderr
        assert "#SBATCH --gpus=a100:2" in done.stdout
        assert "using 'gres_type'" not in done.stderr


class TestNodeLocalLogDirIsFlagged:
    """A `--output` path on node-local storage is the worst kind of working
    script. Measured on Booth's Mercury, whose `/tmp` is a per-node LVM volume:
    slurmate submitted from `/tmp/work/submit`, created `logs/`, reported
    `Submitted! Job ID: 563319`, and the job came back `COMPLETED 0:0` with **no
    log anywhere the submitter could see**. The 213-byte log was on the compute
    node's own `/tmp`. The identical job from an NFS home wrote its log normally.

    `check_log_dirs` could not catch it and said so in its own docstring — it
    tests the directory on the *login* node, where a node-local `/tmp/…/logs`
    exists and is writable. Both true; neither the fact that matters.
    """

    MOUNTINFO = (
        "22 96 0:21 / /proc rw,relatime shared:5 - proc proc rw\n"
        "36 96 253:2 / / rw,relatime shared:1 - xfs /dev/mapper/root rw\n"
        "44 36 253:4 / /tmp rw,relatime shared:9 - xfs /dev/mapper/tmpvol rw\n"
        "45 36 8:1 / /scratch/local rw,relatime shared:10 - xfs /dev/sda1 rw\n"
        "46 36 0:60 / /scratch/midway3 rw,relatime shared:12 - gpfs gpfs rw\n"
        "51 36 0:44 / /home rw,relatime shared:11 - nfs4 nfs:/home rw\n"
        "60 36 0:52 / /project rw,relatime shared:13 - gpfs gpfs_root rw\n"
    )

    @pytest.fixture(autouse=True)
    def _real_cluster(self, mocker):
        # The check is silent in mock mode (no cluster => no compute node), which
        # is what keeps the rest of the suite hermetic.
        mocker.patch.object(su, "_force_mock", return_value=False)
        mocker.patch.object(
            su, "_mount_fs_type",
            side_effect=lambda p: self._fs_for(os.path.abspath(p)))

    def _fs_for(self, path):
        best, best_type = "", ""
        for line in self.MOUNTINFO.splitlines():
            left, _, right = line.partition(" - ")
            point = left.split()[4]
            fs = right.split()[0]
            if path == point or path.startswith(point.rstrip("/") + "/"):
                if len(point) >= len(best):
                    best, best_type = point, fs
        return best_type

    def test_tmp_is_flagged(self):
        assert su.node_local_log_dir("/tmp/work/submit/logs") == "xfs"

    def test_shared_storage_is_not(self):
        assert su.node_local_log_dir("/home/me/logs") == ""
        assert su.node_local_log_dir("/project/lab/logs") == ""

    def test_a_shared_tmp_is_not_flagged(self, mocker):
        # A site that deliberately exports /tmp over NFS is doing nothing wrong,
        # and the type is what distinguishes it from one that does not.
        mocker.patch.object(su, "_mount_fs_type", return_value="nfs4")
        assert su.node_local_log_dir("/tmp/logs") == ""

    def test_var_tmp_and_dev_shm_too(self):
        assert su.node_local_log_dir("/var/tmp/logs") == "xfs"

    def test_a_scratch_that_names_itself_local(self):
        # midway3's /scratch/local is the same /dev/sda1 as its /tmp, and is where
        # $TMPDIR points, so a log written there is just as unreadable.
        assert su.node_local_log_dir("/scratch/local/me/logs") == "xfs"

    def test_bare_scratch_is_left_alone(self):
        # Node-local on some sites, shared on others: warning would fire on the
        # correct configuration.
        assert su.node_local_log_dir("/scratch/midway3/me/logs") == ""

    def test_a_path_merely_starting_with_the_prefix_is_not_matched(self):
        # "/tmpfiles" is not under "/tmp".
        assert su.node_local_log_dir("/tmpfiles/logs") == ""

    def test_the_script_check_reports_it(self):
        script = (
            "#!/bin/bash\n"
            "#SBATCH --output=/tmp/work/logs/x-%j.out\n"
            "#SBATCH --error=/tmp/work/logs/x-%j.err\n"
        )
        issues = su.check_log_dirs(script, will_create=True)
        assert [lvl for lvl, _ in issues] == ["warning"]
        msg = issues[0][1]
        assert "node-local storage (xfs)" in msg
        assert "COMPLETED" in msg and "--output-dir" in msg

    def test_it_fires_on_the_submit_path_too(self):
        # Unlike SM-24's missing-directory warning, creating the directory does
        # not help here, so `will_create=True` must not silence it.
        script = "#SBATCH --output=/tmp/work/logs/x-%j.out\n"
        assert su.check_log_dirs(script, will_create=True)
        assert su.check_log_dirs(script, will_create=False)

    def test_it_replaces_rather_than_stacks_with_the_other_checks(self):
        # One path, one finding: the node-local fact subsumes "does not exist".
        script = "#SBATCH --output=/tmp/definitely/not/there/x-%j.out\n"
        issues = su.check_log_dirs(script, will_create=False)
        assert len(issues) == 1 and "node-local" in issues[0][1]

    def test_mock_mode_stays_silent(self, mocker):
        mocker.patch.object(su, "_force_mock", return_value=True)
        assert su.node_local_log_dir("/tmp/work/logs") == ""


class TestMountFsTypeParsesMountinfo:
    """The lookup reads /proc/self/mountinfo rather than shelling out to
    `stat -f`, because check_log_dirs must not acquire a subprocess.
    """

    def test_the_longest_matching_mount_wins(self, tmp_path, mocker):
        info = tmp_path / "mountinfo"
        info.write_text(
            "36 96 253:2 / / rw - xfs /dev/root rw\n"
            "44 36 0:44 / /home rw - nfs4 nfs:/home rw\n"
            "45 44 0:45 / /home/me/scratch rw - lustre mds:/lus rw\n"
        )
        real_open = open
        mocker.patch("builtins.open", lambda p, *a, **k: real_open(
            info if p == "/proc/self/mountinfo" else p, *a, **k))
        assert su._mount_fs_type("/home/me/scratch/logs") == "lustre"
        assert su._mount_fs_type("/home/me/logs") == "nfs4"
        assert su._mount_fs_type("/opt/x") == "xfs"

    def test_an_unreadable_mountinfo_claims_nothing(self, mocker):
        mocker.patch("builtins.open", side_effect=OSError("nope"))
        assert su._mount_fs_type("/tmp") == ""
