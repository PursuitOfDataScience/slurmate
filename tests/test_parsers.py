from pathlib import Path
from unittest.mock import MagicMock

import pytest

from slurmate.system_utils import (
    fetch_gpu_types_for_partition,
    fetch_partitions,
    fetch_public_partitions,
    fetch_qos_for_partition,
    fetch_queue_eta,
    submit_sbatch,
)
from slurmate.tui import Wizard

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def read_fixture(filename: str) -> str:
    with open(FIXTURES_DIR / filename) as f:
        return f.read()


def mock_run_command(cmd: list[str], timeout: int = 30) -> tuple[str, str, int]:
    # Match sinfo partitions (matched by prefix so the format can gain fields)
    if "sinfo" in cmd and any(a.startswith("%P|%l|%D") for a in cmd):
        return read_fixture("sinfo_partitions.txt"), "", 0
    # The wide, validation-only partition-name list.
    if "sinfo" in cmd and "-a" in cmd and "%P" in cmd:
        return "\n".join(
            line.split("|")[0] for line in read_fixture("sinfo_partitions.txt").splitlines()
        ), "", 0
    # Match scontrol partitions details (both list all and specific partition show)
    if "scontrol" in cmd and "show" in cmd and "partition" in cmd:
        part = None
        for arg in cmd[3:]:
            if arg != "-o":
                part = arg
                break
        scontrol_out = read_fixture("scontrol_partitions.txt")
        if part:
            for line in scontrol_out.splitlines():
                if f"PartitionName={part}" in line:
                    return line, "", 0
            return "", "Partition not found", 1
        else:
            return scontrol_out, "", 0
    # Match gpu types sinfo. The fixture tags each node row with its partition,
    # because the real call is per-partition (`-N -p <part>`) and a fixture that
    # hands every partition every node's features cannot show a partition-scoped
    # detection bug.
    if "sinfo" in cmd and "%f|%G" in cmd:
        want = cmd[cmd.index("-p") + 1] if "-p" in cmd else None
        rows = []
        for line in read_fixture("sinfo_gputypes.txt").splitlines():
            part, _, rest = line.partition("|")
            if want is None or part == want:
                rows.append(rest)
        return "\n".join(rows), "", 0
    # Match squeue jobs
    if "squeue" in cmd:
        return read_fixture("squeue_jobs.txt"), "", 0
    # Match sinfo queue status
    if "sinfo" in cmd and "%D|%a|%t" in cmd:
        return read_fixture("sinfo_queue.txt"), "", 0
    # Node-level free resources: 5 idle + 3 mixed with free cores + 2 fully allocated.
    if "sinfo" in cmd and "-N" in cmd:
        return read_fixture("sinfo_nodes.txt"), "", 0
    # No sbatch in the router, so --test-only yields no start time and
    # fetch_queue_eta falls through to the resource tier — which is what these
    # fixture-driven parser tests are exercising.

    return "", "Unknown mock command", 1


class TestRealParsers:
    @pytest.fixture(autouse=True)
    def setup_mocks(self, mocker):
        # Force is_tool_available to return True for Slurm tools
        mocker.patch("slurmate.system_utils.is_tool_available", return_value=True)
        # Mock _run_command with our fixture router
        mocker.patch("slurmate.system_utils._run_command", side_effect=mock_run_command)

    def test_fetch_partitions_real(self):
        parts = fetch_partitions()
        assert len(parts) == 9

        cpu_shared = next(p for p in parts if p["name"] == "cpu-shared")
        assert cpu_shared["nodes"] == 100
        assert cpu_shared["cpus_per_node"] == 32
        assert cpu_shared["mem_per_node_mb"] == 131072
        assert cpu_shared["gpu_types"] == []
        assert cpu_shared["timelimit"] == "02:00:00"
        # The trailing "*" in sinfo's %P marks the site default partition.
        assert cpu_shared["is_default"] is True

        gpu_shared = next(p for p in parts if p["name"] == "gpu-shared")
        assert gpu_shared["nodes"] == 10
        assert sorted(gpu_shared["gpu_types"]) == ["a100", "v100"]

    def test_node_counts_split_usable_from_dead(self):
        """SM-1: down/drained nodes are counted separately, not as capacity."""
        parts = fetch_partitions()
        by_name = {p["name"]: p for p in parts}
        # idle 60 + allocated 40 — both are live capacity.
        assert by_name["cpu-shared"]["nodes"] == 100
        assert by_name["cpu-shared"]["nodes_up"] == 100
        # 6 nodes, every one down* — advertises capacity nothing can use.
        assert by_name["retired"]["nodes"] == 6
        assert by_name["retired"]["nodes_up"] == 0

    def test_partitions_expose_has_gpu(self):
        parts = fetch_partitions()
        gpu_shared = next(p for p in parts if p["name"] == "gpu-shared")
        cpu_shared = next(p for p in parts if p["name"] == "cpu-shared")
        assert gpu_shared["has_gpu"] is True
        assert cpu_shared["has_gpu"] is False

    def test_fetch_public_partitions_real(self):
        public_parts = fetch_public_partitions()
        # debug has Hidden=YES and AllowAccounts=restricted, so it should be filtered out
        assert len(public_parts) == 8
        assert not any(p["name"] == "debug" for p in public_parts)

    def test_fetch_qos_for_partition_real(self):
        qos = fetch_qos_for_partition("cpu-shared")
        assert qos == ["normal", "high", "express"]

    def test_fetch_gpu_types_for_partition_real(self):
        gpu_types = fetch_gpu_types_for_partition("gpu-shared")
        assert gpu_types == ["a100", "v100"]

    def test_fetch_queue_eta_real(self):
        queue_info = fetch_queue_eta("gpu-shared", req_nodes=2, cpus=8)
        # squeue has 2 running, 2 pending
        assert queue_info["running"] == 2
        assert queue_info["pending"] == 2
        # sinfo_nodes has 5 idle + 3 mixed with >=4 free cores; 2 nodes of 8 cores
        # each is available now. Counted from free cores, not the state label.
        assert queue_info["eta_seconds"] == 0
        assert queue_info["eta_label"] == "now"
        assert queue_info["source"] == "resources"

        # An 8-core share fits the 5 idle nodes, the 2 mixed ones with 16 and 8
        # free cores, and the 2 GPU nodes with 16 free each — 9 in all. The third
        # mixed node has only 4 free and the 2 fully-allocated ones none, so
        # neither counts; nor do the `drained` and `idle*` nodes, which have 32
        # free cores apiece and are still unschedulable.
        assert fetch_queue_eta("gpu-shared", req_nodes=9, cpus=72)["eta_seconds"] == 0
        assert fetch_queue_eta("gpu-shared", req_nodes=10, cpus=80)["eta_seconds"] > 0

        # Per-node share, not the total: the mixed nodes have <32 free cores, so a
        # request needing a whole 32-core node only fits the 5 idle ones.
        assert fetch_queue_eta("gpu-shared", req_nodes=5, cpus=160)["eta_seconds"] == 0
        assert fetch_queue_eta("gpu-shared", req_nodes=6, cpus=192)["eta_seconds"] > 0

    def test_fetch_queue_eta_never_says_now_for_unavailable_gpus(self):
        # Two fixture nodes carry gpu:a100:4 — one with 2 of 4 allocated, one with
        # all 4 — and the rest have gpu:0. So a 2-GPU request fits exactly one
        # node, a 3-GPU request fits none, and the node whose GPUs are fully
        # allocated is "mixed" by state label yet cannot take a GPU job. The old
        # state-label tally reported "now" for all of these.
        assert fetch_queue_eta(
            "gpu-shared", req_nodes=1, cpus=8, gpus_per_node=2
        )["eta_seconds"] == 0
        assert fetch_queue_eta(
            "gpu-shared", req_nodes=2, cpus=8, gpus_per_node=2
        )["eta_seconds"] > 0
        info = fetch_queue_eta("gpu-shared", req_nodes=1, cpus=8, gpus_per_node=3)
        assert info["eta_seconds"] > 0 and info["eta_label"] != "now"


class TestSubmitSbatchReal:
    def test_submit_sbatch_success(self, mocker):
        mocker.patch("slurmate.system_utils.is_tool_available", return_value=True)
        # Mock subprocess.run for sbatch
        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "123456\n"
        mock_run.stderr = ""
        mocker.patch("subprocess.run", return_value=mock_run)

        code, out, err = submit_sbatch("#!/bin/bash\necho test", "test_job")
        assert code == 0
        assert out == "123456"
        assert err == ""

    def test_submit_sbatch_failure(self, mocker):
        mocker.patch("slurmate.system_utils.is_tool_available", return_value=True)
        # Mock subprocess.run error
        mock_run = MagicMock()
        mock_run.returncode = 1
        mock_run.stdout = ""
        mock_run.stderr = "sbatch: error: Invalid partition specification"
        mocker.patch("subprocess.run", return_value=mock_run)

        code, out, err = submit_sbatch("#!/bin/bash\necho test", "test_job")
        assert code == 1
        assert out == ""
        assert "sbatch: error:" in err

    def test_submit_sbatch_timeout(self, mocker):
        mocker.patch("slurmate.system_utils.is_tool_available", return_value=True)
        import subprocess
        mocker.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["sbatch"], 30))

        code, out, err = submit_sbatch("#!/bin/bash\necho test")
        assert code == -1
        assert "timed out" in err


class TestWizardFlow:
    def test_wizard_run_exit(self, mocker):
        wizard = Wizard()
        # Mock wizard.app.run to do nothing
        mocker.patch.object(wizard.app, "run")
        wizard.answers = {"partition": "gpu-shared"}
        res = wizard.run()
        assert res is not None
        # The answers now also carry config provenance (which keys still hold the
        # value a config file supplied), so assert on the job fields rather than
        # the whole dict.
        assert {k: v for k, v in res.items() if not k.startswith("_")} == {
            "partition": "gpu-shared"
        }
        assert res["_config_keys"] == []      # no config in this test

    def test_wizard_partition_mapping(self, mocker):
        wizard = Wizard()
        mocker.patch("slurmate.tui.fetch_public_partitions", return_value=[
            {"name": "cpu-shared", "nodes": 100, "cpus_per_node": 32, "mem_per_node_mb": 131072, "gpu_types": []}
        ])
        mocker.patch("slurmate.tui.fetch_partitions", return_value=[
            {"name": "cpu-shared", "nodes": 100, "cpus_per_node": 32, "mem_per_node_mb": 131072, "gpu_types": []}
        ])

        # Setup partition step
        wizard._setup_partition()
        # choices are: [(CUSTOM, CUSTOM), (PRIVATE, PRIVATE), (fmt, fmt)]
        # Simulate the user arrowing down to the cpu-shared row. The wizard reads
        # the highlighted row (_selected_index), not current_value, because it
        # handles Enter with eager=True.
        wizard.radio_list._selected_index = 2
        wizard._handle_partition_confirm()

        assert wizard.answers["partition"] == "cpu-shared"
        assert wizard.answers["_partition_obj"]["name"] == "cpu-shared"
        assert wizard.answers["_partition_obj"]["nodes"] == 100
