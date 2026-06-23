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
    # Match sinfo partitions
    if "sinfo" in cmd and "%P|%l|%D|%a|%c|%m|%G" in cmd:
        return read_fixture("sinfo_partitions.txt"), "", 0
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
    # Match gpu types sinfo
    if "sinfo" in cmd and "%f|%G" in cmd:
        return read_fixture("sinfo_gputypes.txt"), "", 0
    # Match squeue jobs
    if "squeue" in cmd:
        return read_fixture("squeue_jobs.txt"), "", 0
    # Match sinfo queue status
    if "sinfo" in cmd and "%D|%a|%t" in cmd:
        return read_fixture("sinfo_queue.txt"), "", 0

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
        assert len(parts) == 5

        cpu_shared = next(p for p in parts if p["name"] == "cpu-shared")
        assert cpu_shared["nodes"] == 100
        assert cpu_shared["cpus_per_node"] == 32
        assert cpu_shared["mem_per_node_mb"] == 131072
        assert cpu_shared["gpu_types"] == []
        assert cpu_shared["timelimit"] == "02:00:00"

        gpu_shared = next(p for p in parts if p["name"] == "gpu-shared")
        assert gpu_shared["nodes"] == 10
        assert sorted(gpu_shared["gpu_types"]) == ["a100", "v100"]

    def test_fetch_public_partitions_real(self):
        public_parts = fetch_public_partitions()
        # debug has Hidden=YES and AllowAccounts=restricted, so it should be filtered out
        assert len(public_parts) == 4
        assert not any(p["name"] == "debug" for p in public_parts)

    def test_fetch_qos_for_partition_real(self):
        qos = fetch_qos_for_partition("cpu-shared")
        assert qos == ["normal", "high", "express"]

    def test_fetch_gpu_types_for_partition_real(self):
        gpu_types = fetch_gpu_types_for_partition("gpu-shared")
        assert gpu_types == ["a100", "v100"]

    def test_fetch_queue_eta_real(self):
        queue_info = fetch_queue_eta("gpu-shared", req_nodes=2)
        # squeue has 2 running, 2 pending
        assert queue_info["running"] == 2
        assert queue_info["pending"] == 2
        # sinfo_queue says 5 idle, 3 mix, 2 alloc nodes.
        # since req_nodes=2 and idle_nodes=5 >= 2, ETA should be 0 (labeled "now")
        assert queue_info["eta_seconds"] == 0
        assert queue_info["eta_label"] == "now"

        # If req_nodes=6, idle_nodes=5 < 6, but (idle+mix)=8 >= 6, ETA is 60 (labeled "~60s")
        queue_info_large = fetch_queue_eta("gpu-shared", req_nodes=6)
        assert queue_info_large["eta_seconds"] == 60
        assert queue_info_large["eta_label"] == "~60s"


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
        assert res == {"partition": "gpu-shared"}

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
