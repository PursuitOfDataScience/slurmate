"""Tests for the CLI entry point and batch mode."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["SLURMIFY_MOCK"] = "1"

from slurmify.main import _parse_custom_flags, parse_args, run_batch


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.job_name == ""
        assert args.partition == ""
        assert args.cpus == 4
        assert args.memory == "16G"
        assert args.time == "02:00:00"
        assert args.nodes == 1
        assert args.gpus == 0
        assert args.yes is False

    def test_batch_args(self):
        args = parse_args([
            "--job-name", "test",
            "--partition", "gpu",
            "--cpus", "8",
            "--memory", "32G",
            "--time", "04:00:00",
            "--nodes", "2",
            "--gpus", "4",
            "--gpu-type", "a100",
            "--array", "1-5",
            "--modules", "python/3.10,cuda/12.0",
            "--env", "myenv",
            "--command", "python train.py",
            f"--custom-sbatch=--exclusive,--reservation=abc",
            "--yes",
        ])
        assert args.job_name == "test"
        assert args.partition == "gpu"
        assert args.cpus == 8
        assert args.memory == "32G"
        assert args.time == "04:00:00"
        assert args.nodes == 2
        assert args.gpus == 4
        assert args.gpu_type == "a100"
        assert args.array == "1-5"
        assert args.modules == "python/3.10,cuda/12.0"
        assert args.env == "myenv"
        assert args.command == "python train.py"
        assert args.custom_sbatch == "--exclusive,--reservation=abc"
        assert args.yes is True


class TestRunBatch:
    def test_batch_mode_creates_answers(self):
        from rich.console import Console
        import argparse

        args = argparse.Namespace(
            job_name="test", account=None, partition="gpu", qos=None,
            cpus=4, memory="16G", time="01:00:00", nodes=1, gpus=2,
            gpu_type="a100", array=None, modules="python/3.10",
            env="myenv", command="python train.py",
            custom_sbatch="--exclusive", yes=True,
        )
        answers = run_batch(args, Console())
        assert answers["job_name"] == "test"
        assert answers["partition"] == "gpu"
        assert answers["cpus"] == 4
        assert answers["gpus"] == 2
        assert answers["gpu_type"] == "a100"
        assert answers["command"] == "python train.py"
        assert answers["modules"] == ["python/3.10"]
        assert answers["env_name"] == "myenv"

    def test_batch_no_modules(self):
        from rich.console import Console
        import argparse

        args = argparse.Namespace(
            job_name="t", account=None, partition="cpu", qos=None,
            cpus=1, memory="1G", time="00:01:00", nodes=1, gpus=0,
            gpu_type=None, array=None, modules=None,
            env=None, command="echo hi",
            custom_sbatch=None, yes=True,
        )
        answers = run_batch(args, Console())
        assert answers["modules"] is None
        assert answers["env_name"] is None
        assert answers["gpu_type"] is None

    def test_batch_parse_custom_flags(self):
        flags = _parse_custom_flags("--exclusive,--reservation=test")
        assert flags == ["--exclusive", "--reservation=test"]
