"""Tests for the CLI entry point and batch mode."""

from slurmate.main import _parse_custom_flags, parse_args, run_batch


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.job_name is None
        assert args.partition is None
        assert args.cpus is None
        assert args.memory is None
        assert args.time is None
        assert args.nodes is None
        assert args.gpus is None
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
            env="myenv", env_type="mamba", command="python train.py",
            custom_sbatch="--exclusive", yes=True,
        )
        answers = run_batch(args, Console(), {})
        assert answers["job_name"] == "test"
        assert answers["partition"] == "gpu"
        assert answers["cpus"] == 4
        assert answers["gpus"] == 2
        assert answers["gpu_type"] == "a100"
        assert answers["command"] == "python train.py"
        assert answers["modules"] == ["python/3.10"]
        assert answers["env_name"] == "myenv"
        assert answers["env_type"] == "mamba"

    def test_batch_no_modules(self):
        from rich.console import Console
        import argparse

        args = argparse.Namespace(
            job_name="t", account=None, partition="cpu", qos=None,
            cpus=1, memory="1G", time="00:01:00", nodes=1, gpus=0,
            gpu_type=None, array=None, modules=None,
            env=None, env_type=None, command="echo hi",
            custom_sbatch=None, yes=True,
        )
        answers = run_batch(args, Console(), {})
        assert answers["modules"] is None
        assert answers["env_name"] is None
        assert answers["gpu_type"] is None

    def test_batch_parse_custom_flags(self):
        flags = _parse_custom_flags("--exclusive,--reservation=test")
        assert flags == ["--exclusive", "--reservation=test"]


class TestPartitionLimitsValidation:
    def test_validate_partition_limits_warnings(self):
        from rich.console import Console
        from slurmate.main import _validate_partition_limits

        # Mock partition object
        part_obj = {
            "name": "gpu-test",
            "cpus_per_node": 8,
            "mem_per_node_mb": 32768,  # 32G
            "timelimit": "02:00:00",
            "gpu_types": ["a100"],
        }

        # Case 1: All within limits
        answers = {
            "_partition_obj": part_obj,
            "cpus": 4,
            "memory": "16G",
            "time_limit": "01:00:00",
            "gpus": 1,
            "gpu_type": "a100",
        }
        console = Console(width=80)
        with console.capture() as capture:
            _validate_partition_limits(answers, console)
        assert capture.get() == ""

        # Case 2: Exceeding limits
        answers_exceed = {
            "_partition_obj": part_obj,
            "cpus": 16,
            "memory": "64G",
            "time_limit": "04:00:00",
            "gpus": 2,
            "gpu_type": "v100",  # not in static list, but in MOCK_GPU_TYPES — no warning
        }
        with console.capture() as capture:
            _validate_partition_limits(answers_exceed, console)
        warnings = capture.get()
        assert "CPUs (16) exceeds partition limit" in warnings
        assert "Memory (64G) exceeds partition limit" in warnings
        assert "Time limit (04:00:00) exceeds partition limit" in warnings
        assert "GPU type 'v100' not in partition list" not in warnings  # found via dynamic check

        # Case 3: Not in either static or dynamic GPU types
        answers_exceed2 = {
            "_partition_obj": part_obj,
            "cpus": 4,
            "memory": "16G",
            "time_limit": "01:00:00",
            "gpus": 2,
            "gpu_type": "b200",  # not in static list nor MOCK_GPU_TYPES
        }
        with console.capture() as capture:
            _validate_partition_limits(answers_exceed2, console)
        warnings = capture.get()
        assert "GPU type 'b200' not in partition list" in warnings

        # Case 4: Partition with no GPUs but GPUs requested
        part_no_gpu = {
            "name": "cpu-test",
            "cpus_per_node": 8,
            "mem_per_node_mb": 32768,
            "timelimit": "02:00:00",
            "gpu_types": [],
        }
        answers_gpu = {
            "_partition_obj": part_no_gpu,
            "gpus": 1,
        }
        with console.capture() as capture:
            _validate_partition_limits(answers_gpu, console)
        warnings = capture.get()
        assert "Partition 'cpu-test' does not support GPUs" in warnings


class TestColorSuppression:
    def test_no_color_env_var(self, monkeypatch):
        from slurmate.theme import C
        monkeypatch.setenv("NO_COLOR", "1")
        theme_c = C()
        assert theme_c.PINK == ""
        assert theme_c.RESET == ""


class TestMissingRequired:
    def test_warns_about_blank_required(self):
        from rich.console import Console
        from slurmate.main import _warn_missing_required
        console = Console(width=100)
        with console.capture() as cap:
            missing = _warn_missing_required({"partition": "gpu"}, console)
        assert "Job name" in missing and "Command to run" in missing
        assert "Partition" not in missing
        assert "Missing recommended fields" in cap.get()

    def test_no_warning_when_complete(self):
        from rich.console import Console
        from slurmate.main import _warn_missing_required
        console = Console(width=100)
        with console.capture() as cap:
            missing = _warn_missing_required(
                {"job_name": "j", "partition": "p", "command": "echo"}, console
            )
        assert missing == []
        assert cap.get() == ""
