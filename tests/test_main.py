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
            "--custom-sbatch=--exclusive,--reservation=abc",
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
        import argparse

        from rich.console import Console

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
        import argparse

        from rich.console import Console

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

        # Case 4: Partition KNOWN to have no GPUs (has_gpu: False, as
        # fetch_partitions emits) but GPUs requested — a hard error.
        part_no_gpu = {
            "name": "cpu-test",
            "cpus_per_node": 8,
            "mem_per_node_mb": 32768,
            "timelimit": "02:00:00",
            "gpu_types": [],
            "has_gpu": False,
        }
        answers_gpu = {
            "_partition_obj": part_no_gpu,
            "gpus": 1,
        }
        with console.capture() as capture:
            _validate_partition_limits(answers_gpu, console)
        warnings = capture.get()
        assert "Partition 'cpu-test' does not support GPUs" in warnings

        # Case 5: Partition of UNKNOWN capability (manually-typed / unrecognized
        # → synthetic fallback with no has_gpu key). We must NOT overclaim a hard
        # "does not support GPUs" error when we have no capability information.
        part_unknown = {
            "name": "typo-part", "cpus_per_node": 0, "mem_per_node_mb": 0,
            "gpu_types": [], "timelimit": None, "is_public": True,
        }
        with console.capture() as capture:
            _validate_partition_limits({"_partition_obj": part_unknown, "gpus": 1}, console)
        assert "does not support GPUs" not in capture.get()


class TestVersionConsistency:
    def test_version_matches_metadata(self):
        # P4-2 / P0-1: __version__ is single-sourced from package metadata, so
        # `slurmate --version` can never drift from the published version.
        import importlib.metadata

        import slurmate
        assert slurmate.__version__ == importlib.metadata.version("slurmate")


class TestBatchModeDetection:
    def test_any_job_flag_triggers_batch(self):
        from slurmate.main import _is_batch_mode, parse_args
        assert _is_batch_mode(parse_args(["--job-name", "x", "--command", "c"])) is True
        assert _is_batch_mode(parse_args(["--cpus", "8"])) is True
        assert _is_batch_mode(parse_args(["--yes"])) is True

    def test_no_flags_is_interactive(self):
        from slurmate.main import _is_batch_mode, parse_args
        assert _is_batch_mode(parse_args([])) is False
        # Output-only modes don't force batch by themselves.
        assert _is_batch_mode(parse_args(["--print"])) is False
        assert _is_batch_mode(parse_args(["--no-save-script"])) is False


class TestBatchNumericValidation:
    def _ns(self, **over):
        import argparse
        base = dict(
            job_name=None, account=None, partition="cpu-shared", qos=None,
            cpus=None, memory=None, time=None, nodes=None, ntasks_per_node=None,
            gpus=None, gpu_type=None, gpu_format=None, array=None, modules=None,
            env=None, env_type=None, output_dir=None, output_file=None,
            command="echo hi", custom_sbatch=None, yes=False,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def test_cpus_zero_rejected(self):
        import pytest
        from rich.console import Console

        from slurmate.main import run_batch
        with pytest.raises(SystemExit):
            run_batch(self._ns(cpus=0), Console(), {})

    def test_nodes_negative_rejected(self):
        import pytest
        from rich.console import Console

        from slurmate.main import run_batch
        with pytest.raises(SystemExit):
            run_batch(self._ns(nodes=-2), Console(), {})

    def test_gpus_negative_rejected(self):
        import pytest
        from rich.console import Console

        from slurmate.main import run_batch
        with pytest.raises(SystemExit):
            run_batch(self._ns(gpus=-1), Console(), {})

    def test_valid_numbers_pass(self):
        from rich.console import Console

        from slurmate.main import run_batch
        ans = run_batch(self._ns(cpus=8, nodes=2, gpus=1), Console(), {})
        assert ans["cpus"] == 8 and ans["nodes"] == 2 and ans["gpus"] == 1

    def test_ntasks_non_integer_config_rejected(self):
        # A non-integer ntasks (e.g. a stringy config value) is a single clean
        # hard error naming the value — not the old "using 0" warning followed
        # by a confusing "got 0" that never echoed what the user wrote.
        import pytest
        from rich.console import Console

        from slurmate.main import run_batch
        with pytest.raises(SystemExit):
            run_batch(self._ns(), Console(), {"ntasks_per_node": "x"})

    def test_ntasks_zero_rejected(self):
        import pytest
        from rich.console import Console

        from slurmate.main import run_batch
        with pytest.raises(SystemExit):
            run_batch(self._ns(ntasks_per_node=0), Console(), {})

    def test_ntasks_valid_from_config(self):
        from rich.console import Console

        from slurmate.main import run_batch
        ans = run_batch(self._ns(nodes=2), Console(), {"ntasks_per_node": "4"})
        assert ans["ntasks_per_node"] == 4


class TestBatchStringyConfigCoercion:
    def test_stringy_config_numerics_do_not_crash(self):
        # P0-3: quoted/stringy numbers in config must be coerced, not crash on
        # the `gpus > 0` comparison.
        import argparse

        from rich.console import Console

        from slurmate.main import run_batch
        ns = argparse.Namespace(partition=None, command="echo hi")
        cfg = {"partition": "gpu-shared", "gpus": "2", "cpus": "8", "nodes": "1"}
        ans = run_batch(ns, Console(), cfg)
        assert ans["cpus"] == 8 and ans["gpus"] == 2 and ans["nodes"] == 1


class TestBatchGpuFormatEnv:
    def test_env_seeds_gpu_format(self, monkeypatch):
        # P0-2: SLURMATE_GPU_FORMAT is the default in batch mode.
        import argparse

        from rich.console import Console

        from slurmate.main import run_batch
        monkeypatch.setenv("SLURMATE_GPU_FORMAT", "gpus")
        ns = argparse.Namespace(partition="gpu-shared", gpus=2, command="echo hi")
        ans = run_batch(ns, Console(), {})
        assert ans["gpu_format"] == "gpus"


class TestBatchJobNameSanitized:
    def test_spaces_collapsed(self):
        import argparse

        from rich.console import Console

        from slurmate.main import run_batch
        ns = argparse.Namespace(partition="cpu-shared", job_name="my job", command="x")
        ans = run_batch(ns, Console(), {})
        assert ans["job_name"] == "my_job"


class TestSubmitAndReport:
    _answers = {"job_name": "j", "command": "echo hi"}

    def test_mock_mode_reports_clearly(self, capsys):
        # P1-7: empty job ID (mock) → clear message, no blank ID / broken hints.
        from rich.console import Console

        from slurmate.main import _submit_and_report
        _submit_and_report("#!/bin/bash\necho hi", self._answers, Console())
        out = capsys.readouterr().out
        assert "mock mode" in out
        assert "Job ID:" not in out
        assert "squeue -j" not in out

    def test_failure_goes_to_stderr(self, capsys, mocker):
        # P1-9: submission errors write to stderr, not stdout.
        import pytest
        from rich.console import Console

        from slurmate.main import _submit_and_report
        mocker.patch("slurmate.main.submit_sbatch", return_value=(1, "", "boom"))
        with pytest.raises(SystemExit):
            _submit_and_report("script", self._answers, Console())
        cap = capsys.readouterr()
        assert "boom" in cap.err
        assert "boom" not in cap.out

    def test_no_save_script_skips_copy(self, capsys, mocker, tmp_path, monkeypatch):
        # P1-6: --no-save-script / save_script=False suppresses the CWD copy.
        from rich.console import Console

        from slurmate.main import _submit_and_report
        monkeypatch.chdir(tmp_path)
        mocker.patch("slurmate.main.submit_sbatch", return_value=(0, "12345", ""))
        _submit_and_report("script", self._answers, Console(), save_script=False)
        assert list(tmp_path.glob("*.sh")) == []
        assert "Submitted!" in capsys.readouterr().out


class TestPartitionLimitNtasks:
    def test_cpu_total_accounts_for_ntasks(self):
        # P3-4: ntasks_per_node × cpus is compared to the node core count.
        from rich.console import Console

        from slurmate.main import _validate_partition_limits
        part = {"name": "p", "cpus_per_node": 16, "mem_per_node_mb": 0,
                "timelimit": None, "gpu_types": []}
        console = Console(width=100)
        with console.capture() as cap:
            _validate_partition_limits(
                {"_partition_obj": part, "cpus": 8, "ntasks_per_node": 4}, console)
        assert "exceeds partition limit" in cap.get()  # 4×8=32 > 16


class TestBatchModeOutputModesWithConfig:
    def test_print_alone_is_interactive(self):
        from slurmate.main import _is_batch_mode, parse_args
        assert _is_batch_mode(parse_args(["--print"]), {}) is False
        assert _is_batch_mode(parse_args(["--dry-run"]), {}) is False

    def test_print_with_config_is_batch(self):
        from slurmate.main import _is_batch_mode, parse_args
        assert _is_batch_mode(parse_args(["--print"]), {"partition": "gpu"}) is True
        assert _is_batch_mode(parse_args(["--dry-run"]), {"command": "x"}) is True


class TestBatchGpuFormatValidation:
    def test_invalid_config_gpu_format_clamped(self):
        import argparse

        from rich.console import Console

        from slurmate.main import run_batch
        ns = argparse.Namespace(partition="gpu-shared", gpus=2,
                                gpu_format="bogus", command="x")
        ans = run_batch(ns, Console(), {})
        assert ans["gpu_format"] == "gres_type"


class TestFederatedJobId:
    def test_federated_id_split_for_hints(self, capsys, mocker):
        from rich.console import Console

        from slurmate.main import _submit_and_report
        mocker.patch("slurmate.main.submit_sbatch", return_value=(0, "98765;clusterA", ""))
        _submit_and_report("#SBATCH --output=j-%j.out",
                           {"job_name": "j", "command": "x"}, Console(), save_script=False)
        out = capsys.readouterr().out
        assert "98765" in out
        assert "clusterA" not in out


class TestEditorCommand:
    def test_editor_with_flags_is_split(self, monkeypatch):
        from slurmate.main import _editor_command
        monkeypatch.setenv("EDITOR", "code --wait")
        assert _editor_command() == ["code", "--wait"]

    def test_empty_editor_falls_back_to_vim(self, monkeypatch):
        from slurmate.main import _editor_command
        monkeypatch.setenv("EDITOR", "  ")
        monkeypatch.delenv("VISUAL", raising=False)
        assert _editor_command() == ["vim"]

    def test_bad_editor_does_not_crash(self, monkeypatch, capsys):
        from slurmate.main import _edit_script_in_editor
        monkeypatch.setenv("EDITOR", "definitely-not-a-real-editor-xyz --flag")
        monkeypatch.delenv("VISUAL", raising=False)
        # Returns the original script instead of raising FileNotFoundError.
        assert _edit_script_in_editor("KEEP ME") == "KEEP ME"
        assert "Could not open editor" in capsys.readouterr().out


class TestSaveSubmittedScriptDir:
    def test_saves_into_directory(self, tmp_path):
        from slurmate.main import _save_submitted_script
        d = tmp_path / "logdir"
        path = _save_submitted_script("script-body", "job", "123", directory=str(d))
        assert path == str(d / "job-123.sh")
        assert (d / "job-123.sh").read_text() == "script-body"

    def test_returns_none_on_write_failure(self, tmp_path):
        from slurmate.main import _save_submitted_script
        blocker = tmp_path / "afile"
        blocker.write_text("x")  # a file where a directory is expected
        path = _save_submitted_script("s", "j", "1", directory=str(blocker / "sub"))
        assert path is None


class TestCoerceIntReports:
    def test_bad_config_int_warns_not_silent(self):
        from rich.console import Console

        from slurmate.main import _coerce_int
        console = Console()
        with console.capture() as cap:
            value = _coerce_int("8cores", 4, field="cpus", err_console=console)
        assert value == 4
        assert "cpus" in cap.get()

    def test_none_returns_default_silently(self):
        from slurmate.main import _coerce_int
        assert _coerce_int(None, 7) == 7


class TestGpuWarningHasGpu:
    def test_count_only_gres_no_false_warning(self):
        from rich.console import Console

        from slurmate.main import _validate_partition_limits
        part = {"name": "gpu1", "cpus_per_node": 16, "mem_per_node_mb": 0,
                "timelimit": None, "gpu_types": [], "has_gpu": True}
        console = Console(width=100)
        with console.capture() as cap:
            _validate_partition_limits({"_partition_obj": part, "gpus": 2}, console)
        assert "does not support GPUs" not in cap.get()


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


class TestSummaryMarkupSafety:
    """User-controlled values must never be interpreted as Rich markup (crash)
    or silently dropped from the rendered summary."""

    def _render(self, answers):
        import io

        from rich.console import Console

        from slurmate.builder import build_from_answers, estimate_su
        from slurmate.main import _show_script_and_summary
        buf = io.StringIO()
        console = Console(file=buf, width=200, force_terminal=True)
        script = build_from_answers(answers)
        _show_script_and_summary(console, script, answers, estimate_su(1, "01:00:00"))
        return buf.getvalue()

    def test_closing_tag_in_command_does_not_crash(self):
        # '[/]' used to raise rich.errors.MarkupError and abort the run.
        out = self._render({"job_name": "j", "partition": "p",
                            "command": "grep '[/]' f.txt"})
        assert "grep" in out

    def test_bracket_glob_not_dropped_from_summary(self):
        out = self._render({"job_name": "j", "partition": "p",
                            "command": "cp data/[abc]file.txt out/"})
        assert "[abc]file.txt" in out

    def test_cjk_command_does_not_crash(self):
        # Wide glyphs must not blow up the panel-width math.
        out = self._render({"job_name": "j", "partition": "p",
                            "command": "echo 训练任务开始运行数据处理流程结束"})
        assert "训练任务" in out


class TestPartitionLimitWarningMarkupSafety:
    def test_bracket_gpu_type_warning_does_not_crash(self):
        import io

        from rich.console import Console

        from slurmate.main import _validate_partition_limits
        buf = io.StringIO()
        console = Console(file=buf, width=200, force_terminal=True)
        answers = {
            "gpus": 1, "gpu_type": "[/]", "memory": "16G", "time_limit": "01:00:00",
            "_partition_obj": {"name": "debug", "cpus_per_node": 8,
                               "mem_per_node_mb": 32768, "timelimit": "01:00:00",
                               "gpu_types": [], "has_gpu": False},
        }
        # Must not raise MarkupError.
        _validate_partition_limits(answers, console)
        assert "GPU" in buf.getvalue()


class TestHardErrorsSubmitGuard:
    """Pre-submit guard: errors (not warnings) block submission of a doomed job."""

    CPU = {"name": "cpu1", "cpus_per_node": 48, "mem_per_node_mb": 100000,
           "gpu_types": [], "has_gpu": False, "timelimit": None}

    def test_gpu_on_cpu_partition_is_hard_error(self):
        from slurmate.main import _hard_errors
        errs = _hard_errors({"_partition_obj": self.CPU, "gpus": 2})
        assert any("does not support GPUs" in m for m in errs)

    def test_valid_job_has_no_hard_errors(self):
        from slurmate.main import _hard_errors
        assert _hard_errors({"_partition_obj": self.CPU, "gpus": 0,
                             "cpus": 4, "memory": "16G"}) == []

    def test_warnings_do_not_block(self):
        from slurmate.main import _hard_errors
        # Over-limit memory is advisory (a heterogeneous partition can under-report),
        # so it is a WARNING, not a hard error — it must not block submission.
        assert _hard_errors({"_partition_obj": self.CPU, "cpus": 4,
                             "memory": "512G", "gpus": 0}) == []

    def test_no_partition_object_no_errors(self):
        from slurmate.main import _hard_errors
        assert _hard_errors({"gpus": 4}) == []
