"""Tests for the sbatch script builder."""

from slurmate.builder import build_sbatch_script, estimate_su


class TestBuildSbatchScript:
    def test_minimal_script(self):
        script = build_sbatch_script(
            job_name="test",
            partition="gpu",
            cpus=4,
            memory="16G",
            time_limit="01:00:00",
            command="python train.py",
        )
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=test" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --cpus-per-task=4" in script
        assert "#SBATCH --mem=16G" in script
        assert "#SBATCH --time=01:00:00" in script
        assert "python train.py" in script

    def test_full_script(self):
        script = build_sbatch_script(
            job_name="fulltest",
            partition="gpu_a100",
            account="myaccount",
            qos="high",
            cpus=8,
            memory="32G",
            time_limit="02:00:00",
            nodes=2,
            gpus=4,
            gpu_type="a100",
            array_spec="1-5",
            modules=["python/3.10", "cuda/12.0"],
            env_name="myenv",
            command="python train.py --epochs 100",
            custom_sbatch=["--exclusive", "--constraint=ssd"],
        )
        assert "#SBATCH --job-name=fulltest" in script
        assert "#SBATCH --partition=gpu_a100" in script
        assert "#SBATCH --account=myaccount" in script
        assert "#SBATCH --qos=high" in script
        assert "#SBATCH --cpus-per-task=8" in script
        assert "#SBATCH --mem=32G" in script
        assert "#SBATCH --time=02:00:00" in script
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --gres=gpu:a100:4" in script
        assert "#SBATCH --constraint=a100" not in script
        assert "#SBATCH --array=1-5" in script
        assert "module load python/3.10" in script
        assert "module load cuda/12.0" in script
        assert "source activate myenv" in script
        assert "python train.py --epochs 100" in script
        assert "#SBATCH --exclusive" in script
        assert "#SBATCH --constraint=ssd" in script

    def test_no_gpus(self):
        script = build_sbatch_script(
            job_name="nogpu", partition="cpu", cpus=2, memory="4G",
            time_limit="00:30:00", command="echo hi",
        )
        assert "#SBATCH --gres" not in script
        assert "#SBATCH --gpus" not in script

    def test_gpu_without_type(self):
        script = build_sbatch_script(
            job_name="gpuany", partition="gpu", cpus=4, memory="16G",
            time_limit="01:00:00", gpus=2, command="python train.py",
        )
        assert "#SBATCH --gres=gpu:2" in script

    def test_no_modules(self):
        script = build_sbatch_script(
            job_name="nomod", partition="cpu", cpus=1, memory="1G",
            time_limit="00:01:00", command="echo hi",
        )
        assert "module load" not in script

    def test_no_env(self):
        script = build_sbatch_script(
            job_name="noenv", partition="cpu", cpus=1, memory="1G",
            time_limit="00:01:00", command="echo hi",
        )
        assert "conda activate" not in script

    def test_shebang_first_line(self):
        script = build_sbatch_script(
            job_name="s", partition="cpu", cpus=1, memory="1G",
            time_limit="00:01:00", command="echo hi",
        )
        assert script.startswith("#!/bin/bash\n")

    def test_gpu_format_gres_type(self):
        script = build_sbatch_script(
            job_name="test", partition="gpu", cpus=4, memory="16G",
            time_limit="01:00:00", gpus=2, gpu_type="a100",
            gpu_format="gres_type", command="echo hi",
        )
        assert "#SBATCH --gres=gpu:a100:2" in script
        assert "#SBATCH --constraint" not in script

    def test_gpu_format_gpus(self):
        script = build_sbatch_script(
            job_name="test", partition="gpu", cpus=4, memory="16G",
            time_limit="01:00:00", gpus=2, gpu_type="a100",
            gpu_format="gpus", command="echo hi",
        )
        assert "#SBATCH --gpus=a100:2" in script
        assert "#SBATCH --constraint" not in script

    def test_gpu_format_duplicate_filtering(self):
        script = build_sbatch_script(
            job_name="test", partition="gpu", cpus=4, memory="16G",
            time_limit="01:00:00", gpus=2, gpu_type="a100",
            gpu_format="constraint", command="echo hi",
            custom_sbatch=["--gres=gpu:2", "--constraint=a100", "--constraint=ssd"]
        )
        assert script.count("--gres=gpu:2") == 1
        assert script.count("--constraint=a100") == 1
        assert "#SBATCH --constraint=ssd" in script

    def test_env_activation_strategies(self):
        script_conda = build_sbatch_script(
            job_name="test", partition="cpu", cpus=1, memory="1G",
            time_limit="00:01:00", env_name="myenv", env_type="Conda", command="echo hi"
        )
        assert "source activate myenv" in script_conda

        script_mamba = build_sbatch_script(
            job_name="test", partition="cpu", cpus=1, memory="1G",
            time_limit="00:01:00", env_name="myenv", env_type="Mamba", command="echo hi"
        )
        assert "mamba activate myenv" in script_mamba

        script_venv = build_sbatch_script(
            job_name="test", partition="cpu", cpus=1, memory="1G",
            time_limit="00:01:00", env_name="/path/to/venv", env_type="Virtualenv (venv)", command="echo hi"
        )
        assert "source /path/to/venv/bin/activate" in script_venv

    def test_multi_node_task_layout(self):
        script = build_sbatch_script(
            job_name="test", partition="cpu", cpus=4, memory="16G",
            time_limit="01:00:00", nodes=2, command="echo hi"
        )
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --ntasks-per-node=1" in script


class TestEstimateSu:
    def test_basic_estimate(self):
        result = estimate_su(4, "01:00:00", 1)
        assert result == "4.0"

    def test_zero_cpus(self):
        result = estimate_su(0, "01:00:00", 1)
        assert result == "0.00"

    def test_multi_node(self):
        result = estimate_su(8, "02:00:00", 4)
        assert result == "64.0"

    def test_factors_ntasks_per_node(self):
        # P3-3: tasks-per-node multiplies the per-task core count.
        assert estimate_su(8, "02:00:00", 2, ntasks_per_node=2) == "64.0"
        # None / 0 ntasks behaves like a single task (back-compat).
        assert estimate_su(8, "02:00:00", 2) == "32.0"
        assert estimate_su(8, "02:00:00", 2, ntasks_per_node=0) == "32.0"


class TestSanitizeJobName:
    def test_whitespace_and_unsafe_chars(self):
        from slurmate.builder import sanitize_job_name
        assert sanitize_job_name("my training job") == "my_training_job"
        assert sanitize_job_name("a/b;c") == "abc"
        assert sanitize_job_name("  ok-name_1.2  ") == "ok-name_1.2"
        assert sanitize_job_name("") == ""

    def test_builder_emits_single_token_job_name(self):
        # P1-8: spaces in the name must not split the directive.
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "my training job", "partition": "p"})
        assert "#SBATCH --job-name=my_training_job" in s


class TestErrorPathPreservesPattern:
    """P0-5: a %j/%A/%a in the trailing segment must not be dropped from .err."""

    def test_run_dot_j_keeps_pattern_in_error(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p", "output_file": "run.%j"})
        assert "#SBATCH --output=run.%j.out" in s
        assert "#SBATCH --error=run.%j.err" in s

    def test_x_dot_j_keeps_pattern(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p", "output_file": "%x.%j"})
        assert "#SBATCH --error=%x.%j.err" in s

    def test_base_pattern_still_swaps_real_extension(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p", "output_file": "out_%j.log"})
        assert "#SBATCH --output=out_%j.log" in s
        assert "#SBATCH --error=out_%j.err" in s


class TestArrayLogPattern:
    """P1-10: array jobs default to %A_%a, not %j, when no explicit file given."""

    def test_array_uses_A_a_with_output_dir(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "array_spec": "1-10", "output_dir": "logs"})
        assert "#SBATCH --output=logs/j-%A_%a.out" in s
        assert "#SBATCH --error=logs/j-%A_%a.err" in s

    def test_array_uses_A_a_with_no_output_config(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "array_spec": "1-10", "command": "echo hi"})
        assert "#SBATCH --output=j-%A_%a.out" in s
        assert "#SBATCH --error=j-%A_%a.err" in s

    def test_non_array_still_uses_j(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p", "output_dir": "logs"})
        assert "#SBATCH --output=logs/j-%j.out" in s


class TestJobSummaryRows:
    def test_includes_all_common_fields(self):
        # P3-9/P1-2: a single ordered field list shared by both summaries.
        from slurmate.builder import job_summary_rows
        rows = dict(job_summary_rows({
            "job_name": "j", "partition": "p", "cpus": 8, "memory": "32G",
            "time_limit": "01:00:00", "nodes": 2, "ntasks_per_node": 4,
            "gpus": 2, "gpu_type": "a100", "gpu_format": "gres_type",
            "array_spec": "1-5", "output_dir": "logs", "output_file": "o.out",
            "modules": ["cuda/12.1"], "env_name": "ai",
            "custom_sbatch": ["--exclusive"], "command": "python x.py",
        }))
        for key in ("Job name", "Partition", "Tasks/node", "GPUs", "GPU format",
                    "Modules", "Custom flags", "Command", "Output dir", "Output file"):
            assert key in rows, key
        assert rows["GPUs"] == "2 × a100"
        assert rows["Modules"] == "cuda/12.1"
        assert rows["Custom flags"] == "--exclusive"

    def test_omits_empty_and_gpu_when_zero(self):
        from slurmate.builder import job_summary_rows
        rows = dict(job_summary_rows({"job_name": "j", "partition": "p", "gpus": 0}))
        assert "GPUs" not in rows
        assert "GPU format" not in rows
        assert "Account" not in rows


class TestPartialPreview:
    def test_partial_omits_unentered_fields(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p"}, partial=True)
        assert "--job-name=j" in s
        assert "--partition=p" in s
        # not entered yet -> must not appear as placeholder lines
        assert "--time=" not in s
        assert "--nodes=" not in s
        assert "--cpus-per-task=" not in s
        assert "--mem=" not in s

    def test_partial_hides_partition_until_entered(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j"}, partial=True)
        assert "--partition" not in s
        assert "--job-name=j" in s

    def test_partial_hides_output_until_name(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"cpus": 4}, partial=True)
        assert "--output" not in s
        assert "--cpus-per-task=4" in s

    def test_full_build_still_fills_defaults(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p", "command": "echo hi"})
        assert "--time=02:00:00" in s
        assert "--nodes=1" in s
        assert "--cpus-per-task=1" in s
        assert "--mem=16G" in s
        assert "echo hi" in s


class TestOutputFileWithExtensions:
    def test_non_dot_out_extension_preserved(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_dir": "logs", "output_file": "run.log"})
        assert "#SBATCH --output=logs/run.log" in s
        assert "#SBATCH --error=logs/run.err" in s

    def test_txt_extension_preserved(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_dir": "logs", "output_file": "run.txt"})
        assert "#SBATCH --output=logs/run.txt" in s
        assert "#SBATCH --error=logs/run.err" in s

    def test_dot_out_extension_unchanged(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_dir": "logs", "output_file": "run-%j.out"})
        assert "#SBATCH --output=logs/run-%j.out" in s
        assert "#SBATCH --error=logs/run-%j.err" in s

    def test_bare_name_gets_dot_out(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_dir": "logs", "output_file": "run"})
        assert "#SBATCH --output=logs/run.out" in s
        assert "#SBATCH --error=logs/run.err" in s


class TestEnvTypeNoneWithEnv:
    def test_env_type_none_with_env_still_emits(self):
        from slurmate.builder import build_sbatch_script
        s = build_sbatch_script(
            job_name="test", partition="cpu", cpus=1, memory="1G",
            time_limit="00:01:00", env_name="myenv", env_type="none",
            command="echo hi",
        )
        # env_name is set, so builder enters the env block but no activation
        # line is emitted for unrecognized "none" strategy.
        assert "activate" not in s


class TestQosAndOutputFile:
    def test_qos_default_none_omitted(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p", "qos": "Default (none)"})
        assert "--qos" not in s

    def test_qos_explicit_kept(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p", "qos": "high"})
        assert "#SBATCH --qos=high" in s

    def test_output_file_in_dir_and_derived_error(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_dir": "logs", "output_file": "run-%j.out"})
        assert "#SBATCH --output=logs/run-%j.out" in s
        assert "#SBATCH --error=logs/run-%j.err" in s

    def test_output_file_explicit_path_ignores_dir(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_dir": "logs", "output_file": "/tmp/x.out"})
        assert "#SBATCH --output=/tmp/x.out" in s
        assert "#SBATCH --error=/tmp/x.err" in s


class TestPartialOutputTiming:
    def test_output_hidden_until_dir_or_file(self):
        from slurmate.builder import build_from_answers
        # just job name + partition -> no output lines yet
        s = build_from_answers({"job_name": "train", "partition": "p"}, partial=True)
        assert "--output" not in s
        assert "--error" not in s

    def test_output_shown_once_dir_entered(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "train", "partition": "p", "output_dir": "logs"}, partial=True)
        assert "#SBATCH --output=logs/train-%j.out" in s


class TestArrayOutputFileTag:
    """Array jobs with an explicit output_file must still differentiate per task."""

    def test_array_output_file_gets_per_task_tag(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "array_spec": "0-9", "output_file": "train.log"})
        assert "#SBATCH --output=train-%A_%a.log" in s
        assert "#SBATCH --error=train-%A_%a.err" in s

    def test_array_output_file_no_extension(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "array_spec": "0-9", "output_file": "run"})
        assert "#SBATCH --output=run-%A_%a.out" in s
        assert "#SBATCH --error=run-%A_%a.err" in s

    def test_array_output_file_with_pattern_untouched(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "array_spec": "0-9", "output_file": "run-%A_%a.out"})
        assert "#SBATCH --output=run-%A_%a.out" in s

    def test_non_array_output_file_unchanged(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_dir": "logs", "output_file": "train.log"})
        assert "#SBATCH --output=logs/train.log" in s


class TestEmptyDirectivesOmitted:
    def test_empty_partition_not_emitted(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "command": "echo hi"})
        assert "--partition=" not in s
        assert "#SBATCH --job-name=j" in s

    def test_empty_job_name_not_emitted(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"partition": "p", "command": "echo hi"})
        assert "--job-name=" not in s
        assert "#SBATCH --partition=p" in s


class TestJobNameFallback:
    def test_all_symbol_or_nonlatin_falls_back(self):
        from slurmate.builder import sanitize_job_name
        assert sanitize_job_name("###") == "slurm"
        assert sanitize_job_name("训练任务") == "slurm"
        assert sanitize_job_name("") == ""  # truly empty stays empty

    def test_builder_emits_fallback_not_empty_directive(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "@#%", "partition": "p", "command": "x"})
        assert "#SBATCH --job-name=slurm" in s
        assert "#SBATCH --job-name=\n" not in s


class TestOutputPathQuoting:
    def test_whitespace_path_quoted(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_dir": "/scratch/My Group/logs"})
        assert '#SBATCH --output="/scratch/My Group/logs/j-%j.out"' in s
        assert '#SBATCH --error="/scratch/My Group/logs/j-%j.err"' in s

    def test_spaceless_path_unquoted(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p", "output_dir": "logs"})
        assert "#SBATCH --output=logs/j-%j.out" in s


class TestCustomFlagGpuDedup:
    def _base(self, **kw):
        from slurmate.builder import build_sbatch_script
        args = dict(job_name="j", partition="p", cpus=1, memory="1G",
                    time_limit="01:00:00", gpus=2, gpu_type="v100",
                    gpu_format="gres_type", command="x")
        args.update(kw)
        return build_sbatch_script(**args)

    def test_space_form_exact_dup_gres_deduped(self):
        # The space form of an *exact* duplicate of the emitted directive is
        # dropped (the builder emits --gres=gpu:v100:2 for this _base).
        s = self._base(custom_sbatch=["--gres gpu:v100:2"])
        assert "--gres gpu:v100:2" not in s.replace("#SBATCH --gres=gpu:v100:2", "")
        assert "#SBATCH --gres=gpu:v100:2" in s

    def test_differing_gres_override_kept(self):
        # A custom --gres with a *different* value than the wizard emits is a
        # deliberate override and must survive (previously it was silently
        # dropped by the over-broad "startswith('gpu')" dedup).
        s = self._base(custom_sbatch=["--gres=gpu:a100:8"])
        assert "#SBATCH --gres=gpu:a100:8" in s
        assert "#SBATCH --gres=gpu:v100:2" in s

    def test_gpus_equals_kept_under_gres_type(self):
        # Under gres_type the builder emits no --gpus, so a custom --gpus must
        # survive (it was over-stripped before).
        s = self._base(custom_sbatch=["--gpus=8"])
        assert "#SBATCH --gpus=8" in s

    def test_newline_in_flag_not_injected_into_body(self):
        s = self._base(custom_sbatch=["--comment=a\necho pwned"])
        assert not any(ln.strip() == "echo pwned" for ln in s.splitlines())


class TestEnvNameQuoting:
    def test_venv_path_with_space_quoted(self):
        from slurmate.builder import build_sbatch_script
        s = build_sbatch_script(job_name="j", partition="p", cpus=1, memory="1G",
                                time_limit="01:00:00", env_name="/my envs/ai",
                                env_type="venv", command="x")
        assert "source '/my envs/ai/bin/activate'" in s

    def test_venv_path_no_space_unquoted(self):
        from slurmate.builder import build_sbatch_script
        s = build_sbatch_script(job_name="j", partition="p", cpus=1, memory="1G",
                                time_limit="01:00:00", env_name="/path/to/venv",
                                env_type="venv", command="x")
        assert "source /path/to/venv/bin/activate" in s


class TestTildeExpansion:
    def test_output_dir_tilde_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p", "output_dir": "~/logs"})
        assert f"#SBATCH --output={tmp_path}/logs/j-%j.out" in s
        assert "~/logs" not in s


class TestDirectiveNewlineFolding:
    """A newline in a directive value must not inject a script-body line or
    silently drop the #SBATCH directives that follow it."""

    def _lines(self, **kw):
        from slurmate.builder import build_sbatch_script
        args = dict(job_name="j", partition="p", cpus=4, memory="16G",
                    time_limit="01:00:00", command="echo hi")
        args.update(kw)
        return build_sbatch_script(**args).splitlines()

    def test_partition_newline_folded(self):
        lines = self._lines(partition="gpu\ntouch /tmp/PWNED")
        assert not any(ln.strip() == "touch /tmp/PWNED" for ln in lines)
        assert "#SBATCH --partition=gpu touch /tmp/PWNED" in lines
        # The directives after partition must survive (sbatch stops parsing at the
        # first non-comment line, so an injected line would drop them).
        assert "#SBATCH --cpus-per-task=4" in lines
        assert "#SBATCH --mem=16G" in lines

    def test_account_qos_array_module_newline_folded(self):
        lines = self._lines(account="a\nrm -rf x", qos="q\nevil",
                            array_spec="1-4\nbad", modules=["m\ninjected"])
        assert not any(ln.strip() == "rm -rf x" for ln in lines)
        assert not any(ln.strip() == "evil" for ln in lines)
        assert not any(ln.strip() == "bad" for ln in lines)
        assert not any(ln.strip() == "injected" for ln in lines)
        assert "module load m injected" in lines

    def test_output_path_newline_folded_and_quoted(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_file": "out\nevil.log", "command": "echo hi"})
        assert not any(ln.strip() == 'evil.log"' for ln in s.splitlines())
        # Folded to a space, so it is quoted into a single directive.
        assert '#SBATCH --output="out evil.log"' in s
        assert '#SBATCH --error="out evil.err"' in s

    def test_command_newline_preserved(self):
        # The command body is intentionally multi-line and must NOT be folded.
        lines = self._lines(command="echo a\necho b")
        assert "echo a" in lines
        assert "echo b" in lines

    def test_memory_and_time_newline_folded(self):
        # Free-form memory/time_limit are folded too (same injection class).
        lines = self._lines(memory="16G\necho pwned", time_limit="1:00:00\ninjected")
        assert not any(ln.strip() == "echo pwned" for ln in lines)
        assert not any(ln.strip() == "injected" for ln in lines)
        # Directives after --mem/--time must survive.
        assert "#SBATCH --time=1:00:00 injected" in lines
        assert "#SBATCH --nodes=1" in lines


class TestArrayLogClobberProtection:
    def test_master_only_pattern_gets_per_task_tag(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"array_spec": "1-4", "output_file": "run_%A.log",
                                "command": "run"})
        # %A alone is identical for every task; a per-task token (%a) must be added.
        out = next(ln for ln in s.splitlines() if ln.startswith("#SBATCH --output="))
        err = next(ln for ln in s.splitlines() if ln.startswith("#SBATCH --error="))
        assert "%a" in out and "%a" in err
        assert out != err

    def test_per_task_pattern_trusted(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"array_spec": "1-4", "output_file": "run_%a.log",
                                "command": "run"})
        assert "#SBATCH --output=run_%a.log" in s
        assert "#SBATCH --error=run_%a.err" in s


class TestOutputErrorCollision:
    def test_err_extension_does_not_collapse_streams(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"output_file": "run.err", "command": "run"})
        out = next(ln for ln in s.splitlines() if ln.startswith("#SBATCH --output="))
        err = next(ln for ln in s.splitlines() if ln.startswith("#SBATCH --error="))
        assert out != err, "stdout and stderr must not resolve to the same file"

    def test_err_extension_array_variant(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({"array_spec": "1-4", "output_file": "run.err",
                                "command": "run"})
        out = next(ln for ln in s.splitlines() if ln.startswith("#SBATCH --output="))
        err = next(ln for ln in s.splitlines() if ln.startswith("#SBATCH --error="))
        assert out != err


class TestStringyNumericCoercion:
    def test_stringy_values_do_not_crash(self):
        from slurmate.builder import build_from_answers, build_sbatch_script
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "gpus": "2", "nodes": "2", "gpu_type": "a100"})
        assert "#SBATCH --gres=gpu:a100:2" in s
        assert "#SBATCH --nodes=2" in s
        s2 = build_sbatch_script(job_name="j", partition="p", cpus=1, memory="1G",
                                 time_limit="01:00:00", gpus="3", nodes="2", command="x")
        assert "#SBATCH --gres=gpu:3" in s2


class TestDirectiveOrdering:
    def test_sbatch_directives_in_wizard_order(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({
            "job_name": "j", "partition": "p", "account": "a", "cpus": 4,
            "memory": "16G", "time_limit": "01:00:00", "nodes": 1,
            "output_dir": "logs", "command": "echo hi",
        })
        order = [ln for ln in s.splitlines() if ln.startswith("#SBATCH")]
        keys = [ln.split("=")[0].split()[1] for ln in order]
        assert keys == [
            "--job-name", "--partition", "--account", "--cpus-per-task",
            "--mem", "--time", "--nodes", "--output", "--error",
        ]

    def test_all_sbatch_before_modules_and_command(self):
        from slurmate.builder import build_from_answers
        s = build_from_answers({
            "job_name": "j", "partition": "p", "output_dir": "logs",
            "modules": ["cuda/12.1"], "env_type": "conda", "env_name": "ai",
            "command": "python x.py",
        })
        lines = s.splitlines()
        last_sbatch = max(i for i, ln in enumerate(lines) if ln.startswith("#SBATCH"))
        first_cmd = min(i for i, ln in enumerate(lines)
                        if ln and not ln.startswith("#") and ln.strip())
        assert last_sbatch < first_cmd
