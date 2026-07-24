"""Tests for the sbatch script builder."""

from slurmate.builder import (
    build_from_answers,
    build_sbatch_script,
    estimate_su,
    job_summary_rows,
)


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
        assert "conda activate myenv" in script
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
        # H1: a differing custom --constraint is MERGED into the single directive
        # with "&" instead of being emitted as a second one. Slurm honours only the
        # last --constraint it sees, so two directives silently dropped the GPU
        # type; the exact duplicate ("a100") collapses into the merge.
        assert script.count("#SBATCH --constraint=") == 1
        assert "#SBATCH --constraint=a100&ssd" in script

    def test_env_activation_strategies(self):
        script_conda = build_sbatch_script(
            job_name="test", partition="cpu", cpus=1, memory="1G",
            time_limit="00:01:00", env_name="myenv", env_type="Conda", command="echo hi"
        )
        assert "conda activate myenv" in script_conda
        # Robust form: conda.sh is sourced first so the `conda` shell function is
        # defined in a non-login batch shell; the legacy bare `source activate`
        # (which silently no-ops on modern conda in batch) is no longer emitted.
        assert 'source "$(conda info --base)/etc/profile.d/conda.sh"' in script_conda
        assert "source activate myenv" not in script_conda

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
        for key in ("Job name", "Partition", "Tasks per node", "GPUs", "GPU format",
                    "Modules", "Custom flags", "Command", "Output directory", "Output file"):
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

    def test_leading_dash_dot_plus_stripped(self):
        from slurmate.builder import sanitize_job_name
        # A leading '-'/'+'/'.' is stripped so the saved "<job>-<id>.sh" file and
        # the log path don't look like a CLI option (tail -f -rf-1.out) or hide
        # as a dotfile. Interior dashes/dots are preserved.
        assert sanitize_job_name("-rf") == "rf"
        assert sanitize_job_name("--force") == "force"
        assert sanitize_job_name(".hidden") == "hidden"
        assert sanitize_job_name("+x") == "x"
        assert sanitize_job_name("---") == "slurm"  # only special chars → fallback
        assert sanitize_job_name("ok-name_1.2") == "ok-name_1.2"

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
        # dropped by the over-broad "startswith('gpu')" dedup)…
        s = self._base(custom_sbatch=["--gres=gpu:a100:8"])
        assert "#SBATCH --gres=gpu:a100:8" in s
        # …and P5: it now *replaces* the auto directive instead of sitting next to
        # it. Slurm honours the last option, so the override already won; the auto
        # line was dead weight that contradicted the script and made the summary
        # report a GPU request the job doesn't make.
        assert "#SBATCH --gres=gpu:v100:2" not in s
        assert s.count("#SBATCH --gres") == 1

    def test_gpus_equals_kept_under_gres_type(self):
        # Under gres_type the builder emits no --gpus, so a custom --gpus must
        # survive (it was over-stripped before).
        s = self._base(custom_sbatch=["--gpus=8"])
        assert "#SBATCH --gpus=8" in s

    def test_newline_in_flag_not_injected_into_body(self):
        s = self._base(custom_sbatch=["--comment=a\necho pwned"])
        assert not any(ln.strip() == "echo pwned" for ln in s.splitlines())
        # The folded value now also has a space, so it is quoted into a single
        # well-formed directive rather than left to split at submit time.
        assert '#SBATCH --comment="a echo pwned"' in s


class TestCustomFlagQuoting:
    def _base(self, **kw):
        from slurmate.builder import build_sbatch_script
        args = dict(job_name="j", partition="p", cpus=1, memory="1G",
                    time_limit="01:00:00", command="x")
        args.update(kw)
        return build_sbatch_script(**args)

    def test_spaced_value_quoted(self):
        # A custom-flag value containing a space is wrapped in quotes so Slurm's
        # directive parser keeps it one argument (--comment=my job would
        # otherwise bind --comment=my and leave "job" as a stray token).
        s = self._base(custom_sbatch=["--comment=my job"])
        assert '#SBATCH --comment="my job"' in s

    def test_spaceless_value_unquoted(self):
        s = self._base(custom_sbatch=["--reservation=abc", "--exclusive"])
        assert "#SBATCH --reservation=abc" in s
        assert "#SBATCH --exclusive" in s

    def test_already_quoted_value_not_double_quoted(self):
        s = self._base(custom_sbatch=['--comment="pre quoted"'])
        assert '#SBATCH --comment="pre quoted"' in s
        assert '\\"' not in s  # not escaped / double-wrapped

    def test_full_flow_cli_string_comment_with_space(self):
        # End-to-end: the CLI string form parses and emits one clean directive.
        from slurmate.builder import build_from_answers
        from slurmate.tui import _parse_custom_flags
        flags = _parse_custom_flags('--comment="my job" --exclusive')
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "custom_sbatch": flags, "command": "x"})
        assert '#SBATCH --comment="my job"' in s
        assert "#SBATCH --exclusive" in s
        assert not any(ln.strip().endswith('job"') and "comment" not in ln
                       for ln in s.splitlines())


class TestModulesCoercion:
    def test_stray_string_split_not_iterated(self):
        # A bare string (direct-API misuse) is split on commas, not iterated
        # character-by-character into "module load n / o / t / …".
        from slurmate.builder import build_sbatch_script
        s = build_sbatch_script(job_name="j", partition="p", cpus=1, memory="1G",
                                time_limit="01:00:00", modules="cuda/12.1,gcc",
                                command="x")
        assert "module load cuda/12.1" in s
        assert "module load gcc" in s
        assert "module load c\n" not in s


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

    def test_venv_trailing_slash_no_double_slash(self):
        from slurmate.builder import build_sbatch_script
        s = build_sbatch_script(job_name="j", partition="p", cpus=1, memory="1G",
                                time_limit="01:00:00", env_name="/venv/",
                                env_type="venv", command="x")
        assert "source /venv/bin/activate" in s
        assert "//bin/activate" not in s


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
        # The folded name is now shell-quoted (a space would otherwise split it
        # into two `module load` args, and metacharacters would survive).
        assert "module load 'm injected'" in lines

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


class TestClusterAgnosticBuilder:
    """Fixes from the cluster-agnostic audit: memory/constraint/GPU-format/conda."""

    def _base(self, **kw):
        args = dict(job_name="j", partition="p", cpus=4, memory="16G",
                    time_limit="01:00:00", command="echo hi")
        args.update(kw)
        return build_sbatch_script(**args)

    def test_mem_per_cpu_replaces_mem(self):
        s = self._base(mem_per_cpu="4G")
        assert "#SBATCH --mem-per-cpu=4G" in s
        assert "#SBATCH --mem=" not in s

    def test_empty_memory_omits_mem(self):
        # Whole-node/exclusive sites (e.g. TACC) reject --mem entirely.
        s = self._base(memory="")
        assert "#SBATCH --mem=" not in s
        assert "#SBATCH --mem-per-cpu" not in s

    def test_custom_mem_flag_suppresses_auto_mem(self):
        # A user-supplied memory flag wins; the auto --mem is not also emitted
        # (Slurm rejects a script setting both --mem and --mem-per-cpu).
        s = self._base(memory="16G", custom_sbatch=["--mem-per-cpu=2G"])
        assert "#SBATCH --mem=16G" not in s
        assert "#SBATCH --mem-per-cpu=2G" in s
        assert s.count("--mem-per-cpu") == 1

    def test_constraint_emitted(self):
        s = self._base(constraint="gpu")
        assert "#SBATCH --constraint=gpu" in s

    def test_gpus_per_node_format(self):
        s = self._base(gpus=4, gpu_type="a100", gpu_format="gpus_per_node")
        assert "#SBATCH --gpus-per-node=a100:4" in s
        assert "#SBATCH --gres" not in s

    def test_gpus_per_task_format(self):
        s = self._base(gpus=2, gpu_format="gpus_per_task")
        assert "#SBATCH --gpus-per-task=2" in s

    def test_gpus_per_node_custom_dedup(self):
        s = self._base(gpus=4, gpu_type="a100", gpu_format="gpus_per_node",
                       custom_sbatch=["--gpus-per-node=a100:4"])
        assert s.count("--gpus-per-node=a100:4") == 1

    def test_conda_uses_robust_activation(self):
        s = self._base(env_type="conda", env_name="ml")
        assert 'source "$(conda info --base)/etc/profile.d/conda.sh"' in s
        assert "conda activate ml" in s
        assert "source activate ml" not in s

    def test_nersc_style_script(self):
        # NERSC Perlmutter: mandatory -A and -C, GPUs via --gpus-per-node, no --gres.
        s = self._base(gpus=4, gpu_type="a100", gpu_format="gpus_per_node",
                       constraint="gpu", account="m1234")
        assert "#SBATCH --account=m1234" in s
        assert "#SBATCH --constraint=gpu" in s
        assert "#SBATCH --gpus-per-node=a100:4" in s
        assert "#SBATCH --gres" not in s

    def test_base_case_unchanged(self):
        # Shared-node cluster (e.g. midway3): defaults still produce --mem + --gres.
        s = self._base(gpus=2, gpu_type="a100")
        assert "#SBATCH --mem=16G" in s
        assert "#SBATCH --gres=gpu:a100:2" in s
        assert "#SBATCH --constraint" not in s
        assert "#SBATCH --mem-per-cpu" not in s

    def test_node_and_gpu_constraint_merge(self):
        # A node -C plus gpu_format=constraint must produce ONE merged --constraint,
        # not two conflicting lines (Slurm would keep only the last, dropping the node).
        s = self._base(gpus=2, gpu_type="a100", gpu_format="constraint", constraint="cpu")
        assert s.count("#SBATCH --constraint=") == 1
        assert "#SBATCH --constraint=cpu&a100" in s
        assert "#SBATCH --gres=gpu:2" in s


class TestCustomConstraintMerge:
    """H1: a custom --constraint/-C must be MERGED, never emitted twice.

    Slurm keeps only the LAST --constraint it sees and silently discards the
    earlier one (measured against a real sbatch: an invalid feature placed first
    schedules fine, placed last it fails with "Invalid feature specification").
    Since the custom-flag loop runs last, the discarded directive was always the
    one slurmate derived from the user's other answers.
    """

    def _b(self, **kw):
        base = dict(job_name="t", partition="p", cpus=1, memory="16G",
                    time_limit="01:00:00", command="x")
        base.update(kw)
        return build_sbatch_script(**base)

    def test_custom_constraint_merges_with_gpu_type(self):
        s = self._b(gpus=2, gpu_type="a100", gpu_format="constraint",
                    custom_sbatch=["--constraint=bigmem"])
        assert s.count("#SBATCH --constraint=") == 1
        assert "#SBATCH --constraint=a100&bigmem" in s
        assert "#SBATCH --gres=gpu:2" in s

    def test_custom_constraint_merges_with_constraint_param(self):
        s = self._b(constraint="cpu", custom_sbatch=["--constraint=bigmem"])
        assert s.count("#SBATCH --constraint=") == 1
        assert "#SBATCH --constraint=cpu&bigmem" in s

    def test_short_and_space_forms_merge_too(self):
        for flag in ("-C bigmem", "-C=bigmem", "--constraint bigmem"):
            s = self._b(constraint="cpu", custom_sbatch=[flag])
            assert s.count("#SBATCH --constraint=") == 1, flag
            assert "#SBATCH --constraint=cpu&bigmem" in s, flag

    def test_exact_duplicate_collapses(self):
        s = self._b(gpus=2, gpu_type="a100", gpu_format="constraint",
                    custom_sbatch=["--constraint=a100"])
        assert s.count("#SBATCH --constraint=") == 1
        assert "#SBATCH --constraint=a100\n" in s

    def test_case_differing_values_are_kept_separate(self):
        # Slurm node features are case-sensitive (measured: -C A100 does not match
        # a node advertising a100), so these are two different requirements.
        s = self._b(constraint="a100", custom_sbatch=["--constraint=A100"])
        assert "#SBATCH --constraint=a100&A100" in s

    def test_or_expression_is_parenthesized_when_merged(self):
        s = self._b(constraint="gpu", custom_sbatch=["--constraint=a100|v100"])
        assert "#SBATCH --constraint=gpu&(a100|v100)" in s

    def test_lone_or_expression_is_untouched(self):
        s = self._b(custom_sbatch=["--constraint=a100|v100"])
        assert "#SBATCH --constraint=a100|v100" in s

    def test_multiple_custom_constraints_all_merge(self):
        s = self._b(custom_sbatch=["--constraint=a", "-C b", "--exclusive"])
        assert "#SBATCH --constraint=a&b" in s
        assert "#SBATCH --exclusive" in s


class TestCustomLogFlagDedup:
    """M1: a custom --output/-o (or --error/-e) suppresses the auto directive."""

    def _b(self, custom):
        return build_from_answers({"job_name": "j", "partition": "p",
                                   "output_dir": "logs", "command": "x",
                                   "custom_sbatch": custom})

    def test_custom_output_suppresses_auto_output(self):
        s = self._b(["--output=/real/place/%j.log"])
        assert s.count("#SBATCH --output") == 1
        assert "#SBATCH --output=/real/place/%j.log" in s
        # stderr was not overridden, so its derived directive stays.
        assert "#SBATCH --error=logs/j-%j.err" in s

    def test_short_and_space_forms_also_suppress(self):
        for flag in ("-o /real/place/%j.log", "--output /real/place/%j.log"):
            s = self._b([flag])
            # Only the user's directive survives; the auto one is gone.
            assert "#SBATCH --output=logs/" not in s, flag
            assert f"#SBATCH {flag}" in s, flag
            assert sum(1 for ln in s.splitlines()
                       if ln.startswith(("#SBATCH --output", "#SBATCH -o"))) == 1, flag

    def test_custom_error_suppresses_auto_error_only(self):
        s = self._b(["--error=/real/place/%j.err"])
        assert s.count("#SBATCH --error") == 1
        assert "#SBATCH --output=logs/j-%j.out" in s

    def test_both_overridden(self):
        s = self._b(["--output=/o/%j.out", "--error=/e/%j.err"])
        assert s.count("#SBATCH --output") == 1
        assert s.count("#SBATCH --error") == 1


class TestSummaryMemoryMatchesScript:
    """M3: the summary must report the memory the SCRIPT requests."""

    def test_custom_mem_per_cpu_replaces_memory_row(self):
        ans = {"job_name": "j", "partition": "p", "memory": "16G",
               "custom_sbatch": ["--mem-per-cpu=2G"], "command": "x"}
        rows = dict(job_summary_rows(ans))
        assert rows.get("Mem per CPU") == "2G"
        assert "Memory" not in rows
        script = build_from_answers(ans)
        assert "#SBATCH --mem=" not in script
        assert "#SBATCH --mem-per-cpu=2G" in script

    def test_custom_mem_replaces_memory_row(self):
        ans = {"job_name": "j", "partition": "p", "memory": "16G",
               "custom_sbatch": ["--mem=32G"], "command": "x"}
        assert dict(job_summary_rows(ans)).get("Memory") == "32G"

    def test_space_form_counts_too(self):
        ans = {"job_name": "j", "partition": "p", "memory": "16G",
               "custom_sbatch": ["--mem 32G"], "command": "x"}
        assert dict(job_summary_rows(ans)).get("Memory") == "32G"
        assert "#SBATCH --mem=16G" not in build_from_answers(ans)

    def test_unrelated_mem_flag_does_not_suppress(self):
        # --mem-bind is not --mem; the auto directive must survive.
        ans = {"job_name": "j", "partition": "p", "memory": "16G",
               "custom_sbatch": ["--mem-bind=local"], "command": "x"}
        assert dict(job_summary_rows(ans)).get("Memory") == "16G"
        assert "#SBATCH --mem=16G" in build_from_answers(ans)


class TestTildeWithLeadingWhitespace:
    """L2: expanduser must run on the STRIPPED value."""

    def test_output_dir_leading_space(self):
        import os
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_dir": " ~/logs"})
        assert f"#SBATCH --output={os.path.expanduser('~')}/logs/j-%j.out" in s
        assert "~/logs" not in s

    def test_output_file_leading_space(self):
        import os
        s = build_from_answers({"job_name": "j", "partition": "p",
                               "output_file": " ~/logs/x.out"})
        assert f"#SBATCH --output={os.path.expanduser('~')}/logs/x.out" in s


class TestMambaActivationPortable:
    """L10: `mamba activate` alone dies on mamba >= 2 (no shell hook from conda.sh)."""

    def _b(self, env_type):
        return build_sbatch_script(job_name="t", partition="p", cpus=1,
                                   memory="1G", time_limit="01:00:00",
                                   env_name="myenv", env_type=env_type, command="x")

    def test_mamba_falls_back_to_conda(self):
        s = self._b("Mamba")
        assert 'source "$(conda info --base)/etc/profile.d/conda.sh"' in s
        assert "mamba activate myenv >/dev/null 2>&1 || conda activate myenv" in s

    def test_conda_path_unchanged(self):
        s = self._b("Conda")
        assert "conda activate myenv" in s
        assert "mamba" not in s


class TestGpuHours:
    def test_zero_gpus_is_blank(self):
        from slurmate.builder import estimate_gpu_hours
        assert estimate_gpu_hours(0, "02:00:00") == ""

    def test_per_node_formats_multiply_by_nodes(self):
        from slurmate.builder import estimate_gpu_hours
        # 4 GPUs/node x 2 nodes x 2h = 16
        assert estimate_gpu_hours(4, "02:00:00", 2, "gres_type") == "16.0"
        assert estimate_gpu_hours(4, "02:00:00", 2, "gpus_per_node") == "16.0"

    def test_total_format_does_not_multiply_by_nodes(self):
        from slurmate.builder import estimate_gpu_hours
        # --gpus=4 is job-wide: 4 x 2h = 8
        assert estimate_gpu_hours(4, "02:00:00", 2, "gpus") == "8.0"

    def test_per_task_format_multiplies_by_tasks(self):
        from slurmate.builder import estimate_gpu_hours
        # 1 GPU/task x 4 tasks/node x 2 nodes x 1h = 8
        assert estimate_gpu_hours(1, "01:00:00", 2, "gpus_per_task", 4) == "8.0"


class TestSplitListElementFlags:
    """M5/M1: a TOML/API list that split an option from its value still works.

    ``custom_sbatch = ["-o", "/logs/%j.out"]`` used to emit a valueless
    ``#SBATCH -o`` plus a bare ``#SBATCH /logs/%j.out`` line — a script sbatch
    rejects. All three spellings now converge on one directive.
    """

    def test_normalize_rejoins_option_and_value(self):
        from slurmate.builder import _normalize_custom_flags
        assert _normalize_custom_flags(["-o", "/logs/%j.out"]) == ["-o /logs/%j.out"]
        assert _normalize_custom_flags(["-C", "bigmem"]) == ["-C bigmem"]
        assert _normalize_custom_flags(["--reservation", "abc"]) == ["--reservation abc"]

    def test_boolean_flags_stay_separate(self):
        from slurmate.builder import _normalize_custom_flags
        assert _normalize_custom_flags(["--exclusive", "--hold"]) == [
            "--exclusive", "--hold"]

    def test_all_three_spellings_agree(self):
        from slurmate.builder import _normalize_custom_flags
        from slurmate.tui import _parse_custom_flags
        expected = ["-o /logs/%j.out"]
        assert _parse_custom_flags("-o /logs/%j.out") == expected
        assert _normalize_custom_flags(["-o /logs/%j.out"]) == expected
        assert _normalize_custom_flags(["-o", "/logs/%j.out"]) == expected

    def test_split_constraint_still_merges(self):
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "constraint": "cpu", "command": "x",
                                "custom_sbatch": ["-C", "bigmem"]})
        assert "#SBATCH --constraint=cpu&bigmem" in s
        assert s.count("#SBATCH --constraint") == 1

    def test_split_output_suppresses_auto_and_stays_one_directive(self):
        s = build_from_answers({"job_name": "j", "partition": "p",
                                "output_dir": "logs", "command": "x",
                                "custom_sbatch": ["-o", "/real/%j.log"]})
        assert "#SBATCH -o /real/%j.log" in s
        assert "#SBATCH --output=logs/" not in s
        # The value is not orphaned onto its own line any more.
        assert not any(ln == "#SBATCH /real/%j.log" for ln in s.splitlines())

    def test_newline_in_a_list_entry_is_still_folded(self):
        # Regression guard: the rejoin pass must not reintroduce a second line.
        s = build_from_answers({"job_name": "j", "partition": "p", "command": "x",
                                "custom_sbatch": ["--comment=a\nb"]})
        assert "#SBATCH --comment=\"a b\"" in s
        assert all(ln.startswith(("#", "x", "")) for ln in s.splitlines())


class TestSpaceFormValueQuoting:
    """P1: a space-form value that itself contains a space must be quoted.

    ``--comment "my job"`` reaches the builder as ``--comment my job`` (the parser
    consumes the user's quotes). Emitted bare, Slurm's directive parser splits it
    into ``--comment=my`` plus a stray ``job`` — the same defect the ``=`` form was
    already protected against.
    """

    def _flag_lines(self, raw):
        from slurmate.tui import _parse_custom_flags
        s = build_from_answers({"job_name": "j", "partition": "p", "command": "x",
                                "custom_sbatch": _parse_custom_flags(raw)})
        return [ln for ln in s.splitlines() if ln.startswith("#SBATCH")]

    def test_space_form_value_with_space_is_quoted(self):
        assert '#SBATCH --comment "my job"' in self._flag_lines('--comment "my job"')

    def test_equals_form_still_quoted(self):
        assert '#SBATCH --comment="my job"' in self._flag_lines('--comment="my job"')

    def test_space_form_without_whitespace_is_untouched(self):
        assert "#SBATCH -o /p/x.log" in self._flag_lines("-o /p/x.log")
        assert "#SBATCH -C bigmem" not in self._flag_lines("-C bigmem")  # merged by H1

    def test_unknown_option_is_left_alone(self):
        from slurmate.builder import _quote_custom_flag
        # We can't tell where the value starts, so guessing would corrupt it.
        assert _quote_custom_flag("--madeup a b") == "--madeup a b"

    def test_already_quoted_value_not_double_quoted(self):
        from slurmate.builder import _quote_custom_flag
        assert _quote_custom_flag('--comment "my job"') == '--comment "my job"'

    def test_bare_flag_untouched(self):
        from slurmate.builder import _quote_custom_flag
        assert _quote_custom_flag("--exclusive") == "--exclusive"


class TestConstraintWhitespace:
    """P4: Slurm's feature grammar has no spaces — measured.

    `sbatch -C "a100 & 384g"` fails with "Invalid feature specification" while
    `-C "a100&384g"` schedules, so a spaced expression has to be normalized rather
    than passed through.
    """

    def _c(self, **kw):
        a = {"job_name": "j", "partition": "p", "command": "x"}
        a.update(kw)
        return [ln for ln in build_from_answers(a).splitlines() if "constraint" in ln]

    def test_spaces_around_operators_removed(self):
        assert self._c(constraint="a100 & 384g") == ["#SBATCH --constraint=a100&384g"]
        assert self._c(constraint="a100 | v100") == ["#SBATCH --constraint=a100|v100"]

    def test_surrounding_whitespace_removed(self):
        # Used to emit "#SBATCH --constraint= a100 " — broken twice over.
        assert self._c(constraint=" a100 ") == ["#SBATCH --constraint=a100"]

    def test_custom_constraint_is_cleaned_too(self):
        assert self._c(custom_sbatch=["--constraint=a100 & 384g"]) == [
            "#SBATCH --constraint=a100&384g"]

    def test_merged_value_stays_clean(self):
        assert self._c(constraint="cpu ", custom_sbatch=["-C bigmem"]) == [
            "#SBATCH --constraint=cpu&bigmem"]

    def test_tight_expression_unchanged(self):
        assert self._c(constraint="a100|v100") == ["#SBATCH --constraint=a100|v100"]


class TestCustomGpuFlagReplacesAuto:
    """P5: a differing custom GPU flag replaces the auto directive, not duplicates it."""

    def _s(self, fmt, custom):
        return build_from_answers({"job_name": "j", "partition": "p", "gpus": 2,
                                   "gpu_type": "v100", "gpu_format": fmt,
                                   "custom_sbatch": custom, "command": "x"})

    def test_gres_override_replaces(self):
        s = self._s("gres_type", ["--gres=gpu:a100:8"])
        assert s.count("#SBATCH --gres") == 1
        assert "#SBATCH --gres=gpu:a100:8" in s

    def test_every_format_replaces_its_own_option(self):
        for fmt, flag, val in (
            ("gpus", "--gpus", "a100:8"),
            ("gpus_per_node", "--gpus-per-node", "a100:8"),
            ("gpus_per_task", "--gpus-per-task", "a100:8"),
            ("constraint", "--gres", "gpu:8"),
        ):
            s = self._s(fmt, [f"{flag}={val}"])
            assert s.count(f"#SBATCH {flag}") == 1, fmt
            assert f"#SBATCH {flag}={val}" in s, fmt

    def test_exact_duplicate_keeps_canonical_form(self):
        # No information in the user's spelling, so keep slurmate's `=` form.
        s = self._s("gres_type", ["--gres gpu:v100:2"])
        assert s.count("#SBATCH --gres") == 1
        assert "#SBATCH --gres=gpu:v100:2" in s

    def test_different_option_name_does_not_suppress(self):
        # --gres and --gpus are different requests to Slurm; a custom --gpus must
        # not silently remove the auto --gres.
        s = self._s("gres_type", ["--gpus=8"])
        assert "#SBATCH --gres=gpu:v100:2" in s
        assert "#SBATCH --gpus=8" in s

    def test_constraint_format_keeps_the_type_constraint(self):
        # The GPU type is a separate requirement from the GRES count.
        s = self._s("constraint", ["--gres=gpu:8"])
        assert "#SBATCH --constraint=v100" in s
        assert "#SBATCH --gres=gpu:8" in s

    def test_summary_reports_the_override(self):
        a = {"job_name": "j", "partition": "p", "gpus": 2, "gpu_type": "v100",
             "custom_sbatch": ["--gres=gpu:a100:8"], "command": "x"}
        rows = dict(job_summary_rows(a))
        assert rows["GPUs"] == "--gres=gpu:a100:8 (custom flag)"
        assert "GPU format" not in rows

    def test_summary_unchanged_without_an_override(self):
        a = {"job_name": "j", "partition": "p", "gpus": 2, "gpu_type": "v100",
             "gpu_format": "gres_type", "command": "x"}
        rows = dict(job_summary_rows(a))
        assert rows["GPUs"] == "2 × v100"
        assert rows["GPU format"] == "gres_type"


class TestNoDuplicateOrMalformedDirectives:
    """Property sweep: the defect class behind H1, M1 and P5.

    Slurm silently honours only the last of a repeated option, so a duplicate
    directive is never harmless — it either contradicts slurmate's own request or
    lies to the reader. This walks a matrix of answer combinations and asserts that
    no #SBATCH option is emitted twice and every directive is well-formed.
    """

    import re as _re
    _OPT = _re.compile(r'^--?[A-Za-z][A-Za-z0-9-]*([= ]("[^"]*"|\S+))?$')

    def _scan(self, script):
        import re
        counts, malformed = {}, []
        for ln in script.splitlines():
            if not ln.startswith("#SBATCH "):
                continue
            body = ln[len("#SBATCH "):]
            name = re.split(r"[=\s]", body.strip(), maxsplit=1)[0]
            counts[name] = counts.get(name, 0) + 1
            if not self._OPT.match(body):
                malformed.append(ln)
        return {k: v for k, v in counts.items() if v > 1}, malformed

    def test_matrix_has_no_duplicates_or_malformed_lines(self):
        import itertools
        matrix = {
            "gpus": [0, 2],
            "gpu_type": [None, "a100"],
            "gpu_format": [None, "gres_type", "constraint", "gpus_per_task"],
            "constraint": [None, "cpu", "a|b"],
            "mem_per_cpu": [None, "2G"],
            "output_dir": [None, "logs"],
            "array_spec": [None, "1-4"],
            "custom_sbatch": [None, ["--exclusive"], ["--constraint=bigmem"],
                              ["--mem=8G"], ["--mem-per-cpu=1G"],
                              ["--output=/o/%j.out"], ["-o /o/%j.out"],
                              ["-e", "/e/%j.err"], ["-C", "bigmem"],
                              ["--comment=my job"], ["--gres=gpu:h100:4"],
                              ["--gpus-per-task=4"]],
        }
        keys = list(matrix)
        checked = 0
        for combo in itertools.product(*(matrix[k] for k in keys)):
            answers = dict(zip(keys, combo))
            answers.update(job_name="j", partition="p", cpus=4, memory="16G",
                           command="x")
            script = build_from_answers(answers)
            dupes, malformed = self._scan(script)
            assert not dupes, f"duplicate {dupes} for {answers}"
            assert not malformed, f"malformed {malformed} for {answers}"
            checked += 1
        assert checked > 2000
