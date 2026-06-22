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
