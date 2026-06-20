"""Tests for the sbatch script builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from slurmify.builder import build_sbatch_script, estimate_su


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
        assert "#SBATCH --gres=gpu:4" in script
        assert "#SBATCH --constraint=a100" in script
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


class TestEstimateSu:
    def test_basic_estimate(self):
        result = estimate_su(4, "01:00:00", 1)
        assert isinstance(result, str)
        # 4 CPUs * 1 hour = 4 SU
        assert result is not None

    def test_zero_cpus(self):
        result = estimate_su(0, "01:00:00", 1)
        assert result is not None

    def test_multi_node(self):
        result = estimate_su(8, "02:00:00", 4)
        assert result is not None
