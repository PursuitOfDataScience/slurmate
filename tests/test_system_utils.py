"""Tests for the Slurm system utilities (mock mode)."""

from slurmate.system_utils import (
    MOCK_ACCOUNTS,
    MOCK_CONDA_ENVS,
    MOCK_GPU_TYPES,
    MOCK_MODULES,
    MOCK_PARTITIONS,
    MOCK_QUEUE_INFO,
    fetch_available_modules,
    fetch_conda_envs,
    fetch_gpu_types_for_partition,
    fetch_known_qos,
    fetch_partitions,
    fetch_public_partitions,
    fetch_qos_for_partition,
    fetch_queue_eta,
    fetch_user_accounts,
    submit_sbatch,
)


class TestFetchPartitions:
    def test_fetch_partitions_returns_list(self):
        parts = fetch_partitions()
        assert isinstance(parts, list)
        assert all("name" in p for p in parts)

    def test_fetch_public_partitions(self):
        public = fetch_public_partitions()
        assert len(public) > 0
        for p in public:
            assert p.get("is_public", False) is True

    def test_mock_partitions_have_expected_keys(self):
        for p in MOCK_PARTITIONS:
            assert "name" in p
            assert "cpus_per_node" in p
            assert "mem_per_node_mb" in p
            assert "gpu_types" in p
            assert "nodes" in p


class TestFetchQos:
    def test_fetch_known_qos(self):
        qos = fetch_known_qos()
        assert len(qos) > 0
        assert "normal" in qos

    def test_fetch_qos_for_partition_mock(self):
        qos = fetch_qos_for_partition("gpu-shared")
        assert isinstance(qos, list)


class TestFetchGpuTypes:
    def test_fetch_gpu_types_returns_list(self):
        types = fetch_gpu_types_for_partition("gpu-shared")
        assert isinstance(types, list)

    def test_mock_gpu_types_exist(self):
        assert len(MOCK_GPU_TYPES) > 0


class TestFetchCondaEnvs:
    def test_fetch_conda_envs_returns_list(self):
        envs = fetch_conda_envs()
        assert isinstance(envs, list)
        assert all(isinstance(e, str) for e in envs)

    def test_mock_conda_envs(self):
        assert "pytorch" in MOCK_CONDA_ENVS


class TestFetchModules:
    def test_fetch_modules_returns_list(self):
        mods = fetch_available_modules()
        assert isinstance(mods, list)
        assert len(mods) > 0

    def test_mock_modules(self):
        assert "python/anaconda" in MOCK_MODULES


class TestFetchAccounts:
    def test_fetch_accounts_returns_list(self):
        accounts = fetch_user_accounts()
        assert isinstance(accounts, list)
        assert len(accounts) > 0

    def test_mock_accounts(self):
        assert "my_lab" in MOCK_ACCOUNTS


class TestFetchQueueEta:
    def test_fetch_queue_eta_returns_dict(self):
        info = fetch_queue_eta("gpu-shared", req_nodes=1)
        assert isinstance(info, dict)
        assert "running" in info
        assert "pending" in info
        assert "eta_seconds" in info
        assert "eta_label" in info

    def test_mock_queue_info(self):
        assert MOCK_QUEUE_INFO["running"] >= 0
        assert MOCK_QUEUE_INFO["eta_seconds"] >= 0

    def test_queue_eta_format(self):
        from slurmate.system_utils import _format_eta
        assert _format_eta(0) == "now"
        assert _format_eta(30) == "~30s"
        assert _format_eta(300) == "~5min"
        assert _format_eta(7200) == "~2h"


class TestSubmitSbatch:
    def test_submit_in_mock_mode(self):
        ret, out, err = submit_sbatch("#!/bin/bash\necho hi")
        assert ret == 0
        assert "not available" in err

    def test_submit_creates_log_directories(self, tmp_path):
        out_dir = tmp_path / "test_out_dir"
        err_dir = tmp_path / "test_err_dir"
        assert not out_dir.exists()
        assert not err_dir.exists()
        
        script = f"""#!/bin/bash
#SBATCH --output={out_dir}/job-%j.out
#SBATCH --error={err_dir}/job-%j.err
echo hello
"""
        submit_sbatch(script)
        
        assert out_dir.exists()
        assert err_dir.exists()


class TestHelpers:
    def test_validate_memory(self):
        from slurmate.system_utils import validate_memory
        assert validate_memory("16G") is True
        assert validate_memory("64000M") is True
        assert validate_memory("1T") is True
        assert validate_memory("") is False
        assert validate_memory("0") is False
        assert validate_memory("abc") is False

    def test_parse_mem_to_mb(self):
        from slurmate.system_utils import _parse_mem_to_mb
        assert _parse_mem_to_mb("16G") == 16384
        assert _parse_mem_to_mb("1T") == 1048576
        assert _parse_mem_to_mb("64000M") == 64000

    def test_parse_slurm_time(self):
        from slurmate.system_utils import _parse_slurm_time_to_minutes
        assert _parse_slurm_time_to_minutes("01:00:00") == 60.0
        assert _parse_slurm_time_to_minutes("02:30:00") == 150.0
        assert _parse_slurm_time_to_minutes("1-00:00:00") == 1440.0

    def test_detect_gpu_type(self):
        from slurmate.system_utils import _detect_gpu_type
        # 1. Model from gpu:MODEL:N
        assert _detect_gpu_type("", "gpu:a100:4") == "a100"
        assert _detect_gpu_type("", "gpu:H100:4") == "H100"  # case preserved
        assert _detect_gpu_type("", "gpu:mi300x:8") == "mi300x"
        # 2. Count-only GRES (gpu:N) — scan features with negative filter
        assert _detect_gpu_type("a100", "gpu:4") == "a100"
        assert _detect_gpu_type("gold-6346,256g,a100", "gpu:4") == "a100"
        assert _detect_gpu_type("gold-6346", "gpu:4") == "gpu"
        assert _detect_gpu_type("256g", "gpu:4") == "gpu"
        # 3. Regression: micro-arch/ISA tokens must not be detected as GPU types
        assert _detect_gpu_type("avx512,skylake,a100", "gpu:4") == "a100"
        assert _detect_gpu_type("skylake", "gpu:4") == "gpu"
        assert _detect_gpu_type("avx512,sse42,fma", "gpu:4") == "gpu"
        assert _detect_gpu_type("cascadelake", "gpu:4") == "gpu"
        assert _detect_gpu_type("sapphirerapids", "gpu:4") == "gpu"
        assert _detect_gpu_type("zen3", "gpu:4") == "gpu"
        assert _detect_gpu_type("icelake,broadwell,haswell", "gpu:4") == "gpu"
        # 4. No gpu: at all → empty
        assert _detect_gpu_type("a100", "") == ""
        assert _detect_gpu_type("a30", "") == ""
        assert _detect_gpu_type("gold-6248r", "") == ""
        assert _detect_gpu_type("1536g", "") == ""
        assert _detect_gpu_type("", "") == ""

    def test_detect_gpu_type_with_known_models(self):
        from slurmate.system_utils import _detect_gpu_type
        # known_models is *preferred* — a corroborated token wins even when a
        # non-GPU label appears before it in the features string.
        assert _detect_gpu_type("rack5,gpfs,a40", "gpu:4", known_models={"a40"}) == "a40"
        # Case-insensitive corroboration, original casing preserved.
        assert _detect_gpu_type("rack5,A100", "gpu:4", known_models={"a100"}) == "A100"
        # Fallback: a real GPU model that is NOT in known_models is still
        # detected via negative filtering (regression guard — feature-only GPU
        # types must not be dropped just because some other node had a typed GRES).
        assert _detect_gpu_type("a100", "gpu:4", known_models={"a30"}) == "a100"
        assert _detect_gpu_type("gold-6346,256g,h100", "gpu:4", known_models={"a30"}) == "h100"
        # Fallback still rejects pure CPU/arch junk.
        assert _detect_gpu_type("avx512,skylake", "gpu:4", known_models={"a30"}) == "gpu"
        # Typed GRES overrides everything.
        assert _detect_gpu_type("rack5,gpfs", "gpu:a40:4", known_models={"h100"}) == "a40"


class TestLoadConfig:
    def test_mock_mode_is_hermetic(self, tmp_path, monkeypatch):
        # Even with a real config present, mock mode must ignore it.
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".slurmate.toml").write_text('account = "x"\n')
        monkeypatch.setenv("SLURMATE_MOCK", "1")
        from slurmate.system_utils import load_config
        assert load_config() == {}

    def test_reads_toml_with_section_and_types(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        (tmp_path / ".slurmate.toml").write_text(
            'partition = "gpu"\ncpus = 8\n[defaults]\nmodules = ["a", "b"]\n'
        )
        from slurmate.system_utils import load_config
        cfg = load_config()
        assert cfg["partition"] == "gpu"
        assert cfg["cpus"] == 8
        assert cfg["modules"] == ["a", "b"]

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        from slurmate.system_utils import load_config
        assert load_config() == {}
