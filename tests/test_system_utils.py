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

    def test_validate_memory_rejects_zero_magnitude(self):
        # P3-11: a zero magnitude is invalid regardless of unit; "0G"/"0M" used
        # to slip through because the zero check only fired for the unitless "0".
        from slurmate.system_utils import validate_memory
        assert validate_memory("0G") is False
        assert validate_memory("0M") is False
        assert validate_memory("0.0G") is False
        assert validate_memory("0.5G") is True

    def test_parse_mem_to_mb(self):
        from slurmate.system_utils import _parse_mem_to_mb
        assert _parse_mem_to_mb("16G") == 16384
        assert _parse_mem_to_mb("1T") == 1048576
        assert _parse_mem_to_mb("64000M") == 64000
        assert _parse_mem_to_mb("64000") == 64000  # bare int is MB

    def test_parse_mem_to_mb_malformed_returns_zero(self):
        # P3-12: malformed forms must return 0 (unknown), not a misleading
        # partial like 16 that would masquerade as a tiny valid value.
        from slurmate.system_utils import _parse_mem_to_mb
        assert _parse_mem_to_mb("16GB") == 0
        assert _parse_mem_to_mb("16 G") == 0
        assert _parse_mem_to_mb("1.5.5G") == 0
        assert _parse_mem_to_mb("abc") == 0

    def test_validate_time_broad_formats(self):
        # P0-4: accept the full Slurm --time grammar, 1–2 digit lead fields.
        from slurmate.system_utils import validate_time
        for ok in ("30", "5:00", "2:30:00", "1-12", "1-0:00", "1-12:30:00",
                   "01:00:00", "7-00:00:00", ""):
            assert validate_time(ok) is True, ok
        for bad in ("abc", "1:2:3:4", "-5", "1-"):
            assert validate_time(bad) is False, bad

    def test_validate_time_unpadded_fields(self):
        # Slurm accepts unpadded 1-digit minute/second fields; the wizard must
        # not falsely reject them (the parser already reads them correctly),
        # while genuinely out-of-range fields (60–99) stay rejected.
        from slurmate.system_utils import validate_time
        for ok in ("5:3", "1:2:3", "5:0", "1-0:5", "1-0:5:9"):
            assert validate_time(ok) is True, ok
        for bad in ("1:60", "1:60:60", "1-99:99:99", "1:5:99"):
            assert validate_time(bad) is False, bad

    def test_mock_queue_eta_label_matches_formatter(self):
        # P3-7: the mock label is derived from _format_eta, not hand-written.
        from slurmate.system_utils import MOCK_QUEUE_INFO, _format_eta
        assert MOCK_QUEUE_INFO["eta_label"] == _format_eta(MOCK_QUEUE_INFO["eta_seconds"])
        assert MOCK_QUEUE_INFO["eta_label"] == "~1h"


class TestNaiveConfigParser:
    """P3-13: the no-tomllib/tomli fallback must not corrupt common config."""

    def test_inline_comment_stripped(self):
        from slurmate.system_utils import _parse_config_naive
        cfg = _parse_config_naive('partition = "gpu"  # fav\n')
        assert cfg["partition"] == "gpu"

    def test_unquoted_numeric_array(self):
        from slurmate.system_utils import _parse_config_naive
        cfg = _parse_config_naive("vals = [1, 2, 3]\n")
        assert cfg["vals"] == [1, 2, 3]

    def test_quoted_array_and_scalars(self):
        from slurmate.system_utils import _parse_config_naive
        cfg = _parse_config_naive('mods = ["a", "b"]\ncpus = 8\nratio = 1.5\noff = -2\n')
        assert cfg["mods"] == ["a", "b"]
        assert cfg["cpus"] == 8
        assert cfg["ratio"] == 1.5
        assert cfg["off"] == -2

    def test_hash_inside_quotes_preserved(self):
        from slurmate.system_utils import _parse_config_naive
        cfg = _parse_config_naive('name = "a#b"\n')
        assert cfg["name"] == "a#b"


class TestFetchPublicPartitionsReuse:
    def test_accepts_prefetched_all_parts(self, mocker):
        # P3-5: passing all_parts avoids the internal fetch_partitions() call.
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(
            "PartitionName=p AllowAccounts=ALL Hidden=NO\n", "", 0))
        spy = mocker.patch.object(su, "fetch_partitions")
        pre = [{"name": "p", "nodes": 1, "cpus_per_node": 1, "mem_per_node_mb": 1, "gpu_types": []}]
        out = su.fetch_public_partitions(pre)
        assert spy.call_count == 0  # did not re-fetch
        assert [p["name"] for p in out] == ["p"]

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


class TestMemHeterogeneous:
    def test_plus_suffix_parses_to_min_value(self):
        # sinfo %m emits "515000+" for heterogeneous partitions; it must parse to
        # the min value (not 0, which silently disables the memory-limit check).
        from slurmate.system_utils import _parse_mem_to_mb
        assert _parse_mem_to_mb("515000+") == 515000
        assert _parse_mem_to_mb("250000+") == 250000
        assert _parse_mem_to_mb("256G+") == 256 * 1024

    def test_still_rejects_malformed(self):
        from slurmate.system_utils import _parse_mem_to_mb
        assert _parse_mem_to_mb("16GB") == 0
        assert _parse_mem_to_mb("abc") == 0


class TestNormalizeMemoryNC:
    def test_strips_slurm_nc_suffix(self):
        # `sbatch --mem` accepts only a K/M/G/T unit; the N/C suffix would be
        # rejected, so it must be dropped from the emitted value.
        from slurmate.system_utils import normalize_memory
        assert normalize_memory("16GN") == "16G"
        assert normalize_memory("16GC") == "16G"
        assert normalize_memory("32G") == "32G"


class TestFetchUserAccountsAssoc:
    def test_uses_assoc_scoped_to_current_user(self, mocker, monkeypatch):
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        monkeypatch.setattr(su.getpass, "getuser", lambda: "alice")
        captured: dict = {}

        def fake_run(cmd, timeout=30):
            captured["cmd"] = cmd
            return "acct1\nacct2\nacct1\n", "", 0

        mocker.patch.object(su, "_run_command", side_effect=fake_run)
        accounts = su.fetch_user_accounts()
        assert "assoc" in captured["cmd"]
        assert "user=alice" in captured["cmd"]
        # de-duped, order preserved
        assert accounts == ["acct1", "acct2"]


class TestExtractFirstJson:
    def test_skips_brace_containing_banner(self):
        from slurmate.system_utils import _extract_first_json
        text = 'Welcome {user}!\n{"envs": ["/opt/conda"], "root_prefix": "/opt/conda"}\n'
        data = _extract_first_json(text)
        assert data is not None and data["envs"] == ["/opt/conda"]

    def test_none_when_no_json(self):
        from slurmate.system_utils import _extract_first_json
        assert _extract_first_json("no json here") is None


class TestFetchModulesMockGuard:
    def test_returns_mock_under_mock_mode(self):
        # conftest forces SLURMATE_MOCK=1: must not shell out.
        from slurmate.system_utils import MOCK_MODULES, fetch_available_modules
        assert fetch_available_modules() == MOCK_MODULES


class TestFetchGpuTypesMock:
    def test_known_partition_returns_specific_types(self):
        from slurmate.system_utils import fetch_gpu_types_for_partition
        assert fetch_gpu_types_for_partition("gpu-shared") == ["a100", "v100"]
        assert fetch_gpu_types_for_partition("cpu-shared") == []

    def test_unknown_partition_returns_full_list(self):
        from slurmate.system_utils import MOCK_GPU_TYPES, fetch_gpu_types_for_partition
        assert fetch_gpu_types_for_partition("mystery") == list(MOCK_GPU_TYPES)


class TestNaiveConfigSections:
    def test_section_precedence(self):
        from slurmate.system_utils import _parse_config_naive
        cfg = _parse_config_naive(
            'partition = "top"\n[defaults]\npartition = "def"\ncpus = 4\n'
            '[slurmate]\npartition = "sm"\n'
        )
        assert cfg["partition"] == "sm"  # [slurmate] > [defaults] > top-level
        assert cfg["cpus"] == 4

    def test_multiline_array(self):
        from slurmate.system_utils import _parse_config_naive
        cfg = _parse_config_naive('mods = [\n  "a",\n  "b",\n]\n')
        assert cfg["mods"] == ["a", "b"]


class TestParsingRobustness:
    def test_node_count_sums_across_state_rows(self, mocker):
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(
            "big|infinite|10|up|32|100000|(null)\n"
            "big|infinite|5|up|32|100000|(null)\n", "", 0))
        parts = su.fetch_partitions()
        big = next(p for p in parts if p["name"] == "big")
        assert big["nodes"] == 15  # summed, not max(10, 5)

    def test_mem_plus_suffix_sets_real_limit(self, mocker):
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(
            "het|infinite|4|up|32+|515000+|(null)\n", "", 0))
        parts = su.fetch_partitions()
        assert parts[0]["mem_per_node_mb"] == 515000
        assert parts[0]["cpus_per_node"] == 32

    def test_partition_has_gpu_flag_for_count_only_gres(self, mocker):
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(
            "g|infinite|4|up|32|100000|gpu:4\n", "", 0))
        parts = su.fetch_partitions()
        assert parts[0]["gpu_types"] == []  # count-only GRES has no model
        assert parts[0]["has_gpu"] is True  # but is still a GPU partition

    def test_gpu_types_multiple_models_per_node(self, mocker):
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(
            "nvlink|gpu:a100:2,gpu:v100:2\n", "", 0))
        assert su.fetch_gpu_types_for_partition("p") == ["a100", "v100"]

    def test_queue_eta_tolerates_state_flags(self, mocker):
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)

        def run(cmd, timeout=30):
            if "squeue" in cmd:
                return "", "", 0
            return "5|up|idle~\n3|up|mix*\n", "", 0

        mocker.patch.object(su, "_run_command", side_effect=run)
        info = su.fetch_queue_eta("p", req_nodes=2)
        assert info["eta_seconds"] == 0  # idle~ still counts as 5 idle nodes


class TestRunCommandOSError:
    def test_oserror_returns_nonzero(self, mocker):
        import slurmate.system_utils as su
        mocker.patch("subprocess.run", side_effect=OSError("exec format error"))
        out, err, rc = su._run_command(["sinfo"])
        assert rc == -1
        assert "exec format error" in err


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


class TestDetectGpuType:
    """_detect_gpu_type must return the GPU model, not a CPU vendor/codename,
    when a partition advertises only count-only GRES (gpu:N)."""

    def test_cpu_vendor_before_model_not_returned(self):
        from slurmate.system_utils import _detect_gpu_type
        assert _detect_gpu_type("intel,avx512,a100", "gpu:4") == "a100"
        assert _detect_gpu_type("amd,rome,a100", "gpu:4") == "a100"
        assert _detect_gpu_type("rome,a100", "gpu:4") == "a100"
        assert _detect_gpu_type("milan,h100", "gpu:4") == "h100"
        assert _detect_gpu_type("genoa,a40", "gpu:4") == "a40"
        assert _detect_gpu_type("cascade,v100", "gpu:4") == "v100"

    def test_positive_model_shapes(self):
        from slurmate.system_utils import _detect_gpu_type
        assert _detect_gpu_type("l40s", "gpu:2") == "l40s"
        assert _detect_gpu_type("rack1,t4", "gpu:1") == "t4"
        assert _detect_gpu_type("rtx6000", "gpu:1") == "rtx6000"

    def test_typed_gres_still_wins(self):
        from slurmate.system_utils import _detect_gpu_type
        assert _detect_gpu_type("rack5,gpfs,a40", "gpu:a40:2") == "a40"

    def test_cpu_generation_tags_not_returned(self):
        # Xeon "vN" / POWER "pN" share a GPU-family letter prefix but are CPUs.
        from slurmate.system_utils import _detect_gpu_type
        assert _detect_gpu_type("intel,v4,a100", "gpu:2") == "a100"
        assert _detect_gpu_type("p9,v100", "gpu:2") == "v100"
        assert _detect_gpu_type("amd,v5,h100", "gpu:8") == "h100"

    def test_real_single_digit_gpus_still_detected(self):
        from slurmate.system_utils import _detect_gpu_type
        assert _detect_gpu_type("rack,t4", "gpu:1") == "t4"
        assert _detect_gpu_type("l4", "gpu:1") == "l4"

    def test_no_gpu_returns_empty(self):
        from slurmate.system_utils import _detect_gpu_type
        assert _detect_gpu_type("intel,avx512", "(null)") == ""


class TestFractionalMemory:
    def test_fractional_normalizes_to_integer_megabytes(self):
        from slurmate.system_utils import normalize_memory, validate_memory
        assert validate_memory("1.5G") is True
        assert normalize_memory("1.5G") == "1536M"
        assert normalize_memory("2.5T") == "2621440M"
        # An integer magnitude is untouched.
        assert normalize_memory("16G") == "16G"

    def test_normalized_fractional_is_integer_only(self):
        from slurmate.system_utils import normalize_memory
        out = normalize_memory("0.5G")
        assert "." not in out and out.endswith("M")


class TestNaiveConfigParserParity:
    """The naive fallback parser must agree with tomllib on realistic input."""

    def _both(self, text):
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:
            import tomli as tomllib  # 3.10 (declared dependency)

        from slurmate.system_utils import _parse_config_naive
        return _parse_config_naive(text), tomllib.loads(text)

    def test_quoted_comma_in_array_element(self):
        naive, toml = self._both('custom = ["--foo=a,b", "--bar"]\n')
        assert naive == toml == {"custom": ["--foo=a,b", "--bar"]}

    def test_interior_inline_comment_in_multiline_array(self):
        naive, toml = self._both('modules = [\n  "cuda", # the cuda module\n  "gcc",\n]\n')
        assert naive == toml == {"modules": ["cuda", "gcc"]}

    def test_bracket_inside_string_element(self):
        naive, toml = self._both('x = [\n  "a]b",\n  "c",\n]\n')
        assert naive == toml == {"x": ["a]b", "c"]}

    def test_unclosed_array_warns_not_silent(self, capsys):
        from slurmate.system_utils import _parse_config_naive
        result = _parse_config_naive('modules = [\n  "cuda",\n  "gcc"\n')
        assert "modules" not in result
        assert "unclosed array" in capsys.readouterr().err


class TestValidateJobConfig:
    """The pure, side-effect-free validator shared by the CLI summary and the
    live TUI check."""

    GPU_PART = {"name": "gpu", "cpus_per_node": 16, "mem_per_node_mb": 65536,
                "gpu_types": ["a100"], "has_gpu": True, "timelimit": "04:00:00"}
    CPU_PART = {"name": "caslake", "cpus_per_node": 48, "mem_per_node_mb": 196608,
                "gpu_types": [], "has_gpu": False, "timelimit": "36:00:00"}

    def test_no_partition_object_is_silent(self):
        from slurmate.system_utils import validate_job_config
        assert validate_job_config({"gpus": 4}) == []
        assert validate_job_config({"_partition_obj": None, "gpus": 4}) == []

    def test_within_limits_no_issues(self):
        from slurmate.system_utils import validate_job_config
        assert validate_job_config({
            "_partition_obj": self.GPU_PART, "cpus": 4, "memory": "16G",
            "time_limit": "01:00:00", "gpus": 1, "gpu_type": "a100",
        }) == []

    def test_gpus_on_cpu_only_partition_is_error(self):
        from slurmate.system_utils import validate_job_config
        issues = validate_job_config({"_partition_obj": self.CPU_PART, "gpus": 1})
        assert ("error", "Partition 'caslake' does not support GPUs") in issues

    def test_has_gpu_suppresses_count_only_false_error(self):
        from slurmate.system_utils import validate_job_config
        part = {"name": "gpu1", "cpus_per_node": 16, "mem_per_node_mb": 0,
                "gpu_types": [], "has_gpu": True, "timelimit": None}
        issues = validate_job_config({"_partition_obj": part, "gpus": 2})
        assert all("does not support GPUs" not in m for _, m in issues)

    def test_unknown_partition_capability_no_gpu_error(self):
        from slurmate.system_utils import validate_job_config
        # Synthetic fallback for a manually-typed / unrecognized partition: no
        # has_gpu key means capability is unknown, so requesting GPUs must not
        # produce a hard "does not support GPUs" error (an overclaim).
        part = {"name": "typo", "cpus_per_node": 0, "mem_per_node_mb": 0,
                "gpu_types": [], "timelimit": None}
        assert validate_job_config({"_partition_obj": part, "gpus": 2}) == []

    def test_cpu_mem_time_over_limit_are_warnings(self):
        from slurmate.system_utils import validate_job_config
        issues = validate_job_config({
            "_partition_obj": self.GPU_PART, "cpus": 64, "memory": "128G",
            "time_limit": "08:00:00", "gpus": 0,
        })
        levels = {m.split()[0]: lvl for lvl, m in issues}
        assert levels.get("CPUs") == "warning"
        assert levels.get("Memory") == "warning"
        assert levels.get("Time") == "warning"

    def test_cpu_total_accounts_for_ntasks(self):
        from slurmate.system_utils import validate_job_config
        # 4 tasks x 8 cpus = 32 > 16 per node.
        issues = validate_job_config({
            "_partition_obj": self.GPU_PART, "cpus": 8, "ntasks_per_node": 4,
        })
        assert any("CPUs (4×8=32) exceeds" in m for _, m in issues)

    def test_gpu_type_not_in_list_is_error(self):
        from slurmate.system_utils import validate_job_config
        issues = validate_job_config(
            {"_partition_obj": self.GPU_PART, "gpus": 1, "gpu_type": "h100"})
        assert ("error", "GPU type 'h100' not in partition list (a100)") in issues

    def test_gpu_type_valid_via_extra_types(self):
        from slurmate.system_utils import validate_job_config
        # A model absent from the static list but confirmed by a live lookup
        # must not warn.
        issues = validate_job_config(
            {"_partition_obj": self.GPU_PART, "gpus": 1, "gpu_type": "h100"},
            extra_gpu_types=["h100"])
        assert all("not in partition list" not in m for _, m in issues)

    def test_gpu_type_any_never_warns(self):
        from slurmate.system_utils import validate_job_config
        issues = validate_job_config(
            {"_partition_obj": self.GPU_PART, "gpus": 1, "gpu_type": "Any"})
        assert all("not in partition list" not in m for _, m in issues)

    def test_no_known_types_suppresses_empty_list_warning(self):
        from slurmate.system_utils import validate_job_config
        # Partition advertises GPUs (has_gpu) but no parseable model; requesting a
        # specific type must not produce a "not in partition list ()" against an
        # empty list — the count-only signal, not this one, is authoritative.
        part = {"name": "gpu2", "cpus_per_node": 16, "mem_per_node_mb": 0,
                "gpu_types": [], "has_gpu": True, "timelimit": None}
        issues = validate_job_config(
            {"_partition_obj": part, "gpus": 1, "gpu_type": "a100"})
        assert all("not in partition list" not in m for _, m in issues)

    def test_stringy_and_blank_values_do_not_raise(self):
        from slurmate.system_utils import validate_job_config
        # Live TUI values arrive as raw strings, possibly blank mid-edit.
        assert validate_job_config({
            "_partition_obj": self.CPU_PART, "cpus": "", "memory": "",
            "time_limit": "", "gpus": "", "gpu_type": "",
        }) == []
        # A non-numeric gpus string must not crash and must not warn.
        assert validate_job_config(
            {"_partition_obj": self.CPU_PART, "gpus": "abc"}) == []


class TestNoMockLeakOnRealCluster:
    """A6: demo data appears only under SLURMATE_MOCK, never as a real-cluster fallback."""

    def test_empty_when_tools_absent_and_not_mock(self, monkeypatch, mocker):
        import slurmate.system_utils as su
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        mocker.patch.object(su, "is_tool_available", return_value=False)
        assert su.fetch_user_accounts() == []
        assert su.fetch_partitions() == []
        assert su.fetch_public_partitions() == []
        assert su.fetch_gpu_types_for_partition("gpu") == []

    def test_modules_empty_on_probe_failure(self, monkeypatch, mocker):
        import slurmate.system_utils as su
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        mocker.patch.object(su, "_run_command", return_value=("", "no module", 1))
        assert su.fetch_available_modules() == []

    def test_still_mock_under_mock_mode(self):
        # conftest forces SLURMATE_MOCK=1: demo data stays available for demos.
        from slurmate.system_utils import MOCK_ACCOUNTS, fetch_user_accounts
        assert fetch_user_accounts() == list(MOCK_ACCOUNTS)

    def test_queue_eta_unknown_when_tools_absent(self, monkeypatch, mocker):
        import slurmate.system_utils as su
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        mocker.patch.object(su, "is_tool_available", return_value=False)
        info = su.fetch_queue_eta("gpu")
        assert info["eta_label"] == "unknown"
        assert info["running"] == 0 and info["pending"] == 0


class TestModuleParseLmod:
    """A9: Lmod terse extras (trailing '/', tag/alias markers) are cleaned out."""

    def test_strips_lmod_extras(self, monkeypatch, mocker):
        import slurmate.system_utils as su
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        out = "/opt/modulefiles:\ngcc/\ngcc/12.2 (D)\nopenmpi/4.1 (@ompi)\npython/3.11\n"
        mocker.patch.object(su, "_run_command", return_value=(out, "", 0))
        mods = su.fetch_available_modules()
        assert "gcc" in mods          # trailing "/" stripped -> family short name
        assert "gcc/12.2" in mods
        assert "python/3.11" in mods
        assert "(D)" not in mods
        assert "(@ompi)" not in mods
        assert all(not m.endswith(":") for m in mods)
