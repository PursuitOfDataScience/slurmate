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

    def test_submit_creates_log_directories(self, tmp_path, mocker):
        out_dir = tmp_path / "test_out_dir"
        err_dir = tmp_path / "test_err_dir"
        assert not out_dir.exists()
        assert not err_dir.exists()

        script = f"""#!/bin/bash
#SBATCH --output={out_dir}/job-%j.out
#SBATCH --error={err_dir}/job-%j.err
echo hello
"""
        # Log dirs are created only when a submission is actually going to happen,
        # so pretend sbatch exists (and stub the call itself out).
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su.subprocess, "run", return_value=mocker.Mock(
            returncode=0, stdout="12345", stderr=""))
        submit_sbatch(script)

        assert out_dir.exists()
        assert err_dir.exists()

    def test_submit_creates_log_dirs_for_space_and_short_forms(self, tmp_path, mocker):
        # L4: --output PATH (long option + space) must be recognized too, or the
        # directory goes un-created and Slurm fails the job on a missing dir.
        long_dir = tmp_path / "space_form"
        short_dir = tmp_path / "short_form"
        script = f"""#!/bin/bash
#SBATCH --output {long_dir}/job-%j.out
#SBATCH -e {short_dir}/job-%j.err
echo hello
"""
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su.subprocess, "run", return_value=mocker.Mock(
            returncode=0, stdout="12345", stderr=""))
        submit_sbatch(script)
        assert long_dir.exists()
        assert short_dir.exists()

    def test_mock_mode_creates_nothing(self, tmp_path):
        # L3: nothing is submitted in mock mode, so nothing should be written to
        # the filesystem either (this used to leave stray log dirs behind).
        out_dir = tmp_path / "should_not_exist"
        script = f"#!/bin/bash\n#SBATCH --output={out_dir}/job-%j.out\ntrue\n"
        ret, _, err = submit_sbatch(script)
        assert ret == 0 and "not available" in err
        assert not out_dir.exists()


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
        # Sorted: the flattened view unions the typed/feature sources, matching the
        # live path (which has always returned a sorted list).
        assert fetch_gpu_types_for_partition("mystery") == sorted(MOCK_GPU_TYPES)


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
        # Power-save (~) nodes still run work and must count; not-responding (*)
        # ones must not. Now checked against per-node free resources rather than
        # a state-label tally, but the flag rule is unchanged.
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_scheduler_start_estimate", return_value=None)

        def run(cmd, timeout=30):
            if "squeue" in cmd:
                return "", "", 0
            # StateLong  CPUsState  Memory  AllocMem  Gres  GresUsed
            return (
                "idle~          0/48/0/48       192000    0        (null)  gpu:0\n"
                "idle~          0/48/0/48       192000    0        (null)  gpu:0\n"
                "mix*           0/48/0/48       192000    0        (null)  gpu:0\n"
            ), "", 0

        mocker.patch.object(su, "_run_command", side_effect=run)
        info = su.fetch_queue_eta("p", req_nodes=2, cpus=8)
        assert info["eta_seconds"] == 0  # the two idle~ nodes fit
        assert info["source"] == "resources"
        # Three nodes do not: the mix* one is not schedulable.
        assert su.fetch_queue_eta("p", req_nodes=3, cpus=8)["eta_seconds"] > 0

    def test_queue_eta_does_not_claim_immediate_when_gpus_are_all_allocated(self, mocker):
        # The bug this replaces: every node MIXED with idle cores, so a state-label
        # tally said "immediate" — while every GPU on them was already allocated.
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_scheduler_start_estimate", return_value=None)

        def run(cmd, timeout=30):
            if "squeue" in cmd:
                return "RUNNING|1|2|1\nPENDING|0|2|1\n", "", 0
            return (
                "mixed          28/20/0/48      192000    130000   gpu:4   gpu:4\n"
                "mixed          32/16/0/48      192000    120000   gpu:4   gpu:4\n"
            ), "", 0

        mocker.patch.object(su, "_run_command", side_effect=run)
        gpu_job = su.fetch_queue_eta("p", req_nodes=1, cpus=8, gpus_per_node=1)
        assert gpu_job["eta_seconds"] > 0, "no GPU is free — must not report 'now'"
        assert gpu_job["eta_label"] != "now"
        # The same nodes DO have spare cores, so a CPU-only job still starts now.
        cpu_job = su.fetch_queue_eta("p", req_nodes=1, cpus=8)
        assert cpu_job["eta_seconds"] == 0

    def test_queue_eta_prefers_the_scheduler_when_available(self, mocker):
        # sbatch --test-only is authoritative: it sees QOS/account limits and the
        # site job_submit plugin, none of which sinfo can show.
        from datetime import datetime, timedelta

        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        start = (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")

        def run(cmd, timeout=30):
            if "sbatch" in cmd:
                return "", f"sbatch: Job 42 to start at {start} using 1 processors\n", 0
            if "squeue" in cmd:
                return "", "", 0
            return "idle           0/48/0/48       192000    0        (null)  gpu:0\n", "", 0

        mocker.patch.object(su, "_run_command", side_effect=run)
        info = su.fetch_queue_eta("p", req_nodes=1, cpus=8, account="acct")
        assert info["source"] == "scheduler"
        # ~2 hours, not the "now" the free idle node would otherwise imply.
        assert 7000 < info["eta_seconds"] <= 7200

    def test_queue_eta_falls_back_when_scheduler_rejects(self, mocker):
        # A rejected --test-only (bad account, QOS violation) yields no start time;
        # that must fall through to the resource count, not fabricate one.
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)

        def run(cmd, timeout=30):
            if "sbatch" in cmd:
                return "", "sbatch: error: Account is not specified\n", 1
            if "squeue" in cmd:
                return "", "", 0
            return "idle           0/48/0/48       192000    0        (null)  gpu:0\n", "", 0

        mocker.patch.object(su, "_run_command", side_effect=run)
        info = su.fetch_queue_eta("p", req_nodes=1, cpus=8)
        assert info["source"] == "resources"
        assert info["eta_seconds"] == 0


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


class TestGpuTypeProvenance:
    """H2: a model found only in node FEATURES is not a GRES type.

    Measured on a real count-only-GRES partition: `--gres=gpu:a100:1` fails with
    "Requested node configuration is not available", while `--gres=gpu:1
    --constraint=a100` schedules. So the two sources must be tracked separately.
    """

    def _sinfo(self, mocker, out):
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=(out, "", 0))
        return su

    def test_count_only_gres_yields_feature_source(self, mocker):
        su = self._sinfo(mocker, "gold-6248r,384g,a100|gpu:4\n")
        assert su.fetch_gpu_type_sources("gpu") == {"typed": [], "feature": ["a100"]}
        # The flattened view (used by pickers) still lists it.
        assert su.fetch_gpu_types_for_partition("gpu") == ["a100"]

    def test_typed_gres_yields_typed_source(self, mocker):
        su = self._sinfo(mocker, "gold-6346,256g,a30|gpu:a30:4\n")
        assert su.fetch_gpu_type_sources("p") == {"typed": ["a30"], "feature": []}

    def test_typed_elsewhere_promotes_out_of_feature_only(self, mocker):
        # Mixed partition: one node types the GRES, another only features it. The
        # model IS requestable as a GRES type, so it must not be flagged.
        su = self._sinfo(mocker, "x,a100|gpu:a100:4\ny,a100|gpu:4\n")
        assert su.fetch_gpu_type_sources("p") == {"typed": ["a100"], "feature": []}

    def test_mock_types_count_as_typed(self):
        from slurmate.system_utils import fetch_gpu_type_sources
        # Demos/tests must not see a spurious format mismatch.
        assert fetch_gpu_type_sources("gpu-shared")["feature"] == []

    def test_unreachable_sinfo_returns_empty_sources(self, mocker):
        import slurmate.system_utils as su
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(su, "_run_command", return_value=("", "boom", 1))
        assert su.fetch_gpu_type_sources("p") == {"typed": [], "feature": []}


class TestFeatureOnlyGpuFormatValidation:
    """H2: requesting a feature-only model through a GRES-naming format is an error."""

    PART = {"name": "gpu", "gpu_types": [], "has_gpu": True, "cpus_per_node": 0,
            "mem_per_node_mb": 0, "timelimit": None}

    def _issues(self, fmt, **kw):
        from slurmate.system_utils import validate_job_config
        answers = {"_partition_obj": self.PART, "gpus": 1, "gpu_type": "a100",
                   "gpu_format": fmt}
        answers.update(kw)
        return validate_job_config(answers, extra_gpu_types=["a100"],
                                   feature_only_gpu_types=["a100"])

    def test_gres_type_is_an_error(self):
        errs = [m for lvl, m in self._issues("gres_type") if lvl == "error"]
        assert any("node feature" in m and "constraint" in m for m in errs)

    def test_every_type_naming_format_is_an_error(self):
        for fmt in ("gres_type", "gpus", "gpus_per_node", "gpus_per_task"):
            assert any(lvl == "error" for lvl, _ in self._issues(fmt)), fmt

    def test_constraint_format_is_accepted(self):
        assert self._issues("constraint") == []

    def test_default_format_is_checked(self, monkeypatch):
        # gpu_format unset => the builder's default (gres_type) applies, so the
        # mismatch must still be reported rather than slipping through.
        monkeypatch.delenv("SLURMATE_GPU_FORMAT", raising=False)
        from slurmate.system_utils import validate_job_config
        issues = validate_job_config(
            {"_partition_obj": self.PART, "gpus": 1, "gpu_type": "a100"},
            extra_gpu_types=["a100"], feature_only_gpu_types=["a100"])
        assert any(lvl == "error" for lvl, _ in issues)

    def test_typed_model_is_not_flagged(self):
        from slurmate.system_utils import validate_job_config
        assert validate_job_config(
            {"_partition_obj": self.PART, "gpus": 1, "gpu_type": "a100",
             "gpu_format": "gres_type"},
            extra_gpu_types=["a100"], feature_only_gpu_types=[]) == []

    def test_no_gpus_requested_is_not_flagged(self):
        assert self._issues("gres_type", gpus=0) == []


class TestGpuTypeCaseSensitivity:
    """M6: Slurm node features are case-sensitive; validation lowercased."""

    PART = {"name": "gpu", "gpu_types": ["A100"], "has_gpu": True,
            "cpus_per_node": 0, "mem_per_node_mb": 0, "timelimit": None}

    def _issues(self, gpu_type):
        from slurmate.system_utils import validate_job_config
        return validate_job_config({"_partition_obj": self.PART, "gpus": 1,
                                    "gpu_type": gpu_type,
                                    "gpu_format": "constraint"})

    def test_case_mismatch_warns(self):
        issues = self._issues("a100")
        assert any(lvl == "warning" and "case-sensitive" in m for lvl, m in issues)
        # Still not an error: it does name a model the partition advertises.
        assert all(lvl != "error" for lvl, _ in issues)

    def test_exact_case_is_silent(self):
        assert self._issues("A100") == []

    def test_unknown_model_still_errors_not_warns(self):
        issues = self._issues("h100")
        assert any(lvl == "error" and "not in partition list" in m for lvl, m in issues)
        assert all("case-sensitive" not in m for _, m in issues)


class TestInfraTokensAreNotGpuModels:
    """M2: fabric / rack / form-factor feature tokens are not GPU models."""

    def test_infiniband_generations_rejected(self):
        from slurmate.system_utils import _detect_gpu_type
        for tok in ("hdr100", "hdr200", "edr", "fdr", "ndr", "qdr", "roce", "ib0"):
            assert _detect_gpu_type(tok, "gpu:4") == "gpu", tok

    def test_rack_and_form_factor_labels_rejected(self):
        from slurmate.system_utils import _detect_gpu_type
        for tok in ("rack2", "rack", "row3", "pod1", "chassis4", "blade2",
                    "sxm4", "pcie", "nvlink", "dlc"):
            assert _detect_gpu_type(tok, "gpu:4") == "gpu", tok

    def test_real_model_wins_over_earlier_infra_label(self):
        from slurmate.system_utils import _detect_gpu_type
        # The old shape regex matched "b12"/"t2"/"p2" and returned them because
        # they came first in the feature list, beating the real model.
        assert _detect_gpu_type("b12,a100", "gpu:4") == "a100"
        assert _detect_gpu_type("t2,a100", "gpu:4") == "a100"
        assert _detect_gpu_type("p2,a100", "gpu:4") == "a100"
        assert _detect_gpu_type("rack2,edr,v100", "gpu:4") == "v100"
        assert _detect_gpu_type("gold6248,avx512,hdr100,768g,a100", "gpu:4") == "a100"

    def test_fabric_only_features_fall_back_to_generic(self):
        from slurmate.system_utils import _detect_gpu_type
        assert _detect_gpu_type("gold6248,avx512,hdr100,768g", "gpu:4") == "gpu"

    def test_short_real_models_still_detected(self):
        from slurmate.system_utils import _detect_gpu_type
        for tok in ("t4", "l4", "a30", "a40", "l40s", "k80", "mi50", "h200"):
            assert _detect_gpu_type(f"rack1,{tok}", "gpu:4") == tok, tok

    def test_unknown_future_model_shape_still_detected(self):
        from slurmate.system_utils import _detect_gpu_type
        # Not in the known list, but shaped like one (3+ digits).
        assert _detect_gpu_type("epyc-9335,768g,h300", "gpu:4") == "h300"
        assert _detect_gpu_type("mi450", "gpu:4") == "mi450"

    def test_case_is_preserved(self):
        from slurmate.system_utils import _detect_gpu_type
        # Features are case-sensitive, so the token must come back verbatim.
        assert _detect_gpu_type("epyc-9335,768g,H200,DLC", "gpu:4") == "H200"
        assert _detect_gpu_type("gold-6346,512g,L40S", "gpu:4") == "L40S"


class TestSbatchLogPathForms:
    """L4 + M1: every spelling sbatch accepts, and last-wins resolution."""

    def test_all_forms_parsed(self):
        from slurmate.system_utils import _sbatch_log_path
        assert _sbatch_log_path("#SBATCH --output=/a/%j.out") == "/a/%j.out"
        assert _sbatch_log_path("#SBATCH --output /a/%j.out") == "/a/%j.out"
        assert _sbatch_log_path("#SBATCH -o /a/%j.out") == "/a/%j.out"
        assert _sbatch_log_path("#SBATCH --error /a/%j.err") == "/a/%j.err"
        assert _sbatch_log_path("#SBATCH -e /a/%j.err") == "/a/%j.err"
        assert _sbatch_log_path('#SBATCH --output="/a b/%j.out"') == "/a b/%j.out"

    def test_non_log_directives_and_blanks_ignored(self):
        from slurmate.system_utils import _sbatch_log_path
        assert _sbatch_log_path("#SBATCH --mem=16G") == ""
        assert _sbatch_log_path("#SBATCH -o") == ""
        assert _sbatch_log_path("echo hi") == ""
        assert _sbatch_log_path("#SBATCH --open-mode=append") == ""

    def test_kind_filter(self):
        from slurmate.system_utils import _sbatch_log_path
        assert _sbatch_log_path("#SBATCH -e /a.err", kind="output") == ""
        assert _sbatch_log_path("#SBATCH -e /a.err", kind="error") == "/a.err"

    def test_effective_log_path_takes_the_last(self):
        from slurmate.system_utils import effective_log_path
        script = ("#!/bin/bash\n#SBATCH --output=logs/j-%j.out\n"
                  "#SBATCH --error=logs/j-%j.err\n#SBATCH -o /real/%j.log\ntrue\n")
        assert effective_log_path(script, "output") == "/real/%j.log"
        assert effective_log_path(script, "error") == "logs/j-%j.err"

    def test_effective_log_path_empty_when_absent(self):
        from slurmate.system_utils import effective_log_path
        assert effective_log_path("#!/bin/bash\ntrue\n") == ""


class TestMemoryUnitStrictness:
    def test_petabyte_rejected(self):
        from slurmate.system_utils import validate_memory
        # `sbatch --mem` documents K/M/G/T and rejects 16P client-side with
        # "Invalid --mem specification", so accepting it only deferred the failure.
        assert validate_memory("16P") is False
        assert validate_memory("0.5P") is False

    def test_supported_units_still_accepted(self):
        from slurmate.system_utils import validate_memory
        for v in ("16K", "512M", "16G", "1T", "16GN", "1.5G", "64000"):
            assert validate_memory(v) is True, v


class TestEffectiveMemoryValidation:
    """P2/P3: validate the memory the SCRIPT requests, not the raw answer.

    The builder gives --mem-per-cpu precedence over --mem and lets a custom flag
    suppress both, so checking `answers["memory"]` unconditionally warned about a
    value the job never requests while staying silent about the one it does.
    """

    PART = {"name": "cpu-shared", "cpus_per_node": 32, "mem_per_node_mb": 131072,
            "gpu_types": [], "has_gpu": False, "timelimit": None}

    def _v(self, **kw):
        from slurmate.system_utils import validate_job_config
        a = {"_partition_obj": self.PART, "cpus": 8}
        a.update(kw)
        return validate_job_config(a)

    def test_mem_per_cpu_total_is_checked(self):
        # 64G/CPU x 8 cores = 512G on a 128G node: silent before.
        warns = [m for lvl, m in self._v(mem_per_cpu="64G") if lvl == "warning"]
        assert any("64G/CPU × 8 cores" in m and "exceeds partition limit" in m
                   for m in warns)

    def test_mem_per_cpu_within_limit_is_silent(self):
        assert self._v(mem_per_cpu="2G") == []

    def test_mem_per_cpu_accounts_for_tasks_per_node(self):
        # 8 cpus/task x 4 tasks = 32 cores x 8G = 256G > 128G.
        warns = [m for lvl, m in self._v(mem_per_cpu="8G", ntasks_per_node=4)
                 if lvl == "warning"]
        assert any("8G/CPU × 32 cores" in m for m in warns)

    def test_superseded_memory_is_not_warned_about(self):
        # --mem-per-cpu wins, so the unused --mem value must not raise a warning.
        assert self._v(memory="512G", mem_per_cpu="2G") == []

    def test_custom_mem_flag_is_what_gets_checked(self):
        warns = [m for lvl, m in self._v(memory="16G", custom_sbatch=["--mem=512G"])
                 if lvl == "warning"]
        assert any("Memory (512G)" in m for m in warns)

    def test_custom_mem_flag_suppresses_the_answer_check(self):
        # The auto --mem=512G is suppressed by the custom flag, so don't warn on it.
        assert self._v(memory="512G", custom_sbatch=["--mem=8G"]) == []

    def test_plain_memory_check_unchanged(self):
        warns = [m for lvl, m in self._v(memory="512G") if lvl == "warning"]
        assert any("Memory (512G) exceeds partition limit (131072 MB per node)" in m
                   for m in warns)

    def test_no_cpus_answer_does_not_crash(self):
        from slurmate.system_utils import validate_job_config
        assert validate_job_config({"_partition_obj": self.PART,
                                    "mem_per_cpu": "2G"}) == []
