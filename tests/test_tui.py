"""Tests for the TUI wizard step definitions and logic."""

from slurmate.system_utils import normalize_memory
from slurmate.tui import STEPS, Wizard, _parse_custom_flags


def _idx(key):
    return next(i for i, s in enumerate(STEPS) if s.key == key)


class TestStepDefinitions:
    def test_all_steps_have_keys(self):
        for s in STEPS:
            assert s.key, f"Step missing key: {s.title}"
            assert s.kind in ("text", "select", "autocomplete", "partition", "gpu_type", "gpu_format", "ntasks_per_node", "review")

    def test_no_duplicate_keys(self):
        keys = [s.key for s in STEPS]
        assert len(keys) == len(set(keys)), "Duplicate step keys found"

    def test_all_steps_have_titles(self):
        for s in STEPS:
            assert s.title, f"Step {s.key} missing title"

    def test_required_keys_have_validation(self):
        """job_name and command have inline required-field checks."""
        for s in STEPS:
            if s.key in ("job_name", "command"):
                assert s.validate is None
                # validation is handled inline in _confirm_and_next

    def test_steps_are_in_correct_order(self):
        # Order mirrors the #SBATCH directive order the builder emits, so the live
        # preview grows top-to-bottom: mem_per_cpu sits with memory, and constraint
        # after the GPU block (the builder emits --constraint right after --gres).
        expected_order = [
            "job_name", "partition", "account", "qos", "cpus",
            "memory", "mem_per_cpu", "time_limit", "nodes", "ntasks_per_node",
            "gpus", "gpu_type", "gpu_format", "constraint",
            "array_spec", "output_dir", "output_file", "custom_sbatch",
            "modules", "env_type", "env_name", "command", "review",
        ]
        assert [s.key for s in STEPS] == expected_order

    def test_subtitle_is_string(self):
        for s in STEPS:
            assert isinstance(s.subtitle, str)


class TestWizardConstruction:
    def test_wizard_can_be_created(self):
        w = Wizard()
        assert w.idx == 0
        assert w.answers == {}
        assert not w.submitted

    def test_current_step_is_first_step(self):
        w = Wizard()
        assert w.current_step.key == "job_name"




class TestWizardNavigation:
    def test_advance_increments_idx(self):
        w = Wizard()
        old = w.idx
        w._advance()
        assert w.idx == old + 1

    def test_go_back_decrements_idx(self):
        w = Wizard()
        w.idx = 5
        w._go_back()
        assert w.idx == 4

    def test_go_back_stays_at_zero(self):
        w = Wizard()
        w.idx = 0
        w._go_back()
        assert w.idx == 0

    def test_coerce_cpus(self):
        w = Wizard()
        s = STEPS[4]  # cpus
        assert w._coerce("8", s) == 8
        assert w._coerce("", s) == 4

    def test_coerce_gpus_defaults_zero(self):
        w = Wizard()
        s = STEPS[_idx("gpus")]
        assert w._coerce("4", s) == 4
        assert w._coerce("", s) == 0

    def test_coerce_memory(self):
        w = Wizard()
        s = STEPS[5]  # memory
        assert w._coerce("32G", s) == "32G"
        assert w._coerce("64000", s) == "64000M"

    def test_coerce_modules(self):
        w = Wizard()
        s = STEPS[_idx("modules")]
        assert w._coerce("python/3.10,cuda/12.0", s) == ["python/3.10", "cuda/12.0"]
        assert w._coerce("", s) is None

    def test_coerce_custom_sbatch_returns_list(self):
        # Regression: a raw string here gets iterated char-by-char by the builder
        # (#SBATCH m, #SBATCH i, …); it must be parsed into a flag list.
        w = Wizard()
        s = STEPS[_idx("custom_sbatch")]
        assert w._coerce("--exclusive, --reservation=abc", s) == [
            "--exclusive", "--reservation=abc",
        ]
        assert w._coerce("midway3", s) == ["--midway3"]
        assert w._coerce("", s) is None


class TestWizardStepKinds:
    def test_text_step_kind_check(self):
        w = Wizard()
        for i, s in enumerate(STEPS):
            if s.kind in ("text", "autocomplete", "ntasks_per_node"):
                w.idx = i
                assert w._is_text_active(), f"Step {s.key} should be text-active"

    def test_select_step_kind_check(self):
        w = Wizard()
        for i, s in enumerate(STEPS):
            if s.kind in ("select", "gpu_format"):
                w.idx = i
                assert w._is_select_active(), f"Step {s.key} should be select-active"

    def test_autocomplete_step_kind_check(self):
        w = Wizard()
        for i, s in enumerate(STEPS):
            if s.kind == "autocomplete":
                w.idx = i
                assert w._is_text_active(), f"Step {s.key} should be autocomplete (text)"

    def test_partition_step_kind(self):
        w = Wizard()
        w.idx = _idx("partition")
        w._on_enter_step()
        assert w._is_select_active()  # partition starts in select sub-mode

    def test_partition_text_submode(self):
        w = Wizard()
        w.idx = _idx("partition")
        w.step_cache["partition_sub"] = "text"
        assert w._is_text_active()

    def test_gpu_type_submode_select(self):
        w = Wizard()
        w.idx = _idx("gpu_type")
        w.answers["gpus"] = 2
        w.answers["partition"] = "gpu-shared"
        w._on_enter_step()
        assert w._is_select_active()

    def test_gpu_type_skip_when_zero_gpus(self):
        w = Wizard()
        w.idx = _idx("gpu_type")
        w.answers["gpus"] = 0
        old_idx = w.idx
        w._on_enter_step()
        assert w.idx > old_idx  # should auto-advance


class TestPartitionSubFlow:
    def test_setup_partition_creates_radio(self):
        w = Wizard()
        w.idx = _idx("partition")
        w._on_enter_step()
        assert w.step_cache.get("partition_sub") == "select"
        assert hasattr(w.radio_list, "values")
        values = [v for v, _ in w.radio_list.values]
        assert "Enter partition name manually..." in values

    def test_partition_go_back_from_text(self):
        w = Wizard()
        w.idx = _idx("partition")
        w.step_cache["partition_sub"] = "text"
        w._go_back()
        assert w.step_cache.get("partition_sub") == "select"

    def test_partition_go_back_from_all(self):
        w = Wizard()
        w.idx = _idx("partition")
        w.step_cache["partition_sub"] = "all"
        w._go_back()
        assert w.step_cache.get("partition_sub") == "select"


class TestHelpers:
    def test_normalize_memory(self):
        assert normalize_memory("16") == "16M"
        assert normalize_memory("32G") == "32G"
        assert normalize_memory("64000M") == "64000M"

    def test_parse_custom_flags(self):
        result = _parse_custom_flags("--exclusive,--reservation=abc")
        assert result == ["--exclusive", "--reservation=abc"]

        result = _parse_custom_flags("exclusive, #SBATCH --reservation=abc")
        assert result == ["--exclusive", "--reservation=abc"]

    def test_parse_custom_flags_space_separated(self):
        # Space-separated flags each become their own directive (not one combined).
        assert _parse_custom_flags("--exclusive --reservation=abc") == [
            "--exclusive", "--reservation=abc",
        ]
        # A bare word after an option that takes NO value is its own option — we
        # must not invent --exclusive=<node>. (An option that does take a value
        # absorbs it instead; see TestSpaceSeparatedCustomFlags.)
        assert _parse_custom_flags("--nodelist=midway3-0100") == ["--nodelist=midway3-0100"]
        assert _parse_custom_flags("--exclusive midway3-0100") == [
            "--exclusive", "--midway3-0100",
        ]
        assert _parse_custom_flags("exclusive") == ["--exclusive"]
        # Both flags together, and a comma inside a value (node list) survives.
        assert _parse_custom_flags("--exclusive --exclude=node1,node2") == [
            "--exclusive", "--exclude=node1,node2",
        ]
        assert _parse_custom_flags("--exclusive,--exclude=node1") == [
            "--exclusive", "--exclude=node1",
        ]

    def test_parse_custom_flags_empty(self):
        assert _parse_custom_flags("") == []

    def test_parse_custom_flags_whitespace(self):
        assert _parse_custom_flags("  ,  ,  ") == []

    def test_parse_custom_flags_quoted_value_with_space(self):
        # Regression: a value quoted to hold a space stays ONE flag. The old
        # whitespace split turned --comment="my job" into two broken directives
        # (--comment="my and --job").
        assert _parse_custom_flags('--comment="my job"') == ["--comment=my job"]
        assert _parse_custom_flags('--comment="my job",--exclusive') == [
            "--comment=my job", "--exclusive",
        ]
        assert _parse_custom_flags('--wrap="sleep 60 && echo hi"') == [
            "--wrap=sleep 60 && echo hi",
        ]
        # A comma inside a quoted value is preserved (not a flag separator).
        assert _parse_custom_flags('--comment="a, b"') == ["--comment=a, b"]

    def test_parse_custom_flags_unbalanced_quote_falls_back(self):
        # A half-typed value with no closing quote must not raise; it degrades
        # to a plain whitespace split rather than dropping everything.
        assert _parse_custom_flags('--comment="oops') == ['--comment="oops']


def _step(key):
    """Look a step up by key — index arithmetic breaks whenever a step is added."""
    return next(s for s in STEPS if s.key == key)


class TestStepValidation:
    def test_validate_cpus_valid(self):
        s = _step("cpus")
        assert s.validate is not None
        assert s.validate("4")
        assert s.validate("32")
        assert not s.validate("0")
        assert not s.validate("-1")
        assert not s.validate("abc")

    def test_validate_memory_valid(self):
        s = _step("memory")
        assert s.validate is not None
        assert s.validate("16G")
        assert s.validate("64000M")
        assert s.validate("1T")
        assert not s.validate("abc")

    def test_validate_time_valid(self):
        s = _step("time_limit")
        assert s.validate is not None
        assert s.validate("01:00:00")
        assert s.validate("1-00:00:00")
        assert s.validate("")
        assert not s.validate("abc")

    def test_mem_per_cpu_step_validates_memory_and_allows_blank(self):
        # M4: --mem-per-cpu is now reachable interactively. Blank is allowed (it
        # means "use --mem"), a malformed value is rejected like any memory field.
        s = _step("mem_per_cpu")
        assert s.validate is not None
        assert s.validate("2G")
        assert not s.validate("abc")
        assert s.default == ""


class TestWizardConfigDefaults:
    def test_config_does_not_mutate_global_steps(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        (tmp_path / ".slurmate.toml").write_text('cpus = 99\nenv_type = "venv"\n')
        before = {s.key: s.default for s in STEPS}
        w = Wizard()
        after = {s.key: s.default for s in STEPS}
        assert before == after  # shared STEPS must be untouched
        assert w._config_defaults["cpus"] == "99"
        # lowercase config env_type is normalized to the TUI's choice label
        assert w._config_defaults["env_type"] == "Virtualenv (venv)"


class TestRadioSelection:
    def test_reads_highlighted_row_not_initial_value(self):
        # Regression: the wizard handles Enter eagerly, so RadioList.current_value
        # never syncs to the navigated row. Selecting must read _selected_index.
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        w.idx = next(i for i, s in enumerate(STEPS) if s.key == "gpus")
        s = STEPS[w.idx]
        w._setup_select(s, None)
        # default is "0" (index 0); arrow down to "4"
        target = w.radio_list.values.index(("4", "4"))
        w.radio_list._selected_index = target
        assert w._radio_value() == "4"

    def test_set_radio_default_moves_cursor(self):
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        w.idx = next(i for i, st in enumerate(STEPS) if st.key == "gpus")
        s = STEPS[w.idx]
        w._setup_select(s, "2")  # prev answer "2"
        assert w._radio_value() == "2"  # cursor sits on the default, not index 0


class TestReviewStep:
    def test_review_step_layout_does_not_crash(self):
        w = Wizard()
        w.idx = _idx("review")
        w.answers = {
            "job_name": "test", "partition": "cpu", "cpus": 4,
            "memory": "16G", "time_limit": "01:00:00", "nodes": 1,
            "gpus": 0, "command": "echo hi",
        }
        layout = w._build_layout()
        assert layout is not None


class TestFreeNavigation:
    def test_can_skip_required_empty_field(self):
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        w.idx = next(i for i, s in enumerate(STEPS) if s.key == "job_name")
        w.text_area.text = ""  # leave required job_name blank
        w._confirm_and_next()
        # advanced past it without an error, recording an empty value
        assert "error" not in w.step_cache
        assert w.idx > 0

    def test_invalid_nonempty_still_blocks(self):
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        w.idx = next(i for i, s in enumerate(STEPS) if s.key == "memory")
        start = w.idx
        w.text_area.text = "not-a-size"
        w._confirm_and_next()
        assert w.step_cache.get("error")
        assert w.idx == start  # did not advance


class TestQosCoerceAndPathCompleter:
    def test_qos_default_coerces_to_none(self):
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        qos = next(s for s in STEPS if s.key == "qos")
        assert w._coerce("Default (none)", qos) is None
        assert w._coerce("high", qos) == "high"

    def test_path_completer_completes_last_token(self, tmp_path):
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document
        from prompt_toolkit.formatted_text import fragment_list_to_text

        from slurmate.tui import LastTokenPathCompleter
        (tmp_path / "alpha.txt").write_text("x")
        (tmp_path / "beta.txt").write_text("x")
        pc = LastTokenPathCompleter()
        text = f"python {tmp_path}/al"
        comps = list(pc.get_completions(Document(text, len(text)), CompleteEvent()))
        # completion text is the suffix after "al"; display shows the full name
        names = [fragment_list_to_text(c.display) for c in comps]
        assert any("alpha.txt" in n for n in names)
        assert all("beta.txt" not in n for n in names)

    def test_path_steps_flagged(self):
        from slurmate.tui import STEPS
        path_keys = {s.key for s in STEPS if getattr(s, "path", False)}
        assert {"output_dir", "output_file", "command"} <= path_keys


class TestCoerceConfigDefaults:
    def test_cleared_field_falls_back_to_config(self, tmp_path, monkeypatch):
        # P3-10: clearing a config-defaulted field returns the configured value,
        # not the bare hard-coded literal.
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        (tmp_path / ".slurmate.toml").write_text('cpus = 8\nnodes = 3\nmemory = "32G"\n')
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        assert w._coerce("", STEPS[_idx("cpus")]) == 8
        assert w._coerce("", STEPS[_idx("nodes")]) == 3
        assert w._coerce("", STEPS[_idx("memory")]) == "32G"

    def test_cleared_field_without_config_uses_literal(self):
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        assert w._coerce("", STEPS[_idx("cpus")]) == 4
        assert w._coerce("", STEPS[_idx("nodes")]) == 1
        assert w._coerce("", STEPS[_idx("memory")]) == "16G"


class TestCoerceJobNameSanitized:
    def test_job_name_coerced_safe(self):
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        s = STEPS[_idx("job_name")]
        assert w._coerce("my training job", s) == "my_training_job"


class TestGpuFormatEnvDefault:
    def test_env_seeds_wizard_default(self, monkeypatch):
        # P0-2: SLURMATE_GPU_FORMAT seeds the wizard's GPU-format default.
        monkeypatch.setenv("SLURMATE_GPU_FORMAT", "gpus")
        from slurmate.tui import Wizard
        w = Wizard()
        w.idx = _idx("gpu_format")
        w.answers["gpus"] = 2
        w._setup_gpu_format("forward")
        assert w._radio_value() == "gpus"

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("SLURMATE_GPU_FORMAT", "bogus")
        from slurmate.tui import Wizard
        w = Wizard()
        w.idx = _idx("gpu_format")
        w.answers["gpus"] = 2
        w._setup_gpu_format("forward")
        assert w._radio_value() == "gres_type"


class TestPartitionCaching:
    def test_reentry_reuses_cached_partitions(self, mocker):
        # P3-5: re-entering the partition step reuses the cached result instead
        # of re-running the cluster queries.
        import slurmate.tui as t
        from slurmate.tui import Wizard
        fp = mocker.patch.object(t, "fetch_partitions", return_value=[
            {"name": "p", "nodes": 1, "cpus_per_node": 1, "mem_per_node_mb": 1, "gpu_types": []}])
        mocker.patch.object(t, "fetch_public_partitions", return_value=[])
        w = Wizard()
        w._setup_partition()
        w._setup_partition()  # second entry
        assert fp.call_count == 1


class TestWizardSelectionSmoke:
    def test_walk_select_steps_and_build(self, mocker):
        # P3-1: construct the wizard and exercise selection across a few steps,
        # so a prompt_toolkit change that breaks the private-attr reads we rely
        # on (RadioList._selected_index etc.) fails here rather than in users'
        # terminals.
        from slurmate.builder import build_from_answers
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        # job name (text)
        w.idx = _idx("job_name")
        w.text_area.text = "smoke"
        w._confirm_and_next()
        # env_type (select via radio) — arrow to a value and confirm. Use venv
        # so the follow-on env_name step doesn't pop a conda completion menu
        # (which would warn about an unawaited coroutine with no event loop).
        w.idx = _idx("env_type")
        s = STEPS[w.idx]
        w._setup_select(s, None)
        target = ("Virtualenv (venv)", "Virtualenv (venv)")
        w.radio_list._selected_index = w.radio_list.values.index(target)
        assert w._radio_value() == "Virtualenv (venv)"
        w._confirm_and_next()
        assert w.answers["env_type"] == "Virtualenv (venv)"
        # the collected answers still build a valid script
        assert "#SBATCH --job-name=smoke" in build_from_answers(w.answers)


class TestPreviewDirtyOnBack:
    def test_go_back_marks_preview_dirty(self):
        from slurmate.tui import Wizard
        w = Wizard()
        w.idx = 5
        w.transient["preview_dirty"] = False
        w._go_back()
        assert w.transient.get("preview_dirty") is True


class TestSkippedStepNoStaleSave:
    def test_skipped_env_name_not_saved_with_stale_text(self):
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        env_idx = next(i for i, s in enumerate(STEPS) if s.key == "env_name")
        w.idx = env_idx
        w._skipped_indices.add(env_idx)
        # The shared text widget still holds the modules step's text.
        w.text_area.text = "cuda, python"
        w._go_back()
        # env_name must not be clobbered with the leftover modules string.
        assert w.answers.get("env_name") is None


class TestQosCachePartitionAware:
    def test_qos_refetched_when_partition_changes(self, mocker):
        import slurmate.tui as t
        from slurmate.tui import STEPS, Wizard
        calls: list[str] = []

        def fake_qos(part):
            calls.append(part)
            return ["qos_" + part]

        mocker.patch.object(t, "fetch_qos_for_partition", side_effect=fake_qos)
        mocker.patch.object(t, "fetch_known_qos", return_value=["qos_A", "qos_B"])
        w = Wizard()
        qos_step = next(s for s in STEPS if s.key == "qos")
        w.answers["partition"] = "A"
        r1 = w._resolve_choices(qos_step)
        w.answers["partition"] = "B"
        r2 = w._resolve_choices(qos_step)
        assert r1 == ["Default (none)", "qos_A"]
        assert r2 == ["Default (none)", "qos_B"]
        assert calls == ["A", "B"]


class TestNoneTextAreaGuards:
    def test_gpu_type_text_branch_with_none(self, monkeypatch):
        # Regression: answers["gpu_type"] == None must not crash TextArea.
        import slurmate.tui as t
        from slurmate.tui import STEPS, Wizard
        monkeypatch.setattr(t, "fetch_gpu_type_sources",
                            lambda p: {"typed": [], "feature": []})
        w = Wizard()
        w.idx = next(i for i, s in enumerate(STEPS) if s.key == "gpu_type")
        w.answers.update({"gpus": 2, "partition": "x", "gpu_type": None})
        w._setup_gpu_type("forward")
        assert w.text_area.text == ""

    def test_env_name_venv_with_none(self, monkeypatch):
        from slurmate.tui import STEPS, Wizard
        w = Wizard()
        w.idx = next(i for i, s in enumerate(STEPS) if s.key == "env_name")
        w.answers.update({"env_type": "Virtualenv (venv)", "env_name": None})
        w._setup_env_name("forward")
        assert w.text_area.text == ""


class TestPartitionAndGpuNavigation:
    """State-machine regressions in partition/gpu_type navigation."""

    def _wizard(self):
        w = Wizard()
        w._invalidate = lambda: None
        w._advance = lambda: None
        return w

    def test_gpu_type_back_returns_to_gpus_step(self):
        from prompt_toolkit.widgets import RadioList
        w = self._wizard()
        # Stale radio left over from an earlier select step (e.g. QoS).
        w.radio_list = RadioList([("Default (none)", "Default (none)"),
                                  ("high", "high"), ("gpu", "gpu")])
        w.radio_list._selected_index = 1
        w.answers["gpus"] = 1
        w.answers["partition"] = "debug"  # mock 'debug' advertises no typed GPUs
        w.idx = _idx("gpu_type")
        w._setup_gpu_type("forward")
        assert w.step_cache.get("gpu_sub") == "text"
        w._go_back()
        # Back must move to the gpus step, not trap the user on gpu_type.
        assert w.idx == _idx("gpus")

    def test_partition_back_from_private_resolves_correctly(self):
        from slurmate.tui import PRIVATE
        w = self._wizard()
        w.idx = _idx("partition")
        w._setup_partition()
        vals = [v for v, _ in w.radio_list.values]
        w.radio_list._selected_index = vals.index(PRIVATE)
        w._handle_partition_confirm()
        assert w.step_cache.get("partition_sub") == "all"
        w._go_back()
        assert w.step_cache.get("partition_sub") == "select"
        vals2 = [v for v, _ in w.radio_list.values]
        target = next(v for v in vals2 if v.startswith("cpu-highmem"))
        w.radio_list._selected_index = vals2.index(target)
        w._handle_partition_confirm()
        assert w.answers["partition"] == "cpu-highmem"

    def test_all_restricted_cluster_resolves_real_name(self):
        from prompt_toolkit.widgets import RadioList

        from slurmate.tui import CUSTOM, _fmt_partition
        w = self._wizard()
        allp = [{"name": "restricted-gpu", "nodes": 4, "cpus_per_node": 32,
                 "mem_per_node_mb": 262144, "gpu_types": ["h100"], "timelimit": None}]
        w.transient["all_parts"] = allp
        w.transient["public_parts"] = []
        label = _fmt_partition(allp[0])
        w.radio_list = RadioList([(CUSTOM, CUSTOM), (label, label)])
        w.step_cache["partition_sub"] = "select"
        w._set_partition_from_select(label)
        assert w.answers["partition"] == "restricted-gpu"
        assert w.answers["_partition_obj"]["name"] == "restricted-gpu"


class TestLivePartitionValidation:
    """The work-in-progress script is validated on every step, so a config
    already in a failure mode (e.g. GPUs on a CPU-only partition) keeps warning
    after the user moves past the step that introduced it."""

    CPU_PART = {"name": "caslake", "cpus_per_node": 48, "mem_per_node_mb": 196608,
                "gpu_types": [], "has_gpu": False, "timelimit": "36:00:00"}

    def _wizard_on_caslake(self, gpus=1):
        w = Wizard()
        w.answers.update({"partition": "caslake", "_partition_obj": self.CPU_PART,
                          "gpus": gpus, "cpus": 4, "memory": "16G",
                          "time_limit": "02:00:00"})
        w.transient["gpu_types"] = []
        return w

    def test_no_warning_without_partition_object(self):
        w = Wizard()
        w.answers["gpus"] = 4
        assert w._config_warnings() == []

    def test_gpu_on_cpu_partition_persists_past_gpus_step(self):
        # The reported bug: choosing 1 GPU on caslake (no GPU) must keep warning
        # on the *later* GPU-type step, not just while on the GPUs step.
        w = self._wizard_on_caslake(gpus=1)
        for key in ("gpu_type", "gpu_format", "output_dir", "command", "review"):
            w.idx = _idx(key)
            issues = w._config_warnings()
            assert ("error", "Partition 'caslake' does not support GPUs") in issues, key

    def test_zero_gpus_no_warning(self):
        w = self._wizard_on_caslake(gpus=0)
        w.idx = _idx("gpu_type")
        assert w._config_warnings() == []

    def test_current_field_live_value_overlaid(self):
        # While on the GPUs step, the not-yet-committed typed value drives the
        # check (feedback before Enter), overriding the committed answer.
        w = self._wizard_on_caslake(gpus=0)
        w.idx = _idx("gpus")
        w.text_area.text = "2"
        assert any(lvl == "error" and "does not support GPUs" in m
                   for lvl, m in w._config_warnings())
        w.text_area.text = "0"
        assert w._config_warnings() == []

    def test_manually_typed_unknown_partition_no_false_error(self):
        from slurmate.tui import _get_partition
        # A partition name not in the fetched list resolves to the synthetic
        # fallback (no has_gpu key). Requesting GPUs must not raise a false
        # "does not support GPUs" error, since capability is unknown.
        w = Wizard()
        w.answers["partition"] = "typo-partition"
        w.answers["_partition_obj"] = _get_partition([], "typo-partition")
        w.answers["gpus"] = 2
        w.idx = _idx("gpu_type")
        assert all("does not support GPUs" not in m for _, m in w._config_warnings())


class TestSpaceSeparatedCustomFlags:
    """M5: a bare token after a value-taking option is its VALUE, not a new flag.

    ``-o /logs/x.out`` used to parse as ``['-o', '--/logs/x.out']``, emitting a
    valueless directive plus a nonsense one — a script sbatch rejects outright.
    """

    def test_short_option_with_path_value(self):
        assert _parse_custom_flags("-o /real/place/%j.log") == ["-o /real/place/%j.log"]
        assert _parse_custom_flags("-e /logs/x.err") == ["-e /logs/x.err"]

    def test_short_option_with_word_value(self):
        assert _parse_custom_flags("-C bigmem") == ["-C bigmem"]
        assert _parse_custom_flags("-w node1") == ["-w node1"]
        assert _parse_custom_flags("-J myjob") == ["-J myjob"]

    def test_long_option_with_space_value(self):
        assert _parse_custom_flags("--reservation abc") == ["--reservation abc"]
        assert _parse_custom_flags("--constraint gpu") == ["--constraint gpu"]
        assert _parse_custom_flags("--nodelist n[01-04]") == ["--nodelist n[01-04]"]

    def test_boolean_option_does_not_swallow_the_next_word(self):
        # --exclusive takes no value, so a following bare word is its own option.
        assert _parse_custom_flags("--exclusive hold") == ["--exclusive", "--hold"]
        assert _parse_custom_flags("exclusive requeue") == ["--exclusive", "--requeue"]

    def test_dashless_word_still_becomes_an_option(self):
        assert _parse_custom_flags("exclusive") == ["--exclusive"]

    def test_only_one_bare_token_is_consumed(self):
        # The option is satisfied by the first value; a second bare word is a
        # separate (dashless) option, not appended to the value.
        assert _parse_custom_flags("-C a b") == ["-C a", "--b"]

    def test_equals_form_unchanged(self):
        assert _parse_custom_flags("--constraint=bigmem") == ["--constraint=bigmem"]
        assert _parse_custom_flags("--exclusive --reservation=abc") == [
            "--exclusive", "--reservation=abc"]

    def test_path_value_after_unknown_option(self):
        # Not in the known list, but "/tmp/x" can't be an option name either.
        assert _parse_custom_flags("--madeup /tmp/x") == ["--madeup /tmp/x"]

    def test_multiple_flags_with_space_values(self):
        assert _parse_custom_flags("-C bigmem -w node1 --exclusive") == [
            "-C bigmem", "-w node1", "--exclusive"]


class TestConstraintAndMemPerCpuSteps:
    """M4: both are first-class CLI/config keys and now reachable interactively."""

    def test_steps_exist_in_directive_order(self):
        keys = [s.key for s in STEPS]
        assert keys.index("mem_per_cpu") == keys.index("memory") + 1
        assert keys.index("constraint") == keys.index("gpu_format") + 1

    def test_config_keys_now_reach_the_wizard(self, tmp_path, monkeypatch):
        # Before: Wizard built its defaults by iterating STEPS, so a key with no
        # step was never read — the same .slurmate.toml produced different jobs in
        # batch and interactive mode.
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SLURMATE_MOCK", raising=False)
        (tmp_path / ".slurmate.toml").write_text(
            'constraint = "gpu"\nmem_per_cpu = "2G"\n')
        w = Wizard()
        assert w._config_defaults["constraint"] == "gpu"
        assert w._config_defaults["mem_per_cpu"] == "2G"

    def test_answers_flow_into_the_script(self):
        from slurmate.builder import build_from_answers
        w = Wizard()
        w.idx = _idx("constraint")
        w.text_area.text = "gpu"
        w._confirm_and_next()
        w.idx = _idx("mem_per_cpu")
        w.text_area.text = "2G"
        w._confirm_and_next()
        w.answers.update({"job_name": "j", "partition": "p", "command": "x"})
        script = build_from_answers(w.answers)
        assert "#SBATCH --constraint=gpu" in script
        assert "#SBATCH --mem-per-cpu=2G" in script
        assert "#SBATCH --mem=" not in script

    def test_blank_mem_per_cpu_falls_back_to_memory(self):
        from slurmate.builder import build_from_answers
        w = Wizard()
        w.idx = _idx("mem_per_cpu")
        w.text_area.text = ""
        w._confirm_and_next()
        assert w.answers["mem_per_cpu"] is None
        w.answers.update({"job_name": "j", "partition": "p", "memory": "16G"})
        assert "#SBATCH --mem=16G" in build_from_answers(w.answers)

    def test_mem_per_cpu_is_normalized(self):
        w = Wizard()
        w.idx = _idx("mem_per_cpu")
        w.text_area.text = "2000"
        w._confirm_and_next()
        assert w.answers["mem_per_cpu"] == "2000M"


class TestGpuTypeCacheIsPartitionKeyed:
    """L1: a stale cached type list silently suppressed a real error."""

    def _wizard_on_gpu_shared(self):
        from slurmate.system_utils import MOCK_PARTITIONS
        w = Wizard()
        part = next(p for p in MOCK_PARTITIONS if p["name"] == "gpu-shared")
        w.answers.update({"partition": "gpu-shared", "_partition_obj": part, "gpus": 2})
        w.idx = _idx("gpu_type")
        w._setup_gpu_type("forward")
        w.answers["gpu_type"] = "a100"
        return w

    def test_stale_list_no_longer_hides_the_error(self):
        from slurmate.system_utils import MOCK_PARTITIONS
        w = self._wizard_on_gpu_shared()
        assert w.transient["gpu_types"] == ["a100", "v100"]
        highend = next(p for p in MOCK_PARTITIONS if p["name"] == "gpu-highend")
        w.answers.update({"partition": "gpu-highend", "_partition_obj": highend})
        for key in ("account", "cpus", "memory", "nodes"):
            w.idx = _idx(key)
            msgs = [m for lvl, m in w._config_warnings() if lvl == "error"]
            assert any("not in partition list" in m for m in msgs), key

    def test_cache_is_used_while_the_partition_matches(self):
        w = self._wizard_on_gpu_shared()
        w.idx = _idx("cpus")
        # a100 IS valid on gpu-shared: no error, and the cache is what says so.
        assert w._cached_gpu_types() == ["a100", "v100"]
        assert all(lvl != "error" for lvl, _ in w._config_warnings())


class TestGpuFormatDefaultsToConstraintForFeatureOnlyType:
    """H2: --gres=gpu:<model>:N is invalid when the model is only a node feature."""

    def _wizard(self, feature_only):
        w = Wizard()
        w.answers.update({"partition": "p", "gpus": 2, "gpu_type": "a100"})
        w.transient["gpu_types_part"] = "p"
        w.transient["gpu_types"] = ["a100"]
        w.transient["gpu_types_feature_only"] = feature_only
        w.idx = _idx("gpu_format")
        w._setup_gpu_format("forward")
        return w

    def test_feature_only_type_defaults_to_constraint(self):
        w = self._wizard(["a100"])
        assert w._radio_value() == "constraint"

    def test_typed_gres_keeps_the_default_format(self):
        w = self._wizard([])
        assert w._radio_value() == "gres_type"

    def test_explicit_prior_choice_is_respected(self):
        w = Wizard()
        w.answers.update({"partition": "p", "gpus": 2, "gpu_type": "a100",
                          "gpu_format": "gpus"})
        w.transient.update({"gpu_types_part": "p", "gpu_types_feature_only": ["a100"]})
        w.idx = _idx("gpu_format")
        w._setup_gpu_format("forward")
        assert w._radio_value() == "gpus"

    def test_sources_are_cached_from_the_fetch(self, monkeypatch):
        import slurmate.tui as t
        monkeypatch.setattr(t, "fetch_gpu_type_sources",
                            lambda p: {"typed": ["h100"], "feature": ["a100"]})
        w = Wizard()
        w.answers.update({"partition": "p", "gpus": 1})
        w.idx = _idx("gpu_type")
        w._setup_gpu_type("forward")
        assert w.transient["gpu_types"] == ["a100", "h100"]
        assert w.transient["gpu_types_feature_only"] == ["a100"]
        assert w.transient["gpu_types_part"] == "p"


class TestQueueEtaTracksNodes:
    """L9: the ETA depends on req_nodes, so the cache can't key on partition alone."""

    def test_node_change_refetches(self, monkeypatch):
        import slurmate.system_utils as su
        calls = []

        def spy(part, req_nodes=1):
            calls.append((part, req_nodes))
            return dict(su.MOCK_QUEUE_INFO)

        monkeypatch.setattr(su, "fetch_queue_eta", spy)
        w = Wizard()
        w.answers.update({"partition": "cpu-shared", "_partition_obj": None})
        for key in ("account", "cpus"):
            w.idx = _idx(key)
            w._on_enter_step("forward")
        w.answers["nodes"] = 8
        w.idx = _idx("modules")
        w._on_enter_step("forward")
        assert calls == [("cpu-shared", 1), ("cpu-shared", 8)]

    def test_no_refetch_when_nothing_changed(self, monkeypatch):
        import slurmate.system_utils as su
        calls = []
        monkeypatch.setattr(su, "fetch_queue_eta",
                            lambda part, req_nodes=1: calls.append((part, req_nodes))
                            or dict(su.MOCK_QUEUE_INFO))
        w = Wizard()
        w.answers.update({"partition": "cpu-shared", "_partition_obj": None, "nodes": 2})
        for key in ("account", "cpus", "modules"):
            w.idx = _idx(key)
            w._on_enter_step("forward")
        assert calls == [("cpu-shared", 2)]


class TestReviewColumnAlignment:
    """L6: the label column must fit the longest label, as the CLI summary does."""

    def test_long_labels_do_not_break_the_value_column(self):
        w = Wizard()
        w.answers.update({"job_name": "j", "partition": "p", "nodes": 2,
                          "ntasks_per_node": 4, "array_spec": "1-10",
                          "output_dir": "logs", "command": "a.py\nb.py"})
        rendered = "".join(t for _, t in w._render_review_config())
        lines = [ln for ln in rendered.splitlines() if ln.strip()]
        # Every value starts in the same column.
        starts = set()
        for label, _val in w._review_summary_items():
            line = next(ln for ln in lines if ln.strip().startswith(label))
            starts.add(line.index(line.strip().split(label)[-1].lstrip()))
        assert len(starts) == 1
        # A multi-line value's continuation lines line up with the value column.
        assert any(ln.strip() == "b.py" for ln in lines)
        cont = next(ln for ln in lines if ln.strip() == "b.py")
        assert cont.index("b.py") == starts.pop()


class TestGpuTypeTextPersistsOnBack:
    """Minor: the free-text GPU-type sub-mode was the one input Back discarded."""

    def test_typed_value_survives_back(self):
        w = Wizard()
        # cpu-shared lists no typed GPUs in mock data -> free-text sub-mode.
        w.answers.update({"partition": "cpu-shared", "_partition_obj": None, "gpus": 2})
        w.idx = _idx("gpu_type")
        w._setup_gpu_type("forward")
        assert w.step_cache.get("gpu_sub") == "text"
        w.text_area.text = "h100"
        w._go_back()
        assert w.answers.get("gpu_type") == "h100"

    def test_blank_does_not_overwrite_a_prior_answer(self):
        w = Wizard()
        w.answers.update({"partition": "cpu-shared", "_partition_obj": None,
                          "gpus": 2, "gpu_type": "a100"})
        w.idx = _idx("gpu_type")
        w._setup_gpu_type("forward")
        w.text_area.text = ""
        w._go_back()
        assert w.answers.get("gpu_type") == "a100"
