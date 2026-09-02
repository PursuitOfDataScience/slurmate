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

    def test_comma_completer_ignores_a_trailing_space(self):
        """`gc ` + Tab offered every module in the list instead of `gcc`.

        The token was only `lstrip`ed, so a trailing space survived into the fuzzy
        pattern -- and no module name contains a space, which made the pattern
        match everything. The sibling path completer never had this because
        splitting on whitespace makes a trailing space an empty token.
        """
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document

        from slurmate.tui import LastTokenCommaCompleter

        cc = LastTokenCommaCompleter(["cuda", "cudnn", "python/anaconda", "gcc", "openmpi"])

        def texts(buf):
            doc = Document(buf, len(buf))
            return [c.text for c in cc.get_completions(doc, CompleteEvent())]

        assert texts("gc ") == ["gcc"]
        assert texts("cu ") == ["cuda", "cudnn"]
        assert texts("cuda,  cud ") == ["cuda", "cudnn"]

    def test_the_control_the_comma_completer_still_works_without_a_space(self):
        """The control: stripping too eagerly, or matching nothing at all, would
        also silence the test above."""
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document

        from slurmate.tui import LastTokenCommaCompleter

        cc = LastTokenCommaCompleter(["cuda", "cudnn", "python/anaconda", "gcc", "openmpi"])

        def comps(buf):
            doc = Document(buf, len(buf))
            return list(cc.get_completions(doc, CompleteEvent()))

        assert [c.text for c in comps("cu")] == ["cuda", "cudnn"]
        assert [c.text for c in comps("cuda, cu")] == ["cuda", "cudnn"]
        assert comps("") == [] and comps("cuda,") == []
        # The whole token including its trailing space is replaced, so accepting a
        # completion cannot leave "cu cuda".
        assert all(c.start_position == -3 for c in comps("cu "))

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
            # The qos step now reads BOTH sides of the ACL (AllowQos + DenyQos),
            # so the partition-aware cache is exercised through fetch_qos_acl.
            calls.append(part)
            return {"allow": ["qos_" + part], "deny": []}

        mocker.patch.object(t, "fetch_qos_acl", side_effect=fake_qos)
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

        def spy(part, req_nodes=1, **kw):
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
                            lambda part, req_nodes=1, **kw: calls.append((part, req_nodes))
                            or dict(su.MOCK_QUEUE_INFO))
        w = Wizard()
        w.answers.update({"partition": "cpu-shared", "_partition_obj": None, "nodes": 2})
        for key in ("account", "cpus", "modules"):
            w.idx = _idx(key)
            w._on_enter_step("forward")
        assert calls == [("cpu-shared", 2)]

    def test_gpu_change_refetches(self, monkeypatch):
        # The ETA now depends on the whole request: changing the GPU count must
        # invalidate the cache, or a GPU job keeps showing the CPU-only estimate.
        import slurmate.system_utils as su
        calls = []

        def spy(part, req_nodes=1, **kw):
            calls.append((part, req_nodes, kw.get("gpus_per_node", 0)))
            return dict(su.MOCK_QUEUE_INFO)

        monkeypatch.setattr(su, "fetch_queue_eta", spy)
        w = Wizard()
        w.answers.update({"partition": "gpu-shared", "_partition_obj": None, "nodes": 1})
        w.idx = _idx("account")
        w._on_enter_step("forward")
        w.answers["gpus"] = 4
        w.idx = _idx("modules")
        w._on_enter_step("forward")
        assert calls == [("gpu-shared", 1, 0), ("gpu-shared", 1, 4)]


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

    def test_a_radio_reset_to_any_does_not_overwrite_a_prior_answer(self):
        """The *select* sub-mode still saves nothing on Back — deliberately.

        This is the concern the original fix recorded ("doing the same for the
        select sub-mode would let a radio reset to 'Any' overwrite a previously
        typed model with None"), and it is still guarded. It used to be asserted
        of the *text* sub-mode instead, where an emptied field now means "no
        model", exactly as it does when Enter is pressed — see
        TestClearedFieldIsClearedByBackToo.
        """
        w = Wizard()
        # gpu-shared advertises typed GPUs in mock data -> radio sub-mode.
        w.answers.update({"partition": "gpu-shared", "_partition_obj": None,
                          "gpus": 2, "gpu_type": "a100"})
        w.idx = _idx("gpu_type")
        w._setup_gpu_type("forward")
        assert w.step_cache.get("gpu_sub") == "select"
        w._set_radio_default("Any")
        w._go_back()
        assert w.answers.get("gpu_type") == "a100"


class TestUnknownPartitionIsFlagged:
    """A manually-typed partition the cluster does not have must SAY it wasn't checked.

    Every capacity field on the synthetic record is 0/None, which is what keeps
    the limit checks quiet instead of warning against a limit of zero — so
    without an explicit ``_unknown`` marker the wizard's live panel said nothing
    at all about a 999-CPU request, i.e. the less valid request produced the more
    reassuring screen. The CLI's copy of this record carried the flag; the
    wizard's did not, and the wizard is both the default interface and the only
    one with an "Enter partition name manually..." row.
    """

    OVERSIZED = {"cpus": 999, "memory": "9999G", "time_limit": "999:00:00",
                 "nodes": 50, "gpus": 8}
    REAL_PART = {"name": "amd", "nodes": 4, "nodes_up": 4, "cpus_per_node": 128,
                 "mem_per_node_mb": 256000, "gpu_types": [], "timelimit": "36:00:00",
                 "is_public": True, "is_default": False}

    def _typed_manually(self, name, all_parts):
        """Drive the real CUSTOM (manual-entry) confirm handler, not a stub."""
        w = Wizard()
        w.idx = _idx("partition")
        w.transient["all_parts"] = list(all_parts)
        w.step_cache["partition_sub"] = "text"
        w.text_area.text = name
        w.answers.update(self.OVERSIZED)
        w._handle_partition_confirm()
        return w

    def test_manual_entry_marks_the_record_unknown(self):
        w = self._typed_manually("typo-partition", [self.REAL_PART])
        assert w.answers["partition"] == "typo-partition"
        part = w.answers["_partition_obj"]
        assert part["_unknown"] is True
        # 'absent' vs 'unreadable': the partition list WAS readable here, so this
        # name is genuinely not on the cluster.
        assert part["_unknown_reason"] == "absent"

    def test_live_panel_says_the_limits_were_not_checked(self):
        w = self._typed_manually("typo-partition", [self.REAL_PART])
        w.idx = _idx("cpus")
        msgs = [m for _lvl, m in w._config_warnings()]
        assert any("Capacity limits NOT checked" in m for m in msgs), msgs
        assert any("'typo-partition' is not on this cluster" in m for m in msgs), msgs

    def test_unreadable_list_does_not_claim_the_partition_is_absent(self):
        # No Slurm / sinfo failed: nothing is known about ANY partition, so
        # "not on this cluster" would be a false rejection.
        w = self._typed_manually("amd", [])
        assert w.answers["_partition_obj"]["_unknown_reason"] == "unreadable"
        w.idx = _idx("cpus")
        msgs = [m for _lvl, m in w._config_warnings()]
        assert any("could not be read" in m for m in msgs), msgs
        assert all("is not on this cluster" not in m for m in msgs), msgs

    def test_control_a_real_partition_is_not_flagged_unknown(self):
        """Control: the flag must mark ONLY the synthetic fallback.

        Flagging a partition that WAS found would put "limits not checked" on
        every ordinary run, which is the opposite failure — a warning nobody can
        act on, on the path where the limits were in fact checked.
        """
        from slurmate.tui import _get_partition
        found = _get_partition([self.REAL_PART], "amd")
        assert found is self.REAL_PART
        assert "_unknown" not in found
        assert "_unknown_reason" not in found

        w = Wizard()
        w.answers.update({"partition": "amd", "_partition_obj": self.REAL_PART,
                          "cpus": 4, "memory": "16G", "time_limit": "02:00:00",
                          "nodes": 1, "gpus": 0})
        w.idx = _idx("cpus")
        assert all("Capacity limits NOT checked" not in m
                   for _lvl, m in w._config_warnings())


class TestBlankManualPartitionKeepsValidating:
    """A blank manual partition means "site default", not "stop checking".

    Confirming an EMPTY value in the "Enter partition name manually..." row is a
    legitimate answer — Slurm uses the site default and the builder emits no
    ``--partition`` directive for it (SM-15) — but it left ``_partition_obj``
    None, and ``_config_warnings`` early-returned ``[]`` on that for the whole
    rest of the session. ``validate_job_config`` consults the partition in only
    *some* of its rules, so the ones that never look at one were dropped for a
    reason that has nothing to do with them: a duplicated custom directive
    (a second ``#SBATCH`` line ``sbatch --test-only`` reports ***PASSED*** and
    Slurm then silently honours over slurmate's, while the summary describes the
    value that lost) and an ``--array`` beyond the site's MaxArraySize.
    """

    REAL_PART = {"name": "amd", "nodes": 4, "nodes_up": 4, "cpus_per_node": 128,
                 "mem_per_node_mb": 256000, "gpu_types": [], "has_gpu": False,
                 "timelimit": "36:00:00", "is_public": True, "is_default": True}

    def _blank_manual(self, **answers):
        """Drive the real CUSTOM row and confirm an empty value, not a stub."""
        from slurmate.tui import CUSTOM
        w = Wizard()
        w.idx = _idx("partition")
        w.transient["all_parts"] = [self.REAL_PART]
        w._on_enter_step()
        assert CUSTOM in [v for v, _ in w.radio_list.values]
        w._set_radio_default(CUSTOM)
        w._handle_partition_confirm()          # CUSTOM -> manual-entry text mode
        assert w.step_cache.get("partition_sub") == "text"
        w.text_area.text = ""
        w._handle_partition_confirm()          # confirm the empty value
        # The blank itself is accepted, not rejected: that is the point.
        assert w.answers["partition"] == ""
        assert w.answers["_partition_obj"] is None
        w.answers.update(answers)
        return w

    def test_duplicated_custom_directive_is_still_reported(self):
        w = self._blank_manual(
            job_name="run", cpus=4,
            custom_sbatch=["--job-name=OVERRIDE", "--cpus-per-task=1"],
        )
        for key in ("cpus", "custom_sbatch", "command", "review"):
            w.idx = _idx(key)
            flagged = [m for lvl, m in w._config_warnings()
                       if lvl == "error" and "duplicates a directive" in m]
            assert len(flagged) == 2, (key, w._config_warnings())

    def test_the_script_really_carries_the_duplicate_it_warns_about(self):
        """The harm the warning names, in the generated script itself."""
        from slurmate.builder import build_from_answers
        w = self._blank_manual(
            job_name="run", cpus=4, command="echo hi",
            custom_sbatch=["--job-name=OVERRIDE"],
        )
        names = [ln for ln in build_from_answers(w.answers).splitlines()
                 if "--job-name" in ln]
        # Slurm honours the LAST, so the job runs as OVERRIDE while the summary
        # says "run" — and with no partition nothing said so.
        assert names == ["#SBATCH --job-name=run", "#SBATCH --job-name=OVERRIDE"]
        assert any("duplicates a directive" in m for _lvl, m in w._config_warnings())

    def test_over_maxarraysize_is_still_reported(self, mocker):
        import slurmate.tui as t
        mocker.patch.object(t, "fetch_max_array_size", return_value=65533)
        w = self._blank_manual(cpus=4, array_spec="1-99999")
        w.idx = _idx("review")
        msgs = [m for _lvl, m in w._config_warnings()]
        assert any("MaxArraySize (65533)" in m for m in msgs), msgs

    def test_control_a_fitting_array_stays_silent(self, mocker):
        """Control: the array rule must fire on the limit, not on the blank."""
        import slurmate.tui as t
        mocker.patch.object(t, "fetch_max_array_size", return_value=65533)
        w = self._blank_manual(cpus=4, array_spec="1-100")
        w.idx = _idx("review")
        assert w._config_warnings() == []

    def test_control_partition_dependent_checks_stay_silent(self):
        """Control: this must not become "check everything against nothing".

        With no partition there is no advertised CPU/memory/GPU/time/node figure
        to compare against, so a limit warning here would be an invented limit —
        the false-rejection failure mode SM-4's restraint exists to prevent. Both
        before and after the fix this list is empty; what changed is only the
        rules above, which never consulted a partition in the first place.
        """
        w = self._blank_manual(cpus=999, memory="9999G", time_limit="999:00:00",
                               nodes=50, gpus=8, gpu_type="a100")
        for key in ("cpus", "memory", "nodes", "gpus", "review"):
            w.idx = _idx(key)
            assert w._config_warnings() == [], key

    def test_control_a_named_partition_is_judged_exactly_as_before(self):
        """Control: the same answers on a real partition, in the same order.

        The fix shares two rule bodies between the no-partition path and the main
        one, so this pins the full ordered list a known partition produces —
        a reordered or doubled message would be a regression the assertions
        above cannot see.
        """
        w = Wizard()
        w.answers.update({
            "partition": "amd", "_partition_obj": self.REAL_PART, "job_name": "run",
            "cpus": 999, "memory": "9999G", "time_limit": "999:00:00",
            "nodes": 50, "gpus": 8,
            "custom_sbatch": ["--job-name=OVERRIDE"],
        })
        w.idx = _idx("review")
        assert w._config_warnings() == [
            ("warning", "CPUs (999) exceeds partition limit (128 per node)"),
            ("error", "custom flag --job-name duplicates a directive slurmate "
                      "manages; Slurm would honour it over --job-name and the "
                      "summary would describe the wrong value"),
            ("warning", "Memory (9999G) exceeds partition limit (256000 MB per node)"),
            ("warning", "Time limit (999:00:00) exceeds partition limit (36:00:00)"),
            ("warning", "Nodes (50) exceeds the 4 node(s) in 'amd'"),
            ("error", "Partition 'amd' does not support GPUs"),
        ]

    def test_the_final_summary_agrees_with_the_live_panel(self):
        """The two surfaces share one validator, so a blank partition must not
        split them: the wizard's answers go on to the CLI summary, whose
        ``_partition_issues`` early-returned on the same falsy record."""
        from slurmate.main import _partition_issues
        w = self._blank_manual(job_name="run", cpus=4,
                               custom_sbatch=["--job-name=OVERRIDE"])
        w.idx = _idx("review")
        assert _partition_issues(w.answers) == w._config_warnings()
        assert w._config_warnings()


class TestClearedTimeLimitFallsBackToDefault:
    """Clearing the pre-filled time limit must not silently drop ``--time``.

    ``time_limit`` fell through to ``_coerce``'s bare ``return val``, so an
    emptied field became ``""`` — and the builder omits ``#SBATCH --time`` for an
    empty value. The script then had no time limit, the summary had no "Time
    limit" row, and "Estimated CPU-hours" was still computed from
    ``estimate_su``'s implicit 120-minute assumption. cpus/nodes/memory all
    revert to the config/literal default when cleared (the P3-10 invariant);
    this one did not, and unlike mem_per_cpu/array_spec/constraint its subtitle
    never offers blank as an answer.
    """

    def test_cleared_field_uses_the_declared_default(self):
        w = Wizard()
        assert w._coerce("", STEPS[_idx("time_limit")]) == "02:00:00"

    def test_cleared_field_prefers_a_configured_default(self):
        w = Wizard()
        w._config_defaults["time_limit"] = "08:00:00"
        assert w._coerce("", STEPS[_idx("time_limit")]) == "08:00:00"

    def test_script_and_summary_still_carry_a_time_limit(self):
        from slurmate.builder import build_from_answers, job_summary_rows
        w = Wizard()
        answers = {"job_name": "j", "partition": "amd", "cpus": 4,
                   "memory": "16G", "nodes": 1, "command": "echo hi",
                   "time_limit": w._coerce("", STEPS[_idx("time_limit")])}
        assert "#SBATCH --time=02:00:00" in build_from_answers(answers)
        assert ("Time limit", "02:00:00") in job_summary_rows(answers)

    def test_control_a_typed_value_is_returned_verbatim(self):
        """Control: the default applies to a CLEARED field only.

        Substituting it for a value the user typed would overwrite the answer —
        and every accepted spelling must survive, not just hh:mm:ss.
        """
        w = Wizard()
        w._config_defaults["time_limit"] = "08:00:00"
        s = STEPS[_idx("time_limit")]
        for typed in ("30", "5:00", "01:00:00", "7-00:00:00", "2-12"):
            assert w._coerce(typed, s) == typed

    def test_control_explicit_empty_config_still_omits_time(self):
        """Control: an operator who configures a blank time limit keeps the
        omission — the fallback goes through ``_step_default``, which honours an
        override of ``""``, rather than hardcoding the step literal."""
        w = Wizard()
        w._config_defaults["time_limit"] = ""
        assert w._coerce("", STEPS[_idx("time_limit")]) == ""


class TestCustomFlagsSurviveBack:
    """A quoted custom ``#SBATCH`` value must survive going Back to its step.

    ``_parse_custom_flags`` *consumes* the user's quotes, so writing the parsed
    list back into the field verbatim made the next confirm re-split it. Typing
    ``--comment="my big run"``, pressing Back and confirming again produced
    ``['--comment=my big', '--run']`` — a fabricated ``#SBATCH --run`` that
    ``sbatch --test-only -A rcc-staff -p amd -t 1`` refuses outright
    ("unrecognized option '--run'", rc 255), turning a script sbatch accepted
    (rc 0) into one it rejects.
    """

    def _round_trip(self, typed, key="custom_sbatch"):
        """Answer the step, walk Back into it, confirm it again untouched."""
        w = Wizard()
        w._invalidate = lambda: None
        w.idx = _idx(key)
        w.text_area.text = typed
        w._confirm_and_next()
        forward = w.answers[key]
        forward = list(forward) if isinstance(forward, list) else forward
        guard = 0
        while w.current_step.key != key and guard < 8:
            w._go_back()
            guard += 1
        assert w.current_step.key == key
        restored = w.text_area.text
        w._confirm_and_next()
        after = w.answers[key]
        return forward, restored, (list(after) if isinstance(after, list) else after)

    def test_quoted_value_with_a_space_survives_back(self):
        forward, restored, after = self._round_trip(
            '--comment="my big run" --exclusive'
        )
        assert forward == ["--comment=my big run", "--exclusive"]
        # The field is re-quoted on the way in, so re-parsing it is a no-op.
        assert '--comment="my big run"' in restored
        assert after == forward

    def test_no_fabricated_directive_reaches_the_script(self):
        from slurmate.builder import build_from_answers
        _forward, _restored, after = self._round_trip('--comment="my big run"')
        script = build_from_answers({
            "job_name": "j", "partition": "amd", "cpus": 1, "memory": "1G",
            "nodes": 1, "time_limit": "1", "command": "true",
            "custom_sbatch": after,
        })
        assert '#SBATCH --comment="my big run"' in script
        assert "--run" not in script

    def test_control_unquoted_flags_round_trip_too(self):
        """Control: every custom-flag form whose values hold no whitespace
        round-tripped before the fix and still does. The defect was specific to
        a value that had to be quoted, so this half of the matrix pins that the
        Back-and-reconfirm harness itself is sound rather than vacuous."""
        for typed in ("--exclusive",
                      "--exclude=n1,n2 --exclusive",
                      "-C bigmem",
                      "--exclusive --reservation=abc",
                      "--exclusive,--hold"):
            forward, _restored, after = self._round_trip(typed)
            assert after == forward, typed

    def test_control_module_list_is_not_requoted(self):
        """Control: the same write-back serves the modules step, whose entries
        carry no quoting to restore — it must come back verbatim in both states,
        so the re-quoting stays scoped to custom_sbatch."""
        forward, restored, after = self._round_trip(
            "cuda/11.8, gcc/9.3.0", key="modules"
        )
        assert forward == ["cuda/11.8", "gcc/9.3.0"]
        assert restored == "cuda/11.8, gcc/9.3.0"
        assert after == forward


class TestPrivatePartitionListKeepsTheCurrentChoice:
    """Opening "Include private partitions" must not silently re-pick.

    ``_setup_partition`` deliberately highlights the already-chosen partition so
    a stray Enter doesn't drop into the manual-entry flow. The private list is
    one keypress away from that one and was built with its cursor on row 0, so a
    user who went Back to look at the private partitions and pressed Enter had
    the job moved to whatever partition sorts first — taking the derived memory
    default, the GPU-type list and the QoS list with it.
    """

    def _chose(self, name):
        w = Wizard()
        w._invalidate = lambda: None
        w.idx = _idx("partition")
        w._on_enter_step()
        vals = [v for v, _ in w.radio_list.values]
        w.radio_list._selected_index = next(
            i for i, v in enumerate(vals) if v.startswith(name)
        )
        w._confirm_and_next()
        assert w.answers["partition"] == name
        while w.current_step.key != "partition":
            w._go_back()
        return w

    def _open_private(self, w):
        from slurmate.tui import PRIVATE
        vals = [v for v, _ in w.radio_list.values]
        w.radio_list._selected_index = vals.index(PRIVATE)
        w._confirm_and_next()
        assert w.step_cache.get("partition_sub") == "all"

    def test_stray_enter_in_the_private_list_keeps_the_partition(self):
        w = self._chose("gpu-shared")
        self._open_private(w)
        assert w._radio_value().startswith("gpu-shared")
        w._confirm_and_next()          # a stray Enter, cursor never moved
        assert w.answers["partition"] == "gpu-shared"
        assert w.answers["_partition_obj"]["name"] == "gpu-shared"

    def test_control_a_moved_cursor_still_selects_that_row(self):
        """Control: the restore seeds the cursor, it does not pin it — moving to
        another row in the private list still selects that row in both states.
        Without this the fix could pass by ignoring the user entirely."""
        w = self._chose("gpu-shared")
        self._open_private(w)
        vals = [v for v, _ in w.radio_list.values]
        w.radio_list._selected_index = next(
            i for i, v in enumerate(vals) if v.startswith("debug")
        )
        w._confirm_and_next()
        assert w.answers["partition"] == "debug"

    def test_control_first_visit_opens_on_the_first_row(self):
        """Control: on a first visit there is no answer to restore, so the list
        still opens on row 0 — the behaviour the fix must leave alone."""
        w = Wizard()
        w._invalidate = lambda: None
        w.idx = _idx("partition")
        w._on_enter_step()
        assert "partition" not in w.answers
        self._open_private(w)
        assert w._radio_value() == w.radio_list.values[0][0]


class TestClearedFieldIsClearedByBackToo:
    """Clearing an optional field must clear it whichever key comes next.

    Measured across all 17 free-text steps: Enter committed a cleared field and
    Back did not, so the same visible field state (empty) meant two different
    things depending on which key the user pressed. ``ntasks_per_node`` is where
    that produces a script Slurm refuses *and* a screen that says otherwise: the
    live panel's warning is computed from the field's live value, so it goes quiet
    as the field empties — and Back then puts the deleted directive back.
    Authority, on the real cluster (amd has 128 cores/node):

        sbatch --test-only -A rcc-staff -p amd -t 1 --wrap='true' \\
            -N2 --ntasks-per-node=64 -c 4      # what Back produced
        -> allocation failure: Requested node configuration is not available

        sbatch --test-only -A rcc-staff -p amd -t 1 --wrap='true' \\
            -N2 --ntasks-per-node=1 -c 4       # what Enter produced
        -> Verification: ***PASSED***   (rc 0)
    """

    def _on_ntasks_step(self):
        from slurmate.system_utils import fetch_partitions
        w = Wizard()
        # 32 cores/node in mock data, so 64 tasks x 4 cpus is over the limit.
        part = next(p for p in fetch_partitions() if p["name"] == "cpu-shared")
        w.answers.update({"partition": "cpu-shared", "_partition_obj": part,
                          "cpus": 4, "memory": "16G", "time_limit": "01:00:00",
                          "nodes": 2, "gpus": 0, "ntasks_per_node": 64})
        w.idx = _idx("ntasks_per_node")
        w._on_enter_step("forward")
        return w

    def test_the_screen_confirms_the_clear(self):
        """Not the fix — the reason it is a defect. The live panel drops the
        warning as the field empties, i.e. the screen tells the user the clear
        took effect. Passes in both states."""
        w = self._on_ntasks_step()
        assert w.text_area.text == "64"
        assert any("per node" in m for _lv, m in w._config_warnings())
        w.text_area.text = ""
        assert w._config_warnings() == []

    def test_back_on_a_cleared_optional_field_clears_it(self):
        w = self._on_ntasks_step()
        w.text_area.text = ""
        w._go_back()
        assert w.answers["ntasks_per_node"] is None

    def test_the_deleted_value_leaves_the_script(self):
        """End-to-end: the script no longer asks for 64 tasks x 4 CPUs on a
        32-core node. (A multi-node job always carries the directive, so the
        cleared field shows up as the builder's ``=1``, not as its absence.)"""
        from slurmate.builder import build_from_answers
        w = self._on_ntasks_step()
        w.text_area.text = ""
        w._go_back()
        script = build_from_answers(w.answers)
        assert "#SBATCH --ntasks-per-node=64" not in script
        assert "#SBATCH --ntasks-per-node=1" in script

    def test_both_gestures_now_agree_on_every_optional_step(self):
        """The invariant, stated once: wherever blank is an offered answer
        (``_coerce("") is None``), Enter-on-empty and Back-on-empty produce the
        same answer."""
        agree, differ = [], []
        for s in STEPS:
            if s.kind not in ("text", "autocomplete", "ntasks_per_node"):
                continue
            forward, backward = Wizard(), Wizard()
            if not forward._blank_is_an_answer(s):
                continue
            for w in (forward, backward):
                w.answers.update({"nodes": 2, s.key: "keep-me"})
                w.idx = _idx(s.key)
            forward._confirm_and_next()
            backward._go_back()
            pair = (forward.answers.get(s.key), backward.answers.get(s.key))
            (agree if pair[0] == pair[1] else differ).append((s.key, pair))
        assert differ == []
        assert {k for k, _ in agree} == {
            "account", "mem_per_cpu", "ntasks_per_node", "constraint",
            "array_spec", "output_dir", "output_file", "custom_sbatch", "modules",
            # Joined the set when _coerce stopped spelling this field's "unset"
            # as "" — see TestEnvNameBlankMeansUnset.
            "env_name",
        }

    def test_control_a_typed_value_still_survives_back(self):
        """Control: Back still commits what is *in* the field. Passes in both
        states — without it the fix could pass by making Back save nothing."""
        w = Wizard()
        w.answers.update({"nodes": 2, "ntasks_per_node": 64})
        w.idx = _idx("ntasks_per_node")
        w._on_enter_step("forward")
        w.text_area.text = "8"
        w._go_back()
        assert w.answers["ntasks_per_node"] == 8

    def test_control_an_invalid_value_still_leaves_the_prior_answer(self):
        """Control: the crash guard this branch was written for. A malformed
        non-empty entry is still not fed to _coerce's int(), so the prior answer
        stands and Back does not raise. Passes in both states."""
        w = Wizard()
        w.answers.update({"cpus": 8})
        w.idx = _idx("cpus")
        w._on_enter_step("forward")
        w.text_area.text = "3.5"
        w._go_back()
        assert w.answers["cpus"] == 8

    def test_control_a_default_reverting_field_keeps_its_prior_answer(self):
        """Control: the scope boundary. On the steps where blank is *not* an
        offered answer, Back still keeps the prior answer — it is not made to
        agree with Enter's substituted default. Passes in both states."""
        w = Wizard()
        for key, prior in (("memory", "32G"), ("time_limit", "04:00:00"),
                           ("cpus", 8), ("nodes", 4), ("gpus", 2)):
            w = Wizard()
            w.answers.update({"nodes": 4, key: prior})
            assert not w._blank_is_an_answer(STEPS[_idx(key)])
            w.idx = _idx(key)
            w._on_enter_step("forward")
            w.text_area.text = ""
            w._go_back()
            assert w.answers[key] == prior, key


class TestEnvNameBlankMeansUnset:
    """``_coerce("")`` for ``env_name`` was ``""``; the builder reads blank as unset.

    The two disagreed, and the disagreement was ``_coerce``'s. ``env_name`` is the
    only *optional* text step whose blank did not coerce to ``None``, and it was
    ``""`` only because it fell off the end of ``_coerce``'s ``if`` chain into the
    bare ``return val`` — the same way ``time_limit`` used to. Nothing wanted that
    spelling: ``build_sbatch_script``'s parameter is ``str | None = None`` and it
    gates on ``if env_name:``; ``env_activation_emitted``, ``check_conda_env`` and
    ``job_summary_rows``' ``add`` all reduce it through truthiness or ``or ""``;
    and the wizard's own skip path (``_setup_env_name`` when env_type is "None
    (skip)") already writes ``None``. So one user-visible state had two encodings,
    and the ``""`` one made ``_blank_is_an_answer`` — which reads ``_coerce`` — say
    blank was not an answer for a field where it is.

    What that cost, driven headlessly from the env_name step with ``env_type =
    Conda`` and ``env_name = stale-env`` already answered, field cleared, one
    keystroke apart:

        ENTER -> ... python train.py
        BACK  -> source "$(conda info --base)/etc/profile.d/conda.sh"
                 conda activate stale-env || { echo '...aborting' >&2; exit 1; }

                 python train.py

    Back is the wrong script twice over: it activates an environment the user
    deleted from the field, and the abort guard makes that fatal — the job exits 1
    at activation and ``python train.py`` never runs. ``sbatch --test-only`` cannot
    adjudicate it (an activation line is script body, not a directive), so the
    evidence is the two scripts built from the same visible state.
    """

    ANSWERS = {
        "job_name": "train", "partition": "amd", "cpus": 4, "memory": "16G",
        "time_limit": "01:00:00", "nodes": 1, "gpus": 0,
        "env_type": "Conda", "env_name": "stale-env",
        "command": "python train.py",
    }

    def _on_env_name_step(self, monkeypatch, **over):
        import slurmate.tui as t
        # Keep the conda picker out of it: a populated list pops the completion
        # menu, which warns about an unawaited coroutine with no event loop.
        monkeypatch.setattr(t, "fetch_conda_envs", lambda mods=None: [])
        w = Wizard()
        w.answers.update({**self.ANSWERS, **over})
        w.idx = _idx("env_name")
        w._on_enter_step("forward")
        return w

    def test_coerce_reads_blank_as_unset(self):
        assert Wizard()._coerce("", STEPS[_idx("env_name")]) is None
        assert Wizard()._blank_is_an_answer(STEPS[_idx("env_name")]) is True

    def test_back_on_a_cleared_env_name_clears_it(self, monkeypatch):
        w = self._on_env_name_step(monkeypatch)
        assert w.text_area.text == "stale-env"      # the field was pre-filled
        w.text_area.text = ""
        w._go_back()
        assert w.answers["env_name"] is None

    def test_the_deleted_environment_leaves_the_script(self, monkeypatch):
        """End-to-end: no resurrected activation, and no guard to abort on."""
        from slurmate.builder import build_from_answers
        w = self._on_env_name_step(monkeypatch)
        w.text_area.text = ""
        w._go_back()
        script = build_from_answers(w.answers)
        assert "conda activate stale-env" not in script
        assert "conda info --base" not in script
        assert "python train.py" in script

    def test_both_gestures_build_the_same_script(self, monkeypatch):
        """The invariant, at script level: from one visible field state, one script."""
        from slurmate.builder import build_from_answers
        fwd = self._on_env_name_step(monkeypatch)
        fwd.text_area.text = ""
        fwd._confirm_and_next()
        bwd = self._on_env_name_step(monkeypatch)
        bwd.text_area.text = ""
        bwd._go_back()
        assert fwd.answers["env_name"] == bwd.answers["env_name"]
        assert build_from_answers(fwd.answers) == build_from_answers(bwd.answers)

    def test_control_a_named_environment_still_activates(self, monkeypatch):
        """Control: the forward path is unchanged for a NON-blank env_name — it
        still commits the name and still emits the activation line. Passes in
        both states; without it the fix could pass by never activating anything."""
        from slurmate.builder import build_from_answers, job_summary_rows
        w = self._on_env_name_step(monkeypatch)
        w.text_area.text = "myenv"
        w._confirm_and_next()
        assert w.answers["env_name"] == "myenv"
        script = build_from_answers(w.answers)
        assert 'source "$(conda info --base)/etc/profile.d/conda.sh"' in script
        assert "conda activate myenv" in script
        assert dict(job_summary_rows(w.answers))["Environment"] == "myenv"
        # ...and Back still commits a typed value too, not only Enter.
        b = self._on_env_name_step(monkeypatch)
        b.text_area.text = "myenv"
        b._go_back()
        assert b.answers["env_name"] == "myenv"

    def test_control_the_nine_other_optional_keys_are_untouched(self):
        """Control: the fix widens the derived rule by exactly one key. The nine
        the rule already covered still coerce blank to None and still have Enter
        and Back agreeing. Passes in both states."""
        nine = ["account", "mem_per_cpu", "ntasks_per_node", "constraint",
                "array_spec", "output_dir", "output_file", "custom_sbatch",
                "modules"]
        for key in nine:
            s = STEPS[_idx(key)]
            assert Wizard()._coerce("", s) is None, key
            assert Wizard()._blank_is_an_answer(s) is True, key
            fwd, bwd = Wizard(), Wizard()
            for w in (fwd, bwd):
                w.answers.update({"nodes": 2, key: "keep-me"})
                w.idx = _idx(key)
            fwd._confirm_and_next()
            bwd._go_back()
            assert fwd.answers[key] == bwd.answers[key] is None, key

    def test_control_the_required_free_text_steps_still_keep_theirs(self):
        """Control: the scope boundary the last round drew is still drawn. This
        is a change to ONE key, not to ``_coerce``'s fallthrough — ``job_name``
        and ``command`` are in ``main._REQUIRED_FIELDS``, their builder
        parameters are plain ``str``, and their blank still coerces to ``""`` so
        Back still keeps the prior answer. Passes in both states."""
        from slurmate.main import _REQUIRED_FIELDS
        required = {k for k, _label in _REQUIRED_FIELDS}
        assert required == {"job_name", "partition", "command"}
        assert "env_name" not in required
        for key, prior in (("job_name", "keeper"), ("command", "echo keeper")):
            s = STEPS[_idx(key)]
            assert Wizard()._coerce("", s) == ""
            assert Wizard()._blank_is_an_answer(s) is False
            w = Wizard()
            w.answers.update({key: prior})
            w.idx = _idx(key)
            w._on_enter_step("forward")
            if getattr(s, "multiline", False):
                w.multiline_text_area.text = ""
            else:
                w.text_area.text = ""
            w._go_back()
            assert w.answers[key] == prior, key
