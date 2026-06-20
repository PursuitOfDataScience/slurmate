"""Tests for the TUI wizard step definitions and logic."""

from slurmify.system_utils import normalize_memory
from slurmify.tui import STEPS, Step, Wizard, _parse_custom_flags


class TestStepDefinitions:
    def test_all_steps_have_keys(self):
        for s in STEPS:
            assert s.key, f"Step missing key: {s.title}"
            assert s.kind in ("text", "select", "autocomplete", "partition", "gpu_type", "gpu_format", "ntasks_per_node")

    def test_no_duplicate_keys(self):
        keys = [s.key for s in STEPS]
        assert len(keys) == len(set(keys)), "Duplicate step keys found"

    def test_all_steps_have_titles(self):
        for s in STEPS:
            assert s.title, f"Step {s.key} missing title"

    def test_required_keys_have_validation(self):
        """job_name and command have inline required-field checks."""
        from slurmify.tui import Wizard
        w = Wizard()
        for s in STEPS:
            if s.key in ("job_name", "command"):
                assert s.validate is None
                # validation is handled inline in _confirm_and_next

    def test_steps_are_in_correct_order(self):
        expected_order = [
            "job_name", "account", "partition", "qos", "cpus",
            "memory", "time_limit", "nodes", "ntasks_per_node", "gpus", "gpu_type", "gpu_format",
            "array_spec", "modules", "env_type", "env_name", "output_dir", "output_file",
            "command", "custom_sbatch",
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

    def test_coerce_memory(self):
        w = Wizard()
        s = STEPS[5]  # memory
        assert w._coerce("32G", s) == "32G"
        assert w._coerce("64000", s) == "64000M"

    def test_coerce_modules(self):
        w = Wizard()
        s = STEPS[13]  # modules
        assert w._coerce("python/3.10,cuda/12.0", s) == ["python/3.10", "cuda/12.0"]
        assert w._coerce("", s) is None


class TestWizardStepKinds:
    def test_text_step_kind_check(self):
        w = Wizard()
        for idx in [0, 4, 7, 8, 12, 16, 17]:  # text step indices
            w.idx = idx
            assert w._is_text_active(), f"Step {idx} should be text"

    def test_select_step_kind_check(self):
        w = Wizard()
        for idx in [3, 9, 11, 14]:  # select step indices
            w.idx = idx
            assert w._is_select_active(), f"Step {idx} should be select"

    def test_autocomplete_step_kind_check(self):
        w = Wizard()
        for idx in [1, 5, 6, 13, 15, 18]:  # autocomplete step indices
            w.idx = idx
            assert w._is_text_active(), f"Step {idx} should be autocomplete (text)"

    def test_partition_step_kind(self):
        w = Wizard()
        w.idx = 2  # partition
        w._on_enter_step()
        assert w._is_select_active()  # partition starts in select sub-mode

    def test_partition_text_submode(self):
        w = Wizard()
        w.idx = 2
        w.step_cache["partition_sub"] = "text"
        assert w._is_text_active()

    def test_gpu_type_submode_select(self):
        w = Wizard()
        w.idx = 10
        w.answers["gpus"] = 2
        w.answers["partition"] = "gpu-shared"
        w._on_enter_step()
        assert w._is_select_active()

    def test_gpu_type_skip_when_zero_gpus(self):
        w = Wizard()
        w.idx = 10
        w.answers["gpus"] = 0
        old_idx = w.idx
        w._on_enter_step()
        assert w.idx > old_idx  # should auto-advance


class TestPartitionSubFlow:
    def test_setup_partition_creates_radio(self):
        w = Wizard()
        w.idx = 2
        w._on_enter_step()
        assert w.step_cache.get("partition_sub") == "select"
        assert hasattr(w.radio_list, "values")
        values = [v for v, _ in w.radio_list.values]
        assert "Enter partition name manually..." in values

    def test_partition_go_back_from_text(self):
        w = Wizard()
        w.idx = 2
        w.step_cache["partition_sub"] = "text"
        w._go_back()
        assert w.step_cache.get("partition_sub") == "select"

    def test_partition_go_back_from_all(self):
        w = Wizard()
        w.idx = 2
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

    def test_parse_custom_flags_empty(self):
        assert _parse_custom_flags("") == []

    def test_parse_custom_flags_whitespace(self):
        assert _parse_custom_flags("  ,  ,  ") == []


class TestStepValidation:
    def test_validate_cpus_valid(self):
        s = STEPS[4]  # cpus
        assert s.validate is not None
        assert s.validate("4")
        assert s.validate("32")
        assert not s.validate("0")
        assert not s.validate("-1")
        assert not s.validate("abc")

    def test_validate_memory_valid(self):
        s = STEPS[5]  # memory
        assert s.validate is not None
        assert s.validate("16G")
        assert s.validate("64000M")
        assert s.validate("1T")
        assert not s.validate("abc")

    def test_validate_time_valid(self):
        s = STEPS[6]  # time_limit
        assert s.validate is not None
        assert s.validate("01:00:00")
        assert s.validate("1-00:00:00")
        assert s.validate("")
        assert not s.validate("abc")


class TestWizardConfigDefaults:
    def test_config_does_not_mutate_global_steps(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SLURMIFY_MOCK", raising=False)
        (tmp_path / ".slurmify.toml").write_text('cpus = 99\nenv_type = "venv"\n')
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
        from slurmify.tui import STEPS, Wizard
        w = Wizard()
        w.idx = next(i for i, s in enumerate(STEPS) if s.key == "gpus")
        s = STEPS[w.idx]
        w._setup_select(s, None)
        # default is "0" (index 0); arrow down to "4"
        target = w.radio_list.values.index(("4", "4"))
        w.radio_list._selected_index = target
        assert w._radio_value() == "4"

    def test_set_radio_default_moves_cursor(self):
        from slurmify.tui import STEPS, Wizard
        w = Wizard()
        w.idx = next(i for i, st in enumerate(STEPS) if st.key == "gpus")
        s = STEPS[w.idx]
        w._setup_select(s, "2")  # prev answer "2"
        assert w._radio_value() == "2"  # cursor sits on the default, not index 0


class TestFreeNavigation:
    def test_can_skip_required_empty_field(self):
        from slurmify.tui import STEPS, Wizard
        w = Wizard()
        w.idx = next(i for i, s in enumerate(STEPS) if s.key == "job_name")
        w.text_area.text = ""  # leave required job_name blank
        w._confirm_and_next()
        # advanced past it without an error, recording an empty value
        assert "error" not in w.step_cache
        assert w.idx > 0

    def test_invalid_nonempty_still_blocks(self):
        from slurmify.tui import STEPS, Wizard
        w = Wizard()
        w.idx = next(i for i, s in enumerate(STEPS) if s.key == "memory")
        start = w.idx
        w.text_area.text = "not-a-size"
        w._confirm_and_next()
        assert w.step_cache.get("error")
        assert w.idx == start  # did not advance


class TestQosCoerceAndPathCompleter:
    def test_qos_default_coerces_to_none(self):
        from slurmify.tui import STEPS, Wizard
        w = Wizard()
        qos = next(s for s in STEPS if s.key == "qos")
        assert w._coerce("Default (none)", qos) is None
        assert w._coerce("high", qos) == "high"

    def test_path_completer_completes_last_token(self, tmp_path):
        from prompt_toolkit.completion import CompleteEvent
        from prompt_toolkit.document import Document
        from prompt_toolkit.formatted_text import fragment_list_to_text
        from slurmify.tui import LastTokenPathCompleter
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
        from slurmify.tui import STEPS
        path_keys = {s.key for s in STEPS if getattr(s, "path", False)}
        assert {"output_dir", "output_file", "command"} <= path_keys


class TestNoneTextAreaGuards:
    def test_gpu_type_text_branch_with_none(self, monkeypatch):
        # Regression: answers["gpu_type"] == None must not crash TextArea.
        import slurmify.tui as t
        from slurmify.tui import STEPS, Wizard
        monkeypatch.setattr(t, "fetch_gpu_types_for_partition", lambda p: [])
        w = Wizard()
        w.idx = next(i for i, s in enumerate(STEPS) if s.key == "gpu_type")
        w.answers.update({"gpus": 2, "partition": "x", "gpu_type": None})
        w._setup_gpu_type("forward")
        assert w.text_area.text == ""

    def test_env_name_venv_with_none(self, monkeypatch):
        from slurmify.tui import STEPS, Wizard
        w = Wizard()
        w.idx = next(i for i, s in enumerate(STEPS) if s.key == "env_name")
        w.answers.update({"env_type": "Virtualenv (venv)", "env_name": None})
        w._setup_env_name("forward")
        assert w.text_area.text == ""
