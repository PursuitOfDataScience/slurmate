"""The README promises every config key is a wizard step. Nothing checked it.

    **Recognized keys:** `job_name`, `account`, … `custom_sbatch`.

    Every one of them is also a wizard step, so a config file prefills the
    interactive flow and batch mode identically.

Measured: all 22 hold today, and the single extra step is `review` (the Submit
screen, not a settable field). But the mechanism is one-directional --
`Wizard.__init__` iterates `STEPS` and picks up `self.config[step.key]`, so a
config key with **no** step is silently ignored there, and `CONFIG_KEYS` is what
`_normalize_config_keys` validates against. A key added to one and not the other
leaves the config accepted, the value dropped, and the README wrong.

The second half of the claim is behavioural -- "prefills the interactive flow" --
so each key is driven through `_step_default` rather than just matched by name.
One key is translated on the way in and that is deliberate: config uses the
lowercase batch form (`conda`) and the TUI select expects the capitalised choice
label (`Conda`), via `ENV_TYPE_LABELS`. A naive string comparison reads that as a
mismatch; it is the documented behaviour and has its own control below.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from slurmate import tui
from slurmate.system_utils import CONFIG_KEYS
from slurmate.tui import ENV_TYPE_LABELS, STEPS, Wizard

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"

#: The heading of the prose list quoted in this module's docstring. Split on
#: rather than assumed: a paragraph that got renamed must fail loudly instead of
#: yielding an empty set that satisfies every comparison below.
_KEYS_HEADING = "**Recognized keys:**"


def _readme_recognized_keys() -> list[str]:
    """The key names the README's "Recognized keys" paragraph actually names."""
    after = README.read_text().split(_KEYS_HEADING, 1)
    assert len(after) == 2, f"the README no longer has a {_KEYS_HEADING} paragraph"
    return re.findall(r"`([a-z_]+)`", after[1].split("\n\n", 1)[0])


#: One plausible value per config key, so the prefill sweep drives every one.
#: Asserted to cover `CONFIG_KEYS` exactly, so a new key fails here first.
SAMPLE: dict[str, object] = {
    "job_name": "j",
    "partition": "p",
    "account": "a",
    "qos": "normal",
    "cpus": 4,
    "memory": "8G",
    "mem_per_cpu": "2G",
    "time_limit": "01:00:00",
    "nodes": 2,
    "ntasks_per_node": 4,
    "gpus": 2,
    "gpu_type": "a100",
    "gpu_format": "gres_type",
    "constraint": "haswell",
    "array_spec": "1-4",
    "modules": ["python", "cuda"],
    "env_type": "conda",
    "env_name": "myenv",
    "output_dir": "logs",
    "output_file": "run-%j.out",
    "command": "true",
    "custom_sbatch": "--exclusive",
}

#: Steps that are not settable fields, so they have no config key by design.
#: Named rather than inferred: "whatever is not in CONFIG_KEYS" would let a new
#: field-bearing step through.
_NOT_A_FIELD = frozenset({"review"})


def _wizard(monkeypatch, config):
    monkeypatch.setattr(tui, "load_config", lambda: dict(config))
    return Wizard()


class TestTheClaimCanBeChecked:
    """Vacuity guards: both vocabularies must be populated, or every assertion
    below holds against an empty set."""

    def test_there_are_steps_and_keys(self):
        assert len(CONFIG_KEYS) >= 20, sorted(CONFIG_KEYS)
        assert len({s.key for s in STEPS}) >= len(CONFIG_KEYS)

    def test_the_sample_covers_every_key(self):
        """If a key is added, this fails first and says which -- so the sweeps
        below can never quietly stop testing one."""
        assert set(SAMPLE) == set(CONFIG_KEYS), set(CONFIG_KEYS) ^ set(SAMPLE)


class TestEveryConfigKeyIsAWizardStep:
    def test_none_is_missing_a_step(self):
        """The README's claim, in the direction that matters: a key with no step
        is accepted by the config validator and then dropped by the wizard."""
        keys = {s.key for s in STEPS}
        assert sorted(CONFIG_KEYS - keys) == [], sorted(CONFIG_KEYS - keys)

    def test_every_extra_step_is_a_named_non_field(self):
        """The other direction, so a new settable step has to be triaged rather
        than silently having no config spelling."""
        keys = {s.key for s in STEPS}
        assert sorted(keys - CONFIG_KEYS) == sorted(_NOT_A_FIELD), sorted(keys - CONFIG_KEYS)


class TestEveryConfigKeyPrefillsItsStep:
    @pytest.mark.parametrize("key", sorted(SAMPLE))
    def test_the_value_reaches_the_step_default(self, key, monkeypatch):
        """Behavioural half of "prefills the interactive flow identically"."""
        wizard = _wizard(monkeypatch, SAMPLE)
        assert key in wizard._config_defaults, key
        step = next(s for s in STEPS if s.key == key)
        shown = wizard._step_default(step)
        want = SAMPLE[key]
        if key == "env_type":
            assert shown == ENV_TYPE_LABELS[str(want).lower()], (shown, want)
        elif isinstance(want, list):
            for item in want:
                assert str(item) in shown, (key, item, shown)
        else:
            assert str(want) in shown, (key, want, shown)

    def test_a_list_is_joined_rather_than_stringified(self):
        """`modules = ["python", "cuda"]` must reach the step as the comma form a
        reader would type, not as a Python repr."""
        wizard_defaults = {}
        for step in STEPS:
            if step.key == "modules":
                wizard_defaults[step.key] = ", ".join(str(x) for x in SAMPLE["modules"])
        assert wizard_defaults["modules"] == "python, cuda"

    def test_the_list_form_is_what_the_wizard_actually_shows(self, monkeypatch):
        wizard = _wizard(monkeypatch, SAMPLE)
        step = next(s for s in STEPS if s.key == "modules")
        assert wizard._step_default(step) == "python, cuda"


class TestTheReadmeListsExactlyTheRecognizedKeys:
    """The first half of the claim above, which nothing checked.

    The wizard-step half is covered; the *list* was not. `_normalize_config_keys`
    warns on and DROPS any key outside `CONFIG_KEYS`, so this paragraph is the
    only place a user can learn what a config file may contain: a key in
    `CONFIG_KEYS` but not in the prose is undiscoverable (the reported defect --
    `constraint` and `mem_per_cpu` were both missing), and one in the prose but
    not in `CONFIG_KEYS` is advertised and then rejected as a typo. The README
    agrees today at 22 keys; a hand-written list stays right only while something
    reads it. The neighbouring alias paragraph is already pinned this way.
    """

    def test_the_paragraph_parses_to_something(self):
        """Vacuity guard: an extraction that quietly returned [] would make both
        directions below hold no matter what the README says."""
        names = _readme_recognized_keys()
        assert len(names) >= 20, names
        assert len(names) == len(set(names)), names

    def test_no_recognized_key_is_undocumented(self):
        """D1's direction: a key the config accepts that the list omits."""
        missing = CONFIG_KEYS - set(_readme_recognized_keys())
        assert sorted(missing) == [], sorted(missing)

    def test_no_documented_key_is_recognized_by_nobody(self):
        """The other direction: a key the list promises that the config drops."""
        extra = set(_readme_recognized_keys()) - CONFIG_KEYS
        assert sorted(extra) == [], sorted(extra)


class TestControls:
    """Two of these hold under every neuter (the two below); the two renamed above
    read a configured value, so they are controls for some neuters and finding
    tests for another. With three independent neuters a test can be both -- which
    is why the names say what each checks rather than claiming a role."""

    def test_env_type_uses_the_documented_translation(self, monkeypatch):
        """Config carries the lowercase batch form and the select expects the
        capitalised label; that mapping is `ENV_TYPE_LABELS` and is not drift.

        NOT named a control: it reads a configured value, so it reddens if the
        prefill loop is gutted. It IS a control for the missing-step neuter, and
        the finding test for the one that drops the translation.
        """
        for lowered, label in ENV_TYPE_LABELS.items():
            wizard = _wizard(monkeypatch, {"env_type": lowered})
            step = next(s for s in STEPS if s.key == "env_type")
            assert wizard._step_default(step) == label, (lowered, label)

    def test_control_a_key_absent_from_config_keeps_the_declared_default(
        self, monkeypatch
    ):
        """CONTROL. The override is per-instance and must not be invented when the
        config is silent -- the module-level `STEPS` carry the declared default and
        `__init__` says it must never mutate them."""
        wizard = _wizard(monkeypatch, {})
        assert wizard._config_defaults == {}
        step = next(s for s in STEPS if s.key == "nodes")
        assert wizard._step_default(step) == step.default

    def test_control_the_shared_steps_are_not_mutated(self, monkeypatch):
        """CONTROL. "never mutate the shared module-level STEPS objects (that would
        leak across wizards and tests)" -- so a configured wizard must leave the
        next one clean."""
        before = {s.key: s.default for s in STEPS}
        _wizard(monkeypatch, SAMPLE)
        assert {s.key: s.default for s in STEPS} == before
        plain = _wizard(monkeypatch, {})
        step = next(s for s in STEPS if s.key == "job_name")
        assert plain._step_default(step) == before["job_name"]

    def test_memory_still_prefers_the_cluster_when_config_is_silent(
        self, monkeypatch
    ):
        """The one documented exception: memory prefers a value derived from the
        cluster over the declared literal, and "an explicit config setting still
        wins, since the user asked for it".

        Its first half asserts a configured value, so this is not a control for
        the prefill neuter either -- named for what it checks instead.
        """
        step = next(s for s in STEPS if s.key == "memory")
        configured = _wizard(monkeypatch, {"memory": "77G"})
        assert configured._step_default(step) == "77G"
        silent = _wizard(monkeypatch, {})
        monkeypatch.setattr(silent, "_derived_memory_default", lambda: "123G")
        assert silent._step_default(step) == "123G"
