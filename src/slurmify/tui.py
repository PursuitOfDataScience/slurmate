from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    FuzzyWordCompleter,
    PathCompleter,
)
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import (
    Float,
    FloatContainer,
    HSplit,
    VSplit,
    Window,
    WindowAlign,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.widgets import Frame, RadioList, TextArea

from .builder import build_from_answers
from .system_utils import (
    fetch_available_modules,
    fetch_conda_envs,
    fetch_gpu_types_for_partition,
    fetch_known_qos,
    fetch_partitions,
    fetch_public_partitions,
    fetch_qos_for_partition,
    fetch_user_accounts,
    load_config,
    normalize_memory,
    validate_memory,
    validate_time,
)

# Setup debug logging if SLURMIFY_DEBUG is set
logger = logging.getLogger(__name__)
if os.environ.get("SLURMIFY_DEBUG"):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


_TUI_STYLE = PTStyle([
    ("status-bar", "bg:#0088ff fg:#ffffff bold"),
    ("sidebar-done", "fg:#00ff80 bold"),
    ("sidebar-current", "fg:#bf00ff bold"),
    ("sidebar-pending", "fg:#555555"),
    ("title", "fg:#00ffff bold"),
    ("subtitle", "fg:#888888"),
    ("text-area", "fg:#ffffff bg:#333333"),
    ("text-area focused", "fg:#ffffff bg:#333333"),
    ("radio-list", "fg:#ffffff"),
    ("radio-list.selected", "fg:#00ff80 bold"),
    ("radio-list.pointer", "fg:#bf00ff bold"),
    ("checkbox", "fg:#888888"),
    ("checkbox.selected", "fg:#00ff80"),
    ("preview-header", "fg:#00ffff bold"),
    ("preview-text", "fg:#aaaaaa"),
    ("error", "fg:#ff4444 bold"),
    ("warning", "fg:#ffaa00 bold"),
    ("info", "fg:#888888"),
    ("completion-menu", "bg:#222222 fg:#cccccc"),
    ("completion-menu.completion", "bg:#222222 fg:#cccccc"),
    ("completion-menu.completion.current", "bg:#0088ff fg:#ffffff bold"),
])


CUSTOM = "Enter partition name manually..."
PRIVATE = "Include private partitions"

# Maps the lowercase env_type values used by config/batch mode to the
# capitalized choice labels shown by the interactive "Environment type" step.
ENV_TYPE_LABELS = {
    "conda": "Conda",
    "mamba": "Mamba",
    "venv": "Virtualenv (venv)",
    "none": "None (skip)",
}


class LastTokenPathCompleter(Completer):
    """Filesystem completion for the last whitespace-separated token.

    Lets users tab-complete file/dir paths while typing a command (e.g.
    ``python train.py``) or a single path field, without retyping long paths —
    the more they type, the narrower the suggestions get.
    """

    def __init__(self) -> None:
        self._pc = PathCompleter(expanduser=True)

    def get_completions(self, document: Document, complete_event: CompleteEvent):  # type: ignore[no-untyped-def]
        text = document.text_before_cursor
        cut = max(text.rfind(" "), text.rfind("\t"), text.rfind("\n")) + 1
        token = text[cut:]
        if not token:
            return
        sub = Document(token, len(token))
        for comp in self._pc.get_completions(sub, complete_event):
            yield Completion(
                comp.text,
                start_position=comp.start_position,
                display=comp.display,
                display_meta=comp.display_meta_text,
            )


def _get_partition(partitions: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for p in partitions:
        if p["name"] == name:
            return p
    return {"name": name, "nodes": 0, "cpus_per_node": 0, "mem_per_node_mb": 0,
            "gpu_types": [], "timelimit": None, "is_public": True}


def _fmt_partition(p: dict[str, Any]) -> str:
    name = p["name"]
    nodes = p.get("nodes", "?")
    cpus = p.get("cpus_per_node", "?")
    mem_gb = p.get("mem_per_node_mb", 0) // 1024
    gpus = p.get("gpu_types", [])
    label = f"{name:<12} {nodes} nodes \u00b7 {cpus} CPU \u00b7 {mem_gb}G"
    if gpus:
        label += f" \u00b7 GPU:[{','.join(gpus)}]"
    return label


def _parse_custom_flags(raw: str) -> list[str]:
    flags = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("#SBATCH"):
            part = part.replace("#SBATCH", "", 1).strip()
        if not part.startswith("--"):
            part = f"--{part}"
        flags.append(part)
    return flags


MEMORY_CHOICES = ["4G", "8G", "16G", "32G", "64G", "128G", "256G", "512G", "64000M"]
TIME_CHOICES = ["01:00:00", "02:00:00", "04:00:00", "08:00:00", "12:00:00",
                "24:00:00", "48:00:00", "7-00:00:00"]
SBATCH_FLAGS = [
    "--exclusive", "--reservation=", "--ntasks=", "--ntasks-per-node=",
    "--threads-per-core=", "--mem-per-cpu=", "--constraint=",
    "--licenses=", "--gres=", "--tmp=", "--hint=", "--signal=",
]


# ── Step definitions ─────────────────────────────────────────────────────

class Step:
    key: str
    title: str
    kind: str
    subtitle: str = ""
    choices: list[str] | None = None
    default: str = ""
    validate: Callable[[str], bool] | None = None
    fetch: Callable[..., Any] | None = None
    required: bool = False
    multiline: bool = False
    path: bool = False

    def __init__(self, key: str, title: str, kind: str, **kw: Any) -> None:
        self.key = key
        self.title = title
        self.kind = kind
        self.required = False
        self.multiline = False
        self.path = False
        for k, v in kw.items():
            setattr(self, k, v)


# Steps are ordered to match the #SBATCH directive order the builder emits, so
# the live preview grows top-to-bottom as you answer (no reshuffling). All
# directive-producing steps come first, then setup (modules/env), then command.
STEPS: list[Step] = [
    Step("job_name", "Job name", "text", subtitle="A name for your Slurm job", required=True),
    Step("partition", "Partition", "partition",
         fetch=lambda: (fetch_public_partitions(), fetch_partitions())),
    Step("account", "Account", "autocomplete",
         subtitle="Slurm account to charge (optional)",
         fetch=fetch_user_accounts),
    Step("qos", "QoS", "select", subtitle="Quality of Service",
         choices=["Default (none)"], default="Default (none)",
         fetch=lambda part: fetch_qos_for_partition(part)),
    Step("cpus", "CPU cores", "text",
         subtitle="Number of CPU cores per task", default="4",
         validate=lambda v: v.strip().isdigit() and int(v) > 0),
    Step("memory", "Memory", "autocomplete",
         subtitle="e.g. 16G, 32G, 64000M",
         validate=validate_memory, default="16G",
         choices=MEMORY_CHOICES),
    Step("time_limit", "Time limit", "autocomplete",
         subtitle="Format: hh:mm:ss or d-hh:mm:ss",
         validate=validate_time, default="02:00:00",
         choices=TIME_CHOICES),
    Step("nodes", "Nodes", "text", subtitle="Number of nodes", default="1",
         validate=lambda v: v.strip().isdigit() and int(v) > 0),
    Step("ntasks_per_node", "Tasks per node", "ntasks_per_node",
         subtitle="Tasks per node (optional, for multi-node)", default="1",
         validate=lambda v: not v.strip() or (v.strip().isdigit() and int(v) > 0)),
    Step("gpus", "GPUs", "select", subtitle="Number of GPUs",
         choices=["0", "1", "2", "4", "8"], default="0"),
    Step("gpu_type", "GPU type", "gpu_type", subtitle="GPU hardware type"),
    Step("gpu_format", "GPU format", "gpu_format", subtitle="Format style for GPU requests"),
    Step("array_spec", "Array spec", "text",
         subtitle="e.g. 1-10, 1,3,5-7%4 (optional)"),
    Step("output_dir", "Output directory", "text",
         subtitle="Directory for stdout/stderr logs (optional)", default="logs", path=True),
    Step("output_file", "Output file", "text",
         subtitle="Log file name, %j = job ID (optional, blank = <job>-%j.out)", path=True),
    Step("custom_sbatch", "Custom #SBATCH flags", "autocomplete",
         subtitle="e.g. --exclusive --reservation=abc (optional)",
         choices=SBATCH_FLAGS),
    Step("modules", "Modules", "autocomplete",
         subtitle="Comma-separated, e.g. python/anaconda,cuda (optional)",
         fetch=fetch_available_modules),
    Step("env_type", "Environment type", "select",
         subtitle="Environment activation strategy",
         choices=["None (skip)", "Conda", "Mamba", "Virtualenv (venv)"],
         default="None (skip)"),
    Step("env_name", "Environment name/path", "autocomplete",
         subtitle="Conda environment name or virtualenv path",
         default=""),
    Step("command", "Command to run", "text",
         subtitle="e.g. python train.py  (Tab completes file paths)",
         required=True, multiline=True, path=True),
]


# ── Wizard ───────────────────────────────────────────────────────────────

class Wizard:
    """Full-screen TUI wizard for generating and submitting sbatch scripts."""

    def __init__(self) -> None:
        self.idx = 0
        self.answers: dict[str, Any] = {}
        self.step_cache: dict[str, Any] = {}
        self.transient: dict[str, Any] = {}
        self.submitted = False
        self.config = load_config()
        # Per-instance default overrides from config — never mutate the shared
        # module-level STEPS objects (that would leak across wizards and tests).
        self._config_defaults: dict[str, str] = {}
        for step in STEPS:
            if step.key not in self.config:
                continue
            val = self.config[step.key]
            if isinstance(val, list):
                self._config_defaults[step.key] = ", ".join(str(x) for x in val)
            elif step.key == "env_type":
                # Config uses the lowercase batch form (conda/mamba/venv/none);
                # the TUI select expects the capitalized choice labels.
                self._config_defaults[step.key] = ENV_TYPE_LABELS.get(str(val).lower(), str(val))
            else:
                self._config_defaults[step.key] = str(val)
        self.app: Application[Any]
        # Text input widget (shared across text/autocomplete/partition-text/gpu-text steps)
        self.text_area = TextArea(
            multiline=False,
            style="bg:#333333 fg:#ffffff",
            scrollbar=False,
        )

        self.multiline_text_area = TextArea(
            multiline=True,
            style="bg:#333333 fg:#ffffff",
            scrollbar=True,
        )

        # RadioList (shared across select/partition/gpu_type steps)
        self.radio_list = RadioList([("", "")])

        self.show_help = False
        # Mouse capture OFF by default so the terminal can natively select/copy
        # the script preview. Navigation is fully keyboard-driven; press F2 to
        # enable mouse click/scroll if preferred.
        self.mouse_enabled = False
        self._path_completer = LastTokenPathCompleter()

        self._build_app()

    def _invalidate(self) -> None:
        if self.app and self.app.is_running:
            self.app.layout = self._build_layout()
            if self._is_text_active():
                s = self.current_step
                if getattr(s, "multiline", False):
                    self.app.layout.focus(self.multiline_text_area)
                else:
                    self.app.layout.focus(self.text_area)
            elif self._is_select_active():
                self.app.layout.focus(self.radio_list)
            self.app.invalidate()

    @property
    def current_step(self) -> Step:
        return STEPS[self.idx] if self.idx < len(STEPS) else STEPS[-1]

    def _is_text_active(self) -> bool:
        s = self.current_step
        if s.kind in ("text", "autocomplete", "ntasks_per_node"):
            return True
        if s.kind == "partition":
            return self.step_cache.get("partition_sub") == "text"
        if s.kind == "gpu_type":
            return self.step_cache.get("gpu_sub") == "text"
        return False

    def _is_select_active(self) -> bool:
        s = self.current_step
        if s.kind in ("select", "gpu_format"):
            return True
        if s.kind == "partition":
            sub = self.step_cache.get("partition_sub", "select")
            return sub in ("select", "all")
        if s.kind == "gpu_type":
            return bool(self.step_cache.get("gpu_sub", "select") == "select")
        return False

    def _can_go_back(self) -> bool:
        s = self.current_step
        if s.kind == "partition":
            sub = self.step_cache.get("partition_sub", "select")
            if sub in ("all", "text"):
                return True
        return self.idx > 0

    # ── Key bindings ────────────────────────────────────────────────

    def _build_app(self) -> None:
        kb = KeyBindings()

        @kb.add("f1")
        @kb.add("?")
        def _toggle_help(event: Any) -> None:
            self.show_help = not self.show_help
            self._invalidate()

        @kb.add("escape", eager=True, filter=Condition(lambda: self.show_help))
        @kb.add("enter", eager=True, filter=Condition(lambda: self.show_help))
        def _close_help(event: Any) -> None:
            self.show_help = False
            self._invalidate()

        @kb.add("tab", eager=True, filter=Condition(lambda: not self.show_help))
        def _tab(event: Any) -> None:
            buf = self._focused_buffer()
            if buf is not None and buf.complete_state is not None:
                buf.complete_next()  # cycle suggestions instead of advancing
                return
            self._confirm_and_next()

        @kb.add("enter", eager=True, filter=Condition(lambda: not self.show_help))
        def _enter(event: Any) -> None:
            buf = self._focused_buffer()
            if buf is not None and buf.complete_state is not None \
                    and buf.complete_state.current_completion is not None:
                buf.apply_completion(buf.complete_state.current_completion)
                return
            self._confirm_and_next()

        @kb.add("s-tab", eager=True, filter=Condition(lambda: self._can_go_back() and not self.show_help))
        def _stab(event: Any) -> None:
            self._go_back()

        @kb.add("escape", eager=True, filter=Condition(lambda: self.idx > 0 and not self.show_help))
        def _esc(event: Any) -> None:
            self._go_back()

        @kb.add("f2", eager=True, filter=Condition(lambda: not self.show_help))
        def _toggle_mouse(event: Any) -> None:
            # Releasing mouse capture lets the terminal do native text selection,
            # so users can highlight/copy the script preview. Toggle it back on
            # to use click/scroll inside the TUI again.
            self.mouse_enabled = not self.mouse_enabled
            self._invalidate()

        @kb.add("c-c")
        def _cc(event: Any) -> None:
            raise KeyboardInterrupt

        self.app = Application(
            layout=self._build_layout(),
            key_bindings=kb,
            full_screen=True,
            style=_TUI_STYLE,
            mouse_support=Condition(lambda: self.mouse_enabled),
        )

    # ── Navigation ──────────────────────────────────────────────────

    def _text_val(self) -> str:
        s = self.current_step
        if getattr(s, "multiline", False):
            return self.multiline_text_area.text.strip()
        return self.text_area.text.strip()

    def _focused_buffer(self) -> Any:
        """The active text buffer (single- or multi-line), or None for lists."""
        if not self._is_text_active():
            return None
        s = self.current_step
        if getattr(s, "multiline", False):
            return self.multiline_text_area.buffer
        return self.text_area.buffer

    def _radio_value(self) -> Any:
        """Value of the currently highlighted RadioList row.

        We read ``_selected_index`` (the cursor position) rather than
        ``current_value`` because the wizard binds Enter with ``eager=True``,
        which preempts RadioList's own Enter handler — the only place that would
        otherwise sync ``current_value`` to the highlighted row. Without this,
        every select step returns its initial value regardless of navigation
        (e.g. the partition list always returned "Enter manually...").
        """
        rl = self.radio_list
        idx = getattr(rl, "_selected_index", 0)
        if 0 <= idx < len(rl.values):
            return rl.values[idx][0]
        return rl.current_value

    def _set_radio_default(self, value: str) -> None:
        """Move the RadioList cursor + selection to ``value`` if present."""
        rl = self.radio_list
        for i, (v, _label) in enumerate(rl.values):
            if v == value:
                rl._selected_index = i
                rl.current_value = value
                return

    def _confirm_and_next(self) -> None:
        s = self.current_step

        if s.kind == "partition":
            self._handle_partition_confirm()
            return
        if s.kind == "gpu_type":
            self._handle_gpu_type_confirm()
            return

        if s.kind in ("select", "gpu_format"):
            val = self._radio_value()
            if val:
                self.answers[s.key] = self._coerce(val, s)
                self._advance()
            return

        if s.kind in ("text", "autocomplete", "ntasks_per_node"):
            val = self._text_val()
            # Empty is always allowed — the user can skip a step and come back.
            # Required fields are flagged at the final review instead of blocking
            # navigation here. Only malformed *non-empty* input is rejected.
            if val and s.validate and not s.validate(val):
                self.step_cache["error"] = f"Invalid input: {s.subtitle or ''}"
                self._invalidate()
                return
            self.step_cache.pop("error", None)
            self.answers[s.key] = self._coerce(val, s)
            self._advance()
            return

    def _advance(self) -> None:
        self.idx += 1
        if self.idx >= len(STEPS):
            self.app.exit()
            return
        self.transient["preview_dirty"] = True
        self._on_enter_step("forward")
        self._invalidate()

    def _go_back(self) -> None:
        s = self.current_step
        if s.kind == "partition":
            sub = self.step_cache.get("partition_sub", "select")
            if sub == "all":
                self.step_cache["partition_sub"] = "select"
                self._invalidate()
                return
            if sub == "text":
                self.step_cache["partition_sub"] = "select"
                self._invalidate()
                return
        if s.kind == "gpu_type":
            sub = self.step_cache.get("gpu_sub", "select")
            if sub == "text":
                self.step_cache["gpu_sub"] = "select"
                self.step_cache.pop("gpu_types", None)
                self._invalidate()
                return
        self.step_cache.pop("partition_sub", None)
        self.step_cache.pop("gpu_sub", None)
        self.step_cache.pop("error", None)
        self.idx = max(0, self.idx - 1)
        self._on_enter_step("backward")
        self._invalidate()

    def _coerce(self, val: str, s: Step) -> Any:
        if s.key in ("cpus", "gpus"):
            return int(val) if val else 4
        if s.key == "nodes":
            return int(val) if val else 1
        if s.key == "ntasks_per_node":
            return int(val) if val else None
        if s.key == "memory":
            return normalize_memory(val) if val else "16G"
        if s.key == "modules":
            return [m.strip() for m in val.split(",") if m.strip()] if val else None
        if s.key == "qos":
            return None if (not val or val == "Default (none)") else val
        if s.key in ("account", "array_spec", "gpu_type", "gpu_format",
                     "output_dir", "output_file"):
            return val or None
        return val

    def _get_warning(self) -> str | None:
        part = self.answers.get("_partition_obj")
        if not part:
            return None

        s = self.current_step
        if s.key == "cpus":
            val = self._text_val()
            if val.isdigit():
                cores = int(val)
                limit = part.get("cpus_per_node", 0)
                if limit and cores > limit:
                    return f"CPUs ({cores}) exceeds partition limit ({limit} per node)"
        elif s.key == "memory":
            val = self._text_val()
            if validate_memory(val):
                norm = normalize_memory(val)
                m = re.match(r"^(\d+)([MGT]?)$", norm)
                if m:
                    amt = int(m.group(1))
                    unit = m.group(2)
                    mb = amt
                    if unit == "G":
                        mb = amt * 1024
                    elif unit == "T":
                        mb = amt * 1024 * 1024
                    limit = part.get("mem_per_node_mb", 0)
                    if limit and mb > limit:
                        return f"Memory exceeds partition limit ({limit} MB per node)"
        elif s.key == "time_limit":
            val = self._text_val()
            if val:
                from .system_utils import _parse_slurm_time_to_minutes
                try:
                    req_mins = _parse_slurm_time_to_minutes(val)
                    limit_str = part.get("timelimit")
                    if limit_str:
                        limit_mins = _parse_slurm_time_to_minutes(limit_str)
                        if limit_mins > 0 and req_mins > limit_mins:
                            return f"Time limit exceeds partition limit ({limit_str})"
                except Exception:
                    pass
        elif s.key == "gpus":
            val = self._radio_value()
            if val and val.isdigit():
                gpus = int(val)
                gpu_types = part.get("gpu_types", [])
                if gpus > 0 and not gpu_types:
                    return f"Partition '{part.get('name')}' does not support GPUs"
        elif s.key == "gpu_type":
            val = self._radio_value() if self._is_select_active() else self._text_val()
            gpu_types = part.get("gpu_types", [])
            if val and gpu_types and val not in gpu_types:
                return f"GPU type '{val}' not in partition list ({', '.join(gpu_types)})"

        return None

    def _step_default(self, s: Step) -> str:
        """Default value for a step, preferring a per-instance config override."""
        return self._config_defaults.get(s.key, s.default)

    def _on_enter_step(self, direction: str = "forward") -> None:
        """Called whenever entering a step (forward or backward)."""
        self.step_cache.pop("error", None)
        s = self.current_step
        prev = self.answers.get(s.key)

        part = self.answers.get("partition")
        if part:
            cached_part = self.transient.get("queue_info_part")
            if cached_part != part:
                try:
                    from .system_utils import fetch_queue_eta
                    qinfo = fetch_queue_eta(part, req_nodes=self.answers.get("nodes", 1))
                    self.transient["queue_info"] = qinfo
                    self.transient["queue_info_part"] = part
                except Exception as e:
                    logger.debug(f"Failed to fetch queue info in TUI: {e}")

        if s.kind in ("text", "ntasks_per_node"):
            if s.kind == "ntasks_per_node":
                self._setup_ntasks_per_node(direction)
            elif getattr(s, "multiline", False):
                self.multiline_text_area.text = str(prev or self._step_default(s) or "")
                self._set_multiline_completer(self._path_completer if getattr(s, "path", False) else None)
            else:
                self.text_area.text = str(prev or self._step_default(s) or "")
                self._set_completer(self._path_completer if getattr(s, "path", False) else None)
        elif s.kind == "autocomplete":
            if s.key == "env_name":
                self._setup_env_name(direction)
            else:
                self.text_area.text = str(prev or self._step_default(s) or "")
                self._setup_autocomplete(s)
        elif s.kind == "select":
            self._setup_select(s, prev)
        elif s.kind == "partition":
            self._setup_partition()
        elif s.kind == "gpu_type":
            self._setup_gpu_type(direction)
        elif s.kind == "gpu_format":
            self._setup_gpu_format(direction)

    # ── Autocomplete ────────────────────────────────────────────────

    def _set_completer(self, completer: Completer | None) -> None:
        self.text_area.buffer.completer = completer  # type: ignore[assignment]
        self.text_area.buffer.complete_while_typing = Condition(lambda: completer is not None)

    def _set_multiline_completer(self, completer: Completer | None) -> None:
        self.multiline_text_area.buffer.completer = completer  # type: ignore[assignment]
        self.multiline_text_area.buffer.complete_while_typing = Condition(lambda: completer is not None)

    def _setup_autocomplete(self, s: Step) -> None:
        choices = self._resolve_choices(s)
        if choices:
            self._set_completer(FuzzyWordCompleter(choices))
        else:
            self._set_completer(None)

    # ── Select ──────────────────────────────────────────────────────

    def _setup_select(self, s: Step, prev: Any = None) -> None:
        choices = self._resolve_choices(s)
        if not choices:
            choices = s.choices or []
        if not choices:
            self.answers[s.key] = None
            self._advance()
            return
        pairs = [(c, c) for c in choices]
        self.radio_list = RadioList(pairs)
        default = str(prev or self._step_default(s) or "")
        if default and default in choices:
            self._set_radio_default(default)

    def _resolve_choices(self, s: Step) -> list[str]:
        key = f"choices_{s.key}"
        if key in self.step_cache:
            from typing import cast
            return cast(list[str], self.step_cache[key])
        if s.fetch:
            try:
                if s.key == "qos":
                    part = self.answers.get("partition", "")
                    raw = s.fetch(part)
                    known = set(fetch_known_qos())
                    raw = [q for q in raw if q in known]
                    result = (["Default (none)"] + raw) if raw else []
                elif s.key == "env_name":
                    raw = s.fetch()
                    result = ["None (skip)"] + raw
                else:
                    result = s.fetch()
            except Exception as e:
                logger.debug(f"Failed to fetch {key}: {e}")
                result = s.choices or []
            self.step_cache[key] = result
            return result
        return s.choices or []

    # ── Partition sub-flow ──────────────────────────────────────────

    def _setup_partition(self) -> None:
        self.step_cache["partition_sub"] = "select"
        try:
            public, all_parts = fetch_public_partitions(), fetch_partitions()
        except Exception as e:
            logger.debug(f"Failed to fetch partitions: {e}")
            public, all_parts = [], []
        self.transient["public_parts"] = public
        self.transient["all_parts"] = all_parts
        choices = [CUSTOM]
        if public:
            choices.append(PRIVATE)
            choices.extend(_fmt_partition(p) for p in public)
        elif all_parts:
            choices.extend(_fmt_partition(p) for p in all_parts)
        self.radio_list = RadioList([(c, c) for c in choices])

    def _handle_partition_confirm(self) -> None:
        sub = self.step_cache.get("partition_sub", "select")
        if sub == "select":
            raw = self._radio_value()
            if raw == CUSTOM:
                self.text_area.text = self.answers.get("partition") or ""
                self.step_cache["partition_sub"] = "text"
                self._set_completer(None)
                self._invalidate()
                return
            if raw == PRIVATE:
                all_parts = self.transient.get("all_parts", [])
                fmt_all = [_fmt_partition(p) for p in all_parts]
                self.radio_list = RadioList([(c, c) for c in fmt_all])
                self.step_cache["partition_sub"] = "all"
                self._invalidate()
                return
            self._set_partition_from_select(raw)
            self._advance()
            return
        if sub == "text":
            val = self._text_val()
            all_parts = self.transient.get("all_parts", [])
            self.answers["partition"] = val
            self.answers["_partition_obj"] = _get_partition(all_parts, val) if val else None
            self._advance()
            return
        if sub == "all":
            raw = self._radio_value()
            self._set_partition_from_fmt(raw)
            self._advance()
            return

    def _set_partition_from_select(self, raw: str) -> None:
        choices = self.radio_list.values
        idx = next((i for i, (v, _) in enumerate(choices) if v == raw), -1)
        public = self.transient.get("public_parts", [])
        has_private = PRIVATE in [v for v, _ in choices]
        public_idx = idx - (2 if has_private else 1)
        if 0 <= public_idx < len(public):
            part = public[public_idx]
            self.answers["partition"] = part["name"]
            self.answers["_partition_obj"] = part
        else:
            self.answers["partition"] = raw
            self.answers["_partition_obj"] = _get_partition(
                self.transient.get("all_parts", []), raw)

    def _set_partition_from_fmt(self, raw: str) -> None:
        all_parts = self.transient.get("all_parts", [])
        for p in all_parts:
            if _fmt_partition(p) == raw:
                self.answers["partition"] = p["name"]
                self.answers["_partition_obj"] = p
                return

    # ── GPU type sub-flow ───────────────────────────────────────────

    def _setup_gpu_type(self, direction: str = "forward") -> None:
        gpus = self.answers.get("gpus", 0)
        if isinstance(gpus, str):
            gpus = int(gpus) if gpus.isdigit() else 0
        if gpus == 0:
            self.answers["gpu_type"] = None
            if direction == "forward":
                self._advance()
            else:
                self._go_back()
            return

        part_name = self.answers.get("partition", "")
        try:
            gpu_types = fetch_gpu_types_for_partition(part_name)
        except Exception as e:
            logger.debug(f"Failed to fetch GPU types for partition {part_name}: {e}")
            gpu_types = []
        self.transient["gpu_types"] = gpu_types
        if gpu_types:
            choices = ["Any"] + gpu_types
            self.radio_list = RadioList([(c, c) for c in choices])
            prev = self.answers.get("gpu_type")
            self._set_radio_default(prev if prev and prev in choices else "Any")
            self.step_cache["gpu_sub"] = "select"
        else:
            self.text_area.text = self.answers.get("gpu_type") or ""
            self.step_cache["gpu_sub"] = "text"
            self._set_completer(None)

    def _setup_gpu_format(self, direction: str = "forward") -> None:
        gpus = self.answers.get("gpus", 0)
        if isinstance(gpus, str):
            gpus = int(gpus) if gpus.isdigit() else 0
        if gpus == 0:
            self.answers["gpu_format"] = None
            if direction == "forward":
                self._advance()
            else:
                self._go_back()
            return

        choices = ["gres_type", "constraint", "gpus"]
        self.radio_list = RadioList([(c, c) for c in choices])
        prev = self.answers.get("gpu_format")
        self._set_radio_default(prev if prev and prev in choices else "gres_type")

    def _setup_ntasks_per_node(self, direction: str = "forward") -> None:
        nodes = self.answers.get("nodes", 1)
        if isinstance(nodes, str):
            nodes = int(nodes) if nodes.isdigit() else 1
        if nodes <= 1:
            self.answers["ntasks_per_node"] = None
            if direction == "forward":
                self._advance()
            else:
                self._go_back()
            return

        self.text_area.text = str(self.answers.get("ntasks_per_node") or "1")
        self._set_completer(None)

    def _setup_env_name(self, direction: str = "forward") -> None:
        env_type = self.answers.get("env_type", "None (skip)")
        if env_type == "None (skip)":
            self.answers["env_name"] = None
            if direction == "forward":
                self._advance()
            else:
                self._go_back()
            return

        if env_type in ("Conda", "Mamba"):
            try:
                envs = fetch_conda_envs()
            except Exception as e:
                logger.debug(f"Failed to fetch conda envs: {e}")
                envs = []
            self.step_cache["choices_env_name"] = envs
            self.text_area.text = self.answers.get("env_name") or ""
            self._setup_autocomplete(self.current_step)
        else:  # Virtualenv (venv) — complete filesystem paths
            self.text_area.text = self.answers.get("env_name") or ""
            self._set_completer(self._path_completer)

    def _handle_gpu_type_confirm(self) -> None:
        sub = self.step_cache.get("gpu_sub", "select")
        if sub == "select":
            val = self._radio_value()
            self.answers["gpu_type"] = None if val == "Any" else val
        elif sub == "text":
            val = self._text_val()
            self.answers["gpu_type"] = val or None
        self._advance()



    # ── Layout ──────────────────────────────────────────────────────

    def _help_modal(self) -> Frame:
        help_text = (
            "\n"
            "  \u26a1  Slurmify TUI Help  \u26a1\n\n"
            "  Keyboard Shortcuts:\n"
            "    Enter / Tab      Go to next step\n"
            "    Esc / Shift+Tab  Go back to the previous step\n"
            "    \u2191 \u2193              Move within a list\n"
            "    F2               Toggle mouse capture (off = select/copy text)\n"
            "    F1 / ?           Toggle this help menu\n"
            "    Ctrl + C         Exit wizard\n\n"
            "  You can skip any step (leave it blank) and come back later.\n"
            "  Missing required fields are flagged before you submit.\n\n"
            "  Press F1 or ? to close this menu\n"
        )
        return Frame(
            body=Window(
                content=FormattedTextControl(help_text),
                dont_extend_width=True,
                dont_extend_height=True,
            ),
            style="bg:#222222 fg:#ffffff border:#0088ff",
        )

    def _build_layout(self) -> Layout:
        floats = [Float(xcursor=True, ycursor=True, content=CompletionsMenu())]
        if self.show_help:
            floats.append(Float(content=self._help_modal()))

        focused: Any = self.text_area
        if self._is_text_active():
            s = self.current_step
            if getattr(s, "multiline", False):
                focused = self.multiline_text_area
            else:
                focused = self.text_area
        elif self._is_select_active():
            focused = self.radio_list

        return Layout(
            FloatContainer(
                HSplit([
                    self._header(),
                    VSplit([self._sidebar(), self._content()], padding=1),
                    self._footer(),
                ]),
                floats=floats,
            ),
            focused_element=focused,
        )

    def _header(self) -> VSplit:
        bar_style = "bg:#0088ff fg:#ffffff bold"
        return VSplit([
            Window(
                FormattedTextControl(self._render_header_left),
                height=1, style=bar_style, dont_extend_height=True,
            ),
            Window(
                FormattedTextControl(self._render_header_right),
                height=1, style=bar_style, align=WindowAlign.RIGHT,
            ),
        ])

    def _render_header_left(self) -> list[tuple[str, str]]:
        return [("class:status-bar", "  \u26a1  Slurmify \u2014 sbatch wizard")]

    def _render_header_right(self) -> list[tuple[str, str]]:
        s = self.current_step
        right = f"  Step {self.idx + 1}/{len(STEPS)}  {s.title}"
        progress = "".join(
            "\u25c9" if i <= self.idx else "\u25cb"
            for i in range(len(STEPS))
        )
        return [("class:status-bar", f"  {right}  {progress}  ")]

    def _sidebar(self) -> Window:
        return Window(
            FormattedTextControl(self._render_sidebar),
            width=24,
            style="bg:#1a1a2e",
        )

    def _render_sidebar(self) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = [("class:subtitle", "  Steps\n\n")]
        for i, s in enumerate(STEPS):
            if i < self.idx:
                lines.append(("class:sidebar-done", f"  \u2713 {s.title}\n"))
            elif i == self.idx:
                lines.append(("class:sidebar-current", f"  \u25b6 {s.title}\n"))
            else:
                lines.append(("class:sidebar-pending", f"    {s.title}\n"))
        return lines

    def _content(self) -> HSplit:
        s = self.current_step

        title_text = f"\n  {s.title}\n"
        subtitle_text = f"  {s.subtitle}\n\n"

        error_control: list[Window] = []
        if self.step_cache.get("error"):
            error_control.append(Window(
                FormattedTextControl([("class:error", f"  \u2717 {self.step_cache['error']}\n")]),
                height=1, style="",
            ))
        warning_text = self._get_warning()
        if warning_text:
            error_control.append(Window(
                FormattedTextControl([("class:warning", f"  \u26a0 {warning_text}\n")]),
                height=1, style="",
            ))

        title_win = Window(
            FormattedTextControl([
                ("class:title", title_text),
                ("class:subtitle", subtitle_text),
            ]),
            height=2 + (1 if subtitle_text else 0),
            dont_extend_height=True,
        )

        text_active = self._is_text_active()
        select_active = self._is_select_active()

        children: list[Any] = []
        if text_active:
            if getattr(s, "multiline", False):
                children = [self.multiline_text_area]
            else:
                children = [self.text_area]
        elif select_active:
            children = [self.radio_list]

        return HSplit([title_win] + error_control + children + [self._queue_panel(), self._preview_panel()])

    def _queue_panel(self) -> Window:
        qinfo = self.transient.get("queue_info")
        h = 2 if qinfo else 0
        return Window(
            FormattedTextControl(self._render_queue_text),
            style="bg:#1a1a2e",
            dont_extend_height=True,
            height=h,
        )

    def _render_queue_text(self) -> list[tuple[str, str]]:
        qinfo = self.transient.get("queue_info")
        if not qinfo:
            return []
        part = self.transient.get("queue_info_part", "")
        eta_sec = qinfo.get("eta_seconds", 0)
        eta_color = "fg:#00ff80 bold" if eta_sec < 3600 else "fg:#ffaa00 bold"
        return [
            ("", "\n  "),
            ("class:preview-header", f"Queue status ({part}): "),
            ("class:info", f"{qinfo.get('running', 0)} running / {qinfo.get('pending', 0)} pending   "),
            ("class:preview-header", "Est. ETA (rough): "),
            (eta_color, f"{qinfo.get('eta_label', 'now')}\n"),
        ]

    def _preview_panel(self) -> Window:
        return Window(
            FormattedTextControl(self._render_preview_text),
            style="bg:#1a1a2e",
            dont_extend_height=True,
            height=D(max=10),
        )

    def _tokenize_bash_line(self, line: str) -> list[tuple[str, str]]:
        if not line.strip():
            return [("", "  \n")]

        if line.strip().startswith("#"):
            if line.strip().startswith("#SBATCH"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    return [
                        ("fg:#00ff80 bold", f"  {parts[0]}="),
                        ("fg:#ffffff", f"{parts[1]}\n"),
                    ]
                else:
                    return [("fg:#00ff80 bold", f"  {line}\n")]
            else:
                return [("fg:#555555 italic", f"  {line}\n")]

        tokens = []
        words = line.split(" ")
        for idx, word in enumerate(words):
            space = " " if idx < len(words) - 1 else ""
            if word in ("source", "conda", "activate", "mamba", "module", "load"):
                tokens.append(("fg:#00ffff bold", word + space))
            elif word.startswith("$") or "$(" in word:
                tokens.append(("fg:#ff0080", word + space))
            else:
                tokens.append(("", word + space))

        return [("", "  ")] + tokens + [("", "\n")]

    def _render_preview_text(self) -> list[tuple[str, str]]:
        if self.idx < 1:
            return []
        if self.transient.get("preview_dirty"):
            self.transient.pop("preview_lines", None)
            self.transient["preview_dirty"] = False
        cached = self.transient.get("preview_lines")
        if cached is not None:
            from typing import cast
            return cast(list[tuple[str, str]], cached)
        ans: dict[str, Any] = {}
        for i in range(self.idx):
            step = STEPS[i]
            if step.key in self.answers:
                ans[step.key] = self.answers[step.key]
        if not ans:
            return []
        script = build_from_answers(ans, partial=True)
        if not script.strip():
            return []
        lines: list[tuple[str, str]] = [("class:preview-header", "\n  \u2500\u2500 Script preview (so far) \u2500\u2500\n\n")]
        for line in script.split("\n"):
            lines.extend(self._tokenize_bash_line(line))
        self.transient["preview_lines"] = lines
        return lines

    def _footer(self) -> Window:
        return Window(
            FormattedTextControl(self._render_footer),
            height=1,
            style="bg:#1a1a2e",
        )

    def _render_footer(self) -> list[tuple[str, str]]:
        left = "  Enter/Tab:Next  Esc:Back"
        if self._is_select_active():
            left += "  \u2191\u2193:Move"
        mouse = "Mouse:on (F2 to select text)" if self.mouse_enabled else "Select/copy text freely (F2=mouse nav)"
        left += f"  {mouse}"
        left += "  F1:Help  ^C:Quit"
        return [("class:info", left)]

    # ── Entry point ─────────────────────────────────────────────────

    def run(self) -> dict[str, Any] | None:
        self._on_enter_step()
        try:
            self.app.run()
            return self.answers
        except (KeyboardInterrupt, EOFError):
            return None
