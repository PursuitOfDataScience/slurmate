from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable

from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, FuzzyWordCompleter
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
from prompt_toolkit.widgets import RadioList, TextArea

from .builder import build_from_answers, build_sbatch_script, estimate_su
from .system_utils import (
    fetch_available_modules,
    fetch_conda_envs,
    fetch_gpu_types_for_partition,
    fetch_known_qos,
    fetch_partitions,
    fetch_public_partitions,
    fetch_qos_for_partition,
    fetch_queue_eta,
    fetch_user_accounts,
    normalize_memory,
    validate_memory,
)
from .theme import c


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
    ("info", "fg:#888888"),
    ("completion-menu", "bg:#222222 fg:#cccccc"),
    ("completion-menu.completion", "bg:#222222 fg:#cccccc"),
    ("completion-menu.completion.current", "bg:#0088ff fg:#ffffff bold"),
])


CUSTOM = "Enter partition name manually..."
PRIVATE = "Include private partitions"


def _validate_time(val: str) -> bool:
    if not val.strip():
        return True
    if re.match(r"^\d+-\d{2}:\d{2}:\d{2}$", val.strip()):
        return True
    if re.match(r"^\d{2}:\d{2}:\d{2}$", val.strip()):
        return True
    return False


def _get_partition(partitions: list[dict], name: str) -> dict[str, Any]:
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
    fetch: Callable | None = None

    def __init__(self, key: str, title: str, kind: str, **kw: Any) -> None:
        self.key = key
        self.title = title
        self.kind = kind
        for k, v in kw.items():
            setattr(self, k, v)


STEPS: list[Step] = [
    Step("job_name", "Job name", "text", subtitle="A name for your Slurm job"),
    Step("account", "Account", "autocomplete",
         subtitle="Slurm account to charge (optional)",
         fetch=fetch_user_accounts),
    Step("partition", "Partition", "partition",
         fetch=lambda: (fetch_public_partitions(), fetch_partitions())),
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
         validate=_validate_time, default="02:00:00",
         choices=TIME_CHOICES),
    Step("nodes", "Nodes", "text", subtitle="Number of nodes", default="1",
         validate=lambda v: v.strip().isdigit() and int(v) > 0),
    Step("gpus", "GPUs", "select", subtitle="Number of GPUs",
         choices=["0", "1", "2", "4", "8"], default="0"),
    Step("gpu_type", "GPU type", "gpu_type", subtitle="GPU hardware type"),
    Step("array_spec", "Array spec", "text",
         subtitle="e.g. 1-10, 1,3,5-7%4 (optional)"),
    Step("modules", "Modules", "autocomplete",
         subtitle="Comma-separated, e.g. python/anaconda,cuda (optional)",
         fetch=fetch_available_modules),
    Step("env_name", "Conda environment", "select",
         subtitle="Python/Conda environment",
         choices=["None (skip)"], default="None (skip)",
         fetch=fetch_conda_envs),
    Step("command", "Command to run", "text",
         subtitle="e.g. python train.py --epochs 100"),
    Step("custom_sbatch", "Custom #SBATCH flags", "autocomplete",
         subtitle="e.g. --exclusive --reservation=abc (optional)",
         choices=SBATCH_FLAGS),
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
        # Text input widget (shared across text/autocomplete/partition-text/gpu-text steps)
        self.text_area = TextArea(
            multiline=False,
            style="bg:#333333 fg:#ffffff",
            scrollbar=False,
        )

        # RadioList (shared across select/partition/gpu_type steps)
        self.radio_list = RadioList([("", "")])

        self._build_app()

    def _invalidate(self) -> None:
        if self.app and self.app.is_running:
            self.app.layout = self._build_layout()
            if self._is_text_active():
                self.app.layout.focus(self.text_area)
            elif self._is_select_active():
                self.app.layout.focus(self.radio_list)
            self.app.invalidate()

    @property
    def current_step(self) -> Step: 
        return STEPS[self.idx] if self.idx < len(STEPS) else STEPS[-1]

    def _step_count(self) -> int:
        return len(STEPS)

    def _is_review(self) -> bool:
        return self.idx >= len(STEPS)

    def _is_text_active(self) -> bool:
        if self._is_review():
            return False
        s = self.current_step
        if s.kind in ("text", "autocomplete"):
            return True
        if s.kind == "partition":
            return self.step_cache.get("partition_sub") == "text"
        if s.kind == "gpu_type":
            return self.step_cache.get("gpu_sub") == "text"
        return False

    def _is_select_active(self) -> bool:
        if self._is_review():
            return False
        s = self.current_step
        if s.kind == "select":
            return True
        if s.kind == "partition":
            sub = self.step_cache.get("partition_sub", "select")
            return sub in ("select", "all")
        if s.kind == "gpu_type":
            return self.step_cache.get("gpu_sub", "select") == "select"
        return False

    def _can_go_back(self) -> bool:
        if self._is_review():
            return True
        s = self.current_step
        if s.kind == "partition":
            sub = self.step_cache.get("partition_sub", "select")
            if sub in ("all", "text"):
                return True
        return self.idx > 0

    def _can_go_forward(self) -> bool:
        return not self._is_review()

    def _has_error(self) -> bool:
        return bool(self.step_cache.get("error"))

    # ── Key bindings ────────────────────────────────────────────────

    def _build_app(self) -> None:
        kb = KeyBindings()

        @kb.add("tab", eager=True, filter=Condition(lambda: self._can_go_forward()))
        def _tab(event: Any) -> None:
            self._confirm_and_next()

        @kb.add("enter", eager=True, filter=Condition(lambda: not self._is_review()))
        def _enter(event: Any) -> None:
            self._confirm_and_next()

        @kb.add("s-tab", eager=True, filter=Condition(self._can_go_back))
        def _stab(event: Any) -> None:
            self._go_back()

        @kb.add("escape", eager=True, filter=Condition(lambda: not self._is_review() and self.idx > 0))
        def _esc(event: Any) -> None:
            self._go_back()

        @kb.add("escape", eager=True, filter=Condition(self._is_review))
        def _esc_review(event: Any) -> None:
            self.idx = len(STEPS) - 1
            self._on_enter_step()
            self._invalidate()

        @kb.add("c-c")
        def _cc(event: Any) -> None:
            raise KeyboardInterrupt

        @kb.add("c-s", filter=Condition(self._is_review))
        @kb.add("enter", eager=True, filter=Condition(self._is_review))
        def _review_exit(event: Any) -> None:
            event.app.exit()

        @kb.add("e", filter=Condition(self._is_review))
        def _e(event: Any) -> None:
            self._edit_script()

        self.app = Application(
            layout=self._build_layout(),
            key_bindings=kb,
            full_screen=True,
            style=_TUI_STYLE,
            mouse_support=True,
        )

    # ── Navigation ──────────────────────────────────────────────────

    def _text_val(self) -> str:
        return self.text_area.text.strip()

    def _confirm_and_next(self) -> None:
        if self._is_review():
            return
        s = self.current_step

        if s.kind == "partition":
            self._handle_partition_confirm()
            return
        if s.kind == "gpu_type":
            self._handle_gpu_type_confirm()
            return

        if s.kind == "select":
            val = self.radio_list.current_value
            if val:
                self.answers[s.key] = self._coerce(val, s)
                self._advance()
            return

        if s.kind in ("text", "autocomplete"):
            val = self._text_val()
            if s.key in ("job_name", "command") and not val:
                self.step_cache["error"] = f"Required: {s.subtitle or ''}"
                self._invalidate()
                return
            if s.validate and not s.validate(val):
                self.step_cache["error"] = f"Invalid input: {s.subtitle or ''}"
                self._invalidate()
                return
            self.step_cache.pop("error", None)
            self.answers[s.key] = self._coerce(val, s)
            self._advance()
            return

    def _advance(self) -> None:
        self.idx += 1
        self.transient["preview_dirty"] = True
        self._on_enter_step()
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
        self._on_enter_step()
        self._invalidate()

    def _coerce(self, val: str, s: Step) -> Any:
        if s.key in ("cpus", "gpus"):
            return int(val) if val else 4
        if s.key == "nodes":
            return int(val) if val else 1
        if s.key == "memory":
            return normalize_memory(val) if val else "16G"
        if s.key == "modules":
            return [m.strip() for m in val.split(",") if m.strip()] if val else None
        if s.key in ("account", "array_spec", "gpu_type"):
            return val or None
        return val

    def _on_enter_step(self) -> None:
        """Called whenever entering a step (forward or backward)."""
        self.step_cache.pop("error", None)
        if self._is_review():
            self._build_preview()
            return
        s = self.current_step
        prev = self.answers.get(s.key)

        if s.kind == "text":
            self.text_area.text = str(prev or s.default or "")
            self._set_completer(None)
        elif s.kind == "autocomplete":
            self.text_area.text = str(prev or s.default or "")
            self._setup_autocomplete(s)
        elif s.kind == "select":
            self._setup_select(s, prev)
        elif s.kind == "partition":
            self._setup_partition()
        elif s.kind == "gpu_type":
            self._setup_gpu_type()

    # ── Autocomplete ────────────────────────────────────────────────

    def _set_completer(self, completer: Completer | None) -> None:
        self.text_area.buffer.completer = completer
        self.text_area.buffer.complete_while_typing = completer is not None

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
        default = str(prev or s.default or "")
        if default and default in choices:
            self.radio_list.current_value = default

    def _resolve_choices(self, s: Step) -> list[str]:
        key = f"choices_{s.key}"
        if key in self.step_cache:
            return self.step_cache[key]  # type: ignore[return-value]
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
            raw = self.radio_list.current_value
            if raw == CUSTOM:
                self.text_area.text = self.answers.get("partition", "")
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
            if val:
                all_parts = self.transient.get("all_parts", [])
                self.answers["partition"] = val
                self.answers["_partition_obj"] = _get_partition(all_parts, val)
                self._advance()
            else:
                self.step_cache["error"] = "Partition name is required"
                self._invalidate()
            return
        if sub == "all":
            raw = self.radio_list.current_value
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

    def _setup_gpu_type(self) -> None:
        gpus = self.answers.get("gpus", 0)
        if isinstance(gpus, str):
            gpus = int(gpus) if gpus.isdigit() else 0
        if gpus == 0:
            self.answers["gpu_type"] = None
            self._advance()
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
            self.radio_list.current_value = prev if prev and prev in choices else "Any"
            self.step_cache["gpu_sub"] = "select"
        else:
            self.text_area.text = self.answers.get("gpu_type", "")
            self.step_cache["gpu_sub"] = "text"
            self._set_completer(None)

    def _handle_gpu_type_confirm(self) -> None:
        sub = self.step_cache.get("gpu_sub", "select")
        if sub == "select":
            val = self.radio_list.current_value
            self.answers["gpu_type"] = None if val == "Any" else val
        elif sub == "text":
            val = self._text_val()
            self.answers["gpu_type"] = val or None
        self._advance()

    # ── Preview / Submit ────────────────────────────────────────────

    def _build_preview(self) -> None:
        self.answers.pop("_partition_obj", None)
        script = build_from_answers(self.answers)
        su_estimate = estimate_su(
            self.answers.get("cpus", 1),
            self.answers.get("time_limit", "02:00:00"),
            self.answers.get("nodes", 1),
        )
        try:
            queue_info = fetch_queue_eta(
                self.answers.get("partition", ""),
                req_nodes=self.answers.get("nodes", 1),
            )
        except Exception as e:
            logger.debug(f"Failed to fetch queue info: {e}")
            queue_info = {}
        self.transient["script"] = script
        self.transient["su_estimate"] = su_estimate
        self.transient["queue_info"] = queue_info
        self.transient["preview_built"] = True

    def _edit_script(self) -> None:
        script = self.transient.get("script", "")
        if not script:
            return
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script)
            tmp_path = f.name
        try:
            subprocess.run([editor, tmp_path], check=False)
            with open(tmp_path) as f:
                edited = f.read()
            self.transient["script"] = edited
            self._invalidate()
        finally:
            os.unlink(tmp_path)

    # ── Layout ──────────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        return Layout(
            FloatContainer(
                HSplit([
                    self._header(),
                    VSplit([self._sidebar(), self._content()], padding=1),
                    self._footer(),
                ]),
                floats=[
                    Float(xcursor=True, ycursor=True, content=CompletionsMenu()),
                ],
            ),
            focused_element=self.text_area,
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
        right = ""
        if self._is_review():
            right = "  [ Review ]"
        else:
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
            elif i == self.idx and not self._is_review():
                lines.append(("class:sidebar-current", f"  \u25b6 {s.title}\n"))
            else:
                lines.append(("class:sidebar-pending", f"    {s.title}\n"))
        return lines

    def _content(self) -> HSplit:
        s = self.current_step

        title_text = f"\n  {s.title}\n" if not self._is_review() else "\n"
        subtitle_text = f"  {s.subtitle}\n\n" if not self._is_review() else ""

        error_control: list[Window] = []
        if self.step_cache.get("error"):
            error_control = [Window(
                FormattedTextControl([("class:error", f"  \u2717 {self.step_cache['error']}\n")]),
                height=1, style="",
            )]

        title_win = Window(
            FormattedTextControl([
                ("class:title", title_text),
                ("class:subtitle", subtitle_text),
            ]),
            height=2 + (1 if subtitle_text else 0),
            dont_extend_height=True,
        )

        if self._is_review():
            return self._build_review_content()

        text_active = self._is_text_active()
        select_active = self._is_select_active()

        children: list[Any] = []
        if text_active:
            children = [self.text_area]
        elif select_active:
            children = [self.radio_list]

        return HSplit([title_win] + error_control + children + [self._preview_panel()])

    def _preview_panel(self) -> Window:
        return Window(
            FormattedTextControl(self._render_preview_text),
            style="bg:#1a1a2e",
            dont_extend_height=True,
            height=D(max=10),
        )

    def _render_preview_text(self) -> list[tuple[str, str]]:
        if self._is_review():
            return []
        if self.idx < 1:
            return []
        if self.transient.get("preview_dirty"):
            self.transient.pop("preview_lines", None)
            self.transient["preview_dirty"] = False
        cached = self.transient.get("preview_lines")
        if cached is not None:
            return cached  # type: ignore[return-value]
        ans: dict[str, Any] = {}
        for i in range(self.idx):
            step = STEPS[i]
            if step.key in self.answers:
                ans[step.key] = self.answers[step.key]
        if not ans:
            return []
        script = build_from_answers(ans)
        if not script.strip():
            return []
        lines: list[tuple[str, str]] = [("class:preview-header", "\n  \u2500\u2500 Script preview \u2500\u2500\n\n")]
        for line in script.split("\n"):
            lines.append(("class:preview-text", f"  {line}\n"))
        self.transient["preview_lines"] = lines
        return lines

    def _build_review_content(self) -> HSplit:
        ans = self.answers
        script = self.transient.get("script", "")
        su_est = self.transient.get("su_estimate", "")
        queue_info = self.transient.get("queue_info", {})

        lines: list[tuple[str, str]] = [
            ("class:title", "\n  Review & Submit\n\n"),
            ("class:subtitle", "   Script\n\n"),
        ]
        for line in script.split("\n"):
            lines.append(("class:preview-text", f"  {line}\n"))
        lines.append(("", "\n"))
        lines.append(("class:subtitle", "   Summary\n\n"))
        items: list[tuple[str, str]] = [
            ("Job:", str(ans.get("job_name", ""))),
            ("Partition:", str(ans.get("partition", ""))),
        ]
        if ans.get("account"):
            items.append(("Account:", str(ans["account"])))
        if ans.get("qos"):
            items.append(("QoS:", str(ans["qos"])))
        items.append(("CPUs:", str(ans.get("cpus", ""))))
        items.append(("Memory:", str(ans.get("memory", ""))))
        items.append(("Time:", str(ans.get("time_limit", ""))))
        items.append(("Nodes:", str(ans.get("nodes", "1"))))
        gpus = ans.get("gpus", 0)
        if isinstance(gpus, str):
            gpus = int(gpus) if gpus.isdigit() else 0
        if gpus > 0:
            gt = ans.get("gpu_type") or "any"
            items.append(("GPUs:", f"{gpus} \u00d7 {gt}"))
        if ans.get("array_spec"):
            items.append(("Array:", str(ans["array_spec"])))
        if ans.get("modules"):
            items.append(("Modules:", ", ".join(ans["modules"])))
        if ans.get("env_name"):
            items.append(("Conda env:", str(ans["env_name"])))
        if ans.get("command"):
            items.append(("Command:", str(ans["command"])))
        if ans.get("custom_sbatch"):
            items.append(("Custom flags:", ", ".join(ans["custom_sbatch"])))
        items.append(("SU cost:", str(su_est)))
        if queue_info:
            items.append(("Queue:", f"{queue_info.get('running', '?')} run / {queue_info.get('pending', '?')} wait"))
            items.append(("ETA:", queue_info.get("eta_label", "?")))
        for label, value in items:
            lines.append(("class:preview-text", f"    {label:<14} {value}\n"))
        return HSplit([Window(FormattedTextControl(lines), scrollbar=True)])

    def _footer(self) -> Window:
        return Window(
            FormattedTextControl(self._render_footer),
            height=1,
            style="bg:#1a1a2e",
        )

    def _render_footer(self) -> list[tuple[str, str]]:
        if self._is_review():
            return [
                ("class:info",
                 "  [Enter: Finish]  [e: Edit]  [Esc: Back to wizard]  [^C: Cancel]"),
            ]
        left = "  Tab/Enter:Next  S-Tab:Back"
        if self._is_select_active():
            left += "  \u2191\u2193:Navigate  Enter:Confirm"
        if self.idx > 0:
            left += "  Esc:Prev"
        left += "  ^C:Quit"
        return [("class:info", left)]

    # ── Entry point ─────────────────────────────────────────────────

    def run(self) -> dict[str, Any] | None:
        self._on_enter_step()
        try:
            self.app.run()
            return self.answers
        except (KeyboardInterrupt, EOFError):
            return None
