from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Generator
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
from prompt_toolkit.widgets import RadioList, TextArea

from .builder import build_from_answers, job_summary_rows
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

# Setup debug logging if SLURMATE_DEBUG is set
logger = logging.getLogger(__name__)
if os.environ.get("SLURMATE_DEBUG"):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


# ── Palette ──────────────────────────────────────────────────────────────
# A muted, harmonious dark palette (Tokyo-Night lineage). Every tone is drawn
# just off the pure-channel corners and from one family, so the UI reads as
# *designed* rather than as saturated "neon toy" primaries. One accent
# (periwinkle) carries focus / the current step; green / amber / rosé are
# reserved strictly for validation state, not decoration. Two background tones
# (base stage + raised surface) give panels a subtle card-like depth.
_BASE = "#16161e"      # app stage — the deepest background
_SURFACE = "#1f2335"   # panels / inputs — one step up, reads as a raised card
_SURFACE2 = "#2a2f45"  # focused input / selected row
_OVERLAY = "#3b4261"   # dividers, faint rules, scrollbar thumb
_TEXT = "#c0caf5"       # primary text (soft lavender-white, not pure #fff)
_MUTED = "#8189ad"      # secondary text
_FAINT = "#565f89"      # tertiary — pending steps, comments
_ACCENT = "#7aa2f7"     # periwinkle — focus, headers, current step
_ACCENT2 = "#bb9af7"    # soft violet — list pointer / secondary accent
_GREEN = "#9ece6a"      # done / success (sage, not neon)
_AMBER = "#e0af68"      # warning (gold, not pure orange)
_ROSE = "#f7768e"       # error (rosé, not fire-truck red)
_CYAN = "#7dcfff"       # info — string / variable literals

_BG_BASE = f"bg:{_BASE}"
_BG_SURFACE = f"bg:{_SURFACE}"


_TUI_STYLE = PTStyle([
    ("status-bar", f"bg:{_SURFACE} fg:{_MUTED}"),
    ("status-bar.brand", f"bg:{_SURFACE} fg:{_ACCENT} bold"),
    ("status-bar.meter", f"bg:{_SURFACE} fg:{_ACCENT}"),
    ("status-bar.meter-empty", f"bg:{_SURFACE} fg:{_OVERLAY}"),
    ("sidebar-done", f"fg:{_GREEN}"),
    ("sidebar-current", f"fg:{_ACCENT} bold"),
    ("sidebar-pending", f"fg:{_FAINT}"),
    ("sidebar-gutter", f"fg:{_ACCENT}"),
    ("title", f"fg:{_TEXT} bold"),
    ("subtitle", f"fg:{_MUTED}"),
    ("text-area", f"fg:{_TEXT} {_BG_SURFACE}"),
    # Distinct focused look (a raised selection background) so the active input
    # field is obvious — previously identical to unfocused.
    ("text-area focused", f"fg:{_TEXT} bg:{_SURFACE2} bold"),
    ("radio-list", f"fg:{_TEXT}"),
    ("radio-list.selected", f"fg:{_ACCENT} bold"),
    ("radio-list.pointer", f"fg:{_ACCENT2} bold"),
    ("checkbox", f"fg:{_MUTED}"),
    ("checkbox.selected", f"fg:{_GREEN}"),
    ("preview-header", f"fg:{_ACCENT} bold"),
    ("preview-text", f"fg:{_MUTED}"),
    ("error", f"fg:{_ROSE} bold"),
    ("warning", f"fg:{_AMBER} bold"),
    ("info", f"fg:{_FAINT}"),
    ("rule", f"fg:{_OVERLAY}"),
    ("completion-menu", f"bg:{_SURFACE} fg:{_MUTED}"),
    ("completion-menu.completion", f"bg:{_SURFACE} fg:{_MUTED}"),
    ("completion-menu.completion.current", f"bg:{_ACCENT} fg:{_BASE} bold"),
    ("scrollbar.background", f"bg:{_SURFACE}"),
    ("scrollbar.button", f"bg:{_OVERLAY}"),
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

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Generator[Completion, None, None]:
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


class LastTokenCommaCompleter(Completer):
    """Fuzzy-word completion for the last comma-separated token.

    Lets users type ``python/anaconda,cuda`` and get completions for ``cuda``
    based on the word list, instead of trying to fuzzy-match the whole buffer
    (which would never match a single word).
    """

    def __init__(self, words: list[str]) -> None:
        self._words = words

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Generator[Completion, None, None]:
        text = document.text_before_cursor
        idx = text.rfind(",")
        prefix = text[idx + 1:] if idx >= 0 else text
        stripped = prefix.lstrip()
        if not stripped:
            return
        leading = len(prefix) - len(stripped)
        fuzzy = FuzzyWordCompleter(self._words, WORD=False)
        sub = Document(stripped, len(stripped))
        for comp in fuzzy.get_completions(sub, complete_event):
            yield Completion(
                comp.text,
                start_position=-len(prefix) + leading,
                display=comp.display,
                display_meta=comp.display_meta,
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
    """Parse custom #SBATCH flags from free-form input into one flag per entry.

    Options are separated by spaces or commas, so ``--exclusive --reservation=abc``
    and ``--exclusive,--reservation=abc`` both yield two directives. Only a comma
    that introduces another flag (one followed by ``-``) separates options — a
    comma inside a value is kept, so ``--exclude=node1,node2`` stays a single
    directive. Give an option its value with ``=``; a bare word is taken as a
    standalone option (``exclusive`` -> ``--exclusive``), *not* glued onto the
    previous flag, since the wizard can't know which options take a value. A
    leading ``#SBATCH`` (pasted by mistake) is stripped.
    """
    # Turn only "flag-separating" commas (those before a dash) into spaces;
    # commas inside a value (e.g. a node list) survive.
    raw = re.sub(r",(\s*-)", r" \1", raw)
    flags: list[str] = []
    for tok in raw.split():
        if tok.startswith("#SBATCH"):
            tok = tok[len("#SBATCH"):]
        tok = tok.strip().rstrip(",")
        if not tok:
            continue
        if not tok.startswith("-"):
            tok = f"--{tok}"
        flags.append(tok)
    return flags


MEMORY_CHOICES = ["4G", "8G", "16G", "32G", "64G", "128G", "256G", "512G", "64000M"]
TIME_CHOICES = ["01:00:00", "02:00:00", "04:00:00", "08:00:00", "12:00:00",
                "24:00:00", "48:00:00", "7-00:00:00"]
SBATCH_FLAGS = [
    "--exclusive", "--exclude=", "--nodelist=", "--reservation=",
    "--ntasks=", "--ntasks-per-node=", "--threads-per-core=",
    "--mem-per-cpu=", "--constraint=", "--licenses=", "--gres=",
    "--tmp=", "--hint=", "--signal=",
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
    Step("partition", "Partition", "partition"),
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
         subtitle="e.g. 30 (min), 5:00 (mm:ss), hh:mm:ss, d-hh:mm:ss, d-hh",
         validate=validate_time, default="02:00:00",
         choices=TIME_CHOICES),
    Step("nodes", "Nodes", "text", subtitle="Number of nodes", default="1",
         validate=lambda v: v.strip().isdigit() and int(v) > 0),
    Step("ntasks_per_node", "Tasks per node", "ntasks_per_node",
         subtitle="Tasks per node (optional, for multi-node)", default="1",
         validate=lambda v: not v.strip() or (v.strip().isdigit() and int(v) > 0)),
    Step("gpus", "GPUs", "autocomplete",
         subtitle="Number of GPUs — type any number (suggestions: 0, 1, 2, 4, 8)",
         choices=["0", "1", "2", "4", "8"], default="0",
         validate=lambda v: v.strip().isdigit()),
    Step("gpu_type", "GPU type", "gpu_type", subtitle="GPU hardware type"),
    Step("gpu_format", "GPU format", "gpu_format", subtitle="Format style for GPU requests"),
    Step("array_spec", "Array spec", "text",
         subtitle="e.g. 1-10, 1,3,5-7%4 (optional)"),
    Step("output_dir", "Output directory", "text",
         subtitle="Directory for stdout/stderr logs (optional)", default="logs", path=True),
    Step("output_file", "Output file", "text",
         subtitle="Log name: %j = job ID, %A/%a = array job/task (optional; blank = auto). Bare name gets .out; .err derived", path=True),
    Step("custom_sbatch", "Custom #SBATCH flags", "autocomplete",
         subtitle="e.g. --exclusive --reservation=abc  (space- or comma-separated, optional)",
         choices=SBATCH_FLAGS),
    Step("modules", "Modules", "autocomplete",
         subtitle="Enter a name, press Enter to add (comma auto-inserted); Tab to advance when done",
         fetch=fetch_available_modules),
    Step("env_type", "Environment type", "select",
         subtitle="Environment activation strategy",
         choices=["None (skip)", "Conda", "Mamba", "Virtualenv (venv)"],
         default="None (skip)"),
    Step("env_name", "Environment name/path", "autocomplete",
         subtitle="Conda environment name or virtualenv path",
         default=""),
    Step("command", "Command to run", "text",
         subtitle="e.g. python train.py  (Enter=next, Ctrl+J=newline, Tab=complete)",
         required=True, multiline=True, path=True),
    Step("review", "Review & Submit", "review",
         subtitle="Review your job configuration before submitting"),
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
        # Use the class style (not a literal) so the "text-area focused" pseudo
        # state in _TUI_STYLE actually applies and the focused field stands out.
        self.text_area = TextArea(
            multiline=False,
            style="class:text-area",
            scrollbar=False,
        )

        self.multiline_text_area = TextArea(
            multiline=True,
            style="class:text-area",
            scrollbar=True,
        )

        # RadioList (shared across select/partition/gpu_type steps)
        self.radio_list = RadioList([("", "")])

        # Mouse capture stays OFF so the terminal can natively select/copy the
        # script preview. Navigation is fully keyboard-driven.
        self._path_completer = LastTokenPathCompleter()
        self._skipped_indices: set[int] = set()
        # The review step shows the job config (left) and the final script
        # (right) side by side. The script is taller than the screen, so it must
        # scroll. FormattedTextControl windows don't scroll on their own and
        # their cursor fights ``get_vertical_scroll``; instead we slice the
        # visible lines ourselves by ``_review_scroll`` (driven by the
        # up/down/page key bindings).
        self._review_scroll = 0
        self._review_total_lines = 0
        self._review_config_window = Window(
            FormattedTextControl(self._render_review_config),
            width=D(weight=2), wrap_lines=True, style=_BG_SURFACE,
        )
        self._review_script_window = Window(
            FormattedTextControl(self._render_review_script, focusable=True),
            width=D(weight=3), style=_BG_SURFACE,
        )

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

        @kb.add("tab", eager=True, filter=Condition(lambda: True))
        def _tab(event: Any) -> None:
            buf = self._focused_buffer()
            s = self.current_step
            if buf is not None:
                buf.complete_next()
                if buf.complete_state is not None:
                    return
                # On multiline steps Tab only tries to complete (never advances);
                # Enter proceeds, Ctrl+J inserts a newline.
                if getattr(s, "multiline", False):
                    return
            self._confirm_and_next()

        @kb.add("enter", eager=True)
        def _enter(event: Any) -> None:
            # Enter always proceeds — consistent across every step, including the
            # multiline command step. (Use Ctrl+J for a literal newline there.)
            s = self.current_step
            buf = self._focused_buffer()
            # On non-multiline steps, Enter first applies an active completion
            # rather than advancing; multiline accepts completions with Tab.
            if not getattr(s, "multiline", False) and buf is not None \
                    and buf.complete_state is not None \
                    and buf.complete_state.current_completion is not None:
                buf.apply_completion(buf.complete_state.current_completion)
                if s.key == "modules":
                    buf.insert_text(", ")
                return
            self._confirm_and_next()

        @kb.add("s-tab", eager=True, filter=Condition(lambda: self._can_go_back()))
        def _stab(event: Any) -> None:
            self._go_back()

        @kb.add("escape", eager=True, filter=Condition(lambda: self.idx > 0))
        def _esc(event: Any) -> None:
            self._go_back()

        review_active = Condition(lambda: self.current_step.kind == "review")

        @kb.add("up", eager=True, filter=review_active)
        def _review_up(event: Any) -> None:
            self._review_scroll = max(0, self._review_scroll - 1)
            self.app.invalidate()

        @kb.add("down", eager=True, filter=review_active)
        def _review_down(event: Any) -> None:
            self._review_scroll = min(self._review_max_scroll(), self._review_scroll + 1)
            self.app.invalidate()

        @kb.add("pageup", eager=True, filter=review_active)
        def _review_pgup(event: Any) -> None:
            self._review_scroll = max(0, self._review_scroll - 10)
            self.app.invalidate()

        @kb.add("pagedown", eager=True, filter=review_active)
        def _review_pgdn(event: Any) -> None:
            self._review_scroll = min(self._review_max_scroll(), self._review_scroll + 10)
            self.app.invalidate()

        @kb.add("c-c")
        def _cc(event: Any) -> None:
            raise KeyboardInterrupt

        @kb.add("c-j", eager=True,
                filter=Condition(lambda: getattr(self.current_step, "multiline", False)))
        def _newline(event: Any) -> None:
            # Ctrl+J inserts a literal newline on the multiline command step.
            # (Shift+Enter can't be distinguished from Enter by the terminal.)
            buf = self._focused_buffer()
            if buf is not None:
                buf.insert_text("\n")

        self.app = Application(
            layout=self._build_layout(),
            key_bindings=kb,
            full_screen=True,
            style=_TUI_STYLE,
            mouse_support=False,
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

        if s.kind == "review":
            self._advance()
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
            if sub in ("all", "text"):
                # Return to the initial partition chooser. Rebuild it via
                # _setup_partition() (which resets partition_sub to "select" AND
                # restores the correct select radio); merely flipping the sub back
                # left the private/text radio on screen, so the next confirm
                # resolved to the wrong partition.
                self._setup_partition()
                self._invalidate()
                return
        # NB: gpu_type has no select<-text transition to unwind (unlike partition):
        # the text sub is entered only when the partition lists no typed GPUs, and
        # no radio is built for it. So gpu_type Back must fall through to the
        # general logic below (decrement idx, return to the gpus step) — handling
        # it like partition here trapped the user and confirmed a stale radio value.
        # Whether the step we're leaving was auto-skipped. Its value isn't the
        # user's, and the shared text widget may still hold another step's text —
        # capture this before the pruning below drops the index.
        was_skipped = self.idx in self._skipped_indices
        self._skipped_indices = {i for i in self._skipped_indices if i < self.idx - 1}
        self.step_cache.pop("partition_sub", None)
        self.step_cache.pop("gpu_sub", None)
        self.step_cache.pop("error", None)
        # Save current text so it's preserved when returning to this step — but
        # never for a skipped step, whose shared widget holds a different step's
        # leftover text (e.g. the modules string would otherwise be saved into a
        # skipped env_name when navigating back through it).
        if not was_skipped:
            if s.kind in ("text", "autocomplete", "ntasks_per_node"):
                val = self._text_val()
                if val:
                    self.answers[s.key] = self._coerce(val, s)
            elif s.kind in ("select", "gpu_format"):
                val = self._radio_value()
                if val:
                    self.answers[s.key] = self._coerce(val, s)
        # The live preview is cached; going backward changes which steps feed it,
        # so mark it dirty (forward navigation already does this in _advance).
        self.transient["preview_dirty"] = True
        self.idx = max(0, self.idx - 1)
        self._on_enter_step("backward")
        self._invalidate()

    def _default_int(self, key: str, literal: int) -> int:
        """Config-aware integer default — falls back to the configured value (if
        any) when a field is cleared, not the bare hard-coded literal."""
        raw = self._config_defaults.get(key)
        if raw is None:
            return literal
        try:
            return int(raw)
        except (TypeError, ValueError):
            return literal

    def _coerce(self, val: str, s: Step) -> Any:
        if s.key == "job_name":
            from .builder import sanitize_job_name
            return sanitize_job_name(val)
        if s.key == "cpus":
            return int(val) if val else self._default_int("cpus", 4)
        if s.key == "gpus":
            return int(val) if val.strip().isdigit() else 0
        if s.key == "nodes":
            return int(val) if val else self._default_int("nodes", 1)
        if s.key == "ntasks_per_node":
            return int(val) if val else None
        if s.key == "memory":
            if val:
                return normalize_memory(val)
            return normalize_memory(self._config_defaults.get("memory", "")) or "16G"
        if s.key == "modules":
            return [m.strip() for m in val.split(",") if m.strip()] if val else None
        if s.key == "custom_sbatch":
            # Parse into a list of flags; the builder iterates this, so a raw
            # string would be split character-by-character (#SBATCH m, i, d, …).
            return _parse_custom_flags(val) if val else None
        if s.key == "qos":
            return None if (not val or val == "Default (none)") else val
        if s.key in ("account", "array_spec", "gpu_type", "gpu_format",
                     "output_dir", "output_file"):
            return val or None
        return val

    # Steps whose in-progress value affects the partition-compatibility check.
    # For the active one of these, the field's live (not-yet-committed) value is
    # overlaid before validating, so the current field gives feedback as it's
    # edited — while every *other* field's already-committed value is still
    # checked, so a problem introduced earlier (e.g. GPUs on a CPU-only
    # partition) keeps showing after you move past that step.
    _VALIDATED_KEYS = frozenset(
        {"cpus", "memory", "time_limit", "nodes", "ntasks_per_node", "gpus", "gpu_type"}
    )

    def _config_warnings(self) -> list[tuple[str, str]]:
        """(level, message) issues for the whole work-in-progress config.

        Delegates to the shared, side-effect-free ``validate_job_config`` (the
        same check the final CLI summary runs), so the live preview flags a
        script that's already in a failure mode even while it's unfinished —
        rather than only warning about the step you happen to be on. Safe to
        call on every redraw: no subprocess calls, and the partition's GPU-type
        list is reused from the cache populated when the GPU-type step loaded.
        """
        if not self.answers.get("_partition_obj"):
            return []
        live = dict(self.answers)
        s = self.current_step
        if s.key in self._VALIDATED_KEYS:
            if self._is_text_active():
                live[s.key] = self._text_val()
            elif self._is_select_active():
                live[s.key] = self._radio_value()
        from .system_utils import validate_job_config
        return validate_job_config(live, extra_gpu_types=self.transient.get("gpu_types"))

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
                if isinstance(prev, list):
                    self.text_area.text = ", ".join(prev)
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
        elif s.kind == "review":
            self._setup_review(direction)

    # ── Autocomplete ────────────────────────────────────────────────

    def _set_completer(self, completer: Completer | None) -> None:
        self.text_area.buffer.completer = completer  # type: ignore[assignment]
        self.text_area.buffer.complete_while_typing = Condition(lambda: completer is not None)

    def _set_multiline_completer(self, completer: Completer | None) -> None:
        self.multiline_text_area.buffer.completer = completer  # type: ignore[assignment]
        self.multiline_text_area.buffer.complete_while_typing = Condition(lambda: completer is not None)

    def _setup_autocomplete(self, s: Step) -> None:
        choices = self._resolve_choices(s)
        if not choices:
            self._set_completer(None)
            return
        if s.key == "modules":
            self._set_completer(LastTokenCommaCompleter(choices))
        else:
            self._set_completer(FuzzyWordCompleter(choices))

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
        # QoS is partition-specific (AllowQos differs per partition), so key the
        # cache on the partition too; otherwise changing partition after the QoS
        # step was visited would keep serving the first partition's QoS list.
        if s.key == "qos":
            key = f"choices_qos_{self.answers.get('partition', '')}"
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
        # Cache the partition query for the session: re-entering the step (or
        # navigating back to it) reuses the result instead of re-running sinfo /
        # scontrol. fetch_partitions() is fetched once and shared with
        # fetch_public_partitions() so sinfo isn't run two or three times.
        cached = self.transient.get("all_parts")
        if cached is not None:
            public, all_parts = self.transient.get("public_parts", []), cached
        else:
            try:
                all_parts = fetch_partitions()
                public = fetch_public_partitions(all_parts)
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
        # Resolve the picked row by matching its formatted label against the real
        # partition objects, not by fragile index arithmetic. The old index math
        # assumed the CUSTOM/PRIVATE header rows were always present, so it
        # resolved to the wrong partition after "back" from the private list, and
        # to the raw formatted label (a broken --partition=) on an all-restricted
        # cluster where no partition is public.
        public = self.transient.get("public_parts", [])
        all_parts = self.transient.get("all_parts", [])
        for part in list(public) + list(all_parts):
            if _fmt_partition(part) == raw:
                self.answers["partition"] = part["name"]
                self.answers["_partition_obj"] = part
                return
        self.answers["partition"] = raw
        self.answers["_partition_obj"] = _get_partition(all_parts, raw)

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
            self._skipped_indices.add(self.idx)
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
            self._skipped_indices.add(self.idx)
            if direction == "forward":
                self._advance()
            else:
                self._go_back()
            return

        choices = ["gres_type", "constraint", "gpus"]
        self.radio_list = RadioList([(c, c) for c in choices])
        prev = self.answers.get("gpu_format")
        # Seed from SLURMATE_GPU_FORMAT (documented default) when there's no
        # prior answer, so the env var actually influences the wizard default.
        env_default = os.environ.get("SLURMATE_GPU_FORMAT", "gres_type")
        if env_default not in choices:
            env_default = "gres_type"
        self._set_radio_default(prev if prev and prev in choices else env_default)

    def _setup_ntasks_per_node(self, direction: str = "forward") -> None:
        nodes = self.answers.get("nodes", 1)
        if isinstance(nodes, str):
            nodes = int(nodes) if nodes.isdigit() else 1
        if nodes <= 1:
            self.answers["ntasks_per_node"] = None
            self._skipped_indices.add(self.idx)
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
            self._skipped_indices.add(self.idx)
            if direction == "forward":
                self._advance()
            else:
                self._go_back()
            return

        if env_type in ("Conda", "Mamba"):
            try:
                # Load the modules the user picked first, so envs from a
                # module-provided conda (e.g. `module load anaconda`) show up.
                envs = fetch_conda_envs(self.answers.get("modules"))
            except Exception as e:
                logger.debug(f"Failed to fetch conda envs: {e}")
                envs = []
            self.step_cache["choices_env_name"] = envs
            self.text_area.text = self.answers.get("env_name") or ""
            self._setup_autocomplete(self.current_step)
            if envs:
                # Pop the dropdown so the discovered envs are visible up front,
                # not only once the user starts typing.
                self._open_completion_menu()
        else:  # Virtualenv (venv) — complete filesystem paths
            self.text_area.text = self.answers.get("env_name") or ""
            self._set_completer(self._path_completer)

    def _open_completion_menu(self) -> None:
        """Open the completion dropdown so choices show without typing."""
        try:
            buf = self.text_area.buffer
            if buf.completer is not None and not buf.complete_state:
                buf.start_completion(select_first=False)
        except Exception as e:
            logger.debug(f"Could not open completion menu: {e}")

    def _handle_gpu_type_confirm(self) -> None:
        sub = self.step_cache.get("gpu_sub", "select")
        if sub == "select":
            val = self._radio_value()
            self.answers["gpu_type"] = None if val == "Any" else val
        elif sub == "text":
            val = self._text_val()
            self.answers["gpu_type"] = val or None
        self._advance()

    def _setup_review(self, direction: str = "forward") -> None:
        self.transient["preview_dirty"] = True
        self._review_scroll = 0

    def _review_max_scroll(self) -> int:
        """Largest scroll offset that still keeps the last script line on screen."""
        info = self._review_script_window.render_info
        # The script window's first row is a fixed "── Final Script ──" header
        # that doesn't scroll, so the body viewport is one row shorter.
        visible = (info.window_height - 1) if info else 0
        return max(0, self._review_total_lines - max(1, visible))

    # ── Layout ──────────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        floats = [Float(xcursor=True, ycursor=True, content=CompletionsMenu())]

        focused: Any = self.text_area
        if self._is_text_active():
            s = self.current_step
            if getattr(s, "multiline", False):
                focused = self.multiline_text_area
            else:
                focused = self.text_area
        elif self._is_select_active():
            focused = self.radio_list
        elif self.current_step.kind == "review":
            focused = self._review_script_window

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
        # Reuse the status-bar class rather than re-declaring its colors inline,
        # so the two definitions can't drift apart.
        return VSplit([
            Window(
                FormattedTextControl(self._render_header_left),
                height=1, style="class:status-bar", dont_extend_height=True,
            ),
            Window(
                FormattedTextControl(self._render_header_right),
                height=1, style="class:status-bar", align=WindowAlign.RIGHT,
            ),
        ])

    def _render_header_left(self) -> list[tuple[str, str]]:
        # Brand mark in the accent; the descriptor stays muted so the header
        # reads as calm chrome, not a loud banner bar.
        return [
            ("class:status-bar.brand", "  \u26a1 Slurmate"),
            ("class:status-bar", "  \u2014  sbatch wizard"),
        ]

    _METER_SEGMENTS = 14

    def _render_header_right(self) -> list[tuple[str, str]]:
        s = self.current_step
        visible_total = len(STEPS) - len(self._skipped_indices)
        visible_done = sum(1 for i in range(self.idx) if i not in self._skipped_indices)
        pos = visible_done + 1
        # A segmented progress meter (\u25b0\u25b1) reads at a glance far better than a bare
        # "3/21"; keep the count and the step title alongside it.
        n = self._METER_SEGMENTS
        filled = max(0, min(n, round(pos / max(1, visible_total) * n)))
        filled_bar = "\u25b0" * filled
        empty_bar = "\u25b1" * (n - filled)
        return [
            ("class:status-bar.meter", f"  {filled_bar}"),
            ("class:status-bar.meter-empty", empty_bar),
            ("class:status-bar", f"  {pos}/{visible_total}  "),
            ("class:status-bar.brand", f"{s.title}  "),
        ]

    _SIDEBAR_WIDTH = 26

    def _sidebar(self) -> Window:
        return Window(
            FormattedTextControl(self._render_sidebar),
            width=self._SIDEBAR_WIDTH,
            style=_BG_SURFACE,
        )

    def _render_sidebar(self) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = [("class:subtitle", "  Steps\n\n")]
        # A fixed 4-column prefix keeps every title aligned: done "  \u2713 ", current
        # "  \u258e " (an accent gutter bar), pending "    ". Ellipsize any title that
        # would overflow the fixed width (e.g. "Environment name/path").
        avail = self._SIDEBAR_WIDTH - 4
        for i, s in enumerate(STEPS):
            if i in self._skipped_indices:
                continue
            title = s.title if len(s.title) <= avail else s.title[: avail - 1] + "\u2026"
            if i < self.idx:
                lines.append(("class:sidebar-done", f"  \u2713 {title}\n"))
            elif i == self.idx:
                # A left accent gutter (like an editor's active-line marker) reads
                # as more deliberate than a chevron.
                lines.append(("class:sidebar-gutter", "  \u258e "))
                lines.append(("class:sidebar-current", f"{title}\n"))
            else:
                lines.append(("class:sidebar-pending", f"    {title}\n"))
        return lines

    def _content(self) -> HSplit:
        s = self.current_step

        title_text = f"\n  {s.title}\n"
        subtitle_text = f"  {s.subtitle}\n\n"

        # The central column sits on the base stage; raised surface panels
        # (inputs, preview, review columns) float on top of it for depth.
        content_bg = _BG_BASE

        error_control: list[Window] = []
        if self.step_cache.get("error"):
            error_control.append(Window(
                FormattedTextControl([("class:error", f"  \u2717 {self.step_cache['error']}\n")]),
                height=1, style=content_bg,
            ))
        # Persistent whole-config validation: every issue in the work-in-progress
        # script stays visible on every step, not just the one that introduced it.
        # Errors (a config Slurm will reject \u2014 e.g. GPUs on a CPU-only partition)
        # render red; capacity warnings render orange.
        for level, msg in self._config_warnings():
            cls = "class:error" if level == "error" else "class:warning"
            icon = "\u2717" if level == "error" else "\u26a0"
            error_control.append(Window(
                FormattedTextControl([(cls, f"  {icon} {msg}\n")]),
                height=1, style=content_bg,
            ))

        title_win = Window(
            FormattedTextControl([
                ("class:title", title_text),
                ("class:subtitle", subtitle_text),
            ]),
            height=2 + (1 if subtitle_text else 0),
            dont_extend_height=True,
            style=content_bg,
        )

        text_active = self._is_text_active()
        select_active = self._is_select_active()

        children: list[Any] = []
        if s.kind == "review":
            return HSplit([
                title_win,
                VSplit([
                    self._review_config_window,
                    Window(width=1, char="│", style=f"class:rule {_BG_BASE}"),
                    self._review_script_window,
                ], padding=1, style=content_bg),
            ], style=content_bg)
        if text_active:
            if getattr(s, "multiline", False):
                children = [self.multiline_text_area]
            else:
                children = [self.text_area]
        elif select_active:
            children = [self.radio_list]

        return HSplit(
            [title_win] + error_control + children + [self._queue_panel(), self._preview_panel()],
            style=content_bg,
        )

    def _past_hardware_config(self) -> bool:
        """True once every resource/hardware step is done (modules onward).

        The queue ETA is surfaced here as a heads-up — how long the job will
        wait before modules load and the script runs — rather than during the
        hardware steps, where it would keep shifting as choices change.
        """
        modules_idx = next(
            (i for i, s in enumerate(STEPS) if s.key == "modules"), len(STEPS)
        )
        return self.idx >= modules_idx

    def _queue_panel(self) -> Window:
        show = bool(self.transient.get("queue_info")) and self._past_hardware_config()
        return Window(
            FormattedTextControl(self._render_queue_text),
            style=_BG_BASE,
            dont_extend_height=True,
            height=2 if show else 0,
        )

    def _render_queue_text(self) -> list[tuple[str, str]]:
        qinfo = self.transient.get("queue_info")
        if not qinfo or not self._past_hardware_config():
            return []
        part = self.transient.get("queue_info_part", "")
        eta_sec = qinfo.get("eta_seconds", 0)
        eta_color = f"fg:{_GREEN} bold" if eta_sec < 3600 else f"fg:{_AMBER} bold"
        return [
            ("", "\n  "),
            ("class:preview-header", f"Queue status ({part}): "),
            ("class:info", f"{qinfo.get('running', 0)} running / {qinfo.get('pending', 0)} pending   "),
            ("class:preview-header", "ETA: "),
            (eta_color, f"{qinfo.get('eta_label', 'now')}\n"),
        ]

    def _preview_panel(self) -> Window:
        return Window(
            FormattedTextControl(self._render_preview_text),
            style=_BG_SURFACE,
            height=D(min=8),
        )

    def _review_summary_items(self) -> list[tuple[str, str]]:
        # Shared with the CLI summary panel (job_summary_rows) so both surfaces
        # show the same fields \u2014 including Modules, Custom flags, GPU format, and
        # Tasks/node, which the Review step previously omitted.
        return job_summary_rows(self.answers)

    def _render_review_config(self) -> list[tuple[str, str]]:
        """Left column of the review step \u2014 the job configuration summary."""
        out: list[tuple[str, str]] = [
            ("class:preview-header", " \u2500\u2500 Job Configuration \u2500\u2500\n\n")
        ]
        label_w = 12
        # Continuation lines of a multi-line value (e.g. a multi-command script)
        # line up under the value column instead of starting at column 0.
        indent = " " + " " * label_w + " "
        for label, val in self._review_summary_items():
            if not val:
                continue
            parts = str(val).split("\n")
            out.append(("", f" {label:<{label_w}} "))
            out.append(("class:preview-text", f"{parts[0]}\n"))
            for cont in parts[1:]:
                out.append(("class:preview-text", f"{indent}{cont}\n"))
        return out

    def _build_script_lines(self) -> list[list[tuple[str, str]]]:
        """Final script as a list of lines (each a fragment list, no trailing \\n)."""
        lines: list[list[tuple[str, str]]] = []
        script = build_from_answers(self.answers)
        for line in script.split("\n"):
            frags = self._tokenize_bash_line(line)
            # _tokenize_bash_line appends a trailing "\n"; drop it for the line model.
            if frags:
                style, text = frags[-1]
                text = text[:-1] if text.endswith("\n") else text
                frags = frags[:-1] + ([(style, text)] if text else [])
            lines.append(frags or [("", "")])
        return lines

    def _render_review_script(self) -> list[tuple[str, str]]:
        """Right column \u2014 the final script, manually scrolled by ``_review_scroll``."""
        lines = self._build_script_lines()
        self._review_total_lines = len(lines)
        out: list[tuple[str, str]] = [
            ("class:preview-header", " \u2500\u2500 Final Script \u2500\u2500\n")
        ]
        for frags in lines[self._review_scroll:]:
            out.extend(frags)
            out.append(("", "\n"))
        return out

    def _tokenize_bash_line(self, line: str) -> list[tuple[str, str]]:
        if not line.strip():
            return [("", "  \n")]

        if line.strip().startswith("#"):
            if line.strip().startswith("#SBATCH"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    # Directive name in the accent, value in primary text — two
                    # tones instead of the old green-on-white shout.
                    return [
                        (f"fg:{_ACCENT}", f"  {parts[0]}="),
                        (f"fg:{_TEXT}", f"{parts[1]}\n"),
                    ]
                else:
                    return [(f"fg:{_ACCENT}", f"  {line}\n")]
            else:
                return [(f"fg:{_FAINT} italic", f"  {line}\n")]

        tokens = []
        words = line.split(" ")
        for idx, word in enumerate(words):
            space = " " if idx < len(words) - 1 else ""
            if word in ("source", "conda", "activate", "mamba", "module", "load"):
                tokens.append((f"fg:{_ACCENT}", word + space))
            elif word.startswith("$") or "$(" in word:
                tokens.append((f"fg:{_CYAN}", word + space))
            else:
                tokens.append((f"fg:{_TEXT}", word + space))

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
            style=_BG_SURFACE,
        )

    def _render_footer(self) -> list[tuple[str, str]]:
        s = self.current_step
        if s.key == "review":
            left = "  ↑↓/PgUp/PgDn:Scroll  Tab/Enter:Submit  Esc:Back  ^C:Quit"
        else:
            if getattr(s, "multiline", False):
                left = "  Enter:Next  Ctrl+J:newline  Tab:complete"
            else:
                left = "  Tab/Enter:Next"
            if self._is_select_active():
                left += "  \u2191\u2193:Move"
            left += "  Esc:Back  ^C:Quit"
        return [("class:info", left)]

    # ── Entry point ─────────────────────────────────────────────────

    def run(self) -> dict[str, Any] | None:
        self._on_enter_step()
        try:
            self.app.run()
            return self.answers
        except (KeyboardInterrupt, EOFError):
            return None

    def edit(self) -> dict[str, Any]:
        """Re-enter the wizard (keeping prior answers) at the review step.

        Used to jump back from the post-build action menu to fix a field after
        seeing the generated script. Returns the (in-place updated) answers; a
        Ctrl-C while editing just returns the current answers unchanged.
        """
        self.idx = len(STEPS) - 1
        self.submitted = False
        self._build_app()  # the previous Application already exited; make a fresh one
        self.run()
        return self.answers
