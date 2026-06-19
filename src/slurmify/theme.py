from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.completion import Completer, Completion, FuzzyWordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.validation import ValidationError, Validator as PTValidator
from prompt_toolkit.widgets import Label, RadioList

_SEL_STYLE = PTStyle([
    ("prompt", "bold fg:#00ffff"),
    ("pointer", "bold fg:#bf00ff"),
])


def select_input(prompt: str, choices: list[str], default: str | None = None,
                 qmark: str = "") -> Optional[str]:
    """Arrow-key selectable list with ESC to go back."""
    if not choices:
        return None
    radio = RadioList([(c, c) for c in choices])
    if default and default in choices:
        radio.current_value = default

    sel_kb = KeyBindings()

    @sel_kb.add("enter", eager=True)
    def _enter(event):
        event.app.exit(result=radio.current_value)

    @sel_kb.add("escape", eager=True)
    def _esc(event):
        event.app.exit(result=None)

    app = Application(
        layout=Layout(
            HSplit([
                Window(Label(f"  {prompt}"), height=1, dont_extend_height=True),
                radio,
            ]),
            focused_element=radio,
        ),
        key_bindings=sel_kb,
        style=_SEL_STYLE,
    )
    try:
        return app.run()
    except (KeyboardInterrupt, EOFError):
        return None


class _PathCompleter(Completer):
    """Completes file/directory paths as the user types."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        prefix = text.rsplit(" ", 1)[-1] if " " in text else text
        if not prefix:
            return

        expanded = os.path.expanduser(prefix)
        dir_part = os.path.dirname(expanded) or "."
        partial = os.path.basename(expanded)

        try:
            entries = os.listdir(dir_part)
        except OSError:
            return

        for entry in sorted(entries):
            if not entry.startswith(partial) or entry.startswith("."):
                continue
            full = os.path.join(dir_part, entry)
            suffix = entry + "/" if os.path.isdir(full) else entry
            if partial:
                dir_prefix = prefix[: -len(partial)]
                completion_text = dir_prefix + suffix
            else:
                completion_text = prefix + suffix
            yield Completion(completion_text, start_position=-len(prefix))


_KB = KeyBindings()


@_KB.add("escape", eager=True)
def _exit_on_esc(event):
    event.app.exit(result=None)


def _make_session(choices: list[str] | None = None, validate: Callable[[str], bool] | None = None,
                  path_complete: bool = False) -> PromptSession:
    kwargs = dict(key_bindings=_KB)
    if path_complete:
        kwargs["completer"] = _PathCompleter()
        kwargs["complete_while_typing"] = True
    elif choices:
        kwargs["completer"] = FuzzyWordCompleter(choices)
        kwargs["complete_while_typing"] = True
    if validate:

        class V(PTValidator):
            def validate(self, doc):
                if not validate(doc.text):
                    raise ValidationError(message="Invalid input", cursor_position=len(doc.text))
        kwargs["validator"] = V()
    return PromptSession(**kwargs)


def autocomplete(prompt: str, choices: list[str], default: str = "", qmark: str = "",
                 validate: Callable[[str], bool] | None = None) -> Optional[str]:
    """Like questionary.autocomplete but ESC returns None."""
    try:
        result = _make_session(choices, validate).prompt(f"{qmark}{prompt} ", default=default)
        return result
    except (KeyboardInterrupt, EOFError):
        return None


def text_input(prompt: str, default: str = "", qmark: str = "",
               validate: Callable[[str], bool] | None = None) -> Optional[str]:
    """Like questionary.text but ESC returns None."""
    try:
        result = _make_session(validate=validate).prompt(f"{qmark}{prompt} ", default=default)
        return result
    except (KeyboardInterrupt, EOFError):
        return None


def path_input(prompt: str, default: str = "", qmark: str = "",
               validate: Callable[[str], bool] | None = None) -> Optional[str]:
    """Text input with file/directory path autocomplete."""
    try:
        result = _make_session(validate=validate, path_complete=True).prompt(
            f"{qmark}{prompt} ", default=default
        )
        return result
    except (KeyboardInterrupt, EOFError):
        return None


class C:
    PINK = "\033[38;2;255;0;128m"
    CYAN = "\033[38;2;0;255;255m"
    MAGENTA = "\033[38;2;191;0;255m"
    YELLOW = "\033[38;2;255;255;0m"
    GREEN = "\033[38;2;0;255;128m"
    ORANGE = "\033[38;2;255;128;0m"
    RED = "\033[38;2;255;0;0m"
    BLUE = "\033[38;2;0;128;255m"
    PURPLE = "\033[38;2;128;0;255m"
    WHITE = "\033[38;2;255;255;255m"
    GRAY = "\033[38;2;128;128;128m"
    DARK_GRAY = "\033[38;2;64;64;64m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    G1 = "\033[38;2;255;0;128m"
    G2 = "\033[38;2;255;0;191m"
    G3 = "\033[38;2;191;0;255m"
    G4 = "\033[38;2;128;0;255m"
    G5 = "\033[38;2;0;128;255m"
    G6 = "\033[38;2;0;255;255m"


c = C()

BANNER_LINES = [
    r"    ███████╗██╗     ██╗   ██╗██████╗ ███╗   ███╗██╗███████╗██╗   ██╗",
    r"    ██╔════╝██║     ██║   ██║██╔══██╗████╗ ████║██║██╔════╝╚██╗ ██╔╝",
    r"    ███████╗██║     ██║   ██║██████╔╝██╔████╔██║██║█████╗   ╚████╔╝ ",
    r"    ╚════██║██║     ██║   ██║██╔══██╗██║╚██╔╝██║██║██╔══╝    ╚██╔╝  ",
    r"    ███████║███████╗╚██████╔╝██║  ██║██║ ╚═╝ ██║██║██║        ██║   ",
    r"    ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝        ╚═╝   ",
]

BANNER_GRADIENT = [c.G1, c.G2, c.G3, c.G4, c.G5, c.G6]


def _brighten(rgb, t):
    return (
        min(255, int(rgb[0] + (255 - rgb[0]) * t)),
        min(255, int(rgb[1] + (255 - rgb[1]) * t)),
        min(255, int(rgb[2] + (255 - rgb[2]) * t)),
    )


def _to_rgb(ansi_code):
    m = __import__("re").match(r"\033\[38;2;(\d+);(\d+);(\d+)m", ansi_code)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (255, 0, 128)


BASE_RGB = [_to_rgb(g) for g in BANNER_GRADIENT]


def print_banner(animate=True):
    print()
    for i, line in enumerate(BANNER_LINES):
        print(f"{BANNER_GRADIENT[i]}{c.BOLD}\033[3m{line}\033[23m{c.RESET}")
    print()

    if not animate or not sys.stdout.isatty():
        subtitle = f"{c.CYAN}Slurmify{c.RESET}  {c.GRAY}\u2014  interactive sbatch wizard{c.RESET}"
        print(f"  {subtitle}")
        print(f"  {c.GRAY}ESC to go back{c.RESET}")
        print()
        return

    n = len(BANNER_LINES)
    print(f"\033[{n + 1}A", end="")
    print("\033[s", end="")
    for _ in range(2):
        crest = -2.0
        while crest <= n + 1.0:
            print("\033[u", end="")
            for i, line in enumerate(BANNER_LINES):
                intensity = max(0.0, 1.0 - abs(i - crest) / 2.0) * 0.8
                r, g, b = _brighten(BASE_RGB[i], intensity)
                color = f"\033[38;2;{r};{g};{b}m"
                print(f"\033[2K{color}{c.BOLD}\033[3m{line}\033[23m{c.RESET}\n", end="")
            print("\033[2K\n", end="")
            time.sleep(0.04)
            crest += 0.5
    print("\033[u", end="")
    for i, line in enumerate(BANNER_LINES):
        print(f"\033[2K{BANNER_GRADIENT[i]}{c.BOLD}\033[3m{line}\033[23m{c.RESET}")
    print()
    subtitle = f"{c.CYAN}Slurmify{c.RESET}  {c.GRAY}\u2014  interactive sbatch wizard{c.RESET}"
    print(f"  {subtitle}")
    print(f"  {c.GRAY}ESC to go back{c.RESET}")
    print()


class Spinner:
    def __init__(self, msg="Loading"):
        self.msg = msg
        self.running = False
        self.thread = None

    def _spin(self):
        frames = ["\u25d0", "\u25d3", "\u25d1", "\u25d2"]
        i = 0
        while self.running:
            sys.stdout.write(
                f"\r  {c.CYAN}{frames[i % len(frames)]}{c.RESET} "
                f"{c.GRAY}{self.msg}...\033[K{c.RESET}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self, status="ok"):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        icon = {"ok": f"{c.GREEN}\u2713{c.RESET}", "err": f"{c.RED}\u2717{c.RESET}"}.get(status, "")
        sys.stdout.write(f"\r  {icon} {c.GRAY}{self.msg}\033[K{c.RESET}\n")
        sys.stdout.flush()


def tool_status(name, status="running"):
    if status == "running":
        print(f"\r  {c.CYAN}\u26a1{c.RESET} {c.GRAY}{name}...\033[K{c.RESET}", end="", flush=True)
    elif status == "success":
        print(f"\r  {c.GREEN}\u2713{c.RESET} {c.GRAY}{name}\033[K{c.RESET}")
    elif status == "error":
        print(f"\r  {c.RED}\u2717 {name}\033[K{c.RESET}")


def ok(msg):
    print(f"  {c.GREEN}\u2713{c.RESET} {c.GRAY}{msg}{c.RESET}")


def fail(msg):
    print(f"  {c.RED}\u2717 {msg}{c.RESET}")


def info(msg):
    print(f"  {c.CYAN}\u25b8{c.RESET} {c.GRAY}{msg}{c.RESET}")


def header(title):
    print(f"\n  {c.BOLD}{c.PINK}\u276f {title}{c.RESET}")


def questionary_style():
    import questionary
    return questionary.Style([
        ("qmark", "fg:#00ffff bold"),
        ("question", "fg:#00ffff bold"),
        ("answer", "fg:#00ff80"),
        ("pointer", "fg:#bf00ff bold"),
        ("highlighted", "fg:#bf00ff bold"),
        ("selected", "fg:#00ff80"),
        ("separator", "fg:#808080"),
        ("instruction", "fg:#808080"),
        ("text", "fg:#ffffff"),
        ("disabled", "fg:#808080 italic"),
    ])
