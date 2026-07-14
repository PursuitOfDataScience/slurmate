from __future__ import annotations

import os
import re
import shutil
import sys
import time
from typing import Any


def _env_flag(name: str) -> bool:
    """True when an env var is set to an affirmative value (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _should_use_color() -> bool:
    """Check if we should use color output based on environment."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


class C:
    PINK = "\033[38;2;255;0;128m"
    CYAN = "\033[38;2;0;255;255m"
    MAGENTA = "\033[38;2;191;0;255m"
    # Amber rather than pure yellow: readable on light backgrounds and unified
    # with the TUI's amber `warning` style (used for warnings, "Cancelled",
    # SU/array labels). Pure #ffff00 was nearly invisible on light terminals.
    YELLOW = "\033[38;2;255;170;0m"
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

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        # Decide once per instance and cache it — `__getattribute__` runs on
        # every color access (the banner animation hits it many times per
        # frame), and `_should_use_color()` does an isatty()/env probe each call.
        cache = object.__getattribute__(self, "__dict__")
        use_color = cache.get("_use_color")
        if use_color is None:
            use_color = _should_use_color()
            cache["_use_color"] = use_color
        if not use_color:
            return ""
        return object.__getattribute__(self, name)


c = C()

BANNER_LINES = [
    r"    ███████╗██╗     ██╗   ██╗██████╗ ███╗   ███╗ █████╗ ████████╗███████╗",
    r"    ██╔════╝██║     ██║   ██║██╔══██╗████╗ ████║██╔══██╗╚══██╔══╝██╔════╝",
    r"    ███████╗██║     ██║   ██║██████╔╝██╔████╔██║███████║   ██║   █████╗  ",
    r"    ╚════██║██║     ██║   ██║██╔══██╗██║╚██╔╝██║██╔══██║   ██║   ██╔══╝  ",
    r"    ███████║███████╗╚██████╔╝██║  ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗",
    r"    ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝",
]

# Use class-level access (not the `c` instance) so the gradient codes are not
# blanked by C.__getattribute__'s color gate when this module is imported under
# a non-TTY/NO_COLOR process. print_banner() decides at call time whether to emit
# them.
BANNER_GRADIENT = [C.G1, C.G2, C.G3, C.G4, C.G5, C.G6]


def _brighten(rgb: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        min(255, int(rgb[0] + (255 - rgb[0]) * t)),
        min(255, int(rgb[1] + (255 - rgb[1]) * t)),
        min(255, int(rgb[2] + (255 - rgb[2]) * t)),
    )


def _to_rgb(ansi_code: str) -> tuple[int, int, int]:
    m = re.match(r"\033\[38;2;(\d+);(\d+);(\d+)m", ansi_code)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (255, 0, 128)


BASE_RGB = [_to_rgb(g) for g in BANNER_GRADIENT]


def print_banner(animate: bool | str | None = False, interactive: bool = True) -> None:
    """Print banner, respecting NO_COLOR and SLURMATE_NO_BANNER env vars.

    Args:
        animate: If True, show animation. Default is False (instant display).
                 Can be overridden with SLURMATE_BANNER_ANIMATE=1.
        interactive: When False (batch/non-interactive mode), the "ESC to go
                 back" hint is suppressed \u2014 there's no wizard to go back in.
    """
    if _env_flag("SLURMATE_NO_BANNER"):
        return

    use_color = _should_use_color()
    use_animation = bool(animate) or _env_flag("SLURMATE_BANNER_ANIMATE")

    # The animation drives the cursor with absolute save/restore over the banner
    # region; on a terminal too short to hold it, that garbles the screen. Fall
    # back to the static banner when there isn't enough vertical room.
    if use_animation:
        try:
            rows = shutil.get_terminal_size().lines
        except OSError:
            rows = 24
        if rows < len(BANNER_LINES) + 4:
            use_animation = False

    print()
    if use_color:
        for i, line in enumerate(BANNER_LINES):
            print(f"{BANNER_GRADIENT[i]}{c.BOLD}\033[3m{line}\033[23m{c.RESET}")
    else:
        for line in BANNER_LINES:
            print(line)
    print()

    if not use_animation or not use_color:
        if use_color:
            subtitle = f"{c.CYAN}Slurmate{c.RESET}  {c.GRAY}\u2014  interactive sbatch wizard{c.RESET}"
        else:
            subtitle = "Slurmate  \u2014  interactive sbatch wizard"
        print(f"  {subtitle}")
        if interactive:
            print(f"  {c.GRAY if use_color else ''}ESC to go back{c.RESET if use_color else ''}")
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
    subtitle = f"{c.CYAN}Slurmate{c.RESET}  {c.GRAY}\u2014  interactive sbatch wizard{c.RESET}"
    print(f"  {subtitle}")
    if interactive:
        print(f"  {c.GRAY}ESC to go back{c.RESET}")
    print()


def questionary_style() -> Any:
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
