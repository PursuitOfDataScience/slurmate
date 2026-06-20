from __future__ import annotations

import os
import re
import sys
import time
from typing import Any


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

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        if not _should_use_color():
            return ""
        return object.__getattribute__(self, name)


c = C()

BANNER_LINES = [
    r"    ███████╗██╗     ██╗   ██╗██████╗ ███╗   ███╗██╗███████╗██╗   ██╗",
    r"    ██╔════╝██║     ██║   ██║██╔══██╗████╗ ████║██║██╔════╝╚██╗ ██╔╝",
    r"    ███████╗██║     ██║   ██║██████╔╝██╔████╔██║██║█████╗   ╚████╔╝ ",
    r"    ╚════██║██║     ██║   ██║██╔══██╗██║╚██╔╝██║██║██╔══╝    ╚██╔╝  ",
    r"    ███████║███████╗╚██████╔╝██║  ██║██║ ╚═╝ ██║██║██║        ██║   ",
    r"    ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝        ╚═╝   ",
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


def print_banner(animate: bool | str | None = False) -> None:
    """Print banner, respecting NO_COLOR and SLURMIFY_NO_BANNER env vars.

    Args:
        animate: If True, show animation. Default is False (instant display).
                 Can be overridden with SLURMIFY_BANNER_ANIMATE=1.
    """
    if os.environ.get("SLURMIFY_NO_BANNER"):
        return

    use_color = _should_use_color()
    use_animation = animate or os.environ.get("SLURMIFY_BANNER_ANIMATE") == "1"

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
            subtitle = f"{c.CYAN}Slurmify{c.RESET}  {c.GRAY}\u2014  interactive sbatch wizard{c.RESET}"
        else:
            subtitle = "Slurmify  \u2014  interactive sbatch wizard"
        print(f"  {subtitle}")
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
    subtitle = f"{c.CYAN}Slurmify{c.RESET}  {c.GRAY}\u2014  interactive sbatch wizard{c.RESET}"
    print(f"  {subtitle}")
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
