from __future__ import annotations

import codecs
import os
import re
import shutil
import sys
import time
from typing import Any


def _env_flag(name: str) -> bool:
    """True when an env var is set to an affirmative value (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# ── Output encoding safety ───────────────────────────────────────────────────
# A non-UTF-8 *but valid* locale (el7 has no C.UTF-8; en_US is latin-1) makes any
# non-encodable character in a message raise UnicodeEncodeError mid-print, which
# aborts the run and truncates the output. rich picks a safe box set for its own
# glyphs, but it does not transcode application text, so slurmate's own "⚠"/"✗"
# went straight to the encoder. The markers below therefore have ASCII fallbacks
# — and :func:`make_output_safe` covers everything a table cannot, including
# user-supplied data (a CJK job name, a partition name) that no fallback table
# could anticipate.

_FORCE_ASCII = False


def set_ascii(enabled: bool) -> None:
    """Force (or unforce) ASCII markers — backs ``--ascii``."""
    global _FORCE_ASCII
    _FORCE_ASCII = enabled


def output_encoding() -> str:
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None)
        if encoding:
            return str(encoding)
    return "utf-8"


def _encodable(text: str) -> bool:
    try:
        text.encode(output_encoding(), errors="strict")
    except (UnicodeEncodeError, LookupError, TypeError):
        return False
    return True


def use_ascii() -> bool:
    """Whether to render markers as ASCII: forced, requested, or unencodable."""
    if _FORCE_ASCII or _env_flag("SLURMATE_ASCII"):
        return True
    return not _encodable("".join(_Glyphs.UNICODE.values()))


class _Glyphs:
    """Status markers, resolved per call so ``--ascii`` applies after import."""

    UNICODE = {
        "OK": "\u2713", "ERR": "\u2717", "WARN": "\u26a0",
        "BULLET": "\u25b8", "ARROW": "\u2192", "ELLIPSIS": "\u2026",
        "BOLT": "\u26a1", "UP": "\u2191", "DOWN": "\u2193",
    }
    ASCII = {
        "OK": "+", "ERR": "x", "WARN": "!",
        "BULLET": ">", "ARROW": "->", "ELLIPSIS": "...",
        "BOLT": "*", "UP": "^", "DOWN": "v",
    }

    def __getattr__(self, name: str) -> str:
        table = self.ASCII if use_ascii() else self.UNICODE
        try:
            return table[name]
        except KeyError:
            raise AttributeError(name) from None


g = _Glyphs()


# Typography slurmate writes into its own prose, and the ASCII that carries the
# same meaning. Applied by the codec error handler below, so it covers every
# output path at once — including the 238 em dashes, which are far too many to
# route through the marker table individually.
_TRANSLITERATE = {
    "\u2014": "-", "\u2013": "-", "\u2026": "...", "\u00a0": " ",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2192": "->", "\u2022": "*", "\u00d7": "x",
    "\u2713": "+", "\u2717": "x", "\u26a0": "!", "\u25b8": ">",
    "\u26a1": "*", "\u2191": "^", "\u2193": "v",
}

_ERROR_HANDLER = "slurmate.transliterate"


def _encode_fallback(exc: UnicodeError) -> tuple[str, int]:
    """Transliterate what we can; escape what we cannot. Never raise."""
    if not isinstance(exc, UnicodeEncodeError):
        raise exc
    out: list[str] = []
    for ch in exc.object[exc.start:exc.end]:
        replacement = _TRANSLITERATE.get(ch)
        if replacement is None:
            # Unknown character — escape rather than drop it. This is the case
            # the table cannot cover: a job name, module or partition carrying
            # characters the terminal cannot encode is *data*, not decoration,
            # and "?" would silently destroy it.
            replacement = ch.encode("ascii", "backslashreplace").decode("ascii")
        out.append(replacement)
    return "".join(out), exc.end


def make_output_safe() -> None:
    """Stop a non-encodable character from aborting the run.

    A *valid* non-UTF-8 locale (``en_US`` is latin-1; el7 has no ``C.UTF-8``)
    made any such character raise ``UnicodeEncodeError`` mid-print, which killed
    the run and truncated the output. Registering a codec error handler fixes
    every output path at once, which routing individual strings through the
    marker table cannot: the failures were on *warning and error* paths, so the
    tool was least robust exactly when something had already gone wrong.
    """
    codecs.register_error(_ERROR_HANDLER, _encode_fallback)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors=_ERROR_HANDLER)
        except (ValueError, OSError, LookupError):  # detached / not reconfigurable
            pass


def _should_use_color() -> bool:
    """Check if we should use color output based on environment.

    ``FORCE_COLOR`` is honoured because ``rich`` honours it: without it, piping
    slurmate's output with ``FORCE_COLOR=1`` produced half-coloured output — rich's
    panels kept their colour while every ``c.*``-prefixed status line lost it.

    The test matches rich's exactly, down to the edges: any **non-empty** value
    forces colour (including ``FORCE_COLOR=0``, which rich also treats as "on"),
    while ``FORCE_COLOR=""`` counts as unset — checked against the installed rich
    rather than assumed. ``NO_COLOR`` still wins. ``CLICOLOR_FORCE`` is
    deliberately *not* honoured: rich ignores it, so acting on it here would
    recreate the very mismatch this removes.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
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
    # region, which only means anything on a real terminal — into a pipe or a log
    # file it would just emit escape soup. Colour alone is no longer a proxy for
    # that (FORCE_COLOR can enable colour on a non-TTY), so check isatty directly.
    if use_animation and not sys.stdout.isatty():
        use_animation = False

    # On a terminal too short to hold the banner region, the save/restore garbles
    # the screen. Fall back to the static banner when there isn't enough room.
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
