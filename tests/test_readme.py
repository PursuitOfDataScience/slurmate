"""The README, checked against the real parser.

slurmate was the only one of the five sibling packages with no test touching its
README at all — and it shows seven invocations, using the short forms `-J`, `-p`,
`-c`, `-G`, `-t` and `--mem`. Any of those could be renamed, or an example could
pick up a flag that never existed, and nothing would notice: the suite is large
but it exercises the parser through argument lists written in the tests, never
through the ones written in the docs.

The sibling that already does this says why, and it is the same reason here:
"Documentation that nothing verifies is documentation that will drift."

Deliberately a *parse* check rather than a run: these examples name a `gpu`
partition, an `h100`, and `./run.sh`, none of which exist everywhere, so running
them would fail for reasons that say nothing about the README. What a README can
promise is the shape of the command.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import re
import shlex

import pytest

from slurmate.main import parse_args

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def _invocations() -> list[tuple[str, list[str]]]:
    """Every `slurmate ...` command the README shows, as an argv list.

    Handles the three things a README does to a command line: wraps it with a
    trailing backslash, annotates it with a trailing `# comment`, and shows shell
    plumbing (`> file`, `| cat`) that is not an argument.
    """
    text = re.sub(r"\\\n\s*", " ", README.read_text())
    found = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("$ "):
            line = line[2:].strip()
        if not line.startswith("slurmate ") and line != "slurmate":
            continue
        # Shell plumbing is documentation of usage, not of arguments.
        line = re.split(r"\s(?:>|>>|\||&&|2>)", line)[0]
        try:
            parts = shlex.split(line, comments=True)
        except ValueError:
            continue
        # Prose that opens with the tool's name.
        if len(parts) > 1 and parts[1] in {
            "says",
            "reads",
            "is",
            "and",
            "or",
            "the",
            "then",
            "also",
            "will",
            "can",
        }:
            continue
        found.append((line, parts[1:]))
    return found


def test_the_readme_shows_some_commands():
    """A silent extraction failure would make every test below vacuously pass."""
    found = _invocations()
    assert len(found) >= 5, f"only found {found}"


@pytest.mark.parametrize("shown,argv", _invocations(), ids=lambda v: None)
def test_every_shown_command_parses(shown, argv):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            parse_args(argv)
    except SystemExit:
        message = err.getvalue().strip().splitlines()
        pytest.fail(
            f"README shows `{shown}`, which the parser rejects:\n  "
            + (message[-1] if message else "(no message)")
        )


def _parser_flags() -> set[str]:
    """Every long flag the parser accepts, read from its own `--help`.

    `parse_args` builds its parser internally, so `--help` is the only public way
    to ask -- which is also the right way: it is what a reader of the README would
    check against.
    """
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
        parse_args(["--help"])
    return set(re.findall(r"(--[a-z][a-z0-9-]+)", out.getvalue()))


def test_the_help_lists_flags():
    # Guards the extraction above, not slurmate.
    assert len(_parser_flags()) >= 20


def test_every_flag_the_readme_names_exists():
    """Including flags named in prose, not only inside a full command line.

    An example that parses is not the whole promise: the README also names flags
    in sentences ("pass `--force` to generate it anyway"), and a reader takes those
    as instructions just as readily.
    """
    named = set(re.findall(r"`(--[a-z][a-z0-9-]+)`", README.read_text()))
    assert named, "no flags found in the README; check the extraction"
    #: Flags belonging to OTHER tools the README legitimately quotes -- `sbatch`'s
    #: own options appear when explaining what slurmate emits.
    foreign = {
        "--export",
        "--wrap",
        "--test-only",
        "--parsable",
        "--cpus-per-task",
        "--ntasks-per-node",
        "--mem-per-cpu",
        "--gres",
        "--job-name",
        "--partition-name",
        "--no-requeue",
    }
    missing = sorted(named - _parser_flags() - foreign)
    assert not missing, (
        f"the README names {missing}, which `slurmate --help` does not list -- "
        f"either the flag was renamed or the README invented it"
    )
