from __future__ import annotations

import logging
import os
import re
import shlex
from typing import Any

from .system_utils import (
    _parse_slurm_time_to_minutes,
    normalize_memory,
    time_request_is_unbounded,
)

logger = logging.getLogger(__name__)


def sanitize_job_name(name: str) -> str:
    """Make a job name safe as a single ``sbatch`` token.

    ``sbatch`` splits ``--job-name`` on whitespace, so ``my training job`` would
    silently become just ``my``. Collapse internal whitespace to underscores and
    drop characters outside a conservative safe set, so the emitted directive
    (and the auto-saved ``<job>-<id>.sh`` filename) are always well-formed.

    A truly empty input stays empty (the builder then omits the directive), but a
    non-empty name that sanitizes away entirely (e.g. an all-symbol or non-Latin
    name like ``###`` or ``训练任务``) falls back to ``slurm`` rather than emitting a
    malformed empty ``--job-name=``.

    A leading ``-``/``+``/``.`` is stripped: the sanitized name becomes the
    auto-saved ``<job>-<id>.sh`` filename and the ``%x`` log path, and a leading
    ``-`` makes those look like a CLI option (``tail -f -rf-1.out`` parses
    ``-rf`` as flags), while a leading ``.`` would hide the file. A name that is
    only such characters (``--``) also falls back to ``slurm``.
    """
    name = (name or "").strip()
    if not name:
        return name
    name = re.sub(r"\s+", "_", name)
    cleaned = re.sub(r"[^A-Za-z0-9._+-]", "", name)
    cleaned = cleaned.lstrip("-+.")
    return cleaned or "slurm"


def job_name_change_note(raw: str) -> str:
    """Say when sanitising turned the job name into something else, or "".

    Collapsing whitespace to underscores stays quiet: it is visible in the
    result and nothing is lost. Dropping characters is not, and the fallback is
    the case that really needs saying — *any* all-non-Latin name (``训练任务``)
    becomes ``slurm``, so a user's logs appear as ``logs/slurm-<jobid>.out`` and
    nothing anywhere explained why. The name is not only a directive; it is the
    log filename and the auto-saved script's filename, so a silent rewrite sends
    the user looking in the wrong place.

    Same disclosure the summary already makes for a ``--memory`` that
    ``--mem-per-cpu`` discarded, and for an ``--output-dir`` the job never writes
    to: the value was given, it did not survive, say so.
    """
    original = str(raw or "").strip()
    if not original:
        return ""
    safe = sanitize_job_name(original)
    if not safe or safe == re.sub(r"\s+", "_", original):
        return ""
    return (
        f"job name '{original}' was changed to '{safe}' — sbatch takes a single "
        f"token, and the name is also the log filename, so the output will be "
        f"'{safe}-<jobid>.out'"
    )


def _abort_guard(label: str) -> str:
    """`|| { … exit 1; }` tail that stops the job when a setup line fails.

    A batch script is not run with ``set -e``, so a failed ``module load`` or
    ``conda activate`` prints to stderr and the body runs anyway — Slurm then
    records the job **COMPLETED, exit 0** with the environment absent. The worst
    case is not a confusing failure later, it is a run that quietly proceeds
    against whatever toolchain was already on ``PATH`` and produces results the
    user believes came from the environment they asked for.

    Guarding the line also covers the case validation at generation time cannot:
    a module that exists when the script is written and is retired before the
    job runs.
    """
    # shlex.quote the whole message, not just the command's argument. The label
    # carries a user-supplied module or environment name, and a double-quoted
    # shell string still performs command substitution — so `--modules '$(cmd)'`
    # would have run `cmd` at the moment the guard fired. Single-quoting the
    # message makes it inert text.
    return f" || {{ echo {shlex.quote(f'slurmate: {label} failed; aborting')} >&2; exit 1; }}"


def _fold_directive(value: str) -> str:
    """Collapse CR/LF in a #SBATCH directive value to single spaces.

    Slurm stops parsing ``#SBATCH`` directives at the first non-comment line, so
    a newline smuggled into a value (via a CLI flag or an auto-loaded config)
    would (a) inject a bare command line into the script body and (b) silently
    drop every directive after it — the job then runs mis-sized. Fold any CR/LF
    to a space so the value always stays a single well-formed directive. The
    ``command`` body and ``custom_sbatch`` flags are handled separately (command
    is intentionally multi-line; custom flags fold their own newlines).

    Leading/trailing whitespace is stripped for the same reason, and it is the
    same bug: Slurm's directive parser splits on unquoted whitespace, so
    ``#SBATCH --array= 1-10`` is read as the directive ``--array=`` followed by a
    stray word, and sbatch refuses the whole script with *"Invalid directive
    found in batch script: 1-10"*. Measured on midway3 (Slurm 20.11.8) by
    generating with ``--print --force`` and running ``sbatch --test-only`` on the
    result: ``-a " 1-10 "`` and ``-t " 00:20:00 "`` each produced a script sbatch
    rejected outright (exit 255) while slurmate exited 0, because every consumer
    of the value already strips -- ``validate_array_spec`` calls the padded spec
    valid -- and only the emitter did not, so what was validated was not what was
    written. ``--constraint`` was fixed on its own (see
    ``TestConstraintWhitespace``: *"used to emit ``#SBATCH --constraint= a100 ``
    -- broken twice over"*); doing it here covers ``--array``, ``--time``,
    ``--qos``, ``--partition`` and ``--account`` too. Whitespace *inside* a value
    is untouched: it is meaningful (a log path with a space), and
    :func:`_quote_sbatch_value` quotes it.
    """
    return value.replace("\r", " ").replace("\n", " ").strip()


def _quote_sbatch_value(value: str) -> str:
    """Double-quote a #SBATCH value that contains whitespace or a quote mark.

    Slurm's directive parser splits on unquoted whitespace (so an output path
    like ``/scratch/My Group/log`` would bind only ``/scratch/My``). Slurm strips
    the surrounding quotes and preserves ``%j``/``%A``/``%a`` patterns literally,
    so quoting is safe; plain paths stay unquoted for readability.

    **Whitespace is not the only thing that has to be quoted.** That parser also
    does quote processing, and an *unmatched* ``'`` or ``"`` anywhere in a
    directive value is not a split -- it is fatal, and it takes the whole script
    down. Measured on midway3 (Slurm 20.11.8) on a script this function had
    emitted unquoted, because ``logs/it's-%j.out`` holds no whitespace::

        #SBATCH --output=logs/it's-%j.out
        $ sbatch --test-only script.sh
        sbatch: fatal: script.sh: line 9: Unmatched `'` in [ --output=logs/it's-%j.out]
        rc=1

    An apostrophe in a path is ordinary (``--output-dir /scratch/o'brien``), so
    slurmate was exiting 0 on a script sbatch refuses outright -- the same defect
    :func:`_fold_directive` documents for a padded ``--array``, where what was
    validated was not what was written. Wrapping fixes it: ``--output="logs/it's-%j.out"``
    and ``--output="logs/a\\"b-%j.out"`` both verify rc=0.

    A *balanced* pair of quotes does parse, but Slurm strips it, so an unquoted
    ``logs/a"b"c.out`` silently became ``logs/abc.out``; wrapping keeps the name
    the user asked for. Only a trailing backslash must not be left bare inside
    the quotes -- it would escape the closing one -- so it is doubled, which Slurm
    accepts.

    A newline in the value is folded to a space first (see :func:`_fold_directive`)
    so it can't split the directive across lines or inject a script-body line.
    """
    value = _fold_directive(value)
    if value and (
        any(ch.isspace() for ch in value) or '"' in value or "'" in value
    ):
        inner = value.replace('"', '\\"')
        # A run of backslashes at the very end would swallow the closing quote.
        trailing = len(inner) - len(inner.rstrip("\\"))
        if trailing:
            inner += "\\" * trailing
        return '"' + inner + '"'
    return value


def _quote_custom_flag(flag: str) -> str:
    """Quote the value of a ``--flag=value`` custom directive if it holds whitespace.

    A custom flag whose value contains a space (e.g. ``--comment=my job``, once
    the parser has consumed the user's quotes) would otherwise be split by
    Slurm's directive parser into ``--comment=my`` plus a stray ``job`` token,
    producing a script Slurm rejects. Wrap the value in double quotes — which
    Slurm strips — so it stays a single argument. A bare flag (``--exclusive``),
    a value with no whitespace, or a value the caller already quoted is emitted
    unchanged.

    The **space** form (``--comment my job``) is handled too, but only when the
    option name is a known value-taking one (``_VALUE_TAKING_FLAGS``) — that is
    what tells us where the value starts. Otherwise the flag is left alone, since
    guessing wrong would corrupt it. Without this, ``--comment "my job"`` typed in
    the custom-flags box became ``#SBATCH --comment my job``, which Slurm splits
    into ``--comment=my`` plus a stray ``job`` — the same defect the ``=`` form
    was already protected against.
    """
    if "=" in flag:
        name, value = flag.split("=", 1)
        sep = "="
    else:
        parts = flag.split(None, 1)
        if len(parts) != 2 or parts[0] not in _VALUE_TAKING_FLAGS:
            return flag
        name, value = parts
        sep = " "
    if not value or not any(ch.isspace() for ch in value):
        return flag
    # Already wrapped in matching quotes — don't double-quote it.
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return flag
    return f"{name}{sep}{_quote_sbatch_value(value)}"


# sbatch options that take a value, so a bare token following one of them is
# that value (``-C bigmem``, ``--reservation abc``) rather than a new option.
# Without this, every space-separated Slurm option was shredded into a valueless
# flag plus a nonsense ``--<value>`` one — ``-o /logs/x.out`` became
# ``['-o', '--/logs/x.out']``, a script sbatch rejects outright.
_VALUE_TAKING_FLAGS = frozenset({
    # short forms
    "-a", "-A", "-b", "-c", "-C", "-d", "-D", "-e", "-F", "-G", "-i", "-J", "-L",
    "-m", "-M", "-n", "-N", "-o", "-p", "-q", "-S", "-t", "-w", "-x",
    # long forms plausibly typed by hand (boolean options are deliberately absent:
    # --exclusive, --hold, --requeue, --spread-job, --contiguous, … keep the
    # "next bare word is its own option" behaviour)
    "--account", "--acctg-freq", "--array", "--batch", "--begin", "--chdir",
    "--cluster", "--clusters", "--comment", "--constraint", "--container",
    "--core-spec", "--cpu-freq", "--cpus-per-gpu", "--cpus-per-task", "--deadline",
    "--delay-boot", "--dependency", "--distribution", "--error", "--exclude",
    "--export", "--extra-node-info", "--gid", "--gpu-bind", "--gpu-freq", "--gpus",
    "--gpus-per-node", "--gpus-per-socket", "--gpus-per-task", "--gres",
    "--gres-flags", "--hint", "--input", "--job-name", "--licenses", "--mail-type",
    "--mail-user", "--mem", "--mem-bind", "--mem-per-cpu", "--mem-per-gpu",
    "--mincpus", "--network", "--nodefile", "--nodelist", "--nodes", "--ntasks",
    "--ntasks-per-core", "--ntasks-per-gpu", "--ntasks-per-node",
    "--ntasks-per-socket", "--open-mode", "--output", "--partition", "--prefer",
    "--priority", "--profile", "--qos", "--reservation", "--signal",
    "--sockets-per-node", "--switches", "--thread-spec", "--threads-per-core",
    "--time", "--time-min", "--tmp", "--tres-per-task", "--uid",
    "--wait-all-nodes", "--wckey", "--wrap",
})

# What a plausible option *name* looks like, for the "did the user forget the
# dashes?" fallback. A path, pattern or list (``/logs/%j.out``, ``n[01-04]``,
# ``2G``) can never be one, so such a token is treated as the preceding option's
# value even when that option isn't in the set above.
_OPTION_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*$")


def _join_flag_values(
    tokens: list[str], reassembled: list[tuple[str, str]] | None = None
) -> list[str]:
    """Turn a token list into flags, attaching space-separated values.

    One implementation shared by the free-form parser (:func:`~slurmate.tui.
    _parse_custom_flags`) and the list/API path (:func:`_normalize_custom_flags`),
    so ``--custom-sbatch="-o /p"``, ``custom_sbatch = ["-o /p"]`` and
    ``custom_sbatch = ["-o", "/p"]`` all end up as the same single directive.

    ``reassembled`` collects ``(flag_name, fragment)`` for every bare word that
    was folded into a preceding ``--flag=value``, so a caller can say what it
    did.  See the branch below for why it does it at all.
    """
    flags: list[str] = []
    for raw in tokens:
        tok = raw.strip()
        if not tok:
            continue
        if not tok.startswith("-"):
            prev = flags[-1] if flags else ""
            # An option is still "open" for a value while it has neither an "="
            # nor an already-attached space-separated value.
            prev_open = bool(prev) and "=" not in prev and " " not in prev
            if prev_open and (prev in _VALUE_TAKING_FLAGS
                              or not _OPTION_NAME_RE.match(tok)):
                flags[-1] = f"{prev} {tok}"
                continue
            if prev and "=" in prev and " " not in prev:
                # A bare word after `--flag=value` is the tail of a value the
                # user did not quote. It cannot be an option: sbatch options
                # start with a dash, and this token has none.
                #
                # It used to be given one. `--custom-sbatch='--comment=my run'`
                # emitted `#SBATCH --comment=my` and then `#SBATCH --run`, which
                # sbatch refuses outright -- `unrecognized option '--run'` --
                # silently, from a tool whose output is a script somebody submits.
                # `--help` does say to quote such a value and the quoted forms
                # are correct; what was wrong was writing an option nobody typed
                # and no scheduler has.
                #
                # Bounded to ONE word, by the same `" " not in prev` test the
                # branch above uses. Unbounded, it absorbed every later dashless
                # token -- and that LOST REAL OPTIONS, which is worse than the
                # invalid directive it replaced:
                #
                #   --custom-sbatch='--array=1-10 hold exclusive requeue'
                #       -> #SBATCH --array="1-10 hold exclusive requeue"
                #          (three options gone)
                #   --custom-sbatch='--comment="my run" hold'
                #       -> #SBATCH --comment="my run hold"
                #          (--hold gone, from a CORRECTLY QUOTED value)
                #
                # The second is the one that settles it: `shlex` has already
                # stripped the quotes, so the value arrives holding a space, and
                # requiring `prev` to be space-free leaves it alone. sbatch
                # rejected `#SBATCH --run` loudly; a dropped `--hold` runs.
                #
                # The fold keeps the script VALID for a caller that ignores
                # `reassembled`; the caller that reads it refuses outright, so
                # nothing is lost quietly either way.
                if reassembled is not None:
                    reassembled.append((prev.split("=", 1)[0], tok))
                flags[-1] = f"{prev} {tok}"
                continue
            tok = f"--{tok}"
        flags.append(tok)
    return flags


def _normalize_custom_flags(
    custom_sbatch: Any, reassembled: list[tuple[str, str]] | None = None
) -> list[str]:
    """Coerce ``custom_sbatch`` (list | str | None) into a clean list of flags.

    Single place that (a) tolerates a bare string from a direct API caller — which
    would otherwise be iterated character-by-character — (b) folds CR/LF in an
    entry to a space so one entry can never become two script lines, and (c)
    rejoins an option that was split from its value across two list elements
    (a TOML ``custom_sbatch = ["-o", "/logs/%j.out"]`` used to emit a valueless
    ``#SBATCH -o`` plus a bare path line). Every consumer — the memory override,
    the constraint merge, the output/error dedup, the emit loop and
    ``job_summary_rows`` — works from this same list, so the summary and the
    script can't disagree about what the custom flags say.
    """
    if not custom_sbatch:
        return []
    if isinstance(custom_sbatch, str):
        from .tui import _parse_custom_flags
        custom_sbatch = _parse_custom_flags(custom_sbatch, reassembled)
    cleaned = [
        str(raw).replace("\r", " ").replace("\n", " ").strip() for raw in custom_sbatch
    ]
    return _join_flag_values([c for c in cleaned if c], reassembled)


def _split_flag(flag: str) -> tuple[str, str]:
    """Split a custom flag into ``(name, value)``, however it was written.

    Handles ``--mem=16G``, ``--mem 16G`` and ``-C bigmem`` alike, and strips a
    quote pair the caller left around the value, so callers can compare values
    without caring about the spelling.
    """
    parts = re.split(r"[=\s]", flag, maxsplit=1)
    name = parts[0].strip()
    value = parts[1].strip() if len(parts) > 1 else ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return name, value


# Custom-flag names that collide with a directive the builder emits itself.
_MEM_FLAG_NAMES = ("--mem",)
_MEM_PER_CPU_FLAG_NAMES = ("--mem-per-cpu",)
# The third member of Slurm's mutually exclusive memory family. slurmate has no
# option of its own for it, so --custom-sbatch=--mem-per-gpu=1G is the only way
# to ask for per-GPU memory — and it has to suppress the auto --mem the same way
# a custom --mem/--mem-per-cpu does. Measured on Slurm 20.11.8: a script with
# both is not merely overridden, it is refused outright — `sbatch: fatal: --mem,
# --mem-per-cpu, and --mem-per-gpu are mutually exclusive.` — so slurmate emitted
# an unsubmittable script for a GPU job whose user never mentioned memory at all
# (the --mem comes from the partition default).
_MEM_PER_GPU_FLAG_NAMES = ("--mem-per-gpu",)
_CONSTRAINT_FLAG_NAMES = ("--constraint", "-C")
_OUTPUT_FLAG_NAMES = ("--output", "-o")
_ERROR_FLAG_NAMES = ("--error", "-e")


# Directives slurmate owns and *reconciles* when a custom flag also sets them:
# --mem/--mem-per-cpu (the custom value wins, the auto one is suppressed),
# --constraint/-C (merged into one directive), --output/--error (de-duplicated).
# A custom flag naming one of these is fine — the machinery already accounts for
# it, and the merged constraint case is behaviour the portability report asked to
# keep.
_RECONCILED_CUSTOM_FLAGS = {
    "--mem", "--mem-per-cpu", "--constraint", "-C", "--output", "-o",
    "--error", "-e",
}

# Directives slurmate owns and does NOT reconcile. A custom flag repeating one of
# these emits a second #SBATCH line; Slurm honours the LAST, so the job runs with
# the custom value while slurmate's summary, its cluster validation and its
# queue/ETA figures all describe the first. Each maps to the flag that owns it.
_MANAGED_CUSTOM_FLAGS = {
    "--partition": "--partition", "-p": "--partition",
    "--account": "--account", "-A": "--account",
    "--qos": "--qos", "-q": "--qos",
    "--time": "--time", "-t": "--time",
    # Each maps to a spelling the CLI actually accepts. SM-25 made Slurm's own
    # spellings first-class (--cpus-per-task, --gres, --gpus-per-node/task), so
    # these used to send the user to a *different* flag than the one they typed —
    # and for --gres that lost information, since --gres gpu:a100:2 carries a type
    # that a bare --gpus does not.
    "--cpus-per-task": "--cpus-per-task", "-c": "--cpus-per-task",
    "--nodes": "--nodes", "-N": "--nodes",
    "--ntasks-per-node": "--ntasks-per-node",
    "--job-name": "--job-name", "-J": "--job-name",
    "--array": "--array", "-a": "--array",
    "--gres": "--gres", "--gpus": "--gpus", "-G": "--gpus",
    "--gpus-per-node": "--gpus-per-node", "--gpus-per-task": "--gpus-per-task",
}


def output_dir_is_used(output_dir: Any, output_file: Any) -> bool:
    """Whether ``output_dir`` will actually place the log files.

    The builder puts only a **bare** filename inside it — an absolute or
    directory-bearing ``output_file`` is left alone. So with
    ``--output-file /tmp/x.out --output-dir logs`` the script writes to ``/tmp``
    while the summary still said ``Output directory: logs``, sending the user to
    an empty directory. Shared with the summary so the two cannot disagree.
    """
    if not str(output_dir or "").strip():
        return False
    name = os.path.expanduser(str(output_file or "").strip())
    if not name:
        return True                  # no explicit file: the directory is used
    return not os.path.isabs(name) and not os.path.dirname(name)


def env_activation_emitted(env_name: Any, env_type: Any) -> bool:
    """Whether an ``env_name`` will actually produce an activation line.

    ``--env-type none`` (a documented choice) emits nothing, so an ``--env`` given
    alongside it is silently dropped: the summary still showed
    ``Environment: myenv`` while the script never activated it, and the only
    signal was a ``logger.warning`` no user sees. The predicate lets the summary
    and the checks agree with what the builder actually emits.
    """
    if not str(env_name or "").strip():
        return False
    return str(env_type or "conda").strip().lower() in (
        "conda", "mamba", "venv", "virtualenv (venv)"
    )


def command_injects_directives(command: Any) -> str:
    """The first ``#SBATCH`` line in ``command`` that Slurm would obey, or "".

    Slurm stops reading directives at the first line that is neither blank nor a
    comment. The command body is emitted after the directive block, so a
    ``#SBATCH`` line at the *start* of the body — before any real command — is
    still inside the directive region and takes effect. Measured: a command of
    ``#SBATCH --qos=INJECTED`` produced ``Access/permission denied`` from the
    controller, which is its answer for an invalid QoS, so the directive was
    obeyed. It appears in no summary, is validated by nothing, and bypasses the
    managed-flag check that covers ``--custom-sbatch``.

    Only the *leading* run matters, which is what makes this safe to enforce: a
    ``#SBATCH`` inside a heredoc that writes a nested script is preceded by a
    real command, so Slurm has already stopped parsing and the line is inert.
    """
    for raw in str(command or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("#"):
            return ""            # a real command line: Slurm stops here
        if re.match(r"^#\s*SBATCH\b", line, re.IGNORECASE):
            return line
    return ""


def unquoted_custom_values(custom_sbatch: Any) -> list[tuple[str, str]]:
    """``(flag, whole value)`` for each custom flag whose value was unquoted.

    Empty when every value was quoted, which is what ``--help`` asks for and what
    the vast majority of invocations do.  Reported so the fold that keeps the
    script valid is not also silent: the user typed something with two readings
    and gets told which one was taken.
    """
    notes: list[tuple[str, str]] = []
    flags = _normalize_custom_flags(custom_sbatch, notes)
    if not notes:
        return []
    folded = {name for name, _fragment in notes}
    out: list[tuple[str, str]] = []
    for flag in flags:
        name = flag.split("=", 1)[0].strip().split()[0] if flag.strip() else ""
        if name in folded:
            _n, value = _split_flag(flag)
            out.append((name, value))
    return out


def managed_custom_flags(custom_sbatch: Any) -> list[tuple[str, str]]:
    """``(custom_flag, owning_slurmate_flag)`` for each conflicting custom flag.

    Empty when there is no conflict. Only the *unreconciled* directives count:
    a custom ``--mem`` or ``-C`` is deliberately supported, and refusing those
    would undo behaviour this package is relied on for.
    """
    out: list[tuple[str, str]] = []
    for flag in _normalize_custom_flags(custom_sbatch):
        name = flag.split("=", 1)[0].strip().split()[0] if flag.strip() else ""
        if not name or name in _RECONCILED_CUSTOM_FLAGS:
            continue
        owner = _MANAGED_CUSTOM_FLAGS.get(name)
        if owner:
            out.append((name, owner))
    return out


def custom_ntasks(custom_sbatch: Any) -> int | None:
    """A total task count supplied via ``--custom-sbatch --ntasks``, or ``None``.

    slurmate has no ``--ntasks`` option, so ``--custom-sbatch=--ntasks=N`` is the
    *only* way to express an MPI job with it — which makes this a likely path
    rather than an exotic one. The cost estimate multiplies by tasks, so a custom
    ``--ntasks=100`` left it reporting a hundredth of the real footprint: 2.0
    core-hours for a job asking for 200 cores.

    Last occurrence wins, mirroring Slurm's "later option overrides earlier" and
    :func:`_custom_mem_override`.
    """
    total: int | None = None
    for flag in _normalize_custom_flags(custom_sbatch):
        name, value = _split_flag(flag)
        if name in ("--ntasks", "-n") and value:
            try:
                parsed = int(str(value).strip())
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                total = parsed
    return total


def _custom_mem_override(
    flags: list[str],
) -> tuple[str | None, str | None, str | None]:
    """``(mem, mem_per_cpu, mem_per_gpu)`` from custom flags, else all ``None``.

    All three of Slurm's memory directives are reported, because all three are
    mutually exclusive of one another: whichever one a custom flag carries, the
    auto directive must give way or the controller refuses the script. Matched on
    the exact flag name, so ``--mem-bind`` — which merely starts the same — is
    left alone and does not silently drop the memory request.

    Last occurrence wins, mirroring Slurm's own "later option overrides earlier"
    behaviour, and both the ``=`` and space spellings count.
    """
    mem: str | None = None
    mem_per_cpu: str | None = None
    mem_per_gpu: str | None = None
    for flag in flags:
        name, value = _split_flag(flag)
        if name in _MEM_FLAG_NAMES:
            mem = value
        elif name in _MEM_PER_CPU_FLAG_NAMES:
            mem_per_cpu = value
        elif name in _MEM_PER_GPU_FLAG_NAMES:
            mem_per_gpu = value
    return mem, mem_per_cpu, mem_per_gpu


def _has_custom_flag(flags: list[str], names: tuple[str, ...]) -> bool:
    return any(_split_flag(f)[0] in names for f in flags)


def _clean_constraint(value: str) -> str:
    """Strip all whitespace from a node-feature expression.

    Slurm's feature grammar has no room for spaces: ``-C "a100 & 384g"`` is
    rejected outright ("Invalid feature specification") while ``-C "a100&384g"``
    schedules — measured against a live sbatch. A user typing the spaced form (or
    a stray leading space, which produced ``--constraint= a100``) would otherwise
    get a job Slurm refuses, so normalize instead of passing it through. Feature
    names cannot contain whitespace, so nothing legitimate is lost.
    """
    return re.sub(r"\s+", "", _fold_directive(str(value)))


def _auto_gpu_flag_name(gpu_format: str | None) -> str:
    """The sbatch option the chosen GPU format makes the builder emit.

    Used to spot a custom flag that overrides that exact option (so the auto one
    is suppressed rather than emitted alongside it) and to report the override in
    the summary. Deliberately keyed on the option *name*: ``--gres=gpu:2`` and
    ``--gpus=2`` are different requests to Slurm, so a custom ``--gpus`` must not
    suppress an auto ``--gres``.
    """
    fmt = (gpu_format or os.environ.get("SLURMATE_GPU_FORMAT", "gres_type")).lower()
    return {
        "gpus": "--gpus",
        "gpus_per_node": "--gpus-per-node",
        "gpus_per_task": "--gpus-per-task",
    }.get(fmt, "--gres")


def _constraint_term(value: str) -> str:
    """Wrap an OR-expression in parentheses so ``&``-joining keeps its meaning.

    ``a|b`` merged into ``gpu&a|b`` is ambiguous; ``gpu&(a|b)`` is what the user
    meant. A value that is already grouped (``(…)``) or is a Slurm count-bracket
    expression (``[…]``) is left alone.
    """
    v = value.strip()
    if "|" in v and (v[:1], v[-1:]) not in (("(", ")"), ("[", "]")):
        return f"({v})"
    return v


def _gpus_int(answers: dict[str, Any]) -> int:
    g = answers.get("gpus", 0)
    try:
        return int(g) if g is not None else 0
    except (TypeError, ValueError):
        return 0


def _fold_or_none(value: Any) -> Any:
    """CR/LF-fold a free-text directive value for display, preserving None."""
    if value is None or str(value).strip() == "":
        return None
    return _fold_directive(str(value))


def job_summary_rows(answers: dict[str, Any]) -> list[tuple[str, str]]:
    """Ordered (label, value) rows for the job configuration summary.

    Single source of truth shared by the CLI summary panel and the in-TUI
    Review step, so both surfaces show the same fields in the same order.
    Empty/absent fields are omitted.
    """
    rows: list[tuple[str, str]] = []

    def add(label: str, val: Any) -> None:
        if val is None:
            return
        text = ", ".join(str(x) for x in val) if isinstance(val, list) else str(val)
        if text:
            rows.append((label, text))

    # Show what Slurm will see, not what was typed. These fields are transformed
    # on the way into the script — the name is sanitized, memory is normalized,
    # free-text values are CR/LF-folded — and the CLI happens to pre-transform
    # them before they reach here, so the two agreed by accident rather than by
    # construction. A library caller got a summary describing its input and a
    # script carrying something else.
    add("Job name", sanitize_job_name(str(answers.get("job_name") or "")) or None)
    add("Partition", answers.get("partition"))
    add("Account", _fold_or_none(answers.get("account")))
    qos = answers.get("qos")
    if qos and qos != "Default (none)":
        add("QoS", qos)
    add("CPUs", answers.get("cpus"))
    # Memory must reflect what the SCRIPT will actually request, not what the
    # answers dict happens to hold: a custom --mem / --mem-per-cpu flag suppresses
    # the auto directive in the builder, so showing the (now unused) answer value
    # here made the summary and the generated script disagree.
    custom_flags = _normalize_custom_flags(answers.get("custom_sbatch"))
    custom_mem, custom_mem_per_cpu, custom_mem_per_gpu = _custom_mem_override(
        custom_flags
    )
    if custom_mem_per_cpu or custom_mem or custom_mem_per_gpu:
        # A custom flag wins. Several can be present (Slurm would reject that, but
        # it's the user's script) — show whatever the script really says.
        add("Mem per GPU", custom_mem_per_gpu)
        add("Mem per CPU", custom_mem_per_cpu)
        add("Memory", custom_mem)
    elif answers.get("mem_per_cpu"):
        # Mirror the builder: --mem-per-cpu takes precedence over --mem when set.
        add("Mem per CPU", answers.get("mem_per_cpu"))
        # And say so when a --memory was also supplied. Slurm rejects the two
        # together, so the builder emits only one — but the discarded value was
        # then absent from the summary entirely, which reads as "I never set
        # that". It matters most for the case that cannot be seen: a `memory`
        # key inherited from a config file, silently dropped by a --mem-per-cpu
        # typed on the command line. Same disclosure the Output directory row
        # makes when a flag is given and has no effect.
        if answers.get("memory"):
            add(
                "Memory",
                f"{answers.get('memory')} (not used — --mem-per-cpu takes "
                f"precedence, and Slurm rejects both together)",
            )
    else:
        # normalize_memory is what the builder emits, so the row must show it:
        # "1.5G" becomes "1536M" and "16" becomes "16M" in the directive.
        raw_mem = answers.get("memory")
        add("Memory", normalize_memory(str(raw_mem)) if raw_mem else raw_mem)
    add("Time limit", answers.get("time_limit"))
    # Mirror the builder's own default: build_sbatch_script receives
    # opt("nodes", 1), so an absent value still emits `#SBATCH --nodes=1`. Reading
    # the raw answer here omitted the row, leaving a directive in the script that
    # nothing in the summary accounted for — SM-15's shape in miniature. (The
    # value is not an imposition: 1 node is Slurm's own default too. It just has
    # to be visible, because the summary is what the user checks the script by.)
    nodes = answers.get("nodes")
    if nodes is None or str(nodes) == "":
        nodes = 1
    add("Nodes", nodes)
    if answers.get("ntasks_per_node"):
        add("Tasks per node", answers.get("ntasks_per_node"))
    if _gpus_int(answers) > 0:
        # Same rule as memory above: report the request the SCRIPT makes. A custom
        # flag on the option the chosen format would emit overrides it, so showing
        # "2 × a100" while the script says --gres=gpu:h100:4 would be a plain lie.
        auto_gpu = _auto_gpu_flag_name(answers.get("gpu_format"))
        gpus_n = _gpus_int(answers)
        gpu_t = answers.get("gpu_type")
        typed = bool(gpu_t and str(gpu_t).lower() != "any")
        if auto_gpu == "--gres":
            auto_val = f"gpu:{gpu_t}:{gpus_n}" if (
                typed and (answers.get("gpu_format") or "gres_type") == "gres_type"
            ) else f"gpu:{gpus_n}"
        else:
            auto_val = f"{gpu_t}:{gpus_n}" if typed else f"{gpus_n}"
        override = next(
            (f for f in custom_flags
             for n, v in [_split_flag(f)] if n == auto_gpu and v and v != auto_val),
            None,
        )
        if override:
            add("GPUs", f"{override} (custom flag)")
        else:
            add("GPUs", f"{answers.get('gpus')} × {answers.get('gpu_type') or 'any'}")
            add("GPU format", answers.get("gpu_format"))
    add("Constraint", answers.get("constraint"))
    add("Array specification", answers.get("array_spec"))
    # The --output/--error directives are emitted unconditionally, so a row has to
    # account for them. The CLI and the wizard both default this to "logs", but a
    # direct API caller may omit it — and then the logs land in the working
    # directory, which is worth saying rather than leaving the row out.
    out_dir = answers.get("output_dir")
    if out_dir and not output_dir_is_used(out_dir, answers.get("output_file")):
        # The flag was given and has no effect: say that, rather than naming a
        # directory the job will not write to.
        add("Output directory", f"{out_dir} (not used — output file has its own path)")
    else:
        add("Output directory", out_dir or "(current directory)")
    add("Output file", answers.get("output_file"))
    add("Modules", answers.get("modules"))
    # Say when the name will not be acted on, rather than implying activation.
    env_name = answers.get("env_name")
    if env_name and not env_activation_emitted(env_name, answers.get("env_type")):
        add("Environment", f"{env_name} (not activated — env_type "
                           f"{answers.get('env_type') or 'none'!s})")
    else:
        add("Environment", env_name)
    add("Custom flags", answers.get("custom_sbatch"))
    add("Command", answers.get("command"))
    return rows


def build_from_answers(answers: dict[str, Any], partial: bool = False) -> str:
    """Build an sbatch script from an answers dict.

    Args:
        answers: Collected wizard/CLI answers.
        partial: When True, only emit directives for keys the user has actually
            provided (used by the live preview, so unentered fields don't show
            up as placeholder lines). When False, defaults fill in a complete,
            submittable script.
    """
    # Expand ~ / ~user in log paths at build time: neither Slurm nor
    # os.makedirs expands a leading "~", so an unexpanded "~/logs" would create a
    # literal "./~" directory and send logs to the wrong place.
    # strip() FIRST: expanduser() only acts on a value that *starts* with "~", so
    # a config/CLI value written as " ~/logs" (leading space) would otherwise be
    # stripped later and emitted with a literal "~" still attached.
    output_dir = answers.get("output_dir")
    if output_dir:
        output_dir = os.path.expanduser(str(output_dir).strip())
    output_file = answers.get("output_file")
    if output_file:
        output_file = os.path.expanduser(str(output_file).strip())
    job_name = sanitize_job_name(answers.get("job_name", ""))
    prefix = job_name if job_name else "slurm"

    def _in_dir(name: str) -> str:
        # Place a bare filename inside output_dir; leave explicit paths alone.
        if output_dir and not os.path.isabs(name) and not os.path.dirname(name):
            return f"{output_dir.strip().rstrip('/')}/{name}"
        return name

    # Array jobs conventionally log per task with %A (array job id) + %a (task
    # id); a single %j would collide across tasks. Plain jobs keep %j.
    array_spec = answers.get("array_spec")
    tag = "%A_%a" if array_spec else "%j"

    output_path: str | None
    error_path: str | None
    if output_file:
        of = output_file.strip()
        base, ext = os.path.splitext(of)
        # Only a per-task token (%a, or %j which is the per-task job id) makes an
        # explicit pattern unique across array tasks. %A alone (the array *master*
        # id, identical for every task) or a stray literal "%" would make every
        # task write the same file, so treat those as "no usable pattern" and
        # still insert the per-task %A_%a tag.
        has_task_pattern = "%a" in of or "%j" in of
        if array_spec and not has_task_pattern:
            # An explicit output_file with no per-task pattern would make every
            # array task write the same file (clobbering each other). Insert the
            # per-task %A_%a tag before the extension, mirroring the output_dir
            # branch, so each task gets its own log.
            if ext:
                output_path = _in_dir(f"{base}-{tag}{ext}")
                error_path = _in_dir(f"{base}-{tag}.err")
            else:
                output_path = _in_dir(f"{of}-{tag}.out")
                error_path = _in_dir(f"{of}-{tag}.err")
        # `os.path.splitext("run.%j")` returns ("run", ".%j") — but a suffix that
        # carries a Slurm pattern character (%) is part of the log *pattern*, not
        # a real extension. Treating it as one dropped %j from the derived error
        # path (every task then overwrote the same file). So: only swap a literal
        # extension; otherwise keep the whole name and append .out/.err.
        elif ext and "%" not in ext:
            output_path = _in_dir(of)
            error_path = _in_dir(base + ".err")
        else:
            output_path = _in_dir(of + ".out")
            error_path = _in_dir(of + ".err")
        # A user output_file whose extension is literally ".err" (or the array
        # variant of it) makes the derived error path equal the output path,
        # collapsing stdout and stderr into one file. Give stderr a distinct
        # name so the two streams stay separate as intended.
        if output_path is not None and output_path == error_path:
            e_base, e_ext = os.path.splitext(error_path)
            error_path = f"{e_base}-err{e_ext or '.err'}"
    elif output_dir:
        out_dir = output_dir.strip().rstrip("/")
        output_path = f"{out_dir}/{prefix}-{tag}.out"
        error_path = f"{out_dir}/{prefix}-{tag}.err"
    else:
        output_path = None
        error_path = None

    def opt(key: str, default: Any) -> Any:
        # In partial mode, leave a value unset (None) until the user supplies it.
        if partial and key not in answers:
            return None
        return answers.get(key, default)

    return build_sbatch_script(
        job_name=job_name,
        partition=answers.get("partition", ""),
        account=answers.get("account"),
        qos=answers.get("qos"),
        cpus=opt("cpus", 1),
        memory=opt("memory", "16G"),
        time_limit=opt("time_limit", "02:00:00"),
        nodes=opt("nodes", 1),
        ntasks_per_node=answers.get("ntasks_per_node"),
        gpus=_gpus_int(answers),
        gpu_type=answers.get("gpu_type"),
        array_spec=answers.get("array_spec"),
        output_path=output_path,
        error_path=error_path,
        modules=answers.get("modules"),
        custom_sbatch=answers.get("custom_sbatch"),
        env_name=answers.get("env_name"),
        env_type=answers.get("env_type"),
        gpu_format=answers.get("gpu_format"),
        constraint=answers.get("constraint"),
        mem_per_cpu=opt("mem_per_cpu", None),
        command=answers.get("command", ""),
        partial=partial,
    )


def build_sbatch_script(
    job_name: str,
    partition: str,
    cpus: int | None,
    memory: str | None,
    time_limit: str | None,
    nodes: int | None = 1,
    ntasks_per_node: int | None = None,
    gpus: int = 0,
    gpu_type: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    array_spec: str | None = None,
    output_path: str | None = None,
    error_path: str | None = None,
    modules: list[str] | None = None,
    custom_sbatch: list[str] | None = None,
    env_name: str | None = None,
    env_type: str | None = None,
    gpu_format: str | None = None,
    constraint: str | None = None,
    mem_per_cpu: str | None = None,
    command: str = "",
    partial: bool = False,
) -> str:
    lines = ["#!/bin/bash", ""]

    # Defensive: a raw job name with whitespace would split the directive
    # (`--job-name=my training job` → name becomes `my`). Sanitize here too so
    # direct callers of build_sbatch_script are covered, not just the wizard.
    job_name = sanitize_job_name(job_name)

    # Defensive coercion for direct callers passing stringy numbers (e.g. from a
    # config value) — otherwise the `gpus > 0` / `nodes > 1` comparisons below
    # raise TypeError comparing str and int.
    try:
        gpus = int(gpus)
    except (TypeError, ValueError):
        gpus = 0
    if nodes is not None:
        try:
            nodes = int(nodes)
        except (TypeError, ValueError):
            nodes = 1
    # Same defensive coercion for gpu_type: a direct API caller (or a stringy
    # value that bypassed the CLI/TUI, which already stringify it) may pass a
    # non-string; str() it so the .lower()/_fold_directive calls below don't
    # raise AttributeError on e.g. an int.
    if gpu_type is not None:
        gpu_type = str(gpu_type)

    # One contiguous #SBATCH block, emitted in the same order the wizard asks
    # the questions, so the live preview grows top-to-bottom without reshuffling.
    # Omit an empty job-name/partition rather than emitting a malformed
    # `--job-name=` / `--partition=` (sbatch then auto-names / uses the default
    # partition), matching how account/qos are handled.
    if job_name:
        lines.append(f"#SBATCH --job-name={job_name}")
    if partition:
        lines.append(f"#SBATCH --partition={_fold_directive(partition)}")
    if account:
        lines.append(f"#SBATCH --account={_fold_directive(account)}")
    if qos and qos != "Default (none)":
        lines.append(f"#SBATCH --qos={_fold_directive(qos)}")
    # Every free-form value goes through _fold_directive: a CR/LF in memory or
    # time_limit (both free-form strings, often config-sourced) would otherwise
    # inject a script-body line and silently drop the directives after it, the
    # same hazard partition/account/qos are folded against. cpus/ntasks are
    # normally ints but are folded too for defense-in-depth against stringy
    # callers of the builder API that bypass the CLI/TUI validators.
    if cpus is not None:
        lines.append(f"#SBATCH --cpus-per-task={_fold_directive(str(cpus))}")
    # Normalize the custom flags once, up front: the memory override, the
    # constraint merge and the output/error dedup below all consult this list
    # before their own directive is emitted.
    custom_flags = _normalize_custom_flags(custom_sbatch)
    # If the user supplied their own memory directive via custom flags, don't also
    # emit the auto one: Slurm rejects a script that sets --mem, --mem-per-cpu or
    # --mem-per-gpu together, so a user override wins (mirrors the GPU-flag dedup
    # below). All three count — --mem-per-gpu has no slurmate option, so the
    # passthrough is the only way to ask for it.
    _cm, _cmpc, _cmpg = _custom_mem_override(custom_flags)
    _custom_mem = bool(_cm or _cmpc or _cmpg)
    # Memory: --mem-per-cpu takes precedence over --mem when set (Slurm treats the
    # two as mutually exclusive). A blank memory omits the directive entirely — what
    # whole-node/exclusive sites need: e.g. TACC rejects any script that sets --mem.
    if _custom_mem:
        pass  # a custom --mem / --mem-per-cpu flag is emitted below instead
    elif mem_per_cpu:
        lines.append(
            f"#SBATCH --mem-per-cpu={_fold_directive(normalize_memory(str(mem_per_cpu)))}"
        )
    elif memory:
        # normalize_memory here, not only in the CLI: `sbatch --mem` requires an
        # integer magnitude, so a fractional value that validate_memory accepts
        # ("1.5G") is refused by the controller with "Invalid --mem
        # specification" — measured. The CLI and the wizard both normalized before
        # calling, so the emitted directive was correct *by accident of the
        # caller*; a library caller got an unsubmittable script, and the summary
        # row disagreed with it. Idempotent, so the pre-normalising callers are
        # unaffected.
        lines.append(f"#SBATCH --mem={_fold_directive(normalize_memory(str(memory)))}")
    if time_limit:
        lines.append(f"#SBATCH --time={_fold_directive(str(time_limit))}")
    if nodes is not None:
        lines.append(f"#SBATCH --nodes={nodes}")
    if ntasks_per_node is not None:
        lines.append(f"#SBATCH --ntasks-per-node={_fold_directive(str(ntasks_per_node))}")
    elif nodes is not None and nodes > 1 and custom_ntasks(custom_sbatch) is None:
        # The auto directive gives way to a custom task count, for the reason
        # `_custom_mem_override` already states about the memory trio: "whichever
        # one a custom flag carries, the auto directive must give way or the
        # controller refuses the script." This was the one auto directive that
        # did not follow it, and slurmate has no `--ntasks` of its own, so
        # `--custom-sbatch=--ntasks=N` is the only way to express an MPI job --
        # a likely path rather than an exotic one, as `custom_ntasks` says.
        #
        # Measured on Slurm 20.11.8. `--nodes=4 --ntasks=8` is accepted; the same
        # request with `--ntasks-per-node=1` added is refused with "Requested node
        # configuration is not available", because one task per node over 8 tasks
        # asks for 8 nodes and `--nodes` caps it at 4. So slurmate built a script
        # its own pre-submit check then refused -- `--nodes 4
        # --custom-sbatch=--ntasks=8` printed exactly that.
        #
        # Only the *fallback* gives way. An explicit `--ntasks-per-node` above is
        # the user's own instruction and is emitted whatever else is set; a custom
        # `--ntasks-per-node` is a duplicate rather than a conflict and keeps its
        # existing `_MANAGED_CUSTOM_FLAGS` warning.
        lines.append("#SBATCH --ntasks-per-node=1")

    # Node-feature constraint(s) (Slurm -C), e.g. NERSC Perlmutter's mandatory
    # `-C cpu`/`-C gpu`. Collected here and emitted once *after* the GPU block, so a
    # GPU-as-constraint (gpu_format="constraint") MERGES with a node -C via "&"
    # rather than emitting a second, conflicting --constraint line (Slurm would
    # otherwise keep only the last one, silently dropping the node feature).
    constraint_parts: list[str] = []
    if constraint:
        constraint_parts.append(_clean_constraint(constraint))

    # A custom --constraint / -C is merged into the SAME directive rather than
    # appended as a second one. Slurm keeps only the last --constraint it sees and
    # silently discards the earlier one (measured: an invalid feature placed first
    # schedules fine, placed last it fails with "Invalid feature specification"),
    # and because the custom-flag loop runs last, the directive being discarded was
    # always slurmate's own — dropping the GPU type or the node feature the user
    # asked for, with no error. Merging with "&" (AND) keeps both requirements.
    # Collected here, appended after the GPU block (so the merged value reads
    # param → GPU type → custom), and skipped in the emit loop below.
    custom_constraints: list[str] = []
    consumed_custom: set[int] = set()
    for _i, _flag in enumerate(custom_flags):
        _name, _value = _split_flag(_flag)
        if _name in _CONSTRAINT_FLAG_NAMES and _value:
            custom_constraints.append(_clean_constraint(_value))
            consumed_custom.add(_i)

    gpu_fmt = (gpu_format or os.environ.get("SLURMATE_GPU_FORMAT", "gres_type")).lower()
    gpu_any = gpu_type is not None and gpu_type.lower() == "any"
    if gpu_type:
        gpu_type = _fold_directive(gpu_type)
    # The exact GPU directive values the chosen format emits, used below to drop
    # only a custom flag that *duplicates* them (not a differing user override).
    emitted_gres: str | None = None
    emitted_gpus: str | None = None
    emitted_gpus_per_node: str | None = None
    emitted_gpus_per_task: str | None = None
    if gpus > 0:
        typed = bool(gpu_type and not gpu_any)
        # The value the chosen format would request, computed before emitting so a
        # custom flag on the same option can be compared against it.
        if gpu_fmt in ("gpus", "gpus_per_node", "gpus_per_task"):
            auto_value = f"{gpu_type}:{gpus}" if typed else f"{gpus}"
        elif gpu_fmt == "gres_type" and typed:
            auto_value = f"gpu:{gpu_type}:{gpus}"
        else:  # "constraint" (also gres_type with no/any type)
            auto_value = f"gpu:{gpus}"
        # A custom flag on the SAME option, with a DIFFERENT value, suppresses the
        # auto directive — exactly as a custom --mem/--output does. Slurm honours
        # the last option, so the override already won; leaving the auto directive
        # in the script merely contradicted it, and made the summary describe a GPU
        # request the job doesn't make. An *exact* duplicate is handled the other way
        # round (auto kept, custom dropped by the loop below) so the script keeps
        # slurmate's canonical `=` spelling. Only the option's own name counts:
        # --gres and --gpus are different requests to Slurm.
        auto_flag = _auto_gpu_flag_name(gpu_fmt)
        custom_value = next(
            (v for f in custom_flags
             for n, v in [_split_flag(f)] if n == auto_flag and v),
            None,
        )
        overridden = custom_value is not None and custom_value != auto_value
        if not overridden:
            if gpu_fmt == "gpus":
                emitted_gpus = auto_value
                lines.append(f"#SBATCH --gpus={emitted_gpus}")
            elif gpu_fmt == "gpus_per_node":
                emitted_gpus_per_node = auto_value
                lines.append(f"#SBATCH --gpus-per-node={emitted_gpus_per_node}")
            elif gpu_fmt == "gpus_per_task":
                emitted_gpus_per_task = auto_value
                lines.append(f"#SBATCH --gpus-per-task={emitted_gpus_per_task}")
            else:
                emitted_gres = auto_value
                lines.append(f"#SBATCH --gres={emitted_gres}")
        # The GPU type is a *separate* requirement from the GRES count under the
        # constraint format, so it is recorded either way (the merge above keeps it
        # alongside any node -C).
        if gpu_fmt == "constraint" and typed:
            constraint_parts.append(str(gpu_type))

    constraint_parts.extend(custom_constraints)

    # Emit the merged node/GPU/custom constraint as a single directive. De-dup
    # case-SENSITIVELY: Slurm node features are case-sensitive (measured — a node
    # advertising "a100" is not matched by "-C A100"), so "A100" and "a100" are
    # different requirements and must not be folded together.
    if constraint_parts:
        merged: list[str] = []
        for part in constraint_parts:
            if part and part not in merged:
                merged.append(part)
        joined = "&".join(_constraint_term(p) for p in merged) if len(merged) > 1 else merged[0]
        lines.append(f"#SBATCH --constraint={joined}")

    if array_spec:
        lines.append(f"#SBATCH --array={_fold_directive(array_spec)}")

    # Output/error are auto-derived. In a partial preview, only show them once
    # the user has actually configured an output dir/file (output_path is set).
    # A custom --output/-o (or --error/-e) SUPPRESSES the matching auto directive,
    # exactly as a custom --mem does above: Slurm honours the last one, so emitting
    # both left a contradictory directive in the script and made every consumer that
    # reads the script (the "Log path:"/tail -f hint, the log-dir pre-creation) pick
    # the wrong file. Each stream is handled independently, so overriding only
    # stdout keeps the derived stderr path.
    if not partial or output_path:
        prefix = job_name if job_name else "slurm"
        tag = "%A_%a" if array_spec else "%j"
        out = output_path or f"{prefix}-{tag}.out"
        err = error_path or f"{prefix}-{tag}.err"
        if not _has_custom_flag(custom_flags, _OUTPUT_FLAG_NAMES):
            lines.append(f"#SBATCH --output={_quote_sbatch_value(out)}")
        if not _has_custom_flag(custom_flags, _ERROR_FLAG_NAMES):
            lines.append(f"#SBATCH --error={_quote_sbatch_value(err)}")

    if custom_flags:
        for idx, flag in enumerate(custom_flags):
            # Already merged into the single --constraint directive above.
            if idx in consumed_custom:
                continue
            if gpus > 0:
                # Derive the flag name whether written with '=' or a space, and
                # only drop a custom flag that would *exactly duplicate* the
                # directive the chosen gpu_format already emits. A custom flag
                # with a *different* value (e.g. --gres=gpu:h100:4 overriding the
                # wizard's a100) is a deliberate override and must be kept, so it
                # isn't silently discarded (previously any gpu: --gres and any
                # --gpus were stripped, dropping user overrides).
                flag_name, flag_val = _split_flag(flag)
                if flag_name == "--gres" and emitted_gres is not None \
                        and flag_val == emitted_gres:
                    continue
                if flag_name == "--gpus" and emitted_gpus is not None \
                        and flag_val == emitted_gpus:
                    continue
                if flag_name == "--gpus-per-node" and emitted_gpus_per_node is not None \
                        and flag_val == emitted_gpus_per_node:
                    continue
                if flag_name == "--gpus-per-task" and emitted_gpus_per_task is not None \
                        and flag_val == emitted_gpus_per_task:
                    continue
                # (A custom --constraint needs no dedup here: every one of them was
                # merged into the single --constraint directive above, which drops
                # an exact duplicate of the GPU type as part of the merge.)
            lines.append(f"#SBATCH {_quote_custom_flag(flag)}")

    # Defensive: a bare string here (from a direct API caller) would be iterated
    # character-by-character (module load n, module load o, …). Mirror the
    # custom_sbatch coercion above and split a stray string on commas so a
    # misuse degrades to sensible output instead of one directive per character.
    if isinstance(modules, str):
        modules = [m.strip() for m in modules.split(",") if m.strip()]
    if modules:
        lines.append("")
        for mod in modules:
            # Defensive: str() first so a non-string element (from a direct API
            # caller) can't raise on the .endswith below.
            mod = str(mod)
            # Strip "(default)" annotation that the module system appends
            if mod.endswith("(default)"):
                mod = mod[:-9]
            # Fold any CR/LF so a module name can't inject an extra script line.
            mod = _fold_directive(mod).strip()
            if not mod:
                continue
            # Shell-quote the token (matching env_name below): shell metacharacters
            # in a module name would otherwise survive into the generated script,
            # and a name with a space would split into two `module load` arguments.
            quoted_mod = shlex.quote(mod)
            lines.append(
                f"module load {quoted_mod}{_abort_guard(f'module load {mod}')}"
            )

    if env_name:
        strategy = (env_type or "conda").lower()
        if strategy in ("conda", "mamba"):
            # Robust activation for a non-login batch shell (the script is
            # `#!/bin/bash`): source conda.sh first so the `conda`/`mamba` shell
            # functions are defined, then activate. Bare `source activate <env>`
            # (the old form) silently fails on modern conda (4.4+) whenever the
            # job's shell hasn't been conda-initialized — i.e. the common batch
            # case — leaving the job in the base/system Python.
            lines.append("")
            lines.append('source "$(conda info --base)/etc/profile.d/conda.sh"')
            quoted = shlex.quote(env_name)
            if strategy == "mamba":
                # conda.sh defines the `conda` hook only. mamba >= 2 (miniforge's
                # current default) needs its OWN hook, so a bare `mamba activate`
                # here dies with "critical libmamba Shell not initialized" — and,
                # crucially, the script keeps going, so the job silently runs in
                # whatever interpreter it inherited. Fall back to `conda activate`,
                # which activates a mamba-created env identically (verified on
                # miniforge 25.3 / mamba 2.x: the bare form exits 1, this form
                # exits 0 with the right sys.prefix).
                lines.append("# mamba >= 2 needs its own shell hook; conda activates the same env")
                lines.append(
                    f"mamba activate {quoted} >/dev/null 2>&1 || conda activate {quoted}"
                    f"{_abort_guard(f'activating {env_name}')}"
                )
            else:
                lines.append(
                    f"conda activate {quoted}{_abort_guard(f'activating {env_name}')}"
                )
        elif strategy in ("virtualenv (venv)", "venv"):
            lines.append("")
            # rstrip a trailing "/" so "/venv/" doesn't become "/venv//bin/activate".
            activate = shlex.quote(env_name.rstrip("/") + "/bin/activate")
            lines.append(
                f"source {activate}{_abort_guard(f'activating {env_name}')}"
            )
        else:
            logger.warning(f"env_type '{env_type}' with env_name '{env_name}' — no activation line emitted")

    if command:
        lines.append("")
        lines.append(command.rstrip())

    if partial:
        while len(lines) > 2 and lines[-1] == "":
            lines.pop()
    else:
        lines.append("")
    return "\n".join(lines)


# Requested time limits that mean "no limit": `--time=0` is documented Slurm for
# exactly that, and UNLIMITED/INFINITE appear in config-sourced values. Treating
# them as a *zero-length* job and substituting a two-hour default produced a
# confident core-hour figure for something unbounded — the same shape as quoting
# an ETA for a job the scheduler has refused.
UNBOUNDED_ESTIMATE = "unbounded — no time limit"


def _time_is_unbounded(time_limit: str) -> bool:
    """Whether a *requested* time limit means "no limit".

    Delegates to :func:`time_request_is_unbounded`. It used to be a second
    implementation of the same rule, which is how the partition-limit check came
    to read "no limit" as a *zero-length* job while the estimate here read it
    correctly: two copies of one decision, disagreeing. One copy now.
    """
    return time_request_is_unbounded(time_limit)


def _with_array_total(per_task: float, array_tasks: int | None) -> str:
    """Render a cost as the array total, with the per-task figure kept visible.

    The cost of an array job is per-task cost × task count, and showing only the
    per-task figure understates a 1000-task array a thousandfold — in the
    direction that matters, because it tells the user an enormous job is cheap.
    The ``%N`` throttle is not a divisor: it caps concurrency, so it changes the
    wall-clock and not the bill.
    """
    if not array_tasks or array_tasks <= 1:
        return _fmt_estimate(per_task)
    return (
        f"{_fmt_estimate(per_task * array_tasks)} "
        f"({array_tasks} tasks × {_fmt_estimate(per_task)})"
    )


def estimate_su(cpus: int, time_limit: str, nodes: int = 1,
                ntasks_per_node: int | None = None,
                array_tasks: int | None = None,
                ntasks_total: int | None = None) -> str:
    """Estimate a job's compute cost in CPU core-hours.

    Core-hours = CPUs-per-task × tasks-per-node × nodes × hours. When
    ``ntasks_per_node`` is unset it defaults to 1 task. (Kept the ``estimate_su``
    name for back-compat; the UI shows the cluster-agnostic "CPU-hours" rather
    than a site-specific "SU"/billing-unit name.)

    Args:
        cpus: Number of CPU cores per task.
        time_limit: Time limit string in Slurm format (e.g. "hh:mm:ss" or "d-hh:mm:ss").
        nodes: Number of nodes requested.
        ntasks_per_node: Tasks per node (multiplies the per-task core count).

    Returns:
        Formatted string of the estimated core-hours.
    """
    # Clamp negative core/node counts to 0 (parity with the time clamp below) so
    # a direct-API caller passing a negative can't yield a negative estimate.
    cpus = max(0, cpus)
    nodes = max(0, nodes)
    if _time_is_unbounded(time_limit):
        return UNBOUNDED_ESTIMATE
    # An *absent* limit is a different case: the job will get the partition or
    # site default, and 2 h is what the summary shows for it, so estimating
    # against that is consistent rather than invented.
    minutes = _parse_slurm_time_to_minutes(time_limit) if time_limit else 120.0
    if minutes <= 0:
        minutes = 120.0
    hours = minutes / 60.0
    tasks = ntasks_per_node if (ntasks_per_node and ntasks_per_node > 0) else 1
    # An explicit total (Slurm's --ntasks) is job-wide, so it replaces
    # tasks-per-node × nodes rather than multiplying it. `max` because the two can
    # disagree — --ntasks-per-node is a per-node cap, not a total — and the larger
    # is the number of tasks Slurm will actually run.
    task_units = max(ntasks_total or 0, tasks * max(nodes, 0))
    su = cpus * task_units * hours
    return _with_array_total(su, array_tasks)


def _fmt_estimate(value: float) -> str:
    if value < 1:
        return f"{value:.2f}"
    if value < 100:
        return f"{value:.1f}"
    return f"{value:,.0f}"


def estimate_gpu_hours(gpus: int, time_limit: str, nodes: int = 1,
                       gpu_format: str | None = None,
                       ntasks_per_node: int | None = None,
                       array_tasks: int | None = None,
                       ntasks_total: int | None = None) -> str:
    """Estimate a job's GPU cost in GPU-hours, or "" for a CPU-only job.

    GPU allocation is per-node for ``--gres``/``--gpus-per-node`` (and the
    constraint form), per-task for ``--gpus-per-task``, and job-wide for
    ``--gpus`` — so the multiplier has to follow the chosen ``gpu_format``, not
    just the raw count. Reported next to the CPU-hours figure because on nearly
    every GPU site it is the GPU, not the core, that drives the bill.
    """
    try:
        gpus = int(gpus)
    except (TypeError, ValueError):
        return ""
    if gpus <= 0:
        return ""
    nodes = max(1, nodes or 1)
    if _time_is_unbounded(time_limit):
        return UNBOUNDED_ESTIMATE
    minutes = _parse_slurm_time_to_minutes(time_limit) if time_limit else 120.0
    if minutes <= 0:
        minutes = 120.0
    hours = minutes / 60.0
    fmt = (gpu_format or os.environ.get("SLURMATE_GPU_FORMAT", "gres_type")).lower()
    if fmt == "gpus":
        total = gpus                                    # job-wide total
    elif fmt == "gpus_per_task":
        tasks = ntasks_per_node if (ntasks_per_node and ntasks_per_node > 0) else 1
        # Per *task*, so an explicit --ntasks total governs here as well.
        total = gpus * max(ntasks_total or 0, tasks * nodes)
    else:                                               # gres_type / constraint / gpus_per_node
        total = gpus * nodes
    return _with_array_total(total * hours, array_tasks)
