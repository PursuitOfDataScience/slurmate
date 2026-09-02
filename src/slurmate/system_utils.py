from __future__ import annotations

import contextlib
import difflib
import getpass
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# These rows carry every key ``fetch_partitions`` produces, because a key a real
# cluster always supplies and the fixture omits makes a whole check unreachable in
# mock mode — and so in ``--demo`` and in the test suite. Two were missing and
# each hid a check that works live: no ``is_default`` on any row made the "no
# --partition given, use the site default" path (with its own limit, queue, ETA
# and memory consequences) unreachable, and no ``gpus_per_node`` meant a 99-GPU
# request on a 4-GPU-per-node partition passed in silence here while a real
# cluster warned. TestMockClusterHasTheShapeOfARealOne pins the key set so the
# next divergence is caught rather than discovered on another cluster.
MOCK_PARTITIONS: list[dict[str, Any]] = [
    {"name": "cpu-shared", "nodes": 100, "nodes_up": 100, "state": "up", "cpus_per_node": 32, "mem_per_node_mb": 131072, "heterogeneous": False, "gpu_types": [], "has_gpu": False, "gpus_per_node": 0, "timelimit": "02:00:00", "is_public": True, "is_default": True},
    {"name": "cpu-highmem", "nodes": 20, "nodes_up": 20, "state": "up", "cpus_per_node": 48, "mem_per_node_mb": 524288, "heterogeneous": False, "gpu_types": [], "has_gpu": False, "gpus_per_node": 0, "timelimit": "12:00:00", "is_public": True, "is_default": False},
    {"name": "gpu-shared", "nodes": 10, "nodes_up": 10, "state": "up", "cpus_per_node": 16, "mem_per_node_mb": 196608, "heterogeneous": False, "gpu_types": ["a100", "v100"], "has_gpu": True, "gpus_per_node": 4, "timelimit": "04:00:00", "is_public": True, "is_default": False},
    {"name": "gpu-highend", "nodes": 4, "nodes_up": 4, "state": "up", "cpus_per_node": 32, "mem_per_node_mb": 262144, "heterogeneous": False, "gpu_types": ["h100"], "has_gpu": True, "gpus_per_node": 8, "timelimit": "24:00:00", "is_public": True, "is_default": False},
    {"name": "debug", "nodes": 2, "nodes_up": 2, "state": "up", "cpus_per_node": 8, "mem_per_node_mb": 32768, "heterogeneous": False, "gpu_types": [], "has_gpu": False, "gpus_per_node": 0, "timelimit": "01:00:00", "is_public": True, "is_default": False},
]

MOCK_CONDA_ENVS = ["base", "pytorch", "tensorflow", "jax", "my_project"]

MOCK_GPU_TYPES = ["a100", "h100", "v100", "a40", "rtx6000", "h200", "l40s"]

MOCK_MODULES = ["python/anaconda", "cuda/11.8", "cuda/12.1", "gcc/9.3.0", "openmpi/4.1.1"]

MOCK_ACCOUNTS = ["my_lab", "training", "default"]


_RUN_TIMEOUT = 30

# Advisory cluster facts — the ones used only to *validate* or *enrich*, where a
# failed lookup is already designed to fall through silently. Waiting the full 30 s
# for one of those buys nothing: the answer will be discarded either way, and six
# of them in series meant a hung controller froze a --dry-run for ~170 s with no
# output at all. Healthy latency here is 0.1-0.5 s each, so 10 s is 20-100x
# headroom: a slow-but-working controller is still answered, a dead one is not
# waited on.
_ADVISORY_TIMEOUT = 10

# Total wall clock every Slurm query in one *phase* may spend between them.
#
# The per-call bound above is correct and was the only one there was, so against
# an unresponsive `slurmctld` the worst case was their SUM: one `--dry-run`
# makes four invocations (2x sinfo, squeue, sbatch --test-only), each granted the
# full 30 s independently, so ~120 s of silence with the process looking hung --
# and every fix that teaches slurmate to consult more of the cluster (the
# SelectType read, the association check, the QOS MaxWall) adds another 30 s to
# it.
#
# rapidu solved this in `quota.py:_budget` and wrote down why, and the last
# clause of that docstring applies here verbatim: *"a hanging `lfs quota` is not
# exotic; it is what a Lustre client does when an MDS is degraded, which is the
# same afternoon someone reaches for this tool."* A slow controller is exactly
# when somebody opens a submission helper to see what the cluster looks like.
#
# 45 s: comfortably more than the ~2 s a healthy controller needs for all four
# calls, and short enough that a dead one does not look like a hang. Overridable
# because a site with a genuinely slow controller should be able to wait longer
# rather than lose the enrichment.
DEFAULT_TOTAL_TIMEOUT = 45.0
_TOTAL_TIMEOUT_ENV = "SLURMATE_TIMEOUT"

#: Wall-clock instant every query in the current phase must finish by, or None
#: when no phase has been opened -- in which case each call keeps its own
#: timeout, which is the right behaviour for a long-lived interactive session
#: and for a direct API caller.
_DEADLINE: float | None = None
#: The budget the current phase was opened with, so the "budget exceeded"
#: message names the figure that actually applied rather than re-deriving the
#: default -- which differs whenever a caller passed one explicitly.
_PHASE_BUDGET: float = DEFAULT_TOTAL_TIMEOUT


def total_timeout() -> float:
    """The per-phase budget, from ``$SLURMATE_TIMEOUT`` or the default."""
    raw = os.environ.get(_TOTAL_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            asked = float(raw)
        except ValueError:
            logger.debug(f"ignoring non-numeric {_TOTAL_TIMEOUT_ENV}={raw!r}")
        else:
            if asked > 0:
                return asked
    return DEFAULT_TOTAL_TIMEOUT


@contextlib.contextmanager
def slurm_deadline(total: float | None = None) -> Iterator[float]:
    """Bound the *total* time Slurm queries may take inside this block.

    Opened around each batch of queries rather than once per process, because
    the wizard is interactive: a user takes minutes to answer, and a
    process-wide deadline would expire while they were reading the screen and
    then report the cluster as unreachable.  Nested blocks keep the OUTER
    deadline -- a phase cannot extend its own budget by opening another one.
    """
    global _DEADLINE, _PHASE_BUDGET
    if _DEADLINE is not None:
        yield _DEADLINE
        return
    budget = total if total is not None else total_timeout()
    _DEADLINE = time.monotonic() + budget
    _PHASE_BUDGET = budget
    try:
        yield _DEADLINE
    finally:
        _DEADLINE = None


def _budget(timeout: float) -> float:
    """``timeout``, capped by whatever is left of the phase deadline.

    Ported from `rapidu.quota._budget`, including its shape: a call that falls
    entirely past the deadline gets 0 and returns the existing timeout message,
    so the partition list still renders and only the optional enrichment is
    skipped. That is the right degradation for a tool whose core output is the
    script.
    """
    if _DEADLINE is None:
        return timeout
    return max(0.0, min(timeout, _DEADLINE - time.monotonic()))


def _run_command(
    cmd: list[str], timeout: int = _RUN_TIMEOUT, stdin: str | None = None
) -> tuple[str, str, int]:
    allowed = _budget(timeout)
    if allowed <= 0:
        # Not run at all. `subprocess.run(timeout=0)` starts the process and
        # then kills it, which is a fork, an exec and a controller connection
        # spent to learn nothing -- and against a wedged controller those are
        # the expensive part.
        return "", (
            f"Slurm queries exceeded the {_PHASE_BUDGET:.0f}s total budget "
            f"(${_TOTAL_TIMEOUT_ENV} to change it); skipped {cmd[0]}"
        ), -1
    try:
        # Force UTF-8 decoding with a lossy fallback: under a C/POSIX locale
        # `text=True` would otherwise decode with ASCII and raise on any
        # non-ASCII byte in the command output (crashing the wizard/batch run).
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=allowed,
            encoding="utf-8", errors="replace", input=stdin,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {allowed:.0f}s", -1
    except OSError as e:
        # A Slurm binary that is present but not runnable (bad arch, permission,
        # missing loader) raises here rather than being caught by shutil.which;
        # return a non-zero rc so callers fall back to mock data instead of
        # crashing with a traceback.
        return "", str(e), -1


def is_mock() -> bool:
    """Whether demo data is in force — public alias of :func:`_force_mock`.

    Callers need this to *label* what they render. Synthetic partitions, queue
    depth and ETA presented with no marker are measurement-shaped fiction, and
    the switch is an environment variable, so it arrives from a CI wrapper, a
    container image or a stale ``export`` rather than being chosen deliberately.
    """
    return _force_mock()


# Cluster facts that cannot change during one invocation: the partition name
# list, the caller's accounts and QoS, node features, the select plugin and
# MaxArraySize. Each is queried by more than one code path — the batch path's
# fatal checks and the shared site checks both ask — so an uncached lookup ran
# `sacctmgr show assoc` twice per run, and the report notes sacctmgr is slow
# enough on a busy controller to be worth skipping. Memoised per process; a
# single slurmate run is short enough that staleness is not a concern.
_CLUSTER_CACHE: dict[str, Any] = {}


# The last error Slurm itself gave for a cluster query. The report's RD-2/SW-7
# lesson is that the diagnosis is usually sitting in a stream nobody read: when
# `sinfo` fails it says *why* ("Unable to contact slurm controller (connect
# failure)"), and reporting a generic "sinfo failed" throws that away.
_LAST_CLUSTER_ERROR: str = ""


def last_cluster_error() -> str:
    """Slurm's own words for the most recent failed cluster query, or ""."""
    return _LAST_CLUSTER_ERROR


def _note_cluster_error(stderr: str) -> None:
    global _LAST_CLUSTER_ERROR
    # First line only: Slurm prefixes each with the tool name, and the first is
    # the cause (later ones are usually consequences).
    first = next((ln.strip() for ln in (stderr or "").splitlines() if ln.strip()), "")
    if first:
        _LAST_CLUSTER_ERROR = first


def reset_cluster_cache() -> None:
    """Forget memoised cluster facts (for tests, which vary the mocked output)."""
    global _LAST_CLUSTER_ERROR
    _CLUSTER_CACHE.clear()
    _LAST_CLUSTER_ERROR = ""


def _cached_cluster_fact(key: str, compute: Any) -> Any:
    if key not in _CLUSTER_CACHE:
        _CLUSTER_CACHE[key] = compute()
    return _CLUSTER_CACHE[key]


def fetch_all_partition_names() -> set[str]:
    """Memoised; see :func:`_cached_cluster_fact`."""
    value: set[str] = _cached_cluster_fact("fetch_all_partition_names", _fetch_all_partition_names_uncached)
    return value


def fetch_user_accounts() -> list[str]:
    """Memoised; see :func:`_cached_cluster_fact`."""
    value: list[str] = _cached_cluster_fact("fetch_user_accounts", _fetch_user_accounts_uncached)
    return value


def fetch_known_qos() -> list[str]:
    """Memoised; see :func:`_cached_cluster_fact`."""
    value: list[str] = _cached_cluster_fact("fetch_known_qos", _fetch_known_qos_uncached)
    return value


def fetch_node_features() -> set[str] | None:
    """Memoised; see :func:`_cached_cluster_fact`."""
    value: set[str] | None = _cached_cluster_fact(
        "fetch_node_features", _fetch_node_features_uncached
    )
    return value


def fetch_partition_node_maxima(partition: str) -> tuple[int | None, int | None]:
    """``(max_cpus, max_mem_mb)`` across a partition's nodes, or ``(None, None)``.

    SM-27. ``sinfo``'s aggregate row reports a heterogeneous partition's
    *smallest* node and marks it with ``+`` ("at least this much"). Recording that
    number and then calling it "the partition limit" turns the floor into a
    ceiling, so a request that fits a larger node is warned about: measured on
    midway3, ``test`` reports ``32+|184320+`` while its nodes actually reach 256
    CPUs and 2321910 MB, and 20 of its 87 partitions emit the ``+`` at all.

    One extra ``sinfo`` for the partition the user actually named — and only when
    the aggregate row carried a ``+``, so a homogeneous site pays nothing.

    ``(None, None)`` for "could not tell", which callers must treat as unknown
    rather than as zero; slurmwatch's equivalent returns 0 and guards every use,
    and an unresolved partition there stays unresolved rather than becoming wrong.
    """
    name = str(partition or "").strip().rstrip("*")
    if not name or not is_tool_available("sinfo") or _force_mock():
        return None, None

    def _compute() -> tuple[int | None, int | None]:
        stdout, _, rc = _run_command(
            ["sinfo", "-h", "-N", "-p", name, "-o", "%c|%m"],
            timeout=_ADVISORY_TIMEOUT,
        )
        if rc != 0:
            return None, None
        max_cpus: int | None = None
        max_mem: int | None = None
        for line in stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            # _safe_int / _parse_mem_to_mb already tolerate the "+" and other
            # decoration, and a row that parses to 0 is not evidence of anything.
            cpus = _safe_int(parts[0])
            mem = _parse_mem_to_mb(parts[1])
            if cpus > 0:
                max_cpus = cpus if max_cpus is None else max(max_cpus, cpus)
            if mem > 0:
                max_mem = mem if max_mem is None else max(max_mem, mem)
        return max_cpus, max_mem

    value: tuple[int | None, int | None] = _cached_cluster_fact(
        f"node_maxima:{name}", _compute
    )
    return value


def fetch_select_type() -> str:
    """Memoised; see :func:`_cached_cluster_fact`."""
    value: str = _cached_cluster_fact("fetch_select_type", _fetch_select_type_uncached)
    return value


def fetch_max_array_size() -> int | None:
    """Memoised; see :func:`_cached_cluster_fact`."""
    value: int | None = _cached_cluster_fact("fetch_max_array_size", _fetch_max_array_size_uncached)
    return value


def _force_mock() -> bool:
    return os.environ.get("SLURMATE_MOCK", "").lower() in ("1", "true", "yes")


def is_tool_available(name: str) -> bool:
    if _force_mock():
        return False
    return shutil.which(name) is not None


def _safe_int(raw: str) -> int:
    match = re.search(r"\d+", raw.strip())
    return int(match.group(0)) if match else 0


def _normalize_null(raw: str) -> str:
    value = raw.strip()
    return "" if value.lower() in {"", "(null)", "null", "-", "n/a"} else value


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_mem_to_mb(raw: str) -> int:
    # `sinfo %m` (without -e) reports the minimum node memory with a trailing
    # "+" when a partition's nodes differ (e.g. "515000+"). Strip it so the
    # min value is used, mirroring how _safe_int already tolerates "+" for %c —
    # otherwise the memory-over-limit warning is silently disabled for every
    # heterogeneous partition.
    value = raw.strip().upper().rstrip("+")
    if not value or value == "0":
        return 0
    # The optional trailing "B" matches validate_memory, which accepts "16GB"
    # because sbatch does. Leaving it out here was worse than the refusal it
    # replaced: the value validated, normalised to a correct directive, and then
    # read as **0 MB** in every comparison — so a 64 GB request on a 16 GB
    # partition produced no warning at all. One grammar, three functions.
    match = re.match(r"^(\d+(?:\.\d+)?)([KMGTP])B?(?:[NC])?$", value)
    if match:
        num = float(match.group(1))
        scale = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2, "P": 1024 ** 3}
        mb = int(num * scale[match.group(2)])
        # A positive size below 1 MB (e.g. "1K") would truncate to 0 and read as
        # "unknown"; clamp to 1 MB so it stays a real, if tiny, value.
        return mb if mb > 0 or num == 0 else 1
    # A bare integer is megabytes. Anything else is malformed ("16 G",
    # "1.5.5G", "16GiB") — return 0 (unknown) rather than a misleading partial
    # like "16", which would masquerade as a tiny valid value in limit checks.
    if value.isdigit():
        return int(value)
    return 0


def validate_memory(value: str) -> bool:
    """Validate memory value.

    Accepts formats:
    - Plain digits: "16"
    - With units: "16G", "16g", "512M", "1T"
    - With `sacct`'s N/C suffix: "16GN", "16GC" -- tolerated on INPUT only.
      `sbatch --mem` does not take it (measured: rc=255, "Invalid --mem
      specification"), so this is deliberate leniency towards a value copied out
      of `sacct`'s ReqMem column; `normalize_memory` strips it before the
      directive is written. Do not read this line as Slurm grammar.

    Zero is accepted, in every unit spelling. ``--mem=0`` is documented Slurm and
    means *all the memory on the node* — the whole-node idiom — and it was
    measured accepted here as ``0``, ``0K``, ``0M``, ``0G`` and ``0T``. Rejecting
    it (which this function used to do, deliberately, as "not a valid size") left
    no way to express that request at all: ``--memory ''``/``none`` omits ``--mem``
    entirely, which gets the *site default*, a different thing.

    Rejects:
    - Empty
    - Invalid formats
    """
    v = value.strip()
    if not v:
        return False
    if v.isdigit():
        return True
    # Accepts a unit suffix and optional Slurm N/C. Units are K/M/G/T only: that
    # is all `sbatch --mem` documents, and it rejects anything else client-side
    # ("sbatch: error: Invalid --mem specification" for 16P), so accepting "P"
    # here only let a doomed value through to the submit call.
    # A trailing "B" after the unit is accepted, because sbatch accepts it:
    # measured, --mem=16KB/16MB/16GB/16TB all parse (any case), while --mem=16B
    # (no unit) and --mem=16GiB both get "Invalid --mem specification". Rejecting
    # "16GB" was a false refusal of the most natural way to write memory, and
    # normalize_memory drops the B so the emitted directive stays canonical.
    return bool(re.match(r"^(\d+(?:\.\d+)?)([KMGT])B?(?:[NC])?$", v.upper()))


# Slurm's accepted --time grammar: six shapes, each field a run of digits.
#   minutes | minutes:seconds | hours:minutes:seconds |
#   days-hours | days-hours:minutes | days-hours:minutes:seconds
#
# **No field is range- or width-limited**, because sbatch does not limit them.
# This block used to cap minute/second fields at [0-5]?\d and the days-hours
# field at \d{1,2}, on the stated grounds that "obviously out-of-range values
# like '1:60:60' or '1-99:99:99' are still rejected client-side". They are not.
# Measured against this controller (Slurm 20.11.8), sbatch takes every one of
# them and carries the value straight into the time limit:
#
#   sbatch --test-only -t 0:99         -> PASSED  (--time-min 2    ok, 3 refused)
#   sbatch --test-only -t 1:60         -> PASSED  (--time-min 2    ok, 3 refused)
#   sbatch --test-only -t 1:60:60      -> PASSED  (--time-min 121  ok, 122 refused)
#   sbatch --test-only -t 1-99:99:99   -> PASSED  (--time-min 7481 ok, 7482 refused)
#   sbatch --test-only -t 1-100        -> PASSED
#   sbatch --test-only -t 999:999:999  -> PASSED
#
# and `_parse_slurm_time_to_minutes` already returns exactly the figure sbatch
# arrived at in each case (1.65 / 2.0 / 121.0 / 7480.65 minutes). So the cap was a
# pure false refusal: `slurmate --time 1:60` exited 1 on a request the scheduler
# would have accepted and measured identically -- the same mistake the
# `--time UNLIMITED` and `--mem 0` refusals nearby were fixed for.
#
# What sbatch *does* reject is a shape error, never a magnitude: a fourth field
# ("1:0:0:0"), an empty field ("1:", ":5", "1-:2"), a sign ("+5", "1:-2"), a
# non-digit ("1.5", "10m"), or surrounding whitespace. The six anchored patterns
# below still refuse all of those.
_TIME_PATTERNS = (
    r"^\d+$",                  # minutes
    r"^\d+:\d+$",              # minutes:seconds
    r"^\d+:\d+:\d+$",          # hours:minutes:seconds
    r"^\d+-\d+$",              # days-hours
    r"^\d+-\d+:\d+$",          # days-hours:minutes
    r"^\d+-\d+:\d+:\d+$",      # days-hours:minutes:seconds
)


# The word spellings sbatch accepts for an unlimited --time, measured against a
# live client rather than guessed: `UNLIMITED` and `INFINITE` parse (they reach
# the controller and are judged on policy like any other value), while the
# obvious-looking abbreviation `inf` does **not** — sbatch rejects it outright
# with "Invalid --time specification". So this set is deliberately not "words
# that mean infinity"; it is the two Slurm actually takes. Case-insensitive.
SLURM_UNLIMITED_TIME_WORDS = frozenset({"unlimited", "infinite"})


def time_request_is_unbounded(value: str) -> bool:
    """Whether a *requested* ``--time`` means "no limit at all".

    Both spellings count: the words above, and any time-shaped value that parses
    to zero (Slurm documents a zero limit as "no time limit be imposed"), which
    covers ``0``, ``00:00:00`` and ``0-00:00:00``.

    Only a time-*shaped* string may parse to zero, or "not-a-time" would parse to
    0 and be labelled unbounded — conflating unparseable with unlimited.

    An **absent** limit is deliberately not unbounded: the job takes the partition
    or site default.
    """
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text in SLURM_UNLIMITED_TIME_WORDS:
        return True
    if not re.fullmatch(r"[\d:\-]+", text):
        return False
    return _parse_slurm_time_to_minutes(text) == 0


def validate_time(val: str) -> bool:
    """Validate a time limit string against Slurm's accepted --time formats.

    Includes the word forms. Rejecting ``--time=UNLIMITED`` was a *false*
    refusal: sbatch takes it, so slurmate was blocking a request the scheduler
    would have accepted — the mirror image of the mistakes this module usually
    guards against, and worse in a way, because there is no cluster on which it
    was right. ``inf`` stays rejected because sbatch rejects it too; accepting
    that would trade this bug for its inverse.
    """
    v = val.strip()
    if not v:
        return True
    if v.lower() in SLURM_UNLIMITED_TIME_WORDS:
        return True
    return any(re.match(p, v) for p in _TIME_PATTERNS)


#: What a rejected value should have looked like, phrased for a terminal.
#:
#: Beside the validators rather than at the call sites, so the sentence cannot
#: drift from the grammar it describes -- the time forms are read straight out of
#: `_TIME_PATTERNS`' own comments and the unlimited words out of the set.
#:
#: These exist because three hard rejections (`--memory`, `--mem-per-cpu`,
#: `--time`) printed the offending value and exited 1 without saying what a valid
#: one looks like. Every other refusal in this package names the remedy -- an
#: unknown partition lists the real ones, a rejected flag names `--force` -- so
#: these three were the odd ones out, and they are the errors a first-time user is
#: most likely to hit.
#: Measured against this controller rather than written from the docstrings
#: below, because those disagreed with sbatch: they present the `N`/`C` suffix as
#: Slurm grammar, and `sbatch --mem=16GN` answers
#: "sbatch: error: Invalid --mem specification" (rc=255). Advice on an error path
#: has to name a form that works, so the suffix is not offered here -- slurmate
#: tolerates it on input and `normalize_memory` strips it, which is a kindness to
#: anyone pasting a value out of `sacct`'s ReqMem, not a spelling to recommend.
#: `sbatch --mem` documents its units as [K|M|G|T]; the trailing `B` is measured
#: accepted (`16GB`), and a fraction is not (`1.5G` -> rc=255).
MEMORY_FORMS = (
    "a number of megabytes (`4096`), or a whole number with a unit "
    "(`16G`, `512M`, `1T`; lower case works, and a trailing B is fine: `16GB`). "
    "`0` means \"all the memory on the node\""
)


def time_forms() -> str:
    """Slurm's ``--time`` spellings, listed from the patterns that accept them."""
    words = "/".join(sorted(w.upper() for w in SLURM_UNLIMITED_TIME_WORDS))
    return (
        "minutes (`90`), `MM:SS`, `HH:MM:SS`, `D-HH`, `D-HH:MM`, "
        "`D-HH:MM:SS`, or " + words
    )


def normalize_memory(value: str) -> str:
    """Normalize memory value to a standard format.

    Returns:
    - Plain digits prefixed with "M": "16" -> "16M"
    - Units already present: "16G" -> "16G"
    - STRIPS `sacct`'s per-node/per-cpu suffix: "16GN"/"16GC" -> "16G". This
      previously claimed to preserve it, and preserving it would emit a directive
      sbatch refuses outright ("Invalid --mem specification"). The suffix belongs
      to `sacct`'s ReqMem *output* (`4Gn`, `4Gc`), not to `--mem` input, so a
      value pasted from there is accepted and canonicalised rather than rejected.
    - Folds a fraction into the next unit down: "1.5G" -> "1536M", because sbatch
      takes only whole numbers.
    """
    v = value.strip().upper()
    if not v:
        return ""
    # Every zero spelling ("0", "0G", "0M") is the same request — all the memory
    # on the node — so emit the documented bare form rather than "0M", which
    # reads like a request for nothing.
    if re.match(r"^0+(?:\.0+)?[KMGT]?B?[NC]?$", v):
        return "0"
    # Plain digits: append M
    if v.isdigit():
        return f"{v}M"
    # Already has unit: return as-is, but drop any trailing Slurm N/C suffix —
    # `sbatch --mem` accepts only a K/M/G/T unit, so "16GN" would be rejected.
    m = re.match(r"^(\d+(?:\.\d+)?)([KMGTP])B?(?:[NC])?$", v)
    if m:
        # `sbatch --mem` requires an INTEGER magnitude, so a fractional value like
        # "1.5G" — which validate_memory accepts — would be rejected at submit.
        # Convert it to whole megabytes ("1.5G" -> "1536M") so a value that
        # validates always normalizes to a directive Slurm accepts.
        if "." in m.group(1):
            # Parse the *matched* number and unit, not the raw string: a trailing
            # "B" or Slurm "N"/"C" suffix is not something _parse_mem_to_mb reads,
            # so "1.5GB" and "1.5GN" both came back as 0 — turning a 1.5 GiB
            # request into "--mem=0M", which Slurm reads as *all* the node's
            # memory. The magnitude and unit are already in hand; use them.
            return f"{_parse_mem_to_mb(m.group(1) + m.group(2))}M"
        return f"{m.group(1)}{m.group(2)}"
    # Invalid but return it anyway (validation should catch this)
    return v


# Known GPU model names, checked before any shape heuristic. This is what makes
# the short models (t4, l4, a30, a40, k80, mi50 …) recognizable without also
# accepting two-character rack/chassis labels that happen to share their shape.
_KNOWN_GPU_MODELS = frozenset({
    # NVIDIA datacenter
    "a2", "a10", "a16", "a30", "a40", "a100", "a800", "h20", "h100", "h200",
    "h800", "b100", "b200", "gb200", "gh200", "v100", "v100s", "p100", "p40",
    "p4", "k80", "k40", "k20", "l4", "l40", "l40s", "t4", "t10",
    # Fermi / Kepler / Maxwell. Long EOL, still in service on multi-year
    # clusters' older partitions — and none of them satisfies the shape rule
    # below (two-digit or bare-"m" families), so without these the detector
    # falls through to guessing on exactly the nodes that have them.
    "m2050", "m2070", "m2075", "m2090", "k10", "k20c", "k20m", "k20x",
    "k40c", "k40m", "k40s", "k40st", "k40t", "k80m", "m40", "m60", "m6", "m10",
    # NVIDIA workstation / consumer (as clusters label them)
    "rtx4000", "rtx5000", "rtx6000", "rtx8000", "a4000", "a4500", "a5000",
    "a6000", "rtx2080", "rtx3080", "rtx3090", "rtx4090", "titanv", "titanx",
    "titanrtx",
    # AMD Instinct
    "mi25", "mi50", "mi60", "mi100", "mi210", "mi250", "mi250x", "mi300",
    "mi300a", "mi300x", "mi325x",
    # Intel
    "pvc", "max1100", "max1550", "gaudi", "gaudi2", "gaudi3",
})

# A token *shaped* like a GPU model name: a known GPU-family letter prefix
# followed by at least THREE digits (a100, h200, v100, p100, b200, mi250, gh200,
# gb200, rtx6000, …), so an unreleased future model is still detected. The digit
# run must be 3+ because two-character labels are ambiguous with the rack /
# chassis / blade tags clusters put in node features ("b12", "t2", "p2") — those
# used to win here purely by appearing earlier in the feature list than the real
# model. Shorter real models are covered by _KNOWN_GPU_MODELS above.
_GPU_MODEL_RE = re.compile(
    r"^(?:a|h|v|l|t|p|k|b|rtx|gtx|mi|gh|gb|quadro|tesla)\d{3,}[a-z]?$",
    re.IGNORECASE,
)

# CPU model designations that start with a letter, carry digits, and are not
# caught by the vendor/codename lists: Intel Xeon E3/E5/E7-NNNN and the older
# X/L-series parts. These are what a node's feature string actually holds next
# to the GPU ("tc,e5-2670,160G,ib,m2090,gpu"), so the last-resort scan reaches
# them first and would otherwise return a CPU as the GPU model. "w" is
# deliberately absent: Radeon Pro W6800/W7900 are real GPUs.
# A token shaped like *some* model designation, without assuming a known GPU
# family letter: letters then 3+ digits. Used by the last-resort scan, where the
# alternative is returning whatever appeared first. Three digits rather than one
# because "b12"/"t2"/"n1" are blade/rack/node labels and every GPU model with
# fewer than three digits (l4, t4, a2, k80, m40, mi25 …) is in
# _KNOWN_GPU_MODELS and so never reaches the fallback.
_MODEL_SHAPE_RE = re.compile(r"^[a-z]+[-_]?\d{3,}[a-z0-9]*$", re.IGNORECASE)

_CPU_MODEL_RE = re.compile(
    r"^e[357]-?\d{3,}[a-z0-9]*$"       # e5-2670, e52670, e7-4820v2
    r"|^[xl]\d{4}[a-z]?$"              # X5650, L5520
    r"|^(?:gold|silver|bronze|platinum)-?\d+",
    re.IGNORECASE,
)

# Infrastructure tokens that are not GPU models but pass a naive filter: network
# fabric generations and adapters, rack/chassis/position labels, GPU *form
# factors*, and cooling tags. The pre-existing blocklist already carried
# "ib"/"opa"/"hdr" — these are the same convention's other spellings, which is
# how a partition whose features read "gold6248,avx512,hdr100,768g" came back
# with a GPU type of "hdr100".
_INFRA_TOKEN_RE = re.compile(
    r"^(?:sdr|ddr|qdr|fdr|edr|hdr|ndr|xdr)\d*$"          # InfiniBand generations
    # Fabric/NIC prefixes take a free tail, not just digits: a real cluster
    # labels its InfiniBand spine "ibspine-g20", which has a letter start, a
    # digit and no CPU shape, so the last-resort scan would return it as a GPU.
    r"|^(?:ib|opa|omnipath|roce|eth|ether|bond|enp|eno|mlx|ofed)[\w-]*$"  # fabrics/NICs
    r"|^(?:rack|racks|row|rk|pod|cab|cabinet|chassis|blade|shelf|slot|island|zone|cell|unit"
    r"|spine|leaf|tor|switch|sw)[\w-]*$"
    r"|^(?:sxm|sxm2|sxm3|sxm4|sxm5|pcie|nvlink|nvswitch|mig|dlc|lc|air|water|oam)$",
    re.IGNORECASE,
)

# CPU-generation tags that share a GPU-family letter prefix and would otherwise
# be misread as a GPU model: Intel Xeon "vN" (E5/E7-…-v2…v6) and IBM POWER "pN"
# (POWER8/9/10). No real GPU uses these exact tokens — V100/P100 etc. are
# multi-digit — so excluding them is safe.
_CPU_GEN_TOKENS = frozenset({"v2", "v3", "v4", "v5", "v6", "p8", "p9", "p10"})


def _parse_gpu_count(gres_raw: str) -> int:
    """GPUs a node advertises, from ``sinfo %G``; 0 when none or unreadable.

    Sums every ``gpu`` entry, because a node listing ``gpu:a100:2,gpu:v100:2`` can
    satisfy a request for four. Handles the count-only (``gpu:4``), typed
    (``gpu:a100:4``) and socket-annotated (``gpu:a100:4(S:0-1)``) spellings, and
    ignores ``shard``/``mps`` entries, which are slices of a GPU rather than
    another one.
    """
    text = _normalize_null(gres_raw)
    if not text:
        return 0
    total = 0
    for entry in text.split(","):
        entry = entry.split("(")[0].strip()
        if not entry.lower().startswith("gpu:"):
            continue
        parts = entry.split(":")
        # gpu:N or gpu:TYPE:N — the count is the last numeric field, and a typed
        # entry with no count at all ("gpu:a100") means one.
        counts = [int(x) for x in parts[1:] if x.isdigit()]
        total += counts[-1] if counts else 1
    return total


def _detect_gpu_type(features: str, gres: str, known_models: set[str] | None = None) -> str:
    """Extract GPU model name from sinfo output.

    Priority:
    1. Parse model from ``gpu:MODEL:N`` in GRES.
    2. If GRES is count-only (``gpu:N``), scan node features:
       a. When ``known_models`` is given, *prefer* a feature token that matches
          a model seen in a typed GRES elsewhere in the partition. This
          disambiguates nodes whose features list rack/filesystem labels
          *before* the GPU (e.g. ``rack5,gpfs,a40`` → ``a40``).
       b. A token that is a *known* GPU model name (``_KNOWN_GPU_MODELS``).
       c. A token *shaped* like a model name (family letter + 3-plus digits), so
          a model too new for the list is still found.
       d. Otherwise, fall back to a guess: a token *shaped* like a model at all
          (letters then 3+ digits) that is not a CPU designation, arch, fabric,
          rack or form-factor tag. This keeps detecting GPU types
          that only ever appear in features and never in a typed GRES, without
          returning a site's node-class tag as a GPU model.
    4. Returns ``"gpu"`` when the node has GPUs but nothing identifiable — the
       caller then offers no type at all, which is the right answer: a wrong
       ``--gpu-type`` is worse than none, because nothing prompts the user to
       check it.
    3. If GRES has no ``gpu:`` at all the node has no GPUs — return empty.

    The token's original case is always preserved: Slurm node features are
    case-sensitive (a node advertising ``a100`` is *not* matched by ``-C A100``),
    so a lowercased "model" would produce a constraint that matches nothing.
    """
    text = f"{features},{gres}"
    gres_match = re.search(r"gpu:([a-z0-9._-]+):\d+", text, re.IGNORECASE)
    if gres_match:
        candidate = gres_match.group(1).replace("_", "-")
        if candidate.lower() not in {"gpu", "mps", "shard"}:
            return candidate

    if "gpu:" not in text.lower():
        return ""

    tokens = [t.strip() for t in re.split(r"[,/ ]+", features) if t.strip()]

    # Prefer a feature token corroborated by a typed GRES elsewhere.
    if known_models:
        known_lower = {m.lower() for m in known_models}
        for token in tokens:
            if token.lower() in known_lower:
                return token

    # Positive match, strongest first: an exact known model name anywhere in the
    # feature list beats everything, so "b12,a100" resolves to a100 rather than to
    # the rack label that merely appears earlier.
    for token in tokens:
        if len(token) < 15 and token.lower() in _KNOWN_GPU_MODELS:
            return token

    # Then a token shaped like a model name (family letter + 3-plus digits: a100,
    # h200, v100, rtx6000, mi250, gh200, b200 …). Far more reliable than negative
    # filtering and, crucially, wins over a CPU vendor/codename token that happens
    # to appear first in the features list.
    for token in tokens:
        if token.lower() in _CPU_GEN_TOKENS:
            continue
        # Apply the same length sanity cap as the negative branch below, so a
        # pathologically long feature token (e.g. a concatenated garbage string)
        # can't be returned verbatim as a GPU "model".
        if len(token) >= 15:
            continue
        if _INFRA_TOKEN_RE.match(token):
            continue
        # An explicit CPU designation outranks the shape rule: the NVIDIA L
        # family is L4/L40/L40S, so a four-digit "l5520" is a Xeon, not a GPU,
        # and it satisfies the family-letter-plus-digits shape by coincidence.
        if _CPU_MODEL_RE.match(token):
            continue
        if _GPU_MODEL_RE.match(token):
            return token

    # Fall back to negative filtering: reject obvious non-GPU tokens and return
    # the first plausible one. Never drops a real GPU type that lives only in
    # the features string.
    #
    # Two positive requirements are applied first, because "everything we did not
    # think to exclude" is not a GPU model. A site's node-class tag ("tc", "lc")
    # sat first in the feature string and won on a real cluster, producing
    # `--gres=gpu:tc:1`, which Slurm refuses; the real model was never offered.
    # Every GPU model is letters followed by a run of digits, and a site's
    # class/rack/cooling tags are not.
    for token in tokens:
        if not re.match(r"[a-zA-Z]", token):
            continue
        if not _MODEL_SHAPE_RE.match(token):
            continue
        if _CPU_MODEL_RE.match(token):
            continue
        if token.lower() in _CPU_GEN_TOKENS:
            continue
        if len(token) >= 15:
            continue
        # Network fabric / rack position / form-factor / cooling labels: not GPUs.
        if _INFRA_TOKEN_RE.match(token):
            continue
        if re.match(
            r"(?:gold|xeon|epyc|ryzen|atom|i[3579]|avx\d*|sse\d*|fma)",
            token, re.IGNORECASE
        ):
            continue
        if re.match(
            r"(?:skylake|cascadelake|icelake|sapphirerapids|broadwell|haswell|zen\d*)",
            token, re.IGNORECASE
        ):
            continue
        if token.lower() in {
            "ssd", "nvme", "ib", "opa", "hdr", "hdd",
            "scratch", "fat", "thin", "gpu", "cpu", "mem", "node",
            # Bare CPU vendor / microarch-codename tokens clusters put in node
            # features, ahead of the GPU model. Without these the negative
            # filter would return e.g. "intel"/"rome" as the GPU type.
            "intel", "amd", "arm",
            "rome", "milan", "genoa", "naples", "cascade",
            "sandybridge", "ivybridge", "nehalem", "westmere",
            # Spelled-out IBM POWER CPUs (the short p8/p9/p10 forms are in
            # _CPU_GEN_TOKENS; "power9" etc. would otherwise pass as a GPU model).
            "power", "power8", "power9", "power10",
        }:
            continue
        return token

    return "gpu"


def _extract_token(line: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^ ]+)", line)
    return match.group(1).strip() if match else ""


def _parse_slurm_time_to_minutes(time_str: str) -> float:
    value = time_str.strip()
    if not value or value in {"UNLIMITED", "NOT_SET", "N/A", "INVALID"}:
        return 0.0
    if "-" in value:
        day_part, rest = value.split("-", 1)
        parts = rest.split(":")
        hours = _safe_int(parts[0])
        minutes = _safe_int(parts[1]) if len(parts) > 1 else 0
        seconds = _safe_int(parts[2]) if len(parts) > 2 else 0
        return _safe_int(day_part) * 1440 + hours * 60 + minutes + seconds / 60.0
    parts = value.split(":")
    if len(parts) == 3:
        return _safe_int(parts[0]) * 60 + _safe_int(parts[1]) + _safe_int(parts[2]) / 60.0
    if len(parts) == 2:
        return _safe_int(parts[0]) + _safe_int(parts[1]) / 60.0
    return float(_safe_int(parts[0])) if parts else 0.0


def array_task_count(spec: str) -> int | None:
    """How many tasks an ``--array`` spec launches, or ``None`` when unknowable.

    The cost of an array job is per-task cost × task count, and reporting the
    per-task figure for a 1000-task array understates it a thousandfold — in the
    direction that matters, since it tells the user an enormous job is cheap.

    The ``%N`` throttle is deliberately ignored: it caps how many run *at once*,
    not how many run, so it changes the wall-clock and not the bill. A bare
    ``%N`` with no indices (which Slurm accepts) carries no index count at all, so
    it returns ``None`` rather than guessing 1.
    """
    text = str(spec or "").strip()
    if not text or not validate_array_spec(text):
        return None
    body, _sep, _throttle = text.partition("%")
    body = body.strip()
    if not body:
        return None
    total = 0
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        range_part, _c, step_part = part.partition(":")
        step = _safe_int(step_part) if step_part else 1
        if step <= 0:
            return None
        if "-" in range_part:
            low_s, _d, high_s = range_part.partition("-")
            low, high = _safe_int(low_s), _safe_int(high_s)
            if high < low:
                return None
            total += (high - low) // step + 1
        else:
            total += 1
    return total or None


def _slurm_reads_as_index_number(text: str) -> bool:
    """Whether Slurm's array parser reads ``text`` as a number.

    ``strtol`` semantics, which is what the controller uses: whitespace *before*
    the digits is skipped, whitespace *after* them is a parse error. Measured —
    ``1- 5`` and ``1-10: 2`` verify, ``1 -5`` and ``1-10 :2`` do not.
    """
    return text.lstrip(" \t").isdigit()


def _slurm_would_accept_array_spec(spec: str) -> bool:
    """Whether *sbatch* takes this spec - a laxer question than slurmate's check.

    Deliberately not :func:`validate_array_spec`, which is slurmate's own shape
    check and stricter on purpose. Written from ``sbatch --test-only`` probes
    against a live controller, because Slurm is lax and strict in places an
    intuition-built reimplementation gets backwards:

    * **Empty entries are dropped.** ``1,,3``, ``,1,3``, ``1,3,``, ``1-10,,2-3``
      and even a bare ``,`` all verify.
    * **The text after the first ``%`` is never checked.** ``1-10%``,
      ``1-10%%4``, ``1-10%abc``, ``1-10%4x``, ``1-10%-4``, ``1-10%4%5`` verify.
      A malformed *index* is still refused whatever the throttle says, which is
      why ``10-1%%4`` fails for the range and not for the ``%``.
    * **A step needs a range to step through.** ``1,2-4:2`` verifies but ``3:2``
      and ``1,3:2`` do not - and :func:`validate_array_spec` accepts both, so
      delegating to it would claim the controller takes ``1,,3:2``, which it
      refuses.
    * **Trailing whitespace is fatal, leading whitespace is not** - see
      :func:`_slurm_reads_as_index_number`. ``validate_array_spec`` strips each
      entry, so delegating there would also claim ``1 ,,3`` is accepted.

    Exists so :func:`array_spec_reason` can tell a refusal the controller agrees
    with from one it contradicts, and never call the second kind "invalid".
    """
    body = str(spec or "").strip().partition("%")[0]
    for entry in body.split(","):
        if entry == "":
            continue                       # Slurm drops empty entries
        index, colon, step = entry.partition(":")
        if colon and not (_slurm_reads_as_index_number(step)
                          and int(step) > 0 and "-" in index):
            return False
        bounds = index.split("-")
        if len(bounds) > 2 or not all(_slurm_reads_as_index_number(b) for b in bounds):
            return False
        if len(bounds) == 2 and int(bounds[1]) < int(bounds[0]):
            return False
    return True


def array_spec_reason(spec: str) -> str:
    """Why this array spec is refused, phrased truthfully, or "".

    Some refusals here are ones sbatch makes too - a reversed range, a
    non-numeric index, a zero step - and for those "invalid" is accurate. The
    rest are shapes the controller **accepts**, and for those "invalid" is a
    false claim about the scheduler: slurmate still refuses them, because each is
    a typo Slurm swallows in silence rather than reports, but the reason has to
    be the real one. Measured accepted-yet-refused, via
    :func:`_slurm_would_accept_array_spec`:

    * ``1-10%`` - a ``%`` with no number at all; sbatch drops the throttle and
      runs the array with no cap, which is not what a ``%`` was typed for.
    * ``1-10%%4``, ``1-10%4x`` - any other non-numeric throttle; sbatch never
      reads the text after the ``%`` as one, so the cap is not the written one.
    * ``1,,3`` - an empty entry in the index list; sbatch drops it, so a doubled
      separator quietly runs a smaller array than the one asked for.
    * ``1-10: 2`` - a space inside the spec; sbatch skips over it silently.
    """
    text = str(spec or "").strip()
    if not text or validate_array_spec(text):
        return ""
    body, sep, throttle = text.partition("%")
    if sep and not throttle.strip() and validate_array_spec(body):
        return (
            f"array spec '{text}' ends with '%' and no number. Slurm accepts this "
            f"and runs the array with *no* throttle at all, which is unlikely to "
            f"be what a '%' was typed for — write '%N' (e.g. '{body}%4') to cap "
            f"concurrent tasks, or drop the '%'"
        )
    if _slurm_would_accept_array_spec(text):
        faults: list[str] = []
        if body and any(entry == "" for entry in body.split(",")):
            faults.append("an empty entry in the index list, which Slurm drops")
        if any(ch.isspace() for ch in body):
            faults.append("a space inside the indices, which Slurm skips over")
        if sep and not throttle.isdigit():
            faults.append(
                f"'{throttle}' after the '%', which Slurm does not read as a "
                f"concurrency number"
            )
        kept = [re.sub(r"\s+", "", e) for e in body.split(",") if e.strip()]
        example = (",".join(kept) or "1-10") + ("%4" if sep else "")
        return (
            f"array spec '{text}' has "
            f"{' and '.join(faults or ['a shape slurmate does not accept'])}. "
            f"Slurm accepts this rather than reporting it, so the typo would "
            f"survive into a running array — write it as '{example}' if that is "
            f"what was meant"
        )
    return f"Invalid array specification: {text}"


def validate_array_spec(spec: str) -> bool:
    """Validate an ``sbatch --array`` specification's shape.

    ``--time`` and ``--memory`` values were checked for shape and the array spec
    was not, so ``--array 10-1`` produced a script Slurm refuses with "Invalid
    job array specification". Calibrated against a live controller rather than
    guessed — measured **accepted**: ``5``, ``1-10``, ``0-9``, ``1,3,5``,
    ``1-10:2``, ``1-10%4``, ``1-5,10`` and, unexpectedly, a bare ``%4``;
    measured **rejected**: ``10-1`` (reversed), ``1-10:0`` (zero step), ``1-``
    and ``-5``.
    """
    text = str(spec or "").strip()
    if not text:
        return True                      # no array requested
    body, sep, throttle = text.partition("%")
    if sep and not throttle.isdigit():
        return False
    if not body:
        # A bare "%N" carries no indices at all, which Slurm accepts.
        return bool(sep)
    for part in body.split(","):
        part = part.strip()
        if not part:
            return False
        index, step_sep, step = part.partition(":")
        if step_sep and not (step.isdigit() and int(step) > 0):
            return False
        bounds = index.split("-")
        if len(bounds) > 2 or not all(b.isdigit() for b in bounds):
            return False
        if len(bounds) == 2 and int(bounds[1]) < int(bounds[0]):
            return False
    return True


def _max_array_index(spec: str) -> int | None:
    """Highest task index a Slurm array spec asks for, or None if unreadable.

    Handles the forms ``sbatch --array`` accepts: ``1-10``, ``0-9:2``,
    ``1,3,5``, ``1-5,10``, and a ``%N`` throttle suffix, which bounds concurrency
    rather than the index and must not be read as one.
    """
    text = str(spec or "").split("%")[0].strip()
    if not text:
        return None
    highest: int | None = None
    for part in text.split(","):
        piece = part.split(":")[0].strip()   # drop a step; it cannot raise the max
        if not piece:
            continue
        bounds = piece.split("-")
        if len(bounds) > 2:
            return None
        try:
            values = [int(b) for b in bounds if b != ""]
        except ValueError:
            return None
        if not values:
            return None
        highest = max(values) if highest is None else max(highest, *values)
    return highest


def _module_command() -> list[str] | None:
    """argv prefix for a non-interactive module query, or None if no module system.

    ``module`` is a shell function, not a program, so it cannot be run directly.
    The real entry point is ``$LMOD_CMD`` on Lmod sites and
    ``$MODULESHOME/bin/modulecmd`` on Tcl environment-modules sites; going
    through ``bash -lc 'module ...'`` instead works but costs ~10 s on a real
    login node, against ~30 ms for the direct call.
    """
    lmod = os.environ.get("LMOD_CMD")
    if lmod and os.path.exists(lmod):
        return [lmod, "bash"]
    home = os.environ.get("MODULESHOME")
    if home:
        # Tcl environment-modules moved the real entry point between major
        # versions: 3.x ships $MODULESHOME/bin/modulecmd, 5.x ships
        # $MODULESHOME/libexec/modulecmd.tcl and leaves bin/ without it (measured
        # on Booth's Mercury, MODULESHOME=/usr/share/Modules). Checking only the
        # 3.x path makes every module check silently inert on a 5.x site whose
        # wrapper is not on PATH.
        for candidate in (
            os.path.join(home, "bin", "modulecmd"),
            os.path.join(home, "libexec", "modulecmd.tcl"),
        ):
            if os.path.exists(candidate):
                return [candidate, "bash"]
    found = shutil.which("modulecmd")
    return [found, "bash"] if found else None


def fetch_module_matches(name: str) -> list[str] | None:
    """Module names matching ``name``, or None when no module system can be asked.

    Reads **stderr**, not stdout. ``modulecmd bash -t avail X`` writes its
    listing to stderr and leaves stdout empty, because stdout is reserved for the
    shell code the caller is meant to ``eval``. A stdout-only read — the obvious
    way to write this — reports every module on the system as missing, which is
    the same "the answer was on the channel nobody read" mistake that produced
    several findings in the portability report.

    Exit status is no help either: a hit and a miss both exit 0, so emptiness is
    the only signal.
    """
    cmd = _module_command()
    if cmd is None or _force_mock():
        return None
    stdout, stderr, _rc = _run_command([*cmd, "-t", "avail", name])
    matches: list[str] = []
    for line in f"{stderr}\n{stdout}".splitlines():
        entry = line.strip()
        # Not module names: blank lines, path headers
        # ("/software/modulefiles:"), and Lmod's ruled section banners
        # ("------- /opt/apps/modulefiles -------"). Strip the rule characters
        # first, or the banner reads as a module called "---- /opt ... ----".
        entry = entry.strip("-= \t")
        if not entry or entry.endswith(":") or entry.startswith("/"):
            continue
        entry = entry.split("(")[0].strip().rstrip("/")
        if entry:
            matches.append(entry)
    return matches


def check_modules(modules: Iterable[str]) -> list[tuple[str, str]]:
    """Report ``module load`` names this cluster does not have.

    Module names are the most site-specific thing in a generated script after
    the partition, and they fail late: the job queues, starts, and *then* dies on
    ``module load``. A version that exists on one cluster and not the next is the
    common case, so when the base module is present its available versions are
    listed — that is the answer the user needs.

    Warnings, never errors: hierarchical module trees only expose part of
    themselves at a time, so absence here is strong evidence but not proof.
    Silent when there is no module system to ask.
    """
    out: list[tuple[str, str]] = []
    for raw in modules:
        name = str(raw).strip()
        if not name:
            continue
        matches = fetch_module_matches(name)
        if matches is None:
            return out          # no module system: ask nothing, claim nothing
        if matches:
            continue
        base = name.split("/")[0]
        siblings = fetch_module_matches(base) if base != name else []
        if siblings:
            shown = ", ".join(siblings[:6])
            more = f", ... (+{len(siblings) - 6} more)" if len(siblings) > 6 else ""
            out.append((
                "warning",
                f"module '{name}' not found on this cluster; "
                f"'{base}' is available as: {shown}{more}",
            ))
        else:
            out.append((
                "warning",
                f"module '{name}' not found on this cluster — the job would queue, "
                f"start, and then fail on 'module load'",
            ))
    return out


def check_conda_env(
    env: str, env_type: str, modules: list[str] | None = None
) -> list[tuple[str, str]]:
    """Report a conda/mamba environment name this machine does not have.

    Exactly the shape SM-13 fixed for ``module load``, left in place for the env
    field: the job queues, starts, and then dies on ``conda activate``. The env
    list was already being fetched — the wizard's picker offers it — but a name
    typed on the command line was never checked against it, so ``--modules`` was
    validated and ``--env`` was not.

    Named conda/mamba envs only. A ``venv`` is a *path*, and a path unreadable
    from the login node can be perfectly valid on the compute node — the same
    reason :func:`check_log_dirs` warns rather than refuses — so checking one here
    would manufacture false refusals.

    A warning, never an error. An empty env list still must not read as "your
    environment does not exist" — but it is not nothing either, which is what it
    used to be treated as: see :func:`_conda_unavailable`.
    """
    name = str(env or "").strip()
    if not name or str(env_type or "") not in ("conda", "mamba"):
        return []
    known = fetch_conda_envs(list(modules or []))
    if not known:
        # Could-not-ask, which is a fact about the *script* even though it says
        # nothing about the env: if conda cannot be reached here, the activation
        # line cannot run. Measured on all three clusters tested — conda is on
        # none of their default PATHs — so `--env x` with no conda-providing
        # module was the SM-13 failure exactly, in the field SM-13 did not cover.
        return _conda_unavailable(name, str(env_type), list(modules or []))
    if name in known:
        return []
    # Suggest only *named* envs. fetch_conda_envs also returns full paths for
    # --prefix envs, which is right for activation but useless as a suggestion:
    # they are 100+ characters each, they crowd out the names, and anyone using
    # one already knows its path. Sorting the raw list is worse still — paths
    # begin with "/" and so sort first, burying every name.
    named = sorted(e for e in known if "/" not in e)
    detail = ""
    if named:
        shown = ", ".join(named[:6])
        more = f", ... (+{len(named) - 6} more)" if len(named) > 6 else ""
        detail = f" Available: {shown}{more}"
    return [(
        "warning",
        f"conda environment '{name}' not found here — the job would queue, start, "
        f"and then fail on 'conda activate'.{detail}",
    )]


# Module-name fragments that mean "this module provides conda/mamba". Matched as
# substrings because sites name them every possible way — `conda/23.10` on
# Pythia, `python/anaconda-2025.12` on midway3, `Mambaforge` elsewhere.
_CONDA_MODULE_HINTS = (
    "conda", "mamba", "miniforge", "micromamba",
)


def _module_family(name: str) -> str:
    """A module name with its trailing version removed: ``conda/23.10`` -> ``conda``.

    Used to offer one candidate per family rather than every version of each.
    Handles both site conventions seen: the version as its own path segment
    (``conda/23.10``) and the version glued on with a dash
    (``python/anaconda-2025.12``).
    """
    text = str(name)
    head, _, tail = text.rpartition("/")
    if head and tail[:1].isdigit():
        return head
    stem, dash, ver = text.rpartition("-")
    if dash and ver[:1].isdigit():
        return stem
    return text


def _conda_unavailable(env: str, env_type: str, modules: list[str]) -> list[tuple[str, str]]:
    """Warn that ``conda``/``mamba`` cannot be reached, naming a module that helps.

    The script's first real line is ``source "$(conda info --base)/…"``, so a
    cluster where conda is not on ``PATH`` and no loaded module provides it
    produces a job that starts and immediately dies — the late failure SM-13
    exists to prevent, reached through ``--env`` rather than ``--modules``.

    A warning, and hedged, for the reason the env-name check is: a login node's
    ``PATH`` is not necessarily a compute node's, and a site whose shell profile
    sets conda up only there would make an error a false refusal. The remedy is
    named concretely when the module system can offer one, because "conda is
    missing" is not actionable on a cluster where it is simply a module away.
    """
    tool = "mamba" if env_type == "mamba" else "conda"
    if is_tool_available(tool):
        # It is on PATH and the listing still failed — a broken conda, or one
        # that cannot read its own config. Claim nothing about the environment.
        return []
    # Substring-search the whole module list rather than asking
    # `module -t avail conda`: that matches a **name prefix**, not a substring, so
    # it finds Pythia's `conda/23.10` and misses midway3's
    # `python/anaconda-2025.12` entirely — the cluster where the remedy is most
    # obviously available was the one it could not name.
    hints = (env_type, *_CONDA_MODULE_HINTS) if env_type == "mamba" else _CONDA_MODULE_HINTS
    available = fetch_available_modules()
    candidates: list[str] = []
    for hint in hints:
        for m in available:
            if hint in m.lower() and m not in candidates:
                candidates.append(m)
    if candidates:
        # One entry per family, the highest-sorting version of each. Listing raw
        # matches instead suggested `python/anaconda-2019.03` on a cluster that
        # also has 2025.12, and taking the last of the flat sorted list is no
        # better: it mixes families, so on midway3 (anaconda + miniforge +
        # miniforge3) the "newest" came out as miniforge3-4.8.
        newest: dict[str, str] = {}
        for m in candidates:
            fam = _module_family(m)
            if fam not in newest or m > newest[fam]:
                newest[fam] = m
        picks = list(newest.values())
        shown = ", ".join(picks[:4])
        more = f" (+{len(picks) - 4} more)" if len(picks) > 4 else ""
        # A single candidate can be named as the thing to do; with several,
        # choosing for the user would be a guess, so the list is the answer.
        how = (
            f"add it with --modules {picks[0]}"
            if len(picks) == 1
            else "add one with --modules"
        )
        remedy = f" This cluster provides it as a module ({how}): {shown}{more}."
    else:
        remedy = (
            " Load whatever your site uses for it via --modules, or use "
            "--env-type venv with a path."
        )
    return [(
        "warning",
        f"'{tool}' is not on this cluster's PATH, so activating '{env}' would fail "
        f"as soon as the job starts (the script runs "
        f"'{tool} activate {env}').{remedy}",
    )]


# Slurm's GPU spellings. --gres names a *resource*, so it must say `gpu`; the
# --gpus-per-* flags are already GPU-specific and carry a bare count or
# `<type>:count`.
_GRES_GPU_RE = re.compile(r"^gpu(?::([A-Za-z0-9_.+-]+))?:(\d+)$")
_GPUS_PER_RE = re.compile(r"^(?:([A-Za-z0-9_.+-]+):)?(\d+)$")

# Which --gpu-format each Slurm spelling corresponds to. They are three renderings
# of one request, so they resolve to slurmate's gpus/gpu_type/gpu_format rather
# than to settings of their own.
GPU_SPELLING_FORMATS = {
    "--gres": "gres_type",
    "--gpus-per-node": "gpus_per_node",
    "--gpus-per-task": "gpus_per_task",
}

# ``--gpus`` is not in the table above because it is slurmate's own option rather
# than an alias for it, and a bare ``--gpus 4`` must keep the default format. But
# it takes Slurm's ``[type:]count`` too — slurmate itself *prints*
# ``--gpus=a100:2`` under ``--gpu-format gpus``, and that was the one emitted
# spelling argparse still met with "invalid int value", i.e. the SM-25 defect
# surviving in the flag whose name matches the option. A type given this way does
# imply the format, since that rendering is the only one that produces it.
GPU_COUNT_FLAG = "--gpus"


def parse_gpu_spelling(flag: str, value: str) -> tuple[int, str]:
    """``(count, type)`` from a Slurm GPU flag's value; raises ``ValueError``.

    SM-25's general rule — anything slurmate prints should be typeable back at
    slurmate — applied to the GPU directives, which are also the ones most often
    copied out of somebody else's script.

    ``--gres`` is deliberately strict about the leading ``gpu``. It can carry any
    resource (``lscratch:100``), slurmate manages only GPUs, and quietly treating
    a non-GPU gres as one would drop a request the user actually made.

    Every other flag — ``--gpus``, ``--gpus-per-node``, ``--gpus-per-task`` —
    takes Slurm's ``count`` or ``<type>:count``.
    """
    text = str(value or "").strip()
    if flag == "--gres":
        match = _GRES_GPU_RE.match(text)
        if not match:
            raise ValueError(
                f"--gres {text!r} is not a GPU request slurmate can manage "
                f"(expected 'gpu:N' or 'gpu:<type>:N'). Pass a non-GPU gres "
                f"through --custom-sbatch instead."
            )
    else:
        match = _GPUS_PER_RE.match(text)
        if not match:
            raise ValueError(
                f"{flag} {text!r} is not a count or '<type>:count'"
            )
    return int(match.group(2)), (match.group(1) or "")


def write_private_text(path: str, text: str) -> None:
    """Write ``text`` to ``path``, creating it mode 0600.

    ``open(path, "w")`` leaves the mode to the umask, which is 0002 on both
    clusters measured — so a saved script came out ``-rw-rw-r--``. That is not a
    disclosure everywhere, but it is here: ``/project/rcc`` and the user's
    directory under it are both ``o+x``, so a world-readable file at a known path
    is readable cluster-wide, and 79 of 81 project directories are listable.
    Relying on site policy for that is exactly what a cluster-agnostic tool must
    not do.

    The content is the submitted script, so by construction it contains whatever
    was passed to ``--command`` — a token, an internal hostname, a credential in a
    one-liner. 0600 suits a file whose purpose is the submitter's own
    reproducibility; sharing it should be an explicit ``chmod``, not the umask's
    decision.

    The mode applies at *creation* only, so overwriting a file the user already
    made deliberately shareable leaves their permissions alone — O_TRUNC does not
    re-apply the mode.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    # surrogateescape for the same reason the callers used it: under a non-UTF-8
    # locale a --command with UTF-8 bytes arrives as lone surrogates, and a strict
    # write refuses them.
    with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write(text)


def unexpanded_home(path: str) -> bool:
    """Whether ``path`` still carries an unexpanded leading ``~``.

    ``os.path.expanduser`` does not raise when the home directory cannot be
    determined — it returns the string unchanged. So in the same environment that
    makes ``Path.home()`` raise (no ``$HOME``, no passwd entry), ``~/logs`` stays
    ``~/logs`` and Slurm, which does no tilde expansion of its own, writes the
    job's output to a *relative directory literally named* ``~``. Silent, and the
    log ends up somewhere nobody will look.
    """
    return path.startswith("~")


# Filesystem types that exist only on the machine you are standing on. Compared
# against the *type*, not the path, so a site that deliberately shares /tmp over
# NFS is not accused of anything.
_NODE_LOCAL_FS_TYPES = frozenset({
    "btrfs", "exfat", "ext2", "ext3", "ext4", "f2fs", "jfs", "ntfs", "overlay",
    "overlayfs", "ramfs", "reiserfs", "tmpfs", "vfat", "xfs", "zfs",
})

# Paths that are node-local by near-universal HPC convention, plus the ones that
# say so in their own name. Deliberately not bare `/scratch`: a node-local
# `/scratch` is common and so is a shared one, and warning about shared scratch
# would fire on the correct configuration. `/scratch/local` earns its place —
# midway3's is the same `/dev/sda1` as its `/tmp`, and it is where `$TMPDIR`
# points there, so a log written to it is as invisible as one in `/tmp`.
# Combined with the type check above, a hit is about as certain as this can be
# made without asking a compute node.
_NODE_LOCAL_PREFIXES = (
    "/tmp", "/var/tmp", "/dev/shm",
    "/scratch/local", "/local", "/localscratch", "/local_scratch",
)


def _mount_fs_type(path: str) -> str:
    """Filesystem type of the mount ``path`` lives on, or "" if it can't be read.

    Reads ``/proc/self/mountinfo`` rather than shelling out to ``stat -f``: this
    is called from :func:`check_log_dirs`, which runs on every redraw-adjacent
    path and must not acquire a subprocess.
    """
    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ""
    target = os.path.abspath(path)
    best = ""
    best_type = ""
    for line in lines:
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        fields = left.split()
        rest = right.split()
        if len(fields) < 5 or not rest:
            continue
        # mountinfo escapes spaces/tabs/newlines/backslashes as octal.
        point = (
            fields[4].replace("\\040", " ").replace("\\011", "\t")
            .replace("\\012", "\n").replace("\\134", "\\")
        )
        # One condition: this mount contains the target AND it is at least as
        # specific as the best seen so far. The two were nested, which read as if
        # the second were a separate concern.
        if (
            target == point or target.startswith(point.rstrip("/") + "/")
        ) and len(point) >= len(best):
            best, best_type = point, rest[0]
    return best_type


def node_local_log_dir(path: str) -> str:
    """The filesystem type when ``path`` is node-local storage, else "".

    A job's ``--output`` on node-local storage is the worst kind of working
    script: Slurm opens the file on the *compute* node, so the log lands in that
    node's private copy of the directory and the submitter — standing on the
    login node — sees nothing, while the job reports ``COMPLETED 0:0``. Measured
    on Booth's Mercury, where ``/tmp`` is an LVM volume per node: an identical
    job wrote its log from an NFS home and produced no readable output at all
    from ``/tmp``, with the same exit status both times.

    :func:`check_log_dirs` cannot catch this, and its docstring says why it does
    not try: it tests the directory on the *login* node, where a node-local
    ``/tmp/…/logs`` exists and is writable. Both facts are true and neither is
    the one that matters.
    """
    if _force_mock():
        # No cluster, so no compute node with its own /tmp — and --demo must not
        # lecture a demo user about their temp directory.
        return ""
    try:
        target = os.path.abspath(os.path.expanduser(path))
    except OSError:
        return ""
    if not any(
        target == pre or target.startswith(pre + "/") for pre in _NODE_LOCAL_PREFIXES
    ):
        return ""
    fs_type = _mount_fs_type(target)
    return fs_type if fs_type in _NODE_LOCAL_FS_TYPES else ""


def check_log_dirs(script: str, *, will_create: bool = True) -> list[tuple[str, str]]:
    """Report ``--output``/``--error`` directories that cannot be created here.

    The log path is the most cluster-specific value in a generated script — every
    site mounts its scratch somewhere else — and Slurm kills a job outright when
    it cannot open the file. Until now the failure was invisible twice over: no
    check before submit, and the ``os.makedirs`` attempt at submit time logged
    its ``OSError`` at debug level and submitted anyway.

    A warning, never an error, and deliberately so: a path can be unwritable from
    the login node and perfectly valid on the compute node.

    Node-local storage is checked *first*, by :func:`node_local_log_dir`, and is
    the one case this leniency got backwards: a node-local ``/tmp/x/logs``
    exists here and is writable here, so every test below passes, and the job's
    output still goes somewhere the submitter cannot read.

    ``will_create=False`` says the caller is handing the script to the *user* to
    submit, so nothing will create these directories. A merely absent one then
    matters, and is reported; when slurmate is the one submitting
    (:func:`submit_sbatch` makes them first) it does not, or every default
    ``logs/`` would warn on every run. SM-24: on the ``--print`` path this cost a
    whole job's output — Slurm accepts a path in a missing directory, discards
    what the job writes, and reports ``COMPLETED 0:0``, which is the most
    confusing result a batch user can get.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in script.splitlines():
        val = _sbatch_log_path(line)
        if not val:
            continue
        expanded = os.path.expanduser(val)
        if unexpanded_home(expanded):
            # Not a directory problem: the "~" never resolved, so this path is
            # not the one the user meant.
            out.append((
                "warning",
                f"log path '{val}' still begins with '~' — no home directory could "
                f"be resolved, and Slurm does not expand it, so the log would land "
                f"in a directory literally named '~'",
            ))
            continue
        directory = os.path.dirname(expanded)
        # A "%j"/"%x" component is expanded by Slurm per job; there is no such
        # literal directory to test.
        if not directory or "%" in directory or directory in seen:
            continue
        seen.add(directory)
        # Before existence and writability, which are both about the wrong
        # machine here: a node-local directory can pass both and still swallow
        # the job's entire output.
        local_fs = node_local_log_dir(directory)
        if local_fs:
            try:
                shown = os.path.abspath(os.path.expanduser(directory))
            except OSError:
                shown = directory
            out.append((
                "warning",
                f"log directory '{shown}' is on node-local storage ({local_fs}); "
                f"Slurm opens the log on the compute node, so it lands in that "
                f"node's own copy and is not readable from here — the job still "
                f"reports COMPLETED. Point --output-dir at shared storage (home, "
                f"project or scratch)",
            ))
            continue
        if os.path.isdir(directory):
            if not os.access(directory, os.W_OK):
                out.append((
                    "warning",
                    f"log directory '{directory}' is not writable from here; "
                    f"Slurm fails a job it cannot open the output file for",
                ))
            continue
        # Not there yet — normal for "logs/". The question is whether it could be
        # created, which is decided by the nearest existing ancestor.
        ancestor = directory
        while ancestor and not os.path.exists(ancestor):
            parent = os.path.dirname(ancestor)
            if parent == ancestor:
                break
            ancestor = parent
        # A *relative* directory walks up to "" — dirname("logs") is empty — and
        # reading that as "/" made the check claim that `logs` could not be
        # created whenever it did not exist yet. That is the default output
        # directory, so a first-time user in a perfectly writable directory got a
        # false warning; the effective ancestor there is the working directory.
        if not ancestor:
            ancestor = os.curdir
        if not os.path.isdir(ancestor) or not os.access(ancestor, os.W_OK):
            # Resolve the ancestor for the message: "." is accurate but useless in
            # a CI or job log, where the reader cannot see what the cwd was.
            try:
                shown = os.path.abspath(ancestor)
            except OSError:                     # cwd deleted under us
                shown = ancestor or "/"
            out.append((
                "warning",
                f"log directory '{directory}' cannot be created from here "
                f"(nearest existing parent: '{shown}'); Slurm fails a "
                f"job it cannot open the output file for",
            ))
        elif not will_create:
            # Missing, but creatable — so "cannot be created" does not apply and
            # would be wrong. Reported only when nobody is going to create it:
            # this is the SM-24 case, where the script is handed to the user,
            # Slurm accepts an output path in a missing directory, discards what
            # the job writes, and still reports COMPLETED. Placed *after* the
            # creatability verdict so an unwritable parent keeps the stronger
            # message, which tells the reader that mkdir will not help either.
            out.append((
                "warning",
                f"log directory '{directory}' does not exist, and this script is "
                f"yours to submit — slurmate only creates it when it submits for "
                f"you. Run 'mkdir -p {directory}' first, or Slurm will discard "
                f"this job's output and still report it COMPLETED",
            ))
    return out


# A constraint that is a single plain feature name, as opposed to a Slurm feature
# *expression* — those support "&", "|", "!", "*N", "[a|b]" and counts, and a set
# membership test would reject perfectly valid ones.
_PLAIN_FEATURE_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")


def _fetch_node_features_uncached() -> set[str] | None:
    """Every node feature the cluster advertises; ``None`` when unreadable.

    Cluster-wide rather than per-partition on purpose: a feature is a property of
    nodes, and a user naming one that exists elsewhere on the cluster has made a
    different (and much less likely) mistake than one naming a feature that does
    not exist at all. Rejecting only the latter keeps this from producing false
    refusals.

    ``None`` and an empty set are **different answers** and callers depend on the
    difference. A cluster can legitimately advertise no features at all — every
    node on Booth's Mercury reports ``(null)`` — and there a plain ``-C name``
    matches nothing, which is worth saying. Collapsing that into the same empty
    set as "sinfo could not be asked" is what made the constraint check inert on
    exactly the clusters where it had a definite answer to give.
    """
    if not is_tool_available("sinfo") or _force_mock():
        return None
    stdout, _, rc = _run_command(["sinfo", "-h", "-o", "%f"], timeout=_ADVISORY_TIMEOUT)
    if rc != 0:
        return None
    features: set[str] = set()
    for line in stdout.splitlines():
        for token in _normalize_null(line).split(","):
            token = token.strip()
            if token:
                features.add(token)
    return features


# GPU request syntaxes that only the cons_tres select plugin understands. Under
# select/cons_res or select/linear the *parser* refuses them cluster-wide
# ("Requested GRES option unsupported by configured SelectType plugin"), so this
# is not a partition or a node-config problem and no partition choice avoids it.
_CONS_TRES_ONLY_GPU_FORMATS = frozenset({"gpus", "gpus_per_task"})

# Select plugins that do understand them. Anything else — including an
# unreadable value — is treated as unknown, and unknown must claim nothing.
_CONS_TRES_SELECT_TYPES = frozenset({"select/cons_tres"})


def _fetch_select_type_uncached() -> str:
    """The cluster's ``SelectType``, or "" when it cannot be read.

    Decides which GPU request syntaxes exist at all: ``--gpus`` and
    ``--gpus-per-task`` are cons_tres-only, while ``--gres=gpu:…`` and
    ``--gpus-per-node`` parse everywhere. Two real clusters differ on this —
    one runs ``select/cons_tres``, the other ``select/cons_res`` — which is
    exactly why a ``gpu_format`` carried between them stops working.
    """
    if _force_mock() or not is_tool_available("scontrol"):
        return ""
    stdout, _, rc = _run_command(["scontrol", "show", "config"], timeout=_ADVISORY_TIMEOUT)
    if rc != 0:
        return ""
    match = re.search(r"^SelectType\s*=\s*(\S+)", stdout, re.MULTILINE)
    return match.group(1).strip() if match else ""


def unsupported_gpu_format(gpu_format: str, select_type: str) -> str:
    """Why this ``gpu_format`` cannot work here, or "".

    Silent when ``select_type`` is empty or unrecognised: failing open to the
    default ``gres_type`` is already the safe behaviour, and an unreadable
    ``scontrol`` must not present as "your GPU syntax is wrong".
    """
    fmt = str(gpu_format or "").strip().lower()
    plugin = str(select_type or "").strip().lower()
    if not fmt or not plugin or fmt not in _CONS_TRES_ONLY_GPU_FORMATS:
        return ""
    if plugin in _CONS_TRES_SELECT_TYPES:
        return ""
    if not plugin.startswith("select/"):
        return ""                    # not a value we understand; claim nothing
    return (
        f"gpu_format '{fmt}' needs select/cons_tres, but this cluster runs "
        f"{select_type} — Slurm refuses the request at parse time, on every "
        f"partition. Use 'gres_type' (the default) or 'gpus_per_node'"
    )


def _fetch_max_array_size_uncached() -> int | None:
    """The site's ``MaxArraySize``, or None when it cannot be read.

    A hard scheduler limit that differs wildly between sites — Slurm's own
    default is 1001, this cluster is configured at 65533 — so an ``--array``
    carried from one site is exactly the kind of value that generates a script
    the local controller refuses. None means "unknown"; the caller must stay
    silent rather than guess a limit.
    """
    if _force_mock() or not is_tool_available("scontrol"):
        return None
    stdout, _, rc = _run_command(["scontrol", "show", "config"], timeout=_ADVISORY_TIMEOUT)
    if rc != 0:
        return None
    match = re.search(r"^MaxArraySize\s*=\s*(\d+)", stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def _parse_partition_timelimit(raw: Any) -> float | None:
    """A partition's max time in minutes: ``math.inf`` unbounded, ``None`` unknown.

    Every partition on some clusters is ``TIMELIMIT=infinite``, and collapsing
    that to the same ``None`` as "could not parse this" makes the time check
    inert site-wide with no way to tell the two apart. They are different facts:
    unbounded means any request is acceptable and can be affirmed, unknown means
    no check is possible and the only honest move is silence.
    """
    value = str(raw or "").strip().lower()
    if not value or value in {"n/a", "not_set", "invalid", "unknown", "none"}:
        return None
    if value in {"infinite", "unlimited", "inf"}:
        return math.inf
    minutes = _parse_slurm_time_to_minutes(str(raw))
    return minutes if minutes > 0 else None


def capacity_refusal(
    part: dict[str, Any] | None,
    answers: dict[str, Any],
    max_array_size: int | None = None,
) -> str:
    """Why this request cannot fit, from the partition's own figures, or "".

    A second, scheduler-independent answer to "can this run at all". The ETA's
    first choice is Slurm's own verdict, but when ``sbatch`` cannot be reached the
    estimate used to fall through to a queue-depth heuristic and print a
    confident ``~7min`` on the same screen as a warning saying the request exceeds
    the partition. The warnings already knew; this makes the ETA able to ask them.

    Only **exact** limits are reported. A ``heterogeneous`` partition's cpu/memory
    figures are floors, not ceilings — ``sinfo`` printed its smallest node — so a
    bigger node in the same partition may well take the job, and claiming "never"
    there would trade one confident wrong answer for another. Node counts, array
    indices and the partition time limit are exact and so always count. The array
    index is the one rule here that reads no partition figure at all, so it is
    also the only answer available when no partition was chosen.

    Pure: no subprocess calls, so the ETA path can consult it for free.
    """
    if not part:
        # A blank partition is a legitimate answer ("use the site default"), and
        # returning "" for it dropped the one rule here that never looks at a
        # partition. MaxArraySize is a controller-wide slurm.conf value, not an
        # advertised partition figure: measured, `sbatch --test-only
        # --array=1-99999` against this site's 65533 is refused "Invalid job
        # array specification" both under `-p amd` and under the site default.
        # Same shape as validate_job_config's blank-partition early return, and
        # it calls the same rule body rather than carrying a second copy of the
        # condition -- two surfaces disagreeing about one fact is the failure
        # this module works hardest to avoid.
        issues = _array_size_issues(answers, max_array_size)
        return issues[0][1] if issues else ""
    soft = bool(part.get("heterogeneous"))

    def _as_int(value: Any) -> int | None:
        try:
            if value is None or str(value).strip() == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    req_nodes = _as_int(answers.get("nodes"))
    total_nodes = _as_int(part.get("nodes")) or 0
    if req_nodes and total_nodes and req_nodes > total_nodes:
        return f"partition '{part.get('name')}' has only {total_nodes} node(s)"

    # Left in place rather than hoisted above the no-partition return: the order
    # of these rules is what a refusal *says* when a request breaks several at
    # once, so the node count keeps winning over the array as it always did.
    array_issues = _array_size_issues(answers, max_array_size)
    if array_issues:
        return array_issues[0][1]

    requested_time = str(answers.get("time_limit") or "")
    if requested_time:
        limit_mins = _parse_partition_timelimit(part.get("timelimit"))
        if limit_mins is not None and (
            _parse_slurm_time_to_minutes(requested_time) > limit_mins
        ):
            return f"partition time limit is {part.get('timelimit')}"

    # `soft` used to skip cpu/memory entirely on a mixed partition, because the
    # only figure available was `sinfo`'s floor and refusing against it would have
    # claimed "never" for a request a larger node takes. SM-27's per-node lookup
    # removes that limitation where it has run: an exact maximum *is* a ceiling,
    # so the refusal is sound even on a mixed partition. Without it the old
    # silence is still right. Reading the enriched dict keeps this function pure —
    # the ETA path consults it for free and must not gain a subprocess call.
    cpu_limit, cpu_exact = partition_capacity(part, "cpus")
    cores = _as_int(answers.get("cpus"))
    if cores and cpu_limit and (cpu_exact or not soft):
        tasks = _as_int(answers.get("ntasks_per_node")) or 1
        if cores * max(1, tasks) > cpu_limit:
            return f"no node in '{part.get('name')}' has {cores * max(1, tasks)} cores"

    mem_limit, mem_exact = partition_capacity(part, "mem")
    requested_mb = resolve_request_mem_mb(answers)
    if mem_limit and requested_mb > mem_limit and (mem_exact or not soft):
        return f"no node in '{part.get('name')}' has {requested_mb} MB"

    if soft:
        # Anything below this point still has only aggregate figures to go on.
        return ""

    req_gpus = _as_int(answers.get("gpus")) or 0
    gpu_limit = _as_int(part.get("gpus_per_node")) or 0
    if req_gpus and gpu_limit and req_gpus > gpu_limit:
        return f"no node in '{part.get('name')}' has {req_gpus} GPUs"

    return ""


def _limit_phrase(part: dict[str, Any], amount: str, *, exact: bool = False) -> str:
    """How to describe a per-node capacity: a ceiling, or a floor.

    ``sinfo`` marks a partition whose nodes differ with a trailing ``+`` on
    ``%c``/``%m``, and the number it prints is then the *smallest* node. Calling
    that "the partition limit" makes an over-request warning assert a ceiling
    that a larger node in the same partition may well clear.

    ``exact=True`` means the figure came from the per-node query, so it *is* the
    largest node and the claim is a real ceiling even on a mixed partition.
    """
    if exact and part.get("heterogeneous"):
        return f"exceeds the largest node in this partition ({amount})"
    if part.get("heterogeneous"):
        return f"exceeds the smallest node in this partition ({amount}); nodes differ"
    return f"exceeds partition limit ({amount})"


def _managed_flag_issues(answers: dict[str, Any]) -> list[tuple[str, str]]:
    """Custom ``#SBATCH`` flags that repeat a directive slurmate emits itself.

    Partition-independent: it reads only ``custom_sbatch``, so it is one of the
    two rules :func:`validate_job_config` still runs when no partition is known.
    Reported here (not only on the batch path) so the wizard's live panel, its
    summary and the pre-submit guard all see it.
    """
    from .builder import managed_custom_flags

    return [
        (
            "error",
            f"custom flag {name} duplicates a directive slurmate manages; "
            f"Slurm would honour it over --{owner.lstrip('-')} and the summary "
            f"would describe the wrong value",
        )
        for name, owner in managed_custom_flags(answers.get("custom_sbatch"))
    ]


def _array_size_issues(
    answers: dict[str, Any], max_array_size: int | None
) -> list[tuple[str, str]]:
    """Array indices vs the site's ``MaxArraySize``.

    ``max_array_size`` is passed in, never fetched here — this runs on every TUI
    redraw. ``None`` means unknown, and an unknown limit must not become a claim
    about one.

    Partition-independent: ``MaxArraySize`` is a controller-wide ``slurm.conf``
    value, so the refusal it predicts does not depend on which partition (or
    none) was chosen — measured with ``sbatch --test-only ... --array=1-99999``
    against this site's 65533, which is refused with "Invalid job array
    specification".
    """
    if not max_array_size:
        return []
    top = _max_array_index(str(answers.get("array_spec") or ""))
    if top is None or top < max_array_size:
        return []
    return [(
        "warning",
        f"Array index {top} is at or beyond this cluster's MaxArraySize "
        f"({max_array_size}); Slurm rejects the job with 'Invalid job "
        f"array specification'",
    )]


def partition_capacity(part: dict[str, Any], key: str) -> tuple[int, bool]:
    """``(limit, exact)`` for ``"cpus"`` or ``"mem"`` on this partition.

    SM-27: on a heterogeneous partition the aggregate ``sinfo`` row is a *floor*,
    so comparing against it warns about requests that fit a larger node — 20 of
    87 partitions on midway3 emit the ``+``, and its ``test`` partition reports
    ``32+`` where nodes reach 256 cores. When the per-node maxima have been
    resolved (see :func:`fetch_partition_node_maxima`) the largest node is used
    instead and the warning becomes a real ceiling.

    Falls back to the aggregate figure when the per-node query could not be made,
    which keeps the honest "smallest node; nodes differ" wording rather than going
    silent — an unknown must not turn a floor into a ceiling *or* disable the
    check entirely.
    """
    if key == "cpus":
        exact_val = part.get("max_cpus_per_node")
        floor_val = part.get("cpus_per_node", 0)
    else:
        exact_val = part.get("max_mem_per_node_mb")
        floor_val = part.get("mem_per_node_mb", 0)
    if part.get("heterogeneous") and isinstance(exact_val, int) and exact_val > 0:
        return exact_val, True
    try:
        return int(floor_val or 0), False
    except (TypeError, ValueError):
        return 0, False


def validate_job_config(
    answers: dict[str, Any],
    extra_gpu_types: list[str] | None = None,
    feature_only_gpu_types: list[str] | None = None,
    constraint_gpu_types: list[str] | None = None,
    max_array_size: int | None = None,
) -> list[tuple[str, str]]:
    """Validate a (possibly incomplete) answers dict against the selected
    partition's advertised capabilities.

    Returns a list of ``(level, message)`` tuples, where ``level`` is:

    - ``"error"``   — a configuration Slurm will reject outright (e.g. GPUs on a
      CPU-only partition, or a GPU model the partition doesn't have).
    - ``"warning"`` — a request that exceeds a node's advertised capacity and may
      be rejected or left pending (CPU/memory/time over the per-node limit; the
      advertised value can undercount a heterogeneous partition, so it isn't a
      guaranteed failure).

    An empty list means nothing is known to be wrong. Both the live TUI check
    (every redraw) and the final CLI summary share this single source of truth,
    so the two surfaces can't drift apart.

    This function is pure and side-effect free — it makes **no** subprocess
    calls — so the TUI can safely call it on every keystroke/redraw. Callers
    that can afford a live ``sinfo`` lookup (e.g. the one-shot CLI summary) may
    pass ``extra_gpu_types`` to widen the set of GPU models considered valid
    beyond what ``_partition_obj`` statically lists, and
    ``feature_only_gpu_types`` / ``constraint_gpu_types`` (both from
    :func:`fetch_gpu_type_sources`) to have the GPU *request format* checked
    against how each model can actually be asked for, in either direction.
    ``constraint_gpu_types`` distinguishes "not looked up" (``None``) from "looked
    up, and this partition advertises no GPU-model features" (``[]``) — the second
    is the whole point of the check, so the two cannot share a spelling.
    """
    part = answers.get("_partition_obj")
    out: list[tuple[str, str]] = []
    if not part:
        # No partition is a legitimate answer, not an incomplete one: it means
        # "use the site default", and the builder emits no --partition directive
        # for it (SM-15). The wizard reaches it by confirming an empty value in
        # the "Enter partition name manually..." row, which leaves
        # ``_partition_obj`` None for the rest of the session.
        #
        # So the partition-DEPENDENT checks below have nothing to compare against
        # and must stay silent — but returning [] silenced the whole function,
        # including the rules that never look at a partition at all. A duplicated
        # ``--job-name`` in --custom-sbatch then produced a second #SBATCH line
        # that Slurm accepts (measured: `sbatch --test-only ... -J first -J
        # second` is ***PASSED***) and silently honours over slurmate's, while
        # the summary described the value that lost — the one failure mode this
        # check exists to catch, dropped for a reason unrelated to it.
        out.extend(_managed_flag_issues(answers))
        out.extend(_array_size_issues(answers, max_array_size))
        return out

    # Cores requested per node (cpus-per-task x tasks-per-node), shared by the CPU
    # check and the --mem-per-cpu total below.
    cores_per_node = 0
    try:
        _cpus_raw = answers.get("cpus")
        if _cpus_raw is not None and str(_cpus_raw).strip() != "":
            _cores = int(_cpus_raw)
            _ntpn_raw = answers.get("ntasks_per_node")
            _ntpn = int(_ntpn_raw) if _ntpn_raw else 1
            cores_per_node = _cores * max(1, _ntpn)
            limit, cpu_exact = partition_capacity(part, "cpus")
            if limit and cores_per_node > limit:
                detail = f"{_ntpn}×{_cores}={cores_per_node}" if _ntpn > 1 else str(cores_per_node)
                out.append((
                    "warning",
                    f"CPUs ({detail}) "
                    f"{_limit_phrase(part, f'{limit} per node', exact=cpu_exact)}",
                ))
    except (ValueError, TypeError):
        pass

    # Memory vs the node's advertised memory — checked against what the SCRIPT will
    # actually request, mirroring the builder's precedence: a custom --mem /
    # --mem-per-cpu flag suppresses the auto directive, and --mem-per-cpu wins over
    # --mem. Checking the raw `memory` answer regardless meant warning about a value
    # the script doesn't request (and staying silent about the one it does).
    from .builder import (
        _custom_mem_override,
        _normalize_custom_flags,
    )
    out.extend(_managed_flag_issues(answers))
    _c_mem, _c_mem_per_cpu, _c_mem_per_gpu = _custom_mem_override(
        _normalize_custom_flags(answers.get("custom_sbatch"))
    )
    mem_limit, mem_exact = partition_capacity(part, "mem")
    eff_mem_per_cpu = _c_mem_per_cpu or answers.get("mem_per_cpu")
    # A custom --mem-per-gpu suppresses the auto directive too, so neither the
    # answers' `memory` nor its `mem_per_cpu` is what the script requests.
    if _c_mem_per_gpu:
        eff_mem_per_cpu = _c_mem_per_cpu
    eff_mem = (
        _c_mem
        if (_c_mem or _c_mem_per_cpu or _c_mem_per_gpu)
        else answers.get("memory")
    )
    if eff_mem_per_cpu and validate_memory(str(eff_mem_per_cpu)):
        # --mem-per-cpu is per core, so the per-node request is that x the cores
        # requested on the node. Without this, --mem-per-cpu=64G on an 8-core task
        # (512G/node) passed silently while the equivalent --mem=512G warned.
        per_cpu_mb = _parse_mem_to_mb(str(eff_mem_per_cpu))
        total_mb = per_cpu_mb * cores_per_node
        if mem_limit and total_mb > mem_limit:
            out.append((
                "warning",
                f"Memory ({eff_mem_per_cpu}/CPU × {cores_per_node} cores = {total_mb} MB) "
                f"{_limit_phrase(part, f'{mem_limit} MB per node', exact=mem_exact)}",
            ))
    elif eff_mem and validate_memory(str(eff_mem)):
        mb = _parse_mem_to_mb(str(eff_mem))
        if mem_limit and mb > mem_limit:
            out.append((
                "warning",
                f"Memory ({eff_mem}) "
                f"{_limit_phrase(part, f'{mem_limit} MB per node', exact=mem_exact)}",
            ))

    # Time vs the partition's max time.
    time_limit = answers.get("time_limit")
    if time_limit:
        try:
            req_mins = _parse_slurm_time_to_minutes(str(time_limit))
            # None = the partition told us nothing, so there is nothing to check.
            # math.inf = the partition is unbounded, so the request is fine — a
            # comparison against inf affirms it instead of skipping the check.
            # SM-28 asked for the QoS MaxWall to be compared here. Measured and
            # NOT done, because it is not a cluster-invariant: whether exceeding a
            # QoS MaxWall is refused at submit depends on the QoS's DenyOnLimit
            # flag, which no QoS on midway3 sets. There, `--qos=build
            # --time=30-00:00:00` against `MaxWall=06:00:00` is reported
            # ***PASSED*** by sbatch, so a local comparison warns about a limit the
            # scheduler does not enforce — the false warning this module exists to
            # avoid. Slurm's own verdict is site-accurate and already consulted on
            # every path; a site that does enforce it answers
            # QOSMaxWallDurationPerJobLimit, which refusal_is_permanent() now
            # recognises.
            limit_mins = _parse_partition_timelimit(part.get("timelimit"))
            over = False
            if limit_mins is not None:
                if time_request_is_unbounded(str(time_limit)):
                    # An unbounded request parses to *zero* minutes, so a plain
                    # comparison read "no limit" as a zero-length job.
                    over = limit_mins != math.inf
                else:
                    over = req_mins > limit_mins
            if over:
                out.append((
                    "warning",
                    f"Time limit ({time_limit}) exceeds partition limit "
                    f"({part.get('timelimit')})",
                ))
        except Exception:
            pass

    # GPUs vs what a node advertises. `sinfo %G` carried the count all along
    # (`gpu:4`); it was the one advertised resource with no limit check, so
    # `--gpus 99` on a 4-GPU partition produced a script and said nothing.
    try:
        req_gpus_raw = answers.get("gpus")
        if req_gpus_raw is not None and str(req_gpus_raw).strip() != "":
            req_gpus = int(req_gpus_raw)
            gpu_limit = int(part.get("gpus_per_node") or 0)
            if req_gpus > 0 and gpu_limit and req_gpus > gpu_limit:
                out.append((
                    "warning",
                    f"GPUs ({req_gpus}) {_limit_phrase(part, f'{gpu_limit} per node')}",
                ))
    except (ValueError, TypeError):
        pass

    # Nodes vs how many the partition has at all. Unlike cpu/mem this is a plain
    # count, not a per-node figure, so the heterogeneity caveat does not apply:
    # asking for more nodes than exist cannot be satisfied by a bigger node.
    try:
        req_nodes_raw = answers.get("nodes")
        if req_nodes_raw is not None and str(req_nodes_raw).strip() != "":
            req_nodes = int(req_nodes_raw)
            total_nodes = int(part.get("nodes") or 0)
            if total_nodes and req_nodes > total_nodes:
                out.append((
                    "warning",
                    f"Nodes ({req_nodes}) exceeds the {total_nodes} node(s) in "
                    f"'{part.get('name')}'",
                ))
    except (ValueError, TypeError):
        pass

    # Array indices vs the site's MaxArraySize — see _array_size_issues, which
    # the no-partition path above shares because the limit is a cluster
    # constant rather than a partition figure.
    out.extend(_array_size_issues(answers, max_array_size))

    # A partition slurmate could not resolve has no limits, so every check above
    # compared against nothing and stayed quiet. That inverts the failure mode a
    # user expects: ask for 999 CPUs on a real partition and you are warned; make
    # a typo in the partition name and you are not, so the *less* valid request
    # produces the *more* reassuring screen. Say what was not checked.
    if part.get("_unknown") and any(
        str(answers.get(k) or "").strip() not in ("", "0")
        for k in ("cpus", "memory", "mem_per_cpu", "time_limit", "nodes", "gpus")
    ):
        if part.get("_unknown_reason") == "unreadable":
            detail = last_cluster_error()
            why = (
                f"this cluster's partition list could not be read ({detail})"
                if detail
                else "this cluster's partition list could not be read (no Slurm, "
                     "or sinfo failed)"
            )
        else:
            why = f"partition '{part.get('name')}' is not on this cluster"
        out.append((
            "warning",
            f"Capacity limits NOT checked: {why}, so its CPU, memory, GPU and time "
            f"limits are unknown — the request above has been validated for shape "
            f"only",
        ))

    # The PARTITION's own state, which is a different fact from its nodes': a
    # partition can be UP with every node dead (caught below) or DOWN/INACT with
    # a hundred live nodes (caught here). Slurm accepts a job for a down
    # partition and then never starts it — "queues forever with no indication
    # why", which is the most opaque way for a cross-cluster guess to fail.
    part_state = str(part.get("state") or "").strip().lower()
    if part_state and part_state not in _AVAILABLE_PARTITION_STATES:
        out.append((
            "warning",
            f"Partition '{part.get('name')}' is {part_state} (the partition itself, "
            f"not its nodes) — Slurm accepts the job and never starts it",
        ))

    # A partition whose nodes are every one down/drained can never start the job.
    # A warning rather than an error: nodes come back, and queuing ahead of a
    # repair window is legitimate — but "queues forever with no indication why"
    # is the single most opaque way for a cross-cluster guess to fail.
    # ``nodes_up`` is None when the site's sinfo reported no state column at all,
    # which is "unknown", not "none" — stay silent there.
    nodes_up = part.get("nodes_up")
    total_nodes = part.get("nodes") or 0
    if nodes_up == 0 and total_nodes:
        out.append((
            "warning",
            f"Partition '{part.get('name')}' has no usable nodes right now "
            f"(all {total_nodes} are down/drained/reserved) — the job would queue "
            f"indefinitely",
        ))

    # GPUs requested on a partition *known* to advertise none. Only assert this
    # when ``has_gpu`` is explicitly False: real partition objects (from
    # fetch_partitions / MOCK) always carry it as a bool, so ``is False`` means
    # "we looked and there's no gpu GRES" — a config Slurm will reject. A
    # manually-typed or unrecognized partition falls back to a synthetic object
    # with no ``has_gpu`` key (capability unknown, like the 0/None cpu/mem/time
    # limits the checks above stay silent on), so we must not overclaim a hard
    # "no GPUs" error there. ``has_gpu`` also stays True for count-only
    # ("gpu:4") / typed-without-count GRES that don't populate gpu_types, so a
    # real GPU partition is never flagged as CPU-only.
    gpus = answers.get("gpus", 0)
    try:
        gpus_val = int(gpus) if (gpus is not None and str(gpus).strip() != "") else 0
    except (ValueError, TypeError):
        gpus_val = 0
    gpu_types = list(part.get("gpu_types", []))
    if gpus_val > 0 and not gpu_types and part.get("has_gpu") is False:
        out.append(("error", f"Partition '{part.get('name')}' does not support GPUs"))

    # A specific GPU model the partition doesn't offer. Only meaningful when we
    # actually know which models the partition has (static list plus any
    # caller-supplied dynamic types); with no type info at all, the count-only
    # "does not support GPUs" check above is the right signal, and warning
    # "not in partition list ()" against an empty list would be noise.
    gpu_type = answers.get("gpu_type")
    if gpu_type and str(gpu_type).lower() != "any":
        all_types = gpu_types + [t for t in (extra_gpu_types or []) if t not in gpu_types]
        known = {str(g).lower() for g in all_types}
        recognized = bool(known) and str(gpu_type).lower() in known
        if known and not recognized:
            out.append(("error", f"GPU type '{gpu_type}' not in partition list ({', '.join(all_types)})"))
        elif known:
            # Matched only case-insensitively. Slurm node features ARE
            # case-sensitive — a node advertising "A100" is not matched by
            # "-C a100" ("Invalid feature specification" / "Requested node
            # configuration is not available") — so a case-only difference is a
            # real, and otherwise invisible, way for a validated job to be
            # rejected at submit.
            exact = {str(g) for g in all_types}
            if str(gpu_type) not in exact:
                advertised = next(
                    (str(g) for g in all_types if str(g).lower() == str(gpu_type).lower()), ""
                )
                out.append((
                    "warning",
                    f"GPU type '{gpu_type}' differs in case from the partition's "
                    f"'{advertised}'; Slurm node features are case-sensitive",
                ))

        # Requestability: a model that only ever appears in a node's *feature*
        # list (because the node's GRES is count-only, "gpu:4") is not a GRES
        # type. Every format except "constraint" names the type inside the GRES
        # request, which Slurm then rejects outright — measured on a count-only
        # partition: `--gres=gpu:a100:1` → "Requested node configuration is not
        # available", while `--gres=gpu:1 --constraint=a100` schedules. This is
        # the default path (gres_type is the default format and the type comes
        # from slurmate's own picker), so it has to be caught before submit.
        feature_only = {str(t) for t in (feature_only_gpu_types or [])}
        fmt = str(
            answers.get("gpu_format")
            or os.environ.get("SLURMATE_GPU_FORMAT", "gres_type")
        ).lower()
        if gpus_val > 0 and str(gpu_type) in feature_only:
            if fmt != "constraint":
                out.append((
                    "error",
                    f"GPU type '{gpu_type}' is a node feature on "
                    f"'{part.get('name')}', not a GRES type (the nodes advertise a "
                    f"count-only 'gpu:N'), so gpu_format '{fmt}' would emit a "
                    f"request Slurm rejects — use gpu_format 'constraint'",
                ))
        # The mirror image, and reachable by exactly the remedy the check above
        # recommends: `--constraint` names a node *feature*, so a model that is a
        # real GRES type but appears in no node's feature list cannot be requested
        # that way. Slurm answers "Invalid feature specification" — measured on
        # both Booth clusters, whose nodes advertise typed GRES (`gpu:h100:8`) and
        # no features whatsoever, so on those *every* partition is in this state
        # and `gpu_format constraint` could never work. ``None`` means the lookup
        # did not run, so nothing is claimed; ``[]`` is the measured answer that
        # matters.
        elif (
            gpus_val > 0
            and fmt == "constraint"
            and constraint_gpu_types is not None
            # Only for a model this partition really offers. Without it, a model
            # the partition does not have at all drew a second error asserting it
            # "is a GRES type on <partition>" — a false statement stacked on top
            # of the true "not in partition list" one, measured on midway3's `gpu`
            # partition. When the model is unknown, that first error is the whole
            # answer.
            and recognized
        ):
            requestable = {str(t).lower() for t in constraint_gpu_types}
            if str(gpu_type).lower() not in requestable:
                offer = (
                    "the partition advertises no GPU model as a node feature"
                    if not requestable
                    else f"node features here are {', '.join(sorted(constraint_gpu_types))}"
                )
                out.append((
                    "error",
                    f"GPU type '{gpu_type}' is a GRES type on '{part.get('name')}', "
                    f"not a node feature ({offer}), so gpu_format 'constraint' "
                    f"would emit '--constraint={gpu_type}' and Slurm rejects that "
                    f"with 'Invalid feature specification' — use gpu_format "
                    f"'gres_type'",
                ))

    return out


# ``sinfo %a`` values that mean the partition itself will start work. "drain"
# accepts submissions but starts nothing; "inact" accepts nothing at all; "down"
# is down. Only "up" is usable.
_AVAILABLE_PARTITION_STATES = frozenset({"up"})

# Node states (sinfo ``%T``, long form) that represent real capacity: a job can
# land there now, or as soon as a job already running on the node finishes.
# Everything else — down / drain* / fail* / maint / unknown / future / inval —
# can never start a job, so summing it into a partition's node count is what
# makes a fully-retired partition look like a live choice.
_ALLOCATABLE_NODE_STATES = frozenset({
    "allocated", "alloc", "completing", "comp", "idle", "mixed", "mix",
    "planned", "plnd",
    # Power-save states still accept work — Slurm resumes the node on demand.
    "powereddown", "powerdown", "poweringup", "powerup",
})

# State-flag characters sinfo appends to a node state. Each one means the node
# cannot take work whatever its base state says: "*" not responding, "$" reserved
# for maintenance, "%" powering down, "@" pending reboot, "!" pending power-down.
# So "idle*" is not capacity, and neither is "mixed$".
_UNSCHEDULABLE_FLAGS = ("*", "$", "%", "@", "!")


def _is_allocatable_state(raw: str) -> bool:
    """Whether an ``sinfo %T`` node state can ever start a job.

    Returns ``False`` for an empty state so callers can distinguish "no usable
    nodes" from "state column absent" only by checking emptiness themselves —
    :func:`fetch_partitions` does exactly that and reports ``nodes_up=None``
    (unknown) rather than 0 when a site's sinfo gives it nothing to read.
    """
    state = raw.strip().lower()
    if not state:
        return False
    if any(flag in state for flag in _UNSCHEDULABLE_FLAGS):
        return False
    # Strip separators/digits so the long ("allocated", "powered_down") and
    # short ("alloc") spellings both resolve to the same key.
    return re.sub(r"[^a-z]", "", state) in _ALLOCATABLE_NODE_STATES


def fetch_partitions() -> list[dict[str, Any]]:
    if not is_tool_available("sinfo"):
        # Demo data only under SLURMATE_MOCK; on a real cluster whose sinfo is
        # missing/unrunnable, return nothing (the picker lets the user type a
        # name) rather than fake partitions that can't be submitted to.
        return list(MOCK_PARTITIONS) if _force_mock() else []

    stdout, stderr, rc = _run_command(
        # %T (node state) is appended LAST, after %G: sinfo emits one row per
        # partition+state group, and without the state every down/drained node is
        # summed into the node count, so a partition holding nothing but dead
        # nodes is offered as live capacity. It goes after %G because GRES is the
        # only field that could plausibly carry a stray separator, and keeping it
        # in the final split position leaves it intact.
        ["sinfo", "-h", "-o", "%P|%l|%D|%a|%c|%m|%G|%T"]
    )
    if rc != 0:
        _note_cluster_error(stderr)
        return []

    partitions: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        parts = line.strip().split("|", 7)
        if len(parts) < 5:
            continue
        raw_name = parts[0].strip()
        # sinfo marks the site default partition with a trailing "*" — free
        # information the picker needs to rank it first (no extra scontrol call).
        is_default = raw_name.endswith("*")
        name = raw_name.rstrip("*")
        timelimit = parts[1].strip() if len(parts) > 1 else ""
        nodes = _safe_int(parts[2]) if len(parts) > 2 else 0
        state = parts[3].strip().lower() if len(parts) > 3 else "up"
        cpus_raw = parts[4].strip() if len(parts) > 4 else ""
        cpus = _safe_int(cpus_raw)
        mem_raw = parts[5].strip() if len(parts) > 5 else ""
        # sinfo appends "+" to %c/%m when the nodes in this group are NOT all the
        # same: the printed number is the LOWEST, not the partition maximum.
        # Reading it as a maximum inverts the meaning and makes the limit warning
        # claim a ceiling that some node exceeds.
        heterogeneous = cpus_raw.endswith("+") or mem_raw.endswith("+")
        gres_raw = parts[6].strip() if len(parts) > 6 else ""
        node_state = parts[7].strip() if len(parts) > 7 else ""
        # A site whose sinfo gives us no state column must not be read as "every
        # node is dead"; track whether we ever saw one and report unknown if not.
        state_known = bool(node_state)
        nodes_up = nodes if _is_allocatable_state(node_state) else 0

        gpu_types: list[str] = []
        has_gpu = False
        if gres_raw and gres_raw != "(null)":
            for match in re.finditer(r"gpu:([a-zA-Z0-9._-]+):\d+", gres_raw, re.IGNORECASE):
                gpu_types.append(match.group(1).replace("_", "-"))
            # Detect GPU presence even for count-only ("gpu:4") or typed-without-
            # count ("gpu:a100") GRES that the model regex above doesn't capture,
            # so a real GPU partition isn't misreported as CPU-only downstream.
            has_gpu = bool(re.search(r"gpu[:\d]", gres_raw, re.IGNORECASE))
        gpus_per_node = _parse_gpu_count(gres_raw)

        if name not in partitions:
            partitions[name] = {
                "name": name,
                # Sum node counts: sinfo emits one row per partition+state group,
                # so a partition with idle/mix/alloc nodes spans several rows;
                # max() would report only the largest single group's count.
                "nodes": nodes,
                # Same sum, restricted to states that can actually run a job.
                "nodes_up": nodes_up,
                "_state_known": state_known,
                "state": state,
                "is_default": is_default,
                "cpus_per_node": cpus,
                "mem_per_node_mb": _parse_mem_to_mb(mem_raw) if mem_raw else 0,
                # True when the cpu/mem figures are floors rather than ceilings.
                "heterogeneous": heterogeneous,
                "gpu_types": gpu_types,
                "has_gpu": has_gpu,
                # GPUs a single node advertises — the analogue of cpus_per_node,
                # and until now the one advertised resource with no limit check.
                "gpus_per_node": gpus_per_node,
                # Keep "infinite" rather than nulling it: unbounded is a fact
                # the time check can act on, and None means "unknown" here.
                "timelimit": timelimit or None,
            }
        else:
            p = partitions[name]
            p["nodes"] += nodes
            p["nodes_up"] += nodes_up
            p["_state_known"] = p["_state_known"] or state_known
            p["is_default"] = p["is_default"] or is_default
            # cpus/mem are per-node capacities — keep the max across configs.
            p["cpus_per_node"] = max(p["cpus_per_node"], cpus)
            mem_mb = _parse_mem_to_mb(mem_raw) if mem_raw else 0
            p["mem_per_node_mb"] = max(p["mem_per_node_mb"], mem_mb)
            p["heterogeneous"] = p["heterogeneous"] or heterogeneous
            # sorted(), not list(set()): a partition spanning several sinfo rows
            # merged its GPU types through a set, and Python's per-process string
            # hash randomisation made the order differ between runs. That order is
            # user-visible in the picker's "GPU:[a100,v100]" label and in the
            # "not in partition list (…)" error, so identical input produced
            # different output — measured at four orderings across eight runs.
            p["gpu_types"] = sorted(set(p["gpu_types"] + gpu_types))
            p["has_gpu"] = p["has_gpu"] or has_gpu
            p["gpus_per_node"] = max(p["gpus_per_node"], gpus_per_node)

    for p in partitions.values():
        # "Unknown" (None), not 0: a site whose sinfo never gave us a state
        # column has told us nothing about usability, and reporting 0 there would
        # mark every partition dead. Consumers treat None as "don't filter".
        if not p.pop("_state_known"):
            p["nodes_up"] = None

    return list(partitions.values())


def _fetch_all_partition_names_uncached() -> set[str]:
    """Every partition name the controller knows, hidden ones included.

    Deliberately wider than :func:`fetch_partitions`, which runs a plain ``sinfo``
    so the picker offers what the user can see. Rejecting a user-supplied
    ``--partition`` needs the widest list available, or a hidden-but-submittable
    partition (Slurm's ``Hidden=YES`` is a display setting, not an ACL) gets
    reported as "no such partition on this cluster".

    An empty set means "could not determine", never "this cluster has none" —
    callers must skip validation rather than reject everything.
    """
    if not is_tool_available("sinfo"):
        return {str(p["name"]) for p in MOCK_PARTITIONS} if _force_mock() else set()

    names: set[str] = set()
    stdout, _, rc = _run_command(["sinfo", "-a", "-h", "-o", "%P"], timeout=_ADVISORY_TIMEOUT)
    if rc == 0:
        for line in stdout.splitlines():
            name = line.strip().rstrip("*")
            if name:
                names.add(name)
    if not names:
        # -a is not universally accepted by very old sinfo builds; fall back to
        # the picker's own list rather than validating against nothing.
        names = {str(p["name"]) for p in fetch_partitions()}
    return names


def fetch_public_partitions(all_parts: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return only publicly-usable partitions.

    Pass ``all_parts`` (a prior ``fetch_partitions()`` result) to avoid a
    redundant ``sinfo`` call — the partition step fetches it once and shares it.
    """
    if not is_tool_available("sinfo") or not is_tool_available("scontrol"):
        return [p for p in MOCK_PARTITIONS if p.get("is_public")] if _force_mock() else []

    stdout, _, rc = _run_command(["scontrol", "show", "partition", "-o"])
    if rc != 0:
        return []

    partition_lines: dict[str, str] = {}
    for line in stdout.splitlines():
        name = _extract_token(line, "PartitionName")
        if name:
            partition_lines[name] = line

    if all_parts is None:
        all_parts = fetch_partitions()
    result: list[dict[str, Any]] = []
    for part in all_parts:
        name = part["name"]
        scontrol_line = partition_lines.get(name, "")
        allow_accounts = _extract_token(scontrol_line, "AllowAccounts")
        hidden = _extract_token(scontrol_line, "Hidden")
        state = _extract_token(scontrol_line, "State")

        # "Public" = usable by anyone: open to all accounts, not hidden, and up.
        # (AllowGroups gating can't be evaluated here without the caller's groups;
        # such partitions still appear under the picker's "[Private]"/"[Custom]"
        # paths, so nothing usable is truly hidden — only mis-ranked.)
        is_public = (
            allow_accounts.upper() == "ALL"
            and hidden.upper() != "YES"
            and state.upper() in ("", "UP")
        )
        p = dict(part)
        p["is_public"] = is_public
        if is_public:
            result.append(p)

    return result


def fetch_qos_acl(partition: str) -> dict[str, list[str]]:
    """A partition's QoS ACL as ``{"allow": [...], "deny": [...]}``.

    Slurm expresses this two ways and a site picks one: an explicit ``AllowQos``
    list, or ``AllowQos=ALL`` plus a ``DenyQos`` exclusion list. Reading only the
    allow side means a deny-list site's ``ALL`` expands to every QoS on the
    cluster — including the ones the partition forbids — which is the same defect
    as offering partitions the user has no association for.
    """
    if not is_tool_available("scontrol"):
        return {"allow": [], "deny": []}

    stdout, _, rc = _run_command(["scontrol", "show", "partition", partition, "-o"])
    if rc != 0:
        return {"allow": [], "deny": []}

    return {
        "allow": _split_csv(_normalize_null(_extract_token(stdout, "AllowQos"))),
        "deny": _split_csv(_normalize_null(_extract_token(stdout, "DenyQos"))),
    }


def fetch_qos_for_partition(partition: str) -> list[str]:
    """The partition's ``AllowQos`` list (see :func:`fetch_qos_acl`)."""
    return fetch_qos_acl(partition)["allow"]


MOCK_QOS = ["normal", "high", "express", "gpu", "interactive"]


def _fetch_known_qos_uncached() -> list[str]:
    """Fetch all QoS names known to the system via sacctmgr.

    Returns the demo ``MOCK_QOS`` only in mock mode. When sacctmgr is genuinely
    unavailable (or errors, or lists nothing), returns ``[]`` — an *unknown* set,
    not the demo names — so the TUI can tell "QoS set unknown" apart from a real
    list and skip filtering live ``AllowQos`` against a demo fallback (which
    would otherwise silently drop real, lab-specific QoS names).
    """
    if _force_mock():
        return list(MOCK_QOS)
    if not is_tool_available("sacctmgr"):
        return []

    stdout, _, rc = _run_command(
        ["sacctmgr", "show", "qos", "-P", "format=Name", "--noheader"],
        timeout=_ADVISORY_TIMEOUT,
    )
    if rc != 0:
        return []

    qos: list[str] = []
    for line in stdout.splitlines():
        name = line.strip()
        if name:
            qos.append(name)
    return qos


def fetch_gpu_types_for_partition(partition: str) -> list[str]:
    """Every GPU model a partition offers, from typed GRES and node features."""
    sources = fetch_gpu_type_sources(partition)
    return sorted(set(sources["typed"]) | set(sources["feature"]))


def fetch_gpu_type_sources(partition: str) -> dict[str, list[str]]:
    """GPU models a partition offers, split by **how** they can be requested.

    Returns ``{"typed": [...], "feature": [...], "constraint": [...]}``:

    - ``typed``   — seen in a real ``gpu:MODEL:N`` GRES, so requestable as a GRES
      type (``--gres=gpu:MODEL:N``, ``--gpus=MODEL:N``, …).
    - ``feature`` — only found in the node's *feature* list, because the node's
      GRES is count-only (``gpu:4``). Such a model is **not** a GRES type: asking
      for it with ``--gres=gpu:MODEL:N`` makes Slurm reject the job outright
      ("Requested node configuration is not available"). The only way to request
      it is ``--gres=gpu:N`` plus ``--constraint=MODEL`` — i.e. ``gpu_format
      "constraint"``.
    - ``constraint`` — models that some node in the partition advertises as a
      node *feature*, so ``--constraint=MODEL`` can name them. This is the
      mirror-image fact, and it is not the complement of ``typed``: a model can
      be both (midway3's ``a100`` is a feature *and* a GRES type on some nodes),
      or neither-but-typed — a cluster whose nodes carry typed GRES and no
      features at all lists every model under ``typed`` and **none** here, which
      makes ``gpu_format "constraint"`` impossible there. An empty list is
      therefore a measured answer, not a missing one.

    Keeping the three apart is what lets callers warn about (or avoid) a format
    that cannot express the model they picked; ``fetch_gpu_types_for_partition``
    flattens typed+feature for pickers.
    """
    if not is_tool_available("sinfo"):
        if not _force_mock():
            return {"typed": [], "feature": [], "constraint": []}
        # In mock mode, prefer the specific partition's GPU types so a demo
        # doesn't claim every partition offers all GPU models; fall back to the
        # full list only for an unknown/manually-typed partition name. Mock types
        # stand in for typed GRES *and* for node features, so demos and tests see
        # no format mismatch in either direction.
        for p in MOCK_PARTITIONS:
            if p["name"] == partition:
                mock_types = [str(g) for g in p["gpu_types"]]
                return {"typed": mock_types, "feature": [], "constraint": mock_types}
        return {
            "typed": list(MOCK_GPU_TYPES),
            "feature": [],
            "constraint": list(MOCK_GPU_TYPES),
        }

    stdout, _, rc = _run_command(
        ["sinfo", "-h", "-N", "-p", partition, "-o", "%f|%G"]
    )
    if rc != 0:
        return {"typed": [], "feature": [], "constraint": []}

    # Pass 1: collect typed GPU models from gpu:MODEL:N across all nodes,
    # and stash the raw lines for a second pass.
    typed_models: set[str] = set()
    lines_data: list[tuple[str, str]] = []
    for line in stdout.splitlines():
        parts = line.strip().split("|", 1)
        if len(parts) < 2:
            continue
        features, gres = parts[0].strip(), parts[1].strip()
        lines_data.append((features, gres))
        gres_match = re.search(
            r"gpu:([a-z0-9._-]+):\d+", f"{features},{gres}", re.IGNORECASE
        )
        if gres_match:
            candidate = gres_match.group(1).replace("_", "-")
            if candidate.lower() not in {"gpu", "mps", "shard"}:
                typed_models.add(candidate)

    # Pass 2: collect every typed model on each node (a node can advertise more
    # than one, e.g. "gpu:a100:2,gpu:v100:2" — a single re.search would drop the
    # second). Only when a node has no typed model do we fall back to feature
    # scanning, preferring corroboration against the typed models seen elsewhere.
    typed: set[str] = set()
    feature: set[str] = set()
    # Every feature token any node in the partition advertises. Collected for ALL
    # nodes, not only the count-only ones, because "can --constraint name this
    # model?" is a question a typed-GRES node answers too — and answers "no" on a
    # cluster that publishes no features at all.
    feature_tokens: set[str] = set()
    for features, gres in lines_data:
        if features and features != "(null)":
            feature_tokens.update(t.strip() for t in features.split(",") if t.strip())
        text = f"{features},{gres}"
        typed_here = [
            m.group(1).replace("_", "-")
            for m in re.finditer(r"gpu:([a-z0-9._-]+):\d+", text, re.IGNORECASE)
            if m.group(1).lower() not in {"gpu", "mps", "shard"}
        ]
        if typed_here:
            typed.update(typed_here)
            continue
        gpu_type = _detect_gpu_type(features, gres, known_models=typed_models)
        if gpu_type and gpu_type != "gpu":
            feature.add(gpu_type)
    # A model corroborated by a typed GRES somewhere in the partition is
    # requestable as a GRES type, so it belongs in "typed" even when this node
    # only advertised it as a feature.
    feature -= typed
    # Which of the models this partition offers are also node features. Compared
    # case- and separator-insensitively because the two sources disagree on both
    # ("gpu:h100:8" vs. a "H200" feature, "rtx_6000" vs. "rtx-6000") while naming
    # the same card; the *reported* spelling stays whatever the model list used,
    # since the case-mismatch warning above is what handles a real difference.
    def _norm(text: str) -> str:
        return text.lower().replace("_", "-")

    advertised = {_norm(t) for t in feature_tokens}
    constraint = sorted(m for m in (typed | feature) if _norm(m) in advertised)
    return {
        "typed": sorted(typed),
        "feature": sorted(feature),
        "constraint": constraint,
    }


def _extract_first_json(text: str) -> Any:
    """Return the first parseable JSON object in ``text``, or None.

    A login shell may print a banner before the JSON, and that banner can itself
    contain braces — so a naive first-``{``/last-``}`` slice can capture garbage.
    Walk each ``{`` and try to decode from there, tolerating trailing output.
    """
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            return None
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            idx = start + 1


def fetch_conda_envs(modules: list[str] | None = None) -> list[str]:
    """List conda environment names/paths usable with ``conda activate``.

    Conda is frequently provided by a module (e.g. ``module load anaconda``)
    rather than being on ``PATH`` directly, so when ``modules`` are given we load
    them first — inside a login shell where ``module`` is defined — and then run
    ``conda info --json``. Using ``info`` (not ``env list``) gives the authoritative
    ``root_prefix`` and ``envs_dirs``, so the base env is labelled ``base`` (not by
    its install-dir basename) and a ``--prefix`` env outside the envs dirs is kept
    as a full path (activatable), instead of a bare basename that can't activate.
    """
    if _force_mock():
        return list(MOCK_CONDA_ENVS)

    prefix = ""
    if modules:
        # Quote each module token so a name with shell metacharacters can't break
        # out of (or inject into) the `bash -lc` string.
        names = " ".join(
            shlex.quote((m[:-9] if m.endswith("(default)") else m).strip())
            for m in modules
            if m and m.strip()
        )
        if names.strip():
            prefix = f"module load {names} >/dev/null 2>&1; "

    stdout, _, rc = _run_command(
        ["bash", "-lc", f"{prefix}conda info --json 2>/dev/null"]
    )
    if rc != 0:
        # Real failure (conda/module not found): return nothing rather than
        # misleading mock names so the user can just type their env/path.
        return []

    data = _extract_first_json(stdout)
    if not isinstance(data, dict):
        return []
    root = str(data.get("root_prefix", "")).rstrip("/")
    envs_dirs = {str(d).rstrip("/") for d in data.get("envs_dirs", []) if d}
    env_names: list[str] = []
    for raw_env in data.get("envs", []):
        p = str(raw_env).rstrip("/")
        if not p:
            continue
        if root and p == root:
            env_names.append("base")
        elif os.path.dirname(p) in envs_dirs:
            # A named env under an envs dir — activatable by its basename.
            env_names.append(os.path.basename(p))
        else:
            # A --prefix env elsewhere — only the full path activates it.
            env_names.append(p)
    # De-dup while preserving order.
    return list(dict.fromkeys(env_names))


def fetch_available_modules() -> list[str]:
    """Parse `module avail` output into a sorted unique list of module names."""
    # Mirror every other fetcher: never shell out (into a login shell that
    # sources the user's profile) when mock mode is forced.
    if _force_mock():
        return list(MOCK_MODULES)

    stdout, stderr, rc = _run_command(["bash", "-lc", "command -v module && module -t avail 2>&1"])
    output = stdout + stderr
    if rc != 0:
        return []

    modules: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        # `module -t avail` prints filesystem headers ("/opt/apps/modulefiles:")
        # on their own lines; skip them so they don't pollute the module list.
        if stripped.endswith(":"):
            continue
        for mod in stripped.split():
            # Strip "(default)" annotation that the module system appends
            if mod.endswith("(default)"):
                mod = mod[:-9].strip()
            # Lmod terse output can carry extras a Tcl-modules parser wouldn't: an
            # alias annotation "(@name)", a tag marker like "(D)"/"<F>", and a
            # trailing "/" on a family short-name ("gcc/" — loadable as "gcc").
            if mod.startswith("(@") or (mod.startswith("<") and mod.endswith(">")) \
                    or (mod.startswith("(") and mod.endswith(")")):
                continue
            mod = mod.rstrip("/")
            # Drop the leading `command -v module` probe output — either the
            # bare "module" function name or its resolved path (/usr/bin/module).
            if not mod or mod == "module" or mod.endswith("/module"):
                continue
            modules.add(mod)
    return sorted(modules)


def current_username() -> str:
    """The caller's username, or "" — public alias of :func:`_current_username`."""
    return _current_username()


def _current_username() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER") or os.environ.get("LOGNAME") or ""


def _fetch_user_accounts_uncached() -> list[str]:
    """Fetch the Slurm accounts the current user may submit under.

    Uses ``sacctmgr show assoc user=<me>`` (associations), NOT ``show user``:
    the bare ``user`` entity doesn't populate ``Account`` and isn't scoped to
    the caller (it lists every visible user), so it returns thousands of blank
    lines on a real cluster and the picker silently falls back to mock accounts
    the user can't actually charge to.
    """
    if not is_tool_available("sacctmgr"):
        # Demo accounts only under SLURMATE_MOCK. On a real cluster without
        # sacctmgr, return nothing rather than fake accounts the user can't
        # charge to — the account field is free-text, so they type their own.
        return list(MOCK_ACCOUNTS) if _force_mock() else []

    user = _current_username()
    if not user:
        return []

    stdout, _, rc = _run_command(
        ["sacctmgr", "show", "assoc", f"user={user}", "-P",
         "format=Account", "--noheader"],
        timeout=_ADVISORY_TIMEOUT,
    )
    if rc != 0:
        return []

    accounts: list[str] = []
    for line in stdout.splitlines():
        a = line.strip()
        if a:
            accounts.append(a)
    # De-dupe while preserving order; a user is often associated to the same
    # account through several partitions/QoS, yielding duplicate rows.
    accounts = list(dict.fromkeys(accounts))
    return accounts


def fetch_user_partitions() -> set[str] | None:
    """Partitions the current user holds a Slurm *association* for.

    The gate on most clusters is not the partition ACL. Private PI partitions
    routinely advertise ``AllowGroups=ALL AllowAccounts=ALL`` and still reject
    every submission with *"Invalid account or account/partition combination
    specified"* — what actually decides is the association list in ``sacctmgr``.
    Filtering the picker on the partition ACL therefore cannot work; filtering on
    associations can.

    Returns ``None`` — meaning "no filtering is justified" — when:

    - ``sacctmgr`` is missing, errors, or lists nothing, **or**
    - any association row has an *empty* Partition field. Blank means "every
      partition for that account", i.e. a wildcard, not no access. Sites that put
      the gate on the account rather than the partition look like this on every
      row, and must not have their entire partition list filtered away.

    A real set comes back only when every row names a partition, i.e. the site
    genuinely scopes its associations per partition.
    """
    if not is_tool_available("sacctmgr"):
        return None

    user = _current_username()
    if not user:
        return None

    stdout, _, rc = _run_command(
        ["sacctmgr", "show", "assoc", f"user={user}", "-P",
         "format=Account,Partition", "--noheader"],
        timeout=_ADVISORY_TIMEOUT,
    )
    if rc != 0:
        return None

    named: set[str] = set()
    saw_row = False
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        saw_row = True
        part = fields[1].strip() if len(fields) > 1 else ""
        if not part:
            return None  # wildcard row — the user is not partition-scoped
        named.add(part)
    return named if (saw_row and named) else None


# Partitions that exist for the scheduler rather than for user jobs. "cron" is
# Slurm's own scrontab partition and is near-universal; the rest are common
# site names for the same idea. Matched case-insensitively against the exact
# partition name, never as a substring (a real "system-gpu" must not be caught).
_SYSTEM_PARTITION_NAMES = frozenset({
    "cron", "system", "service", "maint", "maintenance", "admin",
})

# A Slurm hostlist range, e.g. the "[1-2]" in "midway2-login[1-2]".
_HOSTLIST_RANGE_RE = re.compile(r"\[[^\]]*\]")


def _all_nodes_are_login(node_expr: str) -> bool:
    """Whether every node in a Slurm hostlist expression is a login node.

    A cron/service partition's nodes *are* the login nodes, which is a
    structural signal available on any site — unlike the partition's name.
    Bracketed ranges are collapsed first so ``dali-login[1-2],midway2-login[1-2]``
    splits into two host patterns rather than four comma-separated fragments.
    """
    expr = _normalize_null(node_expr)
    if not expr:
        return False
    tokens = [t for t in _HOSTLIST_RANGE_RE.sub("", expr).split(",") if t.strip()]
    return bool(tokens) and all("login" in t.lower() for t in tokens)


def fetch_system_partitions() -> set[str]:
    """Partition names that are for the scheduler, not for user jobs.

    Two cluster-agnostic signals: a small name deny-list, and a partition whose
    nodes are all login nodes. Both are advisory — callers de-prioritise these
    rather than hiding them, since a site can legitimately name a real partition
    anything at all.
    """
    system: set[str] = set()
    if not is_tool_available("scontrol"):
        return system

    stdout, _, rc = _run_command(["scontrol", "show", "partition", "-o"])
    if rc != 0:
        return system
    for line in stdout.splitlines():
        name = _extract_token(line, "PartitionName")
        if not name:
            continue
        if name.lower() in _SYSTEM_PARTITION_NAMES or _all_nodes_are_login(
            _extract_token(line, "Nodes")
        ):
            system.add(name)
    return system


def _near_misses(name: str, candidates: Iterable[str], limit: int = 3) -> list[str]:
    """Plausible intended spellings of ``name`` among ``candidates``, best first.

    Combines edit-distance matches with prefix/substring ones: a user carrying
    ``caslake`` to a cluster that has ``caslake-gpu`` is a substring hit that
    difflib alone scores below the cutoff, and ``bigmem`` vs ``bigmem2`` is the
    single most common cross-cluster near-miss there is.
    """
    pool = [c for c in dict.fromkeys(candidates) if c]
    if not name or not pool:
        return []
    lowered = name.lower()
    ranked = difflib.get_close_matches(lowered, [c.lower() for c in pool], n=limit, cutoff=0.6)
    by_lower = {c.lower(): c for c in pool}
    out = [by_lower[r] for r in ranked if r in by_lower]
    for c in pool:
        if len(out) >= limit:
            break
        cl = c.lower()
        if c not in out and (cl.startswith(lowered) or lowered.startswith(cl) or lowered in cl):
            out.append(c)
    return out[:limit]


def _unknown_target_message(
    kind: str, name: str, known: Iterable[str], default: str = "", listed: int = 8,
    plural: str = "",
) -> str:
    """A multi-line 'no such X on this cluster' message with suggestions."""
    pool = [k for k in dict.fromkeys(known) if k]
    plural = plural or f"{kind}s"
    lines = [f"no {kind} '{name}' on this cluster."]
    suggestions = _near_misses(name, pool)
    if suggestions:
        labelled = [f"{s} (default)" if s == default and default else s for s in suggestions]
        lines.append(f"Did you mean: {', '.join(labelled)}?")
    # Show the default first so a user with no idea what to pick has one.
    ordered = ([default] if default and default in pool else []) + [
        p for p in sorted(pool) if p != default
    ]
    shown = ordered[:listed]
    more = f", ... (+{len(ordered) - len(shown)} more)" if len(ordered) > len(shown) else ""
    lines.append(f"This cluster's {plural}: {', '.join(shown)}{more}")
    return "\n".join(lines)


def validate_cluster_targets(
    partition: str | None,
    account: str | None = None,
    *,
    qos: str | None = None,
    constraint: str | None = None,
    known_partitions: Iterable[str] | None = None,
    known_accounts: Iterable[str] | None = None,
    known_qos: Iterable[str] | None = None,
    known_features: Iterable[str] | None = None,
    default_partition: str = "",
) -> list[tuple[str, str]]:
    """Check ``--partition`` / ``--account`` / ``--qos`` against this cluster.

    This is the check that makes a generated script *correct for the cluster you
    are on*, which is the whole point of generating it. Without it, carrying a
    script or a habit from another site produces a complete, confident, entirely
    unsubmittable script with ``rc=0``.

    Returns ``(level, message)`` tuples, ``"error"`` for a name the cluster does
    not have. Deliberately silent when the cluster's list could not be read
    (empty ``known_*``): an unreadable ``sinfo``/``sacctmgr`` must not turn into
    "your partition doesn't exist".

    Both lookups are pure set membership against lists the caller already has;
    pass them in to avoid re-running ``sinfo``/``sacctmgr``.
    """
    out: list[tuple[str, str]] = []

    parts = [p for p in (known_partitions or []) if p]
    if partition and parts and partition not in parts:
        out.append((
            "error",
            _unknown_target_message("partition", partition, parts, default_partition),
        ))

    accounts = [a for a in (known_accounts or []) if a]
    if account and accounts and account not in accounts:
        out.append((
            "error",
            _unknown_target_message("account", account, accounts),
        ))

    # QoS is the third name Slurm resolves against its own database, and it fails
    # the same way: `--qos` from another site produced a complete script, rc=0,
    # and an "Invalid qos specification" from the controller later. Existence
    # only — whether a QoS is *permitted on this partition* is set by
    # AllowQos/DenyQos, and a site using DenyQos would make that check reject
    # valid combinations.
    # ``None`` means the feature list could not be read and nothing can be said.
    # An empty list is a *different* fact: sinfo answered, and this cluster
    # advertises no features at all, so any plain -C name matches no node here.
    # Reading both as "stay quiet" is why a bad --constraint sailed through on a
    # cluster whose sinfo reported (null) for every node.
    features = None if known_features is None else [f for f in known_features if f]
    # Only a single plain name is checkable; a feature expression is not a set
    # member. Slurm's own verdict on a bad one is a hard refusal
    # ("Invalid feature specification"), so this is an error like the rest.
    if constraint and features is not None and _PLAIN_FEATURE_RE.match(constraint):
        if not features:
            out.append((
                "error",
                f"this cluster advertises no node features, so the constraint "
                f"'{constraint}' (Slurm -C) cannot match any node.",
            ))
        elif constraint not in features:
            out.append((
                "error",
                _unknown_target_message(
                    "node feature", constraint, features, plural="node features"
                ),
            ))

    qos_names = [q for q in (known_qos or []) if q]
    if qos and qos_names and qos not in qos_names:
        out.append((
            "error",
            _unknown_target_message("QoS", qos, qos_names, plural="QoS names"),
        ))

    return out


def _format_eta(seconds: int) -> str:
    if seconds <= 0:
        return "now"
    if seconds < 120:
        return f"~{seconds}s"
    if seconds < 3600:
        return f"~{seconds // 60}min"
    if seconds < 86400:
        return f"~{seconds // 3600}h"
    return f"~{seconds // 86400}d"


# Derive the mock label from _format_eta so the demo display matches the live
# formatter exactly (e.g. "~1h", not a hand-written "~1 hour").
MOCK_QUEUE_INFO = {
    "running": 12,
    "pending": 5,
    "eta_seconds": 3600,
    "eta_label": _format_eta(3600),
    "feasible": True,
    "reason": "",
}


# `sbatch --test-only` prints its verdict on stderr, ending with the scheduler's
# own placement: "Job 123 to start at 2026-07-29T18:07:28 using 1 processors on
# nodes midway3-0008 in partition caslake".
# How far into the past a reported start time may be before it stops meaning
# "now" and starts meaning "these two clocks are not the same clock".
_CLOCK_SKEW_TOLERANCE_S = 120

_TEST_ONLY_START_RE = re.compile(r"to start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")

# How `sbatch --test-only` reports a refusal. The site job_submit plugin's own
# explanation ("Reason: Invalid account [foo]") is far more specific than Slurm's
# generic "allocation failure: Access/permission denied", so both are matched and
# the specific one wins.
_TEST_ONLY_REASON_RE = re.compile(
    r"^\s*(?:sbatch:\s*)?(?:error:\s*)?Reason:\s*(.+?)\s*$", re.MULTILINE
)
_TEST_ONLY_FAILURE_RE = re.compile(r"^\s*allocation failure:\s*(.+?)\s*$", re.MULTILINE)

# Slurm often names the exact limit on a line of its own, immediately above a
# generic bundle that does not say which half was violated. Measured on Booth's
# Mercury: "sbatch: error: QOSMaxSubmitJobPerUserLimit" above "Job violates
# accounting/QOS policy (job submit limit, user's size and/or time limits)". The
# token is the only thing that distinguishes "you already have a job queued"
# (wait) from "this request is too big" (fix it), and it was being discarded.
# Limit names are a single CamelCase word, which is what keeps this from matching
# prose like "error: Batch job submission failed: ...".
_TEST_ONLY_LIMIT_RE = re.compile(
    r"^\s*sbatch:\s*error:\s*([A-Za-z][A-Za-z0-9]{5,})\s*$", re.MULTILINE
)

# sbatch rejecting the SCRIPT'S OWN directives, before the controller sees it.
#
# A third case beside the two `_test_only_refusal` was written for, and it fell
# into the wrong one. It carries neither `allocation failure:` nor `Reason:`, so
# "positive evidence or nothing" -- correctly -- produced no refusal, and the
# caller then fell through to the free-capacity estimate and reported a script
# sbatch had just rejected as `ETA: now`. That is the whole point of `--dry-run`
# answered backwards.
#
# It is still positive evidence, just of a different thing: these are getopt's
# and sbatch's own messages about an option in the file, they name the offending
# token, and none of them is emitted for an unreachable controller or a broken
# binary -- which is what the "nothing" branch exists to protect. A malformed
# directive is also permanently malformed, so `refusal_is_permanent` classifies
# it as such rather than inviting a retry.
# `sbatch:` is MANDATORY, and the token it names is captured.
#
# Optional, the prefix made any line starting with the phrase a permanent
# refusal -- `"   unrecognized option -- policy change 2026"` from a site banner
# was enough. Slurm's clients always prefix their own diagnostics.
_TEST_ONLY_USAGE_RE = re.compile(
    r"^\s*sbatch:\s*(?:error:\s*)?("
    r"unrecognized option\s+(?P<a>\S+)"
    r"|invalid option\s*--\s*(?P<b>\S+)"
    r"|option\s+(?P<c>\S+)\s+requires an argument"
    r"|option requires an argument\s*--\s*(?P<d>\S+)"
    r"|unrecognized argument\s+(?P<e>\S+)"
    r")\s*$",
    re.MULTILINE | re.IGNORECASE,
)

#: Options slurmate passes to `sbatch` ITSELF to run the probe, as opposed to
#: directives it wrote into the script.
#:
#: sbatch rejecting one of these is not a statement about the job -- it is a
#: broken or wrapped local `sbatch`, which is the "controller unreachable" case
#: and must fall through to the estimate. A site wrapper that forwards a
#: whitelist (this module's `parse_submitted_job_id` documents such wrappers)
#: can accept `--parsable` and reject `--test-only`, and reading that as "your
#: job can never run" refused a submit the same wrapper would have accepted.
#: Matched by EQUALITY on the de-quoted option name, never as a substring.
#:
#: `any(own in named for own in ...)` excused any directive whose name merely
#: contained one of these: `--wrapper` contains `--wrap`, `--test-only-ish`
#: contains `--test-only`. Both then got "no verdict", so `--dry-run` fell
#: through to the free-capacity estimate and printed an ETA for a script sbatch
#: had just rejected -- which is SM-30 itself, re-opened for a class of names.
#:
#: The list covers every option slurmate puts on the probe command line, not
#: only the three that identify the probe. All of them are directives slurmate
#: MANAGES and validates -- `managed_custom_flags` refuses a custom flag that
#: duplicates one -- so a rejection naming one of them is evidence about the
#: local `sbatch` (a site wrapper forwarding a whitelist) and not about the job.
_PROBE_OWN_OPTIONS = frozenset({
    "--test-only", "--parsable", "--wrap",
    "-p", "--partition", "-q", "--qos", "-A", "--account",
    "-N", "--nodes", "-c", "--cpus-per-task", "-t", "--time",
    "--mem", "--gres", "--array", "--constraint",
})


def _probe_own_option(named: str) -> bool:
    """Whether a rejected token is one slurmate itself passed to the probe.

    Handles both spellings the messages use: `unrecognized option '--gres=gpu:2'`
    quotes the whole option, while getopt's `invalid option -- q` names a short
    option by its BARE letter, with no dash for the set below to match.
    """
    name = named.strip().strip("'\"").split("=", 1)[0].strip()
    if not name:
        return False
    return name in _PROBE_OWN_OPTIONS or (
        not name.startswith("-") and "-" + name in _PROBE_OWN_OPTIONS
    )

# Anything a `--test-only` run prints as `sbatch: error: <text>` is the site's
# own text: Slurm puts its verdict on the unprefixed `allocation failure:` line,
# and its scaffolding ("Batch job submission failed: …") belongs to a real
# submit. So these lines are the job_submit plugin talking. Read only as a
# fallback, and only when the plugin said exactly ONE thing — Booth's Pythia
# rejects a batch job on its default partition with a single sentence and leaves
# Slurm's half as the contentless "Unspecified error", whereas midway3's plugin
# writes a six-line block whose reason line is already matched above. With
# several lines there is no way to tell which is the reason, so nothing is
# claimed, exactly as before.
_TEST_ONLY_SITE_LINE_RE = re.compile(
    r"^\s*sbatch:\s*error:\s*(.+?)\s*$", re.MULTILINE
)
_SLURM_OWN_ERROR_PREFIXES = ("batch job submission failed",)

# A refusal that describes *this moment* rather than the request. Advisory only:
# the script is fine and will be accepted once the condition clears, so treating
# one of these as fatal would fail a CI run for having a job already queued.
_TRANSIENT_REFUSAL_MARKERS = (
    "maxsubmitjob",      # QOSMaxSubmitJobPerUserLimit, AssocMaxSubmitJobLimit
    "maxjobsper",        # QOSMaxJobsPerUserLimit — a running-count cap
    "not available now",  # Slurm's own "now" marks the transient variant
    "are down",
    "drained",
)

# A refusal that describes the *request*: nothing about waiting will fix it, so
# it is worth blocking a submit and failing --print over. Deliberately a
# whitelist of measured wordings — anything unrecognised stays advisory, because
# a new gate that guesses "permanent" blocks jobs that would have run.
_PERMANENT_REFUSAL_MARKERS = (
    # An option sbatch cannot parse is not going to start parsing later.
    "rejected an option in the script",
    "invalid account",
    "invalid qos",
    "invalid partition",
    "invalid feature",
    "invalid job array",
    "invalid gres",
    "invalid generic resource",
    "requested node configuration is not available",
    "unsupported by configured selecttype",
    "access/permission denied",
    "user's group not permitted",
    # Both measured on Mercury, and both are ordinary mistakes rather than exotic
    # ones: asking for more nodes than the partition allows ("--nodes 2" where the
    # QoS caps it at 1) and a time limit past the partition's maximum. Neither
    # matched the list above, so both were reported as conditions that clear on
    # their own — a confident false claim about a job that can never run.
    "node count specification invalid",
    "requested time limit is invalid",
    # The same two mistakes as above, worded the way an older controller words
    # them. Slurm 25.11 on Mercury answers "node count specification invalid";
    # Slurm 20.11.8 on midway3 answers "More nodes requested than permitted" for
    # the identical request, and "More processors requested than permitted" for
    # the CPU equivalent. Measured with `sbatch --test-only -p amd --cpus 9999`:
    # the label came out "refused" -- the fallback for a refusal whose permanence
    # is unknown -- where the README documents "never" for that very message.
    # Both are per-job ceilings (partition/QoS MaxCPUs, MaxNodes), so no amount
    # of waiting fits the request; a *PerUser* cap is worded as its own
    # QOSMax...PerUserLimit token and is caught by the transient list first.
    "more processors requested than permitted",
    "more nodes requested than permitted",
    # A directive that names something this cluster's *configuration* does not
    # have. Measured on midway3 (Slurm 20.11.8) by generating a script with
    # `--custom-sbatch` and handing it to `sbatch --test-only`:
    #   --nodelist=nosuchnode1     -> "Invalid node name specified"
    #   --exclude=nosuchnode1      -> "Invalid node name specified"
    #   --nodelist=beagle3-0001 -p amd
    #                              -> "Requested nodes not in this partition"
    #   --licenses=nosuchlic:1     -> "Invalid license specification"
    # Unclassified, all three produced the "slurmate cannot tell whether this
    # clears on its own" advisory and an exit status of 0 for a script sbatch had
    # just refused -- so `--print` reported success on a script that can never
    # run, which is the same defect the two markers above were added for.
    # These are the node/partition/license *tables*, not queue state: no amount
    # of waiting adds a node to a partition or a license to slurm.conf, exactly
    # as with "invalid partition"/"invalid feature" already listed above. A cap
    # that really does clear is worded as its own QOSMax.../Max...PerUser token
    # and is caught by the transient list first.
    "invalid node name specified",
    "requested nodes not in this partition",
    "invalid license specification",
)

# Slurm's own limit tokens split cleanly on one word. A "...PerJob" limit is a
# statement about the *request* — no amount of waiting makes a 7-day job fit a
# 6-hour MaxWall — while the "...PerUser"/"...PerAccount" count limits are about
# the moment and clear when something finishes. Measured: sbatch answers
# QOSMaxWallDurationPerJobLimit / QOSMaxCpuPerJobLimit for the former and
# QOSMaxSubmitJobPerUserLimit for the latter. This is what lets slurmate treat a
# site that *does* enforce its QoS wall limit as authoritative, instead of
# reimplementing the policy locally and being wrong on the sites that do not.
_PER_JOB_LIMIT_RE = re.compile(r"\b[A-Za-z]*perjob[A-Za-z]*limit\b")


def refusal_is_transient(reason: str) -> bool:
    """Whether a refusal is *known* to be about the moment rather than the request.

    The counterpart to :func:`refusal_is_permanent`, and deliberately not its
    negation: a refusal can be neither. Slurm has many wordings and this module
    has measured a handful, so "not recognised as permanent" must not become
    "safe to tell the user their script is fine and this will clear" — that is
    the same unfounded confidence as an ETA for a job the scheduler refused.
    Positive evidence, or the caller says it cannot tell.
    """
    text = " ".join(str(reason or "").lower().split())
    if not text:
        return False
    return any(marker in text for marker in _TRANSIENT_REFUSAL_MARKERS)


def refusal_is_permanent(reason: str) -> bool:
    """Whether a scheduler refusal is about the request, not the moment.

    ``sbatch --test-only`` answers two very different questions with the same
    non-zero exit: "this job is misconfigured" and "you cannot submit *right
    now*". Measured on Booth's Mercury, whose ``clay`` QoS allows one submitted
    job per user: a perfectly valid script is refused with
    ``QOSMaxSubmitJobPerUserLimit`` whenever another job is already queued.
    Blocking on that turns a wait into a failure — and in CI, into a red build
    caused by someone else's job.

    Unrecognised wordings are **not** permanent. The cost of guessing wrong in
    that direction is a wasted round-trip and Slurm's own error message; the cost
    of guessing wrong the other way is refusing to submit a job that would run.
    """
    text = " ".join(str(reason or "").lower().split())
    if not text:
        return False
    if any(marker in text for marker in _TRANSIENT_REFUSAL_MARKERS):
        return False
    if _PER_JOB_LIMIT_RE.search(text.replace(" ", "")):
        return True
    return any(marker in text for marker in _PERMANENT_REFUSAL_MARKERS)

# `sinfo -O CPUsState` renders as allocated/idle/other/total.
_CPUS_STATE_RE = re.compile(r"^\s*(\d+)/(\d+)/(\d+)/(\d+)\s*$")

# A node's GRES value: sum every `gpu[:type]:N`, ignoring the `(IDX:0-3)` suffix
# that GresUsed appends, and skipping mps/shard entries.
_NODE_GPU_RE = re.compile(r"(?:^|,)\s*(?:gres/)?gpu(?::[a-zA-Z0-9._-]+)?[:=](\d+)", re.IGNORECASE)

# (_UNSCHEDULABLE_FLAGS — the state flags marking nodes that will not take a
# normal job — is defined next to _is_allocatable_state above; both the
# partition-level node count and the node-level fit check read the same list.)


def _sum_node_gpus(gres: str) -> int:
    if not gres or gres.strip().lower() in ("(null)", "null", "n/a", ""):
        return 0
    return sum(int(n) for n in _NODE_GPU_RE.findall(gres))


def _scheduler_verdict(
    partition: str,
    req_nodes: int,
    cpus: int,
    mem_mb: int,
    gpus_per_node: int,
    gpu_type: str,
    time_limit: str,
    account: str,
    qos: str,
    array_spec: str = "",
    constraint: str = "",
    script: str = "",
) -> tuple[int | None, str]:
    """Slurm's own answer for this request: ``(seconds_until_start, refusal)``.

    Asks Slurm rather than modelling it. ``sbatch --test-only`` queues nothing but
    returns the backfill placement *and* runs the site's ``job_submit`` plugin,
    whose rules are not published anywhere — so this is the only estimate that can
    account for QOS caps, account limits and local policy.

    Exactly one of the two values is meaningful:

    - a start time means Slurm placed the job;
    - a non-empty refusal means Slurm **rejected** it — the request cannot be
      scheduled as written, so any ETA computed for it would be fiction. This is
      the case the older code threw away by collapsing "rejected" into the same
      ``None`` as "couldn't ask", which is how a 35x over-request ended up with a
      confident ``ETA: ~60s``.
    - ``(None, "")`` means the question could not be asked (no sbatch, unparsable
      output); the caller must fall through to its own estimate.
    """
    if not is_tool_available("sbatch"):
        return None, ""

    # Prefer handing Slurm the script it will actually receive. Reconstructing an
    # argv from the individual fields duplicates the builder, and every field the
    # reconstruction forgot produced a confident ETA for a job Slurm refuses:
    # `--array` was missing (so an over-large array read "~22h"), then
    # `--constraint` (a bogus feature read "~21h"), and the reconstruction also
    # rewrote every `--gpu-format` choice as `--gres`, which is a different
    # request on a count-only-GRES site. Piping the script cannot drift, and it is
    # what the portability report asked for: "run `sbatch --test-only` on the
    # generated script and surface whatever Slurm says".
    if script.strip():
        stdout, stderr, rc = _run_command(
            ["sbatch", "--test-only", "--parsable"], timeout=20, stdin=script
        )
        return _read_test_only_output(stdout, stderr, rc)

    cmd = ["sbatch", "--test-only", "--parsable"]
    if partition:
        cmd += ["-p", partition]
    if qos:
        cmd += ["-q", qos]
    # Include the array spec: MaxArraySize is a site limit, so an --array carried
    # from another cluster is refused here, and without it in the probe the ETA
    # happily reports a start time for a job Slurm will not accept.
    if array_spec:
        cmd += [f"--array={array_spec}"]
    # Same reasoning: a feature this cluster does not have is refused
    # ("Invalid feature specification"), and without it in the probe the ETA
    # reported a cheerful "~21h" for a job that cannot be placed.
    if constraint:
        cmd += [f"--constraint={constraint}"]
    if account:
        cmd += ["-A", account]
    if req_nodes > 0:
        cmd += ["-N", str(req_nodes)]
    if cpus > 0:
        cmd += ["-c", str(cpus)]
    if mem_mb > 0:
        cmd += [f"--mem={mem_mb}M"]
    if gpus_per_node > 0:
        cmd += [f"--gres=gpu:{gpu_type}:{gpus_per_node}" if gpu_type else f"--gres=gpu:{gpus_per_node}"]
    if time_limit:
        cmd += ["-t", time_limit]
    cmd += ["--wrap", "true"]

    stdout, stderr, rc = _run_command(cmd, timeout=20)
    return _read_test_only_output(stdout, stderr, rc)


def check_script_with_scheduler(script: str) -> str:
    """Slurm's refusal for this exact script, or "" — submits nothing.

    Needed when the script no longer corresponds to the answers: after a hand
    edit in ``$EDITOR`` the answers dict is stale, so validating *it* checks
    something other than what will be submitted. Asking the controller about the
    bytes themselves cannot drift, and a refusal is authoritative in a way the
    answers-derived checks are not.

    Empty on anything other than a positive refusal — no ``sbatch``, an
    unreachable controller, an unparsable answer — for the same reason the ETA
    does: "could not ask" must never render as "cannot run".
    """
    if not script.strip() or not is_tool_available("sbatch"):
        return ""
    stdout, stderr, rc = _run_command(
        ["sbatch", "--test-only", "--parsable"], timeout=20, stdin=script
    )
    _start, refusal = _read_test_only_output(stdout, stderr, rc)
    return refusal


def _read_test_only_output(stdout: str, stderr: str, rc: int) -> tuple[int | None, str]:
    """Turn one ``sbatch --test-only`` run into ``(seconds_until_start, refusal)``.

    Shared by both probe styles so they cannot disagree about what Slurm said.
    Reads **both** streams: the placement line and the refusal each arrive on
    stderr in some Slurm versions and stdout in others.
    """
    combined = f"{stderr}\n{stdout}"
    match = _TEST_ONLY_START_RE.search(combined)
    if match:
        try:
            start = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None, ""
        delta = int((start - datetime.now()).total_seconds())
        # Slurm reports the placement in the *controller's* local time, compared
        # here against the *login node's* clock. A few seconds in the past is
        # ordinary — "start immediately", plus the latency between asking and
        # parsing — but a large negative gap is evidence the two disagree (a
        # different timezone, or drift), and clamping that to 0 turned it into a
        # confident "ETA: now" for a job starting hours later. Unknown is the
        # honest answer; the caller then falls through to its own estimate.
        if delta < -_CLOCK_SKEW_TOLERANCE_S:
            logger.debug(
                f"sbatch --test-only start {start.isoformat()} is {-delta}s in the "
                f"past; treating the ETA as unknown rather than 'now'"
            )
            return None, ""
        return max(0, delta), ""

    if rc == 0:
        # Accepted but no placement line to parse — no verdict either way.
        return None, ""
    return None, _test_only_refusal(combined)


def _lone_site_message(output: str) -> str:
    """The site plugin's message when it wrote exactly one line, else ``""``.

    The fallback for a plugin that does not use Slurm's ``Reason:`` convention.
    Without it, Pythia's *"Batch jobs cannot use the `interactive_*` partitions."*
    was dropped and the user was shown Slurm's half alone — "Unspecified error",
    which names nothing and suggests nothing — on that cluster's **default**
    partition, i.e. on the shipped-defaults path.

    The caller only consults this once Slurm's own ``allocation failure:`` verdict
    is in hand, so nothing here can turn a broken sbatch into a refusal.
    """
    candidates = []
    for match in _TEST_ONLY_SITE_LINE_RE.finditer(output):
        text = match.group(1).strip()
        low = text.lower()
        if not text or low.startswith(_SLURM_OWN_ERROR_PREFIXES):
            continue
        # A lone limit token is appended separately, with its own brackets.
        if _TEST_ONLY_LIMIT_RE.match(match.group(0)):
            continue
        candidates.append(text)
    return candidates[0] if len(candidates) == 1 else ""


def _test_only_refusal(output: str) -> str:
    """The reason ``sbatch --test-only`` gave for refusing, or ``""``.

    Requires one of Slurm's two verdict markers — ``allocation failure: <why>``
    (its own) or ``Reason: <why>`` (the site ``job_submit`` plugin's, and the more
    specific of the pair, e.g. *"Invalid account [foo]"* vs *"Access/permission
    denied"*). A non-zero exit on its own is **not** enough: sbatch also fails
    when the controller is unreachable or the binary is broken, and reading that
    as "your job can never run" trades one confident wrong answer for another.
    Positive evidence or nothing.
    """
    # Checked first: sbatch never reaches the controller when it cannot parse
    # its own arguments, so there is no `allocation failure:` to find and the
    # verdict is unambiguous.
    usage = _TEST_ONLY_USAGE_RE.search(output)
    if usage:
        named = next(
            (usage.group(g) for g in ("a", "b", "c", "d", "e") if usage.group(g)),
            "",
        )
        if _probe_own_option(named):
            # Our own probe flag, not the script's. See `_PROBE_OWN_OPTIONS`:
            # positive evidence about `sbatch` here, none about the job.
            logger.debug(f"sbatch rejected our own probe option {named!r}; no verdict")
            return ""
        return "sbatch rejected an option in the script: {}".format(
            " ".join(usage.group(1).split())
        )
    reason = _TEST_ONLY_REASON_RE.search(output)
    failure = _TEST_ONLY_FAILURE_RE.search(output)
    specific = reason.group(1).strip() if reason else ""
    generic = failure.group(1).strip() if failure else ""
    # Only enrich a verdict Slurm has actually rendered. A bare `sbatch: error:`
    # line is not positive evidence of one — sbatch prints those for an
    # unreachable controller and for its own usage errors too — so the site's
    # sentence is read as the *specific half of a refusal*, never as the refusal.
    if generic and not specific:
        specific = _lone_site_message(output)
    if specific and generic and specific.lower() != generic.lower():
        # Keep both halves. A site job_submit plugin's Reason is the more useful
        # one to read ("Account is not specified" beats "Access/permission
        # denied"), but Slurm's own generic verdict underneath is the half that
        # can be *classified* — every marker list here is written against Slurm's
        # wordings, not against whatever a site's plugin chooses to say. Showing
        # only the specific reason therefore threw away the classification: on
        # midway3, whose plugin emits a six-line block ending in
        # `Reason: Account is not specified` / `allocation failure:
        # Access/permission denied`, a job that can never run was reported as one
        # slurmate could not judge, while the very next cluster's wording for the
        # same mistake was correctly called permanent.
        base = f"{specific} ({generic})"
    else:
        base = specific or generic
    if not base:
        return ""
    # Append the specific limit when Slurm named one, both so the user gets the
    # exact cause instead of a bundle listing three possibilities, and so
    # refusal_is_permanent() has something to classify.
    token = _TEST_ONLY_LIMIT_RE.search(output)
    if token and token.group(1).lower() not in base.lower():
        base = f"{base} [{token.group(1)}]"
    return base


def _scheduler_start_estimate(
    partition: str,
    req_nodes: int,
    cpus: int,
    mem_mb: int,
    gpus_per_node: int,
    gpu_type: str,
    time_limit: str,
    account: str,
    qos: str,
) -> int | None:
    """Seconds until this request would start, or ``None`` — see
    :func:`_scheduler_verdict`, which also reports *why* when Slurm refuses."""
    start, _ = _scheduler_verdict(
        partition, req_nodes, cpus, mem_mb, gpus_per_node, gpu_type, time_limit, account, qos
    )
    return start


def _nodes_that_fit(
    partition: str, cpus_per_node: int, mem_mb_per_node: int, gpus_per_node: int
) -> int | None:
    """Nodes in ``partition`` with enough *free* CPU, memory and GPU right now.

    The node-centric ``-O`` fields give what the aggregate ``%t`` state cannot:
    ``CPUsState`` (allocated/idle/other/total), ``Memory`` minus ``AllocMem``, and
    ``Gres`` minus ``GresUsed``. Counting node *states* instead treats a MIXED node
    with every GPU allocated as available, which is the bug this replaces.

    ``None`` when the query fails, so the caller can fall back rather than read a
    failure as "nothing fits".
    """
    stdout, _, rc = _run_command(
        [
            "sinfo",
            "-h",
            "-N",
            "-p",
            partition,
            "-O",
            "StateLong:20,CPUsState:24,Memory:16,AllocMem:16,Gres:48,GresUsed:48",
        ]
    )
    if rc != 0 or not stdout.strip():
        return None
    fits = 0
    saw_node = False
    for line in stdout.splitlines():
        fields = [f.strip() for f in re.split(r"\s{2,}", line.strip()) if f.strip()]
        if len(fields) < 4:
            continue
        saw_node = True
        state = fields[0].lower()
        if any(flag in state for flag in _UNSCHEDULABLE_FLAGS):
            continue
        base = re.sub(r"[^a-z]", "", state)
        if not base.startswith(("idle", "mix")):
            continue
        cpu_match = _CPUS_STATE_RE.match(fields[1])
        idle_cpus = int(cpu_match.group(2)) if cpu_match else 0
        # With no CPU count supplied, still require one free core: a MIXED node with
        # every core allocated can take nothing, and counting it is the same class
        # of error as trusting the state label.
        if idle_cpus < max(cpus_per_node, 1):
            continue
        total_mem = _safe_int(fields[2])
        alloc_mem = _safe_int(fields[3])
        if mem_mb_per_node > 0 and (total_mem - alloc_mem) < mem_mb_per_node:
            continue
        if gpus_per_node > 0:
            gres = fields[4] if len(fields) > 4 else ""
            gres_used = fields[5] if len(fields) > 5 else ""
            if (_sum_node_gpus(gres) - _sum_node_gpus(gres_used)) < gpus_per_node:
                continue
        fits += 1
    return fits if saw_node else None


# Last-resort --mem when the cluster tells us nothing about its nodes. Kept only
# as a fallback: see default_memory_for for why a literal is the wrong default.
FALLBACK_MEMORY = "16G"


def default_memory_for(part: dict[str, Any] | None, cpus: int) -> tuple[str, str]:
    """A ``--mem`` value for a user who did not ask for one: ``(value, source)``.

    ``source`` is ``"partition"`` when the number came from the node's advertised
    memory, ``"fallback"`` when the cluster told us nothing and
    :data:`FALLBACK_MEMORY` had to be used.

    A hardcoded default is a number, not a measurement. ``16G`` is harmless on a
    57 GB node and generates a permanently unschedulable script on an 8 GB one,
    and the user who never passed ``--memory`` has no reason to suspect either.
    Sizing it as ``mem_per_node × cores / cpus_per_node`` gives the request the
    same share of the node's memory as of its cores — the same thing a site's own
    ``DefMemPerCPU`` does — and can never exceed what a node has.
    """
    mem_node = _safe_int(str((part or {}).get("mem_per_node_mb") or 0))
    if mem_node <= 0:
        return FALLBACK_MEMORY, "fallback"

    cpus_per_node = _safe_int(str((part or {}).get("cpus_per_node") or 0))
    try:
        cores = max(1, int(cpus))
    except (TypeError, ValueError):
        cores = 1
    if cpus_per_node > 0:
        # A request for more cores than a node has is already flagged elsewhere;
        # here it just means "the whole node".
        share_mb = mem_node * min(cores, cpus_per_node) / cpus_per_node
    else:
        share_mb = mem_node
    share_mb = max(1, min(int(share_mb), mem_node))

    if share_mb >= 1024:
        gb = max(1, min(round(share_mb / 1024), max(1, mem_node // 1024)))
        return f"{gb}G", "partition"
    return f"{share_mb}M", "partition"


def resolve_request_mem_mb(answers: dict[str, Any]) -> int:
    """Per-node memory the built script will request, in MB; 0 when unset.

    Mirrors the builder's precedence — a custom ``--mem`` / ``--mem-per-cpu`` flag
    suppresses the auto directive, and ``--mem-per-cpu`` wins over ``--mem`` — so
    the ETA is computed against the request the script actually makes.
    """
    try:
        from .builder import _custom_mem_override, _normalize_custom_flags

        custom_mem, custom_per_cpu, custom_per_gpu = _custom_mem_override(
            _normalize_custom_flags(answers.get("custom_sbatch"))
        )
    except Exception:  # pragma: no cover - builder import is not worth failing an ETA
        custom_mem, custom_per_cpu, custom_per_gpu = None, None, None

    per_cpu = custom_per_cpu if custom_per_gpu else (
        custom_per_cpu or answers.get("mem_per_cpu")
    )
    flat = (
        custom_mem
        if (custom_mem or custom_per_cpu or custom_per_gpu)
        else answers.get("memory")
    )
    if per_cpu and validate_memory(str(per_cpu)):
        cores = answers.get("cpus") or 1
        try:
            cores = max(1, int(cores))
        except (TypeError, ValueError):
            cores = 1
        return _parse_mem_to_mb(str(per_cpu)) * cores
    if flat and validate_memory(str(flat)):
        return _parse_mem_to_mb(str(flat))
    return 0


def fetch_queue_eta(
    partition: str,
    req_nodes: int = 1,
    *,
    cpus: int = 0,
    mem_mb: int = 0,
    gpus_per_node: int = 0,
    gpu_type: str = "",
    time_limit: str = "",
    account: str = "",
    qos: str = "",
    array_spec: str = "",
    constraint: str = "",
    script: str = "",
) -> dict[str, Any]:
    """Estimate when a request would start in ``partition``.

    Three tiers, best first, with ``source`` in the result naming which one
    answered so the caller can qualify what it shows:

    ``scheduler``   ``sbatch --test-only`` — Slurm's own backfill placement.
    ``resources``   nodes with enough free CPU/memory/GPU, counted per node.
    ``pressure``    a queue-depth heuristic; the last resort.

    The resource arguments are optional for backwards compatibility, but omitting
    them makes the answer worse: a bare ``(partition, req_nodes)`` call cannot know
    that every GPU on an otherwise-free node is already allocated.
    """
    if not is_tool_available("squeue") or not is_tool_available("sinfo"):
        # Demo ETA only under SLURMATE_MOCK; on a real cluster missing squeue/sinfo
        # report "unknown" rather than a fabricated queue depth / wait time.
        if _force_mock():
            return dict(MOCK_QUEUE_INFO)
        return {
            "running": 0,
            "pending": 0,
            "eta_seconds": 0,
            "eta_label": "unknown",
            "source": "unknown",
            "feasible": True,
            "reason": "",
        }

    # Capture the return code. Discarding it made a failed or timed-out squeue
    # indistinguishable from an empty queue, so the summary reported
    # "0 running / 0 pending" as a measurement — the report's own cross-cutting
    # root cause ("a subprocess's error channel is not read"), and SM-19's defect
    # arriving through the failure path instead of a missing partition.
    stdout, _, queue_rc = _run_command(
        ["squeue", "-p", partition, "-o", "%T|%M|%l|%D", "--noheader"],
        timeout=_ADVISORY_TIMEOUT,
    )
    queue_known = queue_rc == 0

    running = 0
    pending = 0

    for line in stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        state = parts[0]
        if state in ("RUNNING", "CONFIGURING", "COMPLETING"):
            running += 1
        elif state in ("PENDING", "SUSPENDED", "WAITING"):
            pending += 1

    def _result(
        eta_sec: int,
        source: str,
        *,
        feasible: bool = True,
        reason: str = "",
        label: str = "",
        permanent: bool = True,
        transient: bool = False,
    ) -> dict[str, Any]:
        return {
            "running": running,
            "pending": pending,
            # False when squeue could not be read: the counts are not a reading,
            # and the caller must not render them as one.
            "queue_known": queue_known,
            "eta_seconds": eta_sec,
            # A rejected request has no wait time. Reporting one — "~60s" for a
            # job Slurm just refused — is worse than reporting nothing, because
            # it is specific and confident and never going to happen.
            "eta_label": label or (_format_eta(eta_sec) if feasible else "never"),
            "source": source,
            "feasible": feasible,
            "reason": reason,
            # Only meaningful when feasible is False: whether the refusal is about
            # the *request* (permanent) or the *moment* (clears on its own). Set
            # here, once, because two surfaces deciding it independently is
            # exactly how the summary came to print "ETA: never" directly above an
            # advisory saying the script was valid and the condition temporary.
            "refusal_is_permanent": permanent,
            # Not the negation of the field above: a refusal can be neither, and
            # only this one licenses telling the user the script is fine.
            "refusal_is_transient": transient,
        }

    # Tier 1 — ask the scheduler.
    scheduled, refusal = _scheduler_verdict(
        partition, req_nodes, cpus, mem_mb, gpus_per_node, gpu_type, time_limit,
        account, qos, array_spec, constraint, script,
    )
    if scheduled is not None:
        return _result(scheduled, "scheduler")
    if refusal:
        # Slurm refused outright: authoritative, and it costs nothing to say so.
        # "never" is only true of a refusal about the request; a submit-count cap
        # clears the moment another job finishes, and calling that "never" is the
        # same over-claim as reporting a wait for a job that will never start.
        permanent = refusal_is_permanent(refusal)
        transient = refusal_is_transient(refusal)
        if permanent:
            label = "never"
        elif transient:
            label = "not right now"
        else:
            # Refused, and we cannot honestly say whether that is about the
            # request or the moment. "refused" states what happened and claims
            # nothing further.
            label = "refused"
        return _result(
            0, "scheduler", feasible=False, reason=refusal,
            label=label, permanent=permanent, transient=transient,
        )

    # Tier 2 — count nodes that genuinely fit the per-node share of the request.
    per_node_cpus = -(-cpus // max(req_nodes, 1)) if cpus > 0 else 0
    per_node_mem = -(-mem_mb // max(req_nodes, 1)) if mem_mb > 0 else 0
    fitting = _nodes_that_fit(partition, per_node_cpus, per_node_mem, gpus_per_node)
    if fitting is not None:
        if fitting >= req_nodes:
            return _result(0, "resources")
        # Nothing fits right now: fall through to a pressure estimate, but never
        # back to "immediate" — the whole point is that state labels lied.
        if running == 0:
            return _result(300, "resources")
        pressure = pending / max(1, running)
        return _result(max(60, int(min(pressure * 120, 7200))), "resources")

    # Tier 3 — neither the scheduler nor per-node data is available. A queue-depth
    # guess is all that is left; it is deliberately never 0, because without
    # resource data there is no evidence anything is actually free.
    #
    # But it is a guess *derived from the queue depth*, so if squeue could not be
    # read there is nothing to derive it from: reporting "~5min" out of a failed
    # query would be inventing a number twice over.
    if not queue_known:
        return _result(0, "unknown")
    if running == 0:
        return _result(300, "pressure")
    pressure = pending / max(1, running)
    return _result(max(60, int(min(pressure * 120, 7200))), "pressure")


def submit_sbatch(script_content: str, job_name: str = "slurm") -> tuple[int, str, str]:
    """Submit sbatch script and return (returncode, job_id_or_output, error_message).

    Args:
        script_content: The sbatch script content
        job_name: Job name for logging purposes

    Returns:
        Tuple of (returncode, job_id_or_stdout, stderr)
        - returncode: 0 on success, non-zero on failure
        - job_id_or_stdout: Job ID (integer as string) on success, stdout on failure
        - stderr: Error message on failure, empty string on success
    """
    # Nothing is going to run, so don't touch the filesystem: creating the script's
    # log directories before this check meant mock mode (and any host without
    # sbatch) left stray "logs/" trees behind while reporting "no job submitted".
    if not is_tool_available("sbatch"):
        return 0, "", "sbatch not available (mock mode) — no job submitted"

    # Create the log directories the script's #SBATCH --output/--error point at,
    # so Slurm doesn't fail the job on a missing directory.
    for line in script_content.splitlines():
        val = _sbatch_log_path(line)
        if not val:
            continue
        dir_name = os.path.dirname(os.path.expanduser(val))
        # Skip a directory component that carries a Slurm filename pattern
        # (%j/%A/%a/%x): those are expanded per-job by Slurm, so creating a
        # literal "%j" directory here would be wrong.
        if dir_name and "%" not in dir_name:
            try:
                os.makedirs(dir_name, exist_ok=True)
            except OSError as e:
                logger.debug(f"Failed to create log directory {dir_name}: {e}")

    try:
        # Use --parsable for clean job ID output
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script_content,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            encoding="utf-8",
            # surrogateescape, not replace: `errors` governs the *input* encoding
            # too, and under a non-UTF-8 locale a --command carrying UTF-8 bytes
            # arrives as lone surrogates. "replace" would send sbatch a "?" for
            # each one — silently running a different command than the user typed
            # — where surrogateescape hands back the original bytes exactly.
            errors="surrogateescape",
        )
    except subprocess.TimeoutExpired:
        return -1, "", "Submission timed out after 30s"
    except OSError as e:
        return -1, "", f"Could not run sbatch: {e}"

    if result.returncode != 0:
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    return result.returncode, result.stdout.strip(), ""


# A line that is exactly what `sbatch --parsable` prints: the job id, optionally
# followed by ";cluster" on a federated setup.
_PARSABLE_ID_RE = re.compile(r"^(\d+)(?:;(\S+))?$")

# Slurm's own non-parsable wording, which is what a site wrapper that drops
# --parsable prints, and what a federated submit prints even with it:
# "Submitted batch job 12345" / "... on cluster mercury". Recognising a second
# *exact* shape is not the number-scraping the docstring below rules out — it is
# still a fixed format, just Slurm's other one — and without it the id is lost
# whenever a wrapper reformats the output, taking the squeue/scancel hints and
# the saved script's filename with it.
_SUBMITTED_ID_RE = re.compile(
    r"^Submitted batch job (\d+)(?:\s+on cluster\s+\S+)?$"
)


def parse_submitted_job_id(raw: str) -> str:
    """The job id from ``sbatch --parsable`` output, or "" if there isn't one.

    ``sbatch`` prints one line, but a site's ``sbatch`` **wrapper** does not:
    a policy notice or MOTD on stdout ends up prepended to the id, and the whole
    banner then travels into the "Job ID:" line, the ``squeue -j`` / ``scancel``
    hints and the saved script's filename. This module already guards the same
    hazard for JSON (:func:`_extract_first_json`, which exists because "a login
    shell may print a banner before the JSON"); the submit path did not.

    Matches only a line of the exact expected shape, and returns "" rather than
    guessing when none is present — a banner can itself contain digits, so
    scraping the first number out of arbitrary text would substitute one wrong
    answer for another. ``;cluster`` is stripped: it is not part of the id that
    ``squeue``/``scancel`` want.
    """
    for line in (raw or "").splitlines():
        stripped = line.strip()
        match = _PARSABLE_ID_RE.match(stripped) or _SUBMITTED_ID_RE.match(stripped)
        if match:
            return match.group(1)
    return ""


_LOG_FLAG_NAMES = {
    "output": ("--output", "-o"),
    "error": ("--error", "-e"),
}

_SBATCH_OPT_RE = re.compile(r"^(--?[A-Za-z][A-Za-z0-9-]*)(?:=|\s+)(.*)$")


def _sbatch_log_path(line: str, kind: str | None = None) -> str:
    """Extract the path from a ``#SBATCH`` output/error directive.

    Handles every spelling ``sbatch`` itself accepts — ``--output=PATH``,
    ``--output PATH`` (long option + space, previously missed, so the directory
    went un-created and Slurm could fail the job), ``-o PATH`` and the ``--error``
    /``-e`` equivalents — strips surrounding quotes, and returns "" for anything
    else (or a valueless directive, which must not raise).

    ``kind`` restricts the match to ``"output"`` or ``"error"``; the default
    accepts either, which is what the log-directory pre-creation wants.
    """
    s = line.strip()
    if not s.startswith("#SBATCH"):
        return ""
    wanted = (
        _LOG_FLAG_NAMES[kind] if kind in _LOG_FLAG_NAMES
        else _LOG_FLAG_NAMES["output"] + _LOG_FLAG_NAMES["error"]
    )
    m = _SBATCH_OPT_RE.match(s[len("#SBATCH"):].strip())
    if not m or m.group(1) not in wanted:
        return ""
    val = m.group(2).strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        val = val[1:-1]
    return val


# Slurm's filename patterns, split by whether slurmate can know the value before
# the job starts. %% is a literal percent and must be consumed as a unit — a
# naive str.replace turns "%%j" into "%<jobid>" when Slurm writes "%j".
_LOG_PATTERN_UNKNOWABLE = frozenset("aNnts")


def expand_log_pattern(
    pattern: str, *, job_id: str = "", job_name: str = "", user: str = ""
) -> tuple[str, list[str]]:
    """Resolve what we can in a Slurm log pattern.

    Returns ``(path, unresolved)`` — the expanded path, and the pattern letters
    left in it because their value does not exist yet (``%a`` per array task,
    ``%N``/``%n``/``%t``/``%s`` per node/task/step). The caller needs that list:
    printing a ``tail -f`` for a path still containing ``%a`` offers the user a
    file that will never exist under that name.

    Single pass, so ``%%`` is consumed as a literal percent instead of leaving a
    bare ``%`` for the next substitution to misread.
    """
    known = {"j": job_id, "A": job_id, "x": job_name, "u": user}
    out: list[str] = []
    unresolved: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch != "%" or i + 1 >= len(pattern):
            out.append(ch)
            i += 1
            continue
        nxt = pattern[i + 1]
        if nxt == "%":
            out.append("%")          # literal percent — do not re-scan it
        elif known.get(nxt):
            out.append(str(known[nxt]))
        else:
            out.append(f"%{nxt}")
            if nxt in _LOG_PATTERN_UNKNOWABLE or nxt in known:
                unresolved.append(f"%{nxt}")
        i += 2
    return "".join(out), unresolved


def effective_log_path(script: str, kind: str = "output") -> str:
    """The log path Slurm will actually use for ``kind``, or "".

    Scans the whole script and keeps the **last** matching directive, because
    that is the one Slurm honours when a script carries more than one (measured:
    with two conflicting options, only the final one takes effect). Reading the
    *first* match is how the submit report came to print — and offer a ``tail -f``
    for — a file the job never wrote.
    """
    val = ""
    for line in script.splitlines():
        found = _sbatch_log_path(line, kind=kind)
        if found:
            val = found
    return val


def _strip_inline_comment(v: str) -> str:
    """Drop a trailing ``# comment`` that sits outside any quotes."""
    in_single = in_double = False
    for i, ch in enumerate(v):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return v[:i].rstrip()
    return v.rstrip()


def _split_top_level_commas(s: str) -> list[str]:
    """Split ``s`` on commas that sit outside single/double quotes.

    A raw ``str.split(',')`` shreds a quoted array element that contains a comma
    (e.g. ``"--constraint=a,b"``) into bogus tokens with dangling quotes; this
    keeps such elements intact, matching how a real TOML parser reads the array.
    """
    items: list[str] = []
    buf: list[str] = []
    in_single = in_double = False
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == "," and not in_single and not in_double:
            items.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    items.append("".join(buf))
    return items


def _has_unquoted_char(s: str, target: str) -> bool:
    """True if ``target`` appears in ``s`` outside single/double quotes."""
    in_single = in_double = False
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == target and not in_single and not in_double:
            return True
    return False


def _coerce_scalar(v: str) -> Any:
    """Coerce a single bare scalar token (string/int/float/bool)."""
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if re.match(r"^-?\d+$", v):
        return int(v)
    if re.match(r"^-?\d+\.\d+$", v):
        return float(v)
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return v


def _coerce_config_value(v: str) -> Any:
    """Parse one value for the naive key=value fallback parser.

    Handles quoted strings, arrays (with quoted *or* bare numeric items), ints,
    floats, negatives and booleans. Best-effort only — real TOML (tomllib/tomli)
    is used whenever available; this is the last resort.
    """
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(x.strip()) for x in _split_top_level_commas(inner) if x.strip()]
    return _coerce_scalar(v)


# ── Configuration file vocabulary ───────────────────────────────────────────
# The keys the batch and wizard paths actually read. Anything else in a config
# file is dropped, so it has to be named rather than silently discarded.
CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "account", "array_spec", "command", "constraint", "cpus", "custom_sbatch",
        "env_name", "env_type", "gpu_format", "gpus", "gpu_type", "job_name",
        "memory", "mem_per_cpu", "modules", "nodes", "ntasks_per_node",
        "output_dir", "output_file", "partition", "qos", "time_limit",
    }
)

# CLI flag spellings that differ from the config key by more than a dash.
# Dashes are normalised to underscores before this is consulted, so
# ``job-name``/``mem-per-cpu``/``ntasks-per-node`` and friends need no entry.
CONFIG_ALIASES: dict[str, str] = {"time": "time_limit", "array": "array_spec"}

# Tables whose contents are merged over the top-level keys, best last.
CONFIG_SECTIONS: tuple[str, ...] = ("defaults", "slurmate")

# (file, tag) pairs already reported this process — see :func:`_config_notice`.
_CONFIG_NOTICES_SHOWN: set[str] = set()

# Display path of the file the last :func:`load_config` read; "" when none.
_CONFIG_SOURCE: str = ""


def _parse_config_naive(text: str, path: Any = None) -> dict[str, Any]:
    """Minimal key=value parser used only when no TOML library is available.

    Best-effort, but section- and array-aware so it doesn't silently disagree
    with the real TOML reader: it tracks ``[section]`` headers, applies the same
    ``[slurmate] > [defaults] > top-level`` precedence as :func:`_flatten_config`,
    and accumulates a multi-line ``key = [`` array until its closing ``]``.
    """
    top: dict[str, Any] = {}
    sections: dict[str, dict[str, Any]] = {}
    current: str | None = None
    pending_key: str | None = None
    pending_parts: list[str] = []

    def store(key: str, raw_value: str) -> None:
        target = sections.setdefault(current, {}) if current else top
        target[key] = _coerce_config_value(raw_value)

    for raw in text.splitlines():
        line = raw.strip()
        if pending_key is not None:
            # Strip a trailing comment from THIS physical line (TOML comments are
            # line-oriented). Doing it once on the joined text would let an
            # interior line's "#" swallow the rest of the array.
            pending_parts.append(_strip_inline_comment(line))
            # Only a "]" outside quotes closes the array; a "]" inside a string
            # element (e.g. "--constraint=a]b") must not terminate it early.
            if _has_unquoted_char(" ".join(pending_parts), "]"):
                store(pending_key, " ".join(pending_parts))
                pending_key, pending_parts = None, []
            continue
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = _strip_inline_comment(v.strip())
            # A multi-line array (`key = [` with no closing `]` on this line).
            if v.startswith("[") and not _has_unquoted_char(v, "]"):
                pending_key, pending_parts = k, [v]
                continue
            store(k, v)

    # An array that opened but never closed: don't silently drop it (and every
    # subsequent line accumulated into it). Warn, matching the tomllib path,
    # which raises + surfaces a "warning: ignoring config" message in load_config.
    if pending_key is not None:
        import sys
        print(
            f"slurmate: warning: unclosed array for '{pending_key}' in the "
            f"configuration file — ignoring it",
            file=sys.stderr,
        )

    if path is not None:
        _warn_unknown_config_sections(sections, path)

    config: dict[str, Any] = dict(top)
    for section in CONFIG_SECTIONS:
        if section in sections:
            config.update(sections[section])
    return config


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    """Take top-level scalar keys, then merge an optional [defaults]/[slurmate] table."""
    config: dict[str, Any] = {k: v for k, v in data.items() if not isinstance(v, dict)}
    for section in CONFIG_SECTIONS:
        sect = data.get(section)
        if isinstance(sect, dict):
            config.update(sect)
    return config


def _config_display_path(path: Any) -> str:
    """The config path as a user would recognise it: ``./x`` or ``~/x``."""
    from pathlib import Path

    p = Path(path)
    try:
        return f"./{p.relative_to(Path.cwd())}"
    except ValueError:
        pass
    try:
        return f"~/{p.relative_to(Path.home())}"
    except ValueError:
        return str(p)


def _config_notice(path: Any, tag: str, message: str) -> None:
    """Print a one-off config notice to stderr.

    Deduplicated per (file, tag) because ``load_config()`` is called once by the
    batch path and again by the wizard in the same process; the same warning
    twice reads like two different problems.
    """
    import sys

    key = f"{path}\x00{tag}"
    if key in _CONFIG_NOTICES_SHOWN:
        return
    _CONFIG_NOTICES_SHOWN.add(key)
    print(message, file=sys.stderr)


def _reset_config_notices() -> None:
    """Forget which config notices have been shown (for tests)."""
    global _CONFIG_SOURCE
    _CONFIG_NOTICES_SHOWN.clear()
    _CONFIG_SOURCE = ""


def config_source() -> str:
    """Where the last :func:`load_config` got its values; "" if nowhere.

    Recorded rather than returned so the existing ``load_config()`` signature —
    used by both the batch path and the wizard — does not have to change.
    """
    return _CONFIG_SOURCE


def _warn_unknown_config_sections(names: Iterable[str], path: Any) -> None:
    """Name any ``[section]`` that is silently discarded.

    A whole table dropped without a word is the same defect as a dropped key,
    one level up and with more of the user's intent in it.
    """
    where = _config_display_path(path)
    for name in dict.fromkeys(n for n in names if n not in CONFIG_SECTIONS):
        _config_notice(
            path,
            f"section:{name}",
            f"slurmate: {where}: ignoring unknown section '[{name}]' — put keys at "
            f"the top level or under [defaults]/[slurmate]",
        )


def _normalize_config_keys(config: dict[str, Any], path: Any) -> dict[str, Any]:
    """Map CLI spellings onto config keys, and report keys that are neither.

    Two separate failures, both silent before: ``time = "36:00:00"`` (the CLI
    flag is ``--time``, the key is ``time_limit``) was dropped and the user got
    the 2-hour default for a 36-hour run, and an outright typo was dropped with
    no warning at all. Dashes are normalised first, so only the genuinely
    different names need to be listed in :data:`CONFIG_ALIASES`.
    """
    where = _config_display_path(path)
    out: dict[str, Any] = {}
    # Which spelling filled each slot, and whether it was the real key name — so
    # `time` losing to `time_limit` is reported whichever order they appear in.
    filled: dict[str, tuple[str, bool]] = {}

    def _dropped(loser: str, winner: str) -> None:
        _config_notice(
            path,
            f"dupe:{loser}",
            f"slurmate: {where}: '{loser}' ignored — '{winner}' is also set",
        )

    for raw_key, value in config.items():
        dashed = str(raw_key).strip().replace("-", "_")
        key = CONFIG_ALIASES.get(dashed, dashed)
        if key not in CONFIG_KEYS:
            hint = _near_misses(dashed, sorted(CONFIG_KEYS), limit=1)
            suffix = f" — did you mean '{hint[0]}'?" if hint else " — ignoring it"
            _config_notice(
                path,
                f"key:{raw_key}",
                f"slurmate: {where}: unknown key '{raw_key}'{suffix}",
            )
            continue
        exact = dashed == key
        if key in filled:
            prev_raw, prev_exact = filled[key]
            if prev_exact and not exact:
                _dropped(str(raw_key), prev_raw)
                continue
            if exact and not prev_exact:
                _dropped(prev_raw, str(raw_key))
            elif prev_raw != str(raw_key):
                # Two spellings of the SAME key: `job-name` and `job_name` are
                # distinct TOML keys but normalise onto one slot, so neither is
                # the other's alias and the pair above cannot separate them. The
                # later one still wins (Slurm's own "last option wins", and the
                # order `_flatten_config` builds means a [defaults] entry keeps
                # beating a top-level one) -- but the loser has to be *named*.
                # It was dropped in silence, which is precisely the failure this
                # notice exists for: `job-name = "a"` above `job_name = "b"` ran
                # under "b" with nothing on stderr, and swapping the two lines
                # silently changed the answer.
                _dropped(prev_raw, str(raw_key))
        out[key] = value
        filled[key] = (str(raw_key), exact)
    return out


def load_config() -> dict[str, Any]:
    """Load configuration defaults, merging the global and project files.

    Reads ``~/.config/slurmate/config.toml`` first, then overlays
    ``$CWD/.slurmate.toml`` on top, so the more specific file wins **per key**
    rather than per file. Keys may sit at the top level or under a ``[defaults]``
    (or ``[slurmate]``) table. Real TOML is used when a parser is available
    (``tomllib`` on 3.11+, ``tomli`` on older Pythons), otherwise a minimal flat
    key=value reader is used.

    First-file-wins was the previous behaviour, and it made a project file
    *destructive*: a one-line ``.slurmate.toml`` naming this cluster's partition
    discarded the global account, memory, time limit and module list — silently,
    and each with its own failure (a rejected or mischarged job, an OOM kill, a
    truncated run, an unloaded environment). That is the most natural use of the
    feature, and every config system a user has met — git, ssh, pip, cargo, npm —
    merges instead. Per-key merging is also what the search order always implied.

    Unrecognised keys are reported rather than dropped, CLI spellings (``time``
    for ``time_limit``, ``array`` for ``array_spec``, and any dashed form) are
    accepted as aliases, and each file is named on stderr with the keys it
    actually **won**, so the precedence is visible rather than inferred.

    A file that cannot be parsed is reported and skipped without taking the other
    one down with it. Returns ``{}`` in mock mode (``SLURMATE_MOCK``) so tests
    stay hermetic, and when no file is readable.
    """
    global _CONFIG_SOURCE
    if _force_mock():
        return {}

    from pathlib import Path

    toml: Any = None
    try:
        import tomllib
        toml = tomllib
    except ModuleNotFoundError:
        try:
            import tomli
            toml = tomli
        except ModuleNotFoundError:
            toml = None

    # Least specific first, so a later overlay wins per key. Both candidates are
    # built defensively, and neither is allowed to take the tool down:
    #
    # `Path.home()` raises RuntimeError when $HOME is unset AND the uid has no
    # passwd entry — which is `sbatch --export=NONE` (standard Slurm, and a
    # cluster-wide default at some sites) on a node whose name service does not
    # resolve the user. Building this list eagerly made that abort *every*
    # invocation before any flag was acted on, including runs with a perfectly
    # good project-local file sitting in the job's working directory.
    #
    # XDG_CONFIG_HOME is the documented location for this file and is honoured
    # here; previously it was ignored.
    paths: list[Any] = []
    xdg = os.environ.get("XDG_CONFIG_HOME")
    try:
        base = Path(xdg) if xdg else Path.home() / ".config"
    except (RuntimeError, OSError):
        base = None                  # no home is discoverable; the CWD one stands
    if base is not None:
        paths.append(base / "slurmate" / "config.toml")
    # cwd deleted under us — nothing to read there.
    with contextlib.suppress(OSError):
        paths.append(Path.cwd() / ".slurmate.toml")

    merged: dict[str, Any] = {}
    origin: dict[str, Any] = {}   # which file each surviving key came from
    used: list[Any] = []

    for path in paths:
        if not path.exists():
            continue
        try:
            if toml is not None:
                with open(path, "rb") as fb:
                    raw = toml.load(fb)
                _warn_unknown_config_sections(
                    [k for k, v in raw.items() if isinstance(v, dict)], path
                )
                config = _normalize_config_keys(_flatten_config(raw), path)
            else:
                with open(path) as f:
                    config = _normalize_config_keys(
                        _parse_config_naive(f.read(), path=path), path
                    )
        except Exception as e:
            # The file exists but couldn't be parsed/read (a TOML syntax error, a
            # permission problem). Surface it and carry on with the other file:
            # dropping both because one is broken would lose values that are
            # perfectly readable.
            import sys
            print(
                f"slurmate: warning: ignoring configuration file {path} — {e}",
                file=sys.stderr,
            )
            logger.debug(f"Failed to load config from {path}: {e}")
            continue
        if not config:
            continue
        used.append(path)
        for key, value in config.items():
            merged[key] = value
            origin[key] = path

    # Name each file with the keys it actually won, in application order, so a
    # value that was overridden is not claimed by the file that lost it.
    contributors: list[Any] = []
    for path in used:
        contributed = [k for k in merged if origin[k] is path]
        if not contributed:
            continue
        contributors.append(path)
        # Emitted here rather than through a helper, because the useful version
        # of this sentence needs `contributed` -- the keys this file actually WON
        # -- which only the merge loop knows. A helper taking one file's whole
        # config was carried above for a while and could only name every key it
        # offered, so with two config files the loser claimed keys it had lost.
        # It is gone; this is the only place the sentence is spelled.
        _config_notice(
            path,
            "loaded",
            f"slurmate: using defaults from {_config_display_path(path)}: "
            f"{', '.join(contributed)}",
        )

    _CONFIG_SOURCE = ", ".join(_config_display_path(p) for p in contributors)
    return merged
