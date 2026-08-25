from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, overload

from prompt_toolkit.key_binding import KeyBindings
from rich.cells import cell_len
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .builder import (
    build_from_answers,
    command_injects_directives,
    custom_ntasks,
    env_activation_emitted,
    estimate_gpu_hours,
    estimate_su,
    job_name_change_note,
    job_summary_rows,
    managed_custom_flags,
    sanitize_job_name,
)
from .system_utils import (
    GPU_COUNT_FLAG,
    GPU_SPELLING_FORMATS,
    array_spec_reason,
    array_task_count,
    capacity_refusal,
    check_conda_env,
    check_log_dirs,
    check_modules,
    check_script_with_scheduler,
    config_source,
    current_username,
    default_memory_for,
    effective_log_path,
    expand_log_pattern,
    fetch_all_partition_names,
    fetch_gpu_type_sources,
    fetch_known_qos,
    fetch_max_array_size,
    fetch_node_features,
    fetch_partition_node_maxima,
    fetch_partitions,
    fetch_queue_eta,
    fetch_select_type,
    fetch_user_accounts,
    is_mock,
    is_tool_available,
    load_config,
    normalize_memory,
    parse_gpu_spelling,
    parse_submitted_job_id,
    refusal_is_permanent,
    refusal_is_transient,
    resolve_request_mem_mb,
    submit_sbatch,
    unsupported_gpu_format,
    validate_array_spec,
    validate_cluster_targets,
    validate_job_config,
    validate_memory,
    validate_time,
    write_private_text,
)
from .theme import c, g, make_output_safe, print_banner, set_ascii
from .tui import Wizard, _parse_custom_flags

logger = logging.getLogger(__name__)

# Sentinel returned by the action menu when the user presses Esc to go back.
_GO_BACK = "\x00__go_back__"

# ── Batch mode helpers ───────────────────────────────────────────────────

def _get_partition(partitions: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for p in partitions:
        if p["name"] == name:
            # Enriched here rather than at the call sites: there are six of them
            # across the CLI and the wizard, and wiring a lookup into some of them
            # is how half the findings in the portability report happened.
            return _enrich_partition_maxima(p)
    # Not in the sinfo list: a manually-typed name, or another cluster's. Every
    # capacity field is unknown (0/None), which is what keeps the limit checks
    # silent rather than warning against a limit of zero. nodes_up is None
    # (unknown), never 0, so it is not read as "all nodes are down".
    # Marked as unknown rather than merely empty: zeros keep the limit checks
    # quiet, but *silence about a 999-CPU request is itself a wrong answer* — the
    # less valid request produced the more reassuring screen. The flag lets the
    # validator say "I could not check" instead.
    #
    # ``_unknown_reason`` distinguishes the two ways we get here, because the
    # honest message differs: with a readable partition list this name is genuinely
    # absent, but with an *empty* list (no Slurm, sinfo down) nothing is known
    # about any partition — and saying "not on this cluster" there is the false
    # rejection the SM-4 restraint exists to prevent.
    return {"name": name, "nodes": 0, "nodes_up": None, "cpus_per_node": 0,
            "mem_per_node_mb": 0, "gpu_types": [], "timelimit": None,
            "is_public": True, "is_default": False, "_unknown": True,
            "_unknown_reason": "absent" if partitions else "unreadable"}


def _enrich_partition_maxima(part: dict[str, Any]) -> dict[str, Any]:
    """Attach the per-node maxima for a heterogeneous partition (SM-27).

    Only for the partition actually in use, and only when ``sinfo``'s aggregate
    row carried a ``+`` — so a homogeneous site makes no extra call at all, and a
    mixed one makes exactly one. Left absent when the query fails, which keeps the
    floor-based warning and its honest "nodes differ" wording rather than
    silencing the check.
    """
    if not part or not part.get("heterogeneous") or part.get("_unknown"):
        return part
    max_cpus, max_mem = fetch_partition_node_maxima(str(part.get("name") or ""))
    if max_cpus is None and max_mem is None:
        return part
    enriched = dict(part)
    if max_cpus:
        enriched["max_cpus_per_node"] = max_cpus
    if max_mem:
        enriched["max_mem_per_node_mb"] = max_mem
    return enriched


def _coerce_int(value: Any, default: int, *, field: str | None = None,
                err_console: Console | None = None) -> int:
    """Coerce a CLI/config value to int, falling back to ``default``.

    Config values can be stringy (e.g. ``gpus = "2"`` in TOML), which used to
    crash batch mode on the later ``gpus > 0`` comparison. A value that is
    present but not an integer (e.g. ``cpus = "8cores"``) is reported to
    ``err_console`` when ``field`` is given, rather than silently reverting to
    the default (which would run the job with the wrong resources, or produce a
    misleading "got 0" error downstream).
    """
    if value is None:
        return default

    def _reject(reason: str) -> int:
        if field and err_console is not None:
            err_console.print(
                f"  {c.YELLOW}{g.WARN} {field} value {value!r} {reason}; "
                f"using {default}{c.RESET}"
            )
        return default

    # A bool is an int subclass, so `cpus = true` in a TOML file silently became
    # a one-core request — a meaningless value accepted as a plausible one.
    if isinstance(value, bool):
        return _reject("is a boolean, not a core/node count")
    # int(2.7) truncates. A config written as `cpus = 2.7` therefore ran a 2-core
    # job with no indication; an integral float (`2.0`) is unambiguous and kept.
    if isinstance(value, float):
        if value != int(value):
            return _reject("is not a whole number")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return _reject("is not an integer")


@overload
def _coerce_str(value: Any, default: str, *, field: str, err_console: Console) -> str: ...
@overload
def _coerce_str(value: Any, default: None, *, field: str, err_console: Console) -> str | None: ...
def _coerce_str(value: Any, default: str | None, *, field: str,
                err_console: Console) -> str | None:
    """Coerce a CLI/config value for a free-form string field.

    A scalar (str/int/float/bool) is accepted and stringified — mirroring how
    ``_coerce_int`` leniently accepts stringy numbers — but a list/dict (or any
    other structured value) can't become a single directive value, so it is
    rejected with a clean error and ``sys.exit(1)`` instead of crashing the
    builder with an AttributeError/TypeError deep in script generation. This
    guards the free-form string fields (partition/account/qos/array/command/
    output paths/env) the way ``_coerce_int`` already guards the numerics.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):  # bool is an int subclass — str() is fine
        return str(value)
    err_console.print(
        f"  {c.RED}{g.ERR} Error: {field} must be a string "
        f"(got {type(value).__name__}){c.RESET}"
    )
    sys.exit(1)


# Config keys whose argparse dest is spelled differently. Used to work out which
# config values actually reached the script (a CLI flag overrides the file), so
# the disclosure names only the ones the user did not type.
CONFIG_ARG_DESTS: dict[str, str] = {
    "time_limit": "time",
    "array_spec": "array",
    "env_name": "env",
}


def _config_keys_in_effect(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    """Config keys not overridden by an explicit flag, in file order."""
    return [
        key
        for key in config
        if getattr(args, CONFIG_ARG_DESTS.get(key, key), None) is None
    ]


def _check_cluster_targets(
    partition: str,
    account: str | None,
    all_parts: list[dict[str, Any]],
    *,
    qos: str | None = None,
    constraint: str | None = None,
    force: bool,
    err_console: Console,
) -> None:
    """Reject a ``--partition``/``--account``/``--qos``/``-C`` this cluster lacks.

    Fatal by default: a script naming a partition that isn't here is not a script,
    it is a queued failure the user finds out about minutes later from sbatch.
    ``--force`` downgrades it to a warning, because writing a script *to carry to
    another cluster* is a legitimate thing to do — it just must not be the silent
    default.

    Stays quiet when the cluster's lists can't be read (no Slurm, sinfo down):
    an unreadable ``sinfo`` must never present as "your partition doesn't exist".
    """
    known_parts = fetch_all_partition_names()
    # The picker's own list is a subset of the validation list, but it is what we
    # already have and it names the site default — used for the suggestion line.
    default_part = next((str(p["name"]) for p in all_parts if p.get("is_default")), "")
    # Only look up accounts when one was actually given: sacctmgr is slow enough
    # on a busy controller to be worth skipping.
    known_accounts = fetch_user_accounts() if account else []
    # Same reasoning for QoS: sacctmgr is slow on a busy controller, so only ask
    # when there is something to check.
    known_qos = fetch_known_qos() if qos else []
    known_features = fetch_node_features() if constraint else None

    issues = validate_cluster_targets(
        partition, account,
        qos=qos,
        constraint=constraint,
        known_partitions=known_parts,
        known_accounts=known_accounts,
        known_qos=known_qos,
        known_features=known_features,
        default_partition=default_part,
    )
    if not issues:
        return

    for _level, msg in issues:
        head, *rest = msg.splitlines()
        if force:
            err_console.print(f"  [yellow]{g.WARN} Warning: {escape(head)}[/]")
        else:
            err_console.print(f"  [red]{g.ERR} Error: {escape(head)}[/]")
        for line in rest:
            err_console.print(f"    [dim]{escape(line)}[/]")
    if not force:
        err_console.print(
            "  [dim]Pass --force to generate the script anyway "
            "(e.g. for another cluster).[/]"
        )
        sys.exit(1)


def run_batch(args: argparse.Namespace, console: Console, config: dict[str, Any]) -> dict[str, Any]:
    err_console = Console(stderr=True)

    # Get values fallback from config
    args_partition = getattr(args, "partition", None)
    partition = _coerce_str(
        args_partition if args_partition is not None else config.get("partition", ""),
        "", field="partition", err_console=err_console)

    args_cpus = getattr(args, "cpus", None)
    cpus = _coerce_int(args_cpus if args_cpus is not None else config.get("cpus", 4), 4,
                       field="cpus", err_console=err_console)

    args_memory = getattr(args, "memory", None)
    # None here means "nobody asked for a memory size", which is different from
    # asking for none: the value is filled in from the partition further down,
    # once sinfo has been read (see SM-7 — a literal default is a number, not a
    # measurement, and 16G is unschedulable on an 8 GB node).
    memory_val = args_memory if args_memory is not None else config.get("memory")
    # An explicit empty / "none" memory omits --mem entirely — required by
    # whole-node/exclusive sites (e.g. TACC) that reject a memory request.
    mem_omit = memory_val is not None and str(memory_val).strip().lower() in ("", "none")

    args_time = getattr(args, "time", None)
    time_val = args_time if args_time is not None else config.get("time_limit", "02:00:00")

    args_nodes = getattr(args, "nodes", None)
    nodes = _coerce_int(args_nodes if args_nodes is not None else config.get("nodes", 1), 1,
                        field="nodes", err_console=err_console)

    args_gpus = getattr(args, "gpus", None)
    gpus = _coerce_int(args_gpus if args_gpus is not None else config.get("gpus", 0), 0,
                       field="gpus", err_console=err_console)

    args_ntasks_per_node = getattr(args, "ntasks_per_node", None)
    raw_ntasks = args_ntasks_per_node if args_ntasks_per_node is not None else config.get("ntasks_per_node")
    ntasks_per_node: int | None
    if raw_ntasks is None:
        ntasks_per_node = None
    else:
        try:
            ntasks_per_node = int(raw_ntasks)
        except (TypeError, ValueError):
            # A non-integer (e.g. a config `ntasks_per_node = "x"`) is a hard
            # error naming the original value — not _coerce_int's "using 0",
            # which would then trip the positive-int guard below with a
            # confusing "got 0" that never echoes what the user wrote.
            err_console.print(
                f"  {c.RED}{g.ERR} Error: --ntasks-per-node must be a positive integer "
                f"(got {raw_ntasks!r}){c.RESET}"
            )
            sys.exit(1)

    args_gpu_type = getattr(args, "gpu_type", None)
    gpu_type = args_gpu_type if args_gpu_type is not None else config.get("gpu_type")
    if gpu_type is not None:
        gpu_type = str(gpu_type)

    args_gpu_format = getattr(args, "gpu_format", None)
    gpu_format = args_gpu_format if args_gpu_format is not None else config.get("gpu_format")

    args_output_dir = getattr(args, "output_dir", None)
    output_dir = _coerce_str(
        args_output_dir if args_output_dir is not None else config.get("output_dir", "logs"),
        "logs", field="output_dir", err_console=err_console)

    args_output_file = getattr(args, "output_file", None)
    output_file = _coerce_str(
        args_output_file if args_output_file is not None else config.get("output_file"),
        None, field="output_file", err_console=err_console)

    # Seed the GPU format from SLURMATE_GPU_FORMAT (default gres_type) so the
    # env var documented in the README actually takes effect in batch mode.
    if gpus > 0 and not gpu_format:
        gpu_format = os.environ.get("SLURMATE_GPU_FORMAT", "gres_type").lower()

    # Validate the resolved GPU format from config/env (the --gpu-format flag is
    # already constrained by argparse choices, but config/env values are not):
    # clamp an unrecognized value to gres_type instead of silently falling
    # through to the constraint-style directives, matching the TUI's behavior.
    _GPU_FORMATS = ("gres_type", "constraint", "gpus", "gpus_per_node", "gpus_per_task")
    if gpu_format is not None:
        gpu_format = str(gpu_format).lower()
        if gpu_format not in _GPU_FORMATS:
            err_console.print(
                f"  {c.YELLOW}{g.WARN} Unknown gpu_format {gpu_format!r}; "
                f"using 'gres_type'{c.RESET}"
            )
            gpu_format = "gres_type"

    # Hard-validate numeric flags so batch mode rejects the same bad input the
    # wizard does (positive cpus/nodes, non-negative gpus/ntasks), instead of
    # emitting Slurm-invalid directives like --cpus-per-task=0 or --nodes=-2.
    if cpus <= 0:
        err_console.print(f"  {c.RED}\u2717 Error: --cpus must be a positive integer (got {cpus}){c.RESET}")
        sys.exit(1)
    if nodes <= 0:
        err_console.print(f"  {c.RED}\u2717 Error: --nodes must be a positive integer (got {nodes}){c.RESET}")
        sys.exit(1)
    if gpus < 0:
        err_console.print(f"  {c.RED}\u2717 Error: --gpus must be a non-negative integer (got {gpus}){c.RESET}")
        sys.exit(1)
    if ntasks_per_node is not None and ntasks_per_node <= 0:
        err_console.print(f"  {c.RED}\u2717 Error: --ntasks-per-node must be a positive integer (got {ntasks_per_node}){c.RESET}")
        sys.exit(1)

    # Hard-validate memory (unless deliberately omitted for a whole-node site,
    # or not supplied at all \u2014 that case is sized from the partition below).
    if memory_val is not None and not mem_omit and not validate_memory(str(memory_val)):
        err_console.print(f"  {c.RED}\u2717 Error: Invalid memory value: {memory_val}{c.RESET}")
        sys.exit(1)

    # --mem-per-cpu (validated as a memory value); takes precedence over --mem.
    args_mem_per_cpu = getattr(args, "mem_per_cpu", None)
    mem_per_cpu = _coerce_str(
        args_mem_per_cpu if args_mem_per_cpu is not None else config.get("mem_per_cpu"),
        None, field="mem_per_cpu", err_console=err_console)
    if mem_per_cpu:
        if not validate_memory(str(mem_per_cpu)):
            err_console.print(f"  {c.RED}\u2717 Error: Invalid --mem-per-cpu value: {mem_per_cpu}{c.RESET}")
            sys.exit(1)
        mem_per_cpu = normalize_memory(str(mem_per_cpu))

    # Node-feature --constraint (Slurm -C), e.g. NERSC Perlmutter's required cpu/gpu.
    args_constraint = getattr(args, "constraint", None)
    constraint = _coerce_str(
        args_constraint if args_constraint is not None else config.get("constraint"),
        None, field="constraint", err_console=err_console)

    # Hard-validate time limit
    if not validate_time(str(time_val)):
        err_console.print(f"  {c.RED}\u2717 Error: Invalid time limit value: {time_val}{c.RESET}")
        sys.exit(1)

    all_parts = fetch_partitions()
    # With no --partition, Slurm uses the site default — and slurmate already
    # knows which that is, from sinfo's "*" marker. Treating the partition as
    # *unknown* instead produced two confidently wrong figures: a queue depth of
    # "0 running / 0 pending" (from `squeue -p ""`) for a job that will land in a
    # partition with hundreds, and the SM-7 memory fallback claiming "this
    # cluster's node memory is unknown" when it is perfectly well known.
    #
    # The default is used for the *derived* figures only. No --partition
    # directive is emitted: adding one the user did not type is what SM-15 was
    # about, and a site's default can differ per user or account.
    effective_partition = partition
    if not partition:
        effective_partition = next(
            (str(p["name"]) for p in all_parts if p.get("is_default")), ""
        )
    part_obj = _get_partition(all_parts, effective_partition)

    # ── Does this cluster actually have what was asked for? ─────────────
    # The point of generating an sbatch script is that it is correct for the
    # cluster you are on, and the commonest way to get a wrong one is to carry a
    # partition or account name over from another site. Checked here, before any
    # script exists, against the sinfo/sacctmgr lists already in hand.
    account_for_check = _coerce_str(
        getattr(args, "account", None) if getattr(args, "account", None) is not None
        else config.get("account"),
        None, field="account", err_console=err_console)
    qos_for_check = _coerce_str(
        getattr(args, "qos", None) if getattr(args, "qos", None) is not None
        else config.get("qos"),
        None, field="qos", err_console=err_console)
    force = bool(getattr(args, "force", False))
    constraint_for_check = _coerce_str(
        getattr(args, "constraint", None) if getattr(args, "constraint", None) is not None
        else config.get("constraint"),
        None, field="constraint", err_console=err_console)
    _check_cluster_targets(
        partition, account_for_check, all_parts,
        qos=qos_for_check, constraint=constraint_for_check,
        force=force, err_console=err_console,
    )

    # No --memory given: size it from the node this partition actually has,
    # rather than emitting a literal that happens to fit the cluster slurmate was
    # written on. Recorded so the summary can say the number was not the user's.
    memory_source: str | None = None
    if memory_val is None and not mem_omit:
        memory_val, memory_source = default_memory_for(part_obj, cpus)

    raw_modules = getattr(args, "modules", None)
    if raw_modules is None:
        cfg_mods = config.get("modules")
        if isinstance(cfg_mods, list):
            mods = [str(m) for m in cfg_mods]
        elif isinstance(cfg_mods, str):
            mods = [m.strip() for m in cfg_mods.split(",") if m.strip()]
        else:
            mods = None
    else:
        mods = [m.strip() for m in raw_modules.split(",") if m.strip()]

    # A module this cluster does not have is the one cross-cluster error that
    # survives submission: sbatch accepts it, the job runs, `module load` prints
    # to stderr, the body executes anyway and Slurm records COMPLETED 0:0 — so
    # the run silently proceeds against whatever toolchain was on PATH. Checked
    # here, before any script exists, and fatal like the partition/account check.
    if mods:
        _check_modules_exist(mods, force=force, err_console=err_console)

    # A gpu_format that this cluster's select plugin does not implement. Checked
    # only when GPUs are actually requested, and fatal like the other
    # cluster-mismatch errors — with --force, since writing a script for a
    # cons_tres cluster from a cons_res one is legitimate. `gpu_format` is a
    # config-file key, so this arrives without the user typing a flag.
    resolved_format = _check_gpu_format(
        getattr(args, "gpu_format", None) or config.get("gpu_format"),
        gpus_requested=bool(gpus),
        force=force,
        err_console=err_console,
        inferred=bool(getattr(args, "gpu_format_inferred", False)),
    )
    # An inferred format that this cluster cannot parse is replaced rather than
    # refused, so the substitution has to reach the script that gets built. Any
    # other outcome returns None and leaves the resolution above untouched.
    if resolved_format:
        gpu_format = resolved_format

    args_env_type = getattr(args, "env_type", None)
    env_type = _coerce_str(
        args_env_type if args_env_type is not None else config.get("env_type"),
        None, field="env_type", err_console=err_console)

    args_env = getattr(args, "env", None)
    env_name = _coerce_str(
        args_env if args_env is not None else config.get("env_name"),
        None, field="env", err_console=err_console)
    if env_name and not env_type:
        env_type = "conda"

    custom_sbatch_val = getattr(args, "custom_sbatch", None)
    if custom_sbatch_val is None:
        cfg_custom = config.get("custom_sbatch")
        if isinstance(cfg_custom, list):
            custom_sbatch_list = [str(f) for f in cfg_custom]
        elif isinstance(cfg_custom, str):
            custom_sbatch_list = _parse_custom_flags(cfg_custom)
        else:
            custom_sbatch_list = None
    else:
        custom_sbatch_list = _parse_custom_flags(custom_sbatch_val)

    args_job_name = getattr(args, "job_name", None)
    args_account = getattr(args, "account", None)
    args_qos = getattr(args, "qos", None)
    args_array = getattr(args, "array", None)
    args_command = getattr(args, "command", None)

    raw_job_name = args_job_name if args_job_name is not None else config.get("job_name", "")
    account = _coerce_str(args_account if args_account is not None else config.get("account"),
                          None, field="account", err_console=err_console)
    qos = _coerce_str(args_qos if args_qos is not None else config.get("qos"),
                      None, field="qos", err_console=err_console)
    array_spec = _coerce_str(args_array if args_array is not None else config.get("array_spec"),
                             None, field="array", err_console=err_console)

    # A custom flag repeating a directive slurmate manages emits a second
    # #SBATCH line. Slurm honours the LAST, so the job would run with the custom
    # value while the summary, the cluster validation and the queue/ETA figures
    # all describe the managed one — and a custom --partition/--account also
    # routes straight past the checks that exist to catch exactly those two.
    # Refused rather than reconciled: for every directive in this set slurmate
    # already has a flag that is validated and reflected everywhere.
    conflicts = managed_custom_flags(getattr(args, "custom_sbatch", None))
    if conflicts:
        for name, owner in conflicts:
            # When the owner *is* the flag they typed — now the common case, since
            # SM-25 made Slurm's own spellings first-class — "use --gres instead"
            # reads like a tautology. Say the actual distinction: pass it as an
            # option rather than inside --custom-sbatch.
            advice = (
                f"Pass {name} as a slurmate option instead of inside "
                f"--custom-sbatch."
                if owner == name
                else f"Use {owner} instead."
            )
            err_console.print(
                f"  {c.RED}\u2717 Error: --custom-sbatch carries {name}, which "
                f"slurmate manages. {advice}{c.RESET}"
            )
        err_console.print(
            f"  {c.GRAY}(A second #SBATCH line for the same directive would win "
            f"over slurmate's, leaving the summary and the ETA describing a "
            f"partition/account the job will not use.){c.RESET}"
        )
        sys.exit(1)

    # Shape-check the array spec the way --time and --memory already are: a
    # reversed range or a zero step is refused by the controller ("Invalid job
    # array specification") after a script that looked fine.
    if array_spec and not validate_array_spec(str(array_spec)):
        err_console.print(
            f"  {c.RED}\u2717 Error: {array_spec_reason(str(array_spec))}{c.RESET}"
        )
        err_console.print(
            f"  {c.GRAY}Expected forms: 1-10, 0-9, 1,3,5, 1-10:2, 1-10%4 "
            f"(a range must not run backwards and a step must be > 0).{c.RESET}"
        )
        sys.exit(1)
    command = _coerce_str(args_command if args_command is not None else config.get("command", ""),
                          "", field="command", err_console=err_console)
    return {
        "job_name": sanitize_job_name(str(raw_job_name)),
        # Kept so the checks can say the name was rewritten. The wizard
        # needs no equivalent: its validator sanitises as the user types,
        # so the change happens in front of them.
        "_job_name_given": str(raw_job_name),
        "account": account,
        "partition": partition,
        # The partition the derived figures (limits, queue depth, ETA, default
        # memory) were computed for — the site default when none was given.
        "_effective_partition": effective_partition,
        "_partition_obj": part_obj,
        "qos": qos,
        "cpus": cpus,
        "memory": None if mem_omit else normalize_memory(str(memory_val)),
        "_memory_source": memory_source,
        "_config_source": config_source(),
        "_config_keys": _config_keys_in_effect(args, config),
        "mem_per_cpu": mem_per_cpu or None,
        "time_limit": str(time_val),
        "nodes": nodes,
        "ntasks_per_node": ntasks_per_node,
        "gpus": gpus,
        "gpu_type": gpu_type or None,
        "gpu_format": gpu_format or None,
        "constraint": constraint,
        "array_spec": array_spec,
        "modules": mods,
        "env_type": env_type,
        "env_name": env_name,
        "output_dir": output_dir,
        "output_file": output_file or None,
        "command": command,
        "custom_sbatch": custom_sbatch_list,
    }


def _partition_issues(
    answers: dict[str, Any], max_array: int | None = None
) -> list[tuple[str, str]]:
    """Resolved ``(level, msg)`` validation issues for the answers.

    A GPU model the partition doesn't statically list may still be valid \u2014 a live
    ``sinfo`` lookup can surface types the cached partition object missed, so widen
    the known set with a one-shot query (only when there's an unrecognized type, to
    avoid a needless call). The same query also reports *how* each model can be
    requested (typed GRES vs. node feature only), which is what catches a
    ``--gres=gpu:<model>:N`` request on a count-only-GRES partition. A model that
    IS in the static list came from a typed ``gpu:MODEL:N`` by construction (that is
    all ``fetch_partitions`` records), so skipping the lookup for it is safe.
    Single source of truth shared by the CLI summary and the pre-submit guard.
    """
    part = answers.get("_partition_obj")
    if not part:
        return []
    # A site limit, so it needs a live query — done by the caller rather than
    # inside validate_job_config, which the TUI calls on every redraw and which
    # must stay subprocess-free. Fetched here only when the caller did not
    # already have it (the pre-submit guard calls this directly).
    if max_array is None and answers.get("array_spec"):
        max_array = fetch_max_array_size()
    extra_gpu_types: list[str] = []
    feature_only: list[str] = []
    constraint_types: list[str] | None = None
    gpu_type = answers.get("gpu_type")
    if gpu_type and str(gpu_type).lower() != "any":
        static = {str(g).lower() for g in part.get("gpu_types", [])}
        # A statically-listed model is a typed GRES by construction, so the
        # lookup buys nothing for the *default* format — but it is the only way
        # to answer the mirror question, "is this model also a node feature?",
        # which is what gpu_format 'constraint' turns on. So it runs for an
        # unrecognized type (as before) or for a constraint request, and stays
        # skipped on the common path.
        wants_constraint = str(
            answers.get("gpu_format")
            or os.environ.get("SLURMATE_GPU_FORMAT", "gres_type")
        ).lower() == "constraint"
        if str(gpu_type).lower() not in static or wants_constraint:
            part_name = part.get("name", "")
            if part_name:
                try:
                    sources = fetch_gpu_type_sources(part_name)
                    extra_gpu_types = sorted(
                        set(sources["typed"]) | set(sources["feature"])
                    )
                    feature_only = list(sources["feature"])
                    # Only when the query actually answered: an empty list has to
                    # mean "no GPU model is a node feature here", so a failed or
                    # skipped lookup must stay None instead of borrowing that.
                    if extra_gpu_types:
                        constraint_types = list(sources.get("constraint") or [])
                except Exception:
                    extra_gpu_types = []
                    feature_only = []
                    constraint_types = None
    return validate_job_config(
        answers, extra_gpu_types=extra_gpu_types, feature_only_gpu_types=feature_only,
        constraint_gpu_types=constraint_types, max_array_size=max_array,
    )


def _check_gpu_format(
    gpu_format: Any, *, gpus_requested: bool, force: bool, err_console: Console,
    inferred: bool = False,
) -> str | None:
    """Reject a ``gpu_format`` whose syntax this cluster's Slurm cannot parse.

    Returns a replacement format, or ``None`` for "leave it alone" — which is
    every case but one. The exception: a format that was *inferred* (from a typed
    ``--gpus a100:2``) and that this cluster cannot parse becomes
    ``"gres_type"``. An inferred format is slurmate's reading of a spelling, not
    a request the user made, so a site running something other than
    ``select/cons_tres`` should get a different rendering rather than a failed job
    the default rendering expresses perfectly well. An explicitly chosen format
    is still an error, because there the user asked for that syntax by name.
    """
    if not gpu_format or not gpus_requested:
        return None
    try:
        reason = unsupported_gpu_format(str(gpu_format), fetch_select_type())
    except Exception as e:                    # a broken probe must not block a job
        logger.debug(f"select type check failed: {e}")
        return None
    if not reason:
        return None
    if inferred:
        err_console.print(
            f"  [dim]--gpus '<type>:count' reads as gpu_format "
            f"'{escape(str(gpu_format))}', which this cluster cannot parse; "
            f"using 'gres_type' instead (pass --gpu-format to choose).[/]"
        )
        return "gres_type"
    if force:
        err_console.print(f"  [yellow]{g.WARN} Warning: {escape(reason)}[/]")
        return None
    err_console.print(f"  [red]{g.ERR} Error: {escape(reason)}[/]")
    err_console.print(
        "  [dim]Pass --force to generate the script anyway "
        "(e.g. for another cluster).[/]"
    )
    sys.exit(1)


def _check_modules_exist(
    modules: list[str], *, force: bool, err_console: Console
) -> None:
    """Reject ``module load`` names this cluster does not have.

    Fatal by default, ``--force`` downgrades to a warning — the same treatment as
    an unknown partition, for the same reason: writing a script to carry to
    another cluster is legitimate, but it must not be the silent default. Stays
    quiet when there is no module system to ask.
    """
    try:
        issues = check_modules(modules)
    except Exception as e:                    # a broken probe must not block a job
        logger.debug(f"module check failed: {e}")
        return
    if not issues:
        return
    for _level, msg in issues:
        if force:
            err_console.print(f"  [yellow]\u26a0 Warning: {escape(msg)}[/]")
        else:
            err_console.print(f"  [red]\u2717 Error: {escape(msg)}[/]")
    if not force:
        err_console.print(
            "  [dim]Pass --force to generate the script anyway "
            "(e.g. for another cluster).[/]"
        )
        sys.exit(1)


def _warn_runtime_targets(
    script: str, answers: dict[str, Any], console: Console, *,
    will_create: bool = True,
) -> None:
    """Warn about log paths that only resolve when the job actually runs.

    A partition or account that does not exist is caught before the script is
    written; a log directory that cannot be created is caught by nothing until
    Slurm tries to open the file and kills the job. It stays a warning because a
    path can be unwritable from the login node and valid on the compute node.
    """
    issues: list[tuple[str, str]] = []
    try:
        issues += check_log_dirs(script, will_create=will_create)
    except Exception as e:                       # never fail a summary over a check
        logger.debug(f"log dir check failed: {e}")
    for _level, msg in issues:
        console.print(f"  [yellow]\u26a0 Warning: {escape(msg)}[/]")


def _validate_partition_limits(
    answers: dict[str, Any], console: Console, max_array: int | None = None
) -> None:
    # escape() the whole message: only user-supplied values can carry Rich-markup
    # metacharacters ('[', ']'); the static text never does, so escaping the lot is
    # equivalent to escaping each interpolated value and can't accidentally miss one.
    for level, msg in _partition_issues(answers, max_array):
        if level == "error":
            console.print(f"  [red]\u2717 Error: {escape(msg)}[/]")
        else:
            console.print(f"  [yellow]\u26a0 Warning: {escape(msg)}[/]")


def site_check_issues(answers: dict[str, Any]) -> list[tuple[str, str]]:
    """Cluster-membership checks as ``(level, message)``, making no exit.

    These all lived on the batch path, where they are fatal before a script
    exists — which left the **wizard** unchecked, and the wizard is the default
    interface *and* offers "Enter partition name manually…". So a name that the
    non-interactive path rejects outright was accepted silently by the
    interactive one. Returning issues rather than exiting lets the wizard offer
    "go back to edit" while the batch path keeps failing fast.
    """
    out: list[tuple[str, str]] = []
    partition = str(answers.get("partition") or "")
    account = str(answers.get("account") or "")
    qos = str(answers.get("qos") or "")
    constraint = str(answers.get("constraint") or "")
    try:
        if partition or account or qos or constraint:
            out += validate_cluster_targets(
                partition or None, account or None,
                qos=qos or None, constraint=constraint or None,
                known_partitions=fetch_all_partition_names() if partition else None,
                known_accounts=fetch_user_accounts() if account else None,
                known_qos=fetch_known_qos() if qos else None,
                known_features=fetch_node_features() if constraint else None,
            )
        note = job_name_change_note(str(answers.get("_job_name_given") or ""))
        if note:
            out.append(("warning", note))
        mods = answers.get("modules") or []
        if mods:
            # Raised to "error" to match the batch path, which SM-13 asked to be
            # fatal-with---force. Keeping it a warning here meant the wizard would
            # submit a job the non-interactive path refuses — and the failure it
            # predicts is real: the job queues, starts, and dies on `module load`.
            # An error still lets the wizard offer "go back to edit"; only the
            # batch path exits.
            out += [("error", msg) for _lvl, msg in check_modules([str(m) for m in mods])]
        # The same late failure, one field over: --modules was validated and
        # --env was not, though the env list was already being fetched for the
        # wizard's picker. Kept a warning rather than raised to an error like the
        # module check: conda's env list is a property of *this* machine, and a
        # site where the compute nodes see an envs dir the login node does not
        # would turn an error into a false refusal.
        out += check_conda_env(
            str(answers.get("env_name") or ""),
            str(answers.get("env_type") or ""),
            [str(m) for m in mods],
        )
        if answers.get("gpus"):
            reason = unsupported_gpu_format(
                str(answers.get("gpu_format") or ""), fetch_select_type()
            )
            if reason:
                out.append(("error", reason))
            # `--gpus-per-task` is per *task*, so Slurm needs a task count to
            # resolve it: on its own it is refused with "Invalid generic resource
            # (gres) specification" — measured — while the same request with
            # --ntasks-per-node is accepted. Cluster-agnostic: the requirement is
            # in the flag, not the site.
            if (
                str(answers.get("gpu_format") or "").strip().lower() == "gpus_per_task"
                and not answers.get("ntasks_per_node")
            ):
                out.append((
                    "error",
                    "gpu_format 'gpus_per_task' needs a task count — Slurm refuses "
                    "--gpus-per-task without one ('Invalid generic resource (gres) "
                    "specification'). Set --ntasks-per-node, or use 'gres_type' "
                    "(the default) or 'gpus_per_node'",
                ))
        spec = str(answers.get("array_spec") or "")
        if spec and not validate_array_spec(spec):
            out.append(("error", array_spec_reason(str(spec))))
        # An env name that will never be activated: the user asked for an
        # environment and for a strategy that emits nothing, and the script
        # silently does neither.
        if answers.get("env_name") and not env_activation_emitted(
            answers.get("env_name"), answers.get("env_type")
        ):
            out.append((
                "warning",
                f"environment '{answers.get('env_name')}' will NOT be activated: "
                f"env_type is '{answers.get('env_type') or 'none'}', which emits no "
                f"activation line — set --env-type conda/mamba/venv, or activate it "
                f"yourself in the command",
            ))
        smuggled = command_injects_directives(answers.get("command"))
        if smuggled:
            out.append((
                "error",
                f"the command begins with a #SBATCH line ({smuggled}) — Slurm is "
                f"still reading directives there, so it would take effect "
                f"unvalidated and unshown. Use the matching flag, or "
                f"--custom-sbatch for anything slurmate does not manage",
            ))
        for name, owner in managed_custom_flags(answers.get("custom_sbatch")):
            out.append((
                "error",
                f"custom flag {name} duplicates a directive slurmate manages; "
                f"use {owner} instead",
            ))
    except Exception as e:            # a probe failure must never block a job
        logger.debug(f"site checks failed: {e}")
    return out


def _hard_errors(answers: dict[str, Any]) -> list[str]:
    """Error-level issues only \u2014 a configuration Slurm will reject outright."""
    return [
        msg
        for level, msg in (_partition_issues(answers) + site_check_issues(answers))
        if level == "error"
    ]


_REQUIRED_FIELDS = [("job_name", "Job name"), ("partition", "Partition"), ("command", "Command to run")]


def _warn_missing_required(answers: dict[str, Any], console: Console) -> list[str]:
    """Print a reminder for any required field left blank; return the labels."""
    missing = [label for key, label in _REQUIRED_FIELDS if not answers.get(key)]
    if missing:
        console.print(
            f"  [yellow]{g.WARN} Missing recommended fields:[/] {', '.join(missing)}"
            f" [dim](go back in the wizard, or pass them as flags)[/]"
        )
    return missing


def build_and_show(answers: dict[str, Any], console: Console) -> tuple[str, dict[str, Any]]:
    script = build_from_answers(answers)

    su_estimate = estimate_su(
        answers.get("cpus", 1),
        answers.get("time_limit", "02:00:00"),
        answers.get("nodes", 1),
        answers.get("ntasks_per_node"),
        # An array's cost is per-task × task count. Without this a 1000-task
        # array reported the same figure as a single job.
        array_task_count(str(answers.get("array_spec") or "")),
        # slurmate has no --ntasks, so --custom-sbatch is the only way to express
        # an MPI job — and the estimate ignored it entirely.
        custom_ntasks(answers.get("custom_sbatch")),
    )

    # Pass the whole request, not just the node count: an ETA computed from node
    # states alone reports "immediate" for a GPU job on a partition whose GPUs are
    # all allocated. fetch_queue_eta needs the resources to ask the scheduler.
    queue_info = fetch_queue_eta(
        answers.get("_effective_partition") or answers.get("partition", ""),
        req_nodes=answers.get("nodes", 1),
        cpus=answers.get("cpus", 0) or 0,
        mem_mb=resolve_request_mem_mb(answers),
        gpus_per_node=answers.get("gpus", 0) or 0,
        gpu_type=answers.get("gpu_type", "") or "",
        time_limit=answers.get("time_limit", "") or "",
        account=answers.get("account", "") or "",
        qos=answers.get("qos", "") or "",
        array_spec=answers.get("array_spec", "") or "",
        constraint=answers.get("constraint", "") or "",
        # Hand Slurm the script it will actually receive, rather than an argv
        # rebuilt from the same fields — see _scheduler_verdict.
        script=script,
    )

    # The ETA's first choice is Slurm's own verdict. When sbatch cannot be
    # reached it falls through to a queue-depth heuristic, which used to print a
    # confident "~7min" on the same screen as a warning saying the request
    # exceeds the partition. Ask the partition's own figures in that case — but
    # only when the scheduler stayed silent: if Slurm *placed* the job it knows
    # better than our advertised capacity does (a heterogeneous partition's
    # figures are floors, and a larger node may well have taken it).
    max_array = fetch_max_array_size() if answers.get("array_spec") else None
    if queue_info.get("feasible", True) and queue_info.get("source") != "scheduler":
        reason = capacity_refusal(answers.get("_partition_obj"), answers, max_array)
        if reason:
            queue_info = {**queue_info, "feasible": False, "reason": reason,
                          "eta_label": "never", "eta_seconds": 0,
                          # Derived from the partition's own figures, so it is a
                          # statement about the request, not about right now.
                          "refusal_is_permanent": True}

    _validate_partition_limits(answers, console, max_array)
    # The wizard never passed through the batch path's cluster checks, so a
    # manually-typed partition/account/qos went unvalidated there. Reported here,
    # where both paths meet, and repeated on the batch path only as a fatal
    # pre-script check (so --print still cannot emit a rejected name).
    for level, msg in site_check_issues(answers):
        head = str(msg).splitlines()[0]
        marker, colour = (g.ERR, "red") if level == "error" else (g.WARN, "yellow")
        console.print(f"  [{colour}]{marker} {escape(head)}[/]")
    _warn_runtime_targets(script, answers, console)
    _show_script_and_summary(console, script, answers, su_estimate, queue_info)
    # After the summary, so the reader has the request in view when told it was
    # refused. Here rather than in each caller because the wizard, --dry-run and
    # --yes all come through this function, and wiring a check into one caller at
    # a time is how --print ended up the only mode that never learned the verdict.
    _note_scheduler_refusal(queue_info, console)
    _warn_missing_required(answers, console)
    _note_defaulted_memory(answers, console)
    _note_config_source(answers, console)
    _note_default_partition(answers, console)
    _note_mock_mode(console)
    return script, queue_info


# How each fetch_queue_eta tier should describe itself. "scheduler" is Slurm's
# own answer and needs no qualifier; the others are inferences of decreasing
# strength and must not look like it.
_ETA_TIER_NOTE = {
    "resources": "estimated from free capacity",
    "pressure": "estimated from queue depth",
    "unknown": "not measurable here",
}


def _qualified_eta(queue_info: dict[str, Any]) -> str:
    """The ETA label, with its provenance when it is not the scheduler's answer."""
    label = str(queue_info.get("eta_label", ""))
    note = _ETA_TIER_NOTE.get(str(queue_info.get("source") or ""))
    return f"{label} ({note})" if note else label


def _note_scheduler_refusal(queue_info: dict[str, Any], console: Console) -> None:
    """Say, as an error, that this job has already been refused.

    The verdict was reaching the screen only as the summary's ``ETA: never —
    <reason>`` row, which renders "this cannot run at all" as a *time estimate*,
    in the same weight as a queue depth, while strictly lesser problems (a time
    limit over the partition's, an array index over MaxArraySize) each got a
    marked warning line of their own. Measured on Booth's Mercury, where a user
    with no default account is refused every account-less script: the fatal fact
    appeared as one unmarked summary row, with rc=0.

    Attribution follows the source. Slurm's own refusal is authoritative; a
    refusal derived from advertised partition figures is ours, and saying "Slurm
    refuses" for it would credit the controller with an answer it never gave.
    """
    if queue_info.get("feasible", True):
        return
    reason = str(queue_info.get("reason") or "").strip()
    if not reason:
        return
    if queue_info.get("source") != "scheduler":
        console.print(
            f"  [red]{g.ERR} This job cannot run as requested: {escape(reason)}[/]"
        )
    elif queue_info.get("refusal_is_permanent", True):
        console.print(f"  [red]{g.ERR} Slurm refuses this job: {escape(reason)}[/]")
    elif queue_info.get("refusal_is_transient", False):
        # The request is fine; the moment isn't. Saying "refuses" here would send
        # the user looking for a mistake in a script that has none.
        console.print(
            f"  [yellow]{g.WARN} Slurm cannot take this job right now: "
            f"{escape(reason)}[/] [dim](the script is valid; this clears on its "
            f"own)[/]"
        )
    else:
        # Refused, and neither list recognises the wording. Report what the
        # controller said and stop there: claiming it clears on its own is how a
        # job asking for too many nodes got told its script was fine.
        console.print(
            f"  [yellow]{g.WARN} Slurm would not accept this job: "
            f"{escape(reason)}[/] [dim](the controller's own words; slurmate "
            f"cannot tell whether this clears on its own)[/]"
        )


def _note_mock_mode(console: Console) -> None:
    """Say once that everything cluster-derived above is demo data.

    Marking each figure individually would mean touching every message; one
    statement in the same stream as the other warnings covers the partition list,
    its limits, the queue depth and the ETA together — and it cannot be mistaken
    for a reading, which a bare "12 running / 5 pending" can.
    """
    if not is_mock():
        return
    console.print(
        f"  [yellow]{g.WARN} SLURMATE_MOCK is set: the partition list, its limits, "
        f"the queue depth and the ETA above are synthetic demo data, not this "
        f"cluster.[/] [dim](unset it, or drop --demo, for real values.)[/]"
    )


def _note_default_partition(answers: dict[str, Any], console: Console) -> None:
    """Say when the figures above describe the site default, not a chosen one."""
    effective = str(answers.get("_effective_partition") or "")
    if not effective or answers.get("partition"):
        return
    console.print(
        f"  [dim]No partition given; the limits, queue depth and ETA above are "
        f"for this cluster's default partition '{escape(effective)}', which is "
        f"where Slurm will place the job. No --partition directive was added.[/]"
    )


def _note_config_source(answers: dict[str, Any], console: Console) -> None:
    """Say which config file supplied directives the user did not type.

    A ``.slurmate.toml`` travels with a project into git and onto the next
    cluster, so the summary — the one surface that says "here is what you are
    about to submit" — has to name where the values came from. The loader also
    says this on stderr; the summary is stdout, which is what a reader of
    ``--dry-run`` output is actually looking at.
    """
    source = str(answers.get("_config_source") or "")
    keys = [str(k) for k in (answers.get("_config_keys") or [])]
    if not source or not keys:
        return
    console.print(
        f"  [dim]Defaults from {escape(source)}: {escape(', '.join(keys))} "
        f"(flags override the file).[/]"
    )


def _note_defaulted_memory(answers: dict[str, Any], console: Console) -> None:
    """Say so when the ``--mem`` in the script is slurmate's guess, not the user's.

    A number nobody typed should never look like a number somebody typed —
    especially the fallback, which is the one case where it has no relationship
    to this cluster at all.
    """
    source = answers.get("_memory_source")
    if not source:
        return
    mem = answers.get("memory")
    if source == "partition":
        # Name the partition the figure was actually derived from. With no
        # --partition the number comes from the site default (which slurmate
        # resolves, and says so two lines further down), but this line read the
        # *user's* answer and printed "from '?' node memory" — a provenance note
        # whose entire job is to say where a number came from, admitting it does
        # not know, on the default path of every cluster tested.
        part = answers.get("_partition_obj") or {}
        origin = str(answers.get("partition") or part.get("name") or "")
        whose = f"'{escape(origin)}'" if origin else "this cluster's"
        console.print(
            f"  [dim]Memory not specified; sized to {escape(str(mem))} from "
            f"{whose} node memory "
            f"(pass --memory to set it, or --memory none to omit --mem).[/]"
        )
    else:
        console.print(
            f"  [yellow]{g.WARN} Memory not specified and this cluster's node memory is "
            f"unknown; defaulted to {escape(str(mem))}[/] "
            f"[dim](pass --memory to set it).[/]"
        )


def _clip_cells(text: str, limit: int) -> tuple[str, bool]:
    """Truncate to ``limit`` display cells; returns ``(text, was_clipped)``.

    Cell-aware rather than character-aware for the same reason the panel measures
    with ``cell_len``: one CJK glyph occupies two columns, and counting code
    points would push the line past the border it was trimmed to fit.
    """
    if limit <= 0 or cell_len(text) <= limit:
        return text, False
    marker = g.ELLIPSIS
    room = max(0, limit - cell_len(marker))
    kept: list[str] = []
    used = 0
    for ch in text:
        width = cell_len(ch)
        if used + width > room:
            break
        kept.append(ch)
        used += width
    return "".join(kept) + marker, True


def _show_script_and_summary(console: Console, script: str, answers: dict[str, Any],
                              su_estimate: str, queue_info: dict[str, Any] | None = None) -> None:
    print()
    script_lines = script.split("\n")
    num_w = len(str(len(script_lines)))
    # Never let this box wrap. rich wraps mid-token, which renders
    # "#SBATCH --output=<long path>" as a bare "#SBATCH" (a no-op directive) on
    # one line and "--output=…" — split mid-path — on the next, where it reads as
    # a *shell command*. `bash -n` accepts that, so a user who copies the box out
    # gets "command not found" at run time and none of the --output they asked
    # for. Clipping makes the box visibly an excerpt instead of a broken script;
    # --print remains the copy-safe form and the note below says so.
    text_w = max(8, console.width - 4 - (num_w + 1))
    clipped = False
    shown_lines: list[str] = []
    for raw_line in script_lines:
        shown, was_clipped = _clip_cells(raw_line, text_w)
        clipped = clipped or was_clipped
        shown_lines.append(shown)
    script_lines = shown_lines
    body = Text()
    for i, ln in enumerate(script_lines, 1):
        body.append(f"{i:>{num_w}} ", style="bright_black")
        if ln.startswith("#!") or (ln.startswith("#") and not ln.startswith("#SBATCH")):
            body.append(ln, style="bright_black")
        elif ln.startswith("#SBATCH") and "=" in ln:
            key, val = ln.split("=", 1)
            body.append(key + "=", style="green")
            body.append(val, style="white")
        elif ln.startswith("#SBATCH"):
            body.append(ln, style="green")
        else:
            body.append(ln, style="cyan")
        if i < len(script_lines):
            body.append("\n")

    # Measure display cells, not code points, so wide (CJK) glyphs in a command
    # don't overflow/misalign the panel border (each renders as 2 cells).
    script_w = max(num_w + 1 + cell_len(ln) for ln in script_lines)
    title_text = "Generated sbatch script"
    script_panel = Panel(
        body,
        title=f"[bold #ff0080]{title_text}[/]",
        border_style="bright_magenta",
        width=script_w + 4,
        padding=(0, 1),
    )

    # Share the ordered field list with the in-TUI Review step (job_summary_rows)
    # so both summaries agree on what's shown; append the CLI-only cost/queue rows.
    rows: list[tuple[str, str, str]] = []
    for label, val in job_summary_rows(answers):
        style = "magenta" if label == "QoS" else "cyan"
        # Collapse a multi-line command to a single summary line (the full text
        # is still in the script panel) so the panel width stays correct.
        rows.append((f"{label}:", val.replace("\n", " \u21b5 "), style))
    rows.append(("Estimated CPU-hours:", f"{su_estimate}", "#ffaa00"))
    # GPU-hours too, when the job asks for GPUs: core-hours are honest but on a GPU
    # site they are not the number that gets billed. The multiplier follows the
    # chosen gpu_format (per-node vs per-task vs job-wide).
    gpu_hours = estimate_gpu_hours(
        answers.get("gpus", 0) or 0,
        answers.get("time_limit", "02:00:00"),
        answers.get("nodes", 1) or 1,
        answers.get("gpu_format"),
        answers.get("ntasks_per_node"),
        array_task_count(str(answers.get("array_spec") or "")),
        custom_ntasks(answers.get("custom_sbatch")),
    )
    if gpu_hours:
        rows.append(("Estimated GPU-hours:", gpu_hours, "#ffaa00"))
    if queue_info:
        # Nothing about a partition this cluster does not have is a measurement:
        # `squeue -p <nonexistent>` returns no rows, which is reported as a real
        # "0 running / 0 pending", and the queue-depth heuristic then answers with
        # a flat constant. Say what it is instead of dressing a guess as a reading.
        part_obj = answers.get("_partition_obj") or {}
        part_unknown = bool(part_obj.get("_unknown"))
        # Distinguish the two reasons, as the capacity message already does: with
        # an unreadable partition list the partition may well exist, and saying it
        # is "not on this cluster" is the false rejection the SM-4 restraint
        # forbids. This renderer keyed off _unknown alone and so made that claim
        # in two more rows.
        why_unknown = (
            "the partition list could not be read"
            if part_obj.get("_unknown_reason") == "unreadable"
            else "partition not on this cluster"
        )
        if part_unknown:
            rows.append(("Queue:", f"unknown — {why_unknown}", "#ffaa00"))
        elif not queue_info.get("queue_known", True):
            # squeue failed or timed out; 0/0 would present a failed query as an
            # idle queue.
            rows.append(("Queue:", "unknown — could not read the queue", "#ffaa00"))
        else:
            depth = f"{queue_info['running']} running / {queue_info['pending']} pending"
            if is_mock():
                depth += " (simulated)"
            rows.append(("Queue:", depth, "white"))
        # A request Slurm has already refused gets its refusal, not a wait time:
        # "~60s" for a job that can never start is a confident lie, and the ETA
        # row is the one line a user reads to decide whether to submit.
        if queue_info.get("feasible", True) is False:
            reason = str(queue_info.get("reason") or "the scheduler rejected this request")
            # The label and its severity come from the result, not from a second
            # copy of the decision here: this row said "never" for a transient
            # submit-count cap while the advisory below it said the script was
            # valid and the condition temporary.
            label = str(queue_info.get("eta_label") or "never")
            permanent = bool(queue_info.get("refusal_is_permanent", True))
            rows.append(("ETA:", f"{label} — {reason}", "red" if permanent else "#ffaa00"))
        elif part_unknown:
            rows.append(("ETA:", f"unknown — {why_unknown}", "#ffaa00"))
        else:
            eta_color = "green" if queue_info["eta_seconds"] < 3600 else "#ffaa00"
            # fetch_queue_eta returns `source` naming which of its three tiers
            # answered, precisely so this row can qualify itself. Dropping it made
            # Slurm's own backfill placement and a queue-depth heuristic — which
            # returns a flat constant — typographically identical.
            rows.append(("ETA:", _qualified_eta(queue_info), eta_color))

    label_w = max(len(label) for label, _, _ in rows)
    summary_w = max(label_w + 2 + cell_len(val) for label, val, _ in rows)
    # escape() every user-controlled value: a command/flag/etc. containing Rich
    # markup like "[/]" would otherwise raise MarkupError (aborting the run) or
    # silently drop bracketed text (e.g. a "[abc]" glob) from the summary.
    summary = "\n".join(
        f"[bold bright_black]{label:<{label_w}}  [/][{style}]{escape(val)}[/]"
        for label, val, style in rows
    )

    s_title = "Summary — SIMULATED (SLURMATE_MOCK)" if is_mock() else "Summary"
    summary_panel = Panel(summary, title=f"[bold cyan]{s_title}[/]", border_style="cyan",
                          width=summary_w + 4, padding=(0, 1))

    # Use the width smartly: place the two panels side by side when the terminal
    # is wide enough, otherwise fall back to stacking them.
    if console.width >= (script_w + 4) + (summary_w + 4) + 2:
        grid = Table.grid(padding=(0, 2))
        grid.add_column()
        grid.add_column()
        grid.add_row(script_panel, summary_panel)
        console.print(grid)
    else:
        console.print(script_panel)
        print()
        console.print(summary_panel)

    if clipped:
        console.print(
            f"  [dim]Lines wider than this terminal are clipped with "
            f"'{g.ELLIPSIS}': the box above is an excerpt, not something to copy. "
            f"Use --print for the exact script.[/]"
        )


def _editor_command() -> list[str]:
    """Resolve $EDITOR/$VISUAL into an argv list.

    Split on shell words so a command with flags (``code --wait``, ``emacs -nw``)
    works, treat an empty/whitespace value as unset, and fall back to vim.
    """
    import shlex
    raw = (os.environ.get("EDITOR") or os.environ.get("VISUAL") or "").strip()
    if not raw:
        return ["vim"]
    try:
        argv = shlex.split(raw)
    except ValueError:
        argv = [raw]
    return argv or ["vim"]


def _edit_script_in_editor(script: str) -> str:
    argv = _editor_command()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False,
                                     encoding="utf-8") as f:
        f.write(script)
        tmp_path = f.name
    try:
        subprocess.run([*argv, tmp_path], check=False)
        with open(tmp_path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        # exec failure (editor not found / not executable) — check=False only
        # suppresses non-zero exit codes, not the exec error. Keep the current
        # script instead of crashing the whole wizard with a traceback.
        print(f"  {c.YELLOW}{g.WARN} Could not open editor {' '.join(argv)!r}: {e}{c.RESET}")
        return script
    finally:
        os.unlink(tmp_path)


def _save_script(script: str, default_name: str) -> None:
    """Prompt for a path and write the script (returns to caller either way)."""
    import questionary

    from .theme import questionary_style
    QS = questionary_style()
    path = questionary.text("Save as (Esc to cancel):", default=default_name, qmark="", style=QS).ask()
    if not path or not path.strip():
        print(f"  {c.GRAY}Save cancelled.{c.RESET}")
        return
    path = os.path.expanduser(path.strip())
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        # surrogateescape reverses what argv decoding did: under a non-UTF-8
        # locale (LC_ALL=C makes the filesystem encoding ASCII) a --command with
        # UTF-8 bytes arrives as lone surrogates, which a strict UTF-8 write
        # refuses. Round-tripping them writes the user's original bytes, so the
        # saved script is what the shell would have seen.
        write_private_text(path, script)
        print(f"  {c.GREEN}{g.OK} Saved to {path}{c.RESET} "
              f"{c.GRAY}(mode 600 — it contains your command verbatim){c.RESET}")
        # The same handover as --print, and so the same SM-24 exposure: this
        # script is the user's to submit, so nothing will create the directories
        # its --output/--error point at, and Slurm accepts the path, discards
        # what the job writes and reports COMPLETED anyway. Wiring that check
        # into --print alone left this second artifact-handover path uncovered —
        # which is the shape of half the findings in the portability report.
        for _level, msg in check_log_dirs(script, will_create=False):
            print(f"  {c.YELLOW}{g.WARN} Warning: {msg}{c.RESET}")
    except (OSError, UnicodeError) as e:
        print(f"  {c.RED}{g.ERR} Could not save: {e}{c.RESET}")


def _save_submitted_script(script: str, job_name: str, job_id: str,
                           directory: str | None = None) -> str | None:
    """Write the exact submitted script for reproducibility; return the path.

    Writes into ``directory`` (e.g. ``SLURMATE_LOG_DIR``) or the working dir, and
    returns ``None`` if the write actually failed — so the caller only reports
    "Script saved" when a file was really written.
    """
    safe = sanitize_job_name(job_name) or "slurm"
    directory = directory or os.getcwd()
    path = os.path.join(directory, f"{safe}-{job_id}.sh")
    try:
        os.makedirs(directory, exist_ok=True)
        write_private_text(path, script)
        return path
    except (OSError, UnicodeError) as e:
        # This runs *after* a successful submission, so an uncaught exception
        # here turns a queued job into a traceback: the job is fine and the
        # bookkeeping is what failed. Reported, never raised.
        print(f"  {c.YELLOW}{g.WARN} Could not save script copy: {e}{c.RESET}")
        return None


def _no_save_requested(save_script: bool) -> bool:
    if not save_script:
        return True
    return os.environ.get("SLURMATE_NO_SAVE", "").lower() in ("1", "true", "yes")


def _submit_and_report(script: str, answers: dict[str, Any], console: Console,
                       save_script: bool = True) -> None:
    """Submit the job and print the result, log path, and follow-up hints."""
    job_name = answers.get("job_name", "") or "slurm"
    retcode, stdout, stderr = submit_sbatch(script, job_name=job_name)
    if retcode != 0:
        # Submission errors go to stderr so they don't pollute stdout pipelines.
        print(f"  {c.RED}{g.ERR} Submission failed (exit {retcode}){c.RESET}", file=sys.stderr)
        if stdout:
            print(f"  {c.GRAY}{stdout}{c.RESET}", file=sys.stderr)
        if stderr:
            print(f"  {c.RED}{stderr}{c.RESET}", file=sys.stderr)
        sys.exit(1)

    # An empty stdout with rc 0 has two very different causes and this inferred
    # the wrong one. Mock mode (or no sbatch at all) short-circuits before
    # running anything — but a real sbatch that exits 0 and prints nothing has
    # very likely *submitted the job*, and telling that user "not actually
    # submitted" invites them to submit a duplicate. Ask which case it is rather
    # than reading it off the emptiness.
    raw_out = stdout.strip()
    if not raw_out:
        if is_mock() or not is_tool_available("sbatch"):
            print(f"  {c.YELLOW}(mock mode — not actually submitted){c.RESET}")
        else:
            print(f"  {c.GREEN}{g.OK} Submitted!{c.RESET} "
                  f"{c.YELLOW}(sbatch exited 0 but printed no job ID — "
                  f"check `squeue -u $USER` before resubmitting){c.RESET}")
        if stderr:
            print(f"  {c.GRAY}{stderr}{c.RESET}")
        return

    # `sbatch --parsable` returns "jobid", or "jobid;cluster" on a federated
    # setup. A site's sbatch *wrapper* can also print a policy notice on stdout,
    # in which case the banner used to become the "job id" and travel into the
    # hints and the saved filename — so match the expected shape instead of
    # trusting the whole of stdout.
    job_id = parse_submitted_job_id(raw_out)
    if not job_id:
        # Submitted (rc 0) but the id is not where it should be. Report the
        # success and hand over the raw output rather than printing a fabricated
        # id and hints built from it.
        print(f"  {c.GREEN}{g.OK} Submitted!{c.RESET} "
              f"{c.YELLOW}(could not read the job ID from sbatch's output){c.RESET}")
        print(f"  {c.GRAY}sbatch said: {raw_out}{c.RESET}")
        return

    print(f"  {c.GREEN}{g.OK} Submitted!{c.RESET} Job ID: {c.CYAN}{job_id}{c.RESET}")

    # Save a copy of the exact submitted script for reproducibility — into
    # SLURMATE_LOG_DIR when set, else the CWD — and only report success when the
    # write actually happened. Skippable via --no-save-script / SLURMATE_NO_SAVE=1.
    if not _no_save_requested(save_script):
        log_dir = os.environ.get("SLURMATE_LOG_DIR")
        saved = _save_submitted_script(script, job_name, job_id, directory=log_dir)
        if saved:
            # Say that a copy of the command now exists on disk: the file is the
            # exact submitted script, so a token pasted into --command is in it.
            print(f"  {c.GRAY}Script saved: {saved} "
                  f"(mode 600 — contains your command){c.RESET}")

    # Read the actual --output path from the generated script (source of truth).
    # effective_log_path takes the LAST output directive (what Slurm honours) and
    # understands the short/space spellings, so a hand-edited or custom-flag
    # `-o`/`--output PATH` no longer makes this point at a file the job never wrote.
    log_path = (effective_log_path(script, "output")
                or f"{answers.get('job_name', '') or 'slurm'}-%j.out")
    # Resolve every pattern whose value we know — %j/%A (this job id), %x (the
    # job name) and %u (the user) — in a single pass, so "%%" stays a literal
    # percent rather than leaving a bare "%" for the next substitution to
    # misread. What is left is genuinely unknowable before the job starts.
    resolved_log, unresolved = expand_log_pattern(
        log_path,
        job_id=job_id,
        job_name=str(answers.get("job_name", "") or ""),
        user=current_username(),
    )
    print(f"  {c.GRAY}Log path: {resolved_log}{c.RESET}")
    print(f"  {c.GRAY}Hints:{c.RESET}")
    print(f"    squeue -j {job_id}")
    if unresolved:
        # Offering `tail -f` on a path that still contains %a/%N would point at a
        # filename Slurm never writes.
        print(f"    ls {os.path.dirname(resolved_log) or '.'}"
              f"    {c.GRAY}# {', '.join(unresolved)} varies per "
              f"{'task' if '%a' in unresolved else 'node/task'}{c.RESET}")
    else:
        print(f"    tail -f {resolved_log}")
    print(f"    scancel {job_id}")


# ── CLI ──────────────────────────────────────────────────────────────────

def _check_custom_sbatch_form(argv: list[str]) -> None:
    """Give a usable error for ``--custom-sbatch --exclusive``.

    argparse treats a value starting with ``-`` as the next option, so the one
    flag whose entire job is passing *other* flags through fails on its most
    natural invocation — with argparse's generic "expected one argument", which
    names neither the cause nor the fix. The ``=`` form works.

    Diagnosed rather than silently repaired: rewriting the pair into the ``=``
    form would make ``slurmate --custom-sbatch --print`` swallow a real slurmate
    flag as an sbatch one, which is a silent wrong answer in place of a loud
    error — the failure mode this whole package is being audited for.
    """
    for i, token in enumerate(argv[:-1]):
        if token != "--custom-sbatch":
            continue
        nxt = argv[i + 1]
        # argparse only mistakes a value for an option when it starts with "-"
        # AND contains no space: `-C bigmem` and `--comment="my run"` are already
        # accepted (verified against argparse), and rejecting those would break
        # the multi-flag form the portability report exercised.
        if nxt.startswith("-") and len(nxt) > 1 and " " not in nxt:
            print(
                f"  {c.RED}\u2717 Error: --custom-sbatch {nxt}: a value starting "
                f"with '-' must use the '=' form{c.RESET}\n"
                f"  {c.GRAY}Use: --custom-sbatch='{nxt}'{c.RESET}\n"
                f"  {c.GRAY}(argparse would otherwise read '{nxt}' as slurmate's "
                f"own option, not as a value.){c.RESET}",
                file=sys.stderr,
            )
            sys.exit(2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from . import __version__
    _check_custom_sbatch_form(list(argv) if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(description="Slurmate \u2014 sbatch wizard")
    parser.add_argument("--job-name", "-J", default=None, help="Job name")
    parser.add_argument("--account", "-A", default=None, help="Slurm account")
    parser.add_argument("--partition", "-p", default=None, help="Target partition")
    parser.add_argument("--qos", "-q", default=None, help="QoS")
    # Slurm's own spellings are accepted alongside slurmate's. SM-25: the
    # documented handoff is "slurmpast tells you what to request -> slurmate
    # builds the script", and it broke on exactly the two flags carrying the
    # sizing, because slurmate *emitted* --cpus-per-task/--mem and accepted
    # neither. Declaring --mem explicitly also removes the ambiguity with
    # --mem-per-cpu, which argparse reported by naming two flags the user had
    # not typed and suggesting neither.
    parser.add_argument("--cpus", "--cpus-per-task", "-c", type=int, default=None,
                        help="CPU cores (Slurm: --cpus-per-task)")
    parser.add_argument("--memory", "--mem", default=None,
                        help="Memory per node, Slurm's --mem (e.g. 16G, 64000M; "
                             "empty or 'none' omits --mem for whole-node sites)")
    parser.add_argument("--mem-per-cpu", default=None,
                        help="Memory per CPU (e.g. 2G); takes precedence over --memory")
    parser.add_argument("--time", "-t", default=None, help="Time limit")
    parser.add_argument("--nodes", "-N", type=int, default=None, help="Node count")
    parser.add_argument("--ntasks-per-node", type=int, default=None, help="Tasks per node")
    # Not type=int: Slurm's own -G/--gpus takes "[<type>:]<count>", and slurmate
    # PRINTS "--gpus=a100:2" under --gpu-format gpus, so int-only was the last
    # emitted directive argparse still rejected outright ("invalid int value").
    # Resolved to a count (+ type, + format) by _resolve_gpu_spellings.
    parser.add_argument("--gpus", "-G", default=None,
                        help="Number of GPUs, or '<type>:<count>'")
    parser.add_argument("--gpu-type", default=None, help="GPU type (e.g. a100, h100)")
    # The three GPU spellings a generated script (or a colleague's) carries. Each
    # is a different *format* of the same request, so they resolve to --gpus /
    # --gpu-type / --gpu-format rather than to a dest of their own.
    # Accepted only to explain itself: slurmate derives --error from the output
    # path rather than tracking it, so it is the one flag a generated script
    # carries that cannot round-trip. argparse's "unrecognized arguments: --error"
    # was technically true and useless. Hidden from --help, since the answer is a
    # redirect rather than an option.
    parser.add_argument("--error", "-e", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gres", default=None,
                        help="Slurm --gres, GPU only (e.g. gpu:2, gpu:a100:2)")
    parser.add_argument("--gpus-per-node", default=None,
                        help="Slurm --gpus-per-node (e.g. 2, a100:2)")
    parser.add_argument("--gpus-per-task", default=None,
                        help="Slurm --gpus-per-task (e.g. 2, a100:2)")
    parser.add_argument("--gpu-format", default=None,
                        choices=["gres_type", "constraint", "gpus", "gpus_per_node", "gpus_per_task"],
                        help="GPU request format")
    parser.add_argument("--constraint", "-C", default=None,
                        help="Node feature constraint / Slurm -C (e.g. 'gpu', 'cpu', 'a100')")
    parser.add_argument("--array", "-a", default=None, help="Array specification (e.g. 1-10)")
    parser.add_argument("--modules", default=None, help="Comma-separated modules")
    parser.add_argument("--env", default=None, help="Conda environment")
    parser.add_argument("--env-type", default=None, choices=["conda", "mamba", "venv", "none"],
                        help="Environment activation strategy (conda, mamba, venv, none)")
    parser.add_argument("--output-dir", default=None, help="Output directory for logs")
    parser.add_argument("--output-file", "--output", "-o", default=None,
                        help="Output log file name/pattern, Slurm's --output "
                             "(%%j = job ID); error derives .err")
    parser.add_argument("--command", default=None, help="Command to run")
    parser.add_argument("--custom-sbatch", default=None,
                        help="Extra #SBATCH flags, space- or comma-separated (e.g. "
                             "--exclusive,--reservation=abc). A value may use '=' or a "
                             "space (-C bigmem); quote one that contains a space "
                             "(--comment=\"my run\")")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data instead of this cluster "
                             "(sets SLURMATE_MOCK=1); output is marked SIMULATED")
    parser.add_argument("--ascii", action="store_true",
                        help="Plain ASCII markers instead of Unicode (also SLURMATE_ASCII=1); "
                             "implied when the terminal's encoding cannot carry them")
    parser.add_argument("--force", action="store_true",
                        help="Downgrade cluster checks (unknown partition/account) to "
                             "warnings — for writing a script to carry to another cluster")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation and submit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the full summary (script, limit warnings, CPU-hours/ETA, "
                             "missing-field reminders) without submitting")
    parser.add_argument("--print", action="store_true",
                        help="Print only the raw script to stdout and exit (nothing else)")
    parser.add_argument("--no-save-script", action="store_true",
                        help="Do not auto-save a <job>-<id>.sh copy on submit")
    parser.add_argument("--version", action="version", version=f"slurmate {__version__}")
    args = parser.parse_args(argv)
    _normalize_gpus_flag(args)
    return args


# Job-defining flags whose presence means the user wants non-interactive
# (batch) mode — not just --partition. Output modes (--print/--dry-run) and
# --no-save-script are deliberately excluded; --yes is handled separately.
_BATCH_FLAGS = (
    "job_name", "account", "partition", "qos", "cpus", "memory", "mem_per_cpu",
    "time", "nodes", "ntasks_per_node", "gpus", "gpu_type", "gpu_format",
    "constraint", "array", "modules", "env", "env_type", "output_dir",
    "output_file", "command", "custom_sbatch",
)


def _is_batch_mode(args: argparse.Namespace, config: dict[str, Any] | None = None) -> bool:
    """Enter batch mode when any job-defining flag (or --yes) is supplied.

    Previously only --partition switched modes, so flags like --cpus/--command
    were silently dropped into the interactive TUI. A config-supplied partition
    still satisfies the partition *requirement* once batch mode is active, but
    by itself doesn't force batch mode (bare `slurmate` stays interactive).

    ``--print``/``--dry-run`` are output modes, not job-defining flags, so on
    their own they stay interactive (a bare ``slurmate --print`` opens the
    wizard). But when a config file already supplies the job, they render from
    it non-interactively instead of launching the full-screen wizard into a pipe.
    """
    if any(getattr(args, f, None) is not None for f in _BATCH_FLAGS):
        return True
    if getattr(args, "yes", False):
        return True
    if (getattr(args, "print", False) or getattr(args, "dry_run", False)) and config:
        return True
    return False


def _isatty(stream: Any) -> bool:
    """``stream.isatty()``, tolerating a detached / closed / replaced stream."""
    try:
        return bool(stream is not None and stream.isatty())
    except (AttributeError, ValueError):
        return False


def _require_terminal_for_wizard() -> None:
    """Refuse to open the full-screen wizard without a terminal on both ends.

    prompt_toolkit detects the problem, prints ``Warning: Input is not a terminal
    (fd=0)``, renders the wizard anyway and then blocks forever on input that
    cannot arrive — so ``slurmate | tee setup.log``, a CI runner, or any wrapper
    script that inherits a pipe hangs until something kills it. Piping is the most
    ordinary thing a user can do to a command, and the non-interactive surface
    already exists and works; this points at it instead of hanging.
    """
    missing = [
        name for name, stream in (("stdin", sys.stdin), ("stdout", sys.stdout))
        if not _isatty(stream)
    ]
    if not missing:
        return
    verb = "is" if len(missing) == 1 else "are"
    print(
        f"  {c.RED}{g.ERR} slurmate: {' and '.join(missing)} {verb} not a terminal — the "
        f"interactive wizard needs one.{c.RESET}\n"
        f"  {c.GRAY}Pass the job as flags for non-interactive use, e.g.{c.RESET}\n"
        f"  {c.GRAY}  slurmate --print --partition <name> --cpus 2 "
        f"--time 01:00:00 --command 'echo hi'{c.RESET}\n"
        f"  {c.GRAY}(--print writes the script to stdout; --dry-run adds the "
        f"summary and checks; --yes submits.){c.RESET}",
        file=sys.stderr,
    )
    sys.exit(1)


def _redirect_error_flag(args: argparse.Namespace) -> None:
    """Explain ``--error`` instead of rejecting it as unknown.

    SM-25's rule is that anything slurmate prints should be typeable back at it.
    ``--error`` is the single exception, because it is *derived* from the output
    path — so the useful response names the escape hatch, which is verified to
    work: a custom directive suppresses the auto one rather than duplicating it.
    """
    raw = getattr(args, "error", None)
    if raw is None:
        return
    print(
        f"  {c.RED}{g.ERR} slurmate derives --error from the output path, so there "
        f"is no --error option. Pass it through as a custom directive instead: "
        f"--custom-sbatch=--error={raw}{c.RESET}",
        file=sys.stderr,
    )
    sys.exit(2)


def _normalize_gpus_flag(args: argparse.Namespace) -> None:
    """Turn ``--gpus`` into an int, splitting off a ``<type>:`` prefix.

    Called from :func:`parse_args` so that ``args.gpus`` is an int (or None) for
    every caller — the flag has to accept Slurm's ``<type>:count`` spelling,
    which argparse's ``type=int`` rejected, but "after parsing, gpus is a number"
    is an invariant the rest of the module and its tests rely on. Idempotent, so
    :func:`_resolve_gpu_spellings` can re-run it without caring who came first.
    """
    raw = getattr(args, "gpus", None)
    if raw is None:
        return
    if str(raw).strip() == "":
        args.gpus = None
        return
    try:
        count, gpu_type = parse_gpu_spelling(GPU_COUNT_FLAG, str(raw))
    except ValueError as e:
        print(f"  {c.RED}{g.ERR} {e}{c.RESET}", file=sys.stderr)
        sys.exit(2)
    args.gpus = count
    if gpu_type:
        if not getattr(args, "gpu_type", None):
            args.gpu_type = gpu_type
        # Only a *typed* --gpus implies the format: that rendering is the only one
        # that produces "--gpus=a100:2", so typing it back reproduces the script
        # it came from. A bare "--gpus 4" says nothing about format and must keep
        # the default. Flagged as inferred, because a format nobody chose must
        # yield to a cluster that cannot parse it rather than block the job.
        if not getattr(args, "gpu_format", None):
            args.gpu_format = "gpus"
            args.gpu_format_inferred = True


def _resolve_gpu_spellings(args: argparse.Namespace) -> None:
    """Fold Slurm's --gres/--gpus-per-* into --gpus/--gpu-type/--gpu-format.

    Mutates ``args`` in place, before anything reads it. A disagreement with an
    explicit ``--gpus`` is an error rather than a precedence rule: these are two
    spellings of one number, so a conflict is a mistake, and picking a winner
    silently would submit a request the user did not make.

    ``--gpus`` itself is normalised first, because it also takes Slurm's
    ``<type>:count`` and everything below compares against it as an int.
    """
    _normalize_gpus_flag(args)
    for flag, fmt in GPU_SPELLING_FORMATS.items():
        raw = getattr(args, flag.lstrip("-").replace("-", "_"), None)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            count, gpu_type = parse_gpu_spelling(flag, str(raw))
        except ValueError as e:
            print(f"  {c.RED}{g.ERR} {e}{c.RESET}", file=sys.stderr)
            sys.exit(2)
        if args.gpus is not None and args.gpus != count:
            print(f"  {c.RED}{g.ERR} {flag} says {count} GPU(s) and --gpus says "
                  f"{args.gpus}; they are two spellings of one number{c.RESET}",
                  file=sys.stderr)
            sys.exit(2)
        args.gpus = count
        if gpu_type and not args.gpu_type:
            args.gpu_type = gpu_type
        if not args.gpu_format:
            args.gpu_format = fmt


def main() -> None:
    # Before any output: a non-encodable character must not be able to abort the
    # run. A *valid* non-UTF-8 locale (en_US is latin-1; el7 has no C.UTF-8) made
    # a "⚠" in a warning raise UnicodeEncodeError mid-print, killing the run and
    # truncating the summary — and the warnings are error paths, so the tool was
    # least robust exactly when something had already gone wrong.
    make_output_safe()
    args = parse_args()
    # Before --demo and before run_batch: these are aliases, so everything
    # downstream must see the resolved --gpus/--gpu-type/--gpu-format.
    _resolve_gpu_spellings(args)
    _redirect_error_flag(args)
    # Before anything reads it: --demo is the discoverable spelling of an
    # environment variable that was documented nowhere, so the deliberate path
    # and the accidental one now produce the same, marked, output.
    if getattr(args, "demo", False):
        os.environ["SLURMATE_MOCK"] = "1"
    if getattr(args, "ascii", False):
        set_ascii(True)
    console = Console()
    config = load_config()
    batch = _is_batch_mode(args, config)
    save_script = not args.no_save_script

    # Before anything is printed: the wizard is the only path that needs a tty,
    # and batch mode must stay fully usable in a pipe.
    if not batch:
        _require_terminal_for_wizard()

    if not (args.print or args.dry_run):
        print_banner(interactive=not batch)

    answers_opt: dict[str, Any] | None = None
    wizard: Wizard | None = None
    if batch:
        # Keep --print's stdout to just the raw script; the mode banner is noise.
        if not (args.print or args.dry_run):
            print(f"  {c.CYAN}{g.BULLET}{c.RESET} {c.GRAY}Running in batch mode{c.RESET}\n")
        answers_opt = run_batch(args, console, config)
    else:
        wizard = Wizard()
        answers_opt = wizard.run()

    if not answers_opt:
        if not (args.print or args.dry_run):
            print(f"  {c.YELLOW}Cancelled.{c.RESET}")
        else:
            sys.exit(1)
        return
    answers: dict[str, Any] = answers_opt

    # --print: emit only the raw script, nothing else (clean for pipes/CI).
    if args.print:
        script = build_from_answers(answers)
        # --print returned before any of the capacity checks ran, so the mode most
        # used in scripts and CI was the one that emitted an unschedulable script
        # in silence: a 999-CPU request that --dry-run warns about twice produced
        # zero bytes on stderr. The checks go to **stderr**, so stdout stays
        # script-only and redirecting it still yields exactly the script.
        err = Console(stderr=True)
        _validate_partition_limits(answers, err, max_array=None)
        # The shared site checks too, so anything added there reaches every mode.
        # Wiring only the two limit reporters here is how a new check ended up
        # covering --dry-run and --yes but not --print.
        fatal = False
        for _level, _msg in site_check_issues(answers):
            head = str(_msg).splitlines()[0]
            marker, colour = (
                (g.ERR, "red") if _level == "error" else (g.WARN, "yellow")
            )
            err.print(f"  [{colour}]{marker} {escape(head)}[/]")
            fatal = fatal or _level == "error"
        # will_create=False: --print is the only mode that hands over a script
        # slurmate will never submit, so it is the only one where a missing log
        # directory stays missing. --dry-run is not that case — a later real run
        # creates it — and warning there would fire on every dry run of the
        # default `logs/`.
        _warn_runtime_targets(script, answers, err, will_create=False)
        # And Slurm's own verdict on the script. Every other mode gets this free
        # from the ETA (build_and_show), which --print does not call, so the mode
        # meant for pipes and CI was the one that could not learn a job was
        # unsubmittable: on Booth's Mercury an account-less script — which that
        # controller refuses outright — printed with zero bytes on stderr and
        # rc=0. Script-based, so it needs no answers, and it submits nothing.
        refusal = check_script_with_scheduler(script)
        if refusal and refusal_is_permanent(refusal):
            err.print(f"  [red]{g.ERR} Slurm refuses this job: {escape(refusal)}[/]")
            fatal = True
        elif refusal and refusal_is_transient(refusal):
            # Reported, not fatal: a transient limit says nothing about the
            # script, and failing here would turn "you already have a job
            # queued" into a red CI build.
            err.print(
                f"  [yellow]{g.WARN} Slurm cannot take this job right now: "
                f"{escape(refusal)}[/] [dim](the script itself is valid)[/]"
            )
        elif refusal:
            # Unrecognised wording: still not fatal (guessing "permanent" fails
            # builds over conditions that clear), but it must not carry the
            # reassurance either.
            err.print(
                f"  [yellow]{g.WARN} Slurm would not accept this job: "
                f"{escape(refusal)}[/] [dim](slurmate cannot tell whether this "
                f"clears on its own)[/]"
            )
        # Reporting an error and then handing over the artifact anyway is the
        # inverse of the silence problem: the tool states the script is wrong and
        # emits it regardless. --force still overrides, as it does for the
        # partition/account checks, since writing a script for another cluster is
        # legitimate.
        if fatal and not getattr(args, "force", False):
            err.print(
                "  [dim]Pass --force to print it anyway "
                "(e.g. for another cluster).[/]"
            )
            sys.exit(1)
        print(script)
        return

    # build_and_show prints the summary panel, partition-limit warnings, CPU-hours/ETA,
    # and missing-field reminders. --dry-run stops here without submitting.
    script, queue_info = build_and_show(answers, console)

    if args.dry_run:
        print(f"  {c.GRAY}Dry run — not submitted.{c.RESET}")
        return

    if args.yes:
        # Unattended submit: a blank / whitespace-only / comment-only command
        # would submit a no-op job (the builder rstrips the body to nothing), so
        # make it a hard error here rather than only an advisory warning. Strip
        # each line and treat a command with no real (non-comment) line as
        # missing. (Partition and job name stay advisory — sbatch defaults them.)
        cmd_lines = [ln.strip() for ln in str(answers.get("command") or "").splitlines()]
        if all(not ln or ln.startswith("#") for ln in cmd_lines):
            print(f"  {c.RED}{g.ERR} Nothing to run — refusing to submit with --yes "
                  f"(pass --command){c.RESET}", file=sys.stderr)
            sys.exit(1)
        # Don't fire off a job Slurm will certainly reject (e.g. GPUs on a CPU-only
        # partition). Errors are hard rejections; warnings stay advisory (a
        # heterogeneous partition can under-report, so they aren't guaranteed fails).
        errs = _hard_errors(answers)
        # Slurm's own refusal blocks the submit too — it is more authoritative
        # than any answers-derived check, and this gate exists precisely to save
        # the round-trip. Two narrowings, both load-bearing: only the
        # *scheduler's* verdict blocks (a refusal derived from advertised
        # partition figures stays advisory, because a heterogeneous partition
        # under-reports — see capacity_refusal), and only a *permanent* one (a
        # submit-count cap means wait, not fix, and blocking on it refused a
        # valid job on Mercury purely because another job was already queued).
        # The message was already printed by build_and_show, so this only
        # decides — repeating it here would say the same thing twice.
        refused = (
            not queue_info.get("feasible", True)
            and queue_info.get("source") == "scheduler"
            and bool(queue_info.get("refusal_is_permanent", True))
        )
        if errs or refused:
            for m in errs:
                print(f"  {c.RED}{g.ERR} {m}{c.RESET}", file=sys.stderr)
            print(f"  {c.RED}{g.ERR} Refusing to submit — Slurm would reject this job "
                  f"(fix the above and pass corrected flags){c.RESET}", file=sys.stderr)
            sys.exit(1)
        _submit_and_report(script, answers, console, save_script=save_script)
        return

    import questionary

    from .theme import questionary_style
    QS = questionary_style()
    # Label the menu with the command that will actually be launched — the raw
    # env lookup disagreed with _editor_command() whenever EDITOR was set but
    # empty (menu said "Open script in ", vim was launched).
    editor = " ".join(_editor_command())
    default_name = f"{answers.get('job_name', '') or 'slurm'}.sh"

    def _resummarize() -> None:
        _show_script_and_summary(console, script, answers, estimate_su(
            answers.get("cpus", 1), answers.get("time_limit", "02:00:00"),
            answers.get("nodes", 1), answers.get("ntasks_per_node"),
            array_task_count(str(answers.get("array_spec") or "")),
            custom_ntasks(answers.get("custom_sbatch")),
        ), queue_info)

    # A navigable action menu instead of a one-way confirm chain: every action
    # returns here. Esc (or the explicit option) re-opens the wizard to edit
    # answers; Ctrl-C/Quit cancels cleanly.
    can_edit = wizard is not None
    manually_edited = False  # set once the user hand-edits the script in $EDITOR
    while True:
        choices = ["Submit to Slurm"]
        if can_edit:
            choices.append("Go back to edit answers")
        choices += [f"Open script in {editor}", "Save script to a file",
                    "Quit without submitting"]

        q = questionary.select(
            "What would you like to do?", choices=choices, qmark="", style=QS,
            instruction="(Esc to go back)" if can_edit else None,
        )
        kb = q.application.key_bindings
        if can_edit and isinstance(kb, KeyBindings):
            @kb.add("escape", eager=True)
            def _back(event: Any) -> None:
                event.app.exit(result=_GO_BACK)
        action = q.ask()

        if action == _GO_BACK or (action is not None and action.startswith("Go back")):
            assert wizard is not None
            # Editing answers regenerates the script from scratch, discarding any
            # manual $EDITOR changes — confirm before throwing them away.
            if manually_edited and not questionary.confirm(
                "Editing answers regenerates the script and discards your manual "
                "edits. Continue?", default=False, qmark="", style=QS,
            ).ask():
                continue
            answers = wizard.edit()
            manually_edited = False
            default_name = f"{answers.get('job_name', '') or 'slurm'}.sh"
            script, queue_info = build_and_show(answers, console)
            continue
        if action is None or action.startswith("Quit"):
            print(f"  {c.YELLOW}Not submitted.{c.RESET}")
            return
        if action.startswith("Submit"):
            # Navigation stays free (the error shows on every step), but block the
            # actual submit — otherwise slurmate fires off a script sbatch rejects,
            # wasting a round-trip. The fix is usually an earlier step (partition).
            if manually_edited:
                # The answers no longer describe the script, so validating them
                # would check something other than what is about to be submitted
                # — and would block a hand edit that *fixed* the problem while
                # passing one that introduced it. Ask the controller about the
                # actual bytes instead.
                refusal = check_script_with_scheduler(script)
                if refusal and refusal_is_permanent(refusal):
                    console.print(f"  [red]{g.ERR} Slurm rejects the edited "
                                  f"script: {escape(refusal)}[/]")
                    console.print("  [dim]Choose \"Open in editor\" to fix it, "
                                  "or \"Go back to edit answers\" to regenerate.[/]")
                    continue
                if refusal:
                    # Not permanent, so the edit is not demonstrably at fault:
                    # blocking would strand a correct script behind a condition
                    # that clears, and "rejects the edited script" would send the
                    # user hunting for a mistake in it. Submit and let the
                    # controller decide at the moment it matters — the same call
                    # the other paths make. The wording still distinguishes a
                    # known-transient limit from one we cannot classify.
                    detail = (
                        "the script is valid; submitting anyway"
                        if refusal_is_transient(refusal)
                        else "slurmate cannot tell whether this clears on its "
                             "own; submitting anyway"
                    )
                    console.print(
                        f"  [yellow]{g.WARN} Slurm would not take this job right "
                        f"now: {escape(refusal)}[/] [dim]({detail})[/]"
                    )
            else:
                errs = _hard_errors(answers)
                if errs:
                    for m in errs:
                        console.print(f"  [red]{g.ERR} {escape(m)}[/]")
                    console.print("  [red]This job has errors Slurm will reject.[/] "
                                  "[dim]Choose \"Go back to edit answers\" to fix, or Quit.[/]")
                    continue
            _submit_and_report(script, answers, console, save_script=save_script)
            return
        if action.startswith("Open"):
            script = _edit_script_in_editor(script)
            manually_edited = True
            _resummarize()
            # Say once that the summary above is now describing the answers, not
            # the edited script — the two can disagree, and the checks that
            # produced that summary no longer apply to what would be submitted.
            console.print(
                f"  [yellow]{g.WARN} Script edited by hand: the summary and checks "
                f"above describe the generated script, not your edits. Slurm's own "
                f"verdict on the edited script is checked at submit.[/]"
            )
        elif action.startswith("Save"):
            _save_script(script, default_name)


if __name__ == "__main__":
    main()
