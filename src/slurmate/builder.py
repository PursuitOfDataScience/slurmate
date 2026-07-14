from __future__ import annotations

import logging
import os
import re
import shlex
from typing import Any

from .system_utils import _parse_slurm_time_to_minutes

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
    """
    name = (name or "").strip()
    if not name:
        return name
    name = re.sub(r"\s+", "_", name)
    cleaned = re.sub(r"[^A-Za-z0-9._+-]", "", name)
    return cleaned or "slurm"


def _quote_sbatch_value(value: str) -> str:
    """Double-quote a #SBATCH value only if it contains whitespace.

    Slurm's directive parser splits on unquoted whitespace (so an output path
    like ``/scratch/My Group/log`` would bind only ``/scratch/My``). Slurm strips
    the surrounding quotes and preserves ``%j``/``%A``/``%a`` patterns literally,
    so quoting is safe; paths without spaces stay unquoted for readability.
    """
    if value and any(ch.isspace() for ch in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _gpus_int(answers: dict[str, Any]) -> int:
    g = answers.get("gpus", 0)
    try:
        return int(g) if g is not None else 0
    except (TypeError, ValueError):
        return 0


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

    add("Job name", answers.get("job_name"))
    add("Partition", answers.get("partition"))
    add("Account", answers.get("account"))
    qos = answers.get("qos")
    if qos and qos != "Default (none)":
        add("QoS", qos)
    add("CPUs", answers.get("cpus"))
    add("Memory", answers.get("memory"))
    add("Time limit", answers.get("time_limit"))
    nodes = answers.get("nodes")
    if nodes is not None and str(nodes) != "":
        add("Nodes", nodes)
    if answers.get("ntasks_per_node"):
        add("Tasks/node", answers.get("ntasks_per_node"))
    if _gpus_int(answers) > 0:
        add("GPUs", f"{answers.get('gpus')} × {answers.get('gpu_type') or 'any'}")
        add("GPU format", answers.get("gpu_format"))
    add("Array spec", answers.get("array_spec"))
    add("Output dir", answers.get("output_dir"))
    add("Output file", answers.get("output_file"))
    add("Modules", answers.get("modules"))
    add("Env", answers.get("env_name"))
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
    output_dir = answers.get("output_dir")
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
    output_file = answers.get("output_file")
    if output_file:
        output_file = os.path.expanduser(output_file)
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
        has_pattern = "%" in of
        if array_spec and not has_pattern:
            # An explicit output_file with no Slurm pattern would make every
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

    # One contiguous #SBATCH block, emitted in the same order the wizard asks
    # the questions, so the live preview grows top-to-bottom without reshuffling.
    # Omit an empty job-name/partition rather than emitting a malformed
    # `--job-name=` / `--partition=` (sbatch then auto-names / uses the default
    # partition), matching how account/qos are handled.
    if job_name:
        lines.append(f"#SBATCH --job-name={job_name}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if qos and qos != "Default (none)":
        lines.append(f"#SBATCH --qos={qos}")
    if cpus is not None:
        lines.append(f"#SBATCH --cpus-per-task={cpus}")
    if memory:
        lines.append(f"#SBATCH --mem={memory}")
    if time_limit:
        lines.append(f"#SBATCH --time={time_limit}")
    if nodes is not None:
        lines.append(f"#SBATCH --nodes={nodes}")
    if ntasks_per_node is not None:
        lines.append(f"#SBATCH --ntasks-per-node={ntasks_per_node}")
    elif nodes is not None and nodes > 1:
        lines.append("#SBATCH --ntasks-per-node=1")

    gpu_fmt = (gpu_format or os.environ.get("SLURMATE_GPU_FORMAT", "gres_type")).lower()
    gpu_any = gpu_type is not None and gpu_type.lower() == "any"
    if gpus > 0:
        if gpu_fmt == "gres_type" and gpu_type and not gpu_any:
            lines.append(f"#SBATCH --gres=gpu:{gpu_type}:{gpus}")
        elif gpu_fmt == "gpus":
            if gpu_type and not gpu_any:
                lines.append(f"#SBATCH --gpus={gpu_type}:{gpus}")
            else:
                lines.append(f"#SBATCH --gpus={gpus}")
        else:  # "constraint"
            lines.append(f"#SBATCH --gres=gpu:{gpus}")
            if gpu_type and not gpu_any:
                lines.append(f"#SBATCH --constraint={gpu_type}")

    if array_spec:
        lines.append(f"#SBATCH --array={array_spec}")

    # Output/error are auto-derived. In a partial preview, only show them once
    # the user has actually configured an output dir/file (output_path is set).
    if not partial or output_path:
        prefix = job_name if job_name else "slurm"
        tag = "%A_%a" if array_spec else "%j"
        out = output_path or f"{prefix}-{tag}.out"
        err = error_path or f"{prefix}-{tag}.err"
        lines.append(f"#SBATCH --output={_quote_sbatch_value(out)}")
        lines.append(f"#SBATCH --error={_quote_sbatch_value(err)}")

    if custom_sbatch:
        # Defensive: a bare string here would be iterated character-by-character
        # (#SBATCH m, #SBATCH i, …). Callers should pass a list; coerce just in
        # case by splitting on commas.
        if isinstance(custom_sbatch, str):
            from .tui import _parse_custom_flags
            custom_sbatch = _parse_custom_flags(custom_sbatch)
        for flag in custom_sbatch:
            # A newline inside a list entry would inject a second, non-#SBATCH
            # line into the script body; fold any newline into a space so the
            # entry stays a single directive.
            flag = str(flag).replace("\n", " ").replace("\r", " ").strip()
            if not flag:
                continue
            if gpus > 0:
                # Derive the flag name whether written with '=' or a space, and
                # only drop a custom flag that would *duplicate* the directive the
                # chosen gpu_format already emits (previously the space form
                # slipped through, and --gpus/--constraint were stripped even under
                # formats that don't emit them).
                name_val = re.split(r"[=\s]", flag, maxsplit=1)
                flag_name = name_val[0].strip()
                flag_val = name_val[1].strip() if len(name_val) > 1 else ""
                if gpu_fmt in ("gres_type", "constraint") and flag_name == "--gres" \
                        and flag_val.startswith("gpu"):
                    continue
                if gpu_fmt == "gpus" and flag_name == "--gpus":
                    continue
                if gpu_fmt == "constraint" and flag_name == "--constraint" \
                        and gpu_type and not gpu_any and flag_val == gpu_type:
                    continue
            lines.append(f"#SBATCH {flag}")

    if modules:
        lines.append("")
        for mod in modules:
            # Strip "(default)" annotation that the module system appends
            if mod.endswith("(default)"):
                mod = mod[:-9]
            lines.append(f"module load {mod}")

    if env_name:
        strategy = (env_type or "conda").lower()
        if strategy == "conda":
            lines.append("")
            lines.append(f"source activate {shlex.quote(env_name)}")
        elif strategy == "mamba":
            lines.append("")
            lines.append(f"mamba activate {shlex.quote(env_name)}")
        elif strategy in ("virtualenv (venv)", "venv"):
            lines.append("")
            lines.append(f"source {shlex.quote(env_name + '/bin/activate')}")
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


def estimate_su(cpus: int, time_limit: str, nodes: int = 1,
                ntasks_per_node: int | None = None) -> str:
    """Estimate Service Units (SU) cost for a job.

    Service Units are typically core-hours: CPUs-per-task × tasks-per-node ×
    nodes × hours. When ``ntasks_per_node`` is unset it defaults to 1 task.

    Args:
        cpus: Number of CPU cores per task.
        time_limit: Time limit string in Slurm format (e.g. "hh:mm:ss" or "d-hh:mm:ss").
        nodes: Number of nodes requested.
        ntasks_per_node: Tasks per node (multiplies the per-task core count).

    Returns:
        Formatted string representation of estimated SUs.
    """
    minutes = _parse_slurm_time_to_minutes(time_limit) if time_limit else 120.0
    if minutes <= 0:
        minutes = 120.0
    hours = minutes / 60.0
    tasks = ntasks_per_node if (ntasks_per_node and ntasks_per_node > 0) else 1
    su = cpus * tasks * hours * nodes
    if su < 1:
        return f"{su:.2f}"
    if su < 100:
        return f"{su:.1f}"
    return f"{su:,.0f}"
