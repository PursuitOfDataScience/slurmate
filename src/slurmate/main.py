from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from typing import Any

from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .builder import build_from_answers, estimate_su, job_summary_rows, sanitize_job_name
from .system_utils import (
    _parse_mem_to_mb,
    _parse_slurm_time_to_minutes,
    fetch_gpu_types_for_partition,
    fetch_partitions,
    fetch_queue_eta,
    load_config,
    normalize_memory,
    submit_sbatch,
    validate_memory,
    validate_time,
)
from .theme import c, print_banner
from .tui import Wizard, _parse_custom_flags

# Sentinel returned by the action menu when the user presses Esc to go back.
_GO_BACK = "\x00__go_back__"

# ── Batch mode helpers ───────────────────────────────────────────────────

def _get_partition(partitions: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for p in partitions:
        if p["name"] == name:
            return p
    return {"name": name, "nodes": 0, "cpus_per_node": 0, "mem_per_node_mb": 0,
            "gpu_types": [], "timelimit": None, "is_public": True}


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
    try:
        return int(value)
    except (TypeError, ValueError):
        if field and err_console is not None:
            err_console.print(
                f"  {c.YELLOW}⚠ {field} value {value!r} is not an integer; "
                f"using {default}{c.RESET}"
            )
        return default


def run_batch(args: argparse.Namespace, console: Console, config: dict[str, Any]) -> dict[str, Any]:
    err_console = Console(stderr=True)

    # Get values fallback from config
    args_partition = getattr(args, "partition", None)
    partition = args_partition if args_partition is not None else config.get("partition", "")

    args_cpus = getattr(args, "cpus", None)
    cpus = _coerce_int(args_cpus if args_cpus is not None else config.get("cpus", 4), 4,
                       field="cpus", err_console=err_console)

    args_memory = getattr(args, "memory", None)
    memory_val = args_memory if args_memory is not None else config.get("memory", "16G")

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
    ntasks_per_node = (
        _coerce_int(raw_ntasks, 0, field="ntasks_per_node", err_console=err_console)
        if raw_ntasks is not None else None
    )

    args_gpu_type = getattr(args, "gpu_type", None)
    gpu_type = args_gpu_type if args_gpu_type is not None else config.get("gpu_type")
    if gpu_type is not None:
        gpu_type = str(gpu_type)

    args_gpu_format = getattr(args, "gpu_format", None)
    gpu_format = args_gpu_format if args_gpu_format is not None else config.get("gpu_format")

    args_output_dir = getattr(args, "output_dir", None)
    output_dir = args_output_dir if args_output_dir is not None else config.get("output_dir", "logs")

    args_output_file = getattr(args, "output_file", None)
    output_file = args_output_file if args_output_file is not None else config.get("output_file")

    # Seed the GPU format from SLURMATE_GPU_FORMAT (default gres_type) so the
    # env var documented in the README actually takes effect in batch mode.
    if gpus > 0 and not gpu_format:
        gpu_format = os.environ.get("SLURMATE_GPU_FORMAT", "gres_type").lower()

    # Validate the resolved GPU format from config/env (the --gpu-format flag is
    # already constrained by argparse choices, but config/env values are not):
    # clamp an unrecognized value to gres_type instead of silently falling
    # through to the constraint-style directives, matching the TUI's behavior.
    _GPU_FORMATS = ("gres_type", "constraint", "gpus")
    if gpu_format is not None:
        gpu_format = str(gpu_format).lower()
        if gpu_format not in _GPU_FORMATS:
            err_console.print(
                f"  {c.YELLOW}⚠ Unknown gpu_format {gpu_format!r}; "
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

    # Hard-validate memory
    if not validate_memory(str(memory_val)):
        err_console.print(f"  {c.RED}\u2717 Error: Invalid memory value: {memory_val}{c.RESET}")
        sys.exit(1)

    # Hard-validate time limit
    if not validate_time(str(time_val)):
        err_console.print(f"  {c.RED}\u2717 Error: Invalid time limit value: {time_val}{c.RESET}")
        sys.exit(1)

    all_parts = fetch_partitions()
    part_obj = _get_partition(all_parts, partition)

    raw_modules = getattr(args, "modules", None)
    if raw_modules is None:
        cfg_mods = config.get("modules")
        if isinstance(cfg_mods, list):
            mods = cfg_mods
        elif isinstance(cfg_mods, str):
            mods = [m.strip() for m in cfg_mods.split(",") if m.strip()]
        else:
            mods = None
    else:
        mods = [m.strip() for m in raw_modules.split(",") if m.strip()]

    args_env_type = getattr(args, "env_type", None)
    env_type = args_env_type if args_env_type is not None else config.get("env_type")

    args_env = getattr(args, "env", None)
    env_name = args_env if args_env is not None else config.get("env_name")
    if env_name and not env_type:
        env_type = "conda"

    custom_sbatch_val = getattr(args, "custom_sbatch", None)
    if custom_sbatch_val is None:
        cfg_custom = config.get("custom_sbatch")
        if isinstance(cfg_custom, list):
            custom_sbatch_list = cfg_custom
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
    return {
        "job_name": sanitize_job_name(str(raw_job_name)),
        "account": args_account if args_account is not None else config.get("account"),
        "partition": partition,
        "_partition_obj": part_obj,
        "qos": args_qos if args_qos is not None else config.get("qos"),
        "cpus": cpus,
        "memory": normalize_memory(str(memory_val)),
        "time_limit": str(time_val),
        "nodes": nodes,
        "ntasks_per_node": ntasks_per_node,
        "gpus": gpus,
        "gpu_type": gpu_type or None,
        "gpu_format": gpu_format or None,
        "array_spec": args_array if args_array is not None else config.get("array_spec"),
        "modules": mods,
        "env_type": env_type,
        "env_name": env_name,
        "output_dir": output_dir,
        "output_file": output_file or None,
        "command": args_command if args_command is not None else config.get("command", ""),
        "custom_sbatch": custom_sbatch_list,
    }


def _validate_partition_limits(answers: dict[str, Any], console: Console) -> None:
    part = answers.get("_partition_obj")
    if not part:
        return

    # Check CPUs \u2014 compare the per-node total (ntasks-per-node \u00d7 cpus-per-task)
    # against the node's core count, so multi-task over-allocation is caught.
    cpus = answers.get("cpus")
    if cpus is not None:
        try:
            cores = int(cpus)
            ntpn_raw = answers.get("ntasks_per_node")
            ntpn = int(ntpn_raw) if ntpn_raw else 1
            total = cores * max(1, ntpn)
            limit = part.get("cpus_per_node", 0)
            if limit and total > limit:
                detail = f"{ntpn}\u00d7{cores}={total}" if ntpn > 1 else str(total)
                console.print(f"  [yellow]\u26a0 Warning: CPUs ({detail}) exceeds partition limit ({limit} per node)[/]")
        except (ValueError, TypeError):
            pass

    # Check Memory
    memory = answers.get("memory")
    if memory:
        if validate_memory(str(memory)):
            mb = _parse_mem_to_mb(str(memory))
            limit = part.get("mem_per_node_mb", 0)
            if limit and mb > limit:
                console.print(f"  [yellow]\u26a0 Warning: Memory ({memory}) exceeds partition limit ({limit} MB per node)[/]")

    # Check Time Limit
    time_limit = answers.get("time_limit")
    if time_limit:
        try:
            req_mins = _parse_slurm_time_to_minutes(str(time_limit))
            limit_str = part.get("timelimit")
            if limit_str:
                limit_mins = _parse_slurm_time_to_minutes(limit_str)
                if limit_mins > 0 and req_mins > limit_mins:
                    console.print(f"  [yellow]\u26a0 Warning: Time limit ({time_limit}) exceeds partition limit ({limit_str})[/]")
        except Exception:
            pass

    # Check GPUs
    gpus = answers.get("gpus", 0)
    try:
        gpus_val = int(gpus) if gpus is not None else 0
    except ValueError:
        gpus_val = 0

    gpu_types = part.get("gpu_types", [])
    # `has_gpu` is set whenever the partition advertises any GRES gpu \u2014 including
    # count-only ("gpu:4") and typed-without-count ("gpu:a100") forms that don't
    # populate gpu_types \u2014 so a real GPU partition isn't flagged as CPU-only.
    if gpus_val > 0 and not gpu_types and not part.get("has_gpu"):
        console.print(f"  [yellow]\u26a0 Warning: Partition '{part.get('name')}' does not support GPUs[/]")

    gpu_type = answers.get("gpu_type")
    if gpu_type and gpu_type.lower() != "any" and gpu_type.lower() not in {g.lower() for g in gpu_types}:
        part_name = part.get("name", "")
        if part_name:
            dyn_types = fetch_gpu_types_for_partition(part_name)
            all_types = gpu_types + [t for t in dyn_types if t not in gpu_types]
        else:
            all_types = gpu_types
        if gpu_type.lower() not in {g.lower() for g in all_types}:
            console.print(f"  [yellow]\u26a0 Warning: GPU type '{gpu_type}' not in partition list ({', '.join(all_types)})[/]")


_REQUIRED_FIELDS = [("job_name", "Job name"), ("partition", "Partition"), ("command", "Command to run")]


def _warn_missing_required(answers: dict[str, Any], console: Console) -> list[str]:
    """Print a reminder for any required field left blank; return the labels."""
    missing = [label for key, label in _REQUIRED_FIELDS if not answers.get(key)]
    if missing:
        console.print(
            f"  [yellow]⚠ Missing recommended fields:[/] {', '.join(missing)}"
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
    )

    queue_info = fetch_queue_eta(
        answers.get("partition", ""),
        req_nodes=answers.get("nodes", 1),
    )

    _validate_partition_limits(answers, console)
    _show_script_and_summary(console, script, answers, su_estimate, queue_info)
    _warn_missing_required(answers, console)
    return script, queue_info


def _show_script_and_summary(console: Console, script: str, answers: dict[str, Any],
                              su_estimate: str, queue_info: dict[str, Any] | None = None) -> None:
    print()
    script_lines = script.split("\n")
    num_w = len(str(len(script_lines)))
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

    script_w = max(num_w + 1 + len(ln) for ln in script_lines)
    title_text = "Generated sbatch script"
    script_panel = Panel(
        body,
        title=f"[bold #ff0080]{title_text}[/]",
        border_style="bright_magenta",
        width=script_w + 4,
        padding=(0, 1),
    )

    # Share the ordered field list with the in-TUI Review step (job_summary_rows)
    # so both summaries agree on what's shown; append the CLI-only SU/queue rows.
    rows: list[tuple[str, str, str]] = []
    for label, val in job_summary_rows(answers):
        style = "magenta" if label == "QoS" else "cyan"
        # Collapse a multi-line command to a single summary line (the full text
        # is still in the script panel) so the panel width stays correct.
        rows.append((f"{label}:", val.replace("\n", " \u21b5 "), style))
    rows.append(("Est. SU:", f"{su_estimate} SU", "#ffaa00"))
    if queue_info:
        rows.append(("Queue:", f"{queue_info['running']} run / {queue_info['pending']} wait", "white"))
        eta_color = "green" if queue_info["eta_seconds"] < 3600 else "#ffaa00"
        rows.append(("ETA:", str(queue_info["eta_label"]), eta_color))

    label_w = max(len(label) for label, _, _ in rows)
    summary_w = max(label_w + 2 + len(val) for label, val, _ in rows)
    summary = "\n".join(
        f"[bold bright_black]{label:<{label_w}}  [/][{style}]{val}[/]"
        for label, val, style in rows
    )

    s_title = "Summary"
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        tmp_path = f.name
    try:
        subprocess.run([*argv, tmp_path], check=False)
        with open(tmp_path) as f:
            return f.read()
    except OSError as e:
        # exec failure (editor not found / not executable) — check=False only
        # suppresses non-zero exit codes, not the exec error. Keep the current
        # script instead of crashing the whole wizard with a traceback.
        print(f"  {c.YELLOW}⚠ Could not open editor {' '.join(argv)!r}: {e}{c.RESET}")
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
        with open(path, "w") as f:
            f.write(script)
        print(f"  {c.GREEN}✓ Saved to {path}{c.RESET}")
    except OSError as e:
        print(f"  {c.RED}✗ Could not save: {e}{c.RESET}")


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
        with open(path, "w") as f:
            f.write(script)
        return path
    except OSError as e:
        print(f"  {c.YELLOW}⚠ Could not save script copy: {e}{c.RESET}")
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
        print(f"  {c.RED}✗ Submission failed (exit {retcode}){c.RESET}", file=sys.stderr)
        if stdout:
            print(f"  {c.GRAY}{stdout}{c.RESET}", file=sys.stderr)
        if stderr:
            print(f"  {c.RED}{stderr}{c.RESET}", file=sys.stderr)
        sys.exit(1)

    # An empty job ID with rc 0 means mock mode (sbatch unavailable) — say so
    # plainly instead of printing a blank ID and broken `squeue -j`/`scancel` hints.
    raw_out = stdout.strip()
    if not raw_out:
        print(f"  {c.YELLOW}(mock mode — not actually submitted){c.RESET}")
        if stderr:
            print(f"  {c.GRAY}{stderr}{c.RESET}")
        return

    # `sbatch --parsable` returns "jobid" or, on a federated/multi-cluster setup,
    # "jobid;cluster". Use just the numeric id for hints, the log path, and the
    # saved filename so none of them carry a stray ";cluster".
    job_id = raw_out.split(";")[0]

    print(f"  {c.GREEN}✓ Submitted!{c.RESET} Job ID: {c.CYAN}{job_id}{c.RESET}")

    # Save a copy of the exact submitted script for reproducibility — into
    # SLURMATE_LOG_DIR when set, else the CWD — and only report success when the
    # write actually happened. Skippable via --no-save-script / SLURMATE_NO_SAVE=1.
    if not _no_save_requested(save_script):
        log_dir = os.environ.get("SLURMATE_LOG_DIR")
        saved = _save_submitted_script(script, job_name, job_id, directory=log_dir)
        if saved:
            print(f"  {c.GRAY}Script saved: {saved}{c.RESET}")

    # Read the actual --output path from the generated script (source of truth).
    log_path = f"{answers.get('job_name', '') or 'slurm'}-%j.out"
    for line in script.splitlines():
        if line.startswith("#SBATCH --output="):
            log_path = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    # Resolve the job-id patterns we can (%j and %A → this job id); leave the
    # per-task %a literal for array jobs since there's no single task to point at.
    resolved_log = log_path.replace("%A", job_id).replace("%j", job_id)
    print(f"  {c.GRAY}Log path: {resolved_log}{c.RESET}")
    print(f"  {c.GRAY}Hints:{c.RESET}")
    print(f"    squeue -j {job_id}")
    print(f"    tail -f {resolved_log}")
    print(f"    scancel {job_id}")


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from . import __version__
    parser = argparse.ArgumentParser(description="Slurmate \u2014 sbatch wizard")
    parser.add_argument("--job-name", default=None, help="Job name")
    parser.add_argument("--account", default=None, help="Slurm account")
    parser.add_argument("--partition", default=None, help="Target partition")
    parser.add_argument("--qos", default=None, help="QoS")
    parser.add_argument("--cpus", type=int, default=None, help="CPU cores")
    parser.add_argument("--memory", default=None, help="Memory (e.g. 16G, 32G, 64000M)")
    parser.add_argument("--time", default=None, help="Time limit")
    parser.add_argument("--nodes", type=int, default=None, help="Node count")
    parser.add_argument("--ntasks-per-node", type=int, default=None, help="Tasks per node")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--gpu-type", default=None, help="GPU type (e.g. a100, h100)")
    parser.add_argument("--gpu-format", default=None, choices=["gres_type", "constraint", "gpus"],
                        help="GPU request format")
    parser.add_argument("--array", default=None, help="Array spec (e.g. 1-10)")
    parser.add_argument("--modules", default=None, help="Comma-separated modules")
    parser.add_argument("--env", default=None, help="Conda environment")
    parser.add_argument("--env-type", default=None, choices=["conda", "mamba", "venv", "none"],
                        help="Environment activation strategy (conda, mamba, venv, none)")
    parser.add_argument("--output-dir", default=None, help="Output directory for logs")
    parser.add_argument("--output-file", default=None,
                        help="Output log file name/pattern (%%j = job ID); error derives .err")
    parser.add_argument("--command", default=None, help="Command to run")
    parser.add_argument("--custom-sbatch", default=None,
                        help="Comma-separated extra #SBATCH flags (e.g. --exclusive,--reservation=abc)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation and submit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the full summary (script, limit warnings, SU/ETA, "
                             "missing-field reminders) without submitting")
    parser.add_argument("--print", action="store_true",
                        help="Print only the raw script to stdout and exit (nothing else)")
    parser.add_argument("--no-save-script", action="store_true",
                        help="Do not auto-save a <job>-<id>.sh copy on submit")
    parser.add_argument("--version", action="version", version=f"slurmate {__version__}")
    return parser.parse_args(argv)


# Job-defining flags whose presence means the user wants non-interactive
# (batch) mode — not just --partition. Output modes (--print/--dry-run) and
# --no-save-script are deliberately excluded; --yes is handled separately.
_BATCH_FLAGS = (
    "job_name", "account", "partition", "qos", "cpus", "memory", "time", "nodes",
    "ntasks_per_node", "gpus", "gpu_type", "gpu_format", "array", "modules", "env",
    "env_type", "output_dir", "output_file", "command", "custom_sbatch",
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


def main() -> None:
    console = Console()
    args = parse_args()
    config = load_config()
    batch = _is_batch_mode(args, config)
    save_script = not args.no_save_script

    if not (args.print or args.dry_run):
        print_banner(interactive=not batch)

    answers_opt: dict[str, Any] | None = None
    wizard: Wizard | None = None
    if batch:
        # Keep --print's stdout to just the raw script; the mode banner is noise.
        if not (args.print or args.dry_run):
            print(f"  {c.CYAN}▸{c.RESET} {c.GRAY}Running in batch mode{c.RESET}\n")
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
        print(build_from_answers(answers))
        return

    # build_and_show prints the summary panel, partition-limit warnings, SU/ETA,
    # and missing-field reminders. --dry-run stops here without submitting.
    script, queue_info = build_and_show(answers, console)

    if args.dry_run:
        print(f"  {c.GRAY}Dry run — not submitted.{c.RESET}")
        return

    if args.yes:
        # Unattended submit: a missing command would submit a no-op job, so make
        # it a hard error here rather than only an advisory warning. (Partition
        # and job name stay advisory — sbatch supplies sensible defaults.)
        if not answers.get("command"):
            print(f"  {c.RED}✗ Nothing to run — refusing to submit with --yes "
                  f"(pass --command){c.RESET}", file=sys.stderr)
            sys.exit(1)
        _submit_and_report(script, answers, console, save_script=save_script)
        return

    import questionary

    from .theme import questionary_style
    QS = questionary_style()
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))
    default_name = f"{answers.get('job_name', '') or 'slurm'}.sh"

    def _resummarize() -> None:
        _show_script_and_summary(console, script, answers, estimate_su(
            answers.get("cpus", 1), answers.get("time_limit", "02:00:00"),
            answers.get("nodes", 1), answers.get("ntasks_per_node"),
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
            _submit_and_report(script, answers, console, save_script=save_script)
            return
        if action.startswith("Open"):
            script = _edit_script_in_editor(script)
            manually_edited = True
            _resummarize()
        elif action.startswith("Save"):
            _save_script(script, default_name)


if __name__ == "__main__":
    main()
