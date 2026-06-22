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

from .builder import build_from_answers, estimate_su
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


def run_batch(args: argparse.Namespace, console: Console, config: dict[str, Any]) -> dict[str, Any]:
    print(f"  {c.CYAN}\u25b8{c.RESET} {c.GRAY}Running in batch mode{c.RESET}\n")

    # Get values fallback from config
    args_partition = getattr(args, "partition", None)
    partition = args_partition if args_partition is not None else config.get("partition", "")

    args_cpus = getattr(args, "cpus", None)
    cpus = args_cpus if args_cpus is not None else config.get("cpus", 4)

    args_memory = getattr(args, "memory", None)
    memory_val = args_memory if args_memory is not None else config.get("memory", "16G")

    args_time = getattr(args, "time", None)
    time_val = args_time if args_time is not None else config.get("time_limit", "02:00:00")

    args_nodes = getattr(args, "nodes", None)
    nodes = args_nodes if args_nodes is not None else config.get("nodes", 1)

    args_gpus = getattr(args, "gpus", None)
    gpus = args_gpus if args_gpus is not None else config.get("gpus", 0)

    args_ntasks_per_node = getattr(args, "ntasks_per_node", None)
    ntasks_per_node = args_ntasks_per_node if args_ntasks_per_node is not None else config.get("ntasks_per_node")

    args_gpu_type = getattr(args, "gpu_type", None)
    gpu_type = args_gpu_type if args_gpu_type is not None else config.get("gpu_type")

    args_gpu_format = getattr(args, "gpu_format", None)
    gpu_format = args_gpu_format if args_gpu_format is not None else config.get("gpu_format")

    args_output_dir = getattr(args, "output_dir", None)
    output_dir = args_output_dir if args_output_dir is not None else config.get("output_dir", "logs")

    args_output_file = getattr(args, "output_file", None)
    output_file = args_output_file if args_output_file is not None else config.get("output_file")

    if gpus > 0 and not gpu_format:
        gpu_format = "gres_type"

    # Hard-validate memory
    if not validate_memory(str(memory_val)):
        Console(stderr=True).print(f"  {c.RED}\u2717 Error: Invalid memory value: {memory_val}{c.RESET}")
        sys.exit(1)

    # Hard-validate time limit
    if not validate_time(str(time_val)):
        Console(stderr=True).print(f"  {c.RED}\u2717 Error: Invalid time limit value: {time_val}{c.RESET}")
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

    return {
        "job_name": args_job_name if args_job_name is not None else config.get("job_name", ""),
        "account": args_account if args_account is not None else config.get("account"),
        "partition": partition,
        "_partition_obj": part_obj,
        "qos": args_qos if args_qos is not None else config.get("qos"),
        "cpus": cpus,
        "memory": normalize_memory(str(memory_val)),
        "time_limit": time_val,
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

    # Check CPUs
    cpus = answers.get("cpus")
    if cpus is not None:
        try:
            cores = int(cpus)
            limit = part.get("cpus_per_node", 0)
            if limit and cores > limit:
                console.print(f"  [yellow]\u26a0 Warning: CPUs ({cores}) exceeds partition limit ({limit} per node)[/]")
        except ValueError:
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
    if gpus_val > 0 and not gpu_types:
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

    rows: list[tuple[str, str, str]] = [
        ("Job:", answers.get("job_name", "") or "", "cyan"),
        ("Partition:", answers.get("partition", "") or "", "cyan"),
    ]
    if answers.get("account"):
        rows.append(("Account:", answers["account"], "cyan"))
    if answers.get("qos") and answers["qos"] != "Default (none)":
        rows.append(("QoS:", answers["qos"], "magenta"))
    rows.append(("CPUs:", str(answers.get("cpus", "")), "cyan"))
    rows.append(("Memory:", answers.get("memory", "") or "", "cyan"))
    rows.append(("Time:", answers.get("time_limit", "") or "", "cyan"))
    rows.append(("Nodes:", str(answers.get("nodes", 1)), "cyan"))
    if answers.get("gpus", 0) > 0:
        gt = answers.get("gpu_type") or "any"
        rows.append(("GPUs:", f"{answers['gpus']} \u00d7 {gt}", "cyan"))
    if answers.get("array_spec"):
        rows.append(("Array:", str(answers["array_spec"]), "yellow"))
    if answers.get("modules"):
        rows.append(("Modules:", ", ".join(answers["modules"]), "cyan"))
    if answers.get("env_name"):
        rows.append(("Env:", answers["env_name"], "cyan"))
    if answers.get("custom_sbatch"):
        rows.append(("Custom flags:", ", ".join(answers["custom_sbatch"]), "cyan"))
    rows.append(("Est. SU:", f"{su_estimate} SU", "yellow"))
    if queue_info:
        rows.append(("Queue:", f"{queue_info['running']} run / {queue_info['pending']} wait", "white"))
        eta_color = "green" if queue_info["eta_seconds"] < 3600 else "yellow"
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


def _edit_script_in_editor(script: str) -> str:
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        tmp_path = f.name
    try:
        subprocess.run([editor, tmp_path], check=False)
        with open(tmp_path) as f:
            return f.read()
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


def _save_submitted_script(script: str, job_name: str, job_id: str) -> str | None:
    """Write the exact submitted script to the working dir for reproducibility."""
    safe = (job_name or "slurm").replace("/", "_").strip() or "slurm"
    path = os.path.join(os.getcwd(), f"{safe}-{job_id}.sh")
    try:
        with open(path, "w") as f:
            f.write(script)
        return path
    except OSError as e:
        print(f"  {c.YELLOW}⚠ Could not save script copy: {e}{c.RESET}")
        return None


def _submit_and_report(script: str, answers: dict[str, Any], console: Console) -> None:
    """Submit the job and print the result, log path, and follow-up hints."""
    retcode, stdout, stderr = submit_sbatch(script, job_name=answers.get("job_name", "slurm"))
    if retcode != 0:
        print(f"  {c.RED}✗ Submission failed (exit {retcode}){c.RESET}")
        if stdout:
            print(f"  {c.GRAY}{stdout}{c.RESET}")
        if stderr:
            print(f"  {c.RED}{stderr}{c.RESET}")
        sys.exit(1)

    job_id = stdout.strip()
    print(f"  {c.GREEN}✓ Submitted!{c.RESET} Job ID: {c.CYAN}{job_id}{c.RESET}")

    # Save a copy of the exact submitted script locally by default, so every
    # submission leaves a reproducible record next to where it was launched.
    if job_id:
        saved = _save_submitted_script(script, answers.get("job_name", "") or "slurm", job_id)
        if saved:
            print(f"  {c.GRAY}Script saved: {saved}{c.RESET}")

    # Read the actual --output path from the generated script (source of truth).
    log_path = f"{answers.get('job_name', '') or 'slurm'}-%j.out"
    for line in script.splitlines():
        if line.startswith("#SBATCH --output="):
            log_path = line.split("=", 1)[1].strip()
            break
    resolved_log = log_path.replace("%j", job_id)
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
    parser.add_argument("--dry-run", action="store_true", help="Print the script and exit without submitting")
    parser.add_argument("--print", action="store_true", help="Print the script to stdout and exit")
    parser.add_argument("--version", action="version", version=f"slurmate {__version__}")
    return parser.parse_args(argv)


def main() -> None:
    console = Console()
    args = parse_args()

    if not (args.print or args.dry_run):
        print_banner(animate=sys.stdout.isatty())

    config = load_config()

    answers_opt: dict[str, Any] | None = None
    wizard: Wizard | None = None
    if args.partition is not None:
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

    if args.print or args.dry_run:
        script = build_from_answers(answers)
        print(script)
        return

    script, queue_info = build_and_show(answers, console)

    if args.yes:
        _submit_and_report(script, answers, console)
        return

    import questionary

    from .theme import questionary_style
    QS = questionary_style()
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))
    default_name = f"{answers.get('job_name', '') or 'slurm'}.sh"

    def _resummarize() -> None:
        _show_script_and_summary(console, script, answers, estimate_su(
            answers.get("cpus", 1), answers.get("time_limit", "02:00:00"), answers.get("nodes", 1),
        ), queue_info)

    # A navigable action menu instead of a one-way confirm chain: every action
    # returns here. Esc (or the explicit option) re-opens the wizard to edit
    # answers; Ctrl-C/Quit cancels cleanly.
    can_edit = wizard is not None
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
            answers = wizard.edit()
            default_name = f"{answers.get('job_name', '') or 'slurm'}.sh"
            script, queue_info = build_and_show(answers, console)
            continue
        if action is None or action.startswith("Quit"):
            print(f"  {c.YELLOW}Not submitted.{c.RESET}")
            return
        if action.startswith("Submit"):
            _submit_and_report(script, answers, console)
            return
        if action.startswith("Open"):
            script = _edit_script_in_editor(script)
            _resummarize()
        elif action.startswith("Save"):
            _save_script(script, default_name)


if __name__ == "__main__":
    main()
