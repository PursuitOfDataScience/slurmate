from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .builder import build_sbatch_script, estimate_su
from .system_utils import (
    fetch_available_modules,
    fetch_conda_envs,
    fetch_gpu_types_for_partition,
    fetch_known_qos,
    fetch_partitions,
    fetch_public_partitions,
    fetch_qos_for_partition,
    fetch_queue_eta,
    fetch_user_accounts,
    submit_sbatch,
)
from .theme import autocomplete, c, ok, path_input, print_banner, questionary_style, select_input, text_input, tool_status


QS = questionary_style()


CUSTOM = "Enter partition name manually..."
PRIVATE = "Include private partitions"


def _validate_time(val: str) -> bool:
    if not val.strip():
        return True
    if re.match(r"^\d+-\d{2}:\d{2}:\d{2}$", val.strip()):
        return True
    if re.match(r"^\d{2}:\d{2}:\d{2}$", val.strip()):
        return True
    return False


def _validate_memory(val: str) -> bool:
    if re.match(r"^\d+[KMGTP]?$", val.strip().upper()):
        return True
    if val.strip().isdigit():
        return True
    return False


def _normalize_memory(val: str) -> str:
    v = val.strip().upper()
    if v.isdigit():
        return f"{v}M"
    if re.match(r"^\d+[KMGTP]$", v):
        return v
    return v


def _get_partition(partitions: list[dict], name: str) -> dict[str, Any]:
    for p in partitions:
        if p["name"] == name:
            return p
    return {"name": name, "nodes": 0, "cpus_per_node": 0, "mem_per_node_mb": 0, "gpu_types": [], "timelimit": None, "is_public": True}


def _fmt_partition(p: dict[str, Any]) -> str:
    name = p["name"]
    nodes = p.get("nodes", "?")
    cpus = p.get("cpus_per_node", "?")
    mem_gb = p.get("mem_per_node_mb", 0) // 1024
    gpus = p.get("gpu_types", [])
    label = f"{name:<12} {nodes} nodes \u00b7 {cpus} CPU \u00b7 {mem_gb}G"
    if gpus:
        label += f" \u00b7 GPU:[{','.join(gpus)}]"
    return label


# ── Steps ─────────────────────────────────────────────────────────────────

def _step_job_name(answers: dict, console: Console) -> dict:
    name = text_input("Job name:")
    if name is None or not name.strip():
        return {"_action": "cancel"}
    return {"_action": "next", "job_name": name.strip()}


def _step_account(answers: dict, console: Console) -> dict:
    tool_status("Checking Slurm accounts")
    accounts = fetch_user_accounts()
    tool_status("Checking Slurm accounts", "success")
    val = autocomplete("Account (optional — start typing or press Tab):", choices=accounts)
    if val is None:
        return {"_action": "back"}
    return {"_action": "next", "account": val.strip() or None}


def _step_partition(answers: dict, console: Console) -> dict:
    tool_status("Fetching partitions")
    public = fetch_public_partitions()
    all_parts = fetch_partitions()
    tool_status("Fetching partitions", "success")

    choices = [CUSTOM]
    if public:
        choices.append(PRIVATE)
        choices.extend(_fmt_partition(p) for p in public)
    else:
        choices.extend(_fmt_partition(p) for p in all_parts)

    raw = select_input("Select partition:", choices=choices)
    if raw is None:
        return {"_action": "back"}

    if raw == CUSTOM:
        custom = text_input("Enter partition name:")
        if custom is None:
            return {"_action": "back"}
        name = custom.strip()
        return {"_action": "next", "partition": name, "_partition_obj": _get_partition(all_parts, name)}

    if raw == PRIVATE:
        fmt_all = [_fmt_partition(p) for p in all_parts]
        raw2 = select_input("Select partition (all):", choices=fmt_all)
        if raw2 is None:
            return {"_action": "back"}
        idx = fmt_all.index(raw2)
        part = all_parts[idx]
        return {"_action": "next", "partition": part["name"], "_partition_obj": part}

    idx = choices.index(raw)
    public_idx = idx - (2 if PRIVATE in choices else 1)
    if public_idx < 0 or public_idx >= len(public):
        return {"_action": "back"}
    part = public[public_idx]
    return {"_action": "next", "partition": part["name"], "_partition_obj": part}


def _step_qos(answers: dict, console: Console) -> dict:
    part = answers.get("partition", "")
    tool_status("Checking QoS options")
    qos_list = fetch_qos_for_partition(part)
    tool_status("Checking QoS options", "success")

    # Validate: values must be known QoS names, not partition names
    if qos_list and part in qos_list:
        qos_list = []
    if qos_list:
        known = set(fetch_known_qos())
        qos_list = [q for q in qos_list if q in known]

    if not qos_list:
        return {"_action": "next", "qos": None}

    choices = ["Default (none)"] + qos_list
    val = select_input("QoS:", choices=choices, default="Default (none)")
    if val is None:
        return {"_action": "back"}
    return {"_action": "next", "qos": None if val == "Default (none)" else val}


def _step_cpus(answers: dict, console: Console) -> dict:
    val = text_input("CPU cores:", default="4",
                     validate=lambda v: v.strip().isdigit() and int(v) > 0)
    if val is None:
        return {"_action": "back"}
    return {"_action": "next", "cpus": int(val.strip())}


MEMORY_CHOICES = ["4G", "8G", "16G", "32G", "64G", "128G", "256G", "512G", "64000M"]

def _step_memory(answers: dict, console: Console) -> dict:
    val = autocomplete("Memory (e.g. 16G, 32G, 64000M):", choices=MEMORY_CHOICES, default="16G",
                       validate=lambda v: _validate_memory(v))
    if val is None:
        return {"_action": "back"}
    return {"_action": "next", "memory": _normalize_memory(val)}


TIME_CHOICES = ["01:00:00", "02:00:00", "04:00:00", "08:00:00", "12:00:00", "24:00:00", "48:00:00", "7-00:00:00"]

def _step_time_limit(answers: dict, console: Console) -> dict:
    val = autocomplete("Time limit (hh:mm:ss or d-hh:mm:ss):", choices=TIME_CHOICES, default="02:00:00",
                       validate=lambda v: _validate_time(v))
    if val is None:
        return {"_action": "back"}
    return {"_action": "next", "time_limit": val.strip()}


def _step_nodes(answers: dict, console: Console) -> dict:
    val = text_input("Nodes:", default="1",
                     validate=lambda v: v.strip().isdigit() and int(v) > 0)
    if val is None:
        return {"_action": "back"}
    return {"_action": "next", "nodes": int(val.strip())}


def _step_gpus(answers: dict, console: Console) -> dict:
    part_obj = answers.get("_partition_obj", {})
    if len(part_obj.get("gpu_types", [])) == 0:
        return {"_action": "next", "gpus": 0, "gpu_type": None}

    choices = ["0", "1", "2", "4", "8"]
    val = select_input("GPUs:", choices=choices, default="1")
    if val is None:
        return {"_action": "back"}
    gpus = int(val)
    return {"_action": "next", "gpus": gpus, "_ask_gpu_type": gpus > 0}


def _step_gpu_type(answers: dict, console: Console) -> dict:
    if not answers.get("_ask_gpu_type"):
        return {"_action": "next", "gpu_type": None}

    part_name = answers.get("partition", "")
    tool_status("Scanning GPU types")
    gpu_types = fetch_gpu_types_for_partition(part_name)
    tool_status("Scanning GPU types", "success")

    if gpu_types:
        choices = ["Any"] + gpu_types
        val = select_input("GPU type:", choices=choices, default="Any")
        if val is None:
            return {"_action": "back"}
        return {"_action": "next", "gpu_type": None if val == "Any" else val}

    val = text_input("GPU type (leave blank for any):")
    if val is None:
        return {"_action": "back"}
    return {"_action": "next", "gpu_type": val.strip() or None}


def _step_array(answers: dict, console: Console) -> dict:
    val = text_input("Array spec (e.g. 1-10, 1,3,5-7%4 — optional):")
    if val is None:
        return {"_action": "back"}
    return {"_action": "next", "array_spec": val.strip() or None}


def _step_modules(answers: dict, console: Console) -> dict:
    tool_status("Fetching available modules")
    all_mods = fetch_available_modules()
    tool_status("Fetching available modules", "success")
    val = autocomplete(
        "Modules (comma-separated, e.g. python/anaconda,cuda — optional):",
        choices=all_mods,
    )
    if val is None:
        return {"_action": "back"}
    mods = [m.strip() for m in val.split(",") if m.strip()] if val.strip() else None
    return {"_action": "next", "modules": mods}


def _step_env(answers: dict, console: Console) -> dict:
    tool_status("Fetching conda environments")
    envs = fetch_conda_envs()
    tool_status("Fetching conda environments", "success")

    choices = ["None (skip)"] + envs
    val = select_input("Conda environment:", choices=choices, default="None (skip)")
    if val is None:
        return {"_action": "back"}
    return {"_action": "next", "env_name": None if val == "None (skip)" else val}


def _step_command(answers: dict, console: Console) -> dict:
    val = path_input("Command to run (e.g. python train.py):")
    if val is None:
        return {"_action": "back"}
    return {"_action": "next", "command": val.strip()}


SBATCH_FLAGS = [
    "--exclusive", "--reservation=", "--ntasks=", "--ntasks-per-node=",
    "--threads-per-core=", "--mem-per-cpu=", "--constraint=",
    "--licenses=", "--gres=", "--tmp=", "--hint=", "--signal=",
]

def _step_custom_sbatch(answers: dict, console: Console) -> dict:
    val = autocomplete(
        "Extra #SBATCH flags (e.g. --exclusive --reservation=abc — optional):",
        choices=SBATCH_FLAGS,
    )
    if val is None:
        return {"_action": "back"}
    raw = val.strip()
    flags = [f.strip() for f in raw.split("--") if f.strip()] if raw else []
    flags = [f"--{f}" if not f.startswith("#") else f for f in flags]
    return {"_action": "next", "custom_sbatch": flags or None}


def _parse_custom_flags(raw: str) -> list[str]:
    flags = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("#SBATCH"):
            part = part.replace("#SBATCH", "", 1).strip()
        if not part.startswith("--"):
            part = f"--{part}"
        flags.append(part)
    return flags


STEPS = [
    ("job_name", _step_job_name),
    ("account", _step_account),
    ("partition", _step_partition),
    ("qos", _step_qos),
    ("cpus", _step_cpus),
    ("memory", _step_memory),
    ("time_limit", _step_time_limit),
    ("nodes", _step_nodes),
    ("gpus", _step_gpus),
    ("gpu_type", _step_gpu_type),
    ("array", _step_array),
    ("modules", _step_modules),
    ("env", _step_env),
    ("command", _step_command),
    ("custom_sbatch", _step_custom_sbatch),
]


# ── Flow ───────────────────────────────────────────────────────────────────

def run_interactive(console: Console) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    step_idx = 0

    while 0 <= step_idx < len(STEPS):
        _, func = STEPS[step_idx]
        result = func(answers.copy(), console)
        if result is None:
            return {}
        action = result.pop("_action", "next")
        answers.update(result)
        if action == "back":
            step_idx -= 1
        elif action == "cancel":
            return {}
        elif action == "next":
            step_idx += 1

    return answers


def run_batch(args: argparse.Namespace, console: Console) -> dict[str, Any]:
    print(f"  {c.CYAN}\u25b8{c.RESET} {c.GRAY}Running in batch mode{c.RESET}\n")

    all_parts = fetch_partitions()
    part_obj = _get_partition(all_parts, args.partition)
    mods = [m.strip() for m in args.modules.split(",") if m.strip()] if args.modules else None

    return {
        "job_name": args.job_name,
        "account": args.account or None,
        "partition": args.partition,
        "_partition_obj": part_obj,
        "qos": args.qos or None,
        "cpus": args.cpus,
        "memory": _normalize_memory(args.memory),
        "time_limit": args.time,
        "nodes": args.nodes,
        "gpus": args.gpus,
        "gpu_type": args.gpu_type or None,
        "array_spec": args.array or None,
        "modules": mods,
        "env_name": args.env or None,
        "command": args.command,
        "custom_sbatch": _parse_custom_flags(args.custom_sbatch) if args.custom_sbatch else None,
    }


def build_and_show(answers: dict[str, Any], console: Console) -> tuple[str, dict]:
    script = build_sbatch_script(
        job_name=answers.get("job_name", ""),
        partition=answers.get("partition", ""),
        account=answers.get("account"),
        qos=answers.get("qos"),
        cpus=answers.get("cpus", 1),
        memory=answers.get("memory", "16G"),
        time_limit=answers.get("time_limit", "02:00:00"),
        nodes=answers.get("nodes", 1),
        gpus=answers.get("gpus", 0),
        gpu_type=answers.get("gpu_type"),
        array_spec=answers.get("array_spec"),
        modules=answers.get("modules"),
        custom_sbatch=answers.get("custom_sbatch"),
        env_name=answers.get("env_name"),
        command=answers.get("command", ""),
    )

    su_estimate = estimate_su(
        answers.get("cpus", 1),
        answers.get("time_limit", "02:00:00"),
        answers.get("nodes", 1),
    )

    tool_status("Checking queue")
    queue_info = fetch_queue_eta(
        answers.get("partition", ""),
        req_nodes=answers.get("nodes", 1),
    )
    tool_status("Checking queue", "success")

    _show_script_and_summary(console, script, answers, su_estimate, queue_info)
    return script, queue_info


def _show_script_and_summary(console: Console, script: str, answers: dict[str, Any],
                              su_estimate: str, queue_info: dict | None = None) -> None:
    print()
    script_w = min(console.width - 2, 60)
    console.print(Panel(
        Syntax(script, "bash", theme="monokai", line_numbers=True),
        title=f"[bold]{c.PINK}Generated sbatch script[/]",
        border_style="bright_magenta",
        padding=(0, 1),
        width=script_w,
    ))

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold bright_black")
    table.add_column(style="cyan")
    table.add_row("Job:", answers.get("job_name", ""))
    table.add_row("Partition:", answers.get("partition", ""))
    if answers.get("account"):
        table.add_row("Account:", answers["account"])
    if answers.get("qos"):
        table.add_row("QoS:", f"[magenta]{answers['qos']}[/]")
    table.add_row("CPUs:", str(answers.get("cpus", "")))
    table.add_row("Memory:", answers.get("memory", ""))
    table.add_row("Time:", answers.get("time_limit", ""))
    table.add_row("Nodes:", str(answers.get("nodes", 1)))
    if answers.get("gpus", 0) > 0:
        gt = answers.get("gpu_type") or "any"
        table.add_row("GPUs:", f"{answers['gpus']} \u00d7 {gt}")
    if answers.get("array_spec"):
        table.add_row("Array:", f"[yellow]{answers['array_spec']}[/]")
    if answers.get("modules"):
        table.add_row("Modules:", ", ".join(answers["modules"]))
    if answers.get("env_name"):
        table.add_row("Conda env:", answers["env_name"])
    if answers.get("custom_sbatch"):
        table.add_row("Custom flags:", ", ".join(answers["custom_sbatch"]))
    table.add_row("SU cost:", f"[yellow]{su_estimate} SU[/]")
    if queue_info:
        table.add_row("Queue:", f"{queue_info['running']} run [dim]/[/] {queue_info['pending']} wait")
        eta_color = "green" if queue_info["eta_seconds"] < 3600 else "yellow"
        table.add_row("ETA:", f"[{eta_color}]{queue_info['eta_label']}[/]")

    print()
    console.print(Panel(table, title=f"[bold]{c.CYAN}Summary[/]", border_style="cyan", padding=(0, 1)))
    print()


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slurmify \u2014 sbatch wizard")
    parser.add_argument("--job-name", default="", help="Job name")
    parser.add_argument("--account", default=None, help="Slurm account")
    parser.add_argument("--partition", default="", help="Target partition")
    parser.add_argument("--qos", default=None, help="QoS")
    parser.add_argument("--cpus", type=int, default=4, help="CPU cores")
    parser.add_argument("--memory", default="16G", help="Memory (e.g. 16G, 32G, 64000M)")
    parser.add_argument("--time", default="02:00:00", help="Time limit")
    parser.add_argument("--nodes", type=int, default=1, help="Node count")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs")
    parser.add_argument("--gpu-type", default=None, help="GPU type (e.g. a100, h100)")
    parser.add_argument("--array", default=None, help="Array spec (e.g. 1-10)")
    parser.add_argument("--modules", default=None, help="Comma-separated modules")
    parser.add_argument("--env", default=None, help="Conda environment")
    parser.add_argument("--command", default="", help="Command to run")
    parser.add_argument("--custom-sbatch", default=None,
                        help="Comma-separated extra #SBATCH flags (e.g. --exclusive,--reservation=abc)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation and submit")
    return parser.parse_args(argv)


def main() -> None:
    console = Console()
    print_banner(animate=sys.stdout.isatty())

    args = parse_args()

    if args.partition:
        answers = run_batch(args, console)
    else:
        answers = run_interactive(console)

    if not answers:
        print(f"  {c.YELLOW}Cancelled.{c.RESET}")
        return

    script, queue_info = build_and_show(answers, console)
    if script is None:
        return

    if not args.yes:
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))
        edit = questionary.confirm(f"Open in {editor} to edit?", default=True, qmark="", style=QS).ask()
        if edit is None:
            print(f"  {c.YELLOW}Cancelled.{c.RESET}")
            return
        if edit:
            script = _edit_script_in_editor(script)
            _show_script_and_summary(console, script, answers, estimate_su(
                answers.get("cpus", 1), answers.get("time_limit", "02:00:00"), answers.get("nodes", 1),
            ), queue_info)

    if args.yes:
        submit = True
    else:
        submit = questionary.confirm("Submit this job to Slurm?", default=True, qmark="", style=QS).ask()
        if submit is None or not submit:
            print(f"  {c.YELLOW}Job not submitted. Exiting.{c.RESET}")
            return

    retcode, stdout, stderr = submit_sbatch(script)
    if retcode != 0:
        print(f"  {c.RED}\u2717 Submission failed (exit {retcode}){c.RESET}")
        if stdout:
            print(f"  {c.GRAY}{stdout}{c.RESET}")
        if stderr:
            print(f"  {c.RED}{stderr}{c.RESET}")
        sys.exit(1)

    print(f"  {c.GREEN}\u2713 Submitted!{c.RESET} {c.GRAY}{stdout}{c.RESET}")


if __name__ == "__main__":
    main()
