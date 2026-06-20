from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .builder import build_from_answers, build_sbatch_script, estimate_su
from .system_utils import (
    fetch_partitions,
    fetch_queue_eta,
    normalize_memory,
    submit_sbatch,
)
from .theme import c, print_banner
from .tui import Wizard, _parse_custom_flags


# ── Batch mode helpers ───────────────────────────────────────────────────

def _get_partition(partitions: list[dict], name: str) -> dict[str, Any]:
    for p in partitions:
        if p["name"] == name:
            return p
    return {"name": name, "nodes": 0, "cpus_per_node": 0, "mem_per_node_mb": 0,
            "gpu_types": [], "timelimit": None, "is_public": True}


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
        "memory": normalize_memory(args.memory),
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
    table.add_row("Est. SU (rough):", f"[yellow]{su_estimate} SU[/]")
    if queue_info:
        table.add_row("Queue:", f"{queue_info['running']} run [dim]/[/] {queue_info['pending']} wait")
        eta_color = "green" if queue_info["eta_seconds"] < 3600 else "yellow"
        table.add_row("Est. ETA (rough):", f"[{eta_color}]{queue_info['eta_label']}[/]")

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


# ── CLI ──────────────────────────────────────────────────────────────────

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
        wizard = Wizard()
        answers = wizard.run()

    if not answers:
        print(f"  {c.YELLOW}Cancelled.{c.RESET}")
        return

    script, queue_info = build_and_show(answers, console)

    if not args.yes:
        import questionary
        from .theme import questionary_style
        QS = questionary_style()
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
        import questionary
        from .theme import questionary_style
        QS = questionary_style()
        submit = questionary.confirm("Submit this job to Slurm?", default=True, qmark="", style=QS).ask()
        if submit is None or not submit:
            print(f"  {c.YELLOW}Job not submitted. Exiting.{c.RESET}")
            return

    retcode, stdout, stderr = submit_sbatch(script, job_name=answers.get("job_name", "slurm"))
    if retcode != 0:
        print(f"  {c.RED}\u2717 Submission failed (exit {retcode}){c.RESET}")
        if stdout:
            print(f"  {c.GRAY}{stdout}{c.RESET}")
        if stderr:
            print(f"  {c.RED}{stderr}{c.RESET}")
        sys.exit(1)

    print(f"  {c.GREEN}\u2713 Submitted!{c.RESET} Job ID: {c.CYAN}{stdout}{c.RESET}")


if __name__ == "__main__":
    main()
