from __future__ import annotations

import re
import sys
from typing import Any, Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .builder import build_sbatch_script, estimate_su
from .system_utils import (
    fetch_conda_envs,
    fetch_gpu_types_for_partition,
    fetch_partitions,
    submit_sbatch,
)


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


def _partition_has_gpu(partition: dict[str, Any]) -> bool:
    return len(partition.get("gpu_types", [])) > 0


def _extract_info(partitions: list[dict]) -> tuple[list[str], list[str]]:
    names: list[str] = []
    descriptions: list[str] = []
    for p in partitions:
        name = p["name"]
        gpus = p.get("gpu_types", [])
        gpu_info = f" GPU:[{','.join(gpus)}]" if gpus else ""
        desc = f"{p.get('nodes', '?')} nodes{p.get('cpus_per_node', '?')} CPU{p.get('mem_per_node_mb', 0)//1024}G{gpu_info}"
        descriptions.append(desc)
        names.append(name)
    return names, descriptions


def main() -> None:
    console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Slurmify[/]  —  interactive sbatch wizard\n"
        "[dim]Generate and submit Slurm jobs without the guesswork[/]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    # ── 1. Job Name ──
    job_name = questionary.text(
        "Job name:",
        default="experiment",
    ).ask()
    if job_name is None:
        return

    # ── 2. Partition Selection ──
    console.print("[bold green]▸[/] Fetching available partitions...")
    partitions = fetch_partitions()
    if not partitions:
        console.print("[red]No partitions found.[/]")
        sys.exit(1)

    part_names, part_descs = _extract_info(partitions)
    part_choice = questionary.select(
        "Select partition:",
        choices=[
            questionary.Choice(
                title=f"{name:<20} {desc}",
                value=name,
            )
            for name, desc in zip(part_names, part_descs)
        ],
    ).ask()
    if part_choice is None:
        return

    selected_part = next(p for p in partitions if p["name"] == part_choice)

    # ── 3. CPU Cores ──
    cpus = questionary.text(
        "CPU cores:",
        default="4",
        validate=lambda v: v.strip().isdigit() and int(v) > 0,
    ).ask()
    if cpus is None:
        return
    cpus = int(cpus.strip())

    # ── 4. Memory ──
    memory = questionary.text(
        "Memory (e.g. 16G, 32G, 64000M):",
        default="16G",
        validate=lambda v: _validate_memory(v),
    ).ask()
    if memory is None:
        return
    memory = _normalize_memory(memory)

    # ── 5. Time Limit ──
    time_limit = questionary.text(
        "Time limit (hh:mm:ss or d-hh:mm:ss):",
        default="02:00:00",
        validate=lambda v: _validate_time(v),
    ).ask()
    if time_limit is None:
        return
    time_limit = time_limit.strip()

    # ── 6. GPUs ──
    gpus = 0
    gpu_type: Optional[str] = None
    if _partition_has_gpu(selected_part):
        gpus_str = questionary.text(
            "Number of GPUs:",
            default="1",
            validate=lambda v: v.strip().isdigit() and int(v) >= 0,
        ).ask()
        if gpus_str is None:
            return
        gpus = int(gpus_str.strip())

        if gpus > 0:
            gpu_types = fetch_gpu_types_for_partition(part_choice)
            if gpu_types:
                gpu_choices = ["Any"] + gpu_types
                gpu_type = questionary.select(
                    "GPU type:",
                    choices=gpu_choices,
                    default="Any",
                ).ask()
                if gpu_type is None:
                    return
                if gpu_type == "Any":
                    gpu_type = None
            else:
                gpu_type_str = questionary.text(
                    "GPU type (leave blank for any):",
                    default="",
                ).ask()
                if gpu_type_str is None:
                    return
                gpu_type = gpu_type_str.strip() or None

    # ── 7. Conda Environment ──
    console.print("[bold green]▸[/] Fetching available conda environments...")
    conda_envs = fetch_conda_envs()
    env_choices = ["None (skip)"] + conda_envs
    env_choice = questionary.select(
        "Conda environment:",
        choices=env_choices,
    ).ask()
    if env_choice is None:
        return
    env_name: Optional[str] = None if env_choice == "None (skip)" else env_choice

    # ── 8. Command ──
    command = questionary.text(
        "Command to run (e.g. python train.py):",
        default="",
    ).ask()
    if command is None:
        return
    command = command.strip()

    # ── 9. Build and Review ──
    script = build_sbatch_script(
        job_name=job_name,
        partition=part_choice,
        cpus=cpus,
        memory=memory,
        time_limit=time_limit,
        gpus=gpus,
        gpu_type=gpu_type,
        env_name=env_name,
        command=command,
    )

    su_estimate = estimate_su(cpus, time_limit)

    console.print()
    console.print(Panel(
        Syntax(script, "bash", theme="monokai", line_numbers=True),
        title="[bold]Generated sbatch script[/]",
        border_style="green",
        padding=(1, 2),
    ))

    summary_table = Table.grid(padding=(0, 2))
    summary_table.add_column(style="bold")
    summary_table.add_column()
    summary_table.add_row("Partition:", part_choice)
    summary_table.add_row("CPUs:", str(cpus))
    summary_table.add_row("Memory:", memory)
    summary_table.add_row("Time:", time_limit)
    if gpus > 0:
        gpu_label = gpu_type if gpu_type else "any"
        summary_table.add_row("GPUs:", f"{gpus} × {gpu_label}")
    if env_name:
        summary_table.add_row("Conda env:", env_name)
    summary_table.add_row("Estimated SU cost:", f"{su_estimate} SU")

    console.print()
    console.print(Panel(
        summary_table,
        title="[bold]Summary[/]",
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()

    # ── 10. Submit ──
    submit = questionary.confirm(
        "Submit this job to Slurm?",
        default=True,
    ).ask()
    if submit is None or not submit:
        console.print("[yellow]Job not submitted. Exiting.[/]")
        return

    retcode, stdout, stderr = submit_sbatch(script)
    if retcode != 0:
        console.print(f"[red]Submission failed[/] (exit {retcode})")
        if stdout:
            console.print(f"stdout: {stdout}")
        if stderr:
            console.print(f"stderr: {stderr}")
        sys.exit(1)

    console.print(f"[green]Submitted![/] {stdout}")


if __name__ == "__main__":
    main()
