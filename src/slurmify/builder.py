from __future__ import annotations

import re
from typing import Any, Optional

from .system_utils import _parse_slurm_time_to_minutes


def build_from_answers(answers: dict[str, Any]) -> str:
    """Build sbatch script from answers dict.
    
    Extracts all required parameters from the answers dict and calls
    build_sbatch_script to generate the script.
    """
    return build_sbatch_script(
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


def build_sbatch_script(
    job_name: str,
    partition: str,
    cpus: int,
    memory: str,
    time_limit: str,
    nodes: int = 1,
    gpus: int = 0,
    gpu_type: Optional[str] = None,
    account: Optional[str] = None,
    qos: Optional[str] = None,
    array_spec: Optional[str] = None,
    output_path: Optional[str] = None,
    error_path: Optional[str] = None,
    modules: Optional[list[str]] = None,
    custom_sbatch: Optional[list[str]] = None,
    env_name: Optional[str] = None,
    command: str = "",
) -> str:
    lines = [
        "#!/bin/bash",
        "",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
    ]

    if account:
        lines.append(f"#SBATCH --account={account}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")

    prefix = job_name if job_name else "slurm"
    out = output_path or f"{prefix}-%j.out"
    err = error_path or f"{prefix}-%j.err"
    lines.extend(["", f"#SBATCH --output={out}", f"#SBATCH --error={err}", ""])

    if time_limit:
        lines.append(f"#SBATCH --time={time_limit}")
    lines.append(f"#SBATCH --nodes={nodes}")
    lines.append(f"#SBATCH --cpus-per-task={cpus}")
    lines.append(f"#SBATCH --mem={memory}")

    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
        if gpu_type:
            lines.append(f"#SBATCH --constraint={gpu_type}")

    if array_spec:
        lines.append(f"#SBATCH --array={array_spec}")

    if custom_sbatch:
        for flag in custom_sbatch:
            lines.append(f"#SBATCH {flag}")
    lines.append("")

    if modules:
        for mod in modules:
            lines.append(f"module load {mod}")
        lines.append("")

    if env_name:
        lines.append(f"source $(conda info --base)/etc/profile.d/conda.sh")
        lines.append(f"conda activate {env_name}")
        lines.append("")

    if command:
        lines.append(command.rstrip())

    lines.append("")
    return "\n".join(lines)


def estimate_su(cpus: int, time_limit: str, nodes: int = 1) -> str:
    minutes = _parse_slurm_time_to_minutes(time_limit) if time_limit else 120.0
    if minutes <= 0:
        minutes = 120.0
    hours = minutes / 60.0
    su = cpus * hours * nodes
    if su < 1:
        return f"{su:.2f}"
    if su < 100:
        return f"{su:.1f}"
    return f"{su:,.0f}"
