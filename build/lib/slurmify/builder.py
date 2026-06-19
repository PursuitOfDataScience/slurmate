from __future__ import annotations

import re
from typing import Optional

from .system_utils import _parse_slurm_time_to_minutes, _safe_int


def build_sbatch_script(
    job_name: str,
    partition: str,
    cpus: int,
    memory: str,
    time_limit: str,
    gpus: int = 0,
    gpu_type: Optional[str] = None,
    env_name: Optional[str] = None,
    command: str = "",
) -> str:
    lines = [
        "#!/bin/bash",
        "",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        "",
        f"#SBATCH --output={job_name}-%j.out",
        f"#SBATCH --error={job_name}-%j.err",
        "",
    ]

    if time_limit:
        lines.append(f"#SBATCH --time={time_limit}")
    lines.append(f"#SBATCH --cpus-per-task={cpus}")
    lines.append(f"#SBATCH --mem={memory}")

    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
        if gpu_type:
            lines.append(f"#SBATCH --constraint={gpu_type}")

    lines.append("")

    if env_name:
        lines.append(f"# Activate conda environment: {env_name}")
        lines.append(f"source $(conda info --base)/etc/profile.d/conda.sh")
        lines.append(f"conda activate {env_name}")
        lines.append("")

    if command:
        lines.append(command.rstrip())

    lines.append("")
    return "\n".join(lines)


def _normalize_time_for_parse(time_str: str) -> str:
    if re.match(r"^\d+-\d+:\d+:\d+$", time_str):
        return time_str
    if re.match(r"^\d+:\d+:\d+$", time_str):
        return time_str
    return time_str


def estimate_su(cpus: int, time_limit: str) -> str:
    minutes = _parse_slurm_time_to_minutes(
        _normalize_time_for_parse(time_limit)
    ) if time_limit else 120.0
    if minutes <= 0:
        minutes = 120.0
    hours = minutes / 60.0
    su = cpus * hours
    if su < 1:
        return f"{su:.2f}"
    if su < 100:
        return f"{su:.1f}"
    return f"{su:,.0f}"


def format_time_limit(hours: int, minutes: int, seconds: int) -> str:
    if hours >= 24:
        days = hours // 24
        hours = hours % 24
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
