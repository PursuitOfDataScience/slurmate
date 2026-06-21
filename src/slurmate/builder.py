from __future__ import annotations

import os
from typing import Any

from .system_utils import _parse_slurm_time_to_minutes


def build_from_answers(answers: dict[str, Any], partial: bool = False) -> str:
    """Build an sbatch script from an answers dict.

    Args:
        answers: Collected wizard/CLI answers.
        partial: When True, only emit directives for keys the user has actually
            provided (used by the live preview, so unentered fields don't show
            up as placeholder lines). When False, defaults fill in a complete,
            submittable script.
    """
    output_dir = answers.get("output_dir")
    output_file = answers.get("output_file")
    job_name = answers.get("job_name", "")
    prefix = job_name if job_name else "slurm"

    def _in_dir(name: str) -> str:
        # Place a bare filename inside output_dir; leave explicit paths alone.
        if output_dir and not os.path.isabs(name) and not os.path.dirname(name):
            return f"{output_dir.strip().rstrip('/')}/{name}"
        return name

    output_path: str | None
    error_path: str | None
    if output_file:
        of = output_file.strip()
        output_path = _in_dir(of)
        if of.endswith(".out"):
            error_path = output_path[:-4] + ".err"
        else:
            output_path = output_path + ".out"
            error_path = output_path[:-4] + ".err"
    elif output_dir:
        out_dir = output_dir.strip().rstrip("/")
        output_path = f"{out_dir}/{prefix}-%j.out"
        error_path = f"{out_dir}/{prefix}-%j.err"
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
        gpus=answers.get("gpus", 0) or 0,
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

    # One contiguous #SBATCH block, emitted in the same order the wizard asks
    # the questions, so the live preview grows top-to-bottom without reshuffling.
    if job_name or not partial:
        lines.append(f"#SBATCH --job-name={job_name}")
    if partition or not partial:
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

    if gpus > 0:
        gpu_fmt = gpu_format or os.environ.get("SLURMATE_GPU_FORMAT", "gres_type").lower()
        gpu_any = gpu_type is not None and gpu_type.lower() == "any"
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
        out = output_path or f"{prefix}-%j.out"
        err = error_path or f"{prefix}-%j.err"
        lines.append(f"#SBATCH --output={out}")
        lines.append(f"#SBATCH --error={err}")

    if custom_sbatch:
        for flag in custom_sbatch:
            if gpus > 0:
                parts = flag.strip().split('=', 1)
                flag_name = parts[0].strip()
                flag_val = parts[1].strip() if len(parts) > 1 else ""
                if flag_name == "--gres" and flag_val.startswith("gpu"):
                    continue
                if flag_name == "--gpus":
                    continue
                if flag_name == "--constraint" and gpu_type and flag_val == gpu_type:
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
            lines.append(f"source activate {env_name}")
        elif strategy == "mamba":
            lines.append("")
            lines.append(f"mamba activate {env_name}")
        elif strategy in ("virtualenv (venv)", "venv"):
            lines.append("")
            lines.append(f"source {env_name}/bin/activate")

    if command:
        lines.append("")
        lines.append(command.rstrip())

    if partial:
        while len(lines) > 2 and lines[-1] == "":
            lines.pop()
    else:
        lines.append("")
    return "\n".join(lines)


def estimate_su(cpus: int, time_limit: str, nodes: int = 1) -> str:
    """Estimate Service Units (SU) cost for a job.

    Service Units are typically core-hours (CPUs * hours * nodes).

    Args:
        cpus: Number of CPU cores per task.
        time_limit: Time limit string in Slurm format (e.g. "hh:mm:ss" or "d-hh:mm:ss").
        nodes: Number of nodes requested.

    Returns:
        Formatted string representation of estimated SUs.
    """
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
