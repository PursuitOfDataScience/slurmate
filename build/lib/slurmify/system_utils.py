from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any


MOCK_PARTITIONS = [
    {"name": "cpu-shared", "nodes": 100, "state": "up", "cpus_per_node": 32, "mem_per_node_mb": 131072, "gpu_types": [], "timelimit": "02:00:00"},
    {"name": "cpu-highmem", "nodes": 20, "state": "up", "cpus_per_node": 48, "mem_per_node_mb": 524288, "gpu_types": [], "timelimit": "12:00:00"},
    {"name": "gpu-shared", "nodes": 10, "state": "up", "cpus_per_node": 16, "mem_per_node_mb": 196608, "gpu_types": ["a100", "v100"], "timelimit": "04:00:00"},
    {"name": "gpu-highend", "nodes": 4, "state": "up", "cpus_per_node": 32, "mem_per_node_mb": 262144, "gpu_types": ["h100"], "timelimit": "24:00:00"},
    {"name": "debug", "nodes": 2, "state": "up", "cpus_per_node": 8, "mem_per_node_mb": 32768, "gpu_types": [], "timelimit": "01:00:00"},
]

MOCK_CONDA_ENVS = ["base", "pytorch", "tensorflow", "jax", "my_project"]

MOCK_GPU_TYPES = ["a100", "h100", "v100", "a40", "rtx6000", "h200", "l40s"]


def _run_command(cmd: list[str]) -> tuple[str, str, int]:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout, result.stderr, result.returncode


def _force_mock() -> bool:
    return os.environ.get("SLURMIFY_MOCK", "").lower() in ("1", "true", "yes")


def is_tool_available(name: str) -> bool:
    if _force_mock():
        return False
    return shutil.which(name) is not None


def _safe_int(raw: str) -> int:
    match = re.search(r"\d+", raw.strip())
    return int(match.group(0)) if match else 0


def _normalize_null(raw: str) -> str:
    value = raw.strip()
    return "" if value.lower() in {"", "(null)", "null", "-", "n/a"} else value


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_mem_to_mb(raw: str) -> int:
    value = raw.strip().upper()
    if not value or value == "0":
        return 0
    match = re.match(r"^(\d+(?:\.\d+)?)([KMGTP])(?:[NC])?$", value)
    if match:
        num = float(match.group(1))
        scale = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2, "P": 1024 ** 3}
        return int(num * scale[match.group(2)])
    plain = _safe_int(value)
    return plain if plain > 0 else 0


def _detect_gpu_type(features: str, gres: str) -> str:
    GPU_TYPE_CANDIDATES = (
        "h200", "h100", "a100", "a40", "a30", "v100", "p100", "k80",
        "t4", "l40", "l40s", "l4", "rtx6000", "rtx5000", "rtx4090", "rtx3090",
        "rtx2080", "rtx", "mi300", "mi250", "mi200", "mi100", "tesla",
    )
    text = f"{features},{gres}".lower()
    gres_model = re.search(r"gpu:([a-z0-9._-]+):\d+", text)
    if gres_model:
        candidate = gres_model.group(1).replace("_", "-")
        if candidate not in {"gpu", "mps", "shard"}:
            return candidate
    for token in GPU_TYPE_CANDIDATES:
        if token in text:
            return token
    return "gpu" if "gpu" in text else ""


def _parse_slurm_time_to_minutes(time_str: str) -> float:
    value = time_str.strip()
    if not value or value in {"UNLIMITED", "NOT_SET", "N/A", "INVALID"}:
        return 0.0
    if "-" in value:
        day_part, rest = value.split("-", 1)
        return _safe_int(day_part) * 1440 + _parse_slurm_time_to_minutes(rest)
    parts = value.split(":")
    if len(parts) == 3:
        return _safe_int(parts[0]) * 60 + _safe_int(parts[1]) + _safe_int(parts[2]) / 60.0
    if len(parts) == 2:
        return _safe_int(parts[0]) + _safe_int(parts[1]) / 60.0
    return float(_safe_int(parts[0])) if parts else 0.0


def fetch_partitions() -> list[dict[str, Any]]:
    if not is_tool_available("sinfo"):
        return list(MOCK_PARTITIONS)

    stdout, _, rc = _run_command(
        ["sinfo", "-h", "-o", "%P|%l|%D|%a|%c|%m|%G"]
    )
    if rc != 0:
        return list(MOCK_PARTITIONS)

    partitions: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        parts = line.strip().split("|", 6)
        if len(parts) < 5:
            continue
        name = parts[0].strip().rstrip("*")
        timelimit = parts[1].strip() if len(parts) > 1 else ""
        nodes = _safe_int(parts[2]) if len(parts) > 2 else 0
        state = parts[3].strip().lower() if len(parts) > 3 else "up"
        cpus = _safe_int(parts[4]) if len(parts) > 4 else 0
        mem_raw = parts[5].strip() if len(parts) > 5 else ""
        gres_raw = parts[6].strip() if len(parts) > 6 else ""

        gpu_types: list[str] = []
        if gres_raw and gres_raw != "(null)":
            for match in re.finditer(r"gpu:([a-z0-9._-]+):\d+", gres_raw.lower()):
                gpu_types.append(match.group(1).replace("_", "-"))

        if name not in partitions:
            partitions[name] = {
                "name": name,
                "nodes": nodes,
                "state": state,
                "cpus_per_node": cpus,
                "mem_per_node_mb": _parse_mem_to_mb(mem_raw) if mem_raw else 0,
                "gpu_types": gpu_types,
                "timelimit": timelimit if timelimit != "infinite" else None,
            }
        else:
            p = partitions[name]
            p["nodes"] = max(p["nodes"], nodes)
            p["cpus_per_node"] = max(p["cpus_per_node"], cpus)
            mem_mb = _parse_mem_to_mb(mem_raw) if mem_raw else 0
            p["mem_per_node_mb"] = max(p["mem_per_node_mb"], mem_mb)
            p["gpu_types"] = list(set(p["gpu_types"] + gpu_types))

    return list(partitions.values())


def fetch_gpu_types_for_partition(partition: str) -> list[str]:
    if not is_tool_available("sinfo"):
        return list(MOCK_GPU_TYPES)

    stdout, _, rc = _run_command(
        ["sinfo", "-h", "-N", "-p", partition, "-o", "%f|%G"]
    )
    if rc != 0:
        return []

    types: set[str] = set()
    for line in stdout.splitlines():
        parts = line.strip().split("|", 1)
        if len(parts) < 2:
            continue
        features, gres = parts[0].strip(), parts[1].strip()
        gpu_type = _detect_gpu_type(features, gres)
        if gpu_type and gpu_type != "gpu":
            types.add(gpu_type)
    return sorted(types)


def fetch_conda_envs() -> list[str]:
    if not is_tool_available("conda"):
        return list(MOCK_CONDA_ENVS)

    stdout, _, rc = _run_command(["conda", "env", "list", "--json"])
    if rc != 0:
        return list(MOCK_CONDA_ENVS)

    try:
        data = json.loads(stdout)
        envs = data.get("envs", [])
        return [env.split("/")[-1] for env in envs]
    except (json.JSONDecodeError, KeyError):
        return list(MOCK_CONDA_ENVS)


def submit_sbatch(script_content: str) -> tuple[int, str, str]:
    if not is_tool_available("sbatch"):
        return 0, "", "sbatch not available (mock mode) — no job submitted"

    result = subprocess.run(
        ["sbatch"],
        input=script_content,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    return result.returncode, result.stdout.strip(), ""
