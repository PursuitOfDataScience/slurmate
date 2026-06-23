from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

MOCK_PARTITIONS = [
    {"name": "cpu-shared", "nodes": 100, "state": "up", "cpus_per_node": 32, "mem_per_node_mb": 131072, "gpu_types": [], "timelimit": "02:00:00", "is_public": True},
    {"name": "cpu-highmem", "nodes": 20, "state": "up", "cpus_per_node": 48, "mem_per_node_mb": 524288, "gpu_types": [], "timelimit": "12:00:00", "is_public": True},
    {"name": "gpu-shared", "nodes": 10, "state": "up", "cpus_per_node": 16, "mem_per_node_mb": 196608, "gpu_types": ["a100", "v100"], "timelimit": "04:00:00", "is_public": True},
    {"name": "gpu-highend", "nodes": 4, "state": "up", "cpus_per_node": 32, "mem_per_node_mb": 262144, "gpu_types": ["h100"], "timelimit": "24:00:00", "is_public": True},
    {"name": "debug", "nodes": 2, "state": "up", "cpus_per_node": 8, "mem_per_node_mb": 32768, "gpu_types": [], "timelimit": "01:00:00", "is_public": True},
]

MOCK_CONDA_ENVS = ["base", "pytorch", "tensorflow", "jax", "my_project"]

MOCK_GPU_TYPES = ["a100", "h100", "v100", "a40", "rtx6000", "h200", "l40s"]

MOCK_MODULES = ["python/anaconda", "cuda/11.8", "cuda/12.1", "gcc/9.3.0", "openmpi/4.1.1"]

MOCK_ACCOUNTS = ["my_lab", "training", "default"]


_RUN_TIMEOUT = 30


def _run_command(cmd: list[str], timeout: int = _RUN_TIMEOUT) -> tuple[str, str, int]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", -1


def _force_mock() -> bool:
    return os.environ.get("SLURMATE_MOCK", "").lower() in ("1", "true", "yes")


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
    # A bare integer is megabytes. Anything else is malformed (e.g. "16GB",
    # "16 G", "1.5.5G") — return 0 (unknown) rather than a misleading partial
    # like "16", which would masquerade as a tiny valid value in limit checks.
    if value.isdigit():
        return int(value)
    return 0


def validate_memory(value: str) -> bool:
    """Validate memory value.

    Accepts formats:
    - Plain digits: "16"
    - With units: "16G", "16g", "512M", "1T"
    - With Slurm N/C suffix: "16GN", "16GC"

    Rejects:
    - Zero or empty
    - Invalid formats
    """
    v = value.strip()
    if not v:
        return False
    # Accepts plain digits — reject a zero magnitude.
    if v.isdigit():
        return int(v) > 0
    # Accepts with unit suffix (KMGTP) and optional Slurm N/C — but reject a
    # zero magnitude regardless of unit ("0G"/"0M" are not valid sizes).
    m = re.match(r"^(\d+(?:\.\d+)?)([KMGTP])(?:[NC])?$", v.upper())
    if m:
        return float(m.group(1)) > 0
    return False


# Slurm's accepted --time grammar, allowing 1–2 digit lead fields:
#   minutes | minutes:seconds | hours:minutes:seconds |
#   days-hours | days-hours:minutes | days-hours:minutes:seconds
_TIME_PATTERNS = (
    r"^\d+$",                       # minutes
    r"^\d+:\d{2}$",                 # minutes:seconds
    r"^\d+:\d{2}:\d{2}$",           # hours:minutes:seconds
    r"^\d+-\d{1,2}$",               # days-hours
    r"^\d+-\d{1,2}:\d{2}$",         # days-hours:minutes
    r"^\d+-\d{1,2}:\d{2}:\d{2}$",   # days-hours:minutes:seconds
)


def validate_time(val: str) -> bool:
    """Validate a time limit string against Slurm's accepted --time formats."""
    v = val.strip()
    if not v:
        return True
    return any(re.match(p, v) for p in _TIME_PATTERNS)


def normalize_memory(value: str) -> str:
    """Normalize memory value to a standard format.

    Returns:
    - Plain digits prefixed with "M": "16" -> "16M"
    - Units already present: "16G" -> "16G"
    - Preserves Slurm N/C suffix if present
    """
    v = value.strip().upper()
    if not v:
        return ""
    # Plain digits: append M
    if v.isdigit():
        return f"{v}M"
    # Already has unit: return as-is
    if re.match(r"^(\d+(?:\.\d+)?)([KMGTP])(?:[NC])?$", v):
        return v
    # Invalid but return it anyway (validation should catch this)
    return v


def _detect_gpu_type(features: str, gres: str, known_models: set[str] | None = None) -> str:
    """Extract GPU model name from sinfo output.

    Priority:
    1. Parse model from ``gpu:MODEL:N`` in GRES.
    2. If GRES is count-only (``gpu:N``), scan node features:
       a. When ``known_models`` is given, *prefer* a feature token that matches
          a model seen in a typed GRES elsewhere in the partition. This
          disambiguates nodes whose features list rack/filesystem labels
          *before* the GPU (e.g. ``rack5,gpfs,a40`` → ``a40``).
       b. Otherwise (no corroborating match, or no ``known_models``), fall back
          to negative filtering: reject obvious CPU/arch/infra tokens and return
          the first plausible one. This keeps detecting GPU types that only ever
          appear in features and never in a typed GRES.
    3. If GRES has no ``gpu:`` at all the node has no GPUs — return empty.
    """
    text = f"{features},{gres}"
    gres_match = re.search(r"gpu:([a-z0-9._-]+):\d+", text, re.IGNORECASE)
    if gres_match:
        candidate = gres_match.group(1).replace("_", "-")
        if candidate.lower() not in {"gpu", "mps", "shard"}:
            return candidate

    if "gpu:" not in text.lower():
        return ""

    tokens = [t.strip() for t in re.split(r"[,/ ]+", features) if t.strip()]

    # Prefer a feature token corroborated by a typed GRES elsewhere.
    if known_models:
        known_lower = {m.lower() for m in known_models}
        for token in tokens:
            if token.lower() in known_lower:
                return token

    # Fall back to negative filtering: reject obvious non-GPU tokens and return
    # the first plausible one. Never drops a real GPU type that lives only in
    # the features string.
    for token in tokens:
        if not re.match(r"[a-zA-Z]", token):
            continue
        if len(token) >= 15:
            continue
        if re.match(
            r"(?:gold|xeon|epyc|ryzen|atom|i[3579]|avx\d*|sse\d*|fma)",
            token, re.IGNORECASE
        ):
            continue
        if re.match(
            r"(?:skylake|cascadelake|icelake|sapphirerapids|broadwell|haswell|zen\d*)",
            token, re.IGNORECASE
        ):
            continue
        if token.lower() in {
            "ssd", "nvme", "ib", "opa", "hdr", "hdd",
            "scratch", "fat", "thin", "gpu", "cpu", "mem", "node",
        }:
            continue
        return token

    return "gpu"


def _extract_token(line: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^ ]+)", line)
    return match.group(1).strip() if match else ""


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
            for match in re.finditer(r"gpu:([a-zA-Z0-9._-]+):\d+", gres_raw, re.IGNORECASE):
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


def fetch_public_partitions(all_parts: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return only publicly-usable partitions.

    Pass ``all_parts`` (a prior ``fetch_partitions()`` result) to avoid a
    redundant ``sinfo`` call — the partition step fetches it once and shares it.
    """
    if not is_tool_available("sinfo") or not is_tool_available("scontrol"):
        return [p for p in MOCK_PARTITIONS if p.get("is_public")]

    stdout, _, rc = _run_command(["scontrol", "show", "partition", "-o"])
    if rc != 0:
        return [p for p in MOCK_PARTITIONS if p.get("is_public")]

    partition_lines: dict[str, str] = {}
    for line in stdout.splitlines():
        name = _extract_token(line, "PartitionName")
        if name:
            partition_lines[name] = line

    if all_parts is None:
        all_parts = fetch_partitions()
    result: list[dict[str, Any]] = []
    for part in all_parts:
        name = part["name"]
        scontrol_line = partition_lines.get(name, "")
        allow_accounts = _extract_token(scontrol_line, "AllowAccounts")
        hidden = _extract_token(scontrol_line, "Hidden")

        is_public = (
            allow_accounts.upper() == "ALL"
            and hidden.upper() != "YES"
        )
        p = dict(part)
        p["is_public"] = is_public
        if is_public:
            result.append(p)

    return result


def fetch_qos_for_partition(partition: str) -> list[str]:
    if not is_tool_available("scontrol"):
        return []

    stdout, _, rc = _run_command(["scontrol", "show", "partition", partition, "-o"])
    if rc != 0:
        return []

    raw = _normalize_null(_extract_token(stdout, "AllowQos"))
    return _split_csv(raw) if raw else []


MOCK_QOS = ["normal", "high", "express", "gpu", "interactive"]


def fetch_known_qos() -> list[str]:
    """Fetch all QoS names known to the system via sacctmgr."""
    if not is_tool_available("sacctmgr"):
        return list(MOCK_QOS)

    stdout, _, rc = _run_command(
        ["sacctmgr", "show", "qos", "-P", "format=Name", "--noheader"]
    )
    if rc != 0:
        return list(MOCK_QOS)

    qos: list[str] = []
    for line in stdout.splitlines():
        name = line.strip()
        if name:
            qos.append(name)
    return qos or list(MOCK_QOS)


def fetch_gpu_types_for_partition(partition: str) -> list[str]:
    if not is_tool_available("sinfo"):
        return list(MOCK_GPU_TYPES)

    stdout, _, rc = _run_command(
        ["sinfo", "-h", "-N", "-p", partition, "-o", "%f|%G"]
    )
    if rc != 0:
        return list(MOCK_GPU_TYPES)

    # Pass 1: collect typed GPU models from gpu:MODEL:N across all nodes,
    # and stash the raw lines for a second pass.
    typed_models: set[str] = set()
    lines_data: list[tuple[str, str]] = []
    for line in stdout.splitlines():
        parts = line.strip().split("|", 1)
        if len(parts) < 2:
            continue
        features, gres = parts[0].strip(), parts[1].strip()
        lines_data.append((features, gres))
        gres_match = re.search(
            r"gpu:([a-z0-9._-]+):\d+", f"{features},{gres}", re.IGNORECASE
        )
        if gres_match:
            candidate = gres_match.group(1).replace("_", "-")
            if candidate.lower() not in {"gpu", "mps", "shard"}:
                typed_models.add(candidate)

    # Pass 2: detect a type per node, *preferring* corroboration against the
    # typed models (to disambiguate nodes that list non-GPU labels first) but
    # still falling back to feature scanning, so count-only GPU nodes whose
    # model lives only in features are never lost.
    types: set[str] = set()
    for features, gres in lines_data:
        gpu_type = _detect_gpu_type(features, gres, known_models=typed_models)
        if gpu_type and gpu_type != "gpu":
            types.add(gpu_type)
    return sorted(types)


def fetch_conda_envs(modules: list[str] | None = None) -> list[str]:
    """List conda environment names.

    Conda is frequently provided by a module (e.g. ``module load anaconda``)
    rather than being on ``PATH`` directly, so when ``modules`` are given we load
    them first — inside a login shell where ``module`` is defined — and then run
    ``conda env list``. This surfaces every env that becomes visible under the
    user's chosen module stack. With no modules we still try a bare ``conda``.
    """
    if _force_mock():
        return list(MOCK_CONDA_ENVS)

    prefix = ""
    if modules:
        # Quote each module token so a name with shell metacharacters can't break
        # out of (or inject into) the `bash -lc` string.
        names = " ".join(
            shlex.quote((m[:-9] if m.endswith("(default)") else m).strip())
            for m in modules
            if m and m.strip()
        )
        if names.strip():
            prefix = f"module load {names} >/dev/null 2>&1; "

    stdout, _, rc = _run_command(
        ["bash", "-lc", f"{prefix}conda env list --json 2>/dev/null"]
    )
    if rc != 0:
        # Real failure (conda/module not found): return nothing rather than
        # misleading mock names so the user can just type their env/path.
        return []

    try:
        # A login shell may emit banner text before the JSON; slice it out.
        start, end = stdout.find("{"), stdout.rfind("}")
        if start == -1 or end == -1:
            return []
        data = json.loads(stdout[start:end + 1])
        envs = [env.rstrip("/").split("/")[-1] for env in data.get("envs", []) if env]
        # De-dup while preserving order.
        return list(dict.fromkeys(envs))
    except (json.JSONDecodeError, KeyError):
        return []


def fetch_available_modules() -> list[str]:
    """Parse `module avail` output into a sorted unique list of module names."""
    stdout, stderr, rc = _run_command(["bash", "-lc", "command -v module && module -t avail 2>&1"])
    output = stdout + stderr
    if rc != 0:
        return list(MOCK_MODULES)

    modules: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        for mod in stripped.split():
            # Strip "(default)" annotation that the module system appends
            if mod.endswith("(default)"):
                mod = mod[:-9]
            if mod:
                modules.add(mod)
    return sorted(modules)


def fetch_user_accounts() -> list[str]:
    """Fetch Slurm accounts for the current user via sacctmgr."""
    if not is_tool_available("sacctmgr"):
        return list(MOCK_ACCOUNTS)

    stdout, _, rc = _run_command(
        ["sacctmgr", "show", "user", "-P", "format=Account", "--noheader"]
    )
    if rc != 0:
        return list(MOCK_ACCOUNTS)

    accounts: list[str] = []
    for line in stdout.splitlines():
        a = line.strip()
        if a:
            accounts.append(a)
    return accounts or list(MOCK_ACCOUNTS)


def _format_eta(seconds: int) -> str:
    if seconds <= 0:
        return "now"
    if seconds < 120:
        return f"~{seconds}s"
    if seconds < 3600:
        return f"~{seconds // 60}min"
    if seconds < 86400:
        return f"~{seconds // 3600}h"
    return f"~{seconds // 86400}d"


# Derive the mock label from _format_eta so the demo display matches the live
# formatter exactly (e.g. "~1h", not a hand-written "~1 hour").
MOCK_QUEUE_INFO = {
    "running": 12,
    "pending": 5,
    "eta_seconds": 3600,
    "eta_label": _format_eta(3600),
}


def fetch_queue_eta(partition: str, req_nodes: int = 1) -> dict[str, Any]:
    """Estimate queue wait time for a partition based on squeue / sinfo data."""
    if not is_tool_available("squeue") or not is_tool_available("sinfo"):
        return dict(MOCK_QUEUE_INFO)

    stdout, _, _ = _run_command(
        ["squeue", "-p", partition, "-o", "%T|%M|%l|%D", "--noheader"]
    )

    running = 0
    pending = 0

    for line in stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        state = parts[0]
        if state in ("RUNNING", "CONFIGURING", "COMPLETING"):
            running += 1
        elif state in ("PENDING", "SUSPENDED", "WAITING"):
            pending += 1

    # Get idle / mix / alloc node counts from sinfo
    sinfo_out, _, _ = _run_command(
        ["sinfo", "-p", partition, "-o", "%D|%a|%t", "--noheader"]
    )
    idle_nodes = 0
    mix_nodes = 0
    total_nodes = 0
    for line in sinfo_out.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3:
            try:
                nnodes = int(parts[0])
            except ValueError:
                nnodes = 0
            total_nodes += nnodes
            state_flag = parts[2].strip()
            if state_flag == "idle":
                idle_nodes += nnodes
            elif state_flag == "mix":
                mix_nodes += nnodes

    # Sensible ETA:
    #   If enough idle/available nodes exist → immediate
    #   Otherwise estimate from queue pressure
    if idle_nodes >= req_nodes:
        eta_sec = 0
    elif (idle_nodes + mix_nodes) >= req_nodes:
        eta_sec = 60  # ~1 min for scheduling shuffle
    elif running == 0:
        eta_sec = 300  # ~5 min conservative
    else:
        # Rough pressure estimate: pending jobs per running job × scheduling interval
        pressure = pending / max(1, running)
        eta_sec = int(min(pressure * 120, 7200))  # cap at 2 hours
        # If the partition has any idle capacity, reduce estimate
        if idle_nodes > 0 or mix_nodes > 0:
            eta_sec = max(60, eta_sec // 2)

    eta_label = _format_eta(eta_sec)

    return {"running": running, "pending": pending, "eta_seconds": eta_sec, "eta_label": eta_label}


def submit_sbatch(script_content: str, job_name: str = "slurm") -> tuple[int, str, str]:
    """Submit sbatch script and return (returncode, job_id_or_output, error_message).

    Args:
        script_content: The sbatch script content
        job_name: Job name for logging purposes

    Returns:
        Tuple of (returncode, job_id_or_stdout, stderr)
        - returncode: 0 on success, non-zero on failure
        - job_id_or_stdout: Job ID (integer as string) on success, stdout on failure
        - stderr: Error message on failure, empty string on success
    """
    # Parse output/error paths to create directories if they don't exist
    for line in script_content.splitlines():
        if line.startswith("#SBATCH --output=") or line.startswith("#SBATCH -o "):
            val = line.split("=", 1)[1].strip() if "=" in line else line.split(None, 2)[2].strip()
            dir_name = os.path.dirname(val)
            if dir_name:
                try:
                    os.makedirs(dir_name, exist_ok=True)
                except OSError as e:
                    logger.debug(f"Failed to create output directory {dir_name}: {e}")
        elif line.startswith("#SBATCH --error=") or line.startswith("#SBATCH -e "):
            val = line.split("=", 1)[1].strip() if "=" in line else line.split(None, 2)[2].strip()
            dir_name = os.path.dirname(val)
            if dir_name:
                try:
                    os.makedirs(dir_name, exist_ok=True)
                except OSError as e:
                    logger.debug(f"Failed to create error directory {dir_name}: {e}")

    if not is_tool_available("sbatch"):
        return 0, "", "sbatch not available (mock mode) — no job submitted"

    try:
        # Use --parsable for clean job ID output
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=script_content,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return -1, "", "Submission timed out after 30s"

    if result.returncode != 0:
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    job_id = result.stdout.strip()

    # Optionally save script to disk for reproducibility
    log_dir = os.environ.get("SLURMATE_LOG_DIR")
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            script_path = os.path.join(log_dir, f"{job_name}-{job_id}.sh")
            with open(script_path, "w") as f:
                f.write(script_content)
        except OSError as e:
            logger.debug(f"Failed to save script copy to SLURMATE_LOG_DIR: {e}")

    return result.returncode, job_id, ""


def _strip_inline_comment(v: str) -> str:
    """Drop a trailing ``# comment`` that sits outside any quotes."""
    in_single = in_double = False
    for i, ch in enumerate(v):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return v[:i].rstrip()
    return v.rstrip()


def _coerce_scalar(v: str) -> Any:
    """Coerce a single bare scalar token (string/int/float/bool)."""
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if re.match(r"^-?\d+$", v):
        return int(v)
    if re.match(r"^-?\d+\.\d+$", v):
        return float(v)
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return v


def _coerce_config_value(v: str) -> Any:
    """Parse one value for the naive key=value fallback parser.

    Handles quoted strings, arrays (with quoted *or* bare numeric items), ints,
    floats, negatives and booleans. Best-effort only — real TOML (tomllib/tomli)
    is used whenever available; this is the last resort.
    """
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(x.strip()) for x in inner.split(",") if x.strip()]
    return _coerce_scalar(v)


def _parse_config_naive(text: str) -> dict[str, Any]:
    """Minimal flat key=value parser used only when no TOML library is available."""
    config: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            config[k.strip()] = _coerce_config_value(_strip_inline_comment(v.strip()))
    return config


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    """Take top-level scalar keys, then merge an optional [defaults]/[slurmate] table."""
    config: dict[str, Any] = {k: v for k, v in data.items() if not isinstance(v, dict)}
    for section in ("defaults", "slurmate"):
        sect = data.get(section)
        if isinstance(sect, dict):
            config.update(sect)
    return config


def load_config() -> dict[str, Any]:
    """Load configuration defaults from a TOML file.

    Looks for ``.slurmate.toml`` in the current directory, then
    ``~/.config/slurmate/config.toml``; the first file found wins. Keys may sit
    at the top level or under a ``[defaults]`` (or ``[slurmate]``) table. Real
    TOML is used when a parser is available (``tomllib`` on 3.11+, ``tomli`` on
    older Pythons), otherwise a minimal flat key=value reader is used.

    Returns ``{}`` in mock mode (``SLURMATE_MOCK``) so tests stay hermetic, and
    on any missing or unreadable file.
    """
    if _force_mock():
        return {}

    from pathlib import Path

    toml: Any = None
    try:
        import tomllib
        toml = tomllib
    except ModuleNotFoundError:
        try:
            import tomli
            toml = tomli
        except ModuleNotFoundError:
            toml = None

    paths = [
        Path.cwd() / ".slurmate.toml",
        Path.home() / ".config" / "slurmate" / "config.toml",
    ]
    for p in paths:
        if not p.exists():
            continue
        try:
            if toml is not None:
                with open(p, "rb") as fb:
                    return _flatten_config(toml.load(fb))
            with open(p) as f:
                return _parse_config_naive(f.read())
        except Exception as e:
            logger.debug(f"Failed to load config from {p}: {e}")
            return {}
    return {}
