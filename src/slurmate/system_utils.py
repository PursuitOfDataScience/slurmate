from __future__ import annotations

import getpass
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

MOCK_PARTITIONS: list[dict[str, Any]] = [
    {"name": "cpu-shared", "nodes": 100, "state": "up", "cpus_per_node": 32, "mem_per_node_mb": 131072, "gpu_types": [], "has_gpu": False, "timelimit": "02:00:00", "is_public": True},
    {"name": "cpu-highmem", "nodes": 20, "state": "up", "cpus_per_node": 48, "mem_per_node_mb": 524288, "gpu_types": [], "has_gpu": False, "timelimit": "12:00:00", "is_public": True},
    {"name": "gpu-shared", "nodes": 10, "state": "up", "cpus_per_node": 16, "mem_per_node_mb": 196608, "gpu_types": ["a100", "v100"], "has_gpu": True, "timelimit": "04:00:00", "is_public": True},
    {"name": "gpu-highend", "nodes": 4, "state": "up", "cpus_per_node": 32, "mem_per_node_mb": 262144, "gpu_types": ["h100"], "has_gpu": True, "timelimit": "24:00:00", "is_public": True},
    {"name": "debug", "nodes": 2, "state": "up", "cpus_per_node": 8, "mem_per_node_mb": 32768, "gpu_types": [], "has_gpu": False, "timelimit": "01:00:00", "is_public": True},
]

MOCK_CONDA_ENVS = ["base", "pytorch", "tensorflow", "jax", "my_project"]

MOCK_GPU_TYPES = ["a100", "h100", "v100", "a40", "rtx6000", "h200", "l40s"]

MOCK_MODULES = ["python/anaconda", "cuda/11.8", "cuda/12.1", "gcc/9.3.0", "openmpi/4.1.1"]

MOCK_ACCOUNTS = ["my_lab", "training", "default"]


_RUN_TIMEOUT = 30


def _run_command(cmd: list[str], timeout: int = _RUN_TIMEOUT) -> tuple[str, str, int]:
    try:
        # Force UTF-8 decoding with a lossy fallback: under a C/POSIX locale
        # `text=True` would otherwise decode with ASCII and raise on any
        # non-ASCII byte in the command output (crashing the wizard/batch run).
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", -1
    except OSError as e:
        # A Slurm binary that is present but not runnable (bad arch, permission,
        # missing loader) raises here rather than being caught by shutil.which;
        # return a non-zero rc so callers fall back to mock data instead of
        # crashing with a traceback.
        return "", str(e), -1


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
    # `sinfo %m` (without -e) reports the minimum node memory with a trailing
    # "+" when a partition's nodes differ (e.g. "515000+"). Strip it so the
    # min value is used, mirroring how _safe_int already tolerates "+" for %c —
    # otherwise the memory-over-limit warning is silently disabled for every
    # heterogeneous partition.
    value = raw.strip().upper().rstrip("+")
    if not value or value == "0":
        return 0
    match = re.match(r"^(\d+(?:\.\d+)?)([KMGTP])(?:[NC])?$", value)
    if match:
        num = float(match.group(1))
        scale = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2, "P": 1024 ** 3}
        mb = int(num * scale[match.group(2)])
        # A positive size below 1 MB (e.g. "1K") would truncate to 0 and read as
        # "unknown"; clamp to 1 MB so it stays a real, if tiny, value.
        return mb if mb > 0 or num == 0 else 1
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
# Minute/second fields are range-limited to [0-5]\d (00–59) so obviously
# out-of-range values like "1:60:60" or "1-99:99:99" are rejected client-side,
# not just at submit. (Lead fields stay \d+ / \d{1,2}: hours in hh:mm:ss can
# legitimately exceed 24, and Slurm accepts a bare "0" as "no limit".)
_TIME_PATTERNS = (
    r"^\d+$",                          # minutes
    r"^\d+:[0-5]\d$",                  # minutes:seconds
    r"^\d+:[0-5]\d:[0-5]\d$",          # hours:minutes:seconds
    r"^\d+-\d{1,2}$",                  # days-hours
    r"^\d+-\d{1,2}:[0-5]\d$",          # days-hours:minutes
    r"^\d+-\d{1,2}:[0-5]\d:[0-5]\d$",  # days-hours:minutes:seconds
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
    # Already has unit: return as-is, but drop any trailing Slurm N/C suffix —
    # `sbatch --mem` accepts only a K/M/G/T unit, so "16GN" would be rejected.
    m = re.match(r"^(\d+(?:\.\d+)?)([KMGTP])(?:[NC])?$", v)
    if m:
        # `sbatch --mem` requires an INTEGER magnitude, so a fractional value like
        # "1.5G" — which validate_memory accepts — would be rejected at submit.
        # Convert it to whole megabytes ("1.5G" -> "1536M") so a value that
        # validates always normalizes to a directive Slurm accepts.
        if "." in m.group(1):
            return f"{_parse_mem_to_mb(v)}M"
        return f"{m.group(1)}{m.group(2)}"
    # Invalid but return it anyway (validation should catch this)
    return v


# A token shaped like a GPU model name: a known GPU-family letter prefix
# immediately followed by a digit (a100, h100/h200, v100, l40s, t4, p100, k80,
# b200, mi250, gh200, gb200, rtx6000/gtx…). Deliberately does NOT match bare CPU
# tokens like "i7"/"gold6248"/"avx512" (their prefixes aren't GPU families).
_GPU_MODEL_RE = re.compile(
    r"^(?:a|h|v|l|t|p|k|b|rtx|gtx|mi|gh|gb|quadro|tesla)\d", re.IGNORECASE
)

# CPU-generation tags that share a GPU-family letter prefix and would otherwise
# be misread as a GPU model: Intel Xeon "vN" (E5/E7-…-v2…v6) and IBM POWER "pN"
# (POWER8/9/10). No real GPU uses these exact tokens — V100/P100 etc. are
# multi-digit — so excluding them is safe.
_CPU_GEN_TOKENS = frozenset({"v2", "v3", "v4", "v5", "v6", "p8", "p9", "p10"})


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

    # Positive match: a token shaped like a GPU model name (a100, h100, v100,
    # l40s, t4, p100, k80, rtx6000, mi250, gh200, b200, quadro/tesla…). This is
    # far more reliable than negative filtering and, crucially, wins over a CPU
    # vendor/codename token that happens to appear first in the features list.
    for token in tokens:
        if token.lower() in _CPU_GEN_TOKENS:
            continue
        # Apply the same length sanity cap as the negative branch below, so a
        # pathologically long feature token (e.g. a concatenated garbage string)
        # can't be returned verbatim as a GPU "model".
        if len(token) >= 15:
            continue
        if _GPU_MODEL_RE.match(token):
            return token

    # Fall back to negative filtering: reject obvious non-GPU tokens and return
    # the first plausible one. Never drops a real GPU type that lives only in
    # the features string.
    for token in tokens:
        if not re.match(r"[a-zA-Z]", token):
            continue
        if token.lower() in _CPU_GEN_TOKENS:
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
            # Bare CPU vendor / microarch-codename tokens clusters put in node
            # features, ahead of the GPU model. Without these the negative
            # filter would return e.g. "intel"/"rome" as the GPU type.
            "intel", "amd", "arm",
            "rome", "milan", "genoa", "naples", "cascade",
            "sandybridge", "ivybridge", "nehalem", "westmere",
            # Spelled-out IBM POWER CPUs (the short p8/p9/p10 forms are in
            # _CPU_GEN_TOKENS; "power9" etc. would otherwise pass as a GPU model).
            "power", "power8", "power9", "power10",
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
        parts = rest.split(":")
        hours = _safe_int(parts[0])
        minutes = _safe_int(parts[1]) if len(parts) > 1 else 0
        seconds = _safe_int(parts[2]) if len(parts) > 2 else 0
        return _safe_int(day_part) * 1440 + hours * 60 + minutes + seconds / 60.0
    parts = value.split(":")
    if len(parts) == 3:
        return _safe_int(parts[0]) * 60 + _safe_int(parts[1]) + _safe_int(parts[2]) / 60.0
    if len(parts) == 2:
        return _safe_int(parts[0]) + _safe_int(parts[1]) / 60.0
    return float(_safe_int(parts[0])) if parts else 0.0


def validate_job_config(
    answers: dict[str, Any],
    extra_gpu_types: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Validate a (possibly incomplete) answers dict against the selected
    partition's advertised capabilities.

    Returns a list of ``(level, message)`` tuples, where ``level`` is:

    - ``"error"``   — a configuration Slurm will reject outright (e.g. GPUs on a
      CPU-only partition, or a GPU model the partition doesn't have).
    - ``"warning"`` — a request that exceeds a node's advertised capacity and may
      be rejected or left pending (CPU/memory/time over the per-node limit; the
      advertised value can undercount a heterogeneous partition, so it isn't a
      guaranteed failure).

    An empty list means nothing is known to be wrong. Both the live TUI check
    (every redraw) and the final CLI summary share this single source of truth,
    so the two surfaces can't drift apart.

    This function is pure and side-effect free — it makes **no** subprocess
    calls — so the TUI can safely call it on every keystroke/redraw. Callers
    that can afford a live ``sinfo`` lookup (e.g. the one-shot CLI summary) may
    pass ``extra_gpu_types`` to widen the set of GPU models considered valid
    beyond what ``_partition_obj`` statically lists.
    """
    part = answers.get("_partition_obj")
    if not part:
        return []
    out: list[tuple[str, str]] = []

    # CPUs — compare the per-node total (cpus-per-task x tasks-per-node) against
    # the node's core count, so multi-task over-allocation is caught.
    cpus = answers.get("cpus")
    if cpus is not None and str(cpus).strip() != "":
        try:
            cores = int(cpus)
            ntpn_raw = answers.get("ntasks_per_node")
            ntpn = int(ntpn_raw) if ntpn_raw else 1
            total = cores * max(1, ntpn)
            limit = part.get("cpus_per_node", 0)
            if limit and total > limit:
                detail = f"{ntpn}×{cores}={total}" if ntpn > 1 else str(total)
                out.append(("warning", f"CPUs ({detail}) exceeds partition limit ({limit} per node)"))
        except (ValueError, TypeError):
            pass

    # Memory vs the node's advertised memory.
    memory = answers.get("memory")
    if memory and validate_memory(str(memory)):
        mb = _parse_mem_to_mb(str(memory))
        limit = part.get("mem_per_node_mb", 0)
        if limit and mb > limit:
            out.append(("warning", f"Memory ({memory}) exceeds partition limit ({limit} MB per node)"))

    # Time vs the partition's max time.
    time_limit = answers.get("time_limit")
    if time_limit:
        try:
            req_mins = _parse_slurm_time_to_minutes(str(time_limit))
            limit_str = part.get("timelimit")
            if limit_str:
                limit_mins = _parse_slurm_time_to_minutes(limit_str)
                if limit_mins > 0 and req_mins > limit_mins:
                    out.append(("warning", f"Time limit ({time_limit}) exceeds partition limit ({limit_str})"))
        except Exception:
            pass

    # GPUs requested on a partition *known* to advertise none. Only assert this
    # when ``has_gpu`` is explicitly False: real partition objects (from
    # fetch_partitions / MOCK) always carry it as a bool, so ``is False`` means
    # "we looked and there's no gpu GRES" — a config Slurm will reject. A
    # manually-typed or unrecognized partition falls back to a synthetic object
    # with no ``has_gpu`` key (capability unknown, like the 0/None cpu/mem/time
    # limits the checks above stay silent on), so we must not overclaim a hard
    # "no GPUs" error there. ``has_gpu`` also stays True for count-only
    # ("gpu:4") / typed-without-count GRES that don't populate gpu_types, so a
    # real GPU partition is never flagged as CPU-only.
    gpus = answers.get("gpus", 0)
    try:
        gpus_val = int(gpus) if (gpus is not None and str(gpus).strip() != "") else 0
    except (ValueError, TypeError):
        gpus_val = 0
    gpu_types = list(part.get("gpu_types", []))
    if gpus_val > 0 and not gpu_types and part.get("has_gpu") is False:
        out.append(("error", f"Partition '{part.get('name')}' does not support GPUs"))

    # A specific GPU model the partition doesn't offer. Only meaningful when we
    # actually know which models the partition has (static list plus any
    # caller-supplied dynamic types); with no type info at all, the count-only
    # "does not support GPUs" check above is the right signal, and warning
    # "not in partition list ()" against an empty list would be noise.
    gpu_type = answers.get("gpu_type")
    if gpu_type and str(gpu_type).lower() != "any":
        all_types = gpu_types + [t for t in (extra_gpu_types or []) if t not in gpu_types]
        known = {str(g).lower() for g in all_types}
        if known and str(gpu_type).lower() not in known:
            out.append(("error", f"GPU type '{gpu_type}' not in partition list ({', '.join(all_types)})"))

    # Memory-per-core advisory: on shared partitions that bill max(cores, memory-
    # fraction), asking for markedly more memory per core than the node provides
    # means the job is allocated/billed for more cores than requested. The site's
    # billing model is unknown here, so this is a soft warning that fires only when
    # the request exceeds 1.5x the node's per-core memory — a normal, roughly
    # proportional request (incl. typical defaults) stays silent.
    try:
        cpus_v = answers.get("cpus")
        mem_v = answers.get("memory")
        node_cpus = part.get("cpus_per_node", 0)
        node_mem = part.get("mem_per_node_mb", 0)
        if (cpus_v is not None and str(cpus_v).strip() and mem_v
                and validate_memory(str(mem_v)) and node_cpus and node_mem):
            ntpn_raw = answers.get("ntasks_per_node")
            ntpn = int(ntpn_raw) if ntpn_raw else 1
            total_cpus = max(1, int(cpus_v) * max(1, ntpn))
            mb = _parse_mem_to_mb(str(mem_v))
            per_core_node = node_mem / node_cpus
            if mb and per_core_node and (mb / total_cpus) > per_core_node * 1.5:
                implied = math.ceil(mb / per_core_node)
                out.append((
                    "warning",
                    f"Memory ({mem_v}) is well above '{part.get('name', '')}'s per-core "
                    f"memory (~{int(per_core_node)} MB/core); a shared partition may "
                    f"bill ~{implied} cores, not {total_cpus}"
                ))
    except (ValueError, TypeError):
        pass

    return out


def fetch_partitions() -> list[dict[str, Any]]:
    if not is_tool_available("sinfo"):
        # Demo data only under SLURMATE_MOCK; on a real cluster whose sinfo is
        # missing/unrunnable, return nothing (the picker lets the user type a
        # name) rather than fake partitions that can't be submitted to.
        return list(MOCK_PARTITIONS) if _force_mock() else []

    stdout, _, rc = _run_command(
        ["sinfo", "-h", "-o", "%P|%l|%D|%a|%c|%m|%G"]
    )
    if rc != 0:
        return []

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
        has_gpu = False
        if gres_raw and gres_raw != "(null)":
            for match in re.finditer(r"gpu:([a-zA-Z0-9._-]+):\d+", gres_raw, re.IGNORECASE):
                gpu_types.append(match.group(1).replace("_", "-"))
            # Detect GPU presence even for count-only ("gpu:4") or typed-without-
            # count ("gpu:a100") GRES that the model regex above doesn't capture,
            # so a real GPU partition isn't misreported as CPU-only downstream.
            has_gpu = bool(re.search(r"gpu[:\d]", gres_raw, re.IGNORECASE))

        if name not in partitions:
            partitions[name] = {
                "name": name,
                # Sum node counts: sinfo emits one row per partition+state group,
                # so a partition with idle/mix/alloc nodes spans several rows;
                # max() would report only the largest single group's count.
                "nodes": nodes,
                "state": state,
                "cpus_per_node": cpus,
                "mem_per_node_mb": _parse_mem_to_mb(mem_raw) if mem_raw else 0,
                "gpu_types": gpu_types,
                "has_gpu": has_gpu,
                "timelimit": timelimit if timelimit != "infinite" else None,
            }
        else:
            p = partitions[name]
            p["nodes"] += nodes
            # cpus/mem are per-node capacities — keep the max across configs.
            p["cpus_per_node"] = max(p["cpus_per_node"], cpus)
            mem_mb = _parse_mem_to_mb(mem_raw) if mem_raw else 0
            p["mem_per_node_mb"] = max(p["mem_per_node_mb"], mem_mb)
            p["gpu_types"] = list(set(p["gpu_types"] + gpu_types))
            p["has_gpu"] = p["has_gpu"] or has_gpu

    return list(partitions.values())


def fetch_public_partitions(all_parts: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return only publicly-usable partitions.

    Pass ``all_parts`` (a prior ``fetch_partitions()`` result) to avoid a
    redundant ``sinfo`` call — the partition step fetches it once and shares it.
    """
    if not is_tool_available("sinfo") or not is_tool_available("scontrol"):
        return [p for p in MOCK_PARTITIONS if p.get("is_public")] if _force_mock() else []

    stdout, _, rc = _run_command(["scontrol", "show", "partition", "-o"])
    if rc != 0:
        return []

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
        state = _extract_token(scontrol_line, "State")

        # "Public" = usable by anyone: open to all accounts, not hidden, and up.
        # (AllowGroups gating can't be evaluated here without the caller's groups;
        # such partitions still appear under the picker's "[Private]"/"[Custom]"
        # paths, so nothing usable is truly hidden — only mis-ranked.)
        is_public = (
            allow_accounts.upper() == "ALL"
            and hidden.upper() != "YES"
            and state.upper() in ("", "UP")
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
    """Fetch all QoS names known to the system via sacctmgr.

    Returns the demo ``MOCK_QOS`` only in mock mode. When sacctmgr is genuinely
    unavailable (or errors, or lists nothing), returns ``[]`` — an *unknown* set,
    not the demo names — so the TUI can tell "QoS set unknown" apart from a real
    list and skip filtering live ``AllowQos`` against a demo fallback (which
    would otherwise silently drop real, lab-specific QoS names).
    """
    if _force_mock():
        return list(MOCK_QOS)
    if not is_tool_available("sacctmgr"):
        return []

    stdout, _, rc = _run_command(
        ["sacctmgr", "show", "qos", "-P", "format=Name", "--noheader"]
    )
    if rc != 0:
        return []

    qos: list[str] = []
    for line in stdout.splitlines():
        name = line.strip()
        if name:
            qos.append(name)
    return qos


def fetch_gpu_types_for_partition(partition: str) -> list[str]:
    if not is_tool_available("sinfo"):
        if not _force_mock():
            return []
        # In mock mode, prefer the specific partition's GPU types so a demo
        # doesn't claim every partition offers all GPU models; fall back to the
        # full list only for an unknown/manually-typed partition name.
        for p in MOCK_PARTITIONS:
            if p["name"] == partition:
                return [str(g) for g in p["gpu_types"]]
        return list(MOCK_GPU_TYPES)

    stdout, _, rc = _run_command(
        ["sinfo", "-h", "-N", "-p", partition, "-o", "%f|%G"]
    )
    if rc != 0:
        return []

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

    # Pass 2: collect every typed model on each node (a node can advertise more
    # than one, e.g. "gpu:a100:2,gpu:v100:2" — a single re.search would drop the
    # second). Only when a node has no typed model do we fall back to feature
    # scanning, preferring corroboration against the typed models seen elsewhere.
    types: set[str] = set()
    for features, gres in lines_data:
        text = f"{features},{gres}"
        typed_here = [
            m.group(1).replace("_", "-")
            for m in re.finditer(r"gpu:([a-z0-9._-]+):\d+", text, re.IGNORECASE)
            if m.group(1).lower() not in {"gpu", "mps", "shard"}
        ]
        if typed_here:
            types.update(typed_here)
            continue
        gpu_type = _detect_gpu_type(features, gres, known_models=typed_models)
        if gpu_type and gpu_type != "gpu":
            types.add(gpu_type)
    return sorted(types)


def _extract_first_json(text: str) -> Any:
    """Return the first parseable JSON object in ``text``, or None.

    A login shell may print a banner before the JSON, and that banner can itself
    contain braces — so a naive first-``{``/last-``}`` slice can capture garbage.
    Walk each ``{`` and try to decode from there, tolerating trailing output.
    """
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            return None
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            idx = start + 1


def fetch_conda_envs(modules: list[str] | None = None) -> list[str]:
    """List conda environment names/paths usable with ``conda activate``.

    Conda is frequently provided by a module (e.g. ``module load anaconda``)
    rather than being on ``PATH`` directly, so when ``modules`` are given we load
    them first — inside a login shell where ``module`` is defined — and then run
    ``conda info --json``. Using ``info`` (not ``env list``) gives the authoritative
    ``root_prefix`` and ``envs_dirs``, so the base env is labelled ``base`` (not by
    its install-dir basename) and a ``--prefix`` env outside the envs dirs is kept
    as a full path (activatable), instead of a bare basename that can't activate.
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
        ["bash", "-lc", f"{prefix}conda info --json 2>/dev/null"]
    )
    if rc != 0:
        # Real failure (conda/module not found): return nothing rather than
        # misleading mock names so the user can just type their env/path.
        return []

    data = _extract_first_json(stdout)
    if not isinstance(data, dict):
        return []
    root = str(data.get("root_prefix", "")).rstrip("/")
    envs_dirs = {str(d).rstrip("/") for d in data.get("envs_dirs", []) if d}
    env_names: list[str] = []
    for raw_env in data.get("envs", []):
        p = str(raw_env).rstrip("/")
        if not p:
            continue
        if root and p == root:
            env_names.append("base")
        elif os.path.dirname(p) in envs_dirs:
            # A named env under an envs dir — activatable by its basename.
            env_names.append(os.path.basename(p))
        else:
            # A --prefix env elsewhere — only the full path activates it.
            env_names.append(p)
    # De-dup while preserving order.
    return list(dict.fromkeys(env_names))


def fetch_available_modules() -> list[str]:
    """Parse `module avail` output into a sorted unique list of module names."""
    # Mirror every other fetcher: never shell out (into a login shell that
    # sources the user's profile) when mock mode is forced.
    if _force_mock():
        return list(MOCK_MODULES)

    stdout, stderr, rc = _run_command(["bash", "-lc", "command -v module && module -t avail 2>&1"])
    output = stdout + stderr
    if rc != 0:
        return []

    modules: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-"):
            continue
        # `module -t avail` prints filesystem headers ("/opt/apps/modulefiles:")
        # on their own lines; skip them so they don't pollute the module list.
        if stripped.endswith(":"):
            continue
        for mod in stripped.split():
            # Strip "(default)" annotation that the module system appends
            if mod.endswith("(default)"):
                mod = mod[:-9].strip()
            # Lmod terse output can carry extras a Tcl-modules parser wouldn't: an
            # alias annotation "(@name)", a tag marker like "(D)"/"<F>", and a
            # trailing "/" on a family short-name ("gcc/" — loadable as "gcc").
            if mod.startswith("(@") or (mod.startswith("<") and mod.endswith(">")) \
                    or (mod.startswith("(") and mod.endswith(")")):
                continue
            mod = mod.rstrip("/")
            # Drop the leading `command -v module` probe output — either the
            # bare "module" function name or its resolved path (/usr/bin/module).
            if not mod or mod == "module" or mod.endswith("/module"):
                continue
            modules.add(mod)
    return sorted(modules)


def _current_username() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER") or os.environ.get("LOGNAME") or ""


def fetch_user_accounts() -> list[str]:
    """Fetch the Slurm accounts the current user may submit under.

    Uses ``sacctmgr show assoc user=<me>`` (associations), NOT ``show user``:
    the bare ``user`` entity doesn't populate ``Account`` and isn't scoped to
    the caller (it lists every visible user), so it returns thousands of blank
    lines on a real cluster and the picker silently falls back to mock accounts
    the user can't actually charge to.
    """
    if not is_tool_available("sacctmgr"):
        # Demo accounts only under SLURMATE_MOCK. On a real cluster without
        # sacctmgr, return nothing rather than fake accounts the user can't
        # charge to — the account field is free-text, so they type their own.
        return list(MOCK_ACCOUNTS) if _force_mock() else []

    user = _current_username()
    if not user:
        return []

    stdout, _, rc = _run_command(
        ["sacctmgr", "show", "assoc", f"user={user}", "-P",
         "format=Account", "--noheader"]
    )
    if rc != 0:
        return []

    accounts: list[str] = []
    for line in stdout.splitlines():
        a = line.strip()
        if a:
            accounts.append(a)
    # De-dupe while preserving order; a user is often associated to the same
    # account through several partitions/QoS, yielding duplicate rows.
    accounts = list(dict.fromkeys(accounts))
    return accounts


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
            # sinfo %t can append status flags to the base state (idle*, idle~,
            # mix#, …: not-responding / power-save / powering-up / maintenance
            # etc.); strip them so nodes aren't dropped from the idle/mix tally.
            state_flag = parts[2].strip().rstrip("*~#!%$@+")
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
    # Create the log directories the script's #SBATCH --output/--error point at,
    # so Slurm doesn't fail the job on a missing directory.
    for line in script_content.splitlines():
        val = _sbatch_log_path(line)
        if not val:
            continue
        dir_name = os.path.dirname(os.path.expanduser(val))
        # Skip a directory component that carries a Slurm filename pattern
        # (%j/%A/%a/%x): those are expanded per-job by Slurm, so creating a
        # literal "%j" directory here would be wrong.
        if dir_name and "%" not in dir_name:
            try:
                os.makedirs(dir_name, exist_ok=True)
            except OSError as e:
                logger.debug(f"Failed to create log directory {dir_name}: {e}")

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
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return -1, "", "Submission timed out after 30s"
    except OSError as e:
        return -1, "", f"Could not run sbatch: {e}"

    if result.returncode != 0:
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    return result.returncode, result.stdout.strip(), ""


def _sbatch_log_path(line: str) -> str:
    """Extract the path from a ``#SBATCH --output=/-o`` or ``--error=/-e`` line.

    Handles both the long ``--output=PATH`` form and the short ``-o PATH`` form,
    strips surrounding quotes, and returns "" for anything else (or a blank
    short-form directive, which must not raise).
    """
    s = line.strip()
    val = ""
    if s.startswith("#SBATCH --output=") or s.startswith("#SBATCH --error="):
        val = s.split("=", 1)[1].strip()
    elif s.startswith("#SBATCH -o ") or s.startswith("#SBATCH -e "):
        parts = s.split(None, 2)
        val = parts[2].strip() if len(parts) > 2 else ""
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        val = val[1:-1]
    return val


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


def _split_top_level_commas(s: str) -> list[str]:
    """Split ``s`` on commas that sit outside single/double quotes.

    A raw ``str.split(',')`` shreds a quoted array element that contains a comma
    (e.g. ``"--constraint=a,b"``) into bogus tokens with dangling quotes; this
    keeps such elements intact, matching how a real TOML parser reads the array.
    """
    items: list[str] = []
    buf: list[str] = []
    in_single = in_double = False
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == "," and not in_single and not in_double:
            items.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    items.append("".join(buf))
    return items


def _has_unquoted_char(s: str, target: str) -> bool:
    """True if ``target`` appears in ``s`` outside single/double quotes."""
    in_single = in_double = False
    for ch in s:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == target and not in_single and not in_double:
            return True
    return False


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
        return [_coerce_scalar(x.strip()) for x in _split_top_level_commas(inner) if x.strip()]
    return _coerce_scalar(v)


def _parse_config_naive(text: str) -> dict[str, Any]:
    """Minimal key=value parser used only when no TOML library is available.

    Best-effort, but section- and array-aware so it doesn't silently disagree
    with the real TOML reader: it tracks ``[section]`` headers, applies the same
    ``[slurmate] > [defaults] > top-level`` precedence as :func:`_flatten_config`,
    and accumulates a multi-line ``key = [`` array until its closing ``]``.
    """
    top: dict[str, Any] = {}
    sections: dict[str, dict[str, Any]] = {}
    current: str | None = None
    pending_key: str | None = None
    pending_parts: list[str] = []

    def store(key: str, raw_value: str) -> None:
        target = sections.setdefault(current, {}) if current else top
        target[key] = _coerce_config_value(raw_value)

    for raw in text.splitlines():
        line = raw.strip()
        if pending_key is not None:
            # Strip a trailing comment from THIS physical line (TOML comments are
            # line-oriented). Doing it once on the joined text would let an
            # interior line's "#" swallow the rest of the array.
            pending_parts.append(_strip_inline_comment(line))
            # Only a "]" outside quotes closes the array; a "]" inside a string
            # element (e.g. "--constraint=a]b") must not terminate it early.
            if _has_unquoted_char(" ".join(pending_parts), "]"):
                store(pending_key, " ".join(pending_parts))
                pending_key, pending_parts = None, []
            continue
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = _strip_inline_comment(v.strip())
            # A multi-line array (`key = [` with no closing `]` on this line).
            if v.startswith("[") and not _has_unquoted_char(v, "]"):
                pending_key, pending_parts = k, [v]
                continue
            store(k, v)

    # An array that opened but never closed: don't silently drop it (and every
    # subsequent line accumulated into it). Warn, matching the tomllib path,
    # which raises + surfaces a "warning: ignoring config" message in load_config.
    if pending_key is not None:
        import sys
        print(
            f"slurmate: warning: unclosed array for '{pending_key}' in the "
            f"configuration file — ignoring it",
            file=sys.stderr,
        )

    config: dict[str, Any] = dict(top)
    for section in ("defaults", "slurmate"):
        if section in sections:
            config.update(sections[section])
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
            # The file exists but couldn't be parsed/read (e.g. a TOML syntax
            # error or a permission problem). Surface it: otherwise every
            # configured default is silently dropped with no hint to the user.
            import sys
            print(
                f"slurmate: warning: ignoring configuration file {p} — {e}",
                file=sys.stderr,
            )
            logger.debug(f"Failed to load config from {p}: {e}")
            return {}
    return {}
