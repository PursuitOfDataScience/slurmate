"""An MCP server, so this works in agents that have never heard of slurmate.

`SKILL.md` is a Claude Code convention and `AGENTS.md` is a file the agent has
to be told to read. Neither is a protocol: both rely on the agent noticing a
document and then shelling out correctly. The Model Context Protocol is the
part that is actually standard, and it is what Claude Code, Cursor, Windsurf,
Zed, Continue and the rest already speak, so a tool that speaks it is
discoverable rather than documented. The agent gets typed schemas, argument
validation and a tool list it can enumerate, none of which a markdown file can
give it.

Implemented directly against the wire format rather than against the `mcp`
package, and that is a deliberate trade. The transport is newline-delimited
JSON-RPC 2.0 over stdin and stdout, the server half of which is the four
methods below; taking the SDK would add slurmate's fifth runtime dependency
(and a heavy one) to a package whose whole promise is that ``pipx install
slurmate`` works on a login node with no build tools. The cost is that a
future protocol revision has to be followed by hand, which is why
:data:`PROTOCOL_VERSION` is echoed rather than asserted.

**stdout belongs to the protocol.** Nothing else may write a byte to it while
this is running, which is the same discipline ``--print`` and ``--json``
already keep; every diagnostic in this module goes to stderr.

Read-only by construction. There is no submit tool and there will not be one:
an agent that can spend a user's allocation unattended is a different product
from one that can read a cluster and check a script.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

from . import __version__
from .agent import (
    SCHEMA_VERSION,
    check_request,
    cluster_brief,
    job_report,
    node_inventory,
    parse_sbatch_script,
)
from .system_utils import cluster_cache_age, reset_cluster_cache

#: The revision this server was written against. A client that asks for a
#: different one is answered with its own string rather than corrected: the
#: methods here have been stable across every revision that has them, and
#: refusing an unfamiliar version would break a client that would have worked.
PROTOCOL_VERSION = "2025-06-18"

#: How long a memoised cluster fact may be served by this server before it is
#: re-read. The facts in that cache change on a human timescale (a partition
#: added, an account granted) and cost a `sacctmgr` round trip each, so the
#: number wants to be long enough that a burst of tool calls pays for them
#: once and short enough that a change lands within a coffee break.
CACHE_TTL_SECONDS = 300.0

#: JSON-RPC's own codes, which are not ours to choose.
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INTERNAL_ERROR = -32603

_PARTITION = {"type": "string", "description": "Partition name"}

#: The job settings both `slurm_check_job` and `slurm_generate_script` accept,
#: declared once. Two copies would drift, and the pair that drifted would be a
#: script this tool generated and then could not check.
_CHECK_FIELDS: dict[str, Any] = {
    "partition": _PARTITION,
    "account": {"type": "string"},
    "qos": {"type": "string"},
    "cpus": {"type": "integer", "description": "--cpus-per-task"},
    "memory": {"type": "string", "description": "--mem, e.g. 32G"},
    "mem_per_cpu": {"type": "string"},
    "time_limit": {"type": "string", "description": "--time, e.g. 04:00:00"},
    "nodes": {"type": "integer"},
    "ntasks_per_node": {"type": "integer"},
    "gpus": {"type": "integer"},
    "gpu_type": {"type": "string", "description": "e.g. a100"},
    "gpu_format": {
        "type": "string",
        "enum": ["gres_type", "constraint", "gpus", "gpus_per_node", "gpus_per_task"],
        "description": "How to spell the GPU request. Check gpu_request_formats in "
                       "the brief first: a model that is feature-only needs "
                       "'constraint', and a typed --gres for it is rejected.",
    },
    "constraint": {"type": "string", "description": "Node feature, Slurm -C"},
    "array_spec": {"type": "string", "description": "--array, e.g. 1-10"},
    "modules": {"type": "array", "items": {"type": "string"},
                "description": "Modules to load, checked against this cluster"},
    "env_name": {"type": "string", "description": "conda env name or venv path"},
    "env_type": {"type": "string", "enum": ["conda", "mamba", "venv", "none"]},
    "output_file": {"type": "string"},
    "command": {"type": "string", "description": "The command the job runs"},
}


def tool_schemas() -> list[dict[str, Any]]:
    """The tools this server offers, as MCP tool definitions.

    The descriptions are written for the model that reads them, not for a
    person browsing a reference: each says when to reach for the tool, because
    that is the decision the description is actually load-bearing for. The
    ordering is the order an agent should use them in.
    """
    return [
        {
            "name": "slurm_cluster_brief",
            "description": (
                "Read the whole Slurm cluster in one call: its name and version, "
                "the partitions you can submit to with their capacity and queue "
                "depth, your accounts, and how GPUs must be requested here. "
                "ALWAYS call this before writing or editing an sbatch script on "
                "an unfamiliar cluster. If the result has mock=true, no cluster "
                "was reachable and nothing in it is real."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "full": {
                        "type": "boolean",
                        "description": "Also return per-node maxima, GPU request "
                                       "formats for every partition, and the module "
                                       "and conda environment lists. Slower.",
                    },
                    "all_partitions": {
                        "type": "boolean",
                        "description": "Include partitions your accounts cannot "
                                       "submit to.",
                    },
                    "partition": dict(_PARTITION, description="Report only this partition"),
                },
            },
        },
        {
            "name": "slurm_list_nodes",
            "description": (
                "What machines the cluster has, grouped by cores, memory, GPUs and "
                "node features, with how many of each are usable. Use this to "
                "choose a --constraint or a --gres, or to answer what hardware is "
                "available; slurm_cluster_brief aggregates to the partition and "
                "cannot tell two node generations apart."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "partition": dict(_PARTITION,
                                      description="Only nodes reachable through this partition"),
                    "gpu_only": {"type": "boolean", "description": "Only GPU nodes"},
                },
            },
        },
        {
            "name": "slurm_list_jobs",
            "description": (
                "The user's jobs and their state. For each PENDING job it returns "
                "Slurm's own reason code plus what that code means to do next, "
                "which is the answer to 'why is my job not running'. The codes are "
                "not interchangeable: some clear on their own, some never will."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "user": {"type": "string",
                             "description": "Another user's jobs (default: yours)"},
                },
            },
        },
        {
            "name": "slurm_check_job",
            "description": (
                "Say whether this cluster would accept a job, WITHOUT submitting "
                "it. Pass either a whole sbatch script or the individual settings. "
                "Runs every slurmate validator plus 'sbatch --test-only', which "
                "enters no queue and spends no allocation, so it is safe to call "
                "repeatedly. ALWAYS call this on a script before giving it to the "
                "user. Reads 'module load' and 'conda activate' out of the script "
                "body, not just the #SBATCH block."
            ),
            "inputSchema": {
                "type": "object",
                "properties": dict(
                    _CHECK_FIELDS,
                    script={"type": "string",
                            "description": "A complete sbatch script. Takes "
                                           "precedence over the fields below."},
                ),
            },
        },
        {
            "name": "slurm_generate_script",
            "description": (
                "Build an sbatch script from job settings, using this cluster's "
                "own GPU syntax and defaults. Returns the script text and the "
                "same findings slurm_check_job would give, so one call both "
                "writes and validates. It does NOT submit: the user runs sbatch."
            ),
            "inputSchema": {
                "type": "object",
                "properties": dict(
                    _CHECK_FIELDS,
                    job_name={"type": "string"},
                    output_dir={"type": "string"},
                ),
                "required": ["command"],
            },
        },
    ]


def _answers(args: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments as the answers dict every validator already reads."""
    known = (
        "job_name", "partition", "account", "qos", "cpus", "memory",
        "mem_per_cpu", "time_limit", "nodes", "ntasks_per_node", "gpus",
        "gpu_type", "gpu_format", "constraint", "array_spec", "modules",
        "env_name", "env_type", "output_dir", "output_file", "command",
    )
    return {k: args[k] for k in known if args.get(k) not in (None, "")}


def _tool_check(args: dict[str, Any]) -> dict[str, Any]:
    script = args.get("script")
    if script:
        answers, extra = parse_sbatch_script(str(script))
        return check_request(answers, script=str(script), extra=extra)
    return check_request(_answers(args))


def _tool_generate(args: dict[str, Any]) -> dict[str, Any]:
    from .builder import build_from_answers

    answers = _answers(args)
    # partial=False: the caller asked for a script to run, so the defaults
    # that make it submittable belong in it. `check` uses partial=True for the
    # opposite reason, that a refusal earned by an unasked-for default is a
    # finding about slurmate rather than about the request.
    script = build_from_answers(answers, partial=False)
    result = _tool_check({"script": script})
    result["script"] = script
    return result


TOOLS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "slurm_cluster_brief": lambda a: cluster_brief(
        full=bool(a.get("full")), all_partitions=bool(a.get("all_partitions")),
        partition=a.get("partition"),
    ),
    "slurm_list_nodes": lambda a: node_inventory(
        partition=a.get("partition"), gpu_only=bool(a.get("gpu_only")),
    ),
    "slurm_list_jobs": lambda a: job_report(a.get("user")),
    "slurm_check_job": _tool_check,
    "slurm_generate_script": _tool_generate,
}


def _result(payload: dict[str, Any]) -> dict[str, Any]:
    """One tool result, in both of MCP's shapes.

    ``content`` is the text every client renders; ``structuredContent`` is the
    parsed object newer ones prefer. Sending both costs one serialisation and
    means a client on either side of that revision gets the usable form rather
    than the fallback.
    """
    text = json.dumps(payload, indent=2, default=str)
    return {
        "content": [{"type": "text", "text": text}],
        "structuredContent": payload,
        "isError": False,
    }


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """One JSON-RPC request in, one response out, or None for a notification.

    Pure, so the whole protocol surface is testable without a subprocess and
    without a pipe: every case below is one dict in and one dict out.
    """
    method = message.get("method")
    ident = message.get("id")
    params = message.get("params") or {}

    # A notification has no id and takes no reply, ever. Answering one is a
    # protocol violation that some clients treat as a fatal desync.
    if ident is None and str(method).startswith("notifications/"):
        return None

    def ok(result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": ident, "result": result}

    def err(code: int, msg: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": msg}}

    if method == "initialize":
        asked = str(params.get("protocolVersion") or PROTOCOL_VERSION)
        return ok({
            # Echo the client's version when it named one. These methods have
            # been stable across every revision that has them, so refusing an
            # unfamiliar string would break a client that would have worked.
            "protocolVersion": asked,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": "slurmate",
                "version": __version__,
                "title": "Slurmate: read a Slurm cluster and check sbatch jobs",
            },
            "instructions": (
                "Call slurm_cluster_brief before writing any sbatch script on this "
                "machine, and slurm_check_job before handing one to the user. "
                "Neither submits anything. If a result has mock=true, no cluster "
                "was reachable and nothing in it describes the real site."
            ),
        })

    if method == "ping":
        return ok({})

    if method == "tools/list":
        return ok({"tools": tool_schemas()})

    if method == "tools/call":
        # An MCP server is one process for the length of an editor session,
        # which is the case the per-process cluster cache was explicitly not
        # designed for. Expired here rather than inside the cache so the
        # one-shot CLI keeps exactly the behaviour it was reasoned about with.
        age = cluster_cache_age()
        if age is not None and age > CACHE_TTL_SECONDS:
            reset_cluster_cache()
        name = str(params.get("name") or "")
        tool = TOOLS.get(name)
        if tool is None:
            return err(_METHOD_NOT_FOUND, f"no such tool: {name}")
        try:
            return ok(_result(tool(params.get("arguments") or {})))
        except Exception as exc:  # noqa: BLE001
            # Reported as a tool result, not as a JSON-RPC error: a cluster
            # query that failed is something the model should see and react
            # to, while a transport error is not. Conflating them hides the
            # former behind a client's generic "the tool broke".
            return ok({
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            })

    if ident is None:
        return None
    return err(_METHOD_NOT_FOUND, f"unknown method: {method}")


def serve(stdin: Any = None, stdout: Any = None) -> int:
    """Read newline-delimited JSON-RPC from stdin until it closes."""
    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout
    for line in source:
        text = line.strip()
        if not text:
            continue
        try:
            message = json.loads(text)
        except ValueError:
            reply: dict[str, Any] | None = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": _PARSE_ERROR, "message": "invalid JSON"},
            }
        else:
            if not isinstance(message, dict):
                reply = {
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": _INVALID_REQUEST, "message": "expected an object"},
                }
            else:
                try:
                    reply = handle(message)
                except Exception as exc:  # noqa: BLE001
                    reply = {
                        "jsonrpc": "2.0", "id": message.get("id"),
                        "error": {"code": _INTERNAL_ERROR, "message": str(exc)},
                    }
        if reply is not None:
            sink.write(json.dumps(reply, default=str) + "\n")
            sink.flush()
    return 0


__all__ = ["CACHE_TTL_SECONDS", "PROTOCOL_VERSION", "SCHEMA_VERSION", "TOOLS", "handle", "serve",
           "tool_schemas"]
