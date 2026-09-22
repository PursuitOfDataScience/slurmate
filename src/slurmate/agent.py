"""Machine-readable surfaces: what a program reads before it writes a script.

Everything else in this package renders for a person: ANSI status lines, rich
panels, a full-screen wizard. None of that is consumable by the caller this
module exists for, which is an AI coding agent (or any script) that has just
landed on an unfamiliar cluster and wants to write a correct ``sbatch`` file
without spending twenty minutes rediscovering the site with ad-hoc ``sinfo``
calls, badly.

The knowledge is already here. :mod:`slurmate.system_utils` knows that a
partition of drained nodes advertises capacity nothing can run, that ``sinfo``
marks a heterogeneous partition's CPU and memory figures as *floors* rather
than ceilings, that a GPU model visible only as a node feature cannot be asked
for with ``--gres=gpu:MODEL:N``, and that a log path on node-local ``/tmp``
vanishes while the job reports ``COMPLETED 0:0``. What was missing was a way to
hand any of that to a caller that cannot read a terminal.

Two verbs do it:

``brief``
    One call, whole site. Identity, partitions, the caller's own accounts and
    QoS, how GPUs may be requested here, and whether any of it is real.
``check``
    One call, one verdict. Every structured validator in the package plus
    Slurm's own ``sbatch --test-only``, over a script or a set of flags. No
    queue entry and no allocation spent, so it is safe in a loop.

Nothing here submits. That is deliberate and is the boundary of this module:
an agent that can spend SUs on a billed partition is a different kind of tool,
and the human who owns the allocation should be the one running ``sbatch``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

from .system_utils import (
    _module_command,
    _near_misses,
    check_log_dirs,
    config_source,
    current_username,
    explain_pending,
    fetch_all_partition_names,
    fetch_available_modules,
    fetch_cluster_identity,
    fetch_conda_envs,
    fetch_gpu_type_sources,
    fetch_known_qos,
    fetch_my_jobs,
    fetch_node_features,
    fetch_node_types,
    fetch_partitions,
    fetch_public_partitions,
    fetch_queue_depth,
    fetch_reachable_partitions,
    fetch_user_accounts,
    fetch_user_partitions,
    is_mock,
    is_tool_available,
    last_cluster_error,
    load_config,
    normalize_memory,
    parse_gpu_spelling,
    scheduler_verdict,
    slurm_deadline,
)
from .theme import c, g

#: Bumped only when a key changes meaning or disappears. A consumer pins this,
#: not the package version: slurmate is pre-1.0 and its *human* surfaces move
#: between releases, which is exactly the churn a machine-readable contract has
#: to be insulated from. Additive keys do not bump it, so a reader must ignore
#: fields it does not know rather than rejecting the document.
SCHEMA_VERSION = 1

#: Subcommands this module owns. `parse_args` in :mod:`slurmate.main` declares
#: no positionals, so ``slurmate brief`` is currently an argparse error and the
#: whole verb namespace is free: adding these breaks no existing invocation.
VERBS = ("brief", "nodes", "jobs", "check", "skill", "mcp")

#: One line each, for the main ``--help`` epilog. Spelled here rather than in
#: `main` so a verb cannot be added without the help learning about it: the
#: only thing worse than an undiscoverable subcommand is a list of them that
#: is missing one.
VERB_HELP = {
    "brief": "what this cluster is, and which partitions you can use",
    "nodes": "what machines it has, grouped by cores, memory and GPUs",
    "jobs": "your jobs, and why a pending one is not running",
    "check": "whether a job would be accepted here, without submitting it",
    "skill": "install agent instructions (Claude skill, AGENTS.md or MCP)",
    "mcp": "serve the above as MCP tools over stdio",
}

#: Where each format lands, relative to the directory given to `--install`.
#:
#: Three formats because there is no single standard and pretending otherwise
#: helps nobody. ``claude`` is Claude Code's skill layout. ``agents`` is
#: AGENTS.md, which Cursor, Codex, Copilot and others read from a project
#: root. ``mcp`` is the only one that is a *protocol* rather than a document:
#: it registers `slurmate mcp` as a tool server, so the agent gets typed
#: schemas and an enumerable tool list instead of prose it has to notice.
#:
#: All three carry the same content, generated from the same file, because the
#: failure mode of shipping three is that they drift and two of them become
#: wrong.
INSTALL_PATHS = {
    "claude": "slurmate/SKILL.md",
    "agents": "AGENTS.md",
    "mcp": ".mcp.json",
}

#: Back-compatible alias for the Claude Code layout.
SKILL_RELATIVE_PATH = INSTALL_PATHS["claude"]


# ---------------------------------------------------------------- brief ----


def _why_unreadable(*tools: str) -> str:
    """Why a query could not run, or "".

    `last_cluster_error` is empty whenever `is_tool_available` short-circuits,
    because nothing was run and so nothing failed. `brief` learned to say so;
    `nodes` and `jobs` did not, and returned ``null`` beside an empty
    ``errors`` list, which tells a reader precisely nothing about whether the
    cluster is empty or absent. Same defect one layer down.
    """
    if is_mock():
        return ""
    recorded = last_cluster_error()
    if recorded:
        return recorded
    missing = [name for name in tools if not is_tool_available(name)]
    if missing:
        return (f"{' and '.join(missing)} not found on PATH: this is not a "
                f"Slurm login node, or Slurm is not installed here")
    return ""


def _bad_partition(name: str | None) -> str:
    """An error for a ``--partition`` this cluster does not have, or "".

    Filtering on a name that does not exist produced an empty result, and an
    empty result reads as a fact about the cluster: `nodes -p typo` said "no
    nodes could be read", which is the wording for an unreadable ``sinfo`` and
    blames the machine for the caller's typo. The same shape as the false pass
    in `check`: nothing came back, so nothing was said.
    """
    if not name:
        return ""
    known = fetch_all_partition_names()
    if not known or name in known:
        # An unreadable partition list claims nothing. Rejecting a name
        # against a list we could not fetch is the SM-4 false rejection.
        return ""
    near = _near_misses(name, sorted(known), limit=3)
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    return f"no partition '{name}' on this cluster.{hint}"


def _tooling() -> dict[str, bool]:
    """Which of the programs slurmate consults are actually present.

    An agent that reads ``"sacctmgr": false`` knows why ``you.accounts`` is
    empty, and can say "this cluster does not expose associations to me"
    instead of "you have no accounts". The distinction is the whole reason
    these are reported rather than inferred from an empty list.
    """
    found = {name: is_tool_available(name)
             for name in ("sinfo", "scontrol", "sacctmgr", "squeue", "sbatch",
                          "conda")}
    # `module` is a shell function rather than a program, so `shutil.which`
    # never finds it and every module-having cluster reported false. The real
    # entry point is what :func:`_module_command` resolves, which is the same
    # answer `check_modules` acts on.
    found["module"] = _module_command() is not None
    return found


def _submittable(partitions: list[dict[str, Any]]) -> set[str] | None:
    """Names the caller holds an association for, or None when unfiltered.

    Thin wrapper over :func:`fetch_user_partitions`, whose ``None`` means "no
    filtering is justified" rather than "no access". Passing that through
    unchanged matters: a site that scopes access by account rather than by
    partition looks identical to a site with no associations at all, and
    reading either as an empty allow-list would hide every partition the
    caller can actually use.
    """
    allowed = fetch_user_partitions()
    if allowed is None:
        return None
    # A partition the association list names but sinfo does not is not useful
    # to report as reachable, and the reverse (sinfo has it, associations do
    # not) is exactly what the filter is for.
    known = {str(p.get("name") or "") for p in partitions}
    return {name for name in allowed if name in known}


def _partition_row(part: dict[str, Any],
                   depth: dict[str, dict[str, int]] | None) -> dict[str, Any]:
    """One partition, with the private bookkeeping keys dropped.

    `fetch_partitions` returns exactly the fields below plus nothing else, so
    this is a projection rather than a translation; it exists to keep an
    underscore-prefixed key from leaking into a published schema, where it
    would immediately become something a consumer depends on.
    """
    row = {
        "name": part.get("name"),
        "is_default": bool(part.get("is_default")),
        "state": part.get("state"),
        "nodes": part.get("nodes"),
        "nodes_up": part.get("nodes_up"),
        "cpus_per_node": part.get("cpus_per_node"),
        "mem_per_node_mb": part.get("mem_per_node_mb"),
        "heterogeneous": bool(part.get("heterogeneous")),
        "timelimit": part.get("timelimit"),
        "has_gpu": bool(part.get("has_gpu")),
        "gpu_types": list(part.get("gpu_types") or []),
        "gpus_per_node": part.get("gpus_per_node"),
        # None, not {"running": 0, "pending": 0}: an unreadable `squeue` must
        # not render as an idle partition, which is the single most inviting
        # thing a chooser can be told.
        "queue": None if depth is None
        else depth.get(str(part.get("name") or ""), {"running": 0, "pending": 0}),
    }
    # Present only after _enrich_partition_maxima has run, i.e. under --full on
    # a heterogeneous partition. Absent means "not looked up", which is a
    # different claim from "the nodes are all the same size".
    for key in ("max_cpus_per_node", "max_mem_per_node_mb"):
        if part.get(key):
            row[key] = part[key]
    return row


def _select(
    partitions: list[dict[str, Any]], allowed: set[str] | None, *,
    all_partitions: bool, partition: str | None,
) -> tuple[list[dict[str, Any]], str]:
    """The partitions to report, and the name of the rule that chose them.

    Ordered by how much the rule actually knows, and each rule is only used
    when the stronger one above it could not answer.

    ``associations``
        The real gate on a multi-PI cluster, when the site scopes them per
        partition. :func:`fetch_user_partitions` returns ``None`` when it does
        not, which is a refusal to guess rather than an empty answer.
    ``accounts``
        :func:`fetch_reachable_partitions`: partitions at least one of the
        caller's accounts may submit to, by the same allow/deny rules
        `partition_account_refusal` was verified with.
    ``public``
        Open to all accounts, not hidden, up. A last resort for a caller with
        no visible accounts at all.
    ``none``
        Nothing could be established, so nothing is filtered.

    **The ``accounts`` rule exists because ``public`` was wrong on its own.**
    Measured here: 88 partitions total, 6 public, 21 the caller's accounts can
    actually reach. ``beagle3`` is one of the 15 in the gap, and the caller
    holds ``beagle3-users`` for it. With SKILL.md telling an agent never to
    name a partition the brief did not list, filtering on "open to everyone"
    talks that agent out of a partition the user pays for. That is a worse
    failure than a long document, and it was shipped for one revision.

    The rule is *named* in the output rather than left implicit, because an
    omission means something different under each: under ``associations`` an
    absent partition is one you cannot use, under ``accounts`` one no account
    of yours reaches, under ``public`` merely one not open to everyone, and
    under ``none`` nothing was dropped at all. A consumer that cannot tell
    them apart would read the weakest as the strongest.

    The site default is always kept. It is the partition Slurm picks when a
    script names none, so a brief without it describes a different cluster
    from the one an omitted ``--partition`` reaches.
    """
    if partition:
        return [p for p in partitions if p.get("name") == partition], "named"
    if all_partitions:
        return list(partitions), "none"

    def keep(names: set[str], label: str) -> tuple[list[dict[str, Any]], str]:
        return [p for p in partitions
                if p.get("name") in names or p.get("is_default")], label

    if allowed is not None:
        return keep(allowed, "associations")
    reachable = fetch_reachable_partitions()
    if reachable:
        return keep(reachable, "accounts")
    public = {p["name"] for p in fetch_public_partitions(partitions)}
    if public:
        return keep(public, "public")
    return list(partitions), "none"


def _environment() -> dict[str, Any]:
    """Module and conda-env names, for the ``module load`` line.

    Gated behind ``--full`` on cost, measured here: ``fetch_available_modules``
    takes **5.3 s** and returns 526 names (about 8 kB of the document), against
    0.8 s and 21 for the conda envs. Paying that on every brief would make the
    common call four times slower for a list most callers never read.

    The cheaper route to the same answer is the check loop, and SKILL.md points
    at it: a wrong module name comes back from ``slurmate check`` naming the
    versions that do exist ("'cuda' is available as: cuda/10.2, cuda/11.2,
    ..."), so an agent can guess once and be corrected, rather than being
    handed every module on the system in advance.
    """
    return {"conda_envs": fetch_conda_envs(), "modules": fetch_available_modules()}


def cluster_brief(
    *, full: bool = False, all_partitions: bool = False,
    partition: str | None = None,
) -> dict[str, Any]:
    """Everything about this site that a script-writer needs, in one document.

    Assembly only: every fact below comes from a fetcher that already existed
    and is already memoised for the process, so the cost is the cluster
    queries, not the shaping.

    Three things about the shape are load-bearing rather than cosmetic:

    **``mock`` is top level.** :func:`is_mock` exists because synthetic
    partitions presented without a marker are measurement-shaped fiction. An
    agent that writes a script against demo data and says nothing is the worst
    failure this function can produce, so the flag is not buried beside the
    data it invalidates.

    **The partition list is filtered by default.** A site with 88 partitions
    returns a document that costs more context than the answer is worth, and
    the partitions a caller cannot submit to are noise in the literal sense:
    they can only mislead a chooser. ``partitions_elided`` says how many were
    dropped so the omission is visible, and ``all_partitions`` turns it off.

    **The expensive reads are gated.** :func:`fetch_partition_node_maxima` and
    :func:`fetch_gpu_type_sources` are one ``sinfo`` per partition. The default
    pays that only for GPU partitions, where the typed-versus-feature
    distinction decides whether a request parses at all; ``full`` pays it
    everywhere.
    """
    # Lazily imported: `main` pulls in rich, and this module is also used as a
    # library. The cycle is real (main dispatches to us) but only at call time.
    from .main import _enrich_partition_maxima

    with slurm_deadline():
        mock = is_mock()
        identity = fetch_cluster_identity()
        partitions = fetch_partitions()
        allowed = _submittable(partitions)

        wanted, how = _select(partitions, allowed,
                              all_partitions=all_partitions, partition=partition)
        elided = len(partitions) - len(wanted)

        # One call for every partition at once, before the loop.
        depth = fetch_queue_depth()
        rows = []
        for part in wanted:
            rows.append(_partition_row(
                _enrich_partition_maxima(part) if full else part, depth
            ))

        gpu_formats: dict[str, Any] = {}
        for part in wanted:
            name = str(part.get("name") or "")
            if not name or not (full or part.get("has_gpu")):
                continue
            gpu_formats[name] = fetch_gpu_type_sources(name)

        config = load_config()
        source = config_source()
        errors = [msg for msg in (_bad_partition(partition), last_cluster_error())
                  if msg]
        # A cluster that answered nothing has to say so. `last_cluster_error`
        # is empty here, because `is_tool_available` short-circuits before any
        # command runs and so records no failure: with Slurm simply absent the
        # document came back with zero partitions, zero errors and every
        # cluster field null, which reads as a real but empty cluster.
        if not partitions and not mock:
            missing = [name for name in ("sinfo", "scontrol")
                       if not is_tool_available(name)]
            if missing:
                errors.append(
                    f"{' and '.join(missing)} not found on PATH: this is not a "
                    f"Slurm login node, or Slurm is not installed here. Nothing "
                    f"below describes a cluster"
                )
            elif not errors:
                errors.append("sinfo returned no partitions")

        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mock": mock,
            # ``qos`` is site-wide, not the caller's: `fetch_known_qos` reads
            # every QoS sacctmgr knows about. It sat under ``you`` for one
            # revision and that was a lie of placement, which on this cluster
            # would have offered an agent 100 names as "yours".
            "cluster": {**identity, "qos": fetch_known_qos()},
            "tools": _tooling(),
            "you": {
                "user": current_username() or None,
                "accounts": fetch_user_accounts(),
                # None, not []: "the site does not scope associations by
                # partition" is not "you may use no partition".
                "partitions": sorted(allowed) if allowed is not None else None,
            },
            "partitions": rows,
            "partition_filter": how,
            "partitions_elided": elided,
            "node_features": sorted(fetch_node_features() or []),
            # Absent rather than empty without --full, so "not looked up" is
            # not readable as "this cluster has no modules".
            **({"environment": _environment()} if full else {}),
            "gpu_request_formats": gpu_formats,
            "config_defaults": {"source": source or None, "keys": dict(config)},
            "errors": errors,
        }


def _render_brief(brief: dict[str, Any]) -> None:
    """The same document, for a person who typed the command by hand."""
    ident = brief["cluster"]
    if brief["mock"]:
        print(f"  {c.YELLOW}{g.WARN} SIMULATED (SLURMATE_MOCK): nothing below "
              f"describes a real cluster.{c.RESET}")
    name = ident.get("name") or "unknown"
    version = ident.get("slurm_version") or "unknown version"
    print(f"  {c.CYAN}{name}{c.RESET} {c.GRAY}(Slurm {version}, "
          f"{ident.get('select_type') or 'unknown select plugin'}, "
          f"on {ident.get('hostname') or 'unknown host'}){c.RESET}")

    you = brief["you"]
    accounts = you["accounts"]
    # Truncated here and only here: 34 accounts on one line is not something a
    # person reads, and the JSON still carries every one of them.
    shown = ", ".join(accounts[:6]) or "none visible"
    if len(accounts) > 6:
        shown += f" (+{len(accounts) - 6} more)"
    print(f"  {c.GRAY}you:{c.RESET} {you.get('user') or 'unknown'}   "
          f"{c.GRAY}accounts:{c.RESET} {shown}")

    rows = brief["partitions"]
    if not rows and brief["errors"]:
        for msg in brief["errors"]:
            print(f"  {c.RED}{g.ERR} {msg}{c.RESET}")
        return
    if not rows:
        print(f"  {c.YELLOW}{g.WARN} no partitions could be read.{c.RESET}")
    else:
        width = max(len("partition"), max(len(str(r["name"])) for r in rows))
        print(f"\n  {c.GRAY}{'partition'.ljust(width + 2)}"
              f"{'nodes':>9}  {'cores':>6}  {'memory':>9}  {'time':<11}  "
              f"{'queued':>10}  gpus{c.RESET}")
        for row in rows:
            up = row["nodes_up"]
            nodes = "?" if up is None else f"{up}/{row['nodes']}"
            mem = row["mem_per_node_mb"] or 0
            # The "+" is the whole point of the `heterogeneous` flag: it marks
            # a figure that is the smallest node, not the largest.
            plus = "+" if row["heterogeneous"] else ""
            gpus = ",".join(row["gpu_types"]) or ("yes" if row["has_gpu"] else "")
            star = "*" if row["is_default"] else " "
            name = f"{str(row['name'])}{star}".ljust(width + 2)
            q = row["queue"]
            queued = "?" if q is None else f"{q['running']}r/{q['pending']}p"
            print(f"  {name}{nodes:>9}  {str(row['cpus_per_node']) + plus:>6}  "
                  f"{str(mem // 1024) + plus + ' GB':>9}  "
                  f"{str(row['timelimit'] or '?'):<11}  {queued:>10}  {gpus}")
    if brief["partitions_elided"]:
        why = {"associations": "you hold no association for",
               "accounts": "none of your accounts can submit to",
               "public": "not open to all accounts",
               "named": "not named"}.get(brief["partition_filter"], "filtered out")
        print(f"  {c.GRAY}({brief['partitions_elided']} more {why}; "
              f"--all-partitions to see them){c.RESET}")
    for msg in brief["errors"]:
        print(f"  {c.YELLOW}{g.WARN} {msg}{c.RESET}")


# ---------------------------------------------------------------- nodes ----

#: How many hardware types the human table shows before it stops. The JSON is
#: never truncated; this is a reading limit, not a data one.
NODE_TYPE_PREVIEW = 12


def node_inventory(*, partition: str | None = None,
                   gpu_only: bool = False) -> dict[str, Any]:
    """What machines this cluster has, grouped by what a job can ask for.

    The question `brief` does not answer. A partition row aggregates its nodes,
    which is right for choosing a partition and wrong for choosing a
    ``--constraint`` or a ``--gres``: it hides that ``gpu`` spans rtx6000 and
    v100 boxes, and that asking for one rather than the other is a different
    wait. Measured here, 608 nodes across 57 shapes, of which the top handful
    are most of the machine.

    Nothing is compressed away. A cluster with 57 configurations really has 57,
    most of them single-node PI machines, and rounding that off would be the
    tool inventing a tidier cluster than the one the caller is on.
    """
    with slurm_deadline():
        bad = _bad_partition(partition)
        types = None if bad else fetch_node_types()
        if types is None:
            return {
                "schema_version": SCHEMA_VERSION, "mock": is_mock(),
                "node_types": None, "nodes_total": None, "nodes_avail": None,
                "errors": [msg for msg in (bad, _why_unreadable("sinfo")) if msg],
            }
        rows = types
        if partition:
            rows = [r for r in rows if partition in r["partitions"]]
        if gpu_only:
            rows = [r for r in rows if r["gpus_per_node"]]
        return {
            "schema_version": SCHEMA_VERSION,
            "mock": is_mock(),
            "node_types": rows,
            "nodes_total": sum(r["count"] for r in rows),
            # Counted from the same rows, so it cannot disagree with the table
            # above it the way a separately-queried total would.
            "nodes_avail": sum(r["avail"] for r in rows),
            "errors": [msg for msg in (last_cluster_error(),) if msg],
        }


def _render_nodes(inv: dict[str, Any], *, show_all: bool) -> None:
    if inv["mock"]:
        print(f"  {c.YELLOW}{g.WARN} SIMULATED (SLURMATE_MOCK): not a real "
              f"cluster.{c.RESET}")
    rows = inv["node_types"]
    if not rows:
        for msg in inv["errors"]:
            print(f"  {c.RED}{g.ERR} {msg}{c.RESET}")
        if not inv["errors"]:
            print(f"  {c.YELLOW}{g.WARN} no nodes could be read.{c.RESET}")
        return
    shown = rows if show_all else rows[:NODE_TYPE_PREVIEW]
    print(f"  {inv['nodes_total']} nodes, {inv['nodes_avail']} usable, "
          f"{len(rows)} hardware types")
    print(f"\n  {c.GRAY}{'nodes':>9}  {'cores':>5}  {'memory':>9}  "
          f"{'gpus':<16}  features{c.RESET}")
    for row in shown:
        # The model from features when the GRES is count-only, which is most
        # of them here; "?" only when the cluster really does not say.
        model = ",".join(row["gpu_types"]) or "?"
        gpus = f"{row['gpus_per_node']}x {model}" if row["gpus_per_node"] else ""
        # One column, so a partly-drained type cannot push every field right.
        count = (f"{row['count']}" if row["avail"] == row["count"]
                 else f"{row['avail']}/{row['count']}")
        print(f"  {count:>9}  {row['cpus']:>5}  "
              f"{row['mem_mb'] // 1024:>6} GB  {gpus:<16}  "
              f"{','.join(row['features'])[:38]}")
    if len(rows) > len(shown):
        print(f"  {c.GRAY}({len(rows) - len(shown)} more types, mostly "
              f"single-node; --all to see them){c.RESET}")


# ----------------------------------------------------------------- jobs ----


def job_report(user: str | None = None) -> dict[str, Any]:
    """The caller's jobs, and for each pending one, what to do about it.

    "Why is my job still pending" is the question that follows a first submit,
    and the answer is in a column most people do not know exists.
    :data:`PENDING_REASONS` turns the code into an action, which matters
    because the codes are not interchangeable: ``Priority`` clears on its own,
    ``Resources`` clears faster if you ask for less, a ``QOSMax...`` cap does
    not clear at all until something of yours finishes, and
    ``PartitionTimeLimit`` means the job will never start as written.
    """
    with slurm_deadline():
        jobs = fetch_my_jobs(user)
        if jobs is None:
            return {
                "schema_version": SCHEMA_VERSION, "mock": is_mock(),
                "user": user or current_username() or None, "jobs": None,
                "running": None, "pending": None,
                "errors": [msg for msg in (_why_unreadable("squeue"),) if msg],
            }
        for job in jobs:
            # "" rather than absent for an unrecognised code: the field is
            # always there, and an empty explanation is the honest answer for
            # a site plugin's reason we have not written down.
            job["reason_explained"] = (
                explain_pending(job["reason"] or "") if job["state"] == "PENDING" else ""
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "mock": is_mock(),
            "user": user or current_username() or None,
            "jobs": jobs,
            "running": sum(1 for j in jobs if j["state"] == "RUNNING"),
            "pending": sum(1 for j in jobs if j["state"] == "PENDING"),
            "errors": [msg for msg in (last_cluster_error(),) if msg],
        }


def _render_jobs(report: dict[str, Any]) -> None:
    if report["mock"]:
        print(f"  {c.YELLOW}{g.WARN} SIMULATED (SLURMATE_MOCK): not a real "
              f"cluster.{c.RESET}")
    jobs = report["jobs"]
    if jobs is None:
        detail = report["errors"][0] if report["errors"] else ""
        print(f"  {c.RED}{g.ERR} the queue could not be read"
              f"{': ' + detail if detail else ''}{c.RESET}")
        return
    if not jobs:
        print(f"  {c.GRAY}no jobs for {report['user'] or 'you'}.{c.RESET}")
        return
    print(f"  {report['running']} running, {report['pending']} pending "
          f"for {report['user']}")
    width = max(len(j["job_id"]) for j in jobs)
    for job in jobs:
        colour = c.GREEN if job["state"] == "RUNNING" else c.YELLOW
        where = job["nodelist"] or job["reason"] or ""
        print(f"\n  {colour}{job['job_id'].ljust(width)}{c.RESET}  "
              f"{job['name'][:24]:<24}  {job['state']:<9}  {job['partition']:<12}  "
              f"{job['elapsed']}/{job['time_limit']}  {where}")
        if job.get("reason_explained"):
            print(f"  {' ' * width}  {c.GRAY}{job['reason_explained']}{c.RESET}")


# ---------------------------------------------------------------- check ----

#: ``#SBATCH`` directives slurmate models as named fields, mapped onto the keys
#: the validators read. Both spellings of each flag, because a script written
#: by hand uses whichever its author remembers.
_DIRECTIVE_FIELDS: dict[str, str] = {
    "--job-name": "job_name", "-J": "job_name",
    "--account": "account", "-A": "account",
    "--partition": "partition", "-p": "partition",
    "--qos": "qos", "-q": "qos",
    "--cpus-per-task": "cpus", "-c": "cpus",
    "--mem": "memory",
    "--mem-per-cpu": "mem_per_cpu",
    "--time": "time_limit", "-t": "time_limit",
    "--nodes": "nodes", "-N": "nodes",
    "--ntasks-per-node": "ntasks_per_node",
    "--constraint": "constraint", "-C": "constraint",
    "--array": "array_spec", "-a": "array_spec",
    "--output": "output_file", "-o": "output_file",
}

#: The four GPU spellings, and the ``gpu_format`` each one implies. Kept apart
#: from `_DIRECTIVE_FIELDS` because they set two fields, not one, and because
#: which spelling was used is itself a finding: `unsupported_gpu_format` can
#: only object to a format it was told about.
_GPU_DIRECTIVES: dict[str, str] = {
    "--gres": "gres_type",
    "--gpus": "gpus", "-G": "gpus",
    "--gpus-per-node": "gpus_per_node",
    "--gpus-per-task": "gpus_per_task",
}

_INT_FIELDS = ("cpus", "nodes", "ntasks_per_node")


def _directive_tokens(text: str) -> list[tuple[str, str]]:
    """``(flag, value)`` for every ``#SBATCH`` directive, in file order.

    Stops at the first non-comment, non-blank line, which is where Slurm stops
    too: a ``#SBATCH`` after the first command is inert, and reading it would
    validate a directive the scheduler ignores.
    """
    from .builder import _split_flag

    out: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or (line.startswith("#") and not line.startswith("#SBATCH")):
            continue
        if not line.startswith("#SBATCH"):
            break
        body = line[len("#SBATCH"):].strip()
        if not body:
            continue
        flag, value = _split_flag(body)
        out.append((flag, value))
    return out


#: `module load a b`, `module add a`, and Lmod's `ml a b`. `module purge` and
#: `module unload` are deliberately not matched: they name modules the script is
#: getting rid of, and reporting one as missing is backwards.
_MODULE_LINE = re.compile(r"^\s*(?:module\s+(?:load|add)|ml)\s+(?P<mods>.+?)\s*$")

#: `conda activate NAME` / `mamba activate NAME`, and the `source activate NAME`
#: spelling older scripts still use.
_CONDA_LINE = re.compile(
    r"^\s*(?:(?P<tool>conda|mamba)\s+activate|source\s+activate)\s+(?P<env>\S+)"
)

#: `source /path/to/venv/bin/activate`, and `.` for the same thing.
_VENV_LINE = re.compile(r"^\s*(?:source|\.)\s+(?P<path>\S*/bin/activate)\s*$")


def parse_script_body(text: str) -> dict[str, Any]:
    """Modules and environment activation, read out of the script's commands.

    The directives are only half of what a script has to get right about a
    cluster. `check_modules` and `check_conda_env` have existed all along and
    catch the failure that costs the most time (the job queues, is scheduled,
    starts, and dies on ``module load`` minutes or hours later), but nothing
    ever handed them a script: they read ``answers["modules"]`` and
    ``answers["env_name"]``, which only the wizard and the CLI flags filled in.

    So `check --script` reported a clean bill on a script whose first command
    was ``module load cuda/99.9-does-not-exist``, while SKILL.md told an agent
    that exact case was covered. This is the wiring that makes the claim true.

    Read from the whole body, not just the top: an activation inside an ``if``
    or after a ``cd`` is still an activation, and a script that loads its
    modules late is not thereby exempt. Being generous here is safe because
    every consumer of these fields reports rather than refuses.
    """
    found: dict[str, Any] = {}
    modules: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _MODULE_LINE.match(line)
        if match:
            for token in match.group("mods").split():
                # A flag (`module load -f x`) or a shell expansion is not a
                # module name, and asking the module system about `$FOO`
                # produces a "not found" about the literal string.
                if token.startswith("-") or "$" in token or "`" in token:
                    continue
                modules.append(token)
            continue
        match = _VENV_LINE.match(line)
        if match:
            found["env_type"] = "venv"
            found["env_name"] = match.group("path")[: -len("/bin/activate")]
            continue
        match = _CONDA_LINE.match(line)
        if match:
            found["env_type"] = match.group("tool") or "conda"
            found["env_name"] = match.group("env").strip("\"'")
    if modules:
        found["modules"] = list(dict.fromkeys(modules))
    return found


def parse_sbatch_script(text: str) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """An answers dict and any parse-level findings, from a real script.

    The builder writes directives and nothing reads them back, so an agent that
    wrote its own script by hand had no way to ask slurmate about it. This is
    the missing direction.

    It returns findings of its own for the one class of defect that is invisible
    downstream: a directive given twice. Slurm accepts the duplicate and
    silently honours the last one (measured: ``sbatch --test-only -J first -J
    second`` reports ``***PASSED***``), so the job runs under a name, an account
    or a time limit that the reader of the script did not choose. Once the
    answers dict is built, the earlier value is simply gone, which is why the
    check has to happen here.
    """
    answers: dict[str, Any] = {}
    findings: list[tuple[str, str]] = []
    custom: list[str] = []
    seen: dict[str, str] = {}

    for flag, value in _directive_tokens(text):
        field = _DIRECTIVE_FIELDS.get(flag)
        if field is None and flag in _GPU_DIRECTIVES:
            try:
                count, gpu_type = parse_gpu_spelling(flag, value)
            except ValueError as exc:
                findings.append(("error", f"{flag}: {exc}"))
                continue
            answers["gpus"] = count
            answers["gpu_format"] = _GPU_DIRECTIVES[flag]
            if gpu_type:
                answers["gpu_type"] = gpu_type
            field = "gpus"
        elif field is None:
            # Unmodelled but legitimate (--exclusive, --dependency, --signal).
            # Kept verbatim so the managed-flag and quoting checks can see it,
            # rather than dropped, which would make the script slurmate
            # validated differ from the one the user holds.
            custom.append(f"{flag} {value}".strip())
            continue
        else:
            answers[field] = value

        prior = seen.get(field)
        if prior is not None and prior != flag:
            findings.append((
                "error",
                f"'{field}' is set twice, by {prior} and by {flag}; Slurm "
                f"accepts both and silently honours the last one",
            ))
        elif prior is not None:
            findings.append((
                "error",
                f"{flag} appears more than once; Slurm silently honours the "
                f"last one",
            ))
        seen[field] = flag

    for field in _INT_FIELDS:
        raw = answers.get(field)
        if raw is None:
            continue
        try:
            answers[field] = int(str(raw).strip())
        except ValueError:
            findings.append(("error", f"--{field.replace('_', '-')}: "
                                      f"'{raw}' is not a whole number"))
            answers.pop(field)

    if answers.get("memory"):
        # Normalised here rather than left raw: every memory check downstream
        # parses this field, and "16GB" is a spelling Slurm rejects but a human
        # writes. normalize_memory returns the value unchanged when it cannot
        # improve it, so an unparseable one still reaches validate_memory.
        answers["memory"] = normalize_memory(str(answers["memory"]))
    if custom:
        answers["custom_sbatch"] = " ".join(custom)
    # Directives first, body second, and the body does not overwrite: a
    # `#SBATCH` line is what the caller declared, and the two disagree only in
    # a script that is already confusing.
    for key, value in parse_script_body(text).items():
        answers.setdefault(key, value)
    return answers, findings


def _resolve_partition(answers: dict[str, Any]) -> None:
    """Attach ``_partition_obj``, which every capacity check reads.

    Left absent, `validate_job_config` runs only the partition-independent
    rules and reports a clean bill for a request no node on the cluster could
    satisfy, which is the most confidently wrong answer this module can give.
    """
    from .main import _get_partition

    name = str(answers.get("partition") or "")
    if not name:
        return
    answers["_partition_obj"] = _get_partition(fetch_partitions(), name)


def _unverifiable(answers: dict[str, Any]) -> list[tuple[str, str]]:
    """Say which checks could not run, rather than passing in silence.

    `check_modules` is correct to claim nothing when there is no module system
    to ask, but "I found no problem" and "I could not look" reach a caller as
    the same clean bill, and an agent acts on a clean bill. Measured on Booth's
    Mercury over a non-login ssh: ``MODULEPATH`` is unset, so every module name
    in a script goes unchecked while `check` prints "nothing to report".

    The mirror of `validate_job_config`'s "Capacity limits NOT checked" line,
    and for the same reason: silence about a check that did not happen is the
    failure mode this whole surface exists to remove.
    """
    out: list[tuple[str, str]] = []
    if answers.get("modules") and _module_command() is None:
        names = ", ".join(str(m) for m in answers["modules"])
        out.append((
            "warning",
            f"module names NOT checked ({names}): no module system is "
            f"reachable from here, so a name this cluster does not have would "
            f"not have been caught. Slurm still accepts the job, and it dies "
            f"on 'module load' after it starts",
        ))
    return out


def check_request(
    answers: dict[str, Any], *, script: str | None = None,
    extra: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Every check slurmate can make, as findings rather than as a screen.

    The aggregation is deliberately not new logic. `_partition_issues` and
    `site_check_issues` already return ``(level, message)`` and already print
    nothing, precisely so the wizard and the CLI summary cannot drift apart;
    this is a third consumer of the same two functions, and it stays correct
    for the same reason they do.

    ``scheduler`` is the one answer that is not ours. ``sbatch --test-only``
    enters no queue and spends no allocation, so it is safe to call in a loop,
    and it is the only opinion here that the controller itself is accountable
    for. It is reported separately from ``findings`` for that reason: a caller
    deciding whether to trust a verdict needs to know who gave it.

    **The flag form gets that verdict too**, by building the script the flags
    describe and asking about that. Without it the most valuable half of the
    answer was available only to a caller that had already written a file,
    which is backwards: the flag form exists to be asked *before* anything is
    written. ``checked_script`` carries the text that was judged, so a refusal
    can be read against the exact directives that earned it.
    """
    from .builder import build_from_answers
    from .main import _partition_issues, site_check_issues

    with slurm_deadline():
        _resolve_partition(answers)
        findings: list[tuple[str, str]] = list(extra or [])
        findings += _partition_issues(answers)
        findings += site_check_issues(answers)
        findings += _unverifiable(answers)

        # partial=True: emit only what the caller actually specified. A full
        # build fills defaults in, and a refusal earned by a default nobody
        # asked for is a finding about slurmate rather than about the request.
        judged = script if script is not None else build_from_answers(answers, partial=True)
        # will_create=False: this caller is not about to submit, so the
        # directories slurmate would have made on the way do not exist yet and
        # must not be assumed into existence.
        findings += check_log_dirs(judged, will_create=False)

        verdict, reason = scheduler_verdict(judged)
        scheduler: dict[str, Any] = {"verdict": verdict, "reason": reason or None}
        if verdict == "unavailable":
            findings.append((
                "warning",
                f"the scheduler was NOT asked about this job ({reason}), so "
                f"nothing here has been confirmed against the controller",
            ))

        out = [{"level": level, "message": msg} for level, msg in findings]
        return {
            "schema_version": SCHEMA_VERSION,
            "mock": is_mock(),
            # True means CHECKED AND CLEAN, not merely "nothing came back".
            # Built the other way round it reported ok=true with zero findings
            # on a machine with no `sbatch` at all, which is the most confident
            # possible wrong answer: an agent reads `ok` and hands the script
            # over. When this is false, `findings` and `scheduler.verdict`
            # together say whether something is wrong or whether nothing could
            # be established.
            "ok": verdict == "accepted"
                  and not any(f["level"] == "error" for f in out),
            "checked": verdict != "unavailable",
            "findings": out,
            "scheduler": scheduler,
            "checked_script": judged,
        }


def _render_check(result: dict[str, Any]) -> None:
    """The verdict, for a person."""
    if result["mock"]:
        print(f"  {c.YELLOW}{g.WARN} SIMULATED (SLURMATE_MOCK): this verdict "
              f"is not about a real cluster.{c.RESET}")
    for finding in result["findings"]:
        colour = c.RED if finding["level"] == "error" else c.YELLOW
        mark = g.ERR if finding["level"] == "error" else g.WARN
        # Several of these carry their own second and third line (the
        # "Did you mean" suggestion, the partition list), and printing them
        # flush left detached the continuation from the error it belongs to.
        lines = str(finding["message"]).splitlines() or [""]
        print(f"  {colour}{mark} {lines[0]}{c.RESET}")
        for cont in lines[1:]:
            print(f"    {colour}{cont}{c.RESET}")
    scheduler = result["scheduler"]
    if scheduler["verdict"] == "refused":
        print(f"  {c.RED}{g.ERR} Slurm would refuse this job: "
              f"{scheduler['reason']}{c.RESET}")
    elif scheduler["verdict"] == "accepted":
        print(f"  {c.GREEN}{g.OK} sbatch --test-only accepts it.{c.RESET}")
    if result["ok"] and not result["findings"]:
        print(f"  {c.GREEN}{g.OK} nothing to report.{c.RESET}")
    elif not result["checked"]:
        print(f"  {c.YELLOW}{g.WARN} unverified: treat this script as "
              f"unchecked rather than as clean.{c.RESET}")


# ---------------------------------------------------------------- skill ----


def skill_text() -> str:
    """The bundled agent instructions, read from the installed package."""
    from importlib.resources import files

    return (files("slurmate") / "data" / "SKILL.md").read_text(encoding="utf-8")


def agents_text() -> str:
    """The same instructions as AGENTS.md rather than as a Claude skill.

    Generated from `SKILL.md` with its YAML frontmatter replaced, not written
    out a second time. The frontmatter is the only Claude-specific thing in
    the file: `name` and `description` are how a skill is *selected*, and
    AGENTS.md is always read, so it needs a heading where the skill needs a
    trigger. Everything below that line is the same knowledge and there is no
    version of this where maintaining two copies of it ends well.
    """
    body = skill_text()
    if body.startswith("---"):
        end = body.find("\n---", 3)
        if end != -1:
            body = body[end + 4:].lstrip("\n")
    # Demoted one level. AGENTS.md is a document with one title, and the
    # skill body opens with its own `#`, so pasting them together gave a file
    # with two H1s and no structure between them.
    body = re.sub(r"^(#{1,5}) ", r"#\1 ", body, flags=re.MULTILINE)
    return (
        "# Running jobs on this Slurm cluster\n\n"
        "This project runs on an HPC cluster managed by Slurm. `slurmate` reads "
        "the live cluster and validates jobs against it; use it rather than "
        "guessing at `sinfo` output or copying an sbatch script from elsewhere.\n\n"
        "Install it with `pipx install slurmate` if the commands below are "
        "missing.\n\n"
        + body
    )


def mcp_config(command: str | None = None) -> dict[str, Any]:
    """The registration block that points an MCP client at this server.

    Written to ``.mcp.json``, which Claude Code reads from a project root and
    several other clients accept. ``sys.executable -m slurmate`` rather than a
    bare ``slurmate``: the entry point is not always on an agent's PATH (a
    pipx or venv install routinely is not), and the interpreter running this
    is by definition the one the package is installed under.
    """
    argv = [command] if command else [sys.executable, "-m", "slurmate"]
    return {
        "mcpServers": {
            "slurmate": {
                "command": argv[0],
                "args": [*argv[1:], "mcp"],
                "env": {},
            }
        }
    }


def install_skill(directory: str, *, force: bool = False,
                  fmt: str = "claude") -> tuple[str, bool]:
    """Write the agent instructions under ``directory``; ``(path, written)``.

    Refuses an existing file without ``force``. All three of these are things
    a user edits for their own site (the traps that matter on their cluster
    are not the ones that matter on ours, and ``.mcp.json`` routinely holds
    other servers), so silently replacing one would destroy exactly the
    content worth keeping.
    """
    target = os.path.join(directory, INSTALL_PATHS[fmt])
    if os.path.exists(target) and not force:
        return target, False
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if fmt == "mcp":
        body = json.dumps(mcp_config(), indent=2) + "\n"
    elif fmt == "agents":
        body = agents_text()
    else:
        body = skill_text()
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(body)
    return target, True


# ------------------------------------------------------------- dispatch ----


def _emit(document: dict[str, Any]) -> None:
    """One JSON document on stdout, and nothing else on stdout ever.

    The discipline `--print` already keeps, for the same reason: a caller that
    pipes this into a parser must not have to strip a banner, a warning or a
    colour code out of the front of it. Everything advisory goes to stderr.
    """
    json.dump(document, sys.stdout, indent=2, sort_keys=False, default=str)
    sys.stdout.write("\n")


def _brief_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurmate brief",
        description="Report this cluster as one document, for a program to read.",
    )
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON (nothing else reaches stdout)")
    parser.add_argument("--full", action="store_true",
                        help="Also read per-node maxima and GPU request formats "
                             "for every partition, not just the GPU ones")
    parser.add_argument("--all-partitions", action="store_true",
                        help="Include partitions you hold no association for")
    parser.add_argument("--partition", "-p", default=None,
                        help="Report only this partition")
    return parser


def _check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurmate check",
        description="Say whether a job would be accepted here, without submitting it.",
    )
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON (nothing else reaches stdout)")
    parser.add_argument("--script", default=None,
                        help="An sbatch script to check ('-' for stdin)")
    parser.add_argument("--partition", "-p", default=None)
    parser.add_argument("--account", "-A", default=None)
    parser.add_argument("--qos", "-q", default=None)
    parser.add_argument("--job-name", "-J", default=None)
    parser.add_argument("--cpus", "--cpus-per-task", "-c", default=None)
    parser.add_argument("--memory", "--mem", default=None)
    parser.add_argument("--mem-per-cpu", default=None)
    parser.add_argument("--time", "-t", default=None)
    parser.add_argument("--nodes", "-N", default=None)
    parser.add_argument("--ntasks-per-node", default=None)
    parser.add_argument("--gpus", "-G", default=None)
    parser.add_argument("--gpu-type", default=None)
    parser.add_argument("--gpu-format", default=None)
    parser.add_argument("--constraint", "-C", default=None)
    parser.add_argument("--array", "-a", default=None)
    parser.add_argument("--modules", default=None,
                        help="Comma-separated module names")
    parser.add_argument("--output-file", "--output", "-o", default=None)
    parser.add_argument("--command", default=None)
    return parser


def _skill_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurmate skill",
        description="Install the bundled agent instructions into a project.",
    )
    parser.add_argument("--install", action="store_true",
                        help="Write the file (without this, it is printed)")
    parser.add_argument("--format", default="claude",
                        choices=sorted(INSTALL_PATHS),
                        help="claude: a Claude Code skill. agents: AGENTS.md, "
                             "which Cursor/Codex/Copilot read. mcp: register "
                             "'slurmate mcp' as a tool server (default: claude)")
    parser.add_argument("--dir", default=None,
                        help="Directory to install into (default: .claude/skills "
                             "for the claude format, the current directory otherwise)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing file")
    return parser


def _nodes_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurmate nodes",
        description="What machines this cluster has, grouped by what a job can ask for.",
    )
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON (nothing else reaches stdout)")
    parser.add_argument("--partition", "-p", default=None,
                        help="Only nodes reachable through this partition")
    parser.add_argument("--gpu", action="store_true", help="Only GPU nodes")
    parser.add_argument("--all", action="store_true",
                        help="Print every hardware type, not just the largest")
    return parser


def _jobs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurmate jobs",
        description="Your jobs, and for each pending one, what to do about it.",
    )
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON (nothing else reaches stdout)")
    parser.add_argument("--user", "-u", default=None,
                        help="Another user's jobs (default: yours)")
    return parser


def _mcp_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurmate mcp",
        description="Serve these commands as MCP tools over stdio.",
    )
    parser.add_argument("--list-tools", action="store_true",
                        help="Print the tool schemas and exit, without serving")
    return parser


#: Verb to parser factory, so a caller can ask for one without knowing which
#: private function builds it.
_PARSERS = {
    "brief": _brief_parser, "nodes": _nodes_parser, "jobs": _jobs_parser,
    "check": _check_parser, "skill": _skill_parser, "mcp": _mcp_parser,
}


def verb_parser(verb: str) -> argparse.ArgumentParser:
    """The argument parser for one verb.

    Public because the README test checks every documented invocation against
    the parser that will actually see it, and these four commands would
    otherwise be the only ones in the file nothing verifies. The sibling rule
    the test states applies to them too: documentation that nothing verifies is
    documentation that will drift.
    """
    return _PARSERS[verb]()


def _answers_from_flags(args: argparse.Namespace) -> dict[str, Any]:
    """The flag form of ``check``, as an answers dict.

    Values stay as the caller typed them. The validators are the layer that
    knows what a well-formed memory size or time limit looks like, and
    coercing here would turn a finding they are built to report into a
    traceback from this function.
    """
    answers: dict[str, Any] = {}
    fields = {
        "partition": args.partition, "account": args.account, "qos": args.qos,
        "job_name": args.job_name, "cpus": args.cpus, "memory": args.memory,
        "mem_per_cpu": args.mem_per_cpu, "time_limit": args.time,
        "nodes": args.nodes, "ntasks_per_node": args.ntasks_per_node,
        "gpus": args.gpus, "gpu_type": args.gpu_type,
        "gpu_format": args.gpu_format, "constraint": args.constraint,
        "array_spec": args.array, "output_file": args.output_file,
        "command": args.command,
    }
    for key, value in fields.items():
        if value is not None:
            answers[key] = value
    for key in _INT_FIELDS + ("gpus",):
        raw = answers.get(key)
        if raw is not None and str(raw).strip().isdigit():
            answers[key] = int(str(raw).strip())
    if args.modules:
        answers["modules"] = [m.strip() for m in args.modules.split(",") if m.strip()]
    return answers


def _read_script(spec: str) -> str:
    if spec == "-":
        return sys.stdin.read()
    with open(spec, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _run_brief(argv: list[str]) -> int:
    args = verb_parser("brief").parse_args(argv)
    brief = cluster_brief(full=args.full, all_partitions=args.all_partitions,
                          partition=args.partition)
    if args.json:
        _emit(brief)
    else:
        _render_brief(brief)
    # Non-zero only when the document describes nothing: a named partition
    # that does not exist, or a machine with no Slurm. A brief that answered,
    # with a warning in `errors`, is still an answer and exits 0.
    return 1 if (brief["errors"] and not brief["partitions"]) else 0


def _run_check(argv: list[str]) -> int:
    args = verb_parser("check").parse_args(argv)
    script: str | None = None
    extra: list[tuple[str, str]] = []
    if args.script:
        try:
            script = _read_script(args.script)
        except OSError as exc:
            print(f"  {c.RED}{g.ERR} {exc}{c.RESET}", file=sys.stderr)
            return 2
        answers, extra = parse_sbatch_script(script)
        # Flags do NOT refine a script. They did for one revision, and it
        # produced a verdict that contradicted itself: `--gpu-format
        # constraint` over a script holding `--gres=gpu:a100:1` cleared
        # slurmate's finding while `sbatch --test-only`, which can only judge
        # the text on disk, still refused it. Two sources for one answer is
        # the defect; edit the script, or drop --script and describe the
        # request in flags.
        clashing = sorted(_answers_from_flags(args))
        if clashing:
            print(f"  {c.RED}{g.ERR} --script judges the file as written, so "
                  f"it cannot be combined with job flags "
                  f"({', '.join(clashing)}). Edit the script, or drop --script "
                  f"to check the flags on their own.{c.RESET}", file=sys.stderr)
            return 2
    else:
        answers = _answers_from_flags(args)
        if not answers:
            print(f"  {c.RED}{g.ERR} nothing to check: pass --script or at "
                  f"least one job flag.{c.RESET}", file=sys.stderr)
            return 2

    result = check_request(answers, script=script, extra=extra)
    if args.json:
        _emit(result)
    else:
        _render_check(result)
    # Warnings do not fail. validate_job_config documents them as "may be
    # rejected" against figures that can undercount a heterogeneous partition,
    # so exiting non-zero on one would make the honest uncertainty look like a
    # refusal, and a caller looping on the exit code could never converge.
    return 0 if result["ok"] else 1


def _run_skill(argv: list[str]) -> int:
    args = verb_parser("skill").parse_args(argv)
    fmt = args.format
    if not args.install:
        if fmt == "mcp":
            _emit(mcp_config())
        else:
            sys.stdout.write(agents_text() if fmt == "agents" else skill_text())
        return 0
    # Only the Claude skill has a conventional home; AGENTS.md and .mcp.json
    # both live at a project root, and putting them under .claude/skills would
    # be a path no client looks in.
    directory = args.dir if args.dir is not None else (
        ".claude/skills" if fmt == "claude" else "."
    )
    path, written = install_skill(directory, force=args.force, fmt=fmt)
    if not written:
        print(f"  {c.YELLOW}{g.WARN} {path} already exists; --force to "
              f"overwrite.{c.RESET}", file=sys.stderr)
        return 1
    print(f"  {c.GREEN}{g.OK} wrote {path}{c.RESET}", file=sys.stderr)
    return 0


def _run_nodes(argv: list[str]) -> int:
    args = verb_parser("nodes").parse_args(argv)
    inv = node_inventory(partition=args.partition, gpu_only=args.gpu)
    if args.json:
        _emit(inv)
    else:
        _render_nodes(inv, show_all=args.all)
    return 0 if inv["node_types"] is not None else 1


def _run_jobs(argv: list[str]) -> int:
    args = verb_parser("jobs").parse_args(argv)
    report = job_report(args.user)
    if args.json:
        _emit(report)
    else:
        _render_jobs(report)
    return 0 if report["jobs"] is not None else 1


def _run_mcp(argv: list[str]) -> int:
    """Serve MCP over stdio, or print the tool schemas and stop.

    Imported here rather than at module scope so the cost of the protocol
    layer is paid only by the caller that asked for it, the same reason the
    wizard's `prompt_toolkit` is deferred in `main`.
    """
    args = verb_parser("mcp").parse_args(argv)
    from .mcp import serve, tool_schemas

    if args.list_tools:
        _emit({"schema_version": SCHEMA_VERSION, "tools": tool_schemas()})
        return 0
    return serve()


def dispatch(argv: list[str]) -> int:
    """Run one verb. ``argv`` starts with the verb itself."""
    verb, rest = argv[0], argv[1:]
    runners = {
        "brief": _run_brief, "nodes": _run_nodes, "jobs": _run_jobs,
        "check": _run_check, "skill": _run_skill, "mcp": _run_mcp,
    }
    try:
        run = runners[verb]
    except KeyError:
        raise ValueError(f"not an agent verb: {verb!r}") from None
    return run(rest)


def is_verb(argv: list[str]) -> bool:
    """Whether this argv opens with one of our subcommands.

    Checked before `parse_args` runs, and deliberately narrow: only the first
    token, only an exact match, and only when it is not an option. A value that
    happens to read ``brief`` (``--partition brief``) is never in that position,
    so there is no spelling of an existing invocation this can capture.
    """
    return bool(argv) and argv[0] in VERBS


def verb_suggestion(argv: list[str]) -> str:
    """A "did you mean" line for a mistyped verb, or "".

    Only for a bare first word, which is the one position a verb can occupy,
    and only when it is close to a real one. A word that resembles nothing is
    left to argparse: it is more likely a misplaced value than a typo, and
    argparse's own message is right about that case.
    """
    from .system_utils import _near_misses

    if not argv or argv[0].startswith("-") or argv[0] in VERBS:
        return ""
    near = _near_misses(argv[0], VERBS, limit=2)
    if not near:
        return ""
    return (f"  {c.RED}{g.ERR} slurmate: no such command '{argv[0]}'. "
            f"Did you mean: {', '.join(near)}?{c.RESET}")


# Kept importable for the schema contract test, which asserts on the key set
# rather than on a live cluster.
__all__ = [
    "SCHEMA_VERSION",
    "VERBS",
    "check_request",
    "cluster_brief",
    "dispatch",
    "install_skill",
    "is_verb",
    "job_report",
    "node_inventory",
    "parse_sbatch_script",
    "parse_script_body",
    "agents_text",
    "mcp_config",
    "skill_text",
    "VERB_HELP",
    "verb_parser",
    "verb_suggestion",
]
