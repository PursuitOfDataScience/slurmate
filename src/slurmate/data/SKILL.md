---
name: slurmate
description: Study an unfamiliar Slurm cluster and write sbatch scripts that it will actually accept. Use when landing on a new HPC login node, choosing a partition, account, QoS or GPU type, writing or fixing a job script, or debugging why sbatch refused one.
---

# Working on a Slurm cluster you do not know

Four read-only commands. Nothing here submits a job.

| Command | Answers |
|---|---|
| `slurmate brief` | What is this cluster, and which partitions can I use? |
| `slurmate nodes` | What machines are in it, and which have the GPU I need? |
| `slurmate jobs` | What is running for me, and why is that one still pending? |
| `slurmate check` | Would this job be accepted here? |

Add `--json` to any of them. There is also an MCP server (`slurmate mcp`) that
exposes the same four as typed tools, if the agent speaks MCP.

## 1. Before writing anything: read the site

```bash
slurmate brief --json
```

One call. Read it before choosing a partition, not after a job is refused.

| Field | What it decides |
|---|---|
| `mock` | **If `true`, none of this is real.** Say so and stop. |
| `cluster.name` / `.slurm_version` | Which site this is |
| `cluster.select_type` | `select/cons_res` cannot parse `--gpus` or `--gpus-per-task`; use `--gres` |
| `cluster.max_array_size` | The ceiling on `--array`. Sites differ wildly (1001 vs 65533) |
| `cluster.qos` | Every QoS on the site. Not all of them are yours |
| `you.accounts` | What may follow `--account`. Empty **plus** `tools.sacctmgr: false` means "not visible", not "none" |
| `partitions[].nodes_up` | `0` advertises capacity nothing can run. `null` means the state is unknown, so do not filter on it |
| `partitions[].heterogeneous` | **`true` means `cpus_per_node` and `mem_per_node_mb` are the smallest node, not the largest.** `--full` adds `max_cpus_per_node` |
| `partitions[].queue` | `{running, pending}`. Two partitions with identical capacity differ by hours of wait. `null` means `squeue` failed, not that it is idle |
| `partitions[].timelimit` | The cap on `--time` |
| `gpu_request_formats` | The trap that wastes the most time. See below |

### The partition list is filtered, and `partition_filter` says how

An omission means something different under each rule:

| `partition_filter` | An absent partition is one that |
|---|---|
| `associations` | you hold no Slurm association for |
| `accounts` | none of your accounts may submit to |
| `public` | is not open to all accounts (you may still hold an account for it) |
| `none` | was not filtered at all |

`partitions_elided` counts them. Under `public` or `none`, do not tell the user
a partition is unavailable: the filter was not strong enough to know. Use
`--all-partitions` to see everything, or `-p NAME` for one.

## 2. What hardware is actually in there

```bash
slurmate nodes --json            # every hardware type, with counts
slurmate nodes -p gpu --gpu      # only GPU nodes reachable through 'gpu'
```

`brief` aggregates to the partition, which is right for choosing a partition
and wrong for choosing a `--constraint`: it hides that one partition spans two
node generations. `nodes` groups by what a job can ask for (cores, memory,
GPUs, features) and reports `count` and `avail` per type.

```console
$ slurmate nodes -p gpu --gpu
  23 nodes, 20 usable, 3 hardware types

      nodes  cores     memory  gpus              features
      13/16     48     180 GB  4x rtx6000        gold-6248r,rtx6000
          5     48     180 GB  4x v100           gold-6248r,v100
          2     48     375 GB  4x a100           a100,gold-6248r
```

`avail` counts only nodes in a state that can run a job, so a type that exists
but is entirely drained reads as zero. The `features` column is what
`--constraint` can name.

## 3. Why a job is not running

```bash
slurmate jobs --json
```

Every pending job carries Slurm's own `reason` **and** a `reason_explained`
saying what to do. The codes are not interchangeable, and this is the whole
point of the field:

| reason | What it means |
|---|---|
| `Priority` | others are ahead; clears on its own, a shorter `--time` moves it up |
| `Resources` | next in line, hardware is busy; asking for less starts sooner |
| `QOSMaxJobsPerUserLimit` | a cap, not a queue. Nothing clears until one of yours finishes |
| `PartitionTimeLimit` | the `--time` exceeds the partition cap, so it will **never** start |
| `DependencyNeverSatisfied` | the job it waits on failed. Cancel it |

An empty `reason_explained` means the code is not one slurmate has written
down. Report the raw code; do not invent a meaning for it.

## 4. Asking for a GPU correctly

`gpu_request_formats` splits each partition's models three ways:

- **`typed`** came from a real `gpu:MODEL:N` GRES. Ask with `--gres=gpu:MODEL:N`.
- **`feature`** appears only in the node's feature list, because the GRES is
  count-only (`gpu:4`). `--gres=gpu:MODEL:N` is **rejected outright** here
  ("Requested node configuration is not available"). Use `--gres=gpu:N` plus
  `--constraint=MODEL`.
- **`constraint`** is what `--constraint=` can name. An empty list is a measured
  answer, not a missing one: no GPU model is a node feature here.

A model can be in both `typed` and `constraint`. Check before you write.

## 5. Before handing the script over: check it

```bash
slurmate check --script job.sbatch --json     # or --script - for stdin
```

Exit `0` clean or warnings only, `1` on an error, `2` on a usage mistake. Safe
in a loop: the scheduler verdict comes from `sbatch --test-only`, which enters
no queue and spends no allocation.

```json
{"ok": false,
 "findings": [{"level": "error", "message": "no partition 'caslake' on this cluster"}],
 "scheduler": {"verdict": "refused", "reason": "More processors requested than permitted"}}
```

`findings[].level` is `error` (Slurm will reject this, or the job will die on
startup) or `warning` (it may, and an advertised figure can undercount a
heterogeneous partition). Warnings do not set `ok: false`.

`scheduler.verdict` is `accepted`, `refused` or `unavailable`, and it is the
controller's own opinion rather than slurmate's, which is why it is reported
separately.

**`ok: true` means checked and clean, never merely quiet.** When it is `false`,
look at `checked`: `false` there means nothing could be confirmed against the
cluster (no `sbatch`, no controller), so the script is *unverified* rather than
*wrong*. Say which one it was. Do not present an unverified script as passing.

Without `--script`, describe the request in flags and get the same verdict
before anything is written. `checked_script` shows the text that was judged:

```bash
slurmate check --json -p gpu -c 8 --mem 32G -t 04:00:00 -G a100:1
```

Flags and `--script` cannot be combined: `--test-only` can only judge the file
as written, so the two would disagree.

### Use it to look modules up

Do not fetch the whole module list. Guess the name, run `check`, and it names
the real ones:

```console
$ slurmate check --script train.sbatch
  ✗ module 'cuda/12.1' not found on this cluster; 'cuda' is available as:
    cuda/10.2, cuda/11.2, cuda/11.3, cuda/11.5, cuda/11.7, cuda/11.8, ... (+11 more)
```

`check` reads `module load` and `conda activate` out of the script **body**, not
just the `#SBATCH` block. `brief --full` has the complete lists under
`environment`, but it costs about five extra seconds.

## 6. Writing the script

Write the `#SBATCH` block yourself, or have slurmate write it:

```bash
slurmate --print -p gpu -c 8 --mem 32G -t 04:00:00 -G a100:1 \
  --command "python train.py" > job.sbatch
```

`--print` puts only the script on stdout. It never submits.

## Traps this catches that the scheduler does not

- **A log path on node-local `/tmp`.** The job writes to the compute node's own
  copy, the submitter sees nothing, and it reports `COMPLETED 0:0` with no log.
- **A module or conda env that does not exist here.** The job queues, is
  scheduled, starts, and dies on `module load` minutes or hours later.
- **A directive given twice.** Slurm accepts `-J first -J second` and silently
  honours the last, so the script reads one way and runs another.
- **A partition whose nodes are all drained.** It advertises capacity; nothing
  runs.

## Rules

1. Run `brief` **first**, on every new cluster, before proposing any script.
   Reach for `nodes` when the choice is about hardware, and `jobs` when the
   question is about something already submitted.
2. If `mock` is `true`, stop and say the cluster could not be read.
3. Never invent a partition, account, QoS or GPU model. Use what `brief`
   returned, and read `partition_filter` before calling anything unavailable.
4. Run `check` before handing a script over. Fix what it reports, run it again,
   and say what the verdict was.
5. Do not run `sbatch` or `slurmate --yes` yourself. Submitting spends the
   user's allocation; that is their call.
