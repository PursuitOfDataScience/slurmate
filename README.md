<div align="center">

```
███████╗██╗     ██╗   ██╗██████╗ ███╗   ███╗ █████╗ ████████╗███████╗
██╔════╝██║     ██║   ██║██╔══██╗████╗ ████║██╔══██╗╚══██╔══╝██╔════╝
███████╗██║     ██║   ██║██████╔╝██╔████╔██║███████║   ██║   █████╗  
╚════██║██║     ██║   ██║██╔══██╗██║╚██╔╝██║██╔══██║   ██║   ██╔══╝  
███████║███████╗╚██████╔╝██║  ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗
╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
```

### ⚡ Stop hand-writing `sbatch` scripts. Let the wizard do it.

A fast, friendly **TUI wizard + CLI** that builds and submits Slurm batch jobs —
on any cluster, as long as `sbatch` is on your `PATH`.

[![CI](https://github.com/PursuitOfDataScience/slurmate/actions/workflows/ci.yml/badge.svg)](https://github.com/PursuitOfDataScience/slurmate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/slurmate.svg?cache=0)](https://pypi.org/project/slurmate/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](#-license)
[![Status: beta](https://img.shields.io/badge/status-beta-orange.svg)](#-status)
[![Linter: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

---

<p align="center">
  <img src="assets/demo.gif" width="840" alt="slurmate building a GPU sbatch job in the TUI wizard, with a live #SBATCH preview">
</p>

---

## ✨ Why Slurmate?

Writing `#SBATCH` headers by hand is fiddly and error-prone: which partition has
H100s? what's the memory limit? did I get the `--gres` syntax right for *this*
cluster? Slurmate turns that into a guided conversation — it reads your cluster
live, validates as you go, and hands you a clean, ready-to-submit script.

```bash
slurmate
```

That's it. Answer a few prompts, watch the script build itself in a live
preview, and submit — or save it for later.

---

## 🚀 Quick start

```bash
# Recommended: isolated install
pipx install slurmate

# or plain pip
pip install slurmate
```

<details>
<summary>Install from source / for development</summary>

```bash
git clone https://github.com/PursuitOfDataScience/slurmate.git
cd slurmate
pip install -e ".[dev]"     # editable + dev tools (pytest, ruff, mypy)
```

</details>

### Interactive mode (the TUI)

```bash
slurmate            # or: python -m slurmate
```

A full-screen wizard walks you through name → resources → environment →
command. The generated script grows **live** in a preview pane as you answer.
A final **Review & Submit** screen shows your full configuration and the
generated script side by side, then a single menu lets you **submit**, **go
back to edit your answers**, **open the script in `$EDITOR`**, **save** it, or
**quit**.

> 💡 You can leave any step blank and come back to it — anything still missing is
> flagged before you submit. `Esc` / `Shift+Tab` go back at any step (including
> from the action menu, to re-edit your answers); navigation is fully
> keyboard-driven.

### Batch mode (scriptable, no TUI)

```bash
slurmate \
  --job-name train_job \
  --partition gpu \
  --cpus 8 --memory 32G --time 04:00:00 \
  --gpus 1 --gpu-type h100 \
  --command "python train.py"
```

Slurm's own spellings work too, so a command line or script copied from
`sbatch` needs no translation — the short flags (`-J -A -p -q -t -N -c -a -C -G
-o`), `--mem`, `--cpus-per-task`, `--output`, and all four GPU renderings
(`--gpus h100:1`, `--gres gpu:h100:1`, `--gpus-per-node`, `--gpus-per-task`):

```bash
slurmate -J train_job -p gpu -c 8 --mem 32G -t 04:00:00 -G h100:1 \
  --command "python train.py"
```

Submit immediately, no prompts:

```bash
slurmate --partition gpu --command "python train.py" --yes
```

Just want the script? `--print` emits **only** the raw script (great for piping
or CI):

```bash
slurmate --partition gpu --command "python train.py" --print > job.sbatch
```

Want a full preview without submitting? `--dry-run` shows the summary panel,
partition-limit warnings, SU/ETA, and any missing-field reminders — everything
except the actual submit:

```bash
slurmate --partition gpu --command "python train.py" --dry-run
```

Batch mode kicks in as soon as you pass any job-defining flag (or `--yes`); a
bare `slurmate` still launches the wizard. If a config file supplies the job,
`--print` and `--dry-run` also render non-interactively from it — so
`slurmate --print` with a `.slurmate.toml` present emits the script straight to
stdout without opening the wizard. `--yes` requires a command to run (it refuses
to submit an empty, no-op job). Every submit also saves a `<job>-<id>.sh` copy
next to where you ran it — pass `--no-save-script` (or set `SLURMATE_NO_SAVE=1`)
to skip that.

The wizard needs a real terminal on both ends. In a pipe or a CI runner
(`slurmate | tee setup.log`) it says so and points at the flags above, rather
than rendering into a stream nobody can type into.

### Checked against *this* cluster

A generated script is only useful if it is correct for the cluster you are on,
so the partition and account you name are checked against the live cluster
before anything is written:

```console
$ slurmate --print --partition caslake --cpus 2 --time 01:00:00 --command ./run.sh
  ✗ Error: no partition 'caslake' on this cluster.
    Did you mean: broadwl (default), build, bigmem2?
    This cluster's partitions: broadwl, build, bigmem2, gpu2, ... (+4 more)
  Pass --force to generate the script anyway (e.g. for another cluster).
```

Writing a script to carry somewhere else is a legitimate thing to do — that is
what `--force` is for; it downgrades the check to a warning. The default just
is not silent.

`--dry-run` additionally reports what Slurm itself says about the request. A
job the scheduler has already refused gets the refusal, not a wait time:

```
│ ETA:  never — More processors requested than permitted │
```

Run `slurmate --help` for the full flag list.

---

## 🎯 Features

| | |
|---|---|
| 🧠 **Live cluster awareness** | Pulls real partitions, GPU types, QoS, accounts, conda envs, and modules from `sinfo` / `scontrol` / `sacctmgr` / `conda`. |
| 👀 **Live preview** | The `#SBATCH` script builds incrementally as you answer — what you see is exactly what gets submitted. |
| 🛡️ **Partition-aware validation** | Inline warnings when CPU / memory / time / GPU requests exceed the selected partition's limits, plus a hard error for a partition or account this cluster does not have (`--force` to override). Under `--dry-run`, Slurm's own `--test-only` verdict is reported instead of an ETA for a job it would refuse. |
| 🩺 **Usable capacity, not node counts** | The picker counts only nodes that can actually run a job, so a partition whose nodes are all `down`/`drained` reads `unavailable` instead of advertising capacity nothing can use. Ranked by the site default, then your own associations, then usable capacity. |
| 📏 **Measured defaults** | An unspecified `--mem` is sized from the partition's own node memory (the same share of memory as of cores) rather than a literal that only fits the cluster slurmate was written on. |
| 📁 **Path autocomplete** | `Tab`-complete file paths while typing your command, virtualenv path, or output files — no more retyping long project paths. |
| ↩️ **Skip & come back** | Leave steps blank, navigate freely with `Esc`, and get reminded of anything missing before submit. |
| 📋 **Copy-friendly** | Mouse capture is off so you can select/copy the preview natively; navigation is fully keyboard-driven. |
| 🧩 **Cluster-agnostic GPU syntax** | Five formats — `--gres=gpu:type:N`, `--gres` + `--constraint`, `--gpus`, `--gpus-per-node`, `--gpus-per-task` — and slurmate flags a GPU model that a site only exposes as a node feature, where a typed `--gres` would be rejected. |
| 🐍 **Env activation** | Conda, Mamba, virtualenv, or none — generated automatically. |
| 🗂️ **Smart output paths** | Set a custom log name/pattern (`%j` = job ID, `%A`/`%a` = array job/task); error path is derived and log dirs are auto-created. Array jobs default to the `%A_%a` pattern. |
| ♻️ **Reproducible** | Every submission is saved locally as `<job>-<job-id>.sh`; you can also save manually or edit in `$EDITOR` before submitting. |
| 🧪 **Safe to explore** | No Slurm? It falls back to realistic mock data so you can try the whole flow anywhere. |

---

## ⚙️ Configuration file

Stop retyping the same account and partition every run. Slurmate reads defaults
from a TOML file (first match wins):

1. `.slurmate.toml` in the current directory
2. `~/.config/slurmate/config.toml`

These prefill the wizard **and** act as fallbacks in batch mode. Explicit CLI
flags always win.

```toml
# .slurmate.toml — keys may be top-level or under a [defaults]/[slurmate] table
account     = "my_lab"
partition   = "gpu-shared"
cpus        = 8
memory      = "32G"
time_limit  = "04:00:00"
gpu_format  = "gres_type"            # gres_type | constraint | gpus | gpus_per_node | gpus_per_task
constraint  = "gpu"                  # node feature / Slurm -C (e.g. Perlmutter's cpu|gpu)
mem_per_cpu = "2G"                   # --mem-per-cpu; overrides `memory` when set
env_type    = "conda"                # conda | mamba | venv | none
modules     = ["cuda/12.1", "gcc/9.3.0"]
output_dir  = "logs"
```

**Recognized keys:** `job_name`, `account`, `partition`, `qos`, `cpus`, `memory`,
`mem_per_cpu`, `time_limit`, `nodes`, `ntasks_per_node`, `gpus`, `gpu_type`,
`gpu_format`, `constraint`, `array_spec`, `modules`, `env_type`, `env_name`,
`output_dir`, `output_file`, `command`, `custom_sbatch`.

Every one of them is also a wizard step, so a config file prefills the
interactive flow and batch mode identically.

**CLI spellings work too.** `time` is accepted for `time_limit`, `array` for
`array_spec`, and any dashed form (`job-name`, `mem-per-cpu`,
`ntasks-per-node`, …) for its underscored key — so a key copied from `--help`
does what it looks like it does.

**Anything else is reported, not dropped.** A key outside the list above gets a
named warning with the likely intent, instead of being silently discarded:

```
slurmate: ./.slurmate.toml: unknown key 'partitions' — did you mean 'partition'?
slurmate: ./.slurmate.toml: ignoring unknown section '[job]' — put keys at the top level or under [defaults]/[slurmate]
```

**The file that supplied the defaults is named.** A `.slurmate.toml` travels
with a project into git and onto whatever cluster it is next checked out on, so
slurmate says where the values came from — on stderr at load, and in the
`--dry-run` summary, listing only the keys no flag overrode:

```
slurmate: using defaults from ./.slurmate.toml: partition, account, cpus, time_limit
  Defaults from ./.slurmate.toml: partition, account, time_limit (flags override the file).
```

`--print` keeps stdout script-only; the disclosure goes to stderr.

**Plain output:** `--ascii` (or `SLURMATE_ASCII=1`) renders status markers as ASCII
(`!`, `x`, `+`) instead of `⚠ ✗ ✓`. It is applied automatically when the terminal's
encoding cannot carry them, so a non-UTF-8 locale degrades rather than failing.

**Config file locations** are `$XDG_CONFIG_HOME/slurmate/config.toml` (falling back to
`~/.config/slurmate/config.toml`) and `./.slurmate.toml`, merged in that order — so a
global config still works in an environment with no resolvable home directory, such as
a job launched with `sbatch --export=NONE`.

Keys may sit at the top level or under a `[defaults]` or `[slurmate]` table.
When the same key appears in more than one place, the effective precedence is
**`[slurmate]` > `[defaults]` > top-level** (a later table wins). Explicit CLI
flags always override the file.

> Real TOML is always used on supported Pythons (`tomllib` on 3.11+, the `tomli`
> dependency on 3.10). A minimal flat `key = value` reader exists only as a
> last-resort fallback.

---

## 🔧 Environment variables

| Variable | Effect |
|---|---|
| `SLURMATE_MOCK=1` | Force mock mode even when Slurm is installed (great for demos/tests). |
| `SLURMATE_GPU_FORMAT` | Default GPU syntax: `gres_type` (default) · `constraint` · `gpus` · `gpus_per_node` · `gpus_per_task`. |
| `SLURMATE_LOG_DIR=…` | Save the submitted script there (instead of the working dir) for reproducibility. |
| `SLURMATE_NO_SAVE=1` | Don't auto-save a `<job>-<id>.sh` copy on submit (same as `--no-save-script`). |
| `SLURMATE_NO_BANNER=1` | Hide the startup banner. |
| `SLURMATE_BANNER_ANIMATE=1` | Animate the startup banner (needs a real TTY; ignored when output is piped). |
| `SLURMATE_DEBUG=1` | Verbose debug logging. |

`NO_COLOR` and non-TTY output are respected automatically; `FORCE_COLOR=1`
forces colour on for both the `rich` panels and the plain status lines.

---

## 🛠️ How it works

1. **Gather** — query the cluster (or fall back to mock data) for partitions,
   limits, GPU types, environments, and modules.
2. **Guide** — a keyboard-first wizard collects name, resources, dependencies,
   and the command, validating against the chosen partition as you go.
3. **Generate & submit** — produce a clean `#SBATCH` script, optionally edit it
   in `$EDITOR`, then pipe it straight to `sbatch` (or save / print it).

---

## 🧪 Status

Slurmate is **beta** and pre-1.0 — the CLI, config keys, and defaults may
change between releases. It's already useful day-to-day; pin a version if you
script around it. Bug reports and cluster-specific quirks are very welcome.

---

## 🤝 Contributing

Issues and PRs are welcome! For local development:

```bash
pip install -e ".[dev]"
ruff check src/        # lint
mypy src/              # types (strict)
pytest                 # tests
```

CI runs the same three checks on Python 3.10–3.13 for every push and PR.

---

## 📄 License

Released under the [MIT License](LICENSE).
