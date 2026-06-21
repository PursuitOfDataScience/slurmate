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
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](#-license)
[![Status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)](#-status)
[![Linter: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

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
slurmate
```

A full-screen wizard walks you through name → resources → environment →
command. The generated script grows **live** in a preview pane as you answer,
and when you're done you get a single menu: **Submit · Edit · Save · Show ·
Quit**.

> 💡 You can leave any step blank and come back to it — anything still missing is
> flagged before you submit. `Esc` / `Shift+Tab` go back; `F1` opens help.

### Batch mode (scriptable, no TUI)

```bash
slurmate \
  --job-name train_job \
  --partition gpu \
  --cpus 8 --memory 32G --time 04:00:00 \
  --gpus 1 --gpu-type h100 \
  --command "python train.py"
```

Submit immediately, no prompts:

```bash
slurmate --partition gpu --command "python train.py" --yes
```

Just want the script? Print it (great for piping or CI):

```bash
slurmate --partition gpu --command "python train.py" --print > job.sbatch
```

Run `slurmate --help` for the full flag list.

---

## 🎯 Features

| | |
|---|---|
| 🧠 **Live cluster awareness** | Pulls real partitions, GPU types, QoS, accounts, conda envs, and modules from `sinfo` / `scontrol` / `sacctmgr` / `conda`. |
| 👀 **Live preview** | The `#SBATCH` script builds incrementally as you answer — what you see is exactly what gets submitted. |
| 🛡️ **Partition-aware validation** | Inline warnings when CPU / memory / time / GPU requests exceed the selected partition's limits. |
| 📁 **Path autocomplete** | `Tab`-complete file paths while typing your command, virtualenv path, or output files — no more retyping long project paths. |
| ↩️ **Skip & come back** | Leave steps blank, navigate freely with `Esc`, and get reminded of anything missing before submit. |
| 📋 **Copy-friendly** | Mouse capture is off by default so you can select/copy the preview natively (`F2` toggles mouse nav). |
| 🧩 **Cluster-agnostic GPU syntax** | Choose `--gres=gpu:type:N`, `--gres` + `--constraint`, or `--gpus` to match your site. |
| 🐍 **Env activation** | Conda, Mamba, virtualenv, or none — generated automatically. |
| 🗂️ **Smart output paths** | Set a custom log name/pattern (`%j` = job ID); error path is derived and log dirs are auto-created. |
| ♻️ **Reproducible** | Save the script, edit it in `$EDITOR`, or keep a copy of every submission. |
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
# .slurmate.toml — keys may be top-level or under a [defaults] table
account     = "my_lab"
partition   = "gpu-shared"
cpus        = 8
memory      = "32G"
time_limit  = "04:00:00"
gpu_format  = "gres_type"            # gres_type | constraint | gpus
env_type    = "conda"                # conda | mamba | venv | none
modules     = ["cuda/12.1", "gcc/9.3.0"]
output_dir  = "logs"
```

**Recognized keys:** `job_name`, `account`, `partition`, `qos`, `cpus`, `memory`,
`time_limit`, `nodes`, `ntasks_per_node`, `gpus`, `gpu_type`, `gpu_format`,
`array_spec`, `modules`, `env_type`, `env_name`, `output_dir`, `output_file`,
`command`, `custom_sbatch`.

> Real TOML is used when available (`tomllib` on 3.11+, `tomli` otherwise);
> without either, a minimal flat `key = value` reader is used as a fallback.

---

## 🔧 Environment variables

| Variable | Effect |
|---|---|
| `SLURMATE_MOCK=1` | Force mock mode even when Slurm is installed (great for demos/tests). |
| `SLURMATE_GPU_FORMAT` | Default GPU syntax: `gres_type` (default) · `constraint` · `gpus`. |
| `SLURMATE_LOG_DIR=…` | Save a copy of every submitted script there for reproducibility. |
| `SLURMATE_NO_BANNER=1` | Hide the startup banner. |
| `SLURMATE_BANNER_ANIMATE=1` | Force the animated banner even when not a TTY. |
| `SLURMATE_DEBUG=1` | Verbose debug logging. |

`NO_COLOR` and non-TTY output are respected automatically.

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

Slurmate is **experimental** and pre-1.0 — the CLI, config keys, and defaults may
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

CI runs the same three checks on Python 3.10–3.12 for every push and PR.

---

## 📄 License

Released under the [MIT License](LICENSE).
