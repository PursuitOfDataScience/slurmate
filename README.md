<div align="center">

# 🧰 slurmate

**Stop hand-writing `sbatch` scripts. Let the wizard do it.**

[![CI](https://github.com/PursuitOfDataScience/slurmate/actions/workflows/ci.yml/badge.svg)](https://github.com/PursuitOfDataScience/slurmate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/slurmate.svg?cache=0)](https://pypi.org/project/slurmate/)
[![PyPI downloads](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/PursuitOfDataScience/slurmate/badges/downloads.json)](https://pypi.org/project/slurmate/)

<img src="assets/demo.gif" width="840" alt="slurmate building a GPU job in the wizard, with the #SBATCH script updating live beside each answer">

</div>

## ✨ Install

```bash
pipx install slurmate     # or: uv tool install slurmate / pip install slurmate
slurmate                  # start the wizard
```

Needs Python 3.10+. The wizard only offers what your cluster actually has: real partitions,
QOS, GPU types and modules.

## 🧰 Use

```bash
slurmate --partition gpu --command "python train.py" --dry-run
slurmate --partition gpu --command "python train.py" --print > job.sbatch
slurmate --partition gpu --command "python train.py" --yes
slurmate -J train_job -p gpu -c 8 --mem 32G -t 04:00:00 -G h100:1 --command ./run.sh
slurmate check --script train.sbatch
slurmate nodes -p gpu --gpu
```

| Flag | Does |
|---|---|
| `--dry-run` | show the script and what would happen |
| `--print` | write the script to stdout |
| `--yes` | submit without asking |

For AI agents, `slurmate skill --install` adds a Claude Code skill. Add `--format agents` or
`--format mcp` for other tools.

## ⚙️ Configuration file

Put your defaults in TOML. **Both files are read and merged per key**: the project file
wins any key it sets, and the global file fills in the rest.

1. `.slurmate.toml` in the current directory
2. `~/.config/slurmate/config.toml`

Command-line flags always win over both.

```toml
account    = "my_lab"
partition  = "gpu-shared"
cpus       = 8
memory     = "32G"
time_limit = "04:00:00"
modules    = ["cuda/12.1", "gcc/9.3.0"]
```

**Recognized keys:** `job_name`, `account`, `partition`, `qos`, `cpus`, `memory`,
`mem_per_cpu`, `time_limit`, `nodes`, `ntasks_per_node`, `gpus`, `gpu_type`,
`gpu_format`, `constraint`, `array_spec`, `modules`, `env_type`, `env_name`,
`output_dir`, `output_file`, `command`, `custom_sbatch`. Each is also a wizard step.

**CLI spellings work too.** `time` is accepted for `time_limit`, `array` for `array_spec`,
`env` for `env_name`, and dashed forms like `mem-per-cpu` for their underscored key.

A misspelled key gets a warning with a suggestion, never silence.

## 📌 Good to know

🎛️ **GPU request syntax varies by site.** Set `gpu_format` to match yours.

🔍 **Check before you submit.** `slurmate check` validates an existing script against this
cluster.

## 🧪 Status

Beta. The flags and the config file are stable; the wizard's screens still change.

## 📄 License

MIT
