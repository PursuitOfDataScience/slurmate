# Slurmify

Interactive TUI wizard and CLI utility for generating and submitting Slurm `sbatch` scripts.

## Installation

From PyPI:

```bash
pipx install slurmify
# or
pip install slurmify
```

From source:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

### Interactive Mode (TUI)

Simply run:

```bash
slurmify
```

Follow the interactive prompts in the full-screen TUI to configure your job. Once finished, you will be prompted to edit the script in your editor and/or submit the job to Slurm.

### Batch Mode (Non-interactive)

You can pass arguments to run in batch mode directly without loading the TUI:

```bash
slurmify --job-name "train_job" --partition "gpu" --cpus 8 --memory "32G" --time "04:00:00" --gpus 1 --command "python train.py"
```

To skip the final submission confirmation prompt and submit immediately:

```bash
slurmify --partition "gpu" --command "python train.py" --yes
```

Use `slurmify --help` to see all available CLI options.

## Features & Configurable Behavior

- **Live Autocompletion:** Fetches live partitions, conda environments, user accounts, and available modules from the cluster.
- **File-Path Completion:** When typing the command, a virtualenv path, or output paths, Tab completes filesystem paths (the last token), so you don't retype long project paths.
- **Skip & Come Back:** Leave any step blank and continue; missing recommended fields are flagged at the final review before you submit. Press Esc (or Shift+Tab) to go back.
- **Copy the Preview:** Press F2 in the TUI to release mouse capture, then select/copy the script with your terminal as usual; F2 again restores click/scroll.
- **Auto-Directory Creation:** If output or error logs are configured to be saved inside subdirectories (e.g. `logs/job-%j.out`), those directories are automatically created before submission.
- **Custom Output Names:** Set an output file name/pattern (`%j` = job ID) via the wizard or `--output-file`; the error path is derived automatically.
- **Partition-Aware Validation:** Inline warnings in the TUI when your CPU, memory, time, or GPU selections exceed limits of the selected partition.

## Environment Variables

Configure Slurmify's behavior using the following environment variables:

- `SLURMIFY_MOCK=1`: Force mock mode even if Slurm commands are available on `PATH`.
- `SLURMIFY_NO_BANNER=1`: Suppress the colorful ASCII banner at startup.
- `SLURMIFY_BANNER_ANIMATE=1`: Force the banner animation even when not in a standard TTY.
- `SLURMIFY_LOG_DIR=/path/to/logs`: If set, a copy of every successfully submitted script is saved to this directory for reproducibility.
- `SLURMIFY_DEBUG=1`: Enable detailed debug log statements.
- `SLURMIFY_GPU_FORMAT`: Set the formatting style for GPU requests. Supported options:
  - `constraint` (default): Emits `#SBATCH --gres=gpu:N` and `#SBATCH --constraint=type`.
  - `gres_type`: Emits `#SBATCH --gres=gpu:type:N`.
  - `gpus`: Emits `#SBATCH --gpus=type:N` (or `#SBATCH --gpus=N`).

## Configuration file

To avoid retyping the same values every run, Slurmify reads default values from a
TOML config file. It looks for, in order (first match wins):

1. `.slurmify.toml` in the current directory
2. `~/.config/slurmify/config.toml`

These defaults are used by **both** the interactive wizard (as prefilled values)
and batch mode (as fallbacks for any flag you don't pass). Explicit CLI flags
always override the config file.

Keys may sit at the top level or under a `[defaults]` table. Example:

```toml
account = "my_lab"
partition = "gpu-shared"
cpus = 8
memory = "32G"
time_limit = "04:00:00"
gpu_format = "gres_type"      # gres_type | constraint | gpus
env_type = "conda"            # conda | mamba | venv | none
modules = ["cuda/12.1", "gcc/9.3.0"]
output_dir = "logs"
```

Recognized keys: `job_name`, `account`, `partition`, `qos`, `cpus`, `memory`,
`time_limit`, `nodes`, `ntasks_per_node`, `gpus`, `gpu_type`, `gpu_format`,
`array_spec`, `modules`, `env_type`, `env_name`, `output_dir`, `output_file`,
`command`, `custom_sbatch`.

> Real TOML parsing is used when available (`tomllib` on Python 3.11+, `tomli`
> on older versions). If neither is installed, a minimal flat `key = value`
> reader is used as a fallback — in that mode `[section]` headers are ignored.

## How it works

1. **Information Gathering:** Fetches partition limits, Conda/Mamba environments, and module data dynamically. If Slurm or Conda is not available, falls back to realistic mock metadata.
2. **Interactive Wizard:** A terminal-based form that guides you through name, resources, dependencies, and shell commands.
3. **Reproducible Script Generation:** Produces clean `#SBATCH` scripts. If configured, launches a temporary file inside your `$EDITOR` (e.g., `vim` or `nano`) for manual post-generation tweaks before piping directly to `sbatch`.
