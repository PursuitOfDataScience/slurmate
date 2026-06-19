# slurmify

Interactive TUI wizard for generating and submitting Slurm `sbatch` scripts.

## Installation

```bash
pip install .
```

## Usage

```bash
slurmify
```

Follow the interactive prompts to configure your job, review the generated
script, and submit it to Slurm.

## How it works

1. Fetches live partition data from `sinfo`
2. Fetches available Conda environments
3. Guides you through resource specification (CPUs, memory, time, GPUs)
4. Builds a complete `#SBATCH` script with syntax highlighting
5. Pipes the script directly to `sbatch` — no temporary files written

If `sinfo`, `sbatch`, or `conda` are not on `PATH`, mock data is returned so
you can develop and explore the UI without a Slurm cluster.

To force mock mode even with Slurm available:
```bash
SLURMIFY_MOCK=1 slurmify
```
