# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Fixed

- Box borders not closing in script/summary panels — raw ANSI escape codes
  (`c.PINK`, `c.CYAN`) in Rich Panel titles caused Rich to ignore the `width=`
  parameter and auto-size incorrectly. Replaced with Rich-native style names.
  (`main.py`)
- Tab advancing from multiline command step — async `PathCompleter` hadn't
  populated `complete_state` by the time the eager Tab handler checked it.
  Tab now only completes on multiline steps and never navigates away;
  `Ctrl+G` advances. (`tui.py`)
- CI failing — removed unused `Frame` import, fixed generator return type
  annotations (`LastTokenPathCompleter`, `LastTokenCommaCompleter`).
  (`tui.py`)

### Added

- Track `issues.md` on GitHub (removed from `.gitignore`).

## [0.1.0] — 2026-06-20

### Added

- Interactive TUI wizard with form steps for job name, partition, account,
  QoS, CPU cores, memory, time limit, nodes, tasks-per-node, GPUs, GPU type,
  GPU format, array spec, modules, conda environment, custom sbatch flags,
  output dir/file, and command.
- Live script preview that grows incrementally as the user fills in each step.
- Batch/CLI mode via flags (`--partition`, `--cpus`, `--command`, etc.).
- Slurm integration — `fetch_partitions()`, `fetch_gpu_types_for_partition()`,
  `fetch_queue_eta()`, `submit_sbatch()`.
- Memory and time-limit validation with user-facing warnings.
- GPU type detection via node features with negative filtering (rejects
  CPU/infrastructure tokens rather than matching against a hardcoded allowlist).
- Post-wizard action menu (Submit / Edit / Save / Show / Quit).
- Color theme constants and ASCII banner.
- Output file extension handling (`.out` / `.err`).
- Case-insensitive GPU type comparisons throughout the codebase.
- GRES-aware GPU scanning — features only inspected when `gpu:` is present.
- Graceful handling of `"Any"` GPU type (omitted from `--gres`, no spurious
  warnings).
- `source activate` syntax for conda environment activation.
- Summary panel width accounting for module lines.
- Multiline command step with `Enter` inserting newlines.
- Comma-separated module autocomplete via `LastTokenCommaCompleter`.
- Step-back preserving current input across navigation.
- `Ctrl+G` as universal next-step key for multiline steps.
- Config file support (`~/.config/slurmate/slurmate.json`).
- Queue status panel in TUI showing running/waiting jobs and ETA.
- Post-submission hints (job ID, log path, `squeue`/`scancel` commands).
- `--version` flag.
- Tasks-per-node support.
- Ordered `#SBATCH` directives matching wizard step order.
- Hidden auto-derived `--output`/`--error` until output is configured by user.
- Responsive live preview height.
- Mouse capture opt-in (off by default; F2 toggles).
- `ruff` and `mypy` CI checks.
- Test suite with fixtures for partition, queue, and GPU type parsing.

[Unreleased]: https://github.com/PursuitOfDataScience/slurmate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PursuitOfDataScience/slurmate/releases/tag/v0.1.0
