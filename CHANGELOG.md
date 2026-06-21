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
- Validation — memory and time-limit parsing with user-friendly warnings.
- GPU type detection via Sinfo features with negative filtering (rejects
  CPU/infrastructure tokens instead of maintaining a hardcoded allowlist).
- Post-wizard action menu (Submit / Edit / Save / Show / Quit).
- `questionary`-theme color constants and ASCII banner.

### Fixed

- Output file extension inconsistency — bare filenames now get `.out`/`.err`.
- GPU type case sensitivity — all comparisons are case-insensitive.
- Non-GPU features falsely detected as GPU types — features are only scanned
  when GRES confirms the node has GPUs.
- "Any" GPU type warning confusing users — warning skipped when type is `Any`.
- "Any" generating invalid `--gres=gpu:Any:N` — now omits type from gres line.
- False GPU type warning when the selected type is actually in the partition
  list — fallback to `fetch_gpu_types_for_partition()`.
- Conda activation using confusing `$(conda info --base)` subshell syntax —
  replaced with `source activate`.
- Modules displayed on one line in summary causing wrapping.
- Command step subtitle not mentioning multiline support.
- GPU type detection only from GRES (missed count-only nodes) — added features
  scanning fallback.
- Conda environment autocomplete (`FuzzyWordCompleter` with fetched envs).
- Multiline command — Enter with `eager=True` intercepted before TextArea could
  insert newline; now checks `multiline` flag first.
- Modules autocomplete broken for comma-separated entry — added
  `LastTokenCommaCompleter`.
- Module list re-rendered with Python brackets on step-back.
- Modules multi-entry workflow — Enter with completion appends `", "`; Tab
  cycles completions, advances only when none found; `Ctrl+G` advances from
  multiline steps.
- Input lost on step-back and Tab advancing prematurely on multiline steps.
- TUI crash when `gpu_type`/`env_name`/`partition` were `None` (TextArea
  rejected `None`).
- Box borders broken — replaced `expand=False` with explicit `width=`.
- `#SBATCH` directives emitted in wrong order (now matches wizard order).
- Auto-derived `--output`/`--error` shown in preview before user configured
  output — now hidden until output dir/file is set.
- Live preview height — now fills available space.
- Mouse capture ON by default (prevented text selection) — now OFF, F2 toggles.
- One-way edit/submit/save confirm chain — replaced with navigable action menu.
- `qos "Default (none)"` leaking into script and summary.
- Banner trailing spaces lost on E rows 3–4.

[Unreleased]: https://github.com/PursuitOfDataScience/slurmate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PursuitOfDataScience/slurmate/releases/tag/v0.1.0
