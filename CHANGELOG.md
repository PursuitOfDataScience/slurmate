# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org).

## [0.2.0] — 2026-06-21

### Added

- Conda environment autocomplete — `_setup_env_name` fetches conda envs via
  `fetch_conda_envs()` and sets `FuzzyWordCompleter` with the results. (#14)

### Fixed

- Output file extension inconsistency — bare filenames now get `.out` appended
  (and `.err` for stderr). (#1)
- Hardcoded GPU type list — replaced positive-pattern-matching with negative
  filtering that rejects CPU/infrastructure tokens instead of matching against
  a static allowlist. (`system_utils.py:_detect_gpu_type`) (#2, #6)
- GPU type case sensitivity — all comparisons made case-insensitive. (#3)
- Broken box borders on panels — raw ANSI escape codes (`c.PINK`, `c.CYAN`)
  in Rich Panel titles caused Rich to ignore the `width=` parameter and
  auto-size incorrectly. Replaced with Rich-native style names
  (`bold #ff0080`, `bold cyan`). Previously `expand=False` was replaced with
  explicit `width=` but that alone was insufficient. (#4)
- Non-GPU features falsely detected as GPU types — features now only scanned
  when GRES contains `gpu:`. (#5)
- "Any" GPU type generating a confusing warning — warning skipped when
  `gpu_type == "any"`. (#7)
- "Any" generating invalid `#SBATCH --gres=gpu:Any:N` — now generates
  `#SBATCH --gres=gpu:N` without type restriction and skips `--constraint`
  entirely. (#8)
- False GPU type warning when the selected type is in the partition list —
  `_validate_partition_limits` falls back to `fetch_gpu_types_for_partition()`
  when static `part.gpu_types` doesn't contain the selected type. (#9)
- Confusing conda activation syntax — replaced `$(conda info --base)`
  subshell with `source activate`. (#10)
- Modules wrapping in summary panel — `width=summary_w + 4` accounts for
  borders and padding. (#11)
- Command step subtitle not mentioning multiline support — updated to
  `"(Enter=newline, Tab=complete, Ctrl+G=next)"`. (#12)
- GPU type detection only from GRES (missed count-only nodes) — added features
  scanning fallback. (#13)
- Multiline command Enter behavior — `eager=True` intercepted Enter before
  TextArea could insert a newline; now checks `multiline` flag first and calls
  `buf.insert_text("\\n")`. (#15)
- Modules autocomplete broken for comma-separated entry — added
  `LastTokenCommaCompleter` that extracts only the last comma-separated token
  for fuzzy-matching. (#16)
- Module list re-rendered with Python brackets on step-back — added
  `isinstance(prev, list)` check that joins with `", ".join(prev)`. (#17)
- Modules multi-entry workflow — Enter with a completion appends `", "`
  automatically; footer cleaned up with consistent key names. (#18)
- Input lost on step-back and Tab advancing prematurely on multiline steps —
  `_go_back()` now saves current input before navigating; Tab handler calls
  `buf.complete_next()` and only advances when `complete_state` is None;
  `Ctrl+G` added as universal next-step key. (#19)
- CI failing — removed unused `Frame` import; fixed generator return type
  annotations on `LastTokenPathCompleter` and `LastTokenCommaCompleter`. (#20)
- Tab still advancing from multiline command step — async `PathCompleter`
  hadn't populated `complete_state` by the time the eager Tab handler checked
  it. Tab now only completes on multiline steps and never navigates away;
  `Ctrl+G` advances. (#21)
- TUI crash when `gpu_type`/`env_name`/`partition` were `None` (TextArea
  rejected `None`).
- `#SBATCH` directives emitted in wrong order — now matches wizard step order.
- Auto-derived `--output`/`--error` shown in preview before output configured
  — now hidden until output dir/file is set.
- Live preview height — now fills available space.
- Mouse capture ON by default (prevented text selection) — now OFF, F2 toggles.
- One-way edit/submit/save confirm chain — replaced with navigable action menu
  (Submit / Edit / Save / Show / Quit).
- `qos "Default (none)"` leaking into script and summary.

### Changed

- Consolidated three different memory parsing grammars into unified
  `validate_memory` / `normalize_memory`.
- `build_from_answers()` helper created to eliminate 14-argument duplication
  across `tui.py` and `main.py`.

## [0.1.0] — 2026-06-20

### Added

- Interactive TUI wizard with form steps for job name, partition, account,
  QoS, CPU cores, memory, time limit, nodes, per-task/node, GPUs, GPU type,
  GPU format, array spec, modules, conda environment, custom sbatch flags,
  output dir/file, and command.
- Live script preview that grows incrementally as the user fills in each step.
- Batch/CLI mode via flags (`--partition`, `--cpus`, `--command`, etc.).
- Slurm integration — `fetch_partitions()`, `fetch_gpu_types_for_partition()`,
  `fetch_queue_eta()`, `submit_sbatch()`.
- Memory and time-limit validation with user-facing warnings.
- GPU type detection via Sinfo features.
- Color theme constants and ASCII banner.
- Config file support (`~/.config/slurmate/slurmate.json`).
- Queue status panel in TUI showing running/waiting jobs and ETA.
- Post-submission hints (job ID, log path, `squeue`/`scancel` commands).
- `--version` flag.
- Tasks-per-node support.
- `ruff` and `mypy` CI checks.
- Test suite with fixtures for partition, queue, and GPU type parsing.

[0.2.0]: https://github.com/PursuitOfDataScience/slurmate/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/PursuitOfDataScience/slurmate/releases/tag/v0.1.0
