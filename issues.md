# Issues & Improvements — Next Release (v0.3.0) Planning

A thorough audit of the codebase after the v0.2.1 PyPI release. This is a
planning backlog only — **no code has been changed**. Items are grouped by
priority and each carries a symptom, the source location, and a suggested fix.
Items marked **(confirmed)** were reproduced by running the code; the rest are
by code inspection.

**Baseline verified 2026-06-22:** `ruff check src/` clean, `mypy src/` clean
(strict), **122 tests pass**. Coverage: overall **57%** — `builder.py` 94%,
`system_utils.py` 73%, `tui.py` 51%, `main.py` 43%, `theme.py` 49%.

**Second audit pass (2026-06-22):** deeper edge-case fuzzing of the builder,
parsers, config loader, and batch path surfaced **P0-5**, **P1-11**, **P2-6**,
and **P3-11 – P3-13**; a focused look at colors/layout added a new
**Priority 5 — UI & visual design** section (**D1 – D8**).

> History: issues **#1–#29** and **U1–U5** were resolved in v0.2.0; the v0.2.1
> release re-synced the PyPI README. See [CHANGELOG.md](CHANGELOG.md) for the
> full record. Those entries are not repeated here.

---

## Priority 0 — Correctness / release blockers

### P0-1 · Version is hard-coded in two places and is already out of sync **(confirmed)**
- **Symptom:** `slurmate --version` prints `slurmate 0.2.0` even though the
  package on PyPI is `0.2.1`. The reported version is wrong *right now*.
- **Where:** `src/slurmate/__init__.py:3` (`__version__ = "0.2.0"`) vs
  `pyproject.toml:7` (`version = "0.2.1"`). `--version` reads `__version__`
  (`src/slurmate/main.py:427,456`).
- **Fix:** Single-source the version. Preferred:
  `__version__ = importlib.metadata.version("slurmate")` in `__init__.py` with a
  `PackageNotFoundError` fallback for editable/uninstalled runs — pyproject
  becomes canonical. (Alternative: make pyproject `dynamic = ["version"]` with
  `[tool.setuptools.dynamic] version = {attr = "slurmate.__version__"}` and treat
  `__init__` as the source.) Also bump to `0.3.0` in the one remaining place.
- **Guard:** add a test asserting `importlib.metadata.version("slurmate")`
  equals `slurmate.__version__` so this never drifts again (see P4-2).

### P0-2 · `SLURMATE_GPU_FORMAT` env var is dead code **(confirmed)**
- **Symptom:** README advertises `SLURMATE_GPU_FORMAT` as the default GPU syntax,
  but it has no effect. `SLURMATE_GPU_FORMAT=gpus slurmate --partition gpu-shared
  --gpus 2 --command ... --print` still emits `#SBATCH --gres=gpu:2`, not
  `--gpus=2`.
- **Where:** the only reader is `src/slurmate/builder.py:132`
  (`gpu_format or os.environ.get("SLURMATE_GPU_FORMAT", "gres_type")`), but both
  CLI paths set `gpu_format` before reaching it:
  - batch mode forces it: `src/slurmate/main.py:82-83`
    (`if gpus > 0 and not gpu_format: gpu_format = "gres_type"`);
  - the wizard always sets a value in `_setup_gpu_format`
    (`src/slurmate/tui.py:917-933`, default `"gres_type"`).
  So the env fallback is unreachable except via the raw library API.
- **Fix:** wire the env var in as the *default* in both places — in `run_batch`
  use `os.environ.get("SLURMATE_GPU_FORMAT", "gres_type")` instead of the literal,
  and seed the wizard's `_setup_gpu_format` default from it. (Or, if undesired,
  delete the feature from the README + builder so docs and behavior agree.)

### P0-3 · Wrong-typed config values crash batch mode **(confirmed)**
- **Symptom:** a `.slurmate.toml` containing `gpus = "2"` (a quoted/stringy
  number) makes `slurmate --partition … --print` crash with
  `TypeError: '>' not supported between instances of 'str' and 'int'`.
- **Where:** `run_batch` consumes config values without coercion —
  `gpus = … config.get("gpus", 0)` (`src/slurmate/main.py:64-65`) then
  `if gpus > 0` (`:82`). Same exposure for `cpus`, `nodes`, `ntasks_per_node`,
  and later `_show_script_and_summary` (`:299` `answers.get("gpus", 0) > 0`).
  The TUI path is safe because it stringifies then `_coerce`s; only batch mode
  trusts raw config types.
- **Fix:** coerce config-sourced numerics (`int(...)` with a `try/except`
  fallback to the default) in `run_batch`, or normalize types once in
  `load_config`. Add a malformed-config test.

### P0-4 · `validate_time` rejects many valid Slurm time formats **(confirmed)**
- **Symptom:** typing common valid times is rejected as "Invalid input":
  `30` (30 min), `2:30:00` (single-digit hour), `5:00` (mm:ss), `1-12`
  (days-hours), `1-0:00` — all accepted by Slurm, all rejected here. Only
  `dd-hh:mm:ss` and `hh:mm:ss` with **two-digit** fields pass.
- **Where:** `src/slurmate/system_utils.py:104-112` — regexes
  `^\d+-\d{2}:\d{2}:\d{2}$` and `^\d{2}:\d{2}:\d{2}$` only.
- **Fix:** broaden to Slurm's accepted set — `minutes`, `minutes:seconds`,
  `hours:minutes:seconds`, `days-hours`, `days-hours:minutes`,
  `days-hours:minutes:seconds` — and allow 1–2 digit fields
  (e.g. `^\d+$`, `^\d{1,2}:\d{2}$`, `^\d{1,2}:\d{2}:\d{2}$`,
  `^\d+-\d{1,2}(:\d{2}(:\d{2})?)?$`). Update the step subtitle
  (`src/slurmate/tui.py:260-263`) to reflect the wider grammar.

### P0-5 · Error-log path silently drops `%j` when the output name's last segment holds it **(confirmed)**
- **Symptom:** an output file like `run.%j` (or `%x.%j`, `log.%j`) yields
  `#SBATCH --output=run.%j` but `#SBATCH --error=run.err` — the **error path
  loses the `%j` placeholder**. Every job (and every array task) then writes
  stderr to the *same* fixed `run.err`, overwriting/interleaving. Confirmed:
  `run.%j → err=run.err`, `log.%j → err=log.err` (while `out_%j.log` is fine,
  because `%j` is in the base, not the trailing segment).
- **Root cause:** `os.path.splitext("run.%j")` returns `("run", ".%j")`, so the
  code treats `.%j` as a real extension and derives the error name from the bare
  base — `build_from_answers` at `src/slurmate/builder.py:36-43`
  (`base, ext = os.path.splitext(of); … error_path = _in_dir(base + ".err")`).
- **Fix:** only treat the suffix as an extension when it doesn't contain a Slurm
  pattern character (`%`). E.g. if `"%" in ext`, fall back to appending/replacing
  on the whole name, or swap only a trailing `.out`/`.log`/`.txt`-style literal
  extension. Add a regression test for `run.%j` and `%x.%j`.

---

## Priority 1 — UX & behavior

### P1-1 · Batch mode only triggers on `--partition`; other flags silently ignored
- **Symptom:** `slurmate --job-name foo --cpus 8 --command "x" --yes` (no
  `--partition`) launches the **interactive TUI** and drops every CLI value.
  A `partition` set only in the config file also won't trigger batch mode.
- **Where:** `src/slurmate/main.py:471` — `if args.partition is not None:` is the
  sole batch/interactive switch.
- **Fix:** enter non-interactive mode when *any* job-defining flag is supplied
  (or add an explicit `--batch`/`--no-tui` flag, and let a config-supplied
  partition satisfy the requirement). At minimum, warn when batch-only flags are
  passed without a partition so they aren't silently discarded.

### P1-2 · In-TUI Review step hides Modules, Custom flags, GPU format, Tasks-per-node **(confirmed)**
- **Symptom:** the "Review & Submit" config column never shows the user's
  modules, custom `#SBATCH` flags, GPU format, or `--ntasks-per-node`, so they
  can't be verified before submit (they *do* appear in the script column, but
  not the labeled summary).
- **Where:** `_review_summary_items` (`src/slurmate/tui.py:1178-1195`) lists only
  a subset of keys.
- **Fix:** add rows for `modules` (joined), `custom_sbatch` (joined),
  `gpu_format` (when GPUs > 0), and `ntasks_per_node`. While there, reconcile
  with the CLI summary (see P3-9) so both surfaces show the same fields.

### P1-3 · `tomli` is not a dependency on Python 3.10 → lossy config parsing
- **Symptom:** the project supports 3.10, where `tomllib` doesn't exist and
  `tomli` isn't installed, so `load_config` silently falls back to the naive
  flat `key=value` reader. That reader mishandles multi-line arrays, inline
  comments, and nested tables — config can be parsed wrong without warning.
- **Where:** `pyproject.toml:30-34` (no `tomli`); fallback at
  `src/slurmate/system_utils.py:667-676` and `_parse_config_naive:627-637`.
- **Fix:** add `tomli>=2.0; python_version < "3.11"` to `dependencies`. The naive
  parser can stay as a last resort but real TOML becomes guaranteed on all
  supported Pythons.

### P1-4 · `--print` and `--dry-run` are identical, redundant flags **(confirmed)**
- **Symptom:** both produce byte-for-byte identical output (just the raw script).
  `--dry-run` conventionally means "do everything except the side effect."
- **Where:** `src/slurmate/main.py:454-455` (defs) and `:485-488`
  (`if args.print or args.dry_run: print(script); return`).
- **Fix:** differentiate — keep `--print` as "emit only the raw script" (script
  to stdout, nothing else) and make `--dry-run` show the full summary panel,
  partition-limit warnings, SU/ETA, and missing-field reminders *without*
  submitting. Or collapse to one flag and document the other as a deprecated
  alias.

### P1-5 · Synchronous cluster queries freeze the wizard (no spinner/async)
- **Symptom:** entering the partition, modules, env, or GPU-type step runs
  `sinfo`/`scontrol`/`sacctmgr`/`module avail`/`conda env list` on the UI thread.
  On a busy cluster (or with thousands of modules) the full-screen TUI hangs for
  seconds with no feedback.
- **Where:** e.g. `_setup_partition` (`src/slurmate/tui.py:811-826`),
  `fetch_available_modules` (`src/slurmate/system_utils.py:419-437`),
  `fetch_conda_envs` (`:376-416`).
- **Fix:** run fetches in a background thread with a "Loading…" indicator and
  `app.invalidate()` on completion, or prefetch at startup. At minimum show a
  transient status line while a step's data loads.

### P1-6 · Auto-saved `<job>-<id>.sh` in CWD has no opt-out, and double-saves with `SLURMATE_LOG_DIR`
- **Symptom:** every successful submit writes a script copy into the current
  directory with no way to disable it (clutters project dirs). If
  `SLURMATE_LOG_DIR` is also set, the script is saved twice.
- **Where:** `_save_submitted_script` (`src/slurmate/main.py:376-386`, called at
  `:404-408`) plus the independent `SLURMATE_LOG_DIR` save in
  `submit_sbatch` (`src/slurmate/system_utils.py:599-607`).
- **Fix:** add an opt-out (`--no-save-script` / `SLURMATE_NO_SAVE=1`) and/or route
  the auto-save through `SLURMATE_LOG_DIR` when set so there's a single,
  configurable location instead of two saves.

### P1-7 · Mock-mode submit prints an empty Job ID and broken hints **(confirmed)**
- **Symptom:** `--yes` in mock mode prints `✓ Submitted! Job ID:` (blank),
  `Log path: logs/j-.out`, and hints `squeue -j` / `scancel` with no ID.
- **Where:** `submit_sbatch` returns `(0, "", "…mock…")`
  (`src/slurmate/system_utils.py:577-578`); `_submit_and_report`
  (`src/slurmate/main.py:400-421`) treats rc 0 as success regardless of an empty
  job ID.
- **Fix:** in `_submit_and_report`, detect the empty/mock job ID and print a
  clear "(mock mode — not actually submitted)" message instead of broken hints.

### P1-8 · Job names aren't sanitized — spaces produce a broken directive **(confirmed)**
- **Symptom:** a job name like `my training job` emits
  `#SBATCH --job-name=my training job`; `sbatch` splits on whitespace, so the
  name becomes `my` (and the extra tokens are mis-parsed). The auto-saved
  filename also keeps the spaces.
- **Where:** name written verbatim in `build_sbatch_script`
  (`src/slurmate/builder.py:111`); only `/` is stripped for the filename
  (`src/slurmate/main.py:378`).
- **Fix:** validate the job-name step (reject/normalize whitespace and shell-
  unsafe characters), or quote/slugify before emitting the directive.

### P1-9 · Submission failures print to stdout, not stderr
- **Symptom:** error text on submit failure goes to stdout, polluting pipelines.
- **Where:** `_submit_and_report` failure branch uses bare `print(...)`
  (`src/slurmate/main.py:392-397`) before `sys.exit(1)`. (Batch validation in
  `run_batch` correctly uses `Console(stderr=True)`.)
- **Fix:** route error output to `Console(stderr=True)` / `sys.stderr` for
  consistency and clean scripting.

### P1-10 · Array jobs use `%j`, not the idiomatic `%A_%a` log pattern **(confirmed)**
- **Symptom:** with `--array 1-10` and no explicit output file, logs default to
  `…/<job>-%j.out`. Array tasks conventionally use `%A_%a` (array id + task id).
- **Where:** default pattern in `build_from_answers`/`build_sbatch_script`
  (`src/slurmate/builder.py:46-47,153-154`).
- **Fix:** when `array_spec` is set and no explicit output file is given, default
  the pattern to `<job>-%A_%a.out` (and `.err`). Mention `%A`/`%a` in the
  output-file step subtitle.

### P1-11 · Batch mode skips the numeric validation the wizard enforces **(confirmed)**
- **Symptom:** `slurmate --partition gpu-shared --command x --cpus 0 --nodes -2
  --gpus -1 --print` emits Slurm-invalid `#SBATCH --cpus-per-task=0` and
  `#SBATCH --nodes=-2` with no error. The wizard rejects these (its `cpus`/`nodes`
  steps require `isdigit() and int(v) > 0`), but batch mode only hard-validates
  memory and time.
- **Where:** `run_batch` validates just `validate_memory`/`validate_time`
  (`src/slurmate/main.py:85-93`); `cpus`, `nodes`, `gpus`, `ntasks_per_node` flow
  straight through.
- **Fix:** validate the numeric flags in `run_batch` (positive int for
  cpus/nodes, non-negative for gpus/ntasks) and exit with a clear message,
  matching the wizard's rules so both entry points reject the same bad input.

---

## Priority 2 — Packaging & infrastructure

### P2-1 · Python 3.13 not tested or advertised
- **Where:** CI matrix `["3.10","3.11","3.12"]` (`.github/workflows/ci.yml:14`)
  and classifiers (`pyproject.toml:23-28`) stop at 3.12. 3.13 is released.
- **Fix:** add `3.13` to the CI matrix and a
  `Programming Language :: Python :: 3.13` classifier once green.

### P2-2 · No guard against the version drift in P0-1
- **Fix:** add a test (P4-2) and consider a release-workflow check that the tag
  (`v[0-9]+.[0-9]+.[0-9]+`, `.github/workflows/release.yml:6-7`) matches the
  package version before publishing.

### P2-3 · Classifier / metadata inaccuracies
- **Where:** `Development Status :: 4 - Beta` (`pyproject.toml:17`) vs the README's
  "experimental"; `Environment :: Console :: Curses` (`:18`) — the app uses
  `prompt_toolkit`, not curses.
- **Fix:** pick one maturity story (Beta vs experimental) and use it in both
  places; change the environment classifier to `Environment :: Console`.

### P2-4 · Deprecated `license` table form
- **Where:** `pyproject.toml:11` `license = {text = "MIT"}`.
- **Fix:** migrate to the PEP 639 SPDX string `license = "MIT"` (+ rely on
  `license-files` auto-glob), which newer setuptools prefers; drop the redundant
  `License :: OSI Approved` classifier when you do.

### P2-5 · No `MANIFEST.in` — verify sdist contents
- **Symptom:** the source distribution may omit `CHANGELOG.md`, `issues.md`, and
  the test suite (tests live outside the package).
- **Fix:** build `python -m build` and inspect `dist/*.tar.gz`; add a
  `MANIFEST.in` if `CHANGELOG`, `LICENSE`, or `tests/` should ship.

### P2-6 · Undocumented config-section precedence **(confirmed)**
- **Symptom:** when the same key appears at the top level, under `[defaults]`, and
  under `[slurmate]`, the effective order is **`[slurmate]` > `[defaults]` >
  top-level** (`_flatten_config` updates in that sequence). The README only says
  "keys may be top-level or under a `[defaults]` table" — the `[slurmate]` table
  and the override order aren't documented.
- **Where:** `_flatten_config` (`src/slurmate/system_utils.py:640-647`); README
  "Configuration file" section.
- **Fix:** document both the `[slurmate]` table and the precedence order (or drop
  one of the two table names to remove the ambiguity).

---

## Priority 3 — Code quality & robustness

### P3-1 · Wizard depends on prompt_toolkit private APIs; deps are unpinned
- **Symptom:** the TUI reads private/internal attributes that can break across
  `prompt_toolkit` releases: `RadioList._selected_index`
  (`src/slurmate/tui.py:530-533,538-541`), `buffer.complete_state`
  (`:442-447`), `window.render_info` (`:1006`). Dependencies are floor-only
  (`prompt_toolkit>=3.0`, `pyproject.toml:31-33`).
- **Fix:** pin a tested upper bound (e.g. `prompt_toolkit>=3.0,<4`) and add a
  smoke test that constructs the wizard and exercises selection, so a pt upgrade
  that moves these internals fails CI rather than users' terminals.

### P3-2 · Unquoted interpolation into `bash -lc`
- **Symptom:** module names are spliced into a shell string
  (`f"module load {names} …"`); a name with shell metacharacters breaks (or, in
  principle, injects into) the command. Self-inflicted only, but fragile.
- **Where:** `fetch_conda_envs` (`src/slurmate/system_utils.py:388-399`) and
  `fetch_available_modules` (`:421`).
- **Fix:** validate module tokens against a safe charset before interpolation, or
  pass them via argv/`shlex.quote` rather than string formatting.

### P3-3 · `estimate_su` ignores tasks-per-node **(confirmed)**
- **Symptom:** SU estimate uses `cpus_per_task × hours × nodes` only;
  `--ntasks-per-node` isn't factored, so multi-task jobs are undercounted
  (`8 cpus × 2h × 2 nodes → 32 SU`, regardless of ntasks).
- **Where:** `src/slurmate/builder.py:211-233`.
- **Fix:** multiply by `ntasks_per_node` when present. It's an estimate, so
  low-severity, but the label invites trust.

### P3-4 · CPU partition-limit check ignores ntasks-per-node
- **Symptom:** the "CPUs exceed partition limit" warning compares
  `cpus-per-task` to `cpus_per_node`, missing over-allocation when
  `ntasks-per-node × cpus-per-task` exceeds the node.
- **Where:** `_validate_partition_limits` (`src/slurmate/main.py:166-175`) and
  the TUI `_get_warning` (`src/slurmate/tui.py:654-660`).
- **Fix:** compare `ntasks_per_node × cpus` against `cpus_per_node`.

### P3-5 · Redundant cluster calls on the partition step
- **Symptom:** `_setup_partition` calls `fetch_public_partitions()` *and*
  `fetch_partitions()`, and `fetch_public_partitions` itself calls
  `fetch_partitions()` — so `sinfo` runs ~2–3× plus `scontrol` every time the
  step is entered, with no caching.
- **Where:** `src/slurmate/tui.py:811-814`;
  `src/slurmate/system_utils.py:268-299`.
- **Fix:** memoize cluster queries for the session (cache by command), or have
  `_setup_partition` reuse a single `fetch_partitions()` result.

### P3-6 · `theme.C.__getattribute__` runs `isatty()` + env reads on every color access
- **Where:** `src/slurmate/theme.py:43-48` calls `_should_use_color()` per
  attribute; the banner animation hits it many times per frame.
- **Fix:** compute the color decision once (cache it) instead of per access.

### P3-7 · Mock queue ETA label doesn't match the real formatter
- **Symptom:** `MOCK_QUEUE_INFO["eta_label"] = "~1 hour"` while
  `_format_eta(3600)` produces `"~1h"`. Mock display is inconsistent with live.
- **Where:** `src/slurmate/system_utils.py:459-464` vs `_format_eta:533-542`.
- **Fix:** derive the mock label from `_format_eta(eta_seconds)`.

### P3-8 · Dead `fetch` lambda on the partition Step
- **Symptom:** `Step("partition", … fetch=lambda: (fetch_public_partitions(),
  fetch_partitions()))` is never invoked — the partition step is handled by
  `_setup_partition`, not `_resolve_choices`.
- **Where:** `src/slurmate/tui.py:245-246`.
- **Fix:** remove the unused `fetch` to avoid implying it runs.

### P3-9 · Two divergent summary renderers
- **Symptom:** the CLI summary (`_show_script_and_summary`) shows Modules and
  Custom flags but not Output dir/file or env type; the TUI Review shows
  Output/Env but not Modules/Custom flags (P1-2). The two "summaries" disagree
  on what matters.
- **Where:** `src/slurmate/main.py:287-321` vs `src/slurmate/tui.py:1178-1195`.
- **Fix:** factor a single ordered "summary fields" helper both surfaces consume.

### P3-10 · Cleared config-defaulted fields fall back to hard-coded defaults
- **Symptom:** with `cpus = 8` in config, clearing the CPU field in the wizard
  coerces to `4` (the literal), not the configured `8`.
- **Where:** `_coerce` literals (`src/slurmate/tui.py:625-634`:
  cpus→4, nodes→1, memory→16G).
- **Fix:** fall back to `self._config_defaults.get(key, <literal>)` instead of the
  bare literal.

### P3-11 · `validate_memory` accepts `0G`/`0M` but rejects bare `0` **(confirmed)**
- **Symptom:** `validate_memory("0")` is `False`, but `validate_memory("0G")` is
  `True` (and `_parse_mem_to_mb("0G") == 0`). The zero check only fires for the
  unit-less form, so `--mem=0G` slips through as valid.
- **Where:** `src/slurmate/system_utils.py:80-101` — the `v == "0"` guard runs
  before the unit regex, which then matches `0G`/`0M`/etc.
- **Fix:** reject a zero magnitude regardless of unit (parse the numeric part and
  require `> 0`), or, if `--mem=0` ("all memory") is intentionally allowed,
  accept both `0` and `0G` consistently and document it.

### P3-12 · `_parse_mem_to_mb` silently mis-parses unsupported forms **(confirmed)**
- **Symptom:** `_parse_mem_to_mb("16GB")` returns `16` (not `16384`) — the regex
  fails on the double unit, then the `_safe_int` fallback grabs the leading `16`
  and treats it as megabytes (a 1000× error). Same for `16 G`, `1.5.5G`, etc.
  Normally shielded by `validate_memory`, but `_parse_mem_to_mb` is also called
  directly by the partition-limit warnings.
- **Where:** `src/slurmate/system_utils.py:67-77`.
- **Fix:** on regex-miss, return `0` (unknown) rather than a misleading partial
  integer, so a malformed value can't masquerade as a tiny valid one.

### P3-13 · Naive TOML fallback corrupts common config **(confirmed)** — compounds P1-3
- **Symptom:** on the no-`tomllib`/no-`tomli` path the flat reader mishandles
  ordinary TOML: an inline comment leaks into the value
  (`partition = "gpu"  # fav` → `'"gpu"  # fav'`, quotes not even stripped), and an
  unquoted-number array (`[1, 2, 3]`) coerces to `[]`. Floats and negatives fall
  back to raw strings.
- **Where:** `_parse_config_naive` / `_coerce_config_value`
  (`src/slurmate/system_utils.py:612-637`).
- **Fix:** the clean resolution is P1-3 (depend on `tomli` for <3.11 so real TOML
  is always used). If the naive reader stays, at least strip inline `#` comments
  outside quotes and handle unquoted numeric array items.

---

## Priority 4 — Testing

### P4-1 · CLI and submit/menu paths are largely untested **(confirmed)**
- **Symptom:** `main.py` at 43% and `tui.py` at 51% coverage. Untested:
  `main()` end-to-end, the post-build action menu loop
  (`src/slurmate/main.py:512-546`), `_submit_and_report`, `build_and_show`,
  `_show_script_and_summary`, `_edit_script_in_editor`, and most wizard
  render/navigation handlers.
- **Fix:** add integration tests for `--print`, `--dry-run`, `--yes` (mock
  `submit_sbatch`), the missing-field reminders, and a wizard smoke test that
  walks several steps and builds a script. Targets each of P0/P1 above should
  ship with a regression test.

### P4-2 · Add a version-consistency test
- **Fix:** assert `slurmate.__version__ == importlib.metadata.version("slurmate")`
  (guards P0-1).

### P4-3 · Stale test count in prose **(confirmed)**
- **Symptom:** the previous `issues.md` cited "120 tests"; the suite is now
  **122**. The README's contributing section is fine but worth a glance.
- **Fix:** avoid hard-coding counts in prose, or update when releasing.

---

## Priority 5 — UI & visual design

### D1 · No focus affordance on text inputs **(confirmed)**
- **Symptom:** the `text-area` and `text-area focused` styles are identical
  (`fg:#ffffff bg:#333333`), and the `TextArea` widgets also hard-code the same
  inline style — so a focused input field looks exactly like an inactive one.
  Nothing draws the eye to where typing will land.
- **Where:** `src/slurmate/tui.py:67-68` and the `TextArea(style=…)` constructors
  (`:331-341`).
- **Fix:** give the focused state a distinct look — a brighter background, an
  accent left-border, or a colored cursor line — so the active field is obvious.

### D2 · Patchwork panel backgrounds **(confirmed)**
- **Symptom:** the chrome (sidebar, queue line, preview, footer) is painted
  `bg:#1a1a2e` (deep navy), but the central column isn't: the title/error/warning
  windows use `style=""` (terminal default), the input box is `#333333`, and the
  review windows set no background. The result is up to three different
  backgrounds stacked together, which looks broken on any terminal whose default
  isn't near-black.
- **Where:** navy chrome at `src/slurmate/tui.py:1068,1151,1174,1302`; bare
  content windows at `:1094,1100` (`style=""`) and the review windows
  (`:358-365`); input at `:331-341`.
- **Fix:** pick one content background (e.g. reuse `#1a1a2e` or a single content
  shade) and apply it consistently to the title/content/review area, or
  deliberately go fully background-transparent and theme only the foreground.

### D3 · Palette assumes a dark terminal; poor contrast on light themes
- **Symptom:** foregrounds are tuned for a black background — cyan titles
  `#00ffff`, pending steps `#555555`, subtitles `#888888`, info `#888888`. On a
  light/solarized-light terminal these wash out (cyan-on-white, gray-on-white).
  There's no light-mode path; `_should_use_color` only checks `NO_COLOR`/`dumb`/
  TTY, not background.
- **Where:** `_TUI_STYLE` (`src/slurmate/tui.py:60-82`) and `theme.C`
  (`src/slurmate/theme.py:19-41`).
- **Fix:** choose colors with adequate contrast on both light and dark (avoid
  pure `#00ffff`/`#ffff00`, lift the grays), and/or honor a
  `SLURMATE_THEME=light|dark` override.

### D4 · CLI warning color is pure yellow; TUI uses amber — unify **(confirmed)**
- **Symptom:** the CLI paints warnings, "Cancelled", and the SU/array labels with
  pure yellow `#ffff00` (`c.YELLOW`), which is nearly invisible on light
  backgrounds; the TUI uses a readable amber `#ffaa00` for the same role. The two
  surfaces disagree and the CLI choice is the weaker one.
- **Where:** `theme.C.YELLOW` (`src/slurmate/theme.py:23`) vs the TUI `warning`
  style (`src/slurmate/tui.py:77`).
- **Fix:** standardize on amber/orange (`#ffaa00`/`#ff8800`) for warnings across
  CLI and TUI.

### D5 · Header bar style duplicated instead of using the `status-bar` class **(confirmed)**
- **Symptom:** `_header` defines `bar_style = "bg:#0088ff fg:#ffffff bold"` inline,
  duplicating the `status-bar` class already in `_TUI_STYLE`. Two copies of the
  same color drift apart over time.
- **Where:** `src/slurmate/tui.py:1042` vs the class at `:61`.
- **Fix:** apply `style="class:status-bar"` to the header windows and delete the
  inline literal.

### D6 · Sidebar fixed width clips the longest step title **(confirmed)**
- **Symptom:** the sidebar is `width=24`, but "Environment name/path" renders as
  25 columns (4-char prefix + 21) and gets clipped. (Related to the carried-over
  "sidebar has no scroll" nit below.)
- **Where:** `_sidebar`/`_render_sidebar` (`src/slurmate/tui.py:1064-1082`).
- **Fix:** widen the sidebar a couple of columns, shorten that title (e.g.
  "Environment"), or truncate long titles with an ellipsis.

### D7 · Banner animation adds ~1.5 s to every interactive launch **(confirmed)**
- **Symptom:** `print_banner(animate=sys.stdout.isatty())` defaults the animation
  **on** for any TTY, and the two-wave loop takes a measured **1.52 s** before the
  wizard appears — a noticeable stall on every `slurmate` run.
- **Where:** `src/slurmate/main.py:465`; animation loop
  `src/slurmate/theme.py:117-135` (`time.sleep(0.04)` × ~38 frames).
- **Fix:** make the animation opt-in (default to the instant banner; keep
  `SLURMATE_BANNER_ANIMATE=1` to enable), or cut the frame count/sleep so it's
  well under ~0.3 s.

### D8 · Banner shows "ESC to go back" even in non-interactive batch mode **(confirmed)**
- **Symptom:** `slurmate --partition … --command … --yes` prints the banner with
  the subtitle "ESC to go back", a hint that's meaningless when there's no wizard.
- **Where:** banner gate only excludes `--print`/`--dry-run`
  (`src/slurmate/main.py:464`); the hint text is in `print_banner`
  (`src/slurmate/theme.py:113,138`).
- **Fix:** suppress the "ESC to go back" line in batch mode (or only print it from
  the wizard entry path).

---

## Roadmap — feature ideas (not defects)

- **More `#SBATCH` coverage:** `--dependency`, `--begin`, `--mail-user`/
  `--mail-type`, `--mem-per-cpu`, `--gpus-per-node` / `--gpus-per-task`,
  `--exclusive` as a first-class toggle (today only via custom flags).
- **`--config PATH`** to point at an explicit config file; **`--save PATH`** in
  batch mode (instead of `--print > file`).
- **`sbatch --test-only`** integration to show the scheduler's real start-time
  estimate and validate the script before a live submit.
- **Profiles / templates:** save a named set of answers and reuse them; or
  remember the last run's answers as defaults.
- **Shell completion** for the CLI (e.g. `argcomplete`).
- **Surface partition defaults** (default time/mem/account) in the wizard.
- **Docs:** clarify per-node vs total semantics for `--mem` (per node) and the
  `--gpus` format (total across nodes) so multi-node users aren't surprised.

---

## Carried-over minor polish (from the v0.2.0 round, still open)

- **Sidebar has no scroll.** On a very short terminal the step list can overflow
  the sidebar height (`_render_sidebar`, `src/slurmate/tui.py:1071-1082`).
- **Visible-step denominator shifts once.** `visible_total` can only mark a
  skippable step (GPU type/format, tasks-per-node, env name) as skipped *after*
  the user passes it, so the `n/total` counter may change once mid-flow
  (`_render_header_right`, `src/slurmate/tui.py:1057-1062`). Inherent to not
  knowing future answers; acceptable.
