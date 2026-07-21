# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org).

## [0.5.1] — 2026-07-21

A bug-fix release from an adversarial edge-case pass over script generation, the
validators, and batch mode. No CLI or config-key changes; the base case is
byte-for-byte unchanged.

### Fixed

- **Custom `#SBATCH` flags with a space in the value were mangled** — a flag like
  `--comment="my job"` was split on the inner space into two broken directives
  (`#SBATCH --comment="my` + `#SBATCH --job"`), a script Slurm rejects. The
  parser (`_parse_custom_flags`) is now quote-aware (`shlex`), so a quoted value
  stays a single flag, and the builder re-quotes any custom-flag value that
  still contains whitespace (mirroring the existing output-path quoting) — so
  even a config-list entry like `custom_sbatch = ["--comment=my job"]` emits one
  well-formed `#SBATCH --comment="my job"` directive. Space- and comma-separated
  flags, comma-bearing values (`--exclude=node1,node2`), and a pasted `#SBATCH`
  prefix all still work; an unbalanced quote falls back to a plain split.
- **`validate_time` falsely rejected unpadded fields** — Slurm accepts
  single-digit minute/second fields (`5:3`, `1:2:3`), and the parser already
  read them correctly, but the wizard/CLI validator required two digits and
  rejected them. Minute/second fields are now `[0-5]?\d`, so unpadded values are
  accepted while genuinely out-of-range ones (`1:60`, `1-99:99:99`) stay rejected.
- **`build_sbatch_script(modules=…)` iterated a stray string** — a bare string
  (from a direct API call) was emitted one `module load <char>` per character;
  it is now split on commas like `custom_sbatch`, matching that field's existing
  defensive coercion.
- **Leading-dash job names produced flag-like filenames** — a name like `-rf`
  yielded `--output=-rf-%j.out` and a saved `-rf-<id>.sh`, so a follow-up
  `tail -f -rf-….out` parsed `-rf` as options. `sanitize_job_name` now strips a
  leading `-`/`+`/`.` (a name made only of those falls back to `slurm`); interior
  dashes/dots are preserved.
- **venv path with a trailing slash** — `--env /venv/` emitted
  `source /venv//bin/activate`; the trailing slash is now trimmed.
- **Confusing batch error for a non-integer `ntasks_per_node`** — a config value
  like `ntasks_per_node = "x"` printed `⚠ … using 0` and then hard-errored
  `… (got 0)`; it now raises a single clean error that names the actual value.

## [0.5.0] — 2026-07-19

Cluster-agnostic hardening from a documentation audit of the major US SLURM centers
(TACC, NERSC, SDSC, OLCF, PSC, Purdue, Harvard, …). New options and safer generation let a
script work on exclusive-node and mandatory-account/constraint sites, while the shared-node
base case (e.g. UChicago Midway3) is byte-for-byte unchanged. See `problems.md` for the
full audit and the rationale behind each fix.

### Added

- **`--mem-per-cpu`** — request memory per CPU instead of per node; takes precedence over
  `--mem` (Slurm treats the two as mutually exclusive).
- **`--constraint` (Slurm `-C`)** — a first-class node-feature constraint, e.g. NERSC
  Perlmutter's mandatory `-C cpu` / `-C gpu`.
- **GPU formats `gpus_per_node` and `gpus_per_task`** for `--gpu-format` /
  `SLURMATE_GPU_FORMAT` (matching NERSC/Anvil conventions), alongside the existing
  `gres_type` (default), `gpus`, and `constraint`.
- **Omit `--mem` entirely** — pass `--memory none` (or empty) so no memory directive is
  emitted, as whole-node/exclusive sites (e.g. TACC, which rejects `--mem`) require.

### Changed

- **conda/mamba activation is now batch-shell-safe** — the generated script sources
  `"$(conda info --base)/etc/profile.d/conda.sh"` before `conda activate <env>`, replacing
  the legacy bare `source activate <env>` that silently no-ops on modern conda (4.4+) in a
  non-login `#!/bin/bash` job (the common batch case).
- **No demo data on real clusters** — mock accounts/partitions/modules/GPU-types/queue-ETA
  now appear ONLY under `SLURMATE_MOCK`. When a real SLURM query is unavailable or errors,
  the corresponding picker is empty (type your own) / the ETA reads "unknown", instead of
  showing fake values that can't be submitted under — most importantly, no fake `--account`.
- **Node and GPU constraints merge** — a node `--constraint` combined with a GPU-as-
  constraint (`--gpu-format constraint`) now emits a single `--constraint=a&b` directive
  instead of two conflicting lines (Slurm would otherwise keep only the last).
- **A user-supplied memory flag wins** — a `--mem`/`--mem-per-cpu` entry in the custom
  flags suppresses the auto memory directive, so a script never sets both at once.
- **`module avail` parsing** tolerates Lmod terse extras (trailing `/` family short names,
  `(D)`/`<F>` tag markers, `(@alias)` annotations).
- **Public-partition detection** also requires `State=UP`.
- **Clearer memory prompt + wrapped warnings** — the Memory step states the value is the
  total per-node request (Slurm `--mem`), and long validation warnings now wrap onto extra
  lines instead of truncating at the card's right edge.
- **Pre-submit error guard** — a job with a hard error (e.g. GPUs on a CPU-only partition)
  is no longer submitted: navigation stays free and the error shows on every step, but
  "Submit" / `--yes` now refuse and point back to the fix, instead of letting `sbatch`
  reject it after a wasted round-trip. Warnings remain advisory and never block.
- **Simpler header** — the top-right shows just the step counter ("Step 9 / 20"); the
  current step's name (already the card title and the highlighted sidebar row) was dropped.

## [0.4.1] — 2026-07-18

A visual-polish release for the wizard TUI. No behavioral or CLI changes — every
job it generates is byte-for-byte identical to 0.4.0; only the on-screen colors
and card layout change.

### Changed

- **Multi-hue wizard palette** — the interactive wizard no longer renders as one
  flat wall of blue. Each structural region now owns a distinct, harmonized hue:
  teal for the header/brand and status labels, violet for the Steps sidebar,
  pink for the progress counter, green for completed steps and the live script
  preview, amber for warnings and the review "Job Configuration" card, and red
  for errors. Blue is now reserved exclusively for the one element your keys
  actually drive — the focused input/selection — so focus is unambiguous.
- **Two-tone header** — the "Slurmate" brand sits in teal with a dimmed tagline,
  and the right-aligned progress counter is pink, echoing the startup banner's
  gradient instead of a single flat bar.
- **Snugger review step** — the "Job Configuration" and "Final Script" cards are
  sized to their content (config summary centered vertically) rather than sprawling
  as two mostly-empty boxes; config values are clipped horizontally instead of
  wrapping (the full, untruncated value is always visible in the Final Script
  card alongside). A top margin and inter-region spacing give the header room to
  breathe.
- **`_card()` internals** — regions now take an explicit accent `color` for their
  border and title (replacing the old `card-border`/`card-title` style classes);
  card interiors remain transparent so the terminal's own (possibly translucent)
  background shows through.

## [0.4.0] — 2026-07-18

Another correctness-focused pass — real-cluster account discovery, more robust
Slurm-output parsing, safer script generation, and clearer CLI behavior — plus a
second adversarial audit that hardened config-driven batch mode and interactive
navigation, a redesigned transparent "card" wizard, and cluster-agnostic wording
throughout.

### Fixed

- **Empty account list on real clusters** — `fetch_user_accounts()` now queries
  the current user's associations (`sacctmgr show assoc user=<you>`) instead of
  `show user`, which returns unscoped, account-less rows and made the picker
  silently fall back to mock accounts you can't submit under.
- **Memory-limit warning silently disabled on heterogeneous partitions** — a
  `sinfo %m` value like `515000+` now parses to the minimum value instead of `0`,
  so the "memory exceeds partition limit" warning fires again.
- **False "partition does not support GPUs" warning** — partitions advertising a
  count-only (`gpu:4`) or typed-without-count (`gpu:a100`) GRES are now detected
  as GPU partitions via a new `has_gpu` flag, so the warning no longer misfires.
- **Partition node counts undercounted** — node totals are summed across
  per-state `sinfo` rows instead of taking the max of a single state group.
- **Multiple GPU models per node dropped** — a node advertising
  `gpu:a100:2,gpu:v100:2` now surfaces both models.
- **`sinfo` node-state flags dropped nodes** — flag-suffixed states (`idle~`,
  `mix*`, …) are normalized, so queue-ETA node tallies aren't undercounted.
- **conda env names** — discovery uses `conda info --json`, so the base env is
  labelled `base` (not its install-dir name) and a `--prefix` env stays an
  activatable path; a login-shell banner containing braces no longer breaks JSON
  parsing.
- **`module avail` pollution** — the module list no longer includes the
  `command -v module` probe output or filesystem path headers, and it honours
  mock mode like every other fetcher.
- **Crash under a non-UTF-8 locale** — subprocess output is decoded as UTF-8 with
  a lossy fallback, and a present-but-unrunnable Slurm binary falls back to mock
  data instead of raising.
- **Malformed config silently dropped every default** — an unreadable/invalid
  `.slurmate.toml` now warns on stderr; the naive fallback reader is section- and
  multi-line-array-aware; a non-integer numeric config value (e.g.
  `cpus = "8cores"`) is reported instead of silently reverting to the default.
- **Script-generation edge cases** — an empty partition/job-name no longer emits
  a malformed `#SBATCH --partition=` / `--job-name=`; a name that sanitizes away
  (all-symbol or non-Latin) falls back to `slurm`; an explicit `output_file` on
  an array job gets a per-task `%A_%a` tag (no more clobbering); output/error
  paths with spaces are quoted; a leading `~` in a log path is expanded;
  `env_name` is shell-quoted; a newline in a `custom_sbatch` entry can no longer
  inject a script-body line; the GPU custom-flag de-dup is space-form- and
  format-aware; and an unrecognized `gpu_format` from config/env is clamped to
  `gres_type` with a warning.
- **`$EDITOR` with arguments/empty/missing crashed** — "Open script in editor"
  now splits `$EDITOR` into words (so `code --wait` works), treats an empty value
  as unset, and reports a failed launch instead of raising; editing answers after
  a manual edit confirms before discarding it.
- **"Script saved" reported even when the write failed** — the
  `SLURMATE_LOG_DIR` copy is written by the CLI and reported only on real success.
- **Federated job IDs** — a `jobid;cluster` from `sbatch --parsable` is split so
  the hints, log path, and saved filename use the numeric id.
- **TUI** — the live preview refreshes after backward navigation; a skipped
  `env_name` no longer captures another step's leftover text; QoS choices are
  re-fetched when the partition changes.
- **Batch mode crashed on a wrong-typed config value** — a TOML array (or wrong
  scalar) for a free-form field (`command`, `partition`, `account`, `qos`,
  `array_spec`, output paths, `env`/`env_type`) now produces a clean
  `✗ Error: <field> must be a string` and exit 1 instead of an uncaught
  `AttributeError`/`TypeError` traceback on `--print`/`--dry-run`/`--yes`.
- **Wizard crashed on "go back" from an invalid numeric field** — pressing Esc /
  Shift-Tab after typing a non-integer into CPU cores / Nodes / Tasks-per-node no
  longer raises `ValueError`; `_go_back` now mirrors the forward validator guard
  (an invalid value is simply not saved, so the prior answer stands).
- **Empty QoS picker on `AllowQos=ALL`** (Slurm's default for most partitions) —
  the wizard now offers the known QoS instead of only `Default (none)`, and when
  `sacctmgr` is unavailable it trusts `scontrol`'s list rather than filtering
  real, lab-specific QoS against the demo names.
- **Crash saving/editing the script under a non-UTF-8 locale** — the temp-file
  and saved-script I/O now force `encoding="utf-8"` (matching the already-hardened
  subprocess paths), so a non-ASCII byte no longer raises a `UnicodeError` — in
  the worst case *after* `sbatch` had already accepted the job.
- **`validate_time` accepted out-of-range fields** — `1:60:60` / `1-99:99:99`
  are now rejected client-side (minute/second fields are `[0-5]\d`); a bare `0`
  (Slurm's "no limit") is still accepted.
- **`_detect_gpu_type` false positives** — a spelled-out CPU codename (`power9`)
  and a pathologically long feature token are no longer surfaced as GPU models.
- **`--yes` submitted a no-op for a blank/comment-only command** — a whitespace-
  or `#comment`-only command is now the same hard error as an empty one.
- **Module names are shell-quoted** in `module load` (matching `env_name`), and
  the partition step restores your prior selection on "go back" instead of
  resetting the cursor to "Enter manually…". `build_sbatch_script` also coerces a
  non-string `gpu_type` and clamps a negative core/node count in the cost estimate.

### Changed

- **`--print` / `--dry-run` read your config** — with a `.slurmate.toml` present
  they render the script non-interactively from it instead of launching the
  wizard (a bare `slurmate --print` with no config still opens the wizard).
- **`--yes` requires a command** — an unattended submit with no command is now a
  hard error rather than silently submitting a no-op job.
- **`SLURMATE_NO_BANNER`** — honours affirmative values (`1`/`true`/`yes`/`on`)
  only, so `SLURMATE_NO_BANNER=0` no longer suppresses the banner.
- **Redesigned wizard UI** — each region (Steps, the current field, the live
  preview, and the Review columns) is now a rounded, fill-less "card", so the
  terminal's own background (including any translucency/blur) shows through
  instead of a flat navy fill. The palette is refined and desaturated — one blue
  accent carries focus/headers/the current step; green/amber/red are reserved for
  state — replacing the previous pure-neon look. The active input card carries an
  accent focus-ring border so it's always clear which field is live.
- **Cluster-agnostic wording** — dropped the misleading "(optional)" from the
  Account field (accounting-enforced clusters reject jobs without a valid
  account); the summary now shows **Estimated CPU-hours** instead of the
  site-specific "SU"; and abbreviated labels are spelled out in full ("Tasks per
  node", "Array specification", "Output directory", "Environment", and
  "N running / M pending").

## [0.3.0] — 2026-06-23

A correctness- and polish-focused release that works through the v0.3.0
planning backlog (see the prior `issues.md`). Highlights: the version is now
single-sourced, batch mode is far more robust, time/memory validation matches
Slurm, and the wizard's visuals are cleaner.

### Fixed

- **Day-hours time parsing** — `_parse_slurm_time_to_minutes()` now reads the
  `D-HH` and `D-HH:MM` Slurm formats correctly (the field after the dash is
  hours, not minutes), fixing SU estimates and partition time-limit warnings
  (e.g. `0-23` is now 1380 min, not 23). (#2)
- **Numeric config values crashed the CLI** — an integer `time_limit` or
  `gpu_type` in `.slurmate.toml` no longer raises `AttributeError`; both are
  coerced to strings in batch mode. (#3)
- **`gpu_format` case-sensitivity** — a non-lowercase `gpu_format` (from the
  `SLURMATE_GPU_FORMAT` env var, a config file, or a programmatic call) is now
  normalised, so it no longer silently emits the constraint directive instead
  of the requested format. (#4)
- **Comma-valued custom flags** — a bare-string `custom_sbatch` with a
  comma-bearing value (e.g. `--nodelist=node1,node2`) is parsed with the
  flag-aware splitter instead of being mangled into an invalid `#SBATCH`
  directive. (#5)
- **Version drift** — `slurmate --version` is now single-sourced from the
  installed package metadata (`importlib.metadata`), so it can never disagree
  with the published version again. (P0-1)
- **`SLURMATE_GPU_FORMAT` had no effect** — the env var is now the actual
  default GPU syntax in both batch mode and the wizard's GPU-format step, as the
  README always advertised. (P0-2)
- **Stringy config values crashed batch mode** — a `.slurmate.toml` with e.g.
  `gpus = "2"` no longer raises `TypeError`; numeric config values are coerced.
  (P0-3)
- **`--time` validation was too strict** — now accepts Slurm's full grammar
  (`minutes`, `mm:ss`, `hh:mm:ss`, `days-hours`, `days-hours:minutes`,
  `days-hours:minutes:seconds`) with 1–2 digit lead fields, so `30`, `5:00`,
  `2:30:00`, and `1-12` are accepted. (P0-4)
- **Error log dropped `%j`** — an output pattern like `run.%j` no longer derives
  a fixed `run.err` (which every task would overwrite); a `%`-bearing suffix is
  treated as part of the log pattern, not a file extension. (P0-5)
- **Batch mode only triggered on `--partition`** — any job-defining flag (or
  `--yes`) now enters non-interactive mode, so flags like `--cpus`/`--command`
  are no longer silently dropped into the TUI. (P1-1)
- **In-TUI Review hid fields** — the Review step now shows Modules, Custom
  `#SBATCH` flags, GPU format, and Tasks-per-node, sharing one ordered field
  list with the CLI summary so the two surfaces always agree. (P1-2, P3-9)
- **Lossy config on Python 3.10** — `tomli` is now a dependency on `<3.11`, so
  real TOML parsing is guaranteed on every supported Python; the naive flat
  reader is only a last resort and now strips inline comments and parses numeric
  arrays/floats/negatives correctly. (P1-3, P3-13)
- **Mock-mode submit printed a blank Job ID** and broken `squeue`/`scancel`
  hints — it now prints a clear "(mock mode — not actually submitted)". (P1-7)
- **Job names weren't sanitized** — whitespace and shell-unsafe characters are
  normalized (`my training job` → `my_training_job`) so the directive and the
  auto-saved filename are always well-formed. (P1-8)
- **Submission errors went to stdout** — failures now go to stderr for clean
  pipelines. (P1-9)
- **Batch mode skipped numeric validation** — `--cpus`/`--nodes` must be
  positive and `--gpus`/`--ntasks-per-node` non-negative, matching the wizard,
  instead of emitting invalid directives like `--cpus-per-task=0`. (P1-11)
- **`validate_memory` accepted `0G`/`0M`** — a zero magnitude is now rejected
  regardless of unit. (P3-11)
- **`_parse_mem_to_mb` mis-parsed bad input** — `16GB`/`16 G`/`1.5.5G` now
  return `0` (unknown) instead of a misleading partial that masqueraded as a
  tiny valid size in partition-limit checks. (P3-12)
- **Redundant cluster queries** — the partition step fetches once and caches for
  the session; re-entering or going back reuses the result instead of re-running
  `sinfo`/`scontrol`. (P1-5, P3-5)
- **Unquoted module names in `bash -lc`** — module tokens are now `shlex`-quoted
  before interpolation. (P3-2)
- **Cleared config-defaulted fields** fell back to hard-coded literals — they now
  fall back to the configured value (e.g. clearing a `cpus = 8` field returns
  `8`, not `4`). (P3-10)
- **Mock queue ETA label** is now derived from the real formatter (`~1h`), not a
  hand-written `~1 hour`. (P3-7)

### Added

- **`--no-save-script` / `SLURMATE_NO_SAVE=1`** to opt out of the auto-saved
  `<job>-<id>.sh` copy; when `SLURMATE_LOG_DIR` is set the script is saved there
  once (no more double-save into the working directory). (P1-6)
- **Array-aware log defaults** — array jobs (`--array`) now default to the
  idiomatic `%A_%a` (array id + task id) pattern instead of `%j`. (P1-10)
- **Python 3.13** added to the CI matrix and the classifier list. (P2-1)
- A release-workflow guard that fails if the pushed tag doesn't match the
  `pyproject` version, and a test asserting `__version__` equals the installed
  metadata. (P2-2, P4-2)
- A `MANIFEST.in` so the sdist ships `CHANGELOG.md` and the full (runnable) test
  suite, including `conftest.py` and the parser fixtures. (P2-5)
- Many regression and integration tests covering each fix above.

### Changed

- **`--print` and `--dry-run` are now distinct** — `--print` emits only the raw
  script (clean for pipes/CI); `--dry-run` shows the full summary panel,
  partition-limit warnings, SU/ETA, and missing-field reminders without
  submitting. (P1-4)
- **SU estimate** now factors in `--ntasks-per-node`, and the CPU
  partition-limit warning compares `ntasks-per-node × cpus-per-task` against the
  node core count. (P3-3, P3-4)
- **UI polish:** focused text inputs now stand out (distinct background); the
  central column uses one consistent background instead of a patchwork; warnings
  are amber across both CLI and TUI (was pure yellow on the CLI); the header
  reuses the `status-bar` style; the sidebar is wider (and ellipsizes long step
  titles) so "Environment name/path" no longer clips; the startup banner is
  instant by default (animate via `SLURMATE_BANNER_ANIMATE=1`); and the "ESC to
  go back" hint is suppressed in batch mode. (D1, D2, D4, D5, D6, D7, D8)
- The color decision is computed once per process instead of on every color
  access (the banner hit it hundreds of times). (P3-6)
- Migrated `pyproject` to the PEP 639 SPDX `license = "MIT"` form, dropped the
  redundant license classifier, fixed the environment classifier
  (`Console`, not `Console :: Curses`), and pinned `prompt_toolkit>=3.0,<4`.
  (P2-3, P2-4, P3-1)

## [0.2.1] — 2026-06-22

### Fixed

- PyPI `README` was out of sync with the GitHub `README` — the `v0.2.0`
  release was cut before a documentation polish commit landed, so PyPI was
  missing the `[PyPI]` badge, had an older "Interactive mode" description
  (lacked the **Review & Submit** walkthrough), and used shorter feature-table
  text. Now resolved for the `v0.2.1` release.

## [0.2.0] — 2026-06-21

### Added

- The exact submitted script is now saved locally by default — on submit it's
  written to `<job-name>-<job-id>.sh` in the working directory, leaving a
  reproducible record next to where the job was launched.
- Post-wizard script + summary panels render **side by side** when the terminal
  is wide enough (stacked otherwise), using a `Table.grid` layout.
- In-TUI "Review & Submit" final step — shows the job configuration and the
  generated script **side by side** for a last look before submitting, without
  leaving the full-screen wizard. The script column scrolls with ↑/↓ and
  PgUp/PgDn (via manual line-slicing, with a pinned "── Final Script ──"
  header) so long scripts aren't cut off, and multi-line commands line up under
  the value column in the config. (U4)
- Conda environment autocomplete — `_setup_env_name` fetches conda envs via
  `fetch_conda_envs()` and sets `FuzzyWordCompleter` with the results. (#14)
- Conda env list now reflects the chosen module stack — `fetch_conda_envs()`
  loads the user's selected modules (in a login shell where `module` is defined)
  before running `conda env list`, so envs from a module-provided conda (e.g.
  `module load anaconda`) are discovered. Login-shell banner text before the
  JSON is sliced out.

### Fixed

- Custom `#SBATCH` flags now split on spaces as well as commas —
  `_parse_custom_flags` treats each whitespace/comma-separated token as its own
  option (`--exclusive --reservation=abc` and `--exclusive,--reservation=abc`
  both → two directives). Only a comma that introduces another flag separates
  options, so a comma *inside* a value survives (`--exclude=node1,node2` stays
  one directive). Values are written with `=` (`--reservation=abc`); a bare word
  is its own option (`exclusive` → `--exclusive`) and is never glued onto the
  previous flag, so the wizard never invents an invalid combination like
  `--exclusive=<node>` from `--exclusive <node>`.
- Custom-flag autocomplete suggestions now include `--exclude=` and
  `--nodelist=` (alongside the existing `--exclusive`).
- Conda env discovery — `fetch_conda_envs` returns `[]` (not misleading mock
  names) when conda/module lookup fails in real mode, de-dups results, and the
  wizard now opens the env dropdown on entry so the discovered envs are visible
  without typing.
- Custom `#SBATCH` flags entered in the wizard were emitted one character per
  line (`#SBATCH m`, `#SBATCH i`, …) — `_coerce` stored the raw string and the
  builder iterated it character-by-character. The wizard now parses the field
  into a flag list via `_parse_custom_flags`, and the builder defensively splits
  a stray string instead of iterating its characters.
- GPU type detection false positives on count-only GRES nodes — when a node
  exposes `gpu:N` (no model), `_detect_gpu_type` now *prefers* a feature token
  that matches a typed GPU model (`gpu:MODEL:N`) seen elsewhere in the
  partition, so nodes that list rack/filesystem labels first (e.g.
  `rack5,gpfs,a40`) resolve to the real GPU (`a40`). When no token corroborates,
  it falls back to negative filtering so GPU types that only ever appear in
  features (and never in a typed GRES) are still detected — every type a
  partition exposes shows up in the picker. (#22)
- `output_file` with a non-`.out` extension no longer gets `.out` appended (the
  old `run.log` → `run.log.out` double extension); uses `os.path.splitext` and
  derives `.err` from the real base. (#23)
- `_coerce` defaulted an empty `gpus` value to 4 — now defaults to 0. (#24)
- Partition memory-limit warning ignored decimal and `K`/`P` values — both
  `_validate_partition_limits` and the TUI's `_get_warning` now use
  `_parse_mem_to_mb` instead of an ad-hoc `[MGT]?` regex. (#25)
- `--env-type none` with an `--env` name silently dropped activation — the
  builder now logs a warning when an env name is set but no activation line is
  emitted. (#26)
- Wizard crashed on reaching the Review step — the review step's focused window
  is now part of the layout, fixing a `Window does not appear in the layout`
  `ValueError`. (#28)
- Footer dropped `Esc:Back` / `^C:Quit` on non-review steps after `F2:Mouse` was
  added — both are restored on every step. (#29)
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
  `"(Enter=next, Ctrl+J=newline, Tab=complete)"`. (#12)
- GPU type detection only from GRES (missed count-only nodes) — added features
  scanning fallback. (#13)
- Multiline command Enter handling — `eager=True` intercepted Enter before the
  TextArea could act on it; the handler now routes Enter explicitly. Final
  behavior: Enter advances on every step (see Changed), Ctrl+J inserts a
  newline. (#15)
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
  Enter advances on multiline steps (see Changed). (#19)
- CI failing — removed unused `Frame` import; fixed generator return type
  annotations on `LastTokenPathCompleter` and `LastTokenCommaCompleter`. (#20)
- Tab still advancing from multiline command step — async `PathCompleter`
  hadn't populated `complete_state` by the time the eager Tab handler checked
  it. Tab now only completes on multiline steps and never navigates away;
  Enter advances. (#21)
- TUI crash when `gpu_type`/`env_name`/`partition` were `None` (TextArea
  rejected `None`).
- `#SBATCH` directives emitted in wrong order — now matches wizard step order.
- Auto-derived `--output`/`--error` shown in preview before output configured
  — now hidden until output dir/file is set.
- Live preview height — now fills available space.
- Mouse capture ON by default (prevented text selection) — now permanently OFF
  so the terminal can natively select/copy.
- One-way edit/submit/save confirm chain — replaced with a navigable action menu
  (Submit / Go back to edit answers / Open script in editor / Save / Quit).
  Pressing **Esc** (or choosing "Go back to edit answers") re-opens the wizard at
  the review step with all answers preserved, so a field can be fixed after
  seeing the generated script. The redundant "Show script again" option was
  removed (the script is already on screen).
- `qos "Default (none)"` leaking into script and summary.

### Changed

- GPUs step accepts any count — it was a fixed radio list (0/1/2/4/8) with no way
  to request e.g. 3 or 16. It's now a free-text field that still suggests the
  common values but validates and accepts any non-negative integer.
- Step counter and sidebar now hide auto-skipped steps (GPU type/format,
  tasks-per-node, env name): the header shows a compact visible `n/total`
  counter instead of a per-step dot row, and skipped steps no longer appear in
  the sidebar or shift the progress count. (U1, U3)
- `output_file` step subtitle clarifies that a bare name gets `.out` and `.err`
  is derived. (U5)
- Command-step keys are now consistent with the rest of the wizard: **Enter
  advances** (instead of inserting a newline), **Ctrl+J** inserts a literal
  newline, and **Tab** completes paths. (Shift+Enter is indistinguishable from
  Enter at the terminal level, so it can't be bound; if you advance by mistake,
  Esc goes back with your input preserved.) The old `Ctrl+G` "next" key was
  removed as redundant.
- Queue ETA in the wizard is now shown only after all hardware/resource steps
  are chosen (from the modules step onward), as a heads-up on the wait before
  modules load and the script runs, instead of flickering during the hardware
  steps.
- Dropped the `(rough)` qualifier from the SU / ETA labels, and the redundant
  `Est.` prefix from `ETA` (the "E" already stands for "Estimated"); the SU
  label stays `Est. SU`.
- Removed the `F2` mouse-capture toggle entirely (no function keys — Mac
  keyboards lack them); mouse capture stays off so the terminal can natively
  select/copy the preview, and navigation is fully keyboard-driven.
- Consolidated three different memory parsing grammars into unified
  `validate_memory` / `normalize_memory`.
- `build_from_answers()` helper created to eliminate 14-argument duplication
  across `tui.py` and `main.py`.

### Docs

- Corrected the v0.1.0 config-path note from the never-shipped
  `~/.config/slurmate/slurmate.json` to the actual TOML paths. (#27)

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
- Config file support (`.slurmate.toml` / `~/.config/slurmate/config.toml`).
- Queue status panel in TUI showing running/waiting jobs and ETA.
- Post-submission hints (job ID, log path, `squeue`/`scancel` commands).
- `--version` flag.
- Tasks-per-node support.
- `ruff` and `mypy` CI checks.
- Test suite with fixtures for partition, queue, and GPU type parsing.

[0.5.1]: https://github.com/PursuitOfDataScience/slurmate/compare/v0.5.0...v0.5.1
[0.4.0]: https://github.com/PursuitOfDataScience/slurmate/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/PursuitOfDataScience/slurmate/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/PursuitOfDataScience/slurmate/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/PursuitOfDataScience/slurmate/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/PursuitOfDataScience/slurmate/releases/tag/v0.1.0
