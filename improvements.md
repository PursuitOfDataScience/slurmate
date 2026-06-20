# Slurmify — Improvements & Issues

A thorough analysis of the current codebase (`src/slurmify/`, `tests/`, packaging,
CI). Items are grouped by category and tagged with severity:

- 🔴 **Critical** — broken behavior, data-loss / wrong-job risk, or blocks PyPI release
- 🟠 **Major** — wrong on real clusters, significant UX problems, or maintainability traps
- 🟡 **Minor** — polish, cleanup, nice-to-have

A suggested ordering is in [Roadmap](#roadmap) at the end.

---

## 1. Correctness bugs (logic)

### 1.1 🔴 Double submission in the interactive path
`main()` runs the `Wizard`, but the wizard **already submits the job itself**
(`^S` → `_submit_and_exit` → `submit_sbatch`). After the wizard returns,
`main()` falls through to `build_and_show()` and a second
`questionary.confirm("Submit this job to Slurm?")` → `submit_sbatch()`.

Consequences:
- If the user submits inside the TUI (`^S`), then quits (`^C`), `Wizard.run()`
  returns `self.answers` (truthy) and `main()` **submits the same job a second time**.
- If the user *doesn't* submit in the TUI and quits, `run()` returns `None` and
  `main()` prints "Cancelled" — so there is effectively **no clean single-submit
  interactive path**. Every interactive submit is either doubled or cancelled.

There are two competing submit/review flows (one in `tui.py`, one in `main.py`).
**Pick one.** Recommended: the wizard returns `answers` only (no submit, no
internal review screen duplication), and `main()` owns review + submit + editor.
Or the inverse. They must not both submit.

### 1.2 🔴 Dead `if script is None` branch hides the design confusion
`build_and_show()` always returns a `str` (never `None`), but `main()` checks
`if script is None: return`. This is dead code and a symptom of 1.1 — the two
flows were stitched together without reconciling control flow.

### 1.3 🟠 `module avail` can never run on a real cluster
`fetch_available_modules()` does:
```python
is_tool_available("module")        # shutil.which("module")
_run_command(["module", "avail", "2>&1"])
```
- `module` is a **shell function** (Lmod/Environment Modules), not an executable
  on `PATH`, so `shutil.which("module")` is almost always `None` → **always falls
  back to mock modules** on real clusters.
- Even if it resolved, `"2>&1"` is passed as a **literal argv** (no shell), so the
  redirect does nothing; `module avail` writes to **stderr**, which is discarded
  here (only `stdout` is read).

Fix: run via a login shell, capturing stderr, e.g.
`bash -lc 'module -t avail'` (the `-t`/terse format is parseable), or read
`$MODULEPATH`. Merge stdout+stderr.

### 1.4 🟠 `fetch_gpu_types_for_partition` returns `[]` (not mock) on command failure
On `rc != 0` it returns `[]`, but on "tool not available" it returns
`MOCK_GPU_TYPES`. Inconsistent with every other fetcher (which fall back to mock
on failure). A transient `sinfo` error then makes the GPU-type step silently
offer nothing.

### 1.5 🟠 `_normalize_memory` doesn't actually normalize bare units / lowercase edge cases
`_validate_memory` accepts `"16g"` (it uppercases), but `_normalize_memory`'s
final `return v` passes through anything that isn't pure-digit or `\d+[KMGTP]`.
`_parse_mem_to_mb` also accepts a trailing `N`/`C` suffix (`16GN`) that the
validator rejects. The three memory functions (`_validate_memory`,
`_normalize_memory`, `_parse_mem_to_mb`) use **three slightly different grammars**.
Consolidate into one parse/validate/format helper.

### 1.6 🟠 Memory `0` is accepted as valid
`_validate_memory("0")` → `True` (matches `^\d+[KMGTP]?$`), producing
`#SBATCH --mem=0M`. Slurm treats `--mem=0` specially (all node memory) which is
almost certainly not the user's intent here. Reject `0`, or document it.

### 1.7 🟡 `estimate_su` and `fetch_queue_eta` are fabricated numbers
- **SU cost** = `cpus × hours × nodes`. Real SU/credit formulas are
  cluster-specific (GPU weighting, partition multipliers, memory charging, min-core
  billing). Presenting a confident `"SU cost: 32 SU"` is **misleading** and not
  cluster-agnostic.
- **Queue ETA** is a hand-rolled heuristic (`pending/running × 120s`, capped),
  not derived from Slurm. `squeue --start` gives Slurm's *actual* estimated start
  time for pending jobs; that's the only defensible source.

Recommendation: either (a) drop both as default and gate behind a clearly-labeled
"rough estimate" with an opt-in/config hook, or (b) compute ETA from
`squeue --start` and make SU a pluggable per-cluster formula (config). Do not show
invented numbers as if authoritative.

### 1.8 🟡 GPU request encoding is cluster-specific and inconsistent
Builder emits `--gres=gpu:N` **plus** `--constraint=<type>`. On many clusters the
GPU type belongs in the gres itself (`--gres=gpu:a100:N`) or via `--gpus`/
`--gpus-per-node`, and `--constraint` is for node features (not GPU model). Worse,
`SBATCH_FLAGS` autocomplete lets the user add another `--constraint=` /`--gres=`,
producing **duplicate directives**. Decide on one GPU model and make type encoding
configurable per cluster.

### 1.9 🟡 `--cpus-per-task` without `--ntasks`
Always emits `--cpus-per-task=N` and `--nodes=M` but never `--ntasks` /
`--ntasks-per-node`. For multi-node jobs this is usually wrong (1 task spanning
nodes). The tool collects `nodes` but the script doesn't meaningfully use it for
task layout.

---

## 2. Cluster-agnostic / Slurm correctness

### 2.1 🟠 No partition-aware limits or validation
The wizard knows each partition's `cpus_per_node`, `mem_per_node_mb`,
`timelimit`, and `gpu_types` (it fetches them!) but **never validates** the
user's CPU/mem/time/GPU requests against them. Users can build a script that
`sbatch` will instantly reject. This is the single biggest UX win available:
validate against the selected partition and warn inline.

### 2.2 🟠 GPU step is not partition-aware
The `gpus` step always offers `0/1/2/4/8` with **default `"1"`**, even on
CPU-only partitions. Default should be `0`, and the step should be skipped (or
restricted) when the selected partition exposes no GPU types.

### 2.3 🟠 `conda activate` hard-coded; no venv / module-only / `srun` options
The script always does `source $(conda info --base)/etc/profile.d/conda.sh`.
Many clusters use `venv`, `pyenv`, `mamba`, or modules only. For a
cluster-agnostic tool, the environment-activation strategy should be selectable
(conda / mamba / venv path / none) rather than conda-only.

### 2.4 🟡 Output/error paths not created
`--output=name-%j.out` is emitted but no log directory is created or chosen. A
common pattern is `logs/%x-%j.out`; offer a log dir and `mkdir -p` it (or warn).

### 2.5 🟡 `fetch_public_partitions` mutates the shared `fetch_partitions()` dicts
It sets `part["is_public"] = ...` on dicts returned by `fetch_partitions()`. Since
both are called separately elsewhere, this in-place mutation of fetched state is a
latent aliasing bug. Build fresh dicts / copy.

### 2.6 🟡 `submit_sbatch` writes no script to disk and can't capture the job ID structurally
Piping to `sbatch` via stdin is fine, but the returned job id is only surfaced as
raw stdout text. Consider `sbatch --parsable` for a clean job-id, and optionally
saving the generated script next to the logs for reproducibility.

---

## 3. UX / TUI

### 3.1 🟠 Two different UIs for one tool
There's a full-screen `prompt_toolkit` wizard (`tui.py`) **and** a `rich`-based
review/summary + `questionary` confirms (`main.py`), plus a third set of
`questionary`/`prompt_toolkit` helpers in `theme.py` (`select_input`,
`autocomplete`, `text_input`, `path_input`) that are **never used**. This is
confusing to maintain and produces the double-submit bug. Consolidate on one.

### 3.2 🟠 Banner animation is intrusive and fragile
`print_banner(animate=...)` runs a 2-pass sleep-driven ANSI cursor animation on
every launch when stdout is a TTY. It:
- adds ~1s of forced latency before the UI,
- relies on `\033[s`/`\033[u` save/restore cursor (not universally supported),
- has no `--no-banner` / `NO_COLOR` / `SLURMIFY_NO_BANNER` opt-out.

Make it instant by default (or a single static gradient), opt-in animation.

### 3.3 🟠 No color/terminal capability detection
`theme.py` emits 24-bit truecolor escapes unconditionally. No handling for
`NO_COLOR`, `TERM=dumb`, non-TTY (piped) output, or Windows legacy consoles.
Piping `slurmify` output will embed raw escape codes. Respect `NO_COLOR` and
`isatty()`, and prefer `rich`'s console (which already does detection) over
hand-rolled ANSI.

### 3.4 🟡 Required-field handling is ad hoc
"Required" is special-cased inside `_confirm_and_next` for `job_name`/`command`
only, rather than being a `Step` attribute. Other steps that should arguably be
required (partition) rely on sub-flow logic. Add `required: bool` to `Step`.

### 3.5 🟡 `command` is single-line only
`Command to run` is a one-line `TextArea`. Real jobs often need multi-line
scripts / `srun` lines. Consider a multi-line editor step or "edit full script"
as a first-class path.

### 3.6 🟡 Inconsistent / undiscoverable keybindings
Review screen uses `^S`/`e`/`Esc`; steps use `Tab`/`Enter`/`S-Tab`/`Esc`. `Enter`
both confirms text and selects in radio — fine — but there's no help overlay and
no mouse-scroll for the long review/script panes (mouse_support is on but the
preview window isn't scrollable). The footer is the only hint.

### 3.7 🟡 Preview re-renders the whole script as plain text
The live preview rebuilds and re-joins the entire script on each step
(`_render_preview_text`) and shows it untokenized (no syntax highlighting),
unlike the `rich` Syntax panel in `main.py`. Inconsistent presentation.

---

## 4. Packaging / PyPI readiness

### 4.1 🔴 Tracked build artifacts in the repo
`git ls-files` shows **committed**:
- `build/lib/slurmify/*.py` (a stale duplicate of the package — 263-line old
  `main.py` vs the current 239-line one)
- `src/slurmify.egg-info/*` (SOURCES, PKG-INFO, etc.)

`.gitignore` lists `build/`, `*.egg-info/`, `.pytest_cache/`, etc., **but these
were committed before `.gitignore` existed**, so they're still tracked. Run
`git rm -r --cached build src/slurmify.egg-info` and commit. (This also violates
the repo's "no artifacts" rule.) The stale `build/lib` copy is actively
misleading.

### 4.2 🔴 `pyproject.toml` missing PyPI-required/expected metadata
Needs before publishing:
- `authors` / `maintainers`
- `license` (+ a `LICENSE` file — currently **none**)
- `readme = "README.md"`
- `classifiers` (Python versions, OS, License, Topic, Development Status)
- `keywords`
- `[project.urls]` (Homepage, Repository, Issues)

Version is hard-coded `0.1.0` in `pyproject.toml`; `src/slurmify/__init__.py` is
**empty** (no `__version__`). Add `__version__` and consider single-sourcing it.

### 4.3 🟠 No `py.typed` marker despite shipping type hints
The package is annotated and ships `from __future__ import annotations`, but has
no `py.typed`, so downstream type-checkers won't use the hints. Add `py.typed`
and include it in package data.

### 4.4 🟡 `tests/` not included/excluded explicitly; no build isolation check
Confirm wheel contents (`python -m build` then inspect) exclude tests and
artifacts. Add `[tool.setuptools.package-data]` for `py.typed`.

---

## 5. Code quality / duplication / dead code

### 5.1 🟠 Duplicated helpers across modules
`_get_partition`, `_normalize_memory`, `_parse_custom_flags` exist in **both**
`tui.py` and `main.py` (main imports two of them from tui but redefines
`_get_partition`). `build_sbatch_script(...)` is called with the **same 14
kwargs** in three places (`main.build_and_show`, `tui._build_preview`,
`tui._render_preview_text`). Extract a single `answers → script` adapter
(e.g. `build_from_answers(answers)`).

### 5.2 🟠 Large blocks of unused code in `theme.py`
`select_input`, `_PathCompleter`, `_make_session`, `autocomplete`, `text_input`,
`path_input`, `Spinner`, `tool_status`, `ok`, `fail`, `info`, `header`,
`_SEL_STYLE`, `_KB` appear unused by the actual wizard. Either delete or wire in.
Dead code inflates the surface and confuses contributors.

### 5.3 🟡 `theme.py` mixes UI helpers, ANSI palette, banner, and a prompt_toolkit
prompt library in one 300-line file. Split: `colors.py` (palette + capability
detection), `banner.py`, and drop the unused prompt helpers.

### 5.4 🟡 `__import__("re")` inside `_to_rgb`
`theme.py` does `__import__("re").match(...)` instead of a top-level
`import re`. Just import it.

### 5.5 🟡 Broad `except Exception` swallowing
`_resolve_choices`, `_setup_partition`, `_setup_gpu_type`, `_build_preview` all
catch bare `Exception` and silently degrade. At minimum log/surface to a debug
channel; otherwise real bugs (and the `module`/`sinfo` issues above) stay
invisible.

### 5.6 🟡 `_run_command` has no timeout
Every Slurm/conda call can hang indefinitely (busy controller, NFS stall). Add a
`timeout=` and handle `subprocess.TimeoutExpired` → mock fallback. A hung
`sinfo` currently freezes the whole TUI launch.

### 5.7 🟡 Type-annotation/runtime mismatch on 3.9
`requires-python = ">=3.9"`, and `from __future__ import annotations` covers
*annotations*, but some `dict[str, Any]`/`list[dict]` appear in **runtime**
positions historically risky on 3.9. (Currently OK because they're annotations,
but worth a 3.9 CI smoke-import to be safe — see 7.2.)

---

## 6. Tests

### 6.1 🟠 Tests assert almost nothing meaningful
Many tests are `assert result is not None` / `isinstance(..., list)` (e.g. all of
`TestEstimateSu`, several fetchers). They'd pass even if the function returned
garbage. Add value assertions: `estimate_su(4, "01:00:00", 1) == "4.0"`,
exact ETA boundaries, exact script content for edge cases.

### 6.2 🟠 No tests for the actual bug surface
Nothing covers: the submit flow (1.1), `_build_preview`/review rendering, the
partition sub-flow selection→`_partition_obj` mapping, GPU step skipping, or
`fetch_*` parsing of **real** `sinfo`/`scontrol`/`squeue` output (only the mock
path is exercised). Add fixtures with captured Slurm command output and test the
parsers against them.

### 6.3 🟡 Tests manipulate `sys.path` and set global env at import time
Each test file does `sys.path.insert(...)` and `os.environ["SLURMIFY_MOCK"]="1"`
at module top. Use an editable install (already in CI) + a `conftest.py`
fixture/autouse for mock mode instead. `tests/__init__.py` is empty; rely on
package discovery.

### 6.4 🟡 No coverage measurement
Add `pytest-cov` and a coverage gate to catch the large untested TUI logic.

---

## 7. CI / tooling

### 7.1 🔴 CI never triggers — wrong branch
`.github/workflows/ci.yml` runs on `push`/`pull_request` to **`main`**, but the
repo's default branch is **`master`**. CI will not run. Fix the branch names (or
rename the branch) — otherwise lint/type/test gates are silently inert.

### 7.2 🟠 `ruff` and `mypy` configured but not installed locally; mypy strictness off
`pyproject` configures ruff + mypy, CI runs them, but they aren't in the dev env
here (couldn't run). Also `mypy` is `strict = false` / `disallow_untyped_defs =
false`, so the annotations aren't enforced. Tighten incrementally.

### 7.3 🟡 No build/publish workflow
For a tool "destined for PyPI", add a release workflow (`python -m build` +
`twine`/`pypa/gh-action-pypi-publish` on tag), and a `python -m build` artifact
check in CI.

### 7.4 🟡 No pre-commit / formatting gate
Add `pre-commit` with ruff-format + ruff-lint so style is enforced before CI.

---

## 8. Documentation

### 8.1 🟠 README understates and partly misdescribes the tool
- Doesn't mention **batch/non-interactive mode** (`--partition ...`,
  `--yes`) which is fully implemented in `main.py`.
- "Pipes the script directly to `sbatch` — no temporary files written" is only
  half true: the **editor** flow (`_edit_script`, `_edit_script_in_editor`)
  writes temp files.
- No documentation of QoS/account/module/GPU autodetection, `SLURMIFY_MOCK`
  caveats, or supported Slurm versions.
- No screenshots/asciinema, no contributing guide, no changelog.

### 8.2 🟡 No docstrings on public builder/util signatures describing units/format
`build_sbatch_script` and `estimate_su` lack docstrings specifying expected
formats (memory units, time format, what SU means). Important for a library.

---

## Roadmap

**Phase 1 — Stop the bleeding (correctness + release blockers)**
1. Fix double-submit / unify submit flow (1.1, 1.2, 3.1) 🔴
2. Untrack `build/` and `egg-info`; fix CI branch (4.1, 7.1) 🔴
3. Add `LICENSE`, PyPI metadata, `__version__`, `py.typed` (4.2, 4.3) 🔴
4. Fix `module avail` detection / fallback consistency (1.3, 1.4) 🟠

**Phase 2 — Make it correct on real clusters**
5. Partition-aware validation + GPU-step awareness (2.1, 2.2) 🟠
6. Rethink SU/ETA: drop or back with real `squeue --start`; make SU pluggable (1.7) 🟡/🟠
7. Selectable env-activation (conda/venv/none) + configurable GPU encoding (2.3, 1.8) 🟠
8. `_run_command` timeouts; stop swallowing exceptions (5.6, 5.5, 5.2-fallbacks) 🟠

**Phase 3 — Consolidate & polish**
9. Remove dead `theme.py` helpers; split modules; single `build_from_answers` (5.1–5.4) 🟠/🟡
10. Color capability detection + banner opt-out (3.2, 3.3) 🟠
11. Required-field model, multi-line command, help overlay (3.4–3.6) 🟡

**Phase 4 — Confidence**
12. Real parser tests against captured Slurm output; submit-flow tests; coverage (6.1–6.4) 🟠
13. Strengthen mypy/ruff; add release + pre-commit workflows (7.2–7.4) 🟡
14. README rewrite (batch mode, temp-file caveat, config) + docstrings (8.1, 8.2) 🟠/🟡

---

## Quick wins (low effort, high value)
- `git rm -r --cached build src/slurmify.egg-info` (4.1)
- Fix CI branch `main` → `master` (7.1)
- Add `LICENSE` + `readme`/`authors`/`classifiers`/`urls` to `pyproject` (4.2)
- Delete unused `theme.py` prompt helpers (5.2)
- Default GPUs to `0`, not `1` (2.2)
- `import re` at top of `theme.py` (5.4)
- Add `timeout=` to `_run_command` (5.6)
