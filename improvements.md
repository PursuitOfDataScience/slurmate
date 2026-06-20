# Slurmify — Improvements & Issues (Third pass)

Re-evaluation after the second fix round. **Verdict: the codebase is in good
shape.** Every release-blocker and almost every item from the previous two
reviews is now genuinely fixed and *verified by running the toolchain* (not by
inspection this time):

```
ruff check src/   → All checks passed!   (1 config-deprecation warning, see R4)
mypy src/          → Success, no issues   (strict = true)
pytest             → 76 passed
```

What remains is a short tail of correctness edge-cases, a real gap against the
**“ship to PyPI”** goal (the release workflow never actually publishes), and a
set of UX upgrades to make the tool effortless on a single cluster.

Severity: 🔴 Critical · 🟠 Major · 🟡 Minor

---

## CI/CD status — answered directly

**Yes, CI is enabled and correct.** `.github/workflows/ci.yml` runs on every
push and PR to `master`, across a Python 3.9–3.12 matrix, and executes:
`ruff check src/` → `mypy src/` → `pytest` (with `SLURMIFY_MOCK=1`). A push to
GitHub will block on any lint, type, or test failure. `.pre-commit-config.yaml`
mirrors this locally (ruff + ruff-format + mypy).

**The release pipeline is incomplete (see R1).** `.github/workflows/release.yml`
triggers on `v*` tags and runs `python -m build` + `twine check dist/*`, but
there is **no `twine upload` / `pypa/gh-action-pypi-publish` step and no PyPI
token**. So tagging a release validates the package but never publishes it —
which is the one thing the stated goal needs.

---

## A. Confirmed resolved since last review ✅

All previously-open items, verified against the current source:

| Area | Item | Status |
|------|------|--------|
| Lint gate | N1 (unused imports, stray `f`, bad indent) | ✅ `ruff` clean |
| Flow | N2 double review screen | ✅ Wizard `_advance` calls `app.exit()` at the review boundary; `main()` owns the single review/edit/submit |
| Submit | N3 `submit_sbatch` timeout | ✅ `timeout=30` + `TimeoutExpired` |
| Honesty | N4 SU/ETA labels | ✅ TUI no longer renders its own review; CLI labels “Est. … (rough)” |
| Artifacts | N5 `build/` on disk | ✅ Gone |
| Errors | N6 `except OSError as e` | ✅ Logged via `logger.debug` |
| Cluster | 2.1 partition-aware validation | ✅ `_validate_partition_limits` (CLI) + `_get_warning` (TUI inline) |
| Cluster | 1.8 GPU encoding | ✅ `gpu_format`: `constraint` / `gres_type` / `gpus`; custom-flag de-dup |
| Cluster | 2.3 env activation hard-coded | ✅ `env_type`: conda / mamba / venv / none, in TUI + `--env-type` |
| Cluster | 1.9 task layout | ✅ `--ntasks-per-node=1` emitted when `nodes > 1` (partial — see R8) |
| Cluster | 2.4 log dir | ✅ `submit_sbatch` `mkdir -p`s `--output`/`--error` dirs (partial — see R5) |
| Cluster | 1.7 ETA | ✅ `fetch_queue_eta` now derives ETA from real idle/mix node counts via `sinfo`+`squeue` |
| UX | 3.4 required flag | ✅ `Step.required` |
| UX | 3.5 multi-line command | ✅ `multiline=True` + `multiline_text_area` |
| UX | 3.6 help overlay | ✅ F1/`?` modal |
| UX | 3.7 highlighted preview | ✅ `_tokenize_bash_line` colorizes the live preview |
| Tests | 6.1 memory rejection cases | ✅ `validate_memory("")/"0"/"abc"` asserted |
| Tests | 6.3 conftest | ✅ `tests/conftest.py` sets path + `SLURMIFY_MOCK` |
| Tooling | 7.2 mypy strict | ✅ `strict = true` |
| Tooling | 7.3 release workflow | ⚠️ Exists but does not publish (R1) |
| Tooling | 7.4 pre-commit | ✅ Added |
| Docs | 8.1 README | ✅ Documents batch mode, env vars, removes the false “no temp files” claim |

Dead code from the prior review (`_step_count`, `_has_error`,
`_parse_squeue_time`) is gone. SU is still a heuristic but is now honestly
labeled (R9).

---

## B. Remaining & newly-found issues

### R1 🔴 The release workflow never publishes to PyPI
This is the single biggest gap versus the project goal. `release.yml` ends at
`twine check`. To actually ship on tag you need a publish step, e.g.:

```yaml
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # with Trusted Publishing (OIDC) configured on PyPI — no token in repo,
        # or an API token in secrets.PYPI_API_TOKEN
```

Also recommended: gate the publish on the test job passing, and add a TestPyPI
dry-run on pre-release tags. Until this exists, “as long as Slurm is installed,
`pip install slurmify`” is not yet true — the package isn’t on the index.

### R2 🟠 `NO_COLOR` / non-TTY is only half-honored
`theme._should_use_color()` correctly checks `NO_COLOR`, `TERM=dumb`, and TTY —
but it **only gates the banner**. The status messages in `main.py`
(“Running in batch mode”, “Cancelled.”, “✗ Submission failed”, “✓ Submitted!”,
the job ID) print raw `c.YELLOW` / `c.GREEN` / `c.RED` escape codes
unconditionally. Result: `NO_COLOR=1 slurmify …` and piping to a file still emit
ANSI in those lines. The `rich` panels are fine (rich auto-detects), but the
hand-rolled `c.*` output is not. Route all `c.*` printing through a helper that
returns `""` when `_should_use_color()` is false (or build a no-op `C` instance).

### R3 🟠 GPU encoding is configurable but not reachable from the UI
`gpu_format` only comes from the `SLURMIFY_GPU_FORMAT` env var — there is **no
wizard step and no `--gpu-format` CLI flag**, and the wizard never sets the
`gpu_format` answer key. So the great new flexibility is invisible to users, and
the default `constraint` mode emits `--gres=gpu:N` **plus** `--constraint=<type>`
— which is wrong on most clusters (`--constraint` is for node features, not GPU
models). For a tool meant to “just work everywhere,” the default should be the
portable `--gres=gpu:<type>:N` (`gres_type`), and the choice should be a wizard
step + `--gpu-format` flag.

### R4 🟡 `ruff` config uses deprecated top-level keys
`[tool.ruff]` `select`/`ignore` are deprecated in favor of `[tool.ruff.lint]`.
Ruff prints a warning today; since dev-deps pin only `ruff>=0.3.0`, a future
ruff release in CI could turn this into an error. Move them under
`[tool.ruff.lint]`.

### R5 🟡 Default job output still lands in the submission CWD
`submit_sbatch` now `mkdir -p`s directories named in `--output`/`--error`, but
the **default** `output_path` is `{job}-%j.out` (no directory) — so logs go to
wherever the user ran `slurmify`, and there is no wizard step or `--output-dir`
flag to put them somewhere sane (e.g. `logs/`). The mkdir only helps users who
already know to type a path. Add an output-directory prompt/flag defaulting to
`logs/`.

### R6 🟡 Batch-mode memory is normalized but never validated
`run_batch` calls `normalize_memory(args.memory)` with no `validate_memory`
guard. A bad `--memory foo` passes straight through (`normalize_memory` returns
it unchanged), and the only feedback is a soft partition-limit warning later.
Batch mode should hard-error on invalid `--memory` / `--time` before building.

### R7 🟡 No `--version` flag
`__version__ = "0.1.0"` exists in `__init__.py` but isn’t wired to the CLI. Add
`parser.add_argument("--version", action="version", version=...)` — expected of
any pip-installed tool.

### R8 🟡 Multi-node task layout is a fixed guess
For `nodes > 1` the builder emits `--ntasks-per-node=1 --cpus-per-task=N`. That’s
a reasonable default but wrong for MPI / multi-task workloads, and there’s no way
to set `--ntasks` / `--ntasks-per-node` from the wizard (only via raw custom
flags). Consider a “tasks per node” step shown only when `nodes > 1`.

### R9 🟡 SU estimate is still an invented formula
`estimate_su` = `cpus × hours × nodes`. It ignores GPU weighting and per-partition
charge multipliers, which is how most allocations actually bill. The “(rough)”
label is honest, but consider making the multiplier pluggable
(`SLURMIFY_SU_PER_*`) or dropping SU entirely and keeping only the Slurm-backed
ETA.

### R10 🟡 `requires-python` vs mypy target mismatch
`pyproject` declares `requires-python = ">=3.9"` and ruff targets `py39`, but
`[tool.mypy] python_version = "3.10"`. Type-checking against 3.10 can miss 3.9
incompatibilities while the package still advertises 3.9. Align them (set mypy to
3.9, or raise the floor to 3.10).

### R11 🟡 Minor dead code remains
`Wizard._is_review()` and the “[ Review ]” header branch can no longer render
(the wizard `app.exit()`s at the review boundary). `_can_go_forward()` always
returns `True`. Harmless, but prune to keep the TUI legible.

---

## C. Test gaps (still open from 6.2 / 6.4)

Coverage breadth improved (76 tests, real value assertions for
`_validate_partition_limits` and `validate_memory`), but:

- **6.2** No test exercises `submit_sbatch` (even the mock “sbatch not available”
  path → `(0, "", …)`), `Wizard.run()`’s exit/return path, or the
  partition-select → `_partition_obj` mapping.
- **No parser fixtures.** `fetch_partitions` / `fetch_public_partitions` /
  `fetch_gpu_types_for_partition` / `fetch_queue_eta` are only tested via the
  mock fallback — none against captured *real* `sinfo`/`scontrol`/`squeue` text.
  These parsers are where cluster-to-cluster breakage will actually happen; add
  a `tests/fixtures/` of recorded command output and parse it.
- **6.4** No coverage measurement — add `pytest-cov` to dev-deps and a
  `--cov=slurmify` step (even non-gating) in CI.

---

## D. UX upgrades toward “effortless on a real cluster”

These aren’t bugs — they’re what would make a returning user’s second run
trivial:

- **Config file / saved defaults.** A `~/.config/slurmify/config.toml` (or
  `.slurmify.toml` in CWD) for `account`, `partition`, `gpu_format`, default
  modules. On a single cluster a user re-types the same account every time;
  remembering it is the highest-value usability win.
- **`--dry-run` / `--print`.** A batch flag that prints the script and exits
  without prompting or submitting — for piping into `sbatch` manually or into CI.
- **Post-submit guidance.** After “✓ Submitted! Job ID: 12345”, also print the
  resolved log path and a copy-paste hint: `squeue -j 12345`, `tail -f <log>`,
  `scancel 12345`. Right now the user is left to find the log themselves.
- **Surface the ETA/queue panel in the TUI**, not only in the CLI summary, so
  interactive users see expected wait before deciding.
- **GPU-format and output-dir steps in the wizard** (folds in R3 + R5) so the
  flexible behavior is discoverable without reading env-var docs.

---

## Roadmap

**Phase 0 — Make the PyPI goal real (do first)**
1. Add a real publish step to `release.yml` (Trusted Publishing or token). 🔴
2. Update README install docs to `pipx install slurmify` / `pip install slurmify`
   once the index has it.

**Phase 1 — Correctness polish**
3. Honor `NO_COLOR`/non-TTY across all `c.*` output (R2). 🟠
4. Default GPU format to `gres_type` + add `--gpu-format`/wizard step (R3). 🟠
5. Validate `--memory`/`--time` in batch mode (R6); add `--version` (R7). 🟡
6. Move ruff config under `[tool.ruff.lint]` (R4); align Python target (R10). 🟡

**Phase 2 — Confidence**
7. Real parser fixtures + `submit_sbatch`/`Wizard.run` tests + coverage (C). 🟠

**Phase 3 — Usability**
8. Config-file defaults, `--dry-run`, post-submit hints, output-dir + GPU-format
   wizard steps (D). 🟡

---

## Quick wins (low effort, high value)
- Add the PyPI publish step (R1) — unblocks the entire point of the project.
- Gate `c.*` prints on `_should_use_color()` (R2).
- `--version` (R7) and `--gpu-format` (R3) CLI flags.
- `[tool.ruff.lint]` migration (R4) — silences the warning, future-proofs CI.
- Hard-validate batch `--memory`/`--time` (R6).
