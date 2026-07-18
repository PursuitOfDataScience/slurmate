# Slurmate — Adversarial Audit (v0.3.0)

A fresh, from-scratch audit of the codebase as it stands on `master` (v0.3.0),
run by feeding **malformed, hostile, and edge-case inputs** through every entry
point and confirming each finding with a runnable reproduction. Two independent
passes were run (a primary sweep plus a separate verification audit) and
cross-checked; **only reproduced findings are listed.** This supersedes the old
pre-v0.3.0 planning backlog (see git history for that).

> Every item below marked **(reproduced)** was triggered by actually running the
> code — not by inspection alone. Line numbers are against the current tree.

## Baseline health (the good news)

The project is in genuinely good shape by the usual gates:

- **`pytest`** — 266 passed.
- **`ruff check src/`** — clean.
- **`mypy src/` (strict)** — clean, no issues in 6 files.

The bugs below are the ones those gates **don't** catch: they live in code paths
the tests don't exercise (config-driven batch mode, interactive `_go_back`
navigation, real-cluster tool-availability permutations, non-UTF-8 locales).
Overall test coverage is ~71%, and the gaps line up exactly with the findings —
`main.main()` / `build_and_show` (batch dispatch) and most of `tui.py`'s
navigation are uncovered.

**What held up well under attack** (verified across both passes, no defect
found): the `#SBATCH` directive CR/LF folding (no cross-field injection); the
output/error-path derivation (stdout≠stderr and per-task uniqueness invariants
hold for the full `output_file × array_spec × output_dir` matrix);
`_submit_and_report` (federated `jobid;cluster` stripping, `%A→id`/`%a`
retention, quoted spaced paths, `SLURMATE_LOG_DIR`, submit-failure and mock
paths); `run_batch` numeric/memory/time/gpu-format validation and the
`--print`/`--dry-run`/`--yes` modes (incl. the `--yes` no-command hard error);
`sanitize_job_name` (strips `/`, no path traversal in the auto-saved filename);
the Slurm-output parsers (`fetch_partitions`, `fetch_gpu_types_for_partition`,
`fetch_queue_eta`, `_parse_mem_to_mb`, …) on malformed input; the Rich-markup
escaping in the CLI summary (survives `[/]`, CJK, emoji, 5 000-char lines); the
naive (no-tomllib) config parser; and the TUI skip-index math (forward/back
through skipped ntasks/gpu/env steps, header counter, `edit()` re-entry).

---

## Severity legend

| | |
|---|---|
| **High** | Uncaught crash (traceback) on realistic input through a normal entry point. |
| **Medium** | Crash on a narrower trigger, or silently wrong behavior a user would act on. |
| **Low** | Cosmetic, hard-to-hit, or defense-in-depth / consistency. |

---

## H1 · Batch mode crashes with a raw traceback on wrong-typed config values **(reproduced)**

**Severity: High.**

**Symptom.** A `.slurmate.toml` that gives a **TOML array**, or a wrong scalar
type, for a free-form field makes `slurmate` die with an uncaught
`AttributeError`/`TypeError` traceback — on `--print`, `--dry-run`, **and**
`--yes` (the crash happens in `build_and_show` *before* any real `sbatch`, so
`--yes` never submits, but the user just sees a Python stack trace).

```toml
# .slurmate.toml
command = ["python", "train.py"]   # array where a string is expected
```
```
$ slurmate --print
Traceback (most recent call last):
  ...
  File ".../builder.py", line 402, in build_sbatch_script
    lines.append(command.rstrip())
AttributeError: 'list' object has no attribute 'rstrip'
```

**Every affected field (each reproduced with a list value; several also with a
wrong scalar):**

| Field | Resolved raw at | Crashes at | Exception |
|---|---|---|---|
| `command` | `main.py:223` | `builder.py:402` `command.rstrip()` | AttributeError |
| `partition` | `main.py:75` | `builder.py:280` `_fold_directive(...)` → `.replace` | AttributeError |
| `account` | `main.py:205` | `builder.py:282` | AttributeError |
| `qos` | `main.py:208` | `builder.py:284` | AttributeError |
| `array_spec` | `main.py:217` | `builder.py:326` | AttributeError |
| `output_dir` | `main.py:111` | `builder.py:132` `os.path.expanduser` | TypeError |
| `output_file` | `main.py:114` | `builder.py:135` `os.path.expanduser` | TypeError |
| `env_name` | `main.py:180` | `builder.py:390` `shlex.quote` | TypeError |
| `env_type` | `main.py:177` | `builder.py:387` `(env_type or "conda").lower()` | AttributeError |
| `modules` (non-str elems, e.g. `[1, 2]`) | `main.py:164` | `builder.py:378` `mod.endswith` | AttributeError |

**Root cause.** `run_batch` carefully coerces the *numeric* and a few string
fields — `cpus`/`nodes`/`gpus`/`ntasks` via `_coerce_int`, and
`memory`/`time`/`job_name`/`gpu_type` via `str(...)`/validation — but passes the
free-form string fields above through **verbatim** from config. The builder then
calls string/path methods on them. (`custom_sbatch` is *safe* only because the
builder happens to `str()` each element at `builder.py:349`.)

**Why the tests miss it.** `load_config()` returns `{}` under `SLURMATE_MOCK`
(`system_utils.py:1075`), so the whole test suite never feeds a real config into
`run_batch`. And the **interactive wizard is immune** — it stringifies every
config value into `_config_defaults` (`tui.py:322-329`), joining lists with
`", "` — so only *batch mode* is exposed. Control check: a well-formed
all-string config builds and prints correctly (exit 0).

**Fix.** Normalize types once, at the top of `run_batch`, the same way
`_coerce_int` already guards the numerics — e.g. a small `_coerce_str(value,
field, err_console)` that either str-coerces or rejects a list/dict with a clean
`✗ Error: <field> must be a string` and `sys.exit(1)`, applied to
partition/account/qos/array_spec/command/output_dir/output_file/env_name/env_type,
and `str()`-map the `modules`/`custom_sbatch` elements. This matches the existing
philosophy (the tree already hardened `gpus = "2"` and other stringy configs);
the standard here is a clean error message, never a traceback. (See also **L7**,
the direct-API `gpu_type` variant of the same "builder assumes types" family.)

---

## M1 · Wizard crashes on "go back" when a numeric field holds a non-integer **(reproduced)**

**Severity: Medium-High.**

**Symptom.** In the interactive wizard, type a non-integer into **CPU cores**,
**Nodes**, or **Tasks per node** (e.g. `abc`, `3.5`, `8 cores`, `1e3`) and then
press **Esc** or **Shift-Tab** to go back and fix an earlier answer. The wizard
dies with an uncaught `ValueError`:

```
ValueError: invalid literal for int() with base 10: '3.5'
```

Reproduced for `cpus="abc"`, `cpus="3.5"`, `nodes="1e3"`,
`ntasks_per_node="two"`, `cpus="8 cores"` — all crash.

**Where.** `Wizard._go_back` re-saves the current field by calling
`self._coerce(val, s)` at **`tui.py:625` / `tui.py:629`** *without first running
the step's validator*. `_coerce` then does `int(val)` for cpus/nodes/ntasks
(`tui.py:653,657,659`).

**Root cause / asymmetry.** The forward path `_confirm_and_next` **does** gate on
the validator first (`tui.py:573`: `if val and s.validate and not s.validate(val):
… return`), so a bad value can never reach `_coerce` going forward. `_go_back`
skips that guard entirely. The raised `ValueError` escapes the prompt_toolkit
key-binding handler; `Wizard.run()` only catches `KeyboardInterrupt`/`EOFError`
(`tui.py:1367`), so it propagates to `main()` as a traceback and leaves the
terminal in the full-screen app's raw state.

**Fix.** Mirror the forward guard in `_go_back`: only persist the value if it
passes `s.validate` (otherwise keep the previous answer), or make `_coerce`
tolerant (wrap the `int(...)` in try/except, falling back to the config-aware
default). The forward-path guard is the model to copy.

---

## M2 · Interactive QoS picker is empty on the common `AllowQos=ALL` case (and drops real QoS when `sacctmgr` is missing) **(reproduced)**

**Severity: Medium** (interactive workaround exists — `--qos` / config).

**Symptom.** On any partition whose `AllowQos=ALL` — **Slurm's default, i.e. most
partitions** — the wizard's QoS step offers only `Default (none)`; you can never
pick a QoS interactively.

```
fetch_qos_for_partition("gpu") -> ["ALL"]        # real Slurm default
after TUI intersection          -> ['Default (none)']   # the QoS list is empty
```

A second trigger: when `sacctmgr` isn't reachable, `fetch_known_qos()` falls back
to the 5 hard-coded `MOCK_QOS` demo names, so any real QoS returned by `scontrol`
that isn't in that list (e.g. a lab-specific QoS) is silently filtered out.

**Where.** `tui.py:812` intersects the partition's `AllowQos`
(`fetch_qos_for_partition`, `system_utils.py:504-513`, returned verbatim) with
`fetch_known_qos()`. Two problems compound: (a) `"ALL"` is a **sentinel, not a
QoS name**, so it never survives the intersection; (b) `fetch_known_qos()` falls
back to `MOCK_QOS` when `sacctmgr` is unavailable (`system_utils.py:522`), turning
a demo list into an authoritative filter over live data.

**Fix.** Special-case `AllowQos=ALL` (offer all known QoS, or skip the filter
entirely and trust `scontrol`); and when `sacctmgr` is genuinely unavailable,
return `[]` from `fetch_known_qos` and have the TUI skip the intersection when the
known-set is empty/unknown, rather than filtering against `MOCK_QOS`.

---

## M3 · Script file I/O uses the locale encoding, not UTF-8 → uncaught `UnicodeError` under a non-UTF-8 locale **(reproduced)**

**Severity: Medium.**

**Symptom.** If a generated script contains any non-ASCII byte (a command,
comment, path, or module name with an accented/unicode char) and the process
locale isn't UTF-8, saving or editing the script crashes with an uncaught
`UnicodeEncodeError`/`DecodeError`. The `_save_submitted_script` variant is the
worst: it runs **after** `sbatch` succeeds, so the job **is** submitted and then
the tool tracebacks before printing the log path and follow-up hints.

```
$ LC_ALL=C LANG=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 slurmate ...
  File ".../main.py", line 444, in _save_submitted_script
    f.write(script)
UnicodeEncodeError: 'ascii' codec can't encode character '\xe9' ...
```

**Where.** The only file-I/O paths missing `encoding="utf-8"`:
`_edit_script_in_editor` (`main.py:391-397` — `NamedTemporaryFile(mode="w")`
write + plain `open()` read), `_save_script` (`main.py:423`), and
`_save_submitted_script` (`main.py:443-444`). The `except OSError` in the two
`_save_*` functions does **not** catch it — `UnicodeEncodeError` is a `ValueError`,
not an `OSError` (verified: `issubclass(UnicodeEncodeError, OSError)` is `False`).

**Root cause.** The sibling *subprocess* I/O was explicitly hardened with
`encoding="utf-8", errors="replace"` against exactly this C/POSIX-locale hazard
(`system_utils._run_command`, `submit_sbatch`), but these file paths were missed.
(Modern Python coerces a bare `LC_ALL=C` to `C.UTF-8`, masking the simplest case
— hence Medium — but genuine non-UTF-8 locales, e.g. `*.ISO-8859-1`, still hit it.)

**Fix.** Pass `encoding="utf-8"` to all three (the `NamedTemporaryFile`, the
read-back `open`, and both `_save_*` opens); `errors="replace"` on the read-back
mirrors the subprocess treatment.

---

## L1 · Module names are injected into `module load` unquoted (inconsistent with `env_name`) **(reproduced)**

**Severity: Low** (same-user threat model — it's the user's own generated
script — but a real inconsistency and a defense-in-depth gap).

**Symptom.** The builder shell-quotes the environment name but **not** module
names, so shell metacharacters in a module survive into the generated script:

```
mods=['cuda; rm -rf ~']   -> 'module load cuda; rm -rf ~'
mods=['$(whoami)']        -> 'module load $(whoami)'
env='e;rm -rf ~'          -> "source activate 'e;rm -rf ~'"   # env_name IS quoted
```

**Where.** `builder.py:384` (`f"module load {mod}"`, only CR/LF-folded) vs the
`shlex.quote(env_name)` at `builder.py:390/393/396`.

**Fix.** `shlex.quote` each module token (they never legitimately need shell
metacharacters), matching `env_name`. Also correctness: a module name with a
space currently becomes two `module load` arguments.

---

## L2 · `validate_time` accepts out-of-range time fields **(reproduced)**

**Severity: Low** (Slurm is the last line of defense at submit).

**Symptom.** The validator checks digit *counts*, not ranges, so obviously
invalid times pass client-side validation:

```
'1:60:60'      validate_time=True    # 60 min / 60 sec
'1-99:99:99'   validate_time=True
'0'            validate_time=True     # zero-length limit
```

**Where.** `_TIME_PATTERNS` / `validate_time`, `system_utils.py:135-150` (the
`\d{2}` groups match `60`–`99`).

**Fix.** Either accept it as "good enough, Slurm will reject it," or tighten the
minutes/seconds groups to `[0-5]\d` and reject an all-zero limit. Low priority.

---

## L3 · `_detect_gpu_type` returns false-positive GPU models **(reproduced)**

**Severity: Low** (only on count-only-GRES clusters with odd feature lists;
cosmetic — surfaces a bogus type in the picker).

**Symptom.**

```
features='p9,power9'  gres='gpu:1'   -> 'power9'   # POWER9 is a CPU
features='h100'*50    gres='gpu:1'   -> 250-char garbage token
```

**Where.** `system_utils.py`. The `_CPU_GEN_TOKENS` set catches `p9` but not the
spelled-out `power9` (`:194`), and the positive-match branch (`:236-240`) returns
the token straight from `_GPU_MODEL_RE` **without** the `len(token) >= 15` sanity
cap that the negative-filter branch applies (`:250`).

**Fix.** Apply the length cap in the positive branch too, and add `power9`/`power`
(and similar spelled-out CPU codenames) to the exclusion set.

---

## L4 · `estimate_su` yields a negative SU for negative CPUs **(reproduced)**

**Severity: Low** (not reachable via CLI/TUI — both reject `cpus <= 0` — so
direct-library-API only).

```
estimate_su(-4, '01:00:00', 1, None) -> '-4.00'
```

**Where.** `builder.py:412-438` — no guard on a negative `cpus`/`nodes`. The time
side is already clamped (`minutes <= 0 → 120`); the core count isn't.

**Fix.** `cpus = max(0, cpus)` (and same for `nodes`) at the top, for parity with
the time clamp. Cheap hardening for a public function.

---

## L5 · `--yes` submits a no-op job for a whitespace/comment-only command **(reproduced)**

**Severity: Low.**

**Symptom.** The unattended-submit guard `if not answers.get("command")`
(`main.py:631`) treats `"   "` or `"# note"` as a real command, so
`slurmate --command "   " --yes …` would submit a job whose body is empty/only a
comment — exactly the no-op the guard is meant to prevent.

**Fix.** Strip and drop a leading-`#`/empty command before the check (e.g.
`cmd = (answers.get("command") or "").strip()`, then treat an empty or
comment-only `cmd` as missing).

---

## L6 · Partition step doesn't restore the prior selection when you navigate back **(reproduced)**

**Severity: Low** (UX only — no data loss).

**Symptom.** Every other step restores its previous value on Back; the partition
step rebuilds its radio with the cursor at index 0 (`Enter partition name
manually…`) instead of re-highlighting the partition you'd chosen. Pressing Enter
without moving therefore drops into the manual-text sub-flow. The prior partition
is preserved in `answers` and pre-filled into the text box (so a second Enter
re-confirms it) — annoying, not corrupting.

**Where.** `tui.py:828-852` (`_setup_partition` rebuilds the radio without calling
`_set_radio_default`), reached via the `_go_back` partition branch
(`tui.py:591-605`).

**Fix.** After building the choices, `_set_radio_default` to the formatted label
of the current `answers["partition"]`.

---

## L7 · `build_sbatch_script` crashes on a non-string `gpu_type` (direct-API only) **(reproduced)**

**Severity: Low** (not reachable through CLI/TUI — `run_batch` does
`str(gpu_type)` at `main.py:104-105` and the wizard coerces all fields; only a
direct library caller can trigger it).

```
build_from_answers({"gpu_type": 0, "command": "x"})
-> AttributeError: 'int' object has no attribute 'lower'   # builder.py:305
```

**Root cause.** Same "builder assumes types" family as **H1** — the builder
defensively `int()`-coerces `gpus`/`nodes` (`builder.py:262-270`) but not
`gpu_type`, an inconsistency. (A latent, effectively-unreachable sibling:
`fetch_conda_envs` iterates a non-list `envs` value char-by-char at
`system_utils.py:653`; `conda info --json` always returns a JSON array, so no
in-app trigger.)

**Fix.** Coerce `gpu_type = str(gpu_type)` (or guard the `.lower()`) alongside the
existing `gpus`/`nodes` coercion.

---

## How to reproduce

```bash
# from a checkout, with a Python 3.10+ interpreter that has the runtime deps:
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# baseline gates
SLURMATE_MOCK=1 pytest -q          # 266 passed
ruff check src/                    # clean
mypy src/                          # clean

# H1 — malformed config crash (do this in a scratch dir; --print never submits):
printf 'command = ["python", "train.py"]\n' > .slurmate.toml
slurmate --print                   # AttributeError traceback

# M1 — go-back crash: launch `slurmate`, on "CPU cores" type `3.5`, press Esc.

# M3 — non-UTF-8 file-I/O crash:
LC_ALL=C LANG=C PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 python -c \
  'import slurmate.main as m; m._save_submitted_script("#!/bin/bash\necho café\n","j","1","/tmp")'
```

**Priority order to fix:** **H1** (worst UX, plausible config typo, hits every
batch flag) → **M1** (interactive crash on a normal navigation action) → **M2**
(empty QoS picker on the default cluster config) → **M3** (crash *after* a
successful submit under a non-UTF-8 locale) → **L1–L7** (hardening / consistency).
