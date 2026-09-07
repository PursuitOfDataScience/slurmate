# Slurmate — Issues & Problems (audit + resolutions)

An audit of the whole codebase at **v0.5.1**, and a record of how each finding was
fixed in **v0.5.2**.

**Status:** every finding below is **resolved** except two that were investigated
and deliberately left alone (see [Withdrawn / not changed](#withdrawn--not-changed)).
**L11** was the last one still open and is now fixed; its entry records the
before/after and the one change inside `_print_indented` that it needed.
Each entry has a **Fix** block saying what
changed, where, why that choice, and how the fix was verified. A post-fix probe of the new code found three further problems
(several introduced or exposed by the fixes themselves), and a round-trip of
generated scripts through the real controller found two more — those are in
[Found while verifying the fixes](#found-while-verifying-the-fixes-p1p6), also fixed.

**Baseline at v0.5.1:** 299 tests passing, `ruff check src/` and `mypy src/` clean.
**After the fixes:** **431 tests** passing (132 added, 7 updated), ruff and mypy
clean, version bumped to 0.5.2 with a CHANGELOG entry.

## How these were verified

Two passes:

1. **Code execution in mock mode** (`SLURMATE_MOCK=1`) for everything that is a pure
   slurmate behaviour — script generation, the summary panels, the wizard state
   machine, the validators.
2. **Against real Slurm**, for every claim about what *Slurm* does with slurmate's
   output. This machine (UChicago Midway3, `/software/slurm-current-el8-x86_64`) has
   a live `sbatch`, so each Slurm-behaviour claim was settled with
   `sbatch --test-only` (validates and reports scheduling, **submits nothing**)
   rather than assumed.

That second pass earned its keep: it **overturned two claims** an earlier mock-only
audit had recorded (L7's time half, L8's `CLICOLOR_FORCE`), **inverted the stated
impact** of a third (M2), and **surfaced a defect mock mode structurally cannot
show** (**H2** — mock partitions all carry typed GPU GRES, while the real cluster's
GPU partitions are count-only).

Reproduction environment for the mock snippets:

```bash
source /software/python-miniforge-25.3.0-el8-x86_64/bin/activate AI   # or any env with the deps
cd /home/youzhi/slurmate
export SLURMATE_MOCK=1
```

Severity reflects likelihood × impact. Line references in the *finding* text are to
**v0.5.1**; references in a **Fix** block are to the current tree.

---

## High severity

### H2 — On a count-only-GRES cluster, the default `gpu_format=gres_type` emits a request Slurm rejects, using a GPU type slurmate itself offered — and validation says nothing is wrong

**Where:** `fetch_gpu_types_for_partition` (`system_utils.py:570-624`) lost the
*provenance* of each GPU model it returned; `build_sbatch_script`
(`builder.py:385-403`) then formatted it as `--gres=gpu:<type>:N`;
`validate_job_config` (`system_utils.py:404-413`) only checked membership, never
requestability.

Many sites configure GPUs **count-only** (`Gres=gpu:4` in `gres.conf`, no type) and
put the GPU model in the node's *feature* list instead. slurmate handled that on the
read side: when a node's GRES has no type, `_detect_gpu_type` scans `%f` features, so
the picker still offers real model names. But the write side then requested them as a
**GRES type**, which does not exist on such a cluster, and the job was rejected.

This is the most impactful finding here because every part of it is the *default*
path: the type comes from slurmate's own picker, `gres_type` is the default
`--gpu-format`, and `--dry-run` reported no problem.

Verified end-to-end on this cluster's `gpu` partition (`sinfo -h -o "%P|%G"` →
`gpu|gpu:4`, i.e. count-only; node features are `gold-6248r,384g,a100`):

```console
$ python -c "from slurmate.system_utils import fetch_gpu_types_for_partition as f; print(f('gpu'))"
['a100', 'rtx6000', 'v100']          # all three are FEATURE-derived, not GRES types

$ slurmate --job-name x --partition gpu --gpus 1 --gpu-type a100 --command true --print | grep gres
#SBATCH --gres=gpu:a100:1

$ slurmate ... --dry-run
(no error, no warning)

$ sbatch --test-only -p gpu --gres=gpu:a100:1 -t 5 --mem=1G --wrap=true
allocation failure: Requested node configuration is not available     # ← rejected

$ sbatch --test-only -p gpu --gres=gpu:1 -C a100 -t 5 --mem=1G --wrap=true
sbatch: Job 52616148 to start at 2026-07-25T01:12:55 using 1 processors on nodes midway3-0294
```

The same applied to `--gpus=<type>:N`, `--gpus-per-node` and `--gpus-per-task`: every
format except `constraint` names the type inside the GRES request. slurmate already
had the information needed to know this — `fetch_gpu_types_for_partition` saw which
models came from a typed `gpu:MODEL:N` and which came from features — it just threw
that distinction away before the caller could use it.

> **Fix — track provenance, then check requestability (not just membership).**
>
> - **`system_utils.fetch_gpu_type_sources(partition)`** (new) returns
>   `{"typed": [...], "feature": [...]}`: models seen in a real `gpu:MODEL:N` vs.
>   models found only in node features. A model corroborated by a typed GRES
>   *anywhere* in the partition is promoted to `typed`, so a mixed partition (some
>   nodes typed, some count-only) is not flagged — the request can be satisfied.
>   `fetch_gpu_types_for_partition` is now a thin flattening wrapper, so every
>   existing caller and the picker keep working.
> - **`validate_job_config(..., feature_only_gpu_types=…)`** raises an
>   **error-level** issue when a feature-only model is requested through any
>   type-naming format, and names the remedy: *"GPU type 'a100' is a node feature on
>   'gpu', not a GRES type (the nodes advertise a count-only 'gpu:N'), so gpu_format
>   'gres_type' would emit a request Slurm rejects — use gpu_format 'constraint'"*.
>   An unset `gpu_format` is resolved the way the builder resolves it
>   (`SLURMATE_GPU_FORMAT`, else `gres_type`), so the default path is covered.
>   Error, not warning, because it is a certainty: no node advertises that GRES type,
>   so Slurm *will* reject it — which means `_hard_errors` now blocks the submit
>   instead of wasting a round-trip.
> - **`main._partition_issues`** passes the provenance through. It still only does
>   the live lookup when the requested type isn't in the partition's static list —
>   anything in that list came from a typed `gpu:MODEL:N` by construction (that is
>   all `fetch_partitions` records), so the "avoid a needless `sinfo` call"
>   optimisation is preserved exactly. A failed lookup degrades to silence.
> - **`tui._setup_gpu_format`** defaults the radio to `constraint` when the chosen
>   model is feature-only, so an interactive user never walks into the error in the
>   first place; an explicit prior choice is still respected.
> - Mock mode reports its GPU types as `typed`, so demos and the test suite see no
>   spurious mismatch.
>
> **Verified** on the real cluster, end to end:
>
> ```console
> $ slurmate --partition gpu --gpus 1 --gpu-type a100 … --dry-run
>   ✗ Error: GPU type 'a100' is a node feature on 'gpu', not a GRES type …
> $ slurmate --partition gpu --gpus 1 --gpu-type a100 … --yes
>   ✗ Refusing to submit — Slurm would reject this job
> $ slurmate --partition gpu --gpus 1 --gpu-type a100 --gpu-format constraint … --print
> #SBATCH --gres=gpu:1
> #SBATCH --constraint=a100
> $ sbatch --test-only -p gpu --gres=gpu:1 --constraint=a100 …
> sbatch: Job 52616717 to start at 2026-07-25T01:12:55 … on nodes midway3-0294   ← accepted
> $ slurmate --partition test --gpus 1 --gpu-type a30 … --dry-run   # typed-GRES partition
> (no errors)  →  #SBATCH --gres=gpu:a30:1                          ← no false alarm
> ```
>
> **Tests:** `TestGpuTypeProvenance` (5), `TestFeatureOnlyGpuFormatValidation` (6),
> `TestGpuFormatDefaultsToConstraintForFeatureOnlyType` (4),
> `TestFeatureOnlyGpuTypeBlocksSubmit` (5) — including "no live lookup when the type
> is statically known" and "lookup failure degrades quietly".

### H1 — A custom `--constraint` flag is never merged, producing two conflicting `#SBATCH --constraint=` lines (Slurm silently drops one)

**Where:** `builder.py:371-407` (constraint merge) vs. the custom-flag loop
`builder.py:422-462` (only an *exact-GPU-type* dedup at `459-461`).

The builder correctly merged the `constraint` parameter and the GPU-type-as-constraint
into one `&`-joined directive. But a user-supplied `--constraint=` in `custom_sbatch`
was **not** merged into it: any custom `--constraint` whose value differed was
appended as a *second* `#SBATCH --constraint=` line — and because the custom loop runs
last, the directive that got discarded was always **slurmate's own**. This is exactly
the hazard the code's own comment (`builder.py:366-370`) says the merge exists to
prevent; the merge just didn't cover the custom-flag path.

```bash
slurmate --job-name t --partition p --gpus 2 --gpu-type a100 \
  --gpu-format constraint --custom-sbatch=--constraint=bigmem --command x --print
```

```
#SBATCH --constraint=a100          ← lost
#SBATCH --constraint=bigmem
```

Also reproduced with the `--constraint` param + a differing custom `--constraint`
(`constraint="cpu"` + `custom_sbatch=["--constraint=bigmem"]` → `cpu` dropped).

**"Slurm keeps the last one" is measured, not assumed.** Ordering a real `sbatch`
request so an invalid feature is first vs. last is a clean discriminator:

```console
$ sbatch --test-only -p gpu --gres=gpu:1 -C NOSUCHFEATURE -C a100 …   # bogus FIRST
sbatch: Verification: ***PASSED***
sbatch: Job 52616146 to start at 2026-07-25T01:12:55 … on nodes midway3-0294

$ sbatch --test-only -p gpu --gres=gpu:1 -C a100 -C NOSUCHFEATURE …   # bogus LAST
allocation failure: Invalid feature specification
```

The bogus-first case schedules fine, so the earlier `-C` is discarded outright and
silently. The remedy is also confirmed valid Slurm syntax:

```console
$ sbatch --test-only -p gpu --gres=gpu:1 -C 'a100&384g' …             # both real
sbatch: Job 52616189 to start at 2026-07-25T01:12:55 … on nodes midway3-0294
$ sbatch --test-only -p gpu --gres=gpu:1 -C 'a100&NOSUCHFEATURE' …
allocation failure: Invalid feature specification                      # AND confirmed
```

> **The old behaviour was codified by a test**, so this was a decision to revisit, not
> merely an oversight: `tests/test_builder.py::test_gpu_format_duplicate_filtering`
> fed exactly this input and asserted `"#SBATCH --constraint=ssd" in script` — it
> *required* a second directive. The intent behind it (`builder.py:436-443`) was
> "never silently discard a user override", which is right for `--gres`/`--gpus`
> (last-wins gives the user what they asked for) but backfires for `--constraint`,
> where last-wins destroys the other requirement.

**Severity note:** the impact when triggered is a silent wrong-node placement, but it
needs `--gpu-format constraint` (not the default) or a `constraint` param, *plus* a
custom `--constraint`. On its own that reads **medium**; it is filed High because
**M4** actively pushed interactive users on constraint-based sites (NERSC Perlmutter's
mandatory `-C cpu`/`-C gpu`) into precisely that combination.

> **Fix — merge every constraint source into one directive.**
>
> - `build_sbatch_script` pre-scans the normalised custom flags for
>   `--constraint`/`-C` (either spelling, `=` or space), collects their values, and
>   marks those entries **consumed** so the emit loop skips them
>   (`builder.py`, `custom_constraints` / `consumed_custom`).
> - Values are appended after the GPU block, so the merged value reads
>   **param → GPU type → custom** (`cpu&a100&bigmem`) — deterministic and readable.
> - **De-dup is case-sensitive**, driven by the M6 measurement below: `A100` and
>   `a100` are *different* Slurm features, so folding them would silently change the
>   request. An exact duplicate still collapses (`a100` + `a100` → `a100`).
> - An OR-expression is parenthesised when merged (`_constraint_term`):
>   `gpu` + `a100|v100` → `gpu&(a100|v100)`, not the ambiguous `gpu&a100|v100`. A
>   value already grouped (`(…)`) or bracketed (`[…]`, Slurm's count syntax) is left
>   alone, and a lone OR-expression is emitted verbatim.
> - The now-redundant exact-GPU-type dedup at the old `459-461` is gone (all
>   constraint flags are consumed before the loop), replaced by a comment saying so.
> - The test that codified the old behaviour was **updated, not deleted**, with a
>   comment recording why the expectation changed.
>
> **Verified** through the real CLI *and* against real Slurm:
>
> ```console
> $ slurmate --partition gpu --gpus 1 --gpu-type a100 --gpu-format constraint \
>     --custom-sbatch="--constraint=384g" … --print
> #SBATCH --gres=gpu:1
> #SBATCH --constraint=a100&384g
> $ sbatch --test-only -p gpu --gres=gpu:1 --constraint='a100&384g' …
> sbatch: Job 52616721 to start at … on nodes midway3-0294        ← accepted
> ```
>
> **Tests:** `TestCustomConstraintMerge` (8) — GPU-type merge, param merge, short/space
> forms, exact-duplicate collapse, case-sensitivity, OR parenthesisation, lone OR
> untouched, multiple custom constraints — plus `test_split_constraint_still_merges`.

---

## Medium severity

### M1 — Custom `--output`/`--error`/`-o`/`-e` create duplicate log directives, and the reported "Log path" / `tail -f` hint points at the wrong file

**Where:** builder emitted the auto `--output`/`--error` at `builder.py:414-420`
*before* the custom-flag loop (`422-462`), with no dedup; reporter read the wrong one
at `main.py:586-599`.

Two problems compounded:

1. A custom `--output=`/`--error=` flag yielded a **second** output directive. Slurm
   uses the last, so the job wrote where the user intended — but the script carried a
   redundant, contradictory auto directive. (Contrast `--mem`, where `_custom_mem` at
   `builder.py:341-356` already suppressed the auto directive for exactly this reason.)
2. `_submit_and_report` scanned for the **first** `#SBATCH --output=` and `break`ed —
   the *auto* one Slurm ignored — so "Log path:" and the `tail -f` hint pointed at a
   file the job never creates.

```
#SBATCH --output=logs/j-%j.out          ← auto (ignored by Slurm)
#SBATCH --error=logs/j-%j.err
#SBATCH --output=/real/place/%j.log     ← user's (wins)
```
→ report said `Log path: logs/j-12345.out`, job wrote `/real/place/12345.log`.

**Precision on the short form** (an earlier audit was vague here): the claim is real
but only reachable through the list-valued paths, and the CLI path failed differently
— and worse.

- `custom_sbatch = ["-o /real/place/%j.log"]` (single config/API element) →
  `#SBATCH -o /real/place/%j.log`, valid for Slurm, duplicate auto directive present,
  reported path wrong (the `startswith("#SBATCH --output=")` scan never matches).
- `--custom-sbatch="-o /real/place/%j.log"` (the CLI) → `_parse_custom_flags` returned
  `['-o', '--/real/place/%j.log']` and the script got **two broken directives**.
  Tracked separately as **M5**.
- The "falls back to the `{job_name}-%j.out` default" wording only applies to a
  hand-edited script in which the `--output=` line was replaced by `-o`; a generated
  script always contains an auto `--output=` for the scan to find.

> **Fix — suppress per stream, and resolve the *effective* path.**
>
> - `build_sbatch_script` now skips the auto `--output` when a custom
>   `--output`/`-o` is present, and the auto `--error` when a custom `--error`/`-e`
>   is present — **independently**, so overriding only stdout keeps the derived
>   stderr path. Mirrors the pre-existing `--mem` suppression.
> - **`system_utils.effective_log_path(script, kind)`** (new) returns the path Slurm
>   will actually use by keeping the **last** matching directive, and understands
>   every spelling (`--output=P`, `--output P`, `-o P`, and the `--error`/`-e`
>   equivalents — see L4). `_submit_and_report` uses it for both the "Log path:" line
>   and the `tail -f` hint, falling back to the job-name default only when the script
>   truly has no output directive.
>
> **Verified:** the M1 repro now emits one `--output` (the user's), and the report
> prints `Log path: /real/place/12345.log` with a matching `tail -f`.
>
> **Tests:** `TestCustomLogFlagDedup` (4), `TestReportedLogPath` (4, including the
> space form and the no-directive fallback), `TestSbatchLogPathForms` (5).

### M5 — `_parse_custom_flags` mangles any space-separated option value into a bogus `--<value>` flag

**Where:** `tui.py:246-283`.

The parser turned a bare (non-`-`) token into a standalone `--token`. Right for
`exclusive` → `--exclusive`, wrong whenever the token is the *value* of the preceding
option — which is how every short Slurm option is written:

```python
p("-o /real/place/%j.log")   # -> ['-o', '--/real/place/%j.log']
p("-C bigmem")               # -> ['-C', '--bigmem']
p("--nodelist n1")           # -> ['--nodelist', '--n1']
```

Each emitted two `#SBATCH` lines, one valueless and one nonsense
(`#SBATCH --/real/place/%j.log`), so sbatch rejected the whole script. The docstring
documented the behaviour ("the wizard can't know which options take a value"), so it
was deliberate — but `--/real/place/%j.log` is not a valid option under any
interpretation, and `-C bigmem` / `-o path` / `-w node1` is how people actually type
Slurm flags. This also widened H1/M1's blast radius: a user typing `-C bigmem` got a
rejected script instead of the merge path.

> **Fix — one shared rule for every input path.**
>
> - **`builder._join_flag_values(tokens)`** (new) is now the single implementation:
>   a bare token is attached to the preceding option when that option is in
>   **`_VALUE_TAKING_FLAGS`** (a curated set of value-taking sbatch short *and* long
>   options; boolean options are deliberately absent) **or** when the token cannot be
>   an option name (`_OPTION_NAME_RE` — a path, pattern or bracket list). Otherwise
>   the old "user forgot the dashes" behaviour stands, so `exclusive hold` still
>   yields two flags. An option already carrying a value (`=` or a space) does not
>   absorb a second token.
> - `tui._parse_custom_flags` keeps its quote-aware `shlex` tokenising and
>   comma-splitting, then delegates to `_join_flag_values`.
> - **`builder._normalize_custom_flags`** also runs it, which fixes the parallel
>   list-path bug: a TOML `custom_sbatch = ["-o", "/logs/%j.out"]` (option and value
>   as two array elements) used to emit `#SBATCH -o` plus a bare path line. All three
>   spellings now converge on one directive — asserted by
>   `test_all_three_spellings_agree`.
> - The set lives in `builder` (which `tui` already imports) so there is no circular
>   import and no duplicated vocabulary.
>
> **Verified** against real Slurm:
>
> ```console
> $ slurmate --custom-sbatch="-C 384g -w midway3-0294" … --print
> #SBATCH --constraint=384g        ← merged via H1
> #SBATCH -w midway3-0294          ← one valid directive
> $ sbatch --test-only -p gpu --constraint=384g -w midway3-0294 …
> sbatch: Job 52616722 to start at … on nodes midway3-0294        ← accepted
> ```
>
> **Tests:** `TestSpaceSeparatedCustomFlags` (10) and `TestSplitListElementFlags` (6),
> including a regression guard that the rejoin pass cannot reintroduce a newline.

### M2 — `_detect_gpu_type` reports fabric / rack / infra node-feature tokens as bogus GPU model names

**Where:** `system_utils.py:203-289` — the positive shape regex `_GPU_MODEL_RE`
(`192-194`) and the negative-filter blocklist (`273-286`); leaked out through
`fetch_gpu_types_for_partition`.

Two separate holes in the count-only-GRES feature scan.

*Negative filter* — the blocklist covered CPU vendors/codenames/ISA and even
`ib`/`opa`/`hdr`/`hdd`, but not the InfiniBand *generation* tags or rack labels that
sit in feature lists at many sites:

```python
_detect_gpu_type("hdr100", "gpu:4")                            # -> 'hdr100'   (expected 'gpu')
_detect_gpu_type("edr",    "gpu:4")                            # -> 'edr'      ; also fdr, ndr, hdr200, roce
_detect_gpu_type("rack2,edr", "gpu:4", known_models={"a100"})  # -> 'rack2'
_detect_gpu_type("gold6248,avx512,hdr100,768g", "gpu:4")       # -> 'hdr100'
```

That the list already contained `ib`/`opa`/`hdr` is itself evidence the fabric-tag
convention was anticipated; the generation variants were just missed.

*Positive shape regex* — the more dangerous half, because it ran **first** and so beat
a real model later in the list:

```python
_detect_gpu_type("b12,a100", "gpu:4")   # -> 'b12'   (a100 is right there)
_detect_gpu_type("t2,a100",  "gpu:4")   # -> 't2'
```

**Impact — the earlier audit had this backwards.** It recorded a "false OK from
validation"; that is vacuous. On a count-only partition `part["gpu_types"]` is empty,
so the membership check is skipped entirely and the result is `[]` either way:

```python
validate_job_config({...,"gpu_type":"hdr100"}, extra_gpu_types=["hdr100"])  # -> []
validate_job_config({...,"gpu_type":"hdr100"}, extra_gpu_types=[])          # -> []   (same)
```

The real consequence is the opposite, and worse — a bogus token becomes the *only*
known model, so a request for a genuinely available GPU is turned into a **false hard
error that blocks submission**:

```python
validate_job_config({...,"gpu_type":"a100"}, extra_gpu_types=["hdr100"])
# -> [('error', "GPU type 'a100' not in partition list (hdr100)")]
```

Plus bogus rows in the picker. Does **not** reproduce on this cluster (features here
are `<cpu>,<mem>,<gpu>` with no fabric tags), so the impact is reasoned rather than
measured — but the two-character-label half is one naming convention away from
breaking a correct request.

> **Fix — allowlist first, stricter shape second, infra filtered from both.**
>
> - **`_KNOWN_GPU_MODELS`** (new): explicit model names (NVIDIA datacenter +
>   workstation, AMD Instinct, Intel). Checked before any heuristic, over *all*
>   tokens, so `b12,a100` now resolves to `a100` — position no longer decides.
> - **`_GPU_MODEL_RE` tightened** from family-letter + 1 digit to **3-plus digits**,
>   which is what removes the `b12`/`t2`/`p2` class entirely while still catching a
>   model too new for the list (`h300`, `mi450`). Short real models (`t4`, `l4`,
>   `a30`, `a40`, `l40s`, `k80`, `mi50`) are covered by the allowlist, and anything
>   the allowlist misses still reaches the negative filter, so no real type is lost.
> - **`_INFRA_TOKEN_RE`** (new) filters IB generations (`sdr…ndr`, with optional
>   digits, so `hdr100`/`hdr200` are covered), fabrics/NICs (`ib*`, `opa`, `roce`,
>   `mlx`, …), rack/chassis/position labels (`rack2`, `row3`, `pod1`, `blade2`, …),
>   GPU *form factors* (`sxm4`, `pcie`, `nvlink`) and cooling tags (`dlc` — a real
>   token on this cluster's H200 nodes). Applied to **both** the shape branch and the
>   negative branch.
> - Case is still preserved verbatim (it has to be — see M6).
>
> **Tests:** `TestInfraTokensAreNotGpuModels` (7) — fabric generations, rack/form
> factors, "real model wins over an earlier infra label", fabric-only features →
> `gpu`, short real models still detected, unknown future shapes still detected, case
> preserved. The pre-existing `_detect_gpu_type` suite (including `rack1,t4` → `t4`
> and `rack5,A100` → `A100`) passes unchanged.

### M3 — Summary panel misreports memory when a custom `--mem`/`--mem-per-cpu` flag is used

**Where:** `job_summary_rows` at `builder.py:127-131` vs. the `_custom_mem`
suppression at `builder.py:341-356`.

The builder suppressed the auto memory directive when `custom_sbatch` contained a
`--mem=`/`--mem-per-cpu=` flag, but `job_summary_rows` ignored custom flags and still
printed the now-unused value — and it feeds *both* the CLI summary and the in-TUI
Review step, so both surfaces were wrong together.

```python
ans = {"job_name":"j","partition":"p","memory":"16G",
       "custom_sbatch":["--mem-per-cpu=2G"],"command":"x"}
dict(job_summary_rows(ans))["Memory"]   # -> '16G'   (shown to the user)
build_from_answers(ans)                 # -> #SBATCH --mem-per-cpu=2G   (no --mem at all)
```

> **Fix — one override helper, used by the script and the summary.**
>
> - **`_custom_mem_override(flags)`** (new) returns `(mem, mem_per_cpu)` as supplied
>   by custom flags, last occurrence winning (Slurm's own rule), and recognises the
>   space form too. **`_split_flag`** compares the exact option *name*, so
>   `--mem-bind=local` no longer counts as `--mem` (the old
>   `startswith("--mem=")` was accidentally right here; exact-name matching makes it
>   intentional, and a new test pins it).
> - `build_sbatch_script` uses it for the suppression decision; `job_summary_rows`
>   uses it to show the value the script really requests, so the panel and the script
>   cannot disagree. When both custom flags are present, both rows are shown — the
>   script does contain both, and hiding one would be the same class of lie.
>
> **Verified:** the M3 repro now prints `Mem per CPU: 2G` with no `Memory` row, and
> `--mem=32G` shows `Memory: 32G`.
>
> **Tests:** `TestSummaryMemoryMatchesScript` (4), including the space form and the
> `--mem-bind` non-match.

### M8 — A multi-node job's `--ntasks-per-node=1` is in the script and in neither summary

**Where:** `job_summary_rows` at `builder.py:699` vs. the emitter and its fallback at
`builder.py:975-997`.

M3's shape, on the row directly below the one whose comment states the rule. That comment
records the `Nodes` row being fixed for exactly this — reading the raw answer "left a
directive in the script that nothing in the summary accounted for, SM-15's shape in
miniature ... the summary is what the user checks the script by" — and the task-count row
kept the defect. The emitter gates on `ntasks_per_node is not None` **plus** an auto
fallback that emits `--ntasks-per-node=1` for any multi-node job with no task count; the
summary gated on truthiness, so the fallback had no row at all.

```python
ans = {"job_name": "j", "partition": "amd", "nodes": 4}
[l for l in build_from_answers(ans).splitlines() if "ntasks" in l]
# -> ['#SBATCH --ntasks-per-node=1']            (4 nodes x 1 task = 4 tasks)
[v for k, v in job_summary_rows(ans) if k == "Tasks per node"]
# -> []                                        (the panel says nothing about tasks)
```

**Reachability, measured — it cut two of the three candidate values.** A raw `""` renders
`#SBATCH --ntasks-per-node=` and a raw `0` renders `=0`, both with no row, but neither is
reachable through the CLI: `main.py:417-421` maps a blank to `None` and `main.py:482`
hard-errors on `<= 0`. Only the fallback is reachable, and it is the ordinary multi-node
case rather than a corner.

> **Fix — mirror the emitter's own condition, and say the value is slurmate's.**
>
> - The summary's `else` arm reproduces the emitter exactly: the same `int()` coercion the
>   builder applies defensively (a config file or the wizard can leave `nodes` a string),
>   and the same exception — `--custom-sbatch=--ntasks=N` suppresses the fallback, so there
>   is no directive to account for and no row.
> - The row reads `1 (automatic for a multi-node job)` rather than a bare `1`. Unlike
>   `--nodes=1`, which the `Nodes` comment calls "not an imposition ... Slurm's own default
>   too", this one is: one task per node over 4 nodes is 4 tasks, where Slurm's default is
>   one task for the whole job. A bare `1` would read as a value the user typed.
> - `job_summary_rows` feeds both the CLI panel and the in-TUI Review step, so one change
>   covers both surfaces — as in M3, they could only ever have been silent together.
>
> **Verified:** `--dry-run -J j -p amd --nodes 4` now shows
> `Tasks per node: 1 (automatic for a multi-node job)` beside the script's
> `#SBATCH --ntasks-per-node=1`; `--nodes 1` shows no such row; an answered count still
> renders verbatim (`2`, and a typed `1` as a bare `1`).
>
> **Tests:** `tests/test_summary_accounts_for_the_fallback.py` (16), built around the rule
> both ways — no unaccounted directive and no phantom row — over 11 answer shapes. Teeth:
> removing the arm reddens 5, all 3 controls green.

### M4 — The wizard has no step for `--constraint` or `--mem-per-cpu`, and silently drops both when they come from a config file

**Where:** `STEPS` in `tui.py:326-381`; both read from CLI/config in
`main.py::run_batch` (`main.py:206-221`).

`--constraint` (Slurm `-C`) and `--mem-per-cpu` were added in 0.5.0 specifically for
cluster-agnostic support (e.g. Perlmutter's mandatory `-C cpu`/`-C gpu`). They worked
in batch mode and from `.slurmate.toml`, but the interactive wizard exposed neither,
so an interactive user on such a site had to fall back to "Custom #SBATCH flags" —
which for `--constraint` then tripped **H1**, and for `-C gpu` (space form) tripped
**M5** — or hand-edit the script.

**It was worse than a missing step.** `Wizard.__init__` builds its config defaults by
iterating `STEPS` (`tui.py:399-410`), so a key with no step was never even read: the
same `.slurmate.toml` produced different jobs in batch and interactive mode.

```python
# .slurmate.toml: constraint = "gpu", mem_per_cpu = "2G", cpus = 8
Wizard()._config_defaults        # -> {'cpus': '8'}     ← constraint/mem_per_cpu absent
```

`job_summary_rows` already had a `Constraint` row (`builder.py:141`) that could
therefore never appear in the TUI review — a tell that the step was intended.

> **Fix — add both steps, in builder-directive order.**
>
> - **`mem_per_cpu`** immediately after `memory` (autocomplete, `MEM_PER_CPU_CHOICES`,
>   `validate_memory`, blank default) and **`constraint`** immediately after
>   `gpu_format` (autocomplete, `CONSTRAINT_CHOICES` = `cpu`/`gpu`/`bigmem`/… as
>   *suggestions* — the field is free text). That order matches where the builder
>   emits the directives, which is what keeps the live preview growing top-to-bottom
>   instead of reshuffling.
> - `_coerce` handles both: blank → `None` (so `mem_per_cpu` falls back to `--mem`),
>   and `mem_per_cpu` is `normalize_memory`d exactly as batch mode does it, so `2000`
>   becomes `2000M` rather than a bare number.
> - Adding the steps **fixes the config drop for free** — `_config_defaults` iterates
>   `STEPS`, so both keys are now picked up. Verified by
>   `test_config_keys_now_reach_the_wizard`.
> - Two index-based tests (`STEPS[5]`, `STEPS[6]`) that the insertion would have
>   silently repointed were converted to a `_step("key")` lookup, so a future
>   insertion cannot quietly change what they assert.
>
> **Tests:** `TestConstraintAndMemPerCpuSteps` (5) — directive-order positions, config
> keys reaching the wizard, answers flowing into the script (`--constraint=gpu` +
> `--mem-per-cpu=2G` and *no* `--mem`), blank falling back to `--mem`, and
> normalisation — plus the step-order and `mem_per_cpu`-validator tests.

### M6 — GPU-type validation lowercases, but Slurm node features are case-sensitive

**Where:** `validate_job_config` (`system_utils.py:408-413`) compared
`str(gpu_type).lower()` against a lowercased set, while `fetch_gpu_types_for_partition`
correctly preserved each feature's original case.

Measured on this cluster (partition `gpu`, whose nodes carry lowercase `a100`/`v100`):

```console
$ sbatch --test-only -p gpu --gres=gpu:1 -C a100 …   → Job … to start … on midway3-0294
$ sbatch --test-only -p gpu --gres=gpu:1 -C A100 …   → allocation failure: Requested node configuration is not available
$ sbatch --test-only -p gpu --gres=gpu:1 -C v100 …   → Job … to start …
$ sbatch --test-only -p gpu --gres=gpu:1 -C V100 …   → allocation failure: Invalid feature specification
```

Two consequences:

1. A user who types `a100` on a cluster whose feature is `A100` (both spellings exist
   here — partition `test` has `A100` and `a100` nodes) passed slurmate's validation
   and was then rejected by Slurm. The check that exists to catch exactly this class
   of typo was blind to it.
2. Any de-duplication of the picker list **must not** be case-folded — see
   [Withdrawn](#withdrawn--not-changed).

> **Fix — warn on a case-only mismatch, and keep every downstream comparison exact.**
>
> - `validate_job_config` now distinguishes "not in the list at all" (error,
>   unchanged) from "matched only case-insensitively" → **warning** naming the
>   advertised spelling: *"GPU type 'a100' differs in case from the partition's
>   'A100'; Slurm node features are case-sensitive"*. Warning rather than error
>   because the mismatch is only fatal for the constraint path, and the check itself
>   is an advisory that a heterogeneous partition can under-report.
> - The H1 constraint de-dup and the H2 feature-only comparison are both
>   **case-sensitive**, and `_detect_gpu_type`'s docstring now records *why* case must
>   be preserved end-to-end.
>
> **Tests:** `TestGpuTypeCaseSensitivity` (3) — mismatch warns but does not error,
> exact case is silent, an unknown model still errors (and does not also warn).

---

### M7 — The partition picker never shows a partition's GPU capacity, so on a count-only-GRES cluster a GPU partition is indistinguishable from a CPU-only one

**Where:** `_fmt_partition` (`tui.py:296-304`) built its GPU segment from
`gpu_types` alone. `fetch_partitions` (`system_utils.py:2416-2462`) records two
other GPU facts per partition — `gpus_per_node`, parsed by `_parse_gpu_count` and
`max()`-merged across the partition's `sinfo` rows, and the boolean `has_gpu` —
and both were read only by two capacity *checks* (`capacity_refusal`
`system_utils.py:1841`, `validate_job_config` `system_utils.py:2110/2222`), which
fire only once a request already exceeds the limit. Neither reached any surface.

This is H2's cluster shape on a different surface. A site that configures GPUs
count-only (`Gres=gpu:4`, no model — the majority spelling here) populates
`gpus_per_node=4` and `has_gpu=True` but leaves `gpu_types` empty, so the row
emitted no GPU marker at all. Measured on midway3 (88 partitions): **18 are
count-only** — `gpu`, `beagle3`, `kicp-gpu`, `ssd-gpu`, `lgagliardi-gpu`,
`gagalli-gpu`, `depablo-gpu` and 11 more — against 2 typed and 68 with no GPU.
Two figures were lost, both shown here with inputs differing *only* in the
missing dimension:

```python
{..., "gpu_types": [],       "has_gpu": True,  "gpus_per_node": 4}
{..., "gpu_types": [],       "has_gpu": False, "gpus_per_node": 0}
both -> 'p            8 nodes · 48 CPU · 180G'          # a 4-GPU partition and a CPU one

{..., "gpu_types": ["a100"], "has_gpu": True,  "gpus_per_node": 1}
{..., "gpu_types": ["a100"], "has_gpu": True,  "gpus_per_node": 8}
both -> 'p            8 nodes · 48 CPU · 180G · GPU:[a100]'
```

The partition step is the *first* step and runs before every GPU step, so nothing
later in the wizard recovers the choice: a user who needs GPUs has 88 rows and no
way to tell which 20 have any. Mock mode cannot show it — every `MOCK_PARTITIONS`
GPU partition carries typed GRES, which is why the picker's own tests passed.

> **Fix — render `gpus_per_node` beside the model list, and emit the segment on
> `has_gpu` rather than on `gpu_types`.**
>
> - `_fmt_partition` now reads `gpus_per_node` and `has_gpu`. The count is the
>   per-node figure the row already prints for CPUs and memory (`48 CPU · 180G`)
>   and was the only one of the three missing, so it goes in the same place:
>   `· GPU:4` (count-only), `· GPU:4 [a100,v100]` (both known), `· GPU:?` (known
>   to have GPUs, neither figure resolvable — the same admission the row already
>   makes for an absent `cpus_per_node`).
> - `· GPU:[a100]` is kept verbatim for a record with models but no resolvable
>   count, so a hand-built partition dict and a library caller see no change.
> - A CPU-only partition (`has_gpu is False`, no models) still gets no GPU segment
>   at all — the widened condition must not make all 68 of them claim GPUs.
> - Not touched: the row is TUI-only. The summary and the script report the
>   *requested* GPUs, not the partition's capacity, so there is no second surface
>   to keep in step here (`validate_job_config`'s "exceeds partition limit (4 per
>   node)" warning already renders the same field, and now agrees with the picker
>   instead of being the only place it appeared). No wizard `subtitle` was
>   lengthened.
>
> **Tests:** `tests/test_partition_gpu_capacity_label.py` (14) —
> `TestGpuCapacityReachesThePicker` (7) pins both lost figures, the `GPU:?`
> admission, and midway3's real `gpu|…|gpu:4|mixed` row plus a two-row `gpu:2` +
> `gpu:8` merge end-to-end through `fetch_partitions`; `TestPickerRowUnchangedElsewhere`
> (7) pins the node/CPU/memory/default segments, the legacy `GPU:[a100]` spelling,
> the CPU-only silence, and the label's round-trip back through
> `_set_partition_from_select` (the picker uses the label as its identity key).

---

## Low severity

### L1 — Stale `transient["gpu_types"]` after a partition change suppresses the live "GPU type not in partition list" error

**Where:** `tui.py:801` passed `extra_gpu_types=self.transient.get("gpu_types")`; the
slot was set only on entering the gpu_type step (`tui.py:1048`) and never cleared or
re-keyed when the partition changed — unlike the partition-keyed QoS cache at
`tui.py:895-896`, which shows the intended pattern.

Pick `gpu-shared` (a100,v100) → gpus=2 → gpu_type=`a100` (caches `["a100","v100"]`);
go Back and switch to `gpu-highend` (h100 only). On every intermediate step
`_config_warnings()` returned `[]` while a fresh `validate_job_config` returned
`('error', "GPU type 'a100' not in partition list (h100)")` — reproduced for the
account, cpus, memory, nodes and gpus steps. Because `extra_gpu_types` only *widens*
the accepted set, a stale value can only suppress a real error, never invent one; it
self-healed on re-entering gpu_type and the submit guard did its own fresh fetch, so
the emitted script was never wrong — only the live warning went missing.

> **Fix — key the cache on the partition it was fetched for.**
> `_setup_gpu_type` stores `transient["gpu_types_part"]`, and a new
> **`_cached_gpu_types(slot)`** helper returns the cached list *only* when that key
> still matches `answers["partition"]` — otherwise `None`, so validation falls back
> to the partition object's own list and the real error surfaces immediately. Used for
> both `extra_gpu_types` and H2's `feature_only_gpu_types`, so neither can go stale.
>
> **Tests:** `TestGpuTypeCacheIsPartitionKeyed` (2) — the error now appears on every
> intermediate step after a partition switch, and the cache is still used (no
> needless re-fetch, no false error) while the partition matches.

### L2 — A leading-whitespace tilde in `output_dir`/`output_file` is not expanded

**Where:** `builder.py:165-170` — `os.path.expanduser()` ran *before* the later
`.strip()` (`builder.py:177`, `226`), so `" ~/logs"` never started with `~` at expand
time.

```python
build_from_answers({... "output_dir":" ~/logs"})   # -> #SBATCH --output=~/logs/j-%j.out
build_from_answers({... "output_dir":"~/logs"})    # -> …=/home/youzhi/logs/j-%j.out  (fine)
```

Slurm does not expand `~`, so the job wrote into a literal `./~/logs/` — while
`submit_sbatch` pre-created the *expanded* `$HOME/logs` (it calls `expanduser`
itself), so the directory the job needed was the one that didn't get created.

> **Fix:** `strip()` before `expanduser()` for both `output_dir` and `output_file`
> (and `str()` first, for a stringy config value), with a comment explaining the
> ordering trap. **Verified:** `" ~/logs"` and `"~/logs"` now both emit
> `/home/youzhi/logs/…`. **Tests:** `TestTildeWithLeadingWhitespace` (2).

### L3 — `submit_sbatch` creates log directories before checking whether `sbatch` exists

**Where:** `system_utils.py:896-911` — the `os.makedirs` loop ran *before* the
`is_tool_available("sbatch")` guard.

```python
# SLURMATE_MOCK=1, empty directory
submit_sbatch("#!/bin/bash\n#SBATCH --output=logs/j-%j.out\n#SBATCH --error=deep/nest/j.err\ntrue\n")
# -> (0, '', 'sbatch not available (mock mode) — no job submitted')
os.listdir(".")   # -> ['deep', 'logs']      ← created anyway
```

> **Fix:** the availability check moved above the `makedirs` loop, so nothing touches
> the filesystem when nothing is going to run. **Tests:** the existing
> `test_submit_creates_log_directories` now stubs `is_tool_available`/`subprocess.run`
> (keeping its real assertion that dirs *are* created for a real submit), and a new
> `test_mock_mode_creates_nothing` pins the fix.

### L4 — `_sbatch_log_path` doesn't handle the space-form long option `--output PATH`

**Where:** `system_utils.py:936-952`. It handled `--output=PATH` and `-o PATH`, but not
`--output PATH`, which sbatch accepts — so the directory went un-created and Slurm
could fail the job on a missing dir.

> **Fix:** rewritten around one option regex (`_SBATCH_OPT_RE`) that accepts `=` *or*
> whitespace as the separator, plus an optional **`kind`** parameter
> (`"output"`/`"error"`/either) that M1's `effective_log_path` needs to resolve the
> two streams separately. Valueless directives and non-log options still return `""`.
> **Tests:** `TestSbatchLogPathForms` (5) and
> `test_submit_creates_log_dirs_for_space_and_short_forms`.

### L5 — The action-menu editor label diverges from the editor actually launched

**Where:** `main.py:752` used a raw `os.environ.get("EDITOR", os.environ.get("VISUAL",
"vim"))` for the label while the invocation used `_editor_command()`
(`main.py:461-475`).

| env | menu label | actually ran |
|---|---|---|
| `EDITOR=""` | `Open script in ` (blank) | `['vim']` |
| `EDITOR="code --wait"` | `Open script in code --wait` | `['code','--wait']` |

> **Fix:** the label is now `" ".join(_editor_command())` — one source of truth, so an
> empty-but-set `EDITOR` shows `vim` (what actually runs). **Tests:**
> `TestEditorLabelMatchesLaunch` (2).

### L6 — The TUI Review "Job Configuration" column misaligns long labels

**Where:** `tui.py:1379-1394` — fixed `label_w = 12`, while the CLI summary computed
the width from the actual labels (`main.py:433`). Three labels exceed 12 characters
(`Tasks per node`, `Output directory`, `Array specification`), so `f"{label:<12}"`
neither padded nor truncated and the value column broke; a multi-line value's
continuation lines indented to a column that matched nothing.

> **Fix:** `label_w = max(len(label) for …)` over the rows actually being rendered
> (empty rows filtered once, up front), matching the CLI. **Tests:**
> `TestReviewColumnAlignment` asserts every value starts in the *same* column and that
> a continuation line lands on that column.

### L7 — `validate_memory` accepts a `P` unit that `sbatch` rejects

**Where:** `system_utils.py:126` (unit class included `P`).

```console
$ python -c "from slurmate.system_utils import validate_memory as v; print(v('16P'))"
True
$ sbatch --test-only --mem=16P --time=5 --wrap=true
sbatch: error: Invalid --mem specification
```

`sbatch --mem` documents only `[K|M|G|T]` and rejects `P` client-side, so slurmate let
through a value that could never submit.

> **Fix:** the `validate_memory` unit class is now `[KMGT]`, with a comment citing the
> measured sbatch error. `_parse_mem_to_mb` keeps `P` deliberately — it parses `sinfo`
> output, a different job. **Verified:** `validate_memory("16P") is False`,
> `("16T") is True`. **Tests:** `TestMemoryUnitStrictness` (2), and the pre-existing
> unit/fraction tests still pass.
>
> The companion `validate_time("1-99")` claim was **withdrawn** — see below.

### L8 — `theme.C` ignores `FORCE_COLOR` while `rich` honours it

**Where:** `theme.py:16-22` (`_should_use_color` checked only `NO_COLOR`, `TERM=dumb`,
`isatty()`), so piped output with `FORCE_COLOR=1` was half-coloured:

```console
$ FORCE_COLOR=1 python -c "...Console().print('[red]rich[/]'); print(repr(c.RED))" | cat -v
^[[31mrich^[[0m          ← rich: coloured
''                       ← theme.c: empty
```

> Correction to an earlier draft: it also named `CLICOLOR_FORCE`. Measured, `rich`
> does **not** honour `CLICOLOR_FORCE`, so honouring it in `theme` would *create* the
> mismatch instead of removing it.

> **Fix:** `_should_use_color` returns `True` when `FORCE_COLOR` is present (any
> value, matching rich's own test), after the `NO_COLOR` and `TERM=dumb` gates so
> those still win. `CLICOLOR_FORCE` is explicitly *not* honoured, with the reason in
> the docstring.
>
> **Consequence caught while fixing it:** colour was previously a proxy for "is a
> TTY", which the banner animation relied on — with `FORCE_COLOR` that proxy breaks,
> and `SLURMATE_BANNER_ANIMATE=1` would have started emitting cursor save/restore
> escapes into a pipe. `print_banner` now checks `sys.stdout.isatty()` for the
> animation explicitly. **Tests:** `TestForceColor` (5), including `NO_COLOR` and
> `TERM=dumb` precedence, `CLICOLOR_FORCE` being ignored, and "animation still
> requires a real TTY" (asserts no `\033[s`/`\033[u` in piped output).

### L9 — Queue-ETA is cached per-partition and ignores a later `nodes` change

**Where:** `tui.py:814-823` refetched only when `cached_part != part`. The first fetch
happens on the step right after partition — before `nodes` is entered — so it always
used `req_nodes=1`, and changing `nodes` later never refreshed it. Instrumenting
`fetch_queue_eta` across a walk that sets `nodes=8` recorded exactly one call:
`[('cpu-shared', 1)]`. `req_nodes` is the one input that makes the estimate
partition-specific rather than generic (the ETA is computed from whether enough idle
nodes exist for it).

> **Fix:** the cache key is now the tuple `(partition, nodes)`
> (`transient["queue_info_key"]`), so a node change refetches and nothing else does.
> **Tests:** `TestQueueEtaTracksNodes` (2) — a node change produces exactly
> `[('cpu-shared', 1), ('cpu-shared', 8)]`, and an unchanged walk still fetches once.

### L10 — The generated `mamba activate` fails on modern mamba (this cluster included), silently leaving the job in the wrong environment

**Where:** `builder.py:488-499`, which emitted:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
mamba activate <env>
```

`conda.sh` defines the `conda` shell function only. mamba ≥ 2 needs its own hook, so
this **failed on this very cluster** (miniforge 25.3.0) — run exactly as slurmd would,
in a clean non-login shell:

```console
$ env -i HOME=$HOME PATH=…/condabin:/usr/bin bash script.sh
critical libmamba Shell not initialized
'mamba' is running as a subprocess and can't modify the parent shell.
exit=1
```

The failure is **non-fatal to the script**, which is the real damage: the job keeps
running in whatever interpreter it inherited and produces plausible-looking wrong
results. Measured alternatives in the same clean shell:

| emitted activation | exit | resulting `sys.prefix` |
|---|---|---|
| `conda.sh` + `mamba activate` | 1 | *(never activated)* |
| `conda.sh` + `mamba.sh` + `mamba activate` | 0 | `…/envs/AI` ✓ (mamba.sh prints WARNINGs) |
| `conda.sh` + `conda activate` | 0 | `…/envs/AI` ✓ |
| `conda.sh` + `mamba activate … \|\| conda activate` | 0 | `…/envs/AI` ✓ |

So this was not the "environment-dependent, worth a note" item an earlier audit
recorded — it is the default outcome on a mainstream miniforge/mamba-2.x install.

> **Fix:** for `env_type=mamba` the builder now emits
>
> ```bash
> source "$(conda info --base)/etc/profile.d/conda.sh"
> # mamba >= 2 needs its own shell hook; conda activates the same env
> mamba activate myenv >/dev/null 2>&1 || conda activate myenv
> ```
>
> Chosen over sourcing `mamba.sh` (doesn't exist on every install, and prints warning
> noise into the job log where it does) and over silently substituting
> `conda activate` (discards the user's stated intent): mamba stays primary where it
> is properly initialised, and a mamba-created env is a conda env, so the fallback is
> exact rather than approximate. The `conda` branch is untouched.
>
> **Verified** by generating a real script and running it as slurmd would:
>
> ```console
> $ slurmate --env AI --env-type mamba --command 'python -c "…print(sys.prefix)"' --print > s.sh
> $ env -i HOME=$HOME PATH=…/condabin:/usr/bin bash s.sh
>   interpreter: /software/python-miniforge-25.3.0-el8-x86_64/envs/AI      ← activated
> ```
>
> **Tests:** `TestMambaActivationPortable` (2). **Known residual limitation
> (documented, not fixed):** **micromamba** has no `conda` at all, so
> `conda info --base` fails and the `source` is a no-op — unreachable without a much
> larger shell-hook probe, and a micromamba user is better served by `env_type=none`
> plus a custom command. Also deliberately *not* added: a hard `|| exit 1` after
> activation. It would arguably be an improvement (a failed `conda activate` for a
> typo'd env still leaves the job running in `base`), but that is a different,
> unreported issue and it would change the semantics of every existing generated
> script.

### L11 — Two bare `err_console.print` calls emitted trailing whitespace and lost their indent when they wrapped

**Where:** `main.py:368` — `err_console.print(f"    [dim]{escape(line)}[/]")`, the
continuation lines of a multi-line site-check issue — and `main.py:370`,
`err_console.print(_FORCE_HINT)`. Both carry their indent as **literal spaces
inside the string**, which is the arrangement `_print_indented`
(`main.py:119-155`) exists to replace, and neither goes through it.

Two consequences, both the ones `_print_indented` was written for:

* `rich` word-wraps the paragraph and keeps the space it broke on, so every
  wrapped line except the last ends in a trailing space.
* the literal prefix indents the **first** line only, so continuation lines come
  back at column 0 — visibly out of line with the `Error:` line above them.

Reproduced through the real report (`_check_cluster_targets` with an unknown
partition and a ten-partition cluster stubbed), **with the `expand=False` fix for
`_print_indented` in place**:

```text
width=60, 3 lines with a trailing space, none of them the helper's:
  0: "  ✗ Error: no partition 'nosuchpartition' on this cluster."   <- _print_indented, clean
  1: "    This cluster's partitions: aaz, aettinger-gpu, "          <- main.py:368
  2: 'ai4s-hackathon, amd, amd-hm, andrewferguson, '                <- main.py:368, indent lost
  3: 'andrewferguson-gpu, astroplasmas, ... (+2 more)'
  4: '  Pass --force to generate the script anyway (e.g. for '      <- main.py:370
  5: 'another cluster).'

width=80 / 90 / 120: 1 such line, always index 1 (main.py:368).
```

Run with `--force`, which suppresses the hint, exactly the two `main.py:368`
lines remain — that is how the two sites were told apart.

Cosmetic, and low for the same reason the fixed sibling was: nothing is
misdrawn. The cost is the one that entry names — a line that copies with junk on
the end, and `grep -nP ' $'` over a captured log flagging it — plus a
continuation line that does not line up with the message it continues.

> **Fixed.** Both calls go through `_print_indented`, at the depth each line
> means: `indent=4` for the continuation lines of an issue (they sit *under* the
> `Error:` line, not level with it) and `indent=2` for the hint. `_FORCE_HINT`
> carried its own `"  [dim]...[/]"` prefix and is printed from three sites
> (now `main.py:404`, `915`, `942`), so the prefix came **out of the constant**
> rather than being indented twice — all three hand it to the helper.
>
> **It needed one change inside the helper as well**, which is why this entry was
> worth its own round. `expand=False` sizes the `Padding` *block*, but `Padding`
> renders its child with `pad=True`, so a paragraph long enough to wrap is
> measured at the full available width and every line of it is filled to the wrap
> column. Routing these two sites through the helper on its own therefore traded
> the lost indent for **more** trailing whitespace, not less — 5 padded lines at
> 60 columns where the literal prefixes had left 3:
>
> ```text
> width=60, routed through the helper as it was:
>   1: "    This cluster's partitions: aaz, aettinger-gpu,          "   <- indent kept, fill gained
>   2: '    ai4s-hackathon, amd, amd-hm, andrewferguson,            '
> ```
>
> So `_print_indented` now wraps first — `rich` still decides where — and hands
> each resulting line to the single-line path, where the block is measured to
> that line's own length and there is nothing left to fill; each line is then
> rstripped, which is what removes the separator space at a wrap point that lands
> inside the width. That closes the "a wrapping paragraph is still padded to the
> wrap column" limit the sibling entry had recorded as acceptable, on the grounds
> that no line the tool emits through the helper wraps — which this site made
> untrue.
>
> **Verified** on the same two surfaces the reproduction used. Real `--dry-run`
> against an unknown partition on this 85-partition cluster: **0** lines with a
> trailing space and **0** at column 0, at 60, 70, 80, 90 and 120 columns (before:
> 3 and 3 at 60/70, 1 and 1 at 80/90/120). The stubbed ten-partition report reads:
>
> ```text
> width=60, after:
>   0: "  ✗ Error: no partition 'nosuchpartition' on this cluster."
>   1: "    This cluster's partitions: aaz, aettinger-gpu,"
>   2: '    ai4s-hackathon, amd, amd-hm, andrewferguson,'
>   3: '    andrewferguson-gpu, astroplasmas, ... (+2 more)'
>   4: '  Pass --force to generate the script anyway (e.g. for'
>   5: '  another cluster).'
> ```
>
> **Tests:** `TestTheSiteCheckReportIsIndentedThroughout` (13 cases) in
> `tests/test_indented_lines_have_no_trailing_fill.py`, which replaces the
> `TestTheSiblingSiteIsStillOpen` class that pinned the open counts, plus
> `TestTheReportItselfIsUnchanged` (10 controls: the text survives the wrap, the
> report still fits the width, the head line and the `--force` downgrade are
> untouched, and the check still exits 1) — all ten green with each of the three
> edits reverted. `test_a_paragraph_that_wraps_still_pads_each_line` became
> `test_a_paragraph_that_wraps_carries_no_fill_on_any_line`.
>
> **~~Deliberately not changed~~ — the deferral was measured and is WRONG; the
> sweep is now finished.** This entry left three groups alone (the `--gpus` format
> hint at `main.py:905-909`, the `_coerce_*` validation lines at `515-557`, the
> TUI's edit-rejection lines at `2451-2480`) on the grounds that "none of them
> interpolates an unbounded list the way this message does, so none has been
> observed to wrap". Each group was then **driven through its own code path** at
> six widths a real terminal is (60, 70, 80, 90, 100, 120) — the rejections from a
> bad flag value through `_run_batch`, the hint from `_check_gpu_format` with
> `select/linear` stubbed, the menu lines from `_main`'s real action loop with the
> scheduler verdict and the human's choices scripted — and **all three wrap**:
>
> | site | text | wraps at (columns) |
> |---|---|---|
> | `905` | inferred `--gpus` format hint | **60 70 80 90 100 120** (147 cells — no width fits it) |
> | `515` / `518` / `521` | `--cpus` / `--nodes` / `--gpus` rejection | — at a 1-digit value (52–56 cells); the number is the user's |
> | `524` | `--ntasks-per-node` rejection | **60** (63 cells) |
> | `530` / `541` / `556` | `Invalid <X> value: <user string>` | **any** — a 200-char `--mem` made 4 lines at 80 |
> | `531` / `544` | `Give {MEMORY_FORMS}.` | **60 70 80 90 100 120** (195 cells of FIXED text) |
> | `557` | `Give {time_forms()}.` | **60 70 80 90** (99 cells) |
> | `2451` | `Slurm rejects the edited script: <sbatch>` | **60 70 80 90 100 120** |
> | `2453` | `Choose "Open in editor" …` | **60 70** (80 cells) |
> | `2470` | `Slurm would not take this job right now: <sbatch>` | **60 80 120** |
> | `2478` | each `_hard_errors` message | **60 80** (85 cells) |
> | `2479` | `This job has errors Slurm will reject. …` | **60 70 80** (90 cells) |
>
> The `Give …` lines are the sharpest counter-example to the deferral's reasoning:
> `MEMORY_FORMS` is fixed text with nothing interpolated at all, and it wraps at
> **every** width. At 60 columns a real `--mem not-a-memory-value` printed 3 lines
> with a trailing space and 3 at column 0.
>
> **Fixed.** All sixteen sites go through `_print_indented` at the two-space depth
> they carried as literal text — the seven `Error:` lines via `_print_issue`, which
> composes the same sentence as markup and hands it on, so the text is byte-for-byte
> what it was. **0 trailing-whitespace lines and 0 column-0 lines at all six widths**
> on all three surfaces (before: 3/3 at 60 for the `--mem` report, 1 col-0 line at
> every width for the `--gpus` hint, 2 for the menu).
>
> **The same measurement turned up a HIGH-severity defect these sites had been
> hiding, fixed with them** — see [H4](#h4--seven-error-lines-printed-their-own-ansi-escape-as-visible-text).
>
> **Tests:** `tests/test_literal_indent_sweep.py`, 84 cases —
> `TestTheBatchRejectionsAreIndentedThroughout` (24),
> `TestTheColourTerminalGetsColourNotEscapeBytes` (2),
> `TestTheInferredGpuHintIsIndentedThroughout` (7),
> `TestTheSubmitMenuRejectionsAreIndentedThroughout` (13) and `TestControls` (36
> controls, every one of them green with the change reverted). Neutered at all 16
> sites — a revert verified code-identical to the pre-round `main.py` — **42 of the
> 84 redden**. The 42 that do not are the cases the measurement says cannot wrap
> (`--cpus`/`--nodes`/`--gpus` at a short value, `Give {time_forms()}.` at 100/120)
> plus the controls, which is the point of measuring rather than asserting.
>
> **Still open, and now recorded rather than assumed absent:** an AST scan of every
> `console.print`/`err_console.print` in `main.py` whose first string literal begins
> with spaces found **34** such sites, not the 16 these three groups contain. The
> other **18** are the same shape and all of them interpolate an unbounded value or
> carry >60 cells of fixed text — `main.py:270`, `314`, `468`, `505`, `723`, `728`,
> `749`, `753`, `764`, `767`, `1123`, `1338`, `1350`, `1370`, `1397`, `1403`,
> `1587`, `2491`. Ten of them (`270`, `314`, `468`, `505`, `723`, `728`, `749`,
> `753`, `764`, `767`) also carry the H4 ANSI defect. Left alone this round to keep
> the change to the three groups that were deferred; they are the next sweep, and
> the scan that finds them is three lines of `ast.walk`.

### H4 — Seven `Error:` lines printed their own ANSI escape as visible text

**Where:** the `_coerce_*` rejections at `main.py:515`, `518`, `521`, `524`, `530`,
`541`, `556` (the group L11 deferred), each of the form
`err_console.print(f"  {c.RED}✗ Error: …{c.RESET}")`.

`theme.c.RED` is a **raw ANSI escape** (`[38;2;255;0;0m`), and `rich` does not
read ANSI in a `print` argument. `[38;2;255;0;0m` is not a markup tag (a tag must
start with a letter, `#`, `/` or `@`), so rich left it as literal text, its repr
highlighter styled the digits *inside* it, and what reached the terminal was a
lone `ESC` followed by rich's own styling of the characters `[38;2;255;0;0m`.

Measured with `FORCE_COLOR=1` at 80 columns, interpreting the captured bytes the
way a terminal would (consume a complete CSI, drop a lone `ESC`):

```text
before:  '  [38;2;255;0;0m✗ Error: Invalid memory value: not-a-memory-value[0m'
         68 visible cells for a 51-cell message — and NOT red
after:   '  ✗ Error: Invalid memory value: not-a-memory-value'
         51 cells, SGR codes on the line: ['\x1b[31m', '\x1b[0m']
```

Invisible to every existing test because they all run with `NO_COLOR=1` or a pipe,
where `theme.C.__getattribute__` returns `""` and the line is plain. It is the
interactive case — the only one a human sees — that was broken.

> **Fixed** by the same edit as L11: routing these lines through `_print_issue`
> replaces the raw escape with rich markup, so nothing un-parsed reaches rich. The
> colour changes from truecolor `#ff0000` to rich's `red`, which is what every
> other `Error:` line in this file already renders (they were already going through
> `_print_issue`), so this also removes a two-reds-for-one-severity split.
>
> **Tests:** `TestTheColourTerminalGetsColourNotEscapeBytes` — one that no escape
> sequence survives into the visible text, one that the line actually carries
> `ESC[31m`. The second was deliberately strengthened after the first draft passed
> in both states: "the line carries some SGR" was true before too, because the
> mangling was itself rich's highlighter colouring the digits.
>
> **Not fixed:** the other ten `c.*`-into-`console.print` sites listed at the end of
> L11 have the identical defect. Same severity, out of scope for a round that was
> asked for three groups.

---

## Documentation issues

### D1 — README "Recognized keys" omits `constraint` and `mem_per_cpu`

`README.md:176-179` listed the recognized config keys but omitted **`constraint`** and
**`mem_per_cpu`**, both of which `run_batch` reads (`main.py:209`, `main.py:220`), so a
user could not discover that they are configurable. Compounds **M4**, where they were
also unreachable interactively.

> **Fix:** both added to the key list, plus a line noting that *every* recognized key
> is now also a wizard step — so a config file prefills the interactive flow and batch
> mode identically (which is only true because of the M4 fix). The config example also
> gained commented `constraint` and `mem_per_cpu` entries.

> **Re-measured 2026-09-05 — the prose half is closed; the pin was the live half.**
> The list at `README.md:221-224` names exactly the 22 keys in
> `system_utils.CONFIG_KEYS`, symmetric difference empty in both directions, and a
> probe through `_normalize_config_keys` keeps all 22 and drops anything else — so the
> reported omission is gone, and the paragraph is byte-identical to `HEAD` rather than
> pending in the working tree. What was still missing was anything holding it there:
> `tests/test_config_keys_are_wizard_steps.py` checked only the *second* sentence of
> the claim (every key is also a wizard step) and never read the list, even though the
> neighbouring "CLI spellings work too" paragraph is pinned against `CONFIG_ALIASES`.
> A hand-written list stays right only while something reads it. **Test:**
> `TestTheReadmeListsExactlyTheRecognizedKeys` (3, same file) extracts the paragraph
> from `README.md` and asserts both directions against `CONFIG_KEYS`, behind a vacuity
> guard on the extraction (>=20 names, no duplicates) so a regex that stopped matching
> cannot pass silently. No source change and no prose change.

### D2 — README lists only 3 of the 5 GPU formats

`gpus_per_node` and `gpus_per_task` (added in 0.5.0, valid `--gpu-format` argparse
choices, handled in `builder.py:390-395`) were missing from three places that still
said "gres_type | constraint | gpus": the env-var table (`README.md:197`), the config
example comment (`README.md:170`), and the "Cluster-agnostic GPU syntax" feature row
(`README.md:144`).

> **Fix:** all five listed in all three places. The feature row also now mentions that
> slurmate flags a GPU model a site only exposes as a node feature (the H2 behaviour),
> since that is the part a user needs to know when choosing a format.

### D3 — README overstates `SLURMATE_BANNER_ANIMATE`

`README.md:201` said it "Force[s] the animated banner even when not a TTY." It didn't:
`print_banner` gated the animation behind `use_color` (`theme.py:137`), and `use_color`
required a TTY. The behaviour was right (cursor-control animation into a pipe would
garble); the documentation was wrong.

> **Fix:** the row now reads "Animate the startup banner (needs a real TTY; ignored
> when output is piped)" — which is also the behaviour L8's explicit `isatty()` check
> now guarantees rather than merely implies. `FORCE_COLOR` was documented alongside the
> existing `NO_COLOR` note.

---

## Minor / by-design notes

- **`python -m slurmate` didn't work** — no `src/slurmate/__main__.py`, so only the
  console script worked.
  > **Fix:** added `src/slurmate/__main__.py` (delegates to `main.main`, with a
  > docstring saying why it exists). Verified: `python -m slurmate --version` →
  > `slurmate 0.5.2`. **Test:** `TestModuleEntryPoint` runs it as a subprocess.
  > README's quick-start now shows it.
- **Un-committed GPU-type text dropped on Back** (`tui.py:698-719`) — in the gpu_type
  *text* sub-mode (used when the partition lists no typed GPUs), Esc/Shift-Tab
  discarded what was typed, because `_go_back`'s save block covered every step kind
  *except* `gpu_type`. Verified: type `h100`, press Back → `answers["gpu_type"] is
  None`, while a select step keeps its value (`env_type` → `'Conda'`).
  > **Fix:** `_go_back` reads the sub-mode *before* clearing it and persists a
  > non-empty typed value. Only the text sub-mode was changed: doing the same for the
  > select sub-mode would let a radio reset to "Any" overwrite a previously typed
  > model with `None`. **Tests:** `TestGpuTypeTextPersistsOnBack` (2) — the typed value
  > survives, and a blank does not clobber a prior answer.
- **`estimate_su` is labeled "Estimated CPU-hours"** even for GPU jobs
  (`main.py:427`) — honest (it *is* CPU core-hours) but it omitted the number that
  actually drives billing on a GPU site.
  > **Fix:** new **`builder.estimate_gpu_hours()`** and an "Estimated GPU-hours" row
  > shown for GPU jobs. The multiplier follows the chosen `gpu_format`, because the
  > allocation semantics differ: per-node for `--gres`/`--gpus-per-node`/constraint,
  > per-task for `--gpus-per-task`, job-wide for `--gpus`. `estimate_su` keeps its name
  > and output (back-compat); the shared number formatting moved to `_fmt_estimate`.
  > **Tests:** `TestGpuHours` (4) + `TestGpuHoursRow` (the row appears only for GPU
  > jobs).

---

## Found while verifying the fixes (P1–P6)

Three further passes over the *new* code, each asking a different question, turned up
five more defects (and one non-defect worth recording):

1. **"What did these changes make reachable that wasn't before?"** → P1, P2, P3.
2. **A property sweep** over 23,040 generated scripts, asserting no `#SBATCH` option
   is ever emitted twice and every directive is well-formed — the defect class behind
   H1, M1 and M3 → P5.
3. **A round trip through the real controller**: 19 generated scripts, each fed to
   `sbatch --test-only` → P4, and P6 (which turned out not to be slurmate's bug).

All are fixed, with tests. The property sweep and the wizard navigation fuzz are now
part of the suite, so this class of regression is caught automatically.

### P1 — A space-form custom value containing a space was emitted unquoted (medium)

Exposed by the **M5** fix. `--comment "my job"` reaches the builder as
`--comment my job` (the parser consumes the user's quotes). `_quote_custom_flag` only
quoted the `=` form, so:

```console
$ # after the M5 fix, before P1:
'--comment "my job"'   -> ['--comment my job'] -> #SBATCH --comment my job
```

Slurm's directive parser splits that into `--comment=my` plus a stray `job` — the
exact defect the `=` form was hardened against in 0.5.1. Before M5 this input was
mangled a *different* way (`#SBATCH --my job`), so it was broken either way; the M5
fix made the option name right and left the quoting wrong, which is arguably worse
because the directive now looks plausible.

> **Fix:** `_quote_custom_flag` handles the space form too, using
> `_VALUE_TAKING_FLAGS` to know where the value starts — the same vocabulary M5
> introduced, which is what makes this solvable at all. An unknown option name with
> spaces is still left untouched (guessing wrong would corrupt it), and an
> already-quoted value is not double-quoted. Verified:
> `#SBATCH --comment "my job"`. **Tests:** `TestSpaceFormValueQuoting` (6).

### P2 — `--mem-per-cpu` was never checked against the node's memory (medium)

`validate_job_config` checked `answers["memory"]` against `mem_per_node_mb` but had no
notion of `--mem-per-cpu`, which is *per core* — so the per-node request went
unchecked:

```python
# cpu-shared: 32 cores, 131072 MB/node
validate_job_config({..., "cpus": 8, "memory": "512G"})       # -> warning ✓
validate_job_config({..., "cpus": 8, "mem_per_cpu": "64G"})   # -> []  ← 512G/node, silent
```

Pre-existing, but the **M4** fix made `--mem-per-cpu` a wizard step, so it went from
"reachable only via CLI/config" to "one of the questions every interactive user is
asked" — which is what promoted it from a footnote to a fix.

> **Fix:** the memory check now multiplies `--mem-per-cpu` by the cores requested on
> the node (`cpus-per-task × tasks-per-node`, computed once and shared with the CPU
> check) and warns with the arithmetic shown: *"Memory (64G/CPU × 8 cores = 524288 MB)
> exceeds partition limit (131072 MB per node)"*. **Tests:**
> `TestEffectiveMemoryValidation` (3 of 8 cover this, including the tasks-per-node
> multiplier).

### P3 — Validation warned about a memory value the script doesn't request (low)

The same root cause as **M3**, one layer over: the builder gives `--mem-per-cpu`
precedence over `--mem` and lets a custom `--mem`/`--mem-per-cpu` flag suppress both,
but validation always read the raw `memory` answer. So a superseded value produced a
phantom warning, while the value actually being requested was never checked:

```python
validate_job_config({..., "memory": "512G", "mem_per_cpu": "2G"})
# -> warning about 512G, which the script does not emit at all
validate_job_config({..., "memory": "16G", "custom_sbatch": ["--mem=512G"]})
# -> [] , although the script requests 512G
```

> **Fix:** validation resolves the **effective** memory first — reusing M3's
> `_custom_mem_override` / `_normalize_custom_flags` via a function-level import (the
> module-level direction is `builder → system_utils`, so a lazy import keeps the one
> source of truth without a cycle) — and checks that. A superseded `--mem` is now
> silent; a custom `--mem` flag is what gets checked. **Tests:** the remaining 5 in
> `TestEffectiveMemoryValidation`.


### P4 — Whitespace inside a `--constraint` value produced a job Slurm rejects (medium)

Found by the round-trip pass. Slurm's feature grammar has no room for spaces, and it is
strict about it:

```console
$ sbatch --test-only -p gpu --gres=gpu:1 -C "a100 & 384g" …
allocation failure: Invalid feature specification
$ sbatch --test-only -p gpu --gres=gpu:1 -C "a100&384g" …
sbatch: Job 52618282 to start at 2026-07-25T03:23:55 … on nodes midway3-0294
```

slurmate passed the value straight through, so a user who typed the spaced form — the
natural way to write a boolean expression — got a job the controller refuses. A stray
leading space was worse: `constraint=" a100 "` emitted `#SBATCH --constraint= a100`,
broken twice over. The **M4** fix made this reachable from the wizard, where a spaced
expression is even more likely than on a command line.

> **Fix:** `_clean_constraint` strips all whitespace from every constraint source (the
> `constraint` answer *and* each merged custom `-C` value) before the H1 merge. Feature
> names cannot contain whitespace, so nothing legitimate is lost, and the
> `a100 & 384g` → `a100&384g` normalization is exactly what Slurm wants.
> **Verified:** the spaced form now round-trips through `sbatch --test-only` as
> "ok". **Tests:** `TestConstraintWhitespace` (5).

### P5 — A custom `--gres` override left a duplicate directive, and the summary described the wrong GPU request (medium)

Found by the property sweep: of 23,040 generated scripts, 576 contained a repeated
`#SBATCH` option — all of them `--gres`, from the deliberate "keep a differing user
override" behaviour:

```
#SBATCH --gres=gpu:v100:2      ← auto (dead: Slurm honours the last)
#SBATCH --gres=gpu:a100:8      ← the user's override
```

The *job* was right (last-wins gives the user their override — which is why the
override is kept), but two things were not: the script contradicted itself, and
`job_summary_rows` still reported `GPUs: 2 × v100` for a job requesting 8×a100 — the
same lie **M3** fixed for memory. The same held for `--gpus`, `--gpus-per-node` and
`--gpus-per-task`.

> **Fix:** a custom flag on the option the chosen format would emit, **with a different
> value**, now suppresses the auto directive (mirroring `--mem` and `--output`), and the
> summary reports `GPUs: --gres=gpu:a100:8 (custom flag)` instead of inventing a count.
> Three deliberate boundaries:
> - An **exact** duplicate is handled the other way round — auto kept, custom dropped —
>   so the script keeps slurmate's canonical `=` spelling rather than the user's
>   incidental one.
> - Only the **same option name** suppresses: `--gres=gpu:2` and `--gpus=2` are
>   different requests to Slurm, so a custom `--gpus` must not remove an auto `--gres`.
> - Under the `constraint` format the GPU **type** is still recorded as a node feature,
>   because it is a separate requirement from the GRES count.
>
> The `emitted_*` values are left unset when suppressing, so the override can never be
> mistaken for a duplicate and dropped by the emit loop — which would have left the
> script with no GPU directive at all.
>
> **Tests:** `TestCustomGpuFlagReplacesAuto` (7) plus
> `TestNoDuplicateOrMalformedDirectives`, which keeps the sweep in the suite (>2,000
> combinations asserting no duplicate option and no malformed directive). The
> pre-existing test that asserted the duplicate was updated with the reason.

### P6 — An unquoted spaced custom value yields a bogus flag (STALE — the example was fixed by SM-30)

> **⚠️ Corrected 2026-09-05.** The example below no longer reproduces.
> `--custom-sbatch="--exclusive,--comment=big run"` now parses as
> `['--exclusive', '--comment=big run']` and reaches the script as
> `#SBATCH --comment="big run"` — byte-identical to the quoted form. **SM-30 fixed
> exactly this case and pinned it**: see
> `tests/test_portability_round82.py::TestAnUnquotedValueDoesNotBecomeADirective`,
> whose `test_all_three_spellings_agree` asserts `--comment=my run`,
> `--comment="my run"` and `--comment='my run'` all yield `["--comment=my run"]`.
> What survives is narrower and is NOT this example: a space-form flag followed by a
> second bare token, `-C a b` → `['-C a', '--b']`. The decline below still applies to
> that residue, for the reason it gives, and `--b` is the loud rejection it argues
> for. Measured, both parsers, 14 inputs.

The round trip's other failure, as originally recorded. `--custom-sbatch="--exclusive,--comment=big run"`
parsed as `['--exclusive', '--comment=big', '--run']`, and sbatch rejected the script
("Try `sbatch --help`" — `--run` is not an option). Unchanged from v0.5.1, and the
quoted form worked correctly (`--comment="big run"` → one flag).

> **Deliberately not "fixed".** The input is ambiguous: `run` could be a continuation of
> the comment *or* the dashless option the parser already supports (`exclusive` →
> `--exclusive`). Guessing "continuation" would silently fabricate values — and for a
> constraint it would interact with P4's whitespace strip to invent a feature name
> (`-C a b` → `-C ab`), turning a loud, immediate rejection into a job that runs on the
> wrong nodes. A rejected script is the better failure mode.
>
> **What did change:** the guidance where the value is typed. The wizard's custom-flags
> subtitle and the `--custom-sbatch` help now both say a value may use `=` **or** a
> space, and that a value containing a space must be quoted.
>
> **⚠️ That was only half true until 2026-09-05.** The `--custom-sbatch` help did say
> it ("A value may use '=' or a space (-C bigmem)"). The wizard subtitle did not: it
> showed `--exclusive --reservation=abc` and the quoting rule, and never mentioned the
> space form — on the surface where most values are actually typed, and for the one
> form the parser had newly gained. The subtitle now carries `-C bigmem` in its
> example. Added to the EXAMPLE rather than as prose because `tui.py` reuses the
> subtitle verbatim as the validation error (`f"Invalid input: {s.subtitle}"`), so the
> line has to stay readable as a rejection; it went from 106 to 116 characters.
> `tests/test_custom_flag_guidance_surfaces.py` pins both surfaces, and pins that the
> forms they advertise are the ones the parser actually accepts.

### Not a slurmate bug: an OR constraint denied by cluster policy

Also from the round trip, recorded so it isn't mistaken for a regression later: the
script carrying `#SBATCH --constraint=a100|v100` failed with
`allocation failure: Access/permission denied`. That is **not** slurmate's doing — the
identical request typed straight into `sbatch`, bypassing slurmate entirely, fails the
same way, as does every OR expression tried (`384g|192g`), while each individual
feature schedules fine. It is a policy of this cluster's configuration for this
account. slurmate emitted exactly what was asked for.

---

## Found in the test suite (T1–T2)

`test_portability_round82.py::TestAFatalSignalPutsTheTerminalBack` was failing in the
whole-suite run and passing when its own class was selected, on a 25 s deadline:
`Failed: the wizard never exited after sig:HUP`. Chasing that split found **no leaked
state** — and two real defects in how the class drives a pty.

Ruled out by measurement rather than by reading:

- **A leaked signal disposition or mask.** `/proc/<pytest>/status` before and after the
  whole suite: `SigBlk 0000000000000000`, `SigIgn` = `SIGPIPE|SIGXFSZ` only,
  `getsignal(SIGHUP)` and `getsignal(SIGTERM)` both `SIG_DFL`, one thread. A
  `pty.fork()` child dumping its *own* status in the same run: identical. So nothing
  leaks a handler, and no wizard is started with `SIGHUP` blocked or ignored — the
  first thing to suspect in a tool that installs handlers, and it is not this.
- **Anything else a wizard inherits.** The child's whole environment dict, open
  descriptors, rlimits, cwd, session, process group and the pty's foreground process
  group, dumped after the class alone and after the whole file: **identical** apart
  from the pids and from three leaked descriptors that are present in *both* orderings
  (that is T2). The file's earlier classes hand the wizard nothing the class alone
  does not.
- **The handler blocking instead of re-raising.** `restore_terminal_on_fatal_signal`
  does two blocking things before `os.kill(os.getpid(), signum)` — `tcsetattr` with
  `TCSADRAIN` (which waits for the output queue) and a buffered write to the tty.
  Driven with the pty reader starved, with the slave's output queue deliberately
  filled, at load 11, and with the child's startup flooded and slowed: the wizard died
  **0.00–0.20 s** after `SIGHUP` in every one of 25+ drives.

The suite is green here — 2 whole-suite runs, 9 runs of the file (3 of them
concurrent), 4 runs of the class, 2,263 tests collected and green — so what remains is the one
thing in the drive that a loaded node changes: **when** the signal is sent.

### T1 — The fatal-signal test bets on interpreter startup, and passes vacuously when it loses the bet (medium)

`_drive` pumped the pty for a hard-coded 5 s and then signalled, whatever state the
wizard was in. Startup is the one part of this that a loaded node stretches: slurmate
imports ~200 modules, `PYTHONDONTWRITEBYTECODE` (set for every run in this project,
with `--cov` tracing on top of it in CI) recompiles them from a shared filesystem on
*every* drive, and the whole-suite run pays that while 2,200 other tests contend for
the same cores. Measured on an idle-ish node, from `execv`: the banner lands at
**0.36 s** and the first composed frame at **0.94 s** cold, 0.48 s warm — comfortably
inside 5 s here, and a bet either way.

Both ways of losing it are bad, and the common one is silent. Signalling at 0.2 s —
i.e. the same fixed-sleep design on a node whose startup outlasts the wait — with the
assertions exactly as they were:

```console
{'signalled': True, 'code': None, 'echo_canon': (True, True), 'alt': (0, 0), 'traceback': False}

  echo_canon == (True, True)    ✓   a master whose termios nobody ever touched
  alt[1] == alt[0]              ✓   0 == 0: the alternate screen was never entered
  not traceback                 ✓
  signalled is True             ✓   the *default* disposition ended the process
```

Four green assertions, and the handler never ran. The other way of losing it is the
reported failure: a signal delivered mid-startup leaves the 25 s exit budget to be
spent waiting for an interpreter that is still importing, and the verdict is then
written against the handler — "the wizard never exited after sig:HUP" — which is a
statement about the test's synchronisation, not about `tui.py`.

> **Fix:** `_drive` waits for the wizard to be *up* before it signals — the alternate
> screen (`\x1b[?1049h`) plus the frame's landmarks (`Steps`, `Job name`, `Step 1`),
> bounded by a 30 s startup budget, and a distinct failure ("never composed a frame …
> nothing was signalled, so this says nothing about the handler") if it never arrives.
> The 25 s exit deadline is untouched; it now measures only the exit, which is what it
> claims to measure. This is what `TestWizardActuallyStarts._render` in
> `test_cluster_portability.py` already does, for the reason its own comment gives —
> waiting for a byte count was a race there too. The new
> `assert got["alt"][0] == 1` is what closes the vacuous pass: it is the one assertion
> that fails when the drive goes back to a fixed sleep. Side effect of waiting for the
> frame instead of 5 s of it: the class runs in **2.6 s instead of 20.9 s**.
> **Tests:** the strengthened `test_the_terminal_is_restored_and_the_signal_is_not_swallowed`
> (2, one per signal) and `test_a_wizard_that_exits_before_it_draws_is_reported_as_exited`
> (1) — the control for the wait, which must still report a `--version` that exits
> before drawing as an honest exit rather than sitting on it for the whole budget.

### T2 — The pty helper leaked its master, and the interrupt helper left a `sleep 300` running for five minutes (low)

Two leaks, both in the same file, both measured:

- `_drive` never closed the pty master. `os.forkpty()` opens it **without
  `O_CLOEXEC`**, so the leak was inherited rather than merely counted: the pytest
  process went from 6 open descriptors at the start of the suite to 9 at the end (one
  per drive), and the wizard under test ran holding `fds 11, 12, 13 -> /dev/ptmx` —
  the masters of the three drives before it. A wizard in a test about terminal state
  should not be holding two other terminals.
- `_interrupt_during` writes `#!/bin/bash\nsleep 300` stubs for the Slurm clients and
  then interrupts slurmate. The `KeyboardInterrupt` unwinds through
  `subprocess.run`, which kills the `bash` it started — and the `sleep` that bash
  forked is reparented to init and runs out its five minutes. Measured: a `sleep 300`
  from this test still running minutes after the suite had finished, one per run.

> **Fix:** `_drive` closes the master in a `finally` (so the assertion paths and the
> `pytest.fail` paths all give the pty back), and `_interrupt_during` starts slurmate
> with `start_new_session=True` and sweeps that group with `os.killpg(proc.pid,
> SIGKILL)` in a `finally`, returning the pgid so a test can check it. Neither changes
> what the two tests assert. **Tests:** `test_a_drive_gives_the_pty_back` (1 — a probe
> descriptor must land on the same number before and after a drive, since POSIX hands
> `open()` the lowest free one) and `test_no_stub_outlives_the_interrupt` (1 — the
> process group must be gone, with `rc == 130` asserted first so the test cannot pass
> by having had nothing to leak).

---

## Withdrawn / not changed

Two items were investigated and deliberately left alone. Both were suspected bugs that
measurement disproved — recorded here so they aren't "re-fixed" later.

- **`validate_time("1-99")` is not a bug.** An earlier audit listed it as "a value
  sbatch rejects (hours should be 0-23)". Slurm accepts it:

  ```console
  $ sbatch --test-only --time=1-99 --wrap=true    # no time error (identical to --time=1-23)
  $ sbatch --test-only --time=abc  --wrap=true    # sbatch: error: Invalid --time specification
  ```

  `1-99` parses as 1 day + 99 hours. `25:99:99` is likewise accepted by Slurm while
  slurmate *rejects* it — the validator is already **stricter** than Slurm, not looser.
  Tightening the days-hours field would have rejected input Slurm honours.
- **Case-duplicated GPU types in the picker are correct.** `fetch_gpu_types_for_partition("test")`
  returning `['A100','H100','H200','L40S','a100','a30','a40','rtx6000','v100']` looks
  like a de-duplication wart, and I had it filed as one until the M6 measurement:
  `A100` and `a100` select *different nodes*, so case-folding the list would silently
  break the constraint. The picker keeps both; the M6 warning is what helps a user who
  picks the wrong spelling.

---

## Areas checked and found solid (ruled out)

Scrutinised, and in most cases executed, without finding a defect beyond what existing
tests cover:

- Newline-injection folding — re-executed with CR/LF smuggled into `job_name`,
  `partition`, `memory`, `time_limit` and a `custom_sbatch` value: every value stays on
  a single directive line, nothing is injected into the script body, and no following
  directive is lost. (A regression guard for this was added to the M5 fix, since the
  new rejoin pass touches the same list.)
- Array `%A_%a` insertion and the `.err` collision fix — re-executed across
  `output_file` = *unset* / `run.log` / `run.%j` / `run.err` / `run` with `array_spec`
  set; each yields distinct per-task stdout/stderr paths (`run.err` correctly becomes
  `run-%A_%a.err` + `run-%A_%a-err.err`).
- Config parsing — the naive `key=value` fallback diffed against `tomllib` on a
  realistic config (comments, arrays, `[defaults]`/`[slurmate]` precedence): identical
  dicts. Divergences exist only for TOML that never appears in a slurmate config *and*
  only on the near-dead path where neither `tomllib` nor `tomli` imports.
- Wizard state machine: `_skipped_indices` pruning, the step counter (`visible_done + 1`
  cannot exceed `visible_total`), value preservation on back/forward, `edit()` re-entry,
  and the partition/gpu_format/ntasks/env_name sub-flows. (The 7,000+ randomised
  navigation trials an earlier audit cites were **not** re-run; the reasoning and
  targeted checks were re-derived instead.)
- Memory/time round-tripping: `validate_memory` ↔ `normalize_memory` ↔
  `_parse_mem_to_mb` and `validate_time` ↔ `_parse_slurm_time_to_minutes` stay mutually
  consistent across boundary, heterogeneous (`+` suffix), N/C-suffix, fractional and
  `infinite`/`UNLIMITED` cases.
- `fetch_partitions` (node-count summing across state rows, mem `+` suffix, socket
  suffix, MIG/mps GRES, count-only `has_gpu`), `fetch_queue_eta` state-flag stripping,
  `_extract_first_json`, `fetch_conda_envs` classification — no crash or misparse on
  realistic input, including this cluster's live `sinfo`.
- Custom-flag quoting (`--comment=my job` → `--comment="my job"`), the GPU-flag
  exact-duplicate dedup, stringy-numeric coercion, job-name sanitising/fallback,
  federated job-id splitting (`jobid;cluster`), and Rich-markup escaping in the summary.

---

## Change summary (v0.5.1 → v0.5.2)

| File | What changed |
|---|---|
| `src/slurmate/builder.py` | Custom-flag normalisation/splitting helpers (`_normalize_custom_flags`, `_split_flag`, `_join_flag_values`, `_VALUE_TAKING_FLAGS`); constraint merge (H1) + `_constraint_term` + `_clean_constraint` (P4); per-stream output/error dedup (M1); GPU-option override suppression + `_auto_gpu_flag_name` (P5); `_custom_mem_override` shared with the summary (M3, P5); space-form value quoting (P1); strip-before-expanduser (L2); portable mamba activation (L10); `estimate_gpu_hours` |
| `src/slurmate/system_utils.py` | `fetch_gpu_type_sources` + provenance-aware `fetch_gpu_types_for_partition` (H2); `validate_job_config` requestability error, case warning, and effective-memory checks (H2, M6, P2, P3); `_KNOWN_GPU_MODELS`/`_INFRA_TOKEN_RE`/tighter shape regex (M2); `_sbatch_log_path` rewrite + `effective_log_path` (L4, M1); makedirs after the sbatch check (L3); `validate_memory` units K/M/G/T (L7) |
| `src/slurmate/tui.py` | `mem_per_cpu` + `constraint` steps and their coercion (M4); `_parse_custom_flags` delegates to the shared rule (M5); custom-flag subtitle guidance (P6); `_cached_gpu_types` partition-keyed cache (L1); GPU-format default for feature-only models (H2); `(partition, nodes)` ETA key (L9); measured review label width (L6); gpu_type text persisted on Back |
| `src/slurmate/main.py` | Effective log path in the submit report (M1); provenance plumbed into `_partition_issues` (H2); editor label from `_editor_command()` (L5); `--custom-sbatch` help (P6); GPU-hours row |
| `src/slurmate/theme.py` | `FORCE_COLOR` honoured with exact rich parity, empty value included (L8); animation requires a real TTY |
| `src/slurmate/__main__.py` | **New** — `python -m slurmate` |
| `README.md` | D1, D2, D3 + `FORCE_COLOR`, `python -m slurmate` |
| `CHANGELOG.md`, `pyproject.toml` | 0.5.2 entry (including the two withdrawn claims) and version bump |
| `tests/` | +132 tests across 5 files, including a 2,000-combination duplicate/malformed-directive sweep; 7 pre-existing tests updated (H1's and P5's codified behaviour, L3's filesystem expectation, the mock GPU-type ordering, the `fetch_gpu_type_sources` patch target, two index-based step lookups → `_step("key")`) |

Final state: **431 passed**, `ruff check src/ tests/` clean, `mypy src/` clean.

Verification beyond the unit suite:

- **19 generated scripts** validated by the real controller via `sbatch --test-only`
  (17 ok; 1 was P4, now fixed; 1 was the cluster's OR-constraint policy, not slurmate).
- **23,040 generated scripts** swept for duplicate or malformed `#SBATCH` directives
  (found P5; now 0, and >2,000 of those combinations run in the suite).
- **700 randomized wizard navigation trials** across the two new steps: 0 crashes,
  0 step-counter errors, 0 back-traps (the 154 no-move Backs were all the documented
  partition sub-mode unwind), and the answers always still build a script.
- The mamba activation, the H1/H2/M5/P4 directives, and the count-only-GRES failure
  and its fix were each executed against real Slurm / a real clean shell, not inferred.
