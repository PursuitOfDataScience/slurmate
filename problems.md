# slurmate v0.4.1 — cluster-agnostic compatibility audit

**What this is:** a simulation of whether slurmate (as released in v0.4.1) works *seamlessly*,
without friction, on the major SLURM-based supercomputing centers in the US (national labs
and universities). It lists every problem found. This began as an audit (no code changes); the findings
were **subsequently fixed in v0.5.0** — see the **Resolution** section below. The problem
write-ups are kept as-is (present tense) as the rationale for each fix.

**Method:** slurmate's exact behavior was modeled from its source
(`src/slurmate/{system_utils,builder,main,tui}.py`). Each center's *current official
documentation* was then read and slurmate's generated script + live queries were simulated
against it. Because a mid-audit session limit cut short several of the parallel doc-research
passes, live-doc verification completed in depth for seven representative centers spanning every
major operational archetype (TACC, NERSC Perlmutter, SDSC Expanse, Purdue Anvil, OLCF Frontier,
PSC Bridges-2, Harvard FASRC) plus a cross-cutting conda/modules pass; the remaining centers are
covered by archetype mapping and well-established published convention. **Every claim is tagged with its verification status so nothing reads as
more certain than it is.**

**Verification legend:**
- ✅ **CONFIRMED** — quoted from the center's official docs during this audit (URL given).
- 🟡 **CONVENTION** — well-documented, widely-known site behavior, not freshly re-fetched this
  session (treat as high-probability, verify before acting).
- 🔵 **CODE** — a fact about slurmate itself, verified in its source (file:line given).

**Severity:**
- 🔴 **MAJOR** — slurmate's default output is *rejected by `sbatch`* or silently does the wrong
  thing (wrong charge / no activation / OOM). User must intervene.
- 🟠 **MODERATE** — works but wastes allocation, misleads, or is non-idiomatic; easily hit.
- 🟢 **MINOR** — cosmetic, rare, or well-mitigated by an existing escape hatch.

---

## Executive summary — the problems, ranked

| # | Problem | Severity | Where it bites |
|---|---------|----------|----------------|
| 1 | **`--mem=16G` is emitted on essentially every job** — hard-rejected on exclusive-node centers | 🔴 MAJOR | TACC (all systems); any whole-node site that disables `--mem` |
| 2 | **Required directives are omitted by default** (`--account`, node-type `-C cpu/gpu`, sometimes `--qos`/`--partition`) | 🔴 MAJOR | NERSC (rejects w/o `-A` **and** `-C`); SDSC/Anvil/Michigan/GaTech/most ACCESS (`--account`) |
| 3 | **conda `source activate <env>` in a non-login batch shell** can silently fail to activate | 🔴 MAJOR (runtime) | Any modern-conda site where the module doesn't pre-source `conda.sh` |
| 4 | **Default GPU form `--gres=gpu:type:N` is right for GRES-style sites but wrong where docs use `--gpus*` (and rejected at TACC)** | 🟠 MODERATE (🔴 at TACC) | 🔴 TACC rejects `--gres`; 🟠 NERSC/SDSC/Anvil prefer `--gpus*`/`--gpus-per-node`; ✅ idiomatic at Bridges-2/Harvard/shared-node sites |
| 5 | **Silent allocation over-charge from the `4 cpu / 16G` defaults** on shared/"max(core,mem)" billing | 🟠 MODERATE | Purdue Anvil (2× SU), SDSC shared, any "whichever is larger" site |
| 6 | **Mock accounts/partitions/QoS leak as real-looking suggestions** when SLURM sub-commands are unavailable | 🟠 MODERATE | No-accounting clusters; centers that don't expose `sacctmgr` to users |
| 7 | **"Public partition" heuristic misses `AllowGroups`/`AllowQos`/`DenyAccounts` gating** | 🟢 MINOR | Group- or QoS-gated partitions (mitigated by `[Private]`/`[Custom]`) |
| 8 | **No native `--mem-per-cpu`, `--gpus-per-node`, `--gpus-per-task`, `-C`, `--exclusive` fields** | 🟢 MINOR | Sites whose docs mandate those forms (custom-flag workaround exists) |

**Bottom line:** slurmate is *architecturally* cluster-agnostic — it hardcodes no site names, queries
everything live, degrades gracefully when SLURM tools are missing, and lets the user type/skip
anything. But its **opinionated defaults** (always emit `--mem`; default GPUs to `--gres`; legacy
`source activate`; omit `--account`/`-C`) are tuned for a *shared-node, account-optional, GRES-style*
cluster (like UChicago Midway3, where it was built). On the two most distinctive national-center
models — **TACC's exclusive-node "no `--mem`, no `--gres`"** and **NERSC's mandatory `-A`+`-C`** — the
out-of-the-box script is **rejected by `sbatch`** until the user adjusts it. Every issue has a
user-side workaround inside slurmate's existing UI; none is a dead end. The gap is *seamlessness*,
not *capability*.

---

## Resolution — fixed in v0.5.0

All findings were addressed (verified on real midway3 + 282 passing tests). **No base-case
regression:** on shared-node clusters the generated script is byte-for-byte unchanged
(`--mem`, `--gres=gpu:type:N`, and every default identical).

| # | Fix |
|---|-----|
| A1 | Added `--mem-per-cpu`; `--memory none`/empty omits `--mem` (batch); a user-supplied `--mem`/`--mem-per-cpu` custom flag suppresses the auto one (no double-directive). |
| A2 | First-class `--constraint` (Slurm `-C`), threaded through CLI/config/builder — covers NERSC's mandatory `-C cpu`/`-C gpu`. |
| A3 | conda/mamba now emit `source "$(conda info --base)/etc/profile.d/conda.sh"` then `conda activate` — works in a non-login batch shell (the old bare `source activate` did not). |
| A4 | Added `--gpu-format gpus_per_node` and `gpus_per_task` (with custom-flag dedup); `--gres` kept as the default (correct for most clusters incl. midway3). |
| A5 | **Reconsidered and dropped.** A memory-per-core warning can't tell a normal request (e.g. 4 CPU / 16G) from genuine waste without the site's billing model — it fired on reasonable defaults on low-per-core-memory partitions (e.g. midway3 `amd`) and was net-negative UX. Removed; the real per-node "Memory exceeds partition limit" check stays. |
| A6 | Mock accounts/partitions/modules/GPU-types appear ONLY under `SLURMATE_MOCK`; a real-cluster query failure now returns empty (the user types their own) instead of fake data. |
| A7 | Public-partition test also requires `State=UP`. (`AllowGroups` still needs the caller's groups; `[Private]`/`[Custom]` remain the escape hatches.) |
| A8 | Subsumed by A1/A2/A4 (mem-per-cpu, constraint, GPU formats now first-class); `--exclusive`/`--reservation` remain via the custom-flags field. |
| A9 | Module list strips Lmod terse extras (trailing `/`, `(D)`/`<F>` tags, `(@alias)`). |

**Applicability to midway3 (this cluster):** verified empirically — A1/A4/A6 never applied
here (shared nodes so `--mem` is valid; `--gres` is the native GPU idiom; real accounts via
`sacctmgr` with default `rcc-staff`). The A5 warning was removed after it wrongly flagged a
normal `4 cpu / 16G` request on the `amd` partition. Only A3 (conda `source activate`)
genuinely affected midway3, and it is now fixed and re-verified on the cluster.

---

## Part A — Tool-level problems (cross-cutting; apply everywhere)

These stem from slurmate's own defaults/logic, independent of any one center.

### A1. `--mem=16G` is emitted on virtually every job 🔴 / 🔵
- 🔵 CODE: `build_sbatch_script` emits `#SBATCH --mem=<memory>` whenever `memory` is truthy
  (`builder.py:299`); batch default is `16G` (`main.py:113`) and the wizard carries a memory
  default too, so `--mem` is present on essentially every generated script. There is **no native
  `--mem-per-cpu`** (only via a typed custom flag).
- ✅ CONFIRMED impact — **TACC Stampede3**: "*Not available. If you attempt to use this option, the
  scheduler will not accept your job.*" for `-mem`, because "*TACC does not implement node-sharing
  on any compute resource.*" → slurmate's default script is **hard-rejected** at submit on all TACC
  systems (Stampede3, Frontera, Lonestar6).
- 🟠 Secondary impact — on shared partitions that bill `max(cores, memory-fraction)` (Anvil, SDSC),
  a 16G default can silently over-charge (see A5/Anvil).
- **Fix (future):** make `--mem` omittable/absent by default, or add a per-cluster profile /
  `--no-mem` / `--mem-per-cpu`; detect exclusive-node partitions (`sinfo` `OverSubscribe=EXCLUSIVE`
  / `%h`) and drop `--mem`. Today's workaround: clear the memory field (the wizard allows empty) or
  `--memory ""` is **not** currently accepted — user must delete it in the review/editor step.

### A2. Site-mandatory directives are omitted by default 🔴 / 🔵
- 🔵 CODE: `--account`, `--qos`, `--partition` are each omitted when blank
  (`builder.py:285-290`); missing ones produce only an *advisory* "Missing recommended fields"
  note (`main.py:309-320`), never a hard stop (except `--yes` requires a non-empty command,
  `main.py:681-685`). There is **no field for node-type `--constraint`/`-C`** (custom flag only).
- ✅ CONFIRMED impact — **NERSC Perlmutter**: `-A/--account` is mandatory *and* `-C cpu`/`-C gpu`
  is mandatory; omitting **either** yields `sbatch: error: Job request does not match any
  supported policy.` slurmate emits neither by default → **hard reject**.
- ✅ CONFIRMED — **SDSC Expanse**: "*Expanse requires users to enter a valid project name*"; every
  sample script has `#SBATCH --account=...`. 🟡 CONVENTION — **most ACCESS/allocation centers**
  (Michigan Great Lakes, Georgia Tech PACE, OSC, TAMU) require `--account`; a job without it is
  rejected or charged to an unintended default.
- **Fix (future):** when the resolved `--account` list has exactly one real entry, default to it;
  add first-class `--constraint` and a "this cluster requires an account/constraint" gate.
  Today's workaround: user supplies `--account`, and `-C ...` via the custom-flags field.

### A3. conda activation uses legacy `source activate` in a non-login shell 🔴(runtime) / 🔵 / ✅
- 🔵 CODE: the generated script starts `#!/bin/bash` (non-login; `builder.py:252`) and, for a conda
  env, emits `source activate <env>` (`builder.py:402`).
- ✅ CONFIRMED (conda docs + HPC centers): since conda 4.4 `conda activate` is the preferred form;
  modern conda's `bin/activate` is a thin shim that needs `_CONDA_ROOT` already set (i.e. `conda.sh`
  already sourced). In a bare non-login batch shell that hasn't run `conda init`/sourced `conda.sh`,
  `source activate <env>` can **silently fail to activate** (job then runs in base/system Python) or
  error `CommandNotFoundError`. Note slurmate's *probe* runs in a login shell
  (`bash -lc "... conda info --json"`, `system_utils:660`) so it succeeds even when the generated
  non-login script won't — a probe/runtime mismatch.
- **Fix (future):** emit the portable pattern
  `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate <env>` (or
  `eval "$(conda shell.bash hook)"`), or generate `#!/bin/bash -l`. Today's workaround: user adds
  a `module load`/init line, or uses the venv option, or hand-edits in the editor step.

### A4. Default GPU directive `--gres=gpu:type:N` is site-dependent 🟠 (🔴 at TACC) / 🔵
- 🔵 CODE: default `gpu_format` is `gres_type` → `#SBATCH --gres=gpu:<type>:<n>`
  (`builder.py:310-321`). Alternatives exist and are one setting away: `gpus` → `--gpus=<type>:<n>`,
  `constraint` → `--gres=gpu:<n>` + `--constraint=<type>` (`--gpu-format` / `SLURMATE_GPU_FORMAT`).
  There is **no native `--gpus-per-node`/`--gpus-per-task`**.
- The default is **correct on GRES-style clusters** (the largest bucket) but wrong on others —
  confirmed conventions of the audited centers:
  - ✅ **PSC Bridges-2** — `--gres=gpu:type:n` (e.g. `--gres=gpu:v100-32:8`). **Matches slurmate's
    default exactly.**
  - ✅ **Harvard FASRC** — `--gres=gpu:n` (also offers `--gpus=N`). Matches.
  - 🟡 Stanford Sherlock, Princeton, NYU Greene, UChicago Midway3, Michigan, GaTech and most
    shared-node university clusters accept/use `--gres` → slurmate's default is idiomatic.
  - ✅ **TACC** — `--gres` is **rejected** ("*Slurm will reject any script with this directive*");
    GPUs via GPU-specific queue (`-p h100`/`pvc`). 🔴
  - ✅ **NERSC** — `--gpus` / `--gpus-per-node` / `--gpus-per-task` (examples use `--gpus-per-task`),
    and `-C gpu` is separately required.
  - ✅ **SDSC** — `--gpus` (e.g. `--gpus=h100:1`).
  - ✅ **Purdue Anvil** — `--gpus-per-node` in every published example.
  - ✅ **OLCF Frontier** — no explicit GPU flag at all; all 8 GCDs come with each whole node.
  - Note: on standard SLURM `--gres=gpu:N` usually still *works* even where docs prefer `--gpus*`,
    so outside TACC this is "non-idiomatic," not fatal.
- **Fix (future):** add `--gpus-per-node`/`--gpus-per-task` formats; keep `--gres` default (it's the
  most broadly accepted) but auto-switch on known exceptions. Today: switch `--gpu-format gpus`
  (equals per-node on a 1-node job), or use custom flags, or drop it entirely on whole-node sites.

### A5. `4 cpu / 16G` defaults can silently over-charge on "max(core, mem)" billing 🟠 / 🔵 / ✅
- 🔵 CODE: batch defaults `cpus=4` (`main.py:109`) and `mem=16G` (`main.py:113`); the validator
  checks only absolute per-node limits, not the memory/core *billing ratio*
  (`system_utils.validate_job_config`).
- ✅ CONFIRMED — **Purdue Anvil**: SU = "one core using <~2G for one hour"; shared jobs billed on
  "*the number of cores or the fraction of the memory requested, whichever is larger*." 16G ÷ 2G/core
  = 8-core-equivalent > the 4 cores requested → **2× silent SU over-charge on every default job** on
  the `shared` (default) partition. Same "whichever is larger" rule is 🟡 CONVENTION at SDSC shared
  and other shared-partition centers.
- **Fix (future):** warn when `mem/cpus` exceeds the partition's memory-per-core ratio (slurmate
  already fetches both `cpus_per_node` and `mem_per_node_mb`). Today: user raises `--cpus` or lowers
  `--memory` to match the ratio.

### A6. Mock accounts / partitions / QoS / modules leak as real-looking suggestions 🟠 / 🔵
- 🔵 CODE: when a SLURM sub-command is missing or errors, fetchers fall back to **demo data** —
  `MOCK_ACCOUNTS = ["my_lab","training","default"]`, `MOCK_PARTITIONS`, `MOCK_QOS`, `MOCK_MODULES`
  (`system_utils.py:15-29,528`). In interactive mode the account step is a free-text autocomplete
  seeded with these (`tui.py:316-318`); a user could pick a **fake** account on a cluster where
  `sacctmgr` isn't exposed → `#SBATCH --account=my_lab` → reject. There is no on-screen "this is
  demo data" marker.
- Scope limiters (why it's MODERATE, not MAJOR): batch mode never calls `fetch_user_accounts`
  (account only from `--account`/config → **no leak in batch**); the field is skippable (empty
  allowed); real clusters with `sacctmgr` show real data. 🟡 Anvil: `sacctmgr` isn't documented for
  users (they use `mybalance`), so the fake-suggestion path is plausibly reachable there.
- **Fix (future):** visually mark demo data, or suppress account/QoS suggestions entirely when the
  query failed (return "unknown" instead of mock, as `fetch_known_qos` already does). Today: user
  types their real account (from `mybalance`/`sacctmgr`/`iam`), or leaves it blank.

### A7. "Public partition" heuristic is narrower than SLURM's real access model 🟢 / 🔵
- 🔵 CODE: a partition is treated as "public" iff `AllowAccounts=ALL` **and** `Hidden!=YES`
  (`system_utils.py:504-507`). It ignores `AllowGroups`, `AllowQos`, `DenyAccounts`, `DenyQos`, and
  partition `State`. So a partition the user *can* use but that's group/QoS-gated is demoted to the
  `[Private]` sub-list, and one that's `AllowAccounts=ALL` but `AllowGroups`-restricted (unusable) is
  shown as public.
- Well-mitigated: the picker always offers `[Custom]` (type any name) and a `[Private]` list of all
  partitions (`tui.py:934-939`), so nothing usable is truly hidden — only mis-ranked. Advisory only.
- **Fix (future):** factor `AllowGroups`/`DenyAccounts`/`State` into the public test. Today: pick
  `[Private]` or `[Custom]`.

### A8. Missing first-class fields for common mandatory forms 🟢 / 🔵
- 🔵 CODE: no built-in field for `--constraint`/`-C`, `--mem-per-cpu`, `--gpus-per-node`,
  `--gpus-per-task`, `--exclusive`, `--reservation` (all reachable only via the free-text
  custom-flags field; that field's allow-list mentions some, `tui.py:277-279`).
- Impact is small because custom flags cover every case, but "seamless" centers that *require*
  `-C`/`--mem-per-cpu` need the user to know the flag.
- **Fix (future):** promote `--constraint` and `--gpus-per-node` to first-class questions.

### A9. Minor robustness/UX notes 🟢 / 🔵 / ✅
- ✅ `module -t avail` writes to **stderr** on both Lmod and TCL — slurmate **handles this
  correctly** already (`bash -lc "... module -t avail 2>&1"` and `output = stdout + stderr`,
  `system_utils.py:697-698`). Listed only to record it was checked and is *not* a bug.
- 🟢 Lmod terse output can include extras slurmate doesn't strip (trailing-`/` short names,
  `(@alias)`, `<F>`/`(D)` tags); it does strip `:`-headers and `(default)`. Cosmetic list noise.
- 🟢 `bash -lc` sources the user's full login profile for the conda/module probes; a heavy or
  banner-printing profile could slow them (30s timeout, `system_utils.py:32`) — `_extract_first_json`
  already tolerates banners.

---

## Part B — Center deep-dives (✅ doc-confirmed this session)

### B1. TACC — Stampede3, Frontera, Lonestar6 → 🔴 MAJOR
The single hardest environment for slurmate's defaults.
- **Exclusive nodes:** "*TACC does not implement node-sharing on any compute resource.*"
- **`--mem` forbidden:** "*Not available. If you attempt to use this option, the scheduler will not
  accept your job.*" → slurmate's always-present `--mem=16G` ⇒ **`sbatch` rejects the job**.
- **`--gres`/`--gpus-per-task` rejected:** "*Slurm will reject any script with this directive*";
  GPUs come from GPU queues (`-p h100`/`pvc`) ⇒ slurmate's default `--gres=gpu:…` ⇒ **rejected**.
- `-A` needed only for multi-project logins; `-p` should be set.
- **Seamless?** No — a default GPU (or even CPU-with-mem) script is rejected until the user (a)
  deletes memory and (b) avoids `--gres` (choose the GPU queue via `--partition`). Both are doable
  in slurmate, none is automatic.
- Source: https://docs.tacc.utexas.edu/hpc/stampede3/ , https://docs.tacc.utexas.edu/hpc/frontera/

### B2. NERSC — Perlmutter → 🔴 MAJOR
- **`-A/--account` mandatory** and **`-C cpu`/`-C gpu` mandatory** — omit either ⇒
  `sbatch: error: Job request does not match any supported policy.` slurmate emits neither by
  default ⇒ **rejected**.
- **GPUs:** "*You must explicitly request GPU resources using … `--gpus`, `--gpus-per-node`, or
  `--gpus-per-task`*" (examples favor `--gpus-per-task`). slurmate default `--gres` is a mismatch
  (and `-C gpu` is still separately required).
- QoS (`-q`) appears in all examples (`shared` for 1–2 GPUs); effectively expected.
- **Seamless?** No — needs `--account`, `-C cpu|gpu` (via custom flag), and ideally `-q`/`--gpus*`.
- Source: https://docs.nersc.gov/systems/perlmutter/running-jobs/

### B3. SDSC — Expanse → 🟠 MODERATE
- **`--account` mandatory:** "*Expanse requires users to enter a valid project name*" (all samples
  carry `#SBATCH --account=…`). slurmate omits ⇒ user must supply.
- **`--partition` required**, and it distinguishes shared vs. exclusive `compute` (which uses
  `--mem=0`). `--mem` is *used* here, so slurmate's `--mem=16G` is fine on `shared` (but see A5
  billing).
- **GPUs via `--gpus`** (e.g. `--gpus=h100:1`), not `--gres` → convention mismatch (usually still
  works; `--gpu-format gpus` matches).
- **Seamless?** Mostly, once the user provides `--account` + `--partition` (both are normal wizard
  steps). GPU form and shared-billing are the rough edges.
- Source: https://www.sdsc.edu/systems/expanse/user_guide.html

### B4. Purdue — Anvil (ACCESS) → 🟠 MODERATE
- **Silent 2× SU over-charge** on the default `shared` partition from `4 cpu / 16G` (A5).
- **Account** effectively required (`mybalance`; distinct CPU vs GPU allocation codes like
  `xxx-gpu`); wrong code ⇒ reject or mis-charge. `sacctmgr` not documented for users ⇒ mock-account
  suggestion risk (A6).
- **GPUs documented as `--gpus-per-node`** (not native in slurmate); `--gres` likely works but is
  non-idiomatic. Leaving partition blank + requesting GPUs lands on `shared` (no GPUs) ⇒ reject, and
  slurmate's own GPU-partition warning is suppressed when partition is blank.
- **Lmod** ⇒ `module -t avail` works.
- **Seamless?** Functional, but with the silent-overcharge and GPU-directive rough edges.
- Source: https://docs.rcac.purdue.edu/userguides/anvil/jobs/

### B5. OLCF — Frontier → 🔴 MAJOR
- **`-A <projid>` mandatory**; **`-p batch` required**; nodes **exclusive whole-node** so **`--mem`
  is not used** (request `-N <nodes>`); **GPUs are automatic** — all 8 MI250X GCDs come with each
  node, no `--gres`/`--gpus` flag. Typical required header is just `-A`, `-p`, `-N`.
- **vs slurmate:** the default script omits `-A` (miss), emits `--mem=16G` (wrong idiom on a
  whole-node system), and would emit `--gres=gpu:…` for a GPU request (not how Frontier works).
  Effectively needs a hand-built script — slurmate's `--print` + editor is the realistic path here.
- Source: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html

### B6. PSC — Bridges-2 → 🟢–🟠 (friendliest ACCESS center for slurmate)
- **`-A` NOT mandatory** ("*If not specified, your default allocation id is used*"; specify only
  with multiple allocations); **`-p` defaults to `RM`** (specify for GPU/shared). RM nodes are
  exclusive; RM-shared/GPU-shared allow partial. `--mem` not emphasized (implicit by cores/node).
- **GPUs via `--gres=gpu:type:n`** (e.g. `--gres=gpu:v100-32:8`) — **matches slurmate's default.**
- **vs slurmate:** close to seamless — account defaults, partition defaults, GPU form matches. Only
  rough edge: `--mem=16G` on an exclusive RM node is redundant (not documented as rejected).
- Source: https://www.psc.edu/resources/bridges-2/user-guide/

### B7. Harvard — FASRC Cannon → 🟢 MINOR (Archetype-3 reference: slurmate ≈ seamless)
- **`--account` NOT mandatory** (only to steer fairshare when in multiple labs); **`-p` NOT required**
  (defaults to `serial_requeue`); nodes **shared**, **`--mem` used** (`--mem-per-cpu` also available);
  **GPUs via `--gres=gpu:n`** (or `--gpus=N`).
- **vs slurmate:** its defaults are all valid here — `--mem=16G` ✓, `--gres=gpu:N` ✓, account/partition
  optional. This is the shared-node university profile slurmate was designed for; near-zero friction.
- Source: https://docs.rc.fas.harvard.edu/kb/running-jobs/

---

## Part C — Broader survey by archetype (🟡 CONVENTION unless marked ✅)

slurmate's friction is predicted well by which *archetype* a center fits. The seven confirmed centers
anchor the archetypes; the rest are mapped by published convention (verify per site).

**Archetype 1 — Exclusive whole-node, `--mem`/`--gres` discouraged or forbidden → 🔴 for slurmate**
- ✅ TACC Stampede3 / Frontera / Lonestar6; ✅ OLCF Frontier (whole-node, `-A` mandatory, no
  per-core `--mem`, GPUs automatic per node).
- 🟡 Other DOE capability systems follow the same whole-node model. ⇒ slurmate's `--mem` and
  `--gres` defaults are the wrong idiom; `--account` omission is a hard miss.

**Archetype 2 — Mandatory `--account` (+ often `-C`/`-q`) → 🔴/🟠 for slurmate**
- ✅ NERSC Perlmutter (`-A` + `-C` hard-required), ✅ SDSC Expanse (`--account` required),
  ✅ Purdue Anvil (effectively).
- 🟡 University of Michigan Great Lakes, Georgia Tech PACE, Ohio Supercomputer Center, Texas A&M
  HPRC (Grace/FASTER), CU Boulder Alpine (also requires `--qos`), Yale (requires `-p`) — all
  require an account/allocation and/or partition that slurmate omits by default. ⇒ user must supply
  `--account` (and `--qos`/`-p` where mandated).

**Archetype 3 — Shared-node, `--mem` expected, `--gres`/`--gpus` accepted → 🟢/🟠 for slurmate**
- 🟡 Harvard FASRC Cannon, Stanford Sherlock, Princeton (Della/Tiger/Adroit), NYU Greene, Yale
  (Grace/McCleary), UChicago Midway3 (slurmate's home cluster), NCSA Delta, PSC Bridges-2, Rutgers
  Amarel, Duke DCC, UFlorida HiPerGator, Vanderbilt ACCRE.
- These match slurmate's design best: `--mem=16G` is valid, `--gres=gpu:type:N` is accepted, GPUs
  via GRES is common. Residual friction: many still **require `-p`/`--account`** (omitted by
  default), and the shared-partition billing note (A5) applies where billing is `max(core,mem)`.
  This is the "seamless-ish" bucket.

**Archetype 4 — Distinctive/edge environments → verify individually**
- 🟡 MIT SuperCloud: unusual module/submission setup and Anaconda module conventions; `LLsub`
  wrapper is promoted alongside `sbatch` — verify `module`/`sacctmgr` availability and conda init.
- 🟡 LLNL Livermore Computing: "bank" accounts (`-A <bank>`) and `--qos`/pool model; SLURM present
  but with lab-specific banks ⇒ `--account` omission is a hard miss.
- Non-SLURM (out of scope, listed to avoid false coverage): ALCF Polaris/Aurora (PBS Pro), some
  legacy UCLA Hoffman2 (SGE), historically Notre Dame CRC (UGE/SGE — verify current scheduler).
  slurmate targets SLURM only, so these are correctly not "friction" but "not applicable."

### Per-center quick table

| Center | System(s) | SLURM? | `--account` req? | `--mem` ok? | GPU form in docs | Predicted friction |
|--------|-----------|--------|------------------|-------------|------------------|--------------------|
| TACC ✅ | Stampede3/Frontera/Lonestar6 | Yes | multi-proj only | **No (rejected)** | queue `-p` (no `--gres`) | 🔴 MAJOR |
| NERSC ✅ | Perlmutter | Yes | **Yes** | n/a (also `-C` req) | `--gpus*` | 🔴 MAJOR |
| SDSC ✅ | Expanse | Yes | **Yes** | Yes | `--gpus` | 🟠 MODERATE |
| Purdue ✅ | Anvil | Yes | effectively | Yes (over-charge) | `--gpus-per-node` | 🟠 MODERATE |
| OLCF ✅ | Frontier | Yes | **Yes** | **No (whole-node)** | none (auto per node) | 🔴 MAJOR |
| NCSA 🟡 | Delta/DeltaAI | Yes | **Yes** | Yes | `--gres`/`--gpus` | 🟠 MODERATE |
| LLNL 🟡 | LC systems | Yes | **Yes (bank)** | Yes | `--gres` | 🟠 MODERATE |
| PSC ✅ | Bridges-2 | Yes | defaults | implicit | `--gres=gpu:type:n` (matches) | 🟢–🟠 |
| Michigan 🟡 | Great Lakes | Yes | **Yes** | Yes | `--gres`/`--gpus` | 🟠 MODERATE |
| GaTech 🟡 | PACE Phoenix | Yes | **Yes** | Yes | `--gres` | 🟠 MODERATE |
| CU Boulder 🟡 | Alpine | Yes | Yes | Yes | `--gres` | 🟠 MODERATE (also `--qos` req) |
| Yale 🟡 | Grace/McCleary | Yes | account+`-p` | Yes | `--gres` | 🟠 MODERATE |
| OSC 🟡 | Pitzer/Owens | Yes | **Yes** | Yes | `--gres`/`--gpus` | 🟠 MODERATE |
| TAMU 🟡 | Grace/FASTER | Yes | **Yes** | Yes | `--gres` | 🟠 MODERATE |
| Harvard ✅ | FASRC Cannon | Yes | No (fairshare) | Yes | `--gres`/`--gpus` | 🟢 MINOR |
| Stanford 🟡 | Sherlock | Yes | `-p` | Yes | `--gres`/`--gpus` | 🟢 MINOR |
| Princeton 🟡 | Della/Tiger | Yes | usually not | Yes | `--gres`/`--gpus` | 🟢 MINOR |
| NYU 🟡 | Greene | Yes | usually not | Yes | `--gres`/`--gpus` | 🟢 MINOR |
| MIT 🟡 | SuperCloud | Yes | varies | Yes | `--gres` | 🟠 MODERATE (verify env/module) |
| UChicago 🟡 | Midway3 | Yes | usually not | Yes | `--gres` | 🟢 MINOR (home cluster) |

(🟡 rows are convention-based archetype mappings, not fresh doc quotes — verify per site before relying.)

---

## Part D — What already works well (portability strengths)

Stated for balance; these are why slurmate is close to seamless on Archetype-3 clusters and never
*crashes* anywhere:
- 🔵 **No hardcoded site names** — partitions/accounts/QoS/modules/GPU types are all live-queried
  (`sinfo`/`scontrol`/`sacctmgr`) or user-typed. Nothing is UChicago-specific.
- 🔵 **Graceful degradation** — every SLURM binary is optional; a missing/erroring tool falls back
  to mock/empty instead of crashing (`_run_command` handles timeouts, `OSError`, non-UTF-8 output).
- 🔵 **Version-safe command set** — the `sinfo`/`squeue`/`scontrol`/`sacctmgr` format codes and
  `sbatch --parsable` used are old and stable; the default GPU form avoids the newer `--gpus`
  (SLURM 19.05+). Federated `jobid;cluster` output is parsed correctly.
- 🔵 **Robust parsing** — heterogeneous partitions (`%m` "515000+"), typed/count-only GRES,
  node-state flags (`idle~`/`mix*`), multi-line TOML config arrays, module stderr capture.
- 🔵 **Safe preview** — `--print`/`--dry-run` render a script on *any* cluster without submitting,
  so a user can inspect+adapt before running. This is the universal escape hatch that makes every
  MAJOR above recoverable.
- 🔵 **Cluster-neutral wording** — "CPU-hours," not a site-specific "SU"/billing-unit name.

---

## Part E — Recommendations (for a future release; nothing changed now)

Priority order, mapped to the problems above:
1. **Exclusive-node awareness (A1, B1):** detect exclusive partitions and omit `--mem`/`--cpus-per-task`;
   allow an empty/`0` memory default; add `--mem-per-cpu`.
2. **First-class `--account` + `-C`/`--constraint` (A2, B2):** auto-select the sole real account;
   add a `--constraint` question; optional per-cluster "required fields" profile that turns advisory
   warnings into blocks.
3. **Portable conda activation (A3):** emit `source "$(conda info --base)/etc/profile.d/conda.sh" &&
   conda activate <env>` (or `#!/bin/bash -l`).
4. **GPU form breadth (A4):** add `--gpus-per-node`/`--gpus-per-task`; consider `--gpus` as default.
5. **Billing-ratio warning (A5):** warn when `mem/cpus` exceeds the partition's memory-per-core ratio.
6. **No fake suggestions (A6):** mark or suppress mock accounts/QoS when the query failed.
7. **Richer access model (A7):** include `AllowGroups`/`DenyAccounts`/`State` in the public test.
8. **Optional site profiles:** ship `~/.config/slurmate/<cluster>.toml`-style presets (e.g. a
   `tacc` profile: no `--mem`, GPUs via queue) so a center's admins can make slurmate seamless.

None of these blocks use today — every problem has an in-UI workaround (custom flags, free-text
account/partition, editable memory, `--gpu-format`, `--print` preview, or the editor step).

---

## Appendix — audit scope, sources, and limits

**Sources verified live this session (✅):**
- TACC Stampede3 — https://docs.tacc.utexas.edu/hpc/stampede3/
- TACC Frontera — https://docs.tacc.utexas.edu/hpc/frontera/
- NERSC Perlmutter — https://docs.nersc.gov/systems/perlmutter/running-jobs/
- SDSC Expanse — https://www.sdsc.edu/systems/expanse/user_guide.html
- Purdue Anvil — https://docs.rcac.purdue.edu/userguides/anvil/jobs/ (+ getting-started, overview)
- OLCF Frontier — https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
- PSC Bridges-2 — https://www.psc.edu/resources/bridges-2/user-guide/
- Harvard FASRC Cannon — https://docs.rc.fas.harvard.edu/kb/running-jobs/
- conda activation — https://www.anaconda.com/blog/how-to-get-ready-for-the-release-of-conda-4-4 ,
  https://github.com/conda/conda/blob/main/conda/shell/bin/activate ,
  https://docs.conda.io/projects/conda/en/stable/dev-guide/deep-dives/activation.html
- modules (Lmod/TCL) — https://lmod.readthedocs.io/en/latest/040_FAQ.html ,
  https://lmod.readthedocs.io/en/latest/105_terse_output.html ,
  https://modules.readthedocs.io/en/latest/FAQ.html

**slurmate behavior verified in source (🔵):** `src/slurmate/system_utils.py`,
`src/slurmate/builder.py`, `src/slurmate/main.py`, `src/slurmate/tui.py` (v0.4.1).

**Limits / honesty notes:**
- A mid-audit session limit (and an exhausted web-search budget) ended several parallel
  doc-research passes early, so the ~17 remaining centers in Part C are 🟡 CONVENTION
  (archetype-mapped), not freshly quoted; seven were confirmed live (Part B). They should be re-verified against
  current docs before being treated as CONFIRMED — center policies change (queues renamed, GPU forms
  updated, accounting toggled).
- This audit simulates against documentation; it did **not** submit real jobs on these clusters.
  The 🔴 rejections (TACC `--mem`/`--gres`, NERSC `-A`/`-C`) are quoted from the centers' own docs,
  which state the scheduler rejects those scripts.
- Scope is SLURM centers only, per the task; PBS/SGE/LSF centers (ALCF Polaris/Aurora, some legacy
  systems) are out of scope and noted as "not applicable," not "friction."
