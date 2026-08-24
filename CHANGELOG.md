# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and this project adheres to [Semantic Versioning](https://semver.org).

## [0.6.0] — 2026-08-24

Portability pass. Every item here came from installing 0.5.3 on a second,
deliberately different cluster (CentOS 7.9 / Python 3.14 / Slurm 23.02, cgroup
v1, a different partition layout) and running it open-box. The common shape of
the bugs: a value read from Slurm was trusted further than it deserved, and the
result was a confident, complete, unsubmittable answer.

### Added

A second open-box run, on a third environment again chosen for how little it
resembles the first (Booth's Mercury: RHEL 9.8 / Python 3.13 / **Slurm 25.11** /
cgroup v2 / NFS home / Tcl environment-modules 5.x / no default account for the
user), produced the next nine. Same shape as before, one layer up: the tool had
the right answer in hand and either never asked for it on the path being used, or
printed it as something weaker than it was.

- **Slurm's own refusal now reaches every mode.** `sbatch --test-only` was
  consulted from exactly one place — the wizard, and only after a hand edit in
  `$EDITOR`. So `--print`, the mode meant for pipes and CI, made no scheduler call
  at all: on a cluster where the user has no default account (Mercury refuses
  every account-less script with *"Invalid account or account/partition
  combination specified"*) it emitted the unsubmittable script with **zero bytes
  on stderr and rc=0**. `--print` now asks about the bytes it is about to hand
  over, and `--yes` will not fire off a job the controller has already rejected.

- **A refusal is no longer rendered as a time estimate.** `--dry-run` did learn
  the verdict — via the ETA probe — and showed it as the summary row `ETA: never
  — <reason>`: the one fact meaning "this cannot run at all" carried the visual
  weight of a queue depth, while strictly lesser problems (a time limit over the
  partition's, an array index over `MaxArraySize`) each got a marked line of
  their own. It is now stated as an error, on every path.

- **"You cannot submit right now" is no longer confused with "this job is
  wrong".** Both come back from `--test-only` as a non-zero exit, and the first
  version of the check above blocked on either — which on Mercury refused a
  perfectly valid job because its `clay` QoS allows one submitted job per user and
  an unrelated job was already queued. Slurm names the specific limit on a line of
  its own (`QOSMaxSubmitJobPerUserLimit`) above a generic bundle that does not say
  which half was violated; that token was being discarded. It is now kept, shown
  to the user, and used to classify: configuration errors block and fail `--print`,
  transient limits are reported and clear on their own. Unrecognised wordings stay
  advisory, because guessing "permanent" refuses jobs that would have run.

- **The refusal check is classified at all four of its call sites, and the label
  agrees with it.** Two follow-on defects of the three above, both the report's own
  recurring shape. The wizard's hand-edited-script branch blocked on *any*
  refusal, so on Mercury a hand edit stranded a valid script behind an unrelated
  queued job — and said "Slurm rejects the edited script", blaming the edit. And
  the summary row's `never` was decided in `main.py` while the permanence was
  decided in `system_utils`, so a transient cap printed `ETA: never` directly
  above an advisory saying the script was valid and the condition temporary. The
  label and its severity now come from the result itself (`refusal_is_permanent`,
  set once where the refusal is parsed) and read `not right now` for a transient
  cap. A source-level guard test pins that every refusal decision classifies,
  since the wizard branch is only reachable interactively.

- **A refusal slurmate cannot classify no longer claims the script is fine.**
  The classifier above had two buckets, so *anything* it did not recognise as
  permanent was reported as a condition that clears on its own. Two ordinary
  mistakes fall there, both measured on Mercury: `--nodes 2` where the QoS caps
  nodes at 1 (*"Node count specification invalid"*) and a time limit past the
  partition maximum (*"Requested time limit is invalid"*). Both were being told
  *"the script is valid; this clears on its own"* about a job that can never run.
  Both wordings are now recognised as permanent, and — the structural half, since
  no marker list will ever enumerate every Slurm wording — `refusal_is_transient`
  is deliberately **not** the negation of `refusal_is_permanent`. An unrecognised
  refusal reports the controller's own words, labels the ETA `refused` rather than
  `never` or `not right now`, and says plainly that slurmate cannot tell whether
  it clears. It still does not block, because guessing "permanent" fails builds
  over conditions that pass on retry.

- **The suite no longer depends on the shell that runs it.** Seven config tests
  passed on midway3 and failed in GitHub CI for a reason unrelated to the code:
  they set `HOME` and wrote `$HOME/.config/slurmate/config.toml`, but
  `load_config()` honours `XDG_CONFIG_HOME` first, and GitHub's runners export it
  while a midway3 login shell does not. An autouse fixture now clears the
  variables slurmate reads that would silently change a result —
  `XDG_CONFIG_HOME`, `SLURMATE_GPU_FORMAT`, `NO_COLOR`/`FORCE_COLOR`,
  `EDITOR`/`VISUAL`, and `LMOD_CMD`/`MODULESHOME` (both set on any Lmod login
  shell and consulted by the module check, so those tests had been reading the
  host's real module system). Verified by running the suite under each variable
  set, and by confirming the seven failures return when the fixture is disabled.

- **A wizard test was passing for the wrong reason, and only here.** It set
  `COLUMNS=150` in the environment but never sized the pty, and env vars do not
  size a pty — prompt_toolkit asks the tty, which reports 80x24 whatever `COLUMNS`
  says, and which of the two wins varies by prompt_toolkit version. So the frame
  it inspected was never the width it thought. Worse, it stopped reading on
  `len(buf) > 1500`, a byte count standing in for "the frame is complete": on
  Mercury (prompt_toolkit 3.0.53) the read broke out at `Step 1 / 23` with the
  sidebar and first prompt still in flight, and the suite failed for reasons
  having nothing to do with the wizard. It now sizes the pty with `TIOCSWINSZ`
  and waits for the landmarks themselves. How prompt_toolkit chunks its first
  frame is not something to assert on. 80x24 — an unresized ssh session, a CI
  pty, a fresh pty's default — is now its own test, and renders in full.

- **`--constraint` is now checked on a cluster that advertises no features.**
  `fetch_node_features()` returned an empty set both for "sinfo could not be
  asked" and for "sinfo answered, and every node reports `(null)`" — and since the
  check must stay silent on the first, it was inert on the second. Mercury is the
  second: `-C a100` there matches nothing and produced no warning, while a bad
  partition, account, QoS or GPU type all reported correctly. The two are now
  distinct answers (`None` vs `set()`), and a featureless cluster says so.

- **Tcl environment-modules 5.x layouts are found.** `_module_command()` looked
  only for `$MODULESHOME/bin/modulecmd` (3.x); 5.x ships
  `$MODULESHOME/libexec/modulecmd.tcl` and leaves `bin/` without it, so on such a
  site every module check went silently inert unless a wrapper happened to be on
  `PATH`. Both layouts are now tried.

- **The "cannot be created" warning now names the resolved directory.** It
  reported `nearest existing parent: '.'`, which is accurate and useless in a CI
  or job log where the reader cannot see what the working directory was. Now the
  absolute path.

- **Pinned the pristine-environment cases.** The previous round's false warning
  was hidden by the dev environment (`logs/` already exists in the repo), so the
  fresh cases are now tests: a clean directory emits a script and says nothing, no
  `$HOME` still works, and a read-only directory warns while still emitting the
  script. Also confirmed the suite itself is location-independent — 1072 pass when
  run from outside the repo.

- **Fixed a false "log directory cannot be created" warning on the default
  path.** The check walked up to the nearest existing ancestor, and a *relative*
  directory walks to `""` — `dirname("logs")` is empty — which was read as `/`. So
  `logs`, the default output directory, was reported as impossible to create
  whenever it did not exist yet: a wrong warning for every first-time user in a
  perfectly writable directory. It never fired in the repo because `logs/` already
  exists there. The effective ancestor for a relative path is now the working
  directory; genuinely unwritable paths still warn.

- **TOML types are no longer silently reinterpreted.** `int()` accepts more than
  it should: `cpus = true` became a **one-core** request (bool is an int subclass)
  and `cpus = 2.7` became 2 by truncation — the SM-9 family, except the value is
  changed rather than discarded. Both are now refused by name. An integral float
  (`2.0`) and a numeric string (`"4"`) still pass, since neither is ambiguous.

- **Pinned the no-false-claim invariant across every surface at once.** Three
  consecutive rounds found the same claim — "partition not on this cluster" when
  the list merely could not be read — in a place the previous fix had not reached.
  Rather than wait for a fourth, I swept every consumer of the unknown flag (three:
  the summary rows, the capacity message, and the wizard's memory default, which
  renders no claim and is correct under either reason) and added a test that
  renders *all* surfaces for an unreadable record and asserts none of them says
  "not on this cluster", plus the converse for a genuinely absent one so the
  honest claim is still made. Verified to have teeth by reverting one row to the
  old wording, which fails it. A companion assertion counts the consumers, so a
  new site reading that flag has to be reviewed for this claim.

- **Slurm's own explanation now reaches the user.** When `sinfo` fails it says
  why — `slurm_load_partitions: Unable to contact slurm controller (connect
  failure)` — and the code reported a generic "no Slurm, or sinfo failed",
  discarding the diagnosis sitting in a stream nobody read. That reason is now
  captured and quoted in the message, falling back to the generic wording when
  Slurm said nothing.

- **Two more rows stopped claiming a partition is absent when the list could not
  be read.** The previous round's `_unknown_reason` fix covered the capacity
  message, but the Queue and ETA rows keyed off the unknown flag alone, so an
  unreadable `sinfo` made them report `unknown — partition not on this cluster`
  about a partition that does exist. Both now distinguish the two reasons, as the
  capacity message does.

  Also confirmed that every remaining `_run_command` call site keeps its return
  code — the discarded one in the queue query was the last — and that each treats
  a failure as "could not ask" rather than as an answer.

- **A failed queue query no longer reads as an empty queue.**
  `stdout, _, _ = _run_command(["squeue", …])` discarded the return code, so a
  failed or timed-out `squeue` was indistinguishable from an idle partition and
  the summary reported `0 running / 0 pending` as a measurement. That is the
  report's cross-cutting root cause verbatim — *"a subprocess's error channel is
  not read"* — and SM-19's defect arriving through the failure path rather than a
  missing partition. The row now reads `unknown — could not read the queue`, and
  the tier-3 ETA guess (which is *derived from* the queue depth) no longer answers
  from a failed query, since that would invent a number twice over. The scheduler
  and free-capacity tiers are unaffected, because neither needs `squeue` —
  verified by breaking `squeue` alone (ETA still `~11h` from the scheduler) and
  then `squeue` plus `sbatch` (falls to `~5min (estimated from free capacity)`,
  honestly labelled).

- **A hung controller no longer freezes a run for nearly three minutes.** Six
  cluster-fact lookups run per invocation and every one is designed to fall
  through *silently* on failure — so at the default 30 s timeout, a dead
  controller froze a `--dry-run` for ~170 s collecting answers it would then
  discard. The advisory lookups now use a 10 s timeout: 20-100x the measured
  healthy latency (0.1-0.5 s each), so a slow-but-working controller is still
  answered, while a dead one is not waited on. Verified against stub binaries that
  never return — 100 s instead of 170 s, still emitting a correct script with the
  honest "could not be read" message.

  `fetch_partitions` deliberately keeps the full timeout: an empty partition list
  is handled, but it costs the user the picker *and* the limit checks, so waiting
  longer is the right trade there. That exclusion is pinned by a test, as is the
  fact that every advisory lookup reports a timeout as "could not ask" rather than
  as an empty answer — which is what makes the shorter timeout safe.

- **Cluster facts are queried once per run instead of twice.** A single
  `--dry-run` made **9 subprocess calls in 2.55 s**, three of them duplicates: the
  batch path's fatal checks and the shared site checks each asked for the
  partition name list, the caller's accounts and the QoS list. `sacctmgr show
  assoc` — the one the report singles out as *"slow enough on a busy controller to
  be worth skipping"* — ran twice, a cost introduced when the site checks were
  shared with the wizard. The six cluster-constant lookups (partition names,
  accounts, QoS, node features, `SelectType`, `MaxArraySize`) are now memoised per
  process; nothing request-specific is cached, so the ETA still reflects the
  request. Now **6 calls in 0.68 s**. Tests get an autouse reset, since they vary
  the mocked output and would otherwise read each other's answers.

- **The builder now normalizes memory when it emits the directive, not only in
  the CLI.** `sbatch --mem` requires an integer magnitude, so a fractional value
  that `validate_memory` accepts — `1.5G` — is refused by the controller with
  `Invalid --mem specification` (measured). `normalize_memory` existed for exactly
  that, but it was applied by the CLI and the wizard *before* calling the builder,
  so the emitted directive was correct only by accident of the caller: a library
  caller got `#SBATCH --mem=1.5G` and an unsubmittable script. Normalizing at the
  point of emission fixes that and makes the summary row agree by construction
  rather than by both layers happening to transform first. Idempotent, so the
  pre-normalizing callers are unaffected; the same directive now round-trips
  through `sbatch --test-only` successfully.

- **The summary rows for transformed fields show what Slurm will see.** The job
  name is sanitized, memory normalized, free-text values CR/LF-folded — and
  `job_summary_rows` read the raw answers, so it described the input rather than
  the directive. Invisible through the CLI, which pre-transforms, and wrong for a
  library caller. Pinned by tests comparing each row against the emitted
  directive.

- **The `Output directory` row no longer names a directory the job will not use.**
  The builder places only a *bare* filename inside `output_dir`; an absolute or
  directory-bearing `output_file` is left alone. So
  `--output-file /tmp/x.out --output-dir logs` wrote to `/tmp` while the summary
  said `Output directory: logs` — sending the user to an empty directory to look
  for their logs. The row now reads `logs (not used — output file has its own
  path)` when the flag has no effect, so an ignored flag is visible rather than
  silently overridden. A shared predicate backs the summary and the emitter, with
  a test asserting the predicate agrees with the emitted `--output=` directive for
  every path shape.

- **Pinned that `--print` and `--yes` produce the same script.** "Inspect it with
  `--print`, then run it with `--yes`" is the documented workflow, and nothing
  asserted the two agree. Verified against a stub `sbatch` across six feature
  combinations: the printed and submitted bytes differ only by the trailing
  newline `print()` adds (377 vs 376 bytes), which is the same shell script. Now
  pinned at the level that makes it true — `build_and_show` returning exactly
  `build_from_answers(answers)`, and the build being a pure function of the
  answers — so a future path that rebuilds the script differently fails the suite.

- **The wizard's live panel and the final summary now reach the same verdict on
  arrays.** Both call `validate_job_config`, but the live one omitted
  `max_array_size`, so an over-large `--array` drew nothing while stepping through
  the wizard and a warning at the summary — the same request judged differently by
  two surfaces. `MaxArraySize` is a cluster constant rather than a per-partition
  one, so it is fetched once per session (~20 ms) and only when an array has
  actually been entered, keeping the redraw subprocess-free for everyone who does
  not use arrays. A failed lookup is cached as "unknown" rather than retried on
  every keystroke.

- **An environment that will never be activated is no longer reported as if it
  were.** `--env-type none` is a documented choice that emits no activation line,
  so `--env myenv` alongside it was silently dropped — while the summary still
  read `Environment: myenv`. The only signal was a `logger.warning`, which no user
  sees. The row now reads `myenv (not activated — env_type none)` and a warning
  names the fix. A shared predicate backs both, so the summary and the emitter
  cannot disagree about whether an activation line exists — pinned by a test that
  builds a script per `env_type` and compares.

- **The summary now accounts for every directive the script carries.** SM-15 was
  the summary and the script disagreeing; nothing asserted the general property,
  and two directives had no row explaining them. `#SBATCH --nodes=1` is emitted
  for every job — the builder receives `opt("nodes", 1)` — while the summary read
  the raw answer and omitted the row, so a CLI run with no `--nodes` produced a
  script that pinned the node count and a summary that never mentioned it. Fixed
  by mirroring the builder's default. (The value is not an imposition: one node is
  Slurm's own default too. It just has to be visible, because the summary is what
  a user checks the script by.)

  `--output`/`--error` are likewise emitted unconditionally; the CLI and wizard
  both default the directory to `logs`, but a direct API caller omitting it got a
  script writing to the working directory and no row saying so. The row now says
  `(current directory)` rather than being absent.

  Pinned by a test that maps every directive slurmate can emit to the row that
  accounts for it, including each of the five GPU spellings, so a new directive
  added without a row fails the suite.

- **Declared Python support now matches what is verified.** The portability
  report's intro noted the packages are 3.14-clean while their classifiers stop at
  3.13 — a claim of *less* support than is true, and classifiers are what PyPI
  shows and what tooling filters on. `requires-python = ">=3.10"` already allowed
  3.14, so nothing was blocked; the metadata simply understated it. Added the
  3.14 classifier, plus a test that every version `requires-python` allows is
  declared, so the two cannot drift apart, and one asserting no removed or
  deprecated stdlib APIs (`distutils`, `imp`, `utcnow`, `getdefaultlocale`,
  `find_loader`, `pkg_resources`) appear in the source.

- **The wizard now has a test that it starts.** Every existing wizard test mocks
  `app.run`, so nothing exercised the real startup path — building the Application
  and composing the first frame. That left the *default* interface (bare
  `slurmate`) with no coverage of the one thing it must do, while these rounds
  changed its step defaults, its imports and one step's validator. A pty-driven
  smoke test now asserts the first frame renders with no traceback; verified to
  have teeth by injecting a `RuntimeError` into the step-default provider, which
  fails it, and reverting, which passes. Skipped where `pty` is unavailable, and
  the child is always terminated.

- **"Capacity limits NOT checked" no longer claims a partition is absent when
  Slurm simply could not be asked.** With no `sbatch`/`sinfo` on `PATH` the
  partition list comes back empty, every name falls through to the unknown
  record, and the message read `partition 'anything' is not on this cluster` —
  the false rejection the SM-4 restraint was written to prevent, reintroduced by
  the SM-20 fix. The record now carries *why* it is unknown, and an unreadable
  list says so: `this cluster's partition list could not be read (no Slurm, or
  sinfo failed)`. A genuinely absent name is still named.

- **A non-UTF-8 locale no longer corrupts the submitted command or crashes the
  bookkeeping.** Under `LC_ALL=C` the filesystem encoding is ASCII, so argv
  decoding turns a `--command` carrying UTF-8 bytes into lone surrogates. Both
  byte paths mishandled them:

  - `_save_submitted_script` wrote with strict UTF-8 and caught only `OSError`,
    so it raised an **unhandled `UnicodeEncodeError` after the job had already
    been submitted** — a queued job reported as a traceback.
  - `submit_sbatch` passed `errors="replace"`, which governs the **input**
    encoding too, so sbatch received a `?` per byte and ran a different command
    than the user typed — silently.

  Both now use `errors="surrogateescape"`, which reverses exactly what argv
  decoding did, so the bytes reaching sbatch and the saved copy are the user's
  originals. Verified with `od -c`: identical UTF-8 in both. The save also catches
  `UnicodeError`, since bookkeeping after a successful submit must report rather
  than raise.

- **`gpu_format = "gpus_per_task"` needs a task count, and now says so.**
  `--gpus-per-task` is per *task*, so Slurm needs a task count to resolve it: on
  its own it is refused with `Invalid generic resource (gres) specification`,
  while the same request plus `--ntasks-per-node` is accepted. slurmate offered
  the format without one, so one of the five `gpu_format` values emitted an
  unschedulable request when used alone. Found by piping the generated script for
  every format to `sbatch --test-only` — it passes SM-18's `SelectType` check,
  because the requirement is in the flag rather than the site. Now a named error
  pointing at `--ntasks-per-node` or a format that needs no task count.

- **An unbounded time limit no longer produces a confident cost estimate.**
  `--time=0` is documented Slurm for *no limit imposed*, and both estimators
  treated it as a zero-length job: `minutes <= 0` substituted a two-hour default,
  so a 48-core job with no time limit reported `Estimated CPU-hours: 96.0` — a
  specific number derived from an assumption nobody typed, for something
  unbounded. The row now reads `unbounded — no time limit`, for CPU-hours and
  GPU-hours alike.

  The check is *shape-based* rather than a list of spellings, because
  enumerating them missed `0-00:00:00` — which the controller accepts, as do `0`
  and `00:00:00`. And it deliberately excludes two neighbouring cases: an
  **absent** limit is not unbounded (the job takes the partition or site default,
  and 2 h is what the summary already shows for it), and an **unparseable** value
  is not unbounded either, since it parses to zero minutes and calling it
  unlimited would be exactly the SM-10 conflation of unknown with infinite.

- **Identical input now produces identical output.** A partition spanning several
  `sinfo` rows merged its GPU types through `list(set(...))`, and Python's
  per-process string-hash randomisation made the order differ between runs —
  measured at **four distinct orderings across eight runs** of the same input.
  That order is user-visible in the picker's `GPU:[a100,v100]` label and in the
  `not in partition list (…)` error, so the same cluster produced different text
  run to run and an error message could not be reproduced from a bug report. Now
  `sorted()`. Pinned by a test that runs fresh interpreters, since hash
  randomisation cannot be exercised within one process, plus a guard against the
  pattern reappearing anywhere in the source.

- **`--print` no longer emits a script it has just called an error.** The shared
  site checks were *reported* on that path and the script printed anyway, `rc=0`
  — the inverse of the silence problem: the tool states the artifact is wrong and
  then hands it over. An error-level issue is now fatal there, with `--force`
  overriding as it does for the partition/account checks. Warnings still print
  and still emit.

- **A missing module blocks the wizard as well as the batch path.** SM-13 asked
  for it to be fatal-with-`--force`, and the batch path implemented that while the
  shared helper emitted a warning — so the wizard would have submitted a job the
  non-interactive path refuses. Levels aligned; the wizard still offers "go back
  to edit" rather than exiting.

- **The command body can no longer smuggle `#SBATCH` directives.** Slurm stops
  reading directives at the first line that is neither blank nor a comment, and
  the command is emitted *after* the directive block — so a `#SBATCH` line at the
  start of the body is still inside the directive region and takes effect.
  Measured: `--command '#SBATCH --qos=INJECTED'` drew `Access/permission denied`
  from the controller, its answer for an invalid QoS, so the directive was obeyed
  — unvalidated, absent from the summary, and bypassing the managed-flag check
  that covers `--custom-sbatch`. Now a named error.

  Only the *leading* run is examined, which is what makes it safe to enforce: a
  `#SBATCH` inside a heredoc that writes a nested script is preceded by a real
  command, so Slurm has already stopped parsing and the line is inert. Both
  spellings Slurm accepts (`#sbatch`, `# SBATCH`) are caught.

- **`--print` now runs the shared site checks, not just the two limit
  reporters.** Wiring only the latter is how the check above reached `--dry-run`
  and `--yes` but not `--print`; routing that mode through the same helper closes
  the class rather than the instance.

- **Pinned the cross-mode invariant.** Three rounds of findings were the same
  shape — a check present on one path and absent on another (`run_batch` vs the
  wizard, `--dry-run` vs `--print`) — so the invariant is now a test rather than
  another audit: a representative bad value must be reported in `--print`,
  `--dry-run` and `--yes` alike, and a clean request must be reported by none of
  them. Hermetic under `SLURMATE_MOCK`, so `--yes` never reaches a controller. The
  one intentional difference is also pinned: `--print` reports on stderr, keeping
  stdout script-only.

- **A hand-edited script is now checked as bytes, not via the stale answers.**
  After "Open in editor" the script holds the user's edits while `answers` still
  describes the generated one, and the pre-submit guard validated `answers`. So an
  edit that *introduced* a bad partition passed the guard, and an edit that
  *fixed* one was still blocked — with the only offered remedy being "go back to
  edit answers", which discards the fix. Both directions check something other
  than what would be submitted, which is SM-15's defect reached through the editor.
  New `check_script_with_scheduler()` pipes the edited script to
  `sbatch --test-only` and reports Slurm's own refusal; nothing is submitted, and
  "could not ask" (no `sbatch`, unreachable controller) still never renders as
  "cannot run". The summary also now says, once, that it describes the generated
  script rather than the edits.

- **The wizard's array step validates as you type.** SM-22's asymmetry, in the
  interactive path: cpus, memory, mem-per-cpu, time, nodes, ntasks and gpus all
  had step-level validators and the array spec was the one resource field left as
  free text, so `10-1` was only caught later at the summary. It now uses the same
  `validate_array_spec`, and accepts the step's own subtitle example
  (`1,3,5-7%4`) plus the empty value, since the field is optional. A test asserts
  *every* resource step has a validator, so adding a field without one fails.

- **The wizard's memory default is derived from the partition, not the literal
  `16G`.** SM-7 was about exactly that number — "the built-in fallback is a
  number, not a measurement" — and its fix landed on the batch path while the
  wizard's memory step kept `default="16G"`. That is the interface that *shows*
  the value pre-filled for the user to accept, so it was the worst place for it:
  on the 8 GB node SM-7 describes, the wizard offered an unschedulable default.
  Now sized from the chosen partition (48-core/180 GB with 8 cores gives `30G`;
  the 4-core/8 GB node gives `2G`), with the literal kept only when no partition
  is chosen yet or it cannot be resolved. An explicit config value still wins,
  since the user asked for it, and clearing the field reverts to the *derived*
  default rather than the literal.

- **The wizard now discloses which values came from a config file.** SM-8's
  disclosure was set inside `run_batch`, so it never fired for a wizard run —
  even though the wizard is what *prefills* from `.slurmate.toml`, which makes it
  the path where "values you did not type" is most likely and the disclosure most
  needed. Recorded on the way *out* of the wizard rather than at prefill time, so
  a field the user then edited is no longer credited to the file, and the string
  and native forms of a number (`8` vs `"8"`) count as the same answer rather
  than reading as an override.

- **The wizard now gets every cluster check the batch path has.** All of them —
  partition/account/qos/constraint membership, module existence, the
  `gpu_format`/`SelectType` match, array-spec shape, and custom-flag conflicts —
  lived only on the non-interactive path, where they are fatal before a script
  exists. So the **wizard**, which is the default interface *and* offers "Enter
  partition name manually…", accepted silently every value the batch path rejects
  outright. `site_check_issues()` runs them where both paths meet, reporting
  rather than exiting so the wizard can offer "go back to edit"; the errors also
  feed the pre-submit guard, so a bad name blocks submission instead of only
  printing. The batch path keeps its fatal pre-script checks, so `--print` still
  cannot emit a rejected name.

  Levels are preserved rather than flattened: a missing module stays a *warning*
  (a hierarchical module tree makes absence evidence, not proof) while a
  nonexistent partition is an *error*. A probe failure yields nothing, so a broken
  `sinfo` cannot block a job.

- **Demo mode is labelled in-band, and discoverable.** `SLURMATE_MOCK` fabricated
  the partition list, its limits, the queue depth and the ETA with **no marker
  anywhere** and appeared nowhere in `--help`, so the realistic way to reach it was
  a stale `export`, a CI wrapper or a container image rather than a deliberate
  choice — synthetic data shaped exactly like measurement. The summary title now
  reads `Summary — SIMULATED (SLURMATE_MOCK)`, the queue row carries
  `(simulated)`, and one warning states that the partition list, limits, queue
  depth and ETA are all demo data. The markers are in the fields themselves, so
  they cannot scroll away from the numbers they qualify.

- **New `--demo`** sets it, so the deliberate path is the documented one, and both
  `--demo` and `SLURMATE_MOCK` now appear in `--help`.

- **`--print` now runs the capacity checks too, on stderr.** It returned before
  any of them, so the mode most used in scripts and CI was the one that emitted an
  unschedulable script in silence: a 999-CPU / 9999 GiB request that `--dry-run`
  warns about twice produced **zero bytes on stderr**. The name checks
  (partition/account/qos/modules/gpu-format/array shape) were already on the batch
  path and did cover `--print`; the *limit* warnings lived behind the summary and
  did not. They now go to stderr, so stdout stays script-only and redirecting it
  still yields exactly the script — pinned by a test asserting the emitted script
  is byte-identical whether a warning fired or not.

- **Pinned the GRES arithmetic the `resources`-tier ETA rests on.** No
  behavioural change — the code was already right — but `_sum_node_gpus()` had
  *zero* test coverage and the `(IDX:…)` suffix Slurm puts on `GresUsed` appeared
  nowhere in the fixtures, which is the same idealised-fixture gap round 21 was
  about. Now pinned across every real spelling (count-only, typed, non-contiguous
  index lists, multi-model, the `gres/gpu:` prefix, `shard` exclusion), and
  `sinfo_nodes.txt` carries real GPU-node rows: one with 2 of 4 GPUs allocated,
  one with all 4, plus `drained` and `idle*` nodes that have free cores and are
  still unschedulable. That matters more now that the ETA row labels this tier
  "estimated from free capacity" — the label is a claim about this subtraction.

- **The ETA now says where its number came from.** `fetch_queue_eta` returns
  `source` naming which of its three tiers answered — its docstring says "so the
  caller can qualify what it shows" — and the renderer dropped it. So Slurm's own
  backfill placement and the last-resort queue-depth heuristic, which returns a
  flat 300 seconds for *any* empty queue, rendered identically as `~5min`. The
  scheduler tier stays unadorned; the others self-label
  (`now (estimated from free capacity)`), and an unrecognised source is left
  unqualified rather than given an invented provenance.

- **A partition this cluster does not have reports `unknown` rather than zero.**
  `squeue -p <nonexistent>` returns no rows, which was rendered as a real
  `0 running / 0 pending`. Together with the previous two entries this closes all
  three unfounded claims a single screen made about an unresolvable partition: the
  queue depth, the absent capacity warnings, and the ETA. A positive refusal from
  Slurm still outranks "unknown", since a named reason beats an absence.

- **An unresolvable partition now says the limits were not checked.** The
  partition/account/qos names are rejected outright, but `--force` deliberately
  reaches the summary with a partition this cluster does not have — and there
  every capacity check compared against an empty record and stayed *silent*, so a
  999-CPU / 9999 GiB request looked unremarkable. That inverts the failure mode a
  user expects: ask for 999 CPUs on a real partition and you are warned; misspell
  the partition and you are not, so the less valid request produces the more
  reassuring screen. The fallback record is now marked unknown, and a concrete
  request against it gets `Capacity limits NOT checked: … validated for shape
  only`. Nothing is claimed when nothing concrete was requested.

- **With no `--partition`, the summary now describes the partition Slurm will
  actually use.** Slurm falls back to the site default, and slurmate already knew
  which that was — sinfo's `*` marker, which it uses for near-miss suggestions.
  Treating the partition as *unknown* instead produced two confidently wrong
  figures: `Queue: 0 running / 0 pending`, straight from `squeue -p ""`, for a job
  landing in a partition with hundreds of jobs; and SM-7's "this cluster's node
  memory is unknown" fallback inventing `16G` when the default partition's memory
  is perfectly well known. Measured on the development cluster: the same run now
  reports `257 running / 825 pending` and derives `30G`.

  The default is used for the *derived* figures only — limits, queue depth, ETA,
  default memory. **No `--partition` directive is added**: emitting one the user
  did not type is what SM-15 was about, and a site's default can differ per user
  or account. The "Missing recommended fields: Partition" warning still fires,
  and the summary says which partition the figures describe. A site whose `sinfo`
  marks no default resolves nothing and claims nothing.

- **`gpu_format` is validated against the cluster's `SelectType`.** `--gpus` and
  `--gpus-per-task` are **cons_tres-only**: under `select/cons_res` or
  `select/linear` Slurm's *parser* refuses them ("Requested GRES option
  unsupported by configured SelectType plugin") cluster-wide, on every partition,
  so no partition choice avoids it. slurmate never read `SelectType`, and nothing
  in its output distinguished a working format from a fatal one.

  The exposure is a config file rather than a flag: `gpu_format` is a
  `.slurmate.toml` key, so a setting that is correct on a cons_tres site produces
  an unsubmittable script on a cons_res one *without the user typing anything*.
  The two clusters in this audit differ on exactly this value — one runs
  `select/cons_tres`, the other `select/cons_res` — which is what makes it a real
  portability failure rather than a theoretical one.

  Now a named error with the working alternatives (`gres_type`, the default, and
  `gpus_per_node`, which both parse everywhere), `--force` to override for
  another cluster, and checked only when GPUs are actually requested. An
  unreadable or unrecognised `SelectType` stays silent — failing open to the
  default is already the safe behaviour, and an unreadable `scontrol` must not
  present as "your GPU syntax is wrong".

- **The reported log path resolves every pattern it can, and stops guessing at
  the ones it cannot.** The submit report built the path with chained
  `str.replace`, which got two things wrong. `%x` (job name) and `%u` (user) were
  left literal although slurmate knows both, so the `tail -f` hint pointed at a
  file like `run-%x-12345.out` that does not exist. And `%%`, which Slurm treats
  as a literal percent, was mis-substituted: `pct-%%j-%j.out` was reported as
  `pct-%12345-12345.out` when Slurm writes `pct-%j-12345.out`. Both sent the user
  to a filename that was never created.

  `expand_log_pattern()` now does a single pass, so `%%` is consumed as a unit,
  and returns the patterns it could *not* resolve — `%a` per array task, `%N`,
  `%n`, `%t`, `%s` per node/task/step. When any remain, the report offers
  `ls <dir>` with a note about what varies instead of a `tail -f` on a path Slurm
  will never write. An unknown letter is left untouched rather than dropped.

- **A partition that is itself down is now reported.** `sinfo %a` was parsed into
  the partition record and never consulted. A partition's own state is a
  different fact from its nodes': it can be UP with every node dead (which the
  SM-1 fix catches) or DOWN with a hundred live nodes, which nothing caught.
  Slurm accepts a job for a down partition and then never starts it — the
  "queues forever with no indication why" failure SM-1 was filed about, one level
  up. The development cluster has a live example: `test` is `State=DOWN` with 177
  nodes in `mix`, so `nodes_up` is high and the node-level check cannot fire.
  `down`, `drain` and `inact` all warn now; an unknown state stays silent, and a
  warning rather than an error because partitions come back, which is the same
  reasoning SM-1 used for drained nodes.

- **`DenyQos` is read, so a deny-list site is not offered forbidden QoS.** Slurm
  expresses a partition's QoS ACL two ways and a site picks one: an explicit
  `AllowQos` list, or `AllowQos=ALL` plus a `DenyQos` exclusion list. Only the
  allow side was read, so on a deny-list site the `ALL` sentinel expanded to
  every QoS on the cluster — *including the ones that partition forbids*. That is
  the same defect as offering partitions the user holds no association for. New
  `fetch_qos_acl()` returns both sides and the deny list is subtracted, whether
  the allow side is `ALL` or an explicit list (Slurm gives deny precedence too).
  `fetch_qos_for_partition()` is kept as the allow-side accessor.

- **A clock disagreement between login node and controller no longer reads as
  "now".** `sbatch --test-only` reports the placement in the *controller's* local
  time; slurmate compared it against the *login node's* clock and clamped a
  negative result to 0. So a timezone difference between the two — a real
  multi-site/federated arrangement — turned into a confident `ETA: now` for a job
  starting hours later. A gap of up to two minutes still means "now" (Slurm says
  "start immediately", plus the latency between asking and parsing); beyond that
  it is evidence the two clocks are not the same clock, and the ETA is now
  reported as unknown so the caller falls through to its own estimate. SM-5's
  defect, reached from a different direction.

- **Every invocation crashed when no home directory could be resolved.**
  `Path.home()` raises `RuntimeError` when `$HOME` is unset *and* the uid has no
  passwd entry — which is `sbatch --export=NONE` (standard Slurm, and a
  cluster-wide default at some sites) on a node whose name service does not
  resolve the user. The config search list was built **eagerly**, so that aborted
  the tool before any flag was acted on, including runs with a perfectly good
  project-local `.slurmate.toml` in the job's working directory: the crash
  happened constructing the list that would have found it. The home candidate is
  now lazy and optional, `Path.cwd()` is guarded the same way, and
  **`XDG_CONFIG_HOME` is honoured** — the documented location, previously
  ignored, and the way to keep a global config in an environment with no home at
  all.

  Its silent sibling is fixed too: `os.path.expanduser` does *not* raise in that
  environment, it returns the string unchanged, so `--output-dir ~/logs` would
  put a job's log in a relative directory literally named `~`. A log path that
  still starts with `~` is now reported.

- **A `⚠` in a warning no longer aborts the run under a non-UTF-8 locale.** A
  *valid* non-UTF-8 locale (`en_US` is latin-1; el7 has no `C.UTF-8`) made any
  non-encodable character raise `UnicodeEncodeError` mid-print, killing the run
  and truncating the summary at ~70%. rich picks a safe box set for its own
  glyphs but does not transcode application text, so slurmate's own markers went
  straight to the encoder. Every affected site was a *warning or error* path, so
  the tool was least robust exactly when something had already gone wrong — the
  observed failure destroyed the "Missing recommended fields" advice the user
  needed.

  Fixed in two layers. A codec error handler transliterates the typography
  slurmate writes (em dash → `-`, ellipsis → `...`, `⚠` → `!`) and *escapes*
  anything unknown rather than dropping it — a job name or module carrying
  characters the terminal cannot encode is data, not decoration, and `?` would
  silently destroy it. And the status markers now resolve through a table with
  ASCII fallbacks, chosen automatically when the encoding cannot carry them.

- **New `--ascii`** (and `SLURMATE_ASCII=1`) forces plain markers on any
  terminal. slurmate was the only package in the suite with no way to ask for
  plain output.

- **`--custom-sbatch` can no longer duplicate a directive slurmate manages.** A
  custom `--partition=caslake` emitted a *second* `#SBATCH --partition` line;
  Slurm honours the last, so the job ran on the custom partition while the
  summary's `Partition` row and the queue-depth and ETA figures derived from it
  all described the managed one. A custom `--partition`/`--account` also routed
  straight past the cluster validation that exists for exactly those two values.
  Of the three available outcomes — reject, reconcile, silently disagree — it was
  doing the third. Now:

  ```
  ✗ Error: --custom-sbatch carries --partition, which slurmate manages. Use --partition instead.
  ```

  Refused rather than reconciled because for every directive in that set slurmate
  already has a flag that is validated and reflected everywhere. The set is
  deliberately narrow: `--mem`/`--mem-per-cpu` (custom wins, auto suppressed),
  `--constraint`/`-C` (merged into one directive) and `--output`/`--error`
  (de-duplicated) are *reconciled* and stay allowed — refusing those would undo
  behaviour this package is relied on for, including the merged `-C bigmem` the
  portability report asked to keep. Reported on both paths, so the wizard's
  summary and the pre-submit guard see it too, not just the batch path.

- **The global and project config files are merged, not chosen between.**
  `load_config()` was first-file-**wins**, which made a project config
  *destructive*: a one-line `.slurmate.toml` naming this cluster's partition
  discarded the global `account`, `memory`, `time_limit` and `modules` entirely.
  Each loss failed differently and all of them silently — a rejected or
  mischarged job, an OOM kill, a twelve-hour run truncated at two, and an
  environment that never loaded (SM-13's silent-success shape, reached through
  config precedence instead of a bad module name). The trigger was the *most
  natural* use of the feature, and the workflow `.slurmate.toml` exists to
  support.

  Now the global file is read first and the project file overlaid on top, so the
  more specific file wins **per key** — which is what the search order always
  implied, and what git, ssh, pip, cargo and npm all do. Each file is named on
  stderr with the keys it actually *won*, so an overridden global value is not
  claimed by the file that lost it and the precedence is visible rather than
  inferred. A file that cannot be parsed is now reported and skipped without
  taking the other one down with it.

- **The config file that supplied the defaults is named.** A `.slurmate.toml`
  travels with a project into git and onto whatever cluster it is next checked
  out on, so a partition, account, CPU count and memory size from another site
  could arrive without the user knowing the file existed — and nothing in the
  output mentioned a config file at all. Now disclosed on stderr at load, and in
  the `--dry-run` summary, listing only the keys no flag overrode:

  ```
  slurmate: using defaults from ./.slurmate.toml: partition, account, cpus, time_limit
    Defaults from ./.slurmate.toml: partition, account, time_limit (flags override the file).
  ```

  `--print` keeps stdout script-only; the disclosure is on stderr. The wrong
  *values* were already caught by the partition/account validation below — a
  config-supplied `caslake` on a cluster without it is a hard error, not a
  silent script.

- **A failed `module load` or environment activation now aborts the job.** This was the
  one cross-cluster error that survived submission: `sbatch` accepted the script, the job
  ran, `module load` printed to stderr, the body executed anyway, and Slurm recorded
  **COMPLETED, exit 0** with the environment absent. The worst outcome is not a confusing
  failure later — it is a run that quietly proceeds against whatever toolchain was already
  on `PATH` and produces results the user believes came from the module they asked for.
  Every setup line now carries a guard:

  ```bash
  module load cuda/11.8 || { echo "slurmate: module load cuda/11.8 failed; aborting" >&2; exit 1; }
  ```

  `module load` exits 1 on a missing modulefile and 0 on success, so the guard fires
  exactly when it should. The same defect existed for environment activation and is
  guarded identically — `conda activate`, the `mamba activate … || conda activate …`
  fallback chain (guard after both), and `source <venv>/bin/activate`. The source
  comments already described that failure mode for mamba ("the script keeps going, so the
  job silently runs in whatever interpreter it inherited") without making it non-zero.

  This is a deliberate behaviour change: a job that previously "succeeded" with the wrong
  environment now fails fast. It also covers the case generation-time validation cannot —
  a module that exists when the script is written and is retired before the job runs.

- **`--memory 0` is accepted — it is Slurm's whole-node idiom, not an invalid
  size.** `validate_memory()` rejected a zero magnitude in every unit,
  deliberately ("0G/0M are not valid sizes"). That was wrong: `--mem=0` is
  documented Slurm for *all the memory on the node*, and `0`, `0K`, `0M`, `0G`
  and `0T` were each measured accepted by a live controller. The rejection left
  no way to express that request at all, since `--memory ''`/`none` omits `--mem`
  entirely and gets the *site default*, which is a different thing. Every zero
  spelling now normalizes to the documented bare `0` rather than `0M`, which
  reads like a request for nothing, and a zero request correctly triggers no
  "exceeds partition limit" warning. Reverses the earlier P3-11 finding, whose
  test asserted the opposite.

- **`--array` is shape-checked like `--time` and `--memory`.** It was the one
  value validated for nothing, so `--array 10-1` produced a script the controller
  refuses with "Invalid job array specification". The grammar was calibrated
  against a live controller rather than guessed — accepted: `5`, `1-10`, `0-9`,
  `1,3,5`, `1-10:2`, `1-10%4`, `1-5,10` and, unexpectedly, a bare `%4`;
  rejected: `10-1`, `1-10:0`, `1-`, `-5`. An intuition-built validator would have
  rejected `%4`.

- **The submitted job ID is parsed, not assumed.** `sbatch --parsable` prints one
  line, but a site's sbatch *wrapper* does not: a policy notice or MOTD on stdout
  was prepended to the id, and the whole banner then travelled into the
  "Job ID:" line, the `squeue -j` / `scancel` hints the user copies, and the
  saved script's filename. New `parse_submitted_job_id()` matches only a line of
  the expected shape (`<id>` or `<id>;<cluster>`), and returns nothing rather
  than guessing when none is present — a banner can itself contain digits, so
  scraping the first number out of arbitrary text would substitute one wrong
  answer for another. When the id genuinely cannot be read, the submission is
  still reported as the success it was, with sbatch's raw output shown and the
  hints suppressed instead of built from a fabricated id.

  Worth noting this module already guarded the same hazard for JSON
  (`_extract_first_json`, which exists because "a login shell may print a banner
  before the JSON"); the submit path was the one place that still trusted stdout
  to be exactly one token.

- **The ETA probe now hands Slurm the real script instead of a rebuilt argv.**
  The old probe reconstructed an `sbatch` command line from the same fields the
  builder reads, which duplicated the builder and kept drifting — every field the
  reconstruction forgot produced a confident ETA for a job Slurm refuses.
  `--array` was missing (an over-large array read `~22h`), then `--constraint` (a
  bogus feature read `~21h`), and it rewrote *every* `--gpu-format` choice as
  `--gres`, which is a different request on a count-only-GRES site. It also
  ignored `--custom-sbatch` flags such as `--exclusive` entirely. Piping the
  generated script to `sbatch --test-only` cannot drift, and it is what the
  portability report asked for in the first place: "run `sbatch --test-only` on
  the generated script and surface whatever Slurm says". The argv path stays for
  callers that have no script yet (the wizard's live preview), and both paths now
  share one output reader so they cannot disagree about what Slurm said.

- **`--custom-sbatch --exclusive` now explains itself.** The one flag whose job
  is passing *other* flags through failed on its most natural invocation, because
  argparse reads a value starting with `-` as the next option — with a generic
  "expected one argument" that named neither the cause nor the fix. Now:

  ```
  ✗ Error: --custom-sbatch --exclusive: a value starting with '-' must use the '=' form
    Use: --custom-sbatch='--exclusive'
  ```

  Diagnosed rather than silently repaired: auto-rewriting the pair would make
  `slurmate --custom-sbatch --print` swallow a real slurmate flag as an sbatch
  one, which is a silent wrong answer in place of a loud error. The check fires
  only when the value starts with `-` **and contains no space** — argparse
  already accepts `-C bigmem` and `--comment="my run"`, and rejecting those would
  have broken the multi-flag form the report exercised. That boundary is pinned
  by a test against argparse itself.

- **GPUs now have a per-node limit check.** `sinfo %G` carried the count all
  along (`gpu:4`), but nothing parsed it, so GPUs were the one advertised
  resource with no limit warning: `--gpus 99` on a 4-GPU partition produced a
  script and said nothing. Partitions now carry `gpus_per_node`, parsed from
  every real GRES spelling — count-only (`gpu:4`), typed (`gpu:a30:4`),
  socket-annotated (`gpu:a100:4(S:0-1)`), multi-model (`gpu:a100:2,gpu:v100:2`
  sums to 4, since either model satisfies the ask) — and ignoring `shard`/`mps`,
  which are slices of a GPU rather than another one. Also wired into
  `capacity_refusal()`, and soft on a heterogeneous partition for the same reason
  cpu/memory are.

- **`--constraint` is validated against the cluster's node features.** A feature
  is as site-specific as a partition name and Slurm refuses a bad one outright
  ("Invalid feature specification"), but it was emitted unchecked. Now a named
  error with the cluster's feature list, `--force` to override. Checked against
  the cluster-wide set rather than the partition's, because naming a feature that
  exists elsewhere is a much less likely mistake than naming one that does not
  exist at all — and **only when the constraint is a single plain name**: Slurm's
  grammar has `&`, `|`, `!`, `*N` and `[…]`, and a set-membership test would
  reject valid expressions.

- **The ETA probe now passes `--constraint` too.** Without it, `--dry-run`
  reported `~21h` for a job Slurm refuses; it now reads
  `never — Invalid feature specification`, while a real feature still gets a real
  estimate. Same omission the array spec had.

- **The ETA no longer guesses when `sbatch` is unreachable but the request
  visibly cannot fit.** `_scheduler_verdict()` handles the case where Slurm
  refuses, but with no `sbatch` on `PATH` the estimate fell through to the
  queue-depth heuristic and printed a confident `~7min` on the same screen as
  `⚠ CPUs (999) exceeds partition limit (48 per node)`. New `capacity_refusal()`
  gives the ETA a second, scheduler-independent source — the partition's own
  figures, which the warnings were already reading — and it now says
  `never — no node in 'caslake' has 999 cores`.

  Two boundaries make this safe rather than a new confident wrong answer. It runs
  **only when the scheduler stayed silent**: if Slurm placed the job, Slurm knows
  better than advertised capacity does. And on a **heterogeneous** partition the
  cpu/memory figures are floors — `sinfo` printed the smallest node — so those
  never refuse; a bigger node may well take the job. Node counts, array indices
  and the partition time limit are exact, so they refuse even there. Verified
  live in all four combinations.

- **Fixed a command-substitution hole in the abort guard itself.** The guard's
  message interpolated a user-supplied module or environment name into a
  *double-quoted* shell string, and double quotes still perform command
  substitution — so `--modules '$(cmd)'` (or a backtick form) executed `cmd` at
  the moment the guard fired. The name reaching that point can come from a
  `.slurmate.toml` committed to a repo, which is the same carried-config path as
  SM-8. The whole message is now `shlex.quote`d, making it inert text; the
  `module load` argument was already quoted, so the command itself was never
  restructurable.

  Also pinned the structural property this rests on: every guarded line is a
  top-level `||` list, never a pipeline or a `( … )` subshell — where `exit`
  would end only the subshell and leave the job running. Verified by executing
  the generated script under a stub `module` that fails (rc=1, body never runs)
  and one that succeeds (rc=0, body runs).

- **An unknown module is now fatal at generation, not just a warning.** Same treatment as
  an unknown partition: rc=1 with empty stdout, `--force` downgrades to a warning, and it
  runs before any script exists so `--print` is covered.

- **`module load` names are checked against the cluster.** Modules are the most
  site-specific thing in a generated script after the partition, and they fail
  *late*: the job queues, starts, and only then dies on `module load`. Nothing
  checked them. The common case is a module that exists at a different version,
  so that is what the message answers:

  ```
  ⚠ Warning: module 'python/3.99' not found on this cluster; 'python' is available as:
    python/2.7, python/3.8.0, python/3.11.5+gcc-13.2.0, python/3.11.9
  ```

  Warnings rather than errors, because a hierarchical module tree only exposes
  part of itself at a time, so absence is strong evidence and not proof — and
  silent when there is no module system to ask. Two implementation notes worth
  recording, both of which are ways to get this wrong: the answer arrives on
  **stderr** (stdout carries shell code for the caller to `eval`, so a
  stdout-only read reports every module on the cluster as missing, and both a hit
  and a miss exit 0), and the query must go to `$LMOD_CMD` /
  `$MODULESHOME/bin/modulecmd` directly — `bash -lc 'module -t avail'` returns
  the same answer but takes ~10 s on a real login node against ~30 ms.

- **`--output`/`--error` directories are checked before submit.** The log path is
  the most cluster-specific value there is — every site mounts its scratch
  somewhere else — and Slurm kills a job outright when it cannot open the file.
  The failure was invisible twice over: nothing checked before submit, and the
  `os.makedirs` attempt inside `submit_sbatch` logged its `OSError` at *debug*
  level and submitted anyway. Now named, with the nearest existing parent so the
  reason is visible. Also a warning and deliberately so: a path can be unwritable
  from the login node and perfectly valid on the compute node — the test
  cluster's own `/tmp` is node-local, which is exactly that case.

- **`--qos` is validated against the cluster too.** The partition/account check
  covered two of the three names Slurm resolves against its own database;
  `--qos` was emitted unchecked, so a QoS carried from another site produced a
  complete script with `rc=0` and an "Invalid qos specification" from the
  controller later. Now the same named error with near-miss suggestions, the
  same `--force` downgrade, and the same silence when `sacctmgr` cannot be read.
  Existence only — whether a QoS is *permitted on a given partition* is set by
  `AllowQos`/`DenyQos`, and checking that would reject valid combinations on a
  site that uses `DenyQos`.

- **`--array` is checked against the site's `MaxArraySize`.** Another hard site
  limit that differs wildly — Slurm's default is 1001, the development cluster
  is configured at 65533 — so `--array 1-5000` is fine on one cluster and
  refused on the next with "Invalid job array specification". A warning names
  the local limit, and the spec parser reads `1-10`, `0-9:2`, `1,3,5`, `1-5,10`
  and the `%N` throttle suffix (which bounds concurrency, not the index).
  `scontrol show config` is queried on the CLI path only and only when an array
  was requested: `validate_job_config` runs on every wizard keystroke and stays
  subprocess-free, so the limit is passed in, and an unreadable value stays
  `None` rather than becoming a claim.

- **The ETA probe now includes the array spec.** Without it, `--dry-run`
  reported `ETA: ~22h` for an array Slurm refuses outright — the SM-5 defect in
  a narrower case. It now reads `never — Invalid job array specification`, from
  Slurm's own refusal, while a valid array still gets a real estimate.

- **A node count over the partition's size is warned about.** `--dry-run`
  advertises limit warnings and had them for CPUs, memory and time but not
  nodes, so `--nodes 9999` on a 190-node partition said nothing. Unlike the
  CPU/memory figures this one is not softened for heterogeneous partitions: a
  count is exact, and a bigger node cannot satisfy a request for more nodes.

- **GPU-type detection no longer returns a site's node-class tag as a GPU
  model.** On a partition whose GRES is count-only (`gpu:1`), the model is mined
  from node features, and the last-resort scan returned whatever appeared first
  that it had not thought to exclude. On real nodes reading
  `tc,e5-2670,160G,ib,m2090,gpu,ibspine-g20` that was **`tc`** — the site's
  node-class tag, carried by unrelated partitions — producing
  `--gres=gpu:tc:1`, which Slurm refuses, while `m2090` was never offered. The
  scan now requires a token *shaped* like a model (letters then 3+ digits) and
  rejects CPU designations (`e5-2670`, `x5650`, `l5520`, `gold-6148`) and fabric
  topology labels (`ibspine-g20`); the Fermi/Kepler/Maxwell families
  (`m2090`, `k20m`, `k40s`, `m40`, `m60`, …) were added to the known-model list,
  since none of them satisfies the shape rule. `l5520` is the instructive case:
  the NVIDIA L family is L4/L40/L40S, so a four-digit `l` token is a Xeon that
  matched the GPU shape rule by coincidence. When nothing is identifiable the
  answer is now no type at all, which is right — a wrong `--gpu-type` is worse
  than none, because nothing prompts the user to check it.

- **`infinite` is unbounded, not unknown.** A partition's `TIMELIMIT=infinite`
  parsed to the same `None` as "could not read that", so the time-limit check
  was skipped rather than satisfied — and the two cases were indistinguishable.
  Now `math.inf` vs `None`, so an unbounded partition *affirms* the request
  while an unreadable one stays silent. This was never site-specific: **all 87
  partitions on the development cluster are `infinite` too**, so the check had
  been inert there from the start.

- **A `+` on `sinfo`'s `%c`/`%m` means the figure is a floor, not a ceiling.**
  Slurm appends it when the nodes in a group differ, and the number printed is
  the *smallest*. Treating it as the maximum inverted the meaning: a 40-CPU
  request against a partition reading `28+` drew "exceeds partition limit
  (28 per node)" when a larger node in the same partition may well take it.
  Partitions now carry `heterogeneous`, and the CPU/memory warnings say
  `exceeds the smallest node in this partition (48 per node); nodes differ`
  instead of asserting a bound. Also not site-specific — 14 of the development
  cluster's 87 partitions are heterogeneous, including its two busiest.

- **The test fixtures were idealised, which is why the three above got through.**
  `sinfo_partitions.txt` had clean `HH:MM:SS` limits, bare integers and typed
  GRES — none of `infinite`, `+` suffixes or count-only `gpu:N` appeared in any
  row, so no fixture-driven test could ever exercise them. The fixtures now
  carry all three, plus a node-features row in the real shape (class tag and CPU
  model ahead of the GPU, fabric label after). `sinfo_gputypes.txt` is also
  partition-scoped now: the real call is `sinfo -N -p <part>`, and a fixture that
  answered every partition with every node's features could not show a
  partition-scoped detection bug.

- **Config keys accept their CLI spellings, and unrecognised keys are
  reported.** The flag is `--time`, the key was only `time_limit`, and the
  natural translation was dropped in silence: `time = "36:00:00"` produced
  `#SBATCH --time=02:00:00` — a 36-hour run silently truncated to the two-hour
  default, which kills it mid-flight. `time` and `array` are now accepted as
  aliases, as is any dashed form (`job-name`, `mem-per-cpu`, …), and anything
  outside the recognised set gets a named warning with the likely intent
  instead of vanishing:

  ```
  slurmate: ./.slurmate.toml: unknown key 'partitions' — did you mean 'partition'?
  slurmate: ./.slurmate.toml: ignoring unknown section '[job]' — put keys at the top level or under [defaults]/[slurmate]
  ```

  When both a key and its alias are set, the real key wins in either order and
  the alias is reported as ignored. A config parser that silently discards what
  it does not understand turns a typo into a wrong job.

- **`--partition` and `--account` are validated against the live cluster.**
  Previously a partition from another site produced a full script and `rc=0`
  with no warning — the exact failure the tool exists to prevent, since the
  value of generating an sbatch script is that it is correct *for the cluster
  you are on*. Five midway3 partition names (`caslake`, `amd`, `test`,
  `beagle3`, `gpu`) each generated a clean script on a cluster that has none of
  them. Now:

  ```
  ✗ Error: no partition 'caslake' on this cluster.
    Did you mean: broadwl (default), build, bigmem2?
    This cluster's partitions: broadwl, build, bigmem2, gpu2, ... (+4 more)
    Pass --force to generate the script anyway (e.g. for another cluster).
  ```

  Accounts are checked the same way against the caller's `sacctmgr`
  associations. Validation is against `sinfo -a` (hidden partitions included),
  so a hidden-but-submittable partition is not rejected, and it stays silent
  when the cluster's lists cannot be read at all — an unreadable `sinfo` must
  never present as "your partition doesn't exist".

- **`--force`** — downgrades those checks to warnings, for the legitimate case
  of writing a script to carry to another cluster. The default just is not
  silent.

- **A terminal guard on the wizard.** `slurmate | cat` used to hang forever
  (killed at 20 s in testing): prompt_toolkit warned `Input is not a terminal
  (fd=0)`, slurmate rendered the wizard anyway, and it blocked on input that
  could not arrive. Piping is the most ordinary thing a user can do to a
  command. It now exits with a message pointing at `--print` / `--dry-run`,
  which already work. Batch mode is unaffected — it must stay usable in a pipe.

### Fixed

- **The partition picker counted `down` and `drained` nodes as capacity.**
  `sinfo` emits one row per partition+state group and the code summed those rows
  without reading the state, so a partition holding nothing but dead nodes
  advertised the same node count as a healthy one. On the test cluster, five of
  the first ten choices had zero usable nodes; picking one gets a job that
  queues forever with no indication why. `fetch_partitions` now reads `%T` and
  reports `nodes_up` alongside `nodes`; the picker shows `13 of 17 nodes` and
  marks a fully-dead partition `unavailable`, and `validate_job_config` warns
  when the selected partition has no usable nodes. Nothing is hidden — a
  partition drained today can be the right answer tomorrow.

  A site whose `sinfo` reports no state column gets `nodes_up=None` (unknown),
  never `0`, so absence of evidence is not read as evidence of absence.

- **The picker offered partitions the user cannot submit to.** Private PI
  partitions routinely advertise `AllowGroups=ALL AllowAccounts=ALL` and still
  reject every submission with *"Invalid account or account/partition
  combination specified"* — the partition ACL is not the gate, the `sacctmgr`
  association list is. The picker now filters on associations when the site
  scopes them per partition. An association row with a **blank** Partition means
  "all partitions for that account" and is treated as a wildcard, so sites that
  gate on the account instead do not have their whole list filtered away.

- **Picker ordering was raw `sinfo` order**, which put a scheduler partition and
  a run of retired PI partitions ahead of the two partitions anybody actually
  uses. Now ranked: site default first (from `sinfo`'s `*` marker), then the
  user's own associations, then usable capacity, with fully-dead partitions and
  scheduler/system partitions last. `cron` is detected both by name and
  structurally — a partition whose nodes are all login nodes.

- **A confident false ETA for a job the scheduler had already refused.** A 35x
  over-request on a single-node partition reported `ETA: ~60s`. The
  `sbatch --test-only` tier collapsed "rejected" into the same `None` as "could
  not ask" and fell through to a queue-depth heuristic, which duly produced a
  specific, confident prediction for a job that can never start. The refusal is
  now surfaced verbatim:

  ```
  │ ETA:  never — More processors requested than permitted │
  ```

  `fetch_queue_eta` gained `feasible` and `reason` keys. A rejection is only
  claimed on positive evidence — Slurm's `allocation failure:` or the site
  plugin's more specific `Reason:` — so an unreachable controller or a broken
  `sbatch` still falls through to an estimate rather than trading one confident
  wrong answer for another.

- **The default `--mem` was a hardcoded `16G` with no relation to the cluster.**
  Harmless on a 57 GB node, permanently unschedulable on an 8 GB one, and the
  user who never passed `--memory` had no reason to suspect either — it was
  emitted even with no scheduler present at all. An unspecified memory is now
  sized from the partition's advertised node memory as
  `mem_per_node × cores / cpus_per_node`: the same share of the node's memory as
  of its cores, which is what a site's own `DefMemPerCPU` does, and never more
  than a node has. The literal remains only as a last resort when the cluster
  says nothing, and the summary states which happened. `--memory ''`/`none`
  still omits `--mem` entirely.

## [0.5.3] — 2026-07-29

### Fixed

- **The queue ETA was computed from node *state labels* and ignored free
  resources entirely, so it reported "~1 min" for jobs that could not start for
  hours — or at all.** `fetch_queue_eta` counted `sinfo` `idle`/`mix` nodes and
  returned "immediate" if that count reached `req_nodes`. A node's *state* says
  nothing about what is left on it: a MIXED node with 44 idle cores and every GPU
  allocated was counted as available for a 4-GPU job. Nor were CPUs, memory or
  GRES ever consulted — `req_nodes` was the only part of the request that reached
  the estimator.

  Measured on a live cluster while fixing this:

  | request | before | actual |
  |---|---|---|
  | `caslake`, 1 cpu, 30 min | "~1 min" | 5 h 55 m (`sbatch --test-only`) |
  | `gpu`, 4 GPUs | "~1 min" | 0 of 44 GPUs free — could not start; now ~23 h |

  The estimate now has three tiers, and reports which one answered in a new
  `source` key:

  - `scheduler` — `sbatch --test-only`, Slurm's own backfill placement. It queues
    nothing, and it is the only tier that sees QOS caps, account limits and the
    site `job_submit` plugin.
  - `resources` — nodes with enough genuinely free CPU, memory and GPU, from the
    per-node `CPUsState` / `Memory`−`AllocMem` / `Gres`−`GresUsed` fields.
  - `pressure` — the previous queue-depth heuristic, now a last resort only, and
    no longer able to return "now" (without resource data there is no evidence
    anything is free).

  `fetch_queue_eta` takes the rest of the request as keyword arguments
  (`cpus`, `mem_mb`, `gpus_per_node`, `gpu_type`, `time_limit`, `account`,
  `qos`); the old two-argument call still works but gives a weaker answer. Both
  the wizard and batch mode now pass the full request, and the TUI's ETA cache
  keys on all of it, so changing the GPU count refreshes the estimate.

- **New:** `resolve_request_mem_mb(answers)` resolves the per-node memory the
  built script will actually request, mirroring the builder's
  `--mem-per-cpu` over `--mem` over auto-directive precedence.

## [0.5.2] — 2026-07-24

A correctness release from a full-codebase audit that verified every
Slurm-behaviour claim against a live `sbatch --test-only` rather than by
inspection. Two findings were *only* visible that way (**a typed `--gres` request
on a count-only-GRES cluster**, and **`mamba activate` failing under mamba 2.x**),
and two previously-suspected issues were withdrawn as non-bugs.

### Fixed

- **A GPU model that a site exposes only as a node feature was requested as a
  GRES type, and Slurm rejected the job** — many clusters configure GPUs
  count-only (`Gres=gpu:4`) and put the model in the node's *feature* list.
  slurmate read those models correctly and offered them in the picker, but then
  emitted `--gres=gpu:<model>:N` (the default `gres_type` format), which fails
  with "Requested node configuration is not available", while validation reported
  nothing wrong. `fetch_gpu_type_sources()` now reports *how* each model can be
  requested (typed GRES vs. node feature); `validate_job_config` raises a hard
  error naming the fix when a feature-only model is requested through any
  type-naming format, and the wizard's GPU-format step defaults to `constraint`
  for such a model. Requesting `--gres=gpu:N` + `--constraint=<model>` — the form
  that actually schedules — is now what you get.
- **A custom `--constraint`/`-C` produced two conflicting `#SBATCH --constraint`
  lines** — Slurm keeps only the last one and silently discards the earlier,
  which was always slurmate's own (the GPU type or the `--constraint` answer), so
  the job landed on the wrong nodes with no error. Custom constraint flags are
  now merged into the single `&`-joined directive, values de-duplicated
  case-sensitively (Slurm features are case-sensitive) and an OR-expression
  parenthesised so `a&(b|c)` keeps its meaning.
- **A custom `--output`/`--error`/`-o`/`-e` left a contradictory auto directive**,
  and the submit report read the *first* `--output` — the one Slurm ignores — so
  "Log path:" and the `tail -f` hint pointed at a file the job never wrote. The
  auto directive is now suppressed per stream (as a custom `--mem` already did),
  and the report resolves the effective (last-wins) path, understanding
  `--output=P`, `--output P` and `-o P`.
- **Space-separated option values were shredded into nonsense flags** — `-C
  bigmem` became `['-C', '--bigmem']` and `-o /logs/x.out` became
  `['-o', '--/logs/x.out']`, emitting a valueless directive plus an invalid one
  that sbatch rejects. A bare token is now attached to the preceding option when
  that option takes a value (or when the token can't be an option name); a bare
  word after a boolean option is still its own flag. The same rule is shared by
  the free-form parser and the list/API path, so `--custom-sbatch="-o /p"`,
  `custom_sbatch = ["-o /p"]` and `custom_sbatch = ["-o", "/p"]` all agree.
- **The summary panel misreported memory** when a custom `--mem`/`--mem-per-cpu`
  flag suppressed the auto directive: it kept showing the unused answer value, so
  the panel and the script disagreed. Both surfaces now derive the row from the
  same custom-flag override the builder uses.
- **`mamba activate` failed on modern mamba, leaving the job in the wrong
  environment** — `conda.sh` defines only the `conda` hook, so on mamba ≥ 2
  (miniforge's current default) the emitted line died with "critical libmamba
  Shell not initialized" *without* stopping the script, and the job silently ran
  in whatever interpreter it inherited. The generated activation now falls back
  to `conda activate`, which activates a mamba-created env identically.
- **Fabric, rack and form-factor node features were reported as GPU models** —
  the blocklist covered `ib`/`opa`/`hdr` but not `hdr100`/`edr`/`fdr`/`ndr`, and
  the shape heuristic matched two-character labels like `b12`/`t2`, which then
  beat the real model that appeared later in the feature list. Detection is now
  known-model-first, then a stricter shape rule (family letter + 3-plus digits),
  with fabric/rack/form-factor/cooling tokens filtered from both branches.
- **A GPU type differing only in case passed validation and then failed at
  submit** — Slurm node features are case-sensitive (`-C A100` does not match a
  node advertising `a100`), while the check lowercased both sides. A case-only
  mismatch is now a warning naming the advertised spelling. (The picker keeps
  both spellings when a partition really has both — they select different nodes.)
- **A stale GPU-type cache suppressed a live "not in partition list" error** in
  the wizard after switching partitions; the cache is now keyed on the partition
  it was fetched for, like the QoS cache.
- **`submit_sbatch` created log directories before checking for `sbatch`**, so
  mock mode (and any host without Slurm) left stray `logs/` trees behind while
  reporting "no job submitted".
- **A leading space defeated tilde expansion** — `output_dir = " ~/logs"` emitted
  a literal `~/logs`, which Slurm does not expand; the value is now stripped
  before `expanduser`.
- **`validate_memory` accepted a `P` unit** that `sbatch --mem` rejects
  client-side ("Invalid --mem specification"); units are now K/M/G/T.
- **The action menu's editor label disagreed with the editor launched** (blank
  when `EDITOR` was set-but-empty); it now shows `_editor_command()`'s result.
- **The TUI review's label column used a fixed width**, so labels longer than 12
  characters broke the value column and multi-line indentation; it now measures
  the actual labels, as the CLI summary already did.
- **The queue-ETA cache ignored a later `nodes` change** (it is computed from
  whether enough idle nodes exist), and so always reported the `req_nodes=1`
  estimate; the cache key now includes the node count.
- **`theme.C` ignored `FORCE_COLOR`** while `rich` honours it, so piping with
  `FORCE_COLOR=1` produced half-coloured output. (`CLICOLOR_FORCE` is
  deliberately still ignored — rich ignores it too.) Banner animation now
  requires a real TTY explicitly, since colour is no longer a proxy for one.
- **An un-confirmed GPU-type edit was discarded by Back** in the free-text
  sub-mode — the only input the wizard's Back path didn't persist.
- **A space-form custom value containing a space was emitted unquoted** — once the
  parser consumes the user's quotes, `--comment "my job"` arrives as
  `--comment my job`, and only the `=` form was re-quoted, so Slurm split it into
  `--comment=my` plus a stray `job`. The space form is now quoted too, using the
  known value-taking option names to find where the value starts.
- **Whitespace inside a `--constraint` value produced a job Slurm rejects** —
  measured: `-C "a100 & 384g"` fails with "Invalid feature specification" while
  `-C "a100&384g"` schedules. All whitespace is now stripped from every constraint
  source (feature names cannot contain any), and a stray leading space no longer
  emits `--constraint= a100`.
- **A custom `--gres`/`--gpus*` override left a duplicate directive** and the summary
  then described a GPU request the job doesn't make (`2 × v100` for a script asking
  for `gpu:a100:8`). A differing custom flag on the option the chosen format emits now
  replaces the auto directive, as a custom `--mem`/`--output` already did; an *exact*
  duplicate still keeps slurmate's canonical `=` spelling, and a custom `--gpus` does
  not suppress an auto `--gres` (different requests to Slurm).
- **`--mem-per-cpu` was never checked against the node's memory** — it is per *core*,
  so `--mem-per-cpu=64G` with 8 cores (512G/node) passed silently while the equivalent
  `--mem=512G` warned. The check now multiplies by the cores requested per node and
  shows the arithmetic.
- **Validation warned about a memory value the script doesn't request** — a `--mem`
  superseded by `--mem-per-cpu` (or by a custom flag) still produced a limit warning,
  while the value actually requested went unchecked. Validation now resolves the
  effective memory the same way the builder does.

### Added

- **`--constraint` and `--mem-per-cpu` wizard steps.** Both were already CLI
  flags and config keys, but had no step — and because `Wizard` builds its
  defaults by iterating the step list, a config file's `constraint`/`mem_per_cpu`
  was silently dropped in interactive mode, so the same `.slurmate.toml` produced
  different jobs in batch and interactive mode. Both now appear in
  builder-directive order (`mem_per_cpu` with `memory`, `constraint` after the
  GPU block).
- **`python -m slurmate`** works alongside the console script.
- **An "Estimated GPU-hours" summary row** for GPU jobs, alongside CPU-hours;
  the multiplier follows the chosen `gpu_format` (per-node vs. per-task vs.
  job-wide).
- **`fetch_gpu_type_sources()`** — GPU models split by how they can be requested.
- **`effective_log_path()`** — the log path Slurm will actually use for a script.

### Documentation

- README: `constraint` and `mem_per_cpu` added to the recognized config keys; all
  five `--gpu-format` values listed (`gpus_per_node`/`gpus_per_task` were missing
  from three places); the `SLURMATE_BANNER_ANIMATE` row no longer claims it forces
  animation on a non-TTY (it cannot — and should not); `FORCE_COLOR` documented.
- The wizard's custom-flags subtitle and the `--custom-sbatch` help now state that a
  value may use `=` or a space, and that a value containing a space must be quoted —
  an unquoted `--comment=big run` is genuinely ambiguous, so slurmate rejects it loudly
  rather than guessing (guessing would fabricate values).

### Not changed (investigated, found correct)

- **`validate_time("1-99")`** was suspected of accepting a value sbatch rejects.
  It does not: Slurm accepts `1-99` (1 day + 99 hours) — measured — while
  slurmate already *rejects* input Slurm accepts (`25:99:99`). The validator is
  stricter than Slurm, not looser; tightening the days-hours field would have
  rejected valid input.
- **Case-duplicated GPU types in the picker** (`A100` *and* `a100`) look like a
  de-duplication bug but are correct: the two select different nodes, so folding
  them would break the constraint.
- **An OR node-feature constraint denied by cluster policy** (`--constraint=a100|v100`
  → "Access/permission denied" on the audit cluster) is not slurmate's doing: the
  identical request typed straight into `sbatch` fails the same way, while each
  individual feature schedules.

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
base case (e.g. UChicago Midway3) is byte-for-byte unchanged.

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
planning backlog. Highlights: the version is now
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
