# Issues and Resolutions

_Entries #1 – #21 were resolved in v0.2.0 (see [CHANGELOG.md](CHANGELOG.md))._

_Second-round issues #22 – #27 and the regressions #28 – #29 introduced by the
UI work are all **resolved** and folded into the v0.2.0 entry of the CHANGELOG.
Verified 2026-06-22: ruff + mypy clean, 120 tests pass, and the
GPU-corroboration, review-step, and footer fixes were checked behaviorally. No
open correctness issues remain._

---

## Open — minor polish (non-blocking)

These are cosmetic/edge-case nits only; nothing affects correctness.

- **Sidebar has no scroll.** On a very short terminal the visible-step list can
  still overflow the sidebar height. (Header overflow was already fixed by the
  compact `n/total` counter that replaced the per-step dot row.)
- **Visible-step denominator shifts once.** `visible_total` can only count a
  skippable step (GPU type/format, tasks-per-node, env name) as skipped after the
  user passes it, so the `n/total` denominator may change once mid-flow. Inherent
  to not knowing future answers; acceptable.

## Verified resolved this round (kept for traceability)

- **#22** GPU detection false positives — `_detect_gpu_type` now *prefers* a
  feature token corroborated by a typed GPU model (`gpu:MODEL:N`) seen elsewhere
  in the partition (`rack5,gpfs,a40` → `a40`), but **falls back** to negative
  filtering when nothing corroborates, so feature-only GPU types are still
  detected. (Fixes a regression where requiring corroboration dropped every type
  that lacked a typed GRES — a partition with only `a30` typed showed just
  `a30`.) Verified a mixed partition now reports all of
  `a100, a30, a40, h100, h200, l40s, rtx6000, v100`.
- **#23** `output_file` non-`.out` extension no longer double-appends `.out`.
- **#24** `_coerce` empty `gpus` → 0 (was 4).
- **#25** memory-limit warnings use `_parse_mem_to_mb` (decimals + K/P handled).
- **#26** `--env-type none` with an env name now logs a warning.
- **#27** CHANGELOG config path corrected to the TOML paths.
- **#28** Review step no longer crashes — focused window is in the layout.
- **#29** Footer restores `Esc:Back` / `^C:Quit` on every step.
- **U1/U3** skipped steps hidden from counter + sidebar; compact `n/total`.
- **U4** in-TUI Review & Submit step.
- **U5** `output_file` subtitle clarifies `.out`/`.err` derivation.
