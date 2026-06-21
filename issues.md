# Issues and Resolutions

_Entries #1 – #21 were resolved in v0.2.0. Entries from #22 onward are tracked moving forward._

## 1. Output file extension inconsistency
**Problem:** `#SBATCH --output=logs/test` (no `.out` extension) but `#SBATCH --error=logs/test.err` has `.err`. Bare filenames lacked the `.out` extension.

**Fix:** `builder.py` — bare filenames now have `.out` appended (and `.err` for stderr).

---

## 2. Hardcoded GPU type list
**Problem:** Code relied on a static `GPU_TYPE_CANDIDATES` list that missed newer GPUs (B100, Intel Xeons, etc.) and was impossible to keep complete.

**Fix:** `system_utils.py:_detect_gpu_type` — removed all hardcoded GPU name lists. Uses **negative filtering**: instead of matching against known GPU names, rejects only what is clearly NOT a GPU:
- Tokens starting with a digit → memory sizes (`256g`, `1536g`)
- CPU architecture prefixes (`gold-`, `xeon-`, `epyc-`, `ryzen-`, `atom-`, `i[3579]`)
- Infrastructure keywords (`ssd`, `nvme`, `ib`, `opa`, `hdr`, `hdd`, `scratch`, `fat`, `thin`)
- Tokens ≥ 15 chars (too long for a GPU model)

Everything else passes — unknown future GPUs like `b200`, `gb200`, `w7800`, `intel-max-1550`, `nvidia-h100` are all detected automatically.

---

## 3. GPU type case sensitivity
**Problem:** Case mismatches (e.g., `H100` vs `h100`) caused false warnings.

**Fix:** All GPU type comparisons are case-insensitive (`main.py`, `tui.py`).

---

## 4. Broken box borders on panels
**Problem:** Panel borders were shorter than the content — top border `╭─ Generated sbatch script ─╮` (1 dash each side, ~29 visible chars) while content/bottom border stretched to 45+ chars. Right corners didn't align.

**Root cause (two‑part):**
1. `expand=False` relies on Rich's broken `Text()` width measurement (total char count, not max line width) → fixed by replacing with explicit `width=…`.
2. Panel titles used raw ANSI escape codes (`c.PINK` = `\033[38;2;255;0;128m`, `c.CYAN` = `\033[36m`) embedded in Rich markup. Rich's title‑width calculation chokes on raw ANSI escapes, **ignoring the `width=` parameter** entirely and auto‑sizing to a wrong value, while rendering the title without enough filler dashes.

**Fix:**
- `main.py:285` — `title=f"[bold]{c.PINK}{title_text}[/]"` → `title=f"[bold #ff0080]{title_text}[/]"`
- `main.py:329` — `title=f"[bold]{c.CYAN}{s_title}[/]"` → `title=f"[bold cyan]{s_title}[/]"`
- Use Rich-native style names (`bold #ff0080`, `bold cyan`) instead of raw ANSI escapes so Rich handles width correctly. Both panels now render with matching borders at every terminal width.

---

## 5. Non-GPU features falsely detected as GPU types
**Problem:** `gold-6248r`, `1536g`, `a30` (from partitions without GPUs) appeared as GPU types because features were scanned without checking whether GRES actually confirms GPUs exist.

**Fix:** Features are only scanned when GRES contains `gpu:` — confirming the node actually has GPUs. Without `gpu:` in GRES, the function returns empty.

---

## 6. Features fallback with hardcoded heuristic
**Problem:** The heuristic `len ≤ 8, starts with letter, has digits` is a hardcoded assumption about GPU naming patterns that could break on clusters with differently-named GPUs.

**Fix:** Replaced positive-pattern-matching with **negative filtering** (see #2). This rejects only what is universally non-GPU rather than guessing what looks GPU-like.

---

## 7. "Any" GPU type warning is confusing
**Problem:** `⚠ Warning: GPU type 'Any' not in partition list (a30)` — "Any" is not a real GPU type, and showing a warning for it confused users.

**Fix:** `main.py:215`, `tui.py:592` — warning is skipped when `gpu_type.lower() == "any"`.

---

## 8. "Any" should not generate `--gres=gpu:Any:N`
**Problem:** Selecting "Any" caused `#SBATCH --gres=gpu:Any:N` which is invalid — "Any" means the user doesn't care about GPU type.

**Fix:** `builder.py:130-139` — "Any" now generates `#SBATCH --gres=gpu:N` (no type restriction), and skips `--constraint` lines entirely.

---

## 9. False GPU type warning — selected type IS in partition list
**Problem:** `⚠ Warning: GPU type 'a100' not in partition list (a30)` appeared even though the user selected `a100` from the radio list (which was populated from the partition). This happened because `_validate_partition_limits` checked `part.gpu_types` (from `fetch_partitions()`, GRES-only) while the radio list used `fetch_gpu_types_for_partition()` (GRES + features).

**Fix:** Two-part fix:
- `tui.py:591` — TUI warning now uses `self.transient["gpu_types"]` (same dynamic list as the radio options).
- `main.py:215-221` — `_validate_partition_limits` falls back to `fetch_gpu_types_for_partition()` when the static `part.gpu_types` doesn't contain the selected type.

---

## 10. Confusing conda activation syntax
**Problem:** `source $(conda info --base)/etc/profile.d/conda.sh; conda activate AI` — the `$(conda info --base)` subshell syntax was confusing to users who are used to `source activate`.

**Fix:** `builder.py:179-183` — replaced with `source activate AI`.

---

## 11. Modules displayed on one line in summary
**Problem:** Long module names (`python/anaconda-2022.05`) wrapped to a new line inside the summary panel because the panel was 2 chars too narrow (missing default padding of `(0, 1)`).

**Fix:** `main.py:330` — `width=summary_w + 4` accounts for both borders (2) and default padding (2).

---

## 12. Command step — unclear it supports multiple lines
**Problem:** The command field subtitle said "e.g. python train.py (Tab completes file paths)" with no mention of multiline support, even though the step already had `multiline=True`.

**Fix:** `tui.py:248` — updated subtitle to `"e.g. python train.py  (Tab completes file paths, multiline supported)"`.

---

## 13. GPU type detection only from GRES (missed count-only nodes)
**Problem:** Nodes configured with `gpu:4` (count-only, no model in GRES) returned `"gpu"` (unknown), which was filtered out by `fetch_gpu_types_for_partition`. This caused the test partition's `a100` GPUs to be invisible.

**Fix:** Added features scanning fallback in `_detect_gpu_type` (see #2 and #6). The test partition now correctly reports all its GPU types including `a100`, `a30`, `A100`, `H100`, `H200`, `L40S`, `rtx6000`, `v100`, `a40`.

---

## 14. Conda environment autocomplete
**Problem:** User wanted conda environment names to auto-complete when typing in the environment name field, similar to how file paths complete in the command field.

**Status:** Already implemented — `_setup_env_name` fetches conda envs via `fetch_conda_envs()` and sets `FuzzyWordCompleter` with the results.

---

## 15. Multiline command — Enter always advances instead of inserting newline
**Problem:** The command step had `multiline=True`, but pressing Enter with `eager=True` intercepted the key before the TextArea could insert a newline, always advancing to the next step instead. Users could not enter multi-line commands like `python xyz.py` followed by `Rscript x.R`.

**Fix:** `tui.py:377-384` — the Enter handler now checks `getattr(s, "multiline", False)` first; if true, it calls `buf.insert_text("\\n")` to insert a newline and returns without advancing. Use Tab to advance from multiline steps.

---

## 16. Modules autocomplete broken for comma-separated entry
**Problem:** The modules step used `FuzzyWordCompleter` which matched the **entire buffer** as a single token. After typing the first module and a comma (e.g. `python/anaconda,`), the FuzzyWordCompleter tried to match `python/anaconda,cuda` against module names — which never matched, so auto-completion broke after the first entry.

**Fix:** `tui.py:127-153` — added `LastTokenCommaCompleter`, a custom `Completer` that extracts only the **last comma-separated token** and fuzzy-matches it against the word list. Now typing `python/anaconda,cuda` correctly offers `cuda/11.8`, `cuda/12.1`, etc. The first module's autocomplete is also unaffected.

---

## 17. Module list re-rendered with Python brackets on step-back
**Problem:** When navigating back to the modules step, `prev = self.answers.get("modules")` returned the list `["python", "R/4.4.1"]`. `str(prev)` produced `"['python', 'R/4.4.1']"`, which was set as the text area value and then comma-split on the next advance, generating corrupted module names like `module load ['python'` and `module load R/4.4.1]`.

**Fix:** `tui.py:668` — added an `isinstance(prev, list)` check that joins with `", ".join(prev)` instead of calling `str()`.

---

## 18. Modules multi-entry workflow and command-step advancement
**Problem:** Two issues:
- Modules step required manually typing commas; Enter with a completion applied the name but gave no clear next action.
- Command step (`multiline=True`) had Enter inserting newlines but NO WAY to advance — Tab cycled path completions forever, trapping the user on the step.
- Footer contained useless copy-paste/mouse instructions and inconsistent key names ("Enter:Next" vs "Tab:advance").
- Command subtitle mentioned "Tab" twice, confusing users.

**Fix:**
- `tui.py` (Enter handler) — Enter with a completion on the modules step now appends `", "` automatically; repeat to add more modules, then Tab to advance when no completion.
- `tui.py:389-393` — Tab handler changed: always calls `buf.complete_next()` first (cycles existing completions or starts a new one). Only advances if `complete_state` remains None (no completions found). For multiline steps with path completions, Ctrl+G advances instead.
- `tui.py:268` — Modules subtitle: `"Enter a name, press Enter to add (comma auto-inserted); Tab to advance when done"`.
- `tui.py:278` — Command subtitle: `"e.g. python train.py  (Enter=newline, Tab=complete, Ctrl+G=next)"`.
- `tui.py:1127-1135` — Footer cleaned up: removed all mouse/F2 garbage. Shows `Tab/Enter:Next` for single-line, `Enter:newline  Tab:complete  Ctrl+G:next` for multiline, plus `Esc:Back  ^C:Quit`.
- Removed the entire F1/? help modal (user feedback: it made no sense).

---

## 19. Input lost on step-back and Tab still advancing on multiline
**Problem:** Two issues:
1. Pressing Esc on the command step to go back, then returning, showed a blank input — `_go_back()` never saved the current text to `self.answers`, so `_on_enter_step` restored the default (empty).
2. Tab handler checked `buf.complete_state is not None` and immediately advanced if None — without ever trying to trigger completion. On the command step with `complete_while_typing`, if async completion hadn't populated `complete_state`, Tab advanced instead of completing paths.

**Fix:**
- `tui.py:526-560` — `_go_back()` now saves the current step's input to `self.answers` before navigating away (text/autocomplete/ntasks_per_node via `_text_val()`, select/gpu_format via `_radio_value()`). Returning to the step restores the saved value.
- `tui.py:386-393` — Tab handler rewritten: `buf.complete_next()` is called unconditionally, which either cycles an existing completion or calls `start_completion()` to fetch new ones. Only advances when `complete_state` remains None (no completions at all).
- `tui.py:432-434` — Added `Ctrl+G` (`c-g`) as a universal "next step" key that always calls `_confirm_and_next()` regardless of completions. Footer for multiline steps shows `Tab:complete Ctrl+G:next`.

---

## 20. CI failing — unused Frame import and generator return type
**Problem:** `.github/workflows/ci.yml` runs `ruff check src/` and `mypy src/` on push/PR. The commit `bd54be1` introduced:
- `ruff` error F401: `prompt_toolkit.widgets.Frame` imported but unused (`tui.py:33`).
- `mypy` error misc: `LastTokenCommaCompleter.get_completions` annotated `-> Completion` but it is a generator (yields) → must return `Generator[Completion, None, None]` or an Iterable supertype (`tui.py:136`). The existing `# type: ignore[override]` did not suppress the `misc` code.

**Fix:**
- `tui.py:33` — Removed `Frame` from the import.
- `tui.py:6` — Added `Generator` to `from collections.abc import`.
- `tui.py:109,136` — Changed return type of both `LastTokenPathCompleter.get_completions` and `LastTokenCommaCompleter.get_completions` to `Generator[Completion, None, None]`, removed stale `type: ignore` comments.

---

## 21. Tab still advances from multiline command step (async path completer)
**Problem:** The fix from #19 claimed Tab "only advances when `complete_state` remains None (no completions at all)". But `PathCompleter` generates completions asynchronously — `buf.complete_next()` schedules the completion via `start_completion()` but `complete_state` is still `None` on the next line of the eager handler. So Tab always advanced from the multiline command step instead of doing path completion.

**Fix:** `tui.py:389-395` — Added a guard: Tab for multiline steps only attempts completion and returns without advancing regardless of whether completions were found. Ctrl+G remains the designated "next step" key for multiline steps.
