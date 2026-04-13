# GitHub-as-Bridge Workflow

> Ground rules for the three-environment development model.
> **Effective from Phase 1 onward. All future sessions must follow this document.**

---

## The Three-Environment Model

There are exactly three environments. Each has a fixed role. No role bleeds into another.

```
Claude Code (Windows, no GPU)
  Role: AUTHOR
  Writes: src/, tests/, config/, requirements.txt,
          run_pipeline.py, dashboard/, docs/,
          notebooks (cell structure only — not executed)
  Tool: plain git (no gh CLI)
          ↓ git push
GitHub (Rutwik1000/trailer-counter)
  Role: BRIDGE — single source of truth
  Stores: all versioned code, configs, docs, notebook structure
  Does NOT store: models/, data/frames/, annotated mp4s
          ↓ !git pull (notebook Cell 1)
Kaggle (primary) / Colab (fallback)
  Role: EXECUTOR
  Runs: notebooks, GPU inference
  Writes back: data/results/*.json, config/loading_zone.json, docs/PROGRESS.md
```

### Non-Negotiable Boundaries

| Rule | Reason |
|---|---|
| Claude Code never executes notebooks | No GPU; execution is Kaggle's role |
| Kaggle/Colab never edits `src/` modules | Bypasses TDD and ownership |
| GitHub never receives files > ~50 KB | No weights, frames, or mp4s |
| `pytest` runs only in Claude Code | Tests must not depend on GPU or Kaggle's dataset |

---

## Work Cycle Protocol

Every unit of work follows this cycle. Do not skip steps.

```
[1]  PULL    Claude Code: git pull origin master (before any new work)
[2]  TEST    Write failing test in tests/ BEFORE any src/ code
[3]  IMPL    Write src/ module to pass the test; author notebook cell structure
[4]  VERIFY  pytest tests/ -v — must be green before committing
[5]  COMMIT  Stage by explicit file path (never git add -A or git add .)
[6]  PUSH    git push origin master
             ── Kaggle side ──────────────────────────────────────────────
[7]  SYNC    Cell 1: git pull origin master
             Cell 2: pip install -r requirements.txt -q
[8]  EXEC    Run notebook cells in order; capture all outputs
[9]  BACK    Commit data/results/, config/loading_zone.json, PROGRESS.md
             git push origin master
             ── Back to Claude Code ───────────────────────────────────────
[10] PULL    git pull origin master (receive Kaggle commits); begin next cycle
```

---

## File Ownership Matrix

| File | Owner | Committed? | Notes |
|---|---|---|---|
| `src/*.py` | Claude Code | YES | Never edited on Kaggle |
| `tests/*.py` | Claude Code | YES | Never run on Kaggle |
| `notebooks/0N_*.ipynb` | Claude Code (structure) | YES — outputs stripped | Kaggle executes; outputs not committed |
| `requirements.txt` | Claude Code | YES | Single source of truth for both envs |
| `run_pipeline.py` | Claude Code | YES | CLI entry; authored in Phase 4 |
| `dashboard/app.py` | Claude Code | YES | |
| `config/loading_zone.json` | Kaggle (once, Phase 3) | YES | Commit immediately after calibration |
| `config/detector_choice.json` | Claude Code | YES | Set in Phase 2 |
| `data/results/YYYY-MM-DD.json` | Kaggle | YES (small files) | Primary pipeline output |
| `data/frames/*.jpg` | Kaggle | NO | Too large; stays on Kaggle disk |
| `data/outputs/*.mp4` | Kaggle | NO | Add to .gitignore before Phase 6 |
| `models/*.pth` | Kaggle (hf_hub_download) | NO | Re-downloaded each session |
| `docs/PROGRESS.md` | Both (append-only) | YES | Claude Code: structure; Kaggle: phase entries |

---

## Mandatory Notebook Structure

Every notebook must open with these three cells, in this exact order:

```python
# Cell 1 — Repo sync (ALWAYS run first, even if "already cloned")
import os, sys
repo_path = "/kaggle/working/trailer-counter"
if not os.path.exists(repo_path):
    os.system(f"git clone https://github.com/Rutwik1000/trailer-counter.git {repo_path}")
os.chdir(repo_path)
os.system("git fetch origin")
os.system("git reset --hard origin/master")  # always match remote exactly — never fails on divergent branches
os.system("git log --oneline -3")  # confirm latest commit is present

# Cell 2 — Install dependencies
os.system("pip install -r requirements.txt -q")
# Force reimport of src modules in case install changed them
for mod in list(sys.modules.keys()):
    if mod.startswith("src"):
        del sys.modules[mod]

# Cell 3 — Verify environment
import torch
print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
# Add phase-specific imports below this line
```

### Forbidden in Notebook Cells

| Pattern | Why | Correct Alternative |
|---|---|---|
| Class/function definitions duplicating `src/` | Silent divergence — untested version runs | Import the `src/` version |
| `if/elif` chains with pipeline logic | Logic belongs in `src/` | Call a `src/` function |
| Hardcoded absolute paths (`/kaggle/working/...`) | Breaks on Colab, breaks on session reset | `os.path.join(repo_path, ...)` |
| `pip install X` without updating `requirements.txt` | Breaks next session, breaks local env | Add to `requirements.txt` first |
| `torch.save()` into repo paths | Weights must not enter the repo | Use `hf_hub_download`; save to `/kaggle/working/` |
| `!pytest` | Tests are local only | Run tests in Claude Code |
| Committed output cells | Binary blobs inflate repo | `jupyter nbconvert --clear-output --inplace` |

### Allowed in Notebook Cells

- `import` statements
- Top-of-notebook constants (`BATCH_SIZE = 8`)
- Calls to `src/` functions
- Display/visualization (`cv2.imshow`, `plt.imshow`, `sv.plot_image`)
- File I/O for outputs (`json.dump`, `cv2.imwrite`)
- Simple assertions (`assert len(frames) == 50, "Expected 50 frames"`)
- The git sync/commit cells from the work cycle

### Before Committing Any Notebook from Claude Code

```bash
# Strip all output cells — keep code only
jupyter nbconvert --clear-output --inplace notebooks/0N_phase.ipynb
# Confirm outputs are empty
grep -c '"outputs": \[\]' notebooks/0N_phase.ipynb
```

---

## Output Management

| Output | Where | Committed? |
|---|---|---|
| Daily count JSON | `data/results/YYYY-MM-DD.json` | YES |
| Calibrated zone polygon | `config/loading_zone.json` | YES (once) |
| Sample frames | Kaggle disk only | NO |
| Annotated mp4 | Kaggle disk or Kaggle Dataset storage | NO |

### Result Commit Protocol (from Kaggle)

```python
from kaggle_secrets import UserSecretsClient
import os

os.system("git config user.email 'kaggle@trailer-counter'")
os.system("git config user.name 'Kaggle Executor'")
os.system("git add data/results/")
os.system("git diff --stat HEAD")   # confirm ONLY result JSONs are staged
os.system("git commit -m 'data: daily count results YYYY-MM-DD — phase N output'")

gh_token = UserSecretsClient().get_secret("github_token")
remote_auth = "https://" + gh_token + "@github.com/Rutwik1000/trailer-counter.git"
os.system("git remote set-url origin " + remote_auth)
os.system("git push origin HEAD:master")
os.system("git remote set-url origin https://github.com/Rutwik1000/trailer-counter.git")
print("Pushed to master")
```

**Required Kaggle Secret:** `github_token` — GitHub PAT with `repo` scope.
Create at: github.com → Settings → Developer settings → Personal access tokens → Tokens (classic).

Only commit after the full pipeline run completes without error. Never commit partial results.

### Loading Zone Config (Phase 3, one-time)

1. Claude Code commits `config/loading_zone.json` with `{"polygon": null}` as placeholder
2. Phase 3 calibration cell overwrites it with real coordinates
3. Kaggle commits immediately: `git add config/loading_zone.json && git commit`
4. Claude Code pulls: `git pull origin master`
5. This file is never overwritten again unless the camera is physically repositioned

---

## Environment Parity Rules

- `requirements.txt` is the **only** source of truth for installed packages in both environments
- Never `pip install X` on Kaggle without first adding X to `requirements.txt` in Claude Code
- Use `>=` version pins, not `==` pins — exact pins cause resolver conflicts with Kaggle's pre-installed packages
- Python 3.10 or 3.11 only — rfdetr incompatible with 3.12+. Confirm via Cell 3 `sys.version` output
- After Phase 1 install on Kaggle, run `!pip freeze` and document surprising resolved versions in `docs/PROGRESS.md`
- Every `import` used in `src/` must appear in `requirements.txt`

---

## Branching Strategy

| Scenario | Branch? | Notes |
|---|---|---|
| Phase N implementation | No — commit to `master` | Phases are sequential; no parallel work |
| Uncertain approach | No — use Kaggle-only scratch notebook | Not in repo; winner implemented clean |
| Contingency path | Yes — `contingency/<id>` | Merge to master immediately on validation |
| Phase completion | No — tag on `master` | `git tag -a phase-N-complete` |
| Kaggle output commits | No — direct to `master` | Small non-code changes |

### Tagging Phase Completions

```bash
# On Claude Code, after PROGRESS.md is updated:
git tag -a phase-N-complete -m "Phase N: <one-line summary>"
git push origin phase-N-complete
```

---

## Emergency Debugging Protocol

**Step 1 — Confirm latest code (2 min)**
```python
!git log --oneline -5   # does HEAD match last Claude Code push?
!git status
```
If not: re-run Cell 1.

**Step 2 — Isolate failure layer**

| Error | Layer | Action |
|---|---|---|
| `ModuleNotFoundError` | requirements / install | Re-run pip install; check requirements.txt |
| `AttributeError: X has no attribute Y` | Library API changed | Check installed version vs pin |
| `FileNotFoundError: models/...` | Weights not downloaded | Re-run hf_hub_download cell |
| `CUDA out of memory` | Batch size | Halve the batch size constant |
| `KeyError: "video"` | Dataset schema | Print `list(sample.keys())` |
| `RuntimeError` in PyTorch | dtype / device | Add `.to(device)` checks |

**Step 3 — Reproduce in isolation**
New notebook, minimal reproduction. Not reproducible → restart runtime + re-run from Cell 1.

**Step 4 — Determine bug location**
- Bug in notebook glue code → fix on Kaggle, copy corrected cell back to Claude Code, commit
- Bug in `src/` module → **do NOT fix on Kaggle**. Close notebook. Write failing test in Claude Code. Fix `src/`. `pytest` green. Commit. Push. Pull on Kaggle.

**Step 5 — Document in `docs/PROGRESS.md`**
```
## YYYY-MM-DD — Debug note: <description>
**Phase:** N
**Symptom:** <error message>
**Root cause:** <one sentence>
**Fix:** <what changed and where>
```

**Common false alarms**
- "Worked yesterday, fails today" — Kaggle session reset. Re-run Cell 1 + re-download models (not persisted on free tier).
- "Push rejected" — Kaggle committed something. Fix: `git pull origin master --rebase`, then push. Never `--force`.

---

## Dos and Don'ts

### DOs

1. `git pull origin master` in Claude Code **before** starting any new work cycle
2. Write the failing test **before** writing any `src/` implementation (TDD mandatory)
3. Stage files by explicit path (`git add src/zone.py`) — never `git add -A`
4. Keep `pytest tests/ -v` green on `master` at all times
5. Strip notebook output cells before committing: `jupyter nbconvert --clear-output --inplace`
6. Tag every phase completion: `git tag -a phase-N-complete`
7. Use `os.path.join(repo_path, ...)` for all file paths in notebooks
8. Commit `config/loading_zone.json` from Kaggle **in the same session** it was calibrated
9. Document any deviation from the implementation plan as a new ADR in `docs/DECISIONS.md`
10. Add `data/outputs/*.mp4` to `.gitignore` **before Phase 6 begins**

### DON'Ts

1. **Never edit `src/*.py` on Kaggle** — untested, bypasses TDD, invisible to Claude Code
2. Never `git add -A` or `git add .` — stages weights, frames, or `.env` accidentally
3. Never commit notebooks with output cells — binary blobs inflate repo history permanently
4. Never `pip install X` on Kaggle without first adding X to `requirements.txt`
5. Never change module isolation boundaries without an ADR
6. Never create a second notebook for the same phase (`02_detection_v2.ipynb`)
7. Never commit `models/` weights — not even "just 5 MB"
8. Never mix phase logic across notebooks — Phase 2 code belongs only in `02_detection.ipynb`
9. Never leave a broken test on `master` — broken tests block all future cycles
10. Never `git push --force` — destroys Kaggle's committed result JSONs

---

## Per-Phase Commit Responsibility

| Phase | Claude Code Commits | Kaggle Commits Back |
|---|---|---|
| 1 — Foundation | `src/preprocessor.py`, `tests/`, `notebooks/01_foundation.ipynb`, `requirements.txt` | `docs/PROGRESS.md` |
| 2 — Detection | `src/detector.py`, `tests/`, `notebooks/02_detection.ipynb`, `config/detector_choice.json` | `docs/PROGRESS.md` |
| 3 — Tracking + Zone | `src/tracker.py`, `src/zone.py`, `tests/`, `notebooks/03_tracking_zone.ipynb` | `config/loading_zone.json`, `docs/PROGRESS.md` |
| 4 — Fill Counter | `src/event_counter.py`, `tests/`, `run_pipeline.py`, `notebooks/04_counting.ipynb` | `data/results/*.json`, `docs/PROGRESS.md` |
| 5 — Re-ID | `src/reid_gallery.py`, `tests/`, `notebooks/05_reid.ipynb` | `data/results/*.json` (updated), `docs/PROGRESS.md` |
| 6 — Dashboard | `dashboard/app.py`, `src/video_annotator.py`, `notebooks/06_dashboard.ipynb` | `docs/PROGRESS.md` |

---

## Git Commands Reference

```bash
# Start of every session — receive any Kaggle commits
git pull origin master

# Push new work
git push origin master

# Recover from rejected push (Kaggle committed while you worked)
git pull origin master --rebase
git push origin master

# Tag phase completion
git tag -a phase-N-complete -m "Phase N: <summary>"
git push origin phase-N-complete

# Inspect what Kaggle committed
git show HEAD --stat

# View history with tags
git log --oneline --tags -10

# Undo a commit that has NOT been pushed yet
git reset HEAD~1    # keeps file changes, unstages the commit
```
