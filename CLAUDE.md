# CLAUDE.md — Project Instructions for Claude Code

## Project Summary
Trailer Counter: a computer vision post-processing pipeline that counts filled dump truck/trailer loads per unique vehicle per day, using video from a CCTV camera mounted inside an excavator cab roof.

## Critical Context (Read Before Helping)

### Camera Setup
- Camera is fixed inside the excavator cab roof, looking downward at the loading area
- The cab (upper body) rotates — the camera rotates with it
- Because the camera rotates with the cab, the trailer always appears in roughly the same image region when being loaded
- This is why a **static polygon zone** works (see ADR-002 in docs/DECISIONS.md)

### Fill Event Definition
- A "fill" = trailer enters loading zone polygon AND then departs
- No minimum dwell time (trailer sizes vary too much)
- Any departure = count 1, regardless of full/partial load
- There are passing vehicles and other excavators' trucks in frame — only count vehicles that enter the calibrated loading zone

### Re-ID Scope
- Gallery resets each day — no cross-day vehicle persistence
- Identification is by visual appearance only: color, shape, body surface, wear patterns
- DINOv3 ViT-B/16 + SiteSense projection head → 128-d L2-normalized embeddings
- Cosine similarity threshold: 0.75 (tune based on real footage)

### Current Status
**No code has been written yet.** Project is in planning phase. See `docs/PROGRESS.md` for phase status.

---

## Architecture Rules (Enforce These)

Each `src/` module has exactly one responsibility. **These isolation boundaries must be respected:**

| Module | Allowed to know about | Must NOT know about |
|---|---|---|
| `preprocessor.py` | OpenCV | Detectors, trackers, zones |
| `detector.py` | RF-DETR or YOLOv8 | Tracking, zones, counts |
| `tracker.py` | BoxMOT | Zones, counts, Re-ID |
| `zone.py` | Polygon math, OpenCV | Trackers, models, counts |
| `event_counter.py` | Track IDs, zone bool | Video, models, dashboard |
| `reid_gallery.py` | numpy, sklearn | Video, detectors, dashboard |

**Data contracts between modules are fixed** — do not change them without updating `docs/ARCHITECTURE.md`:
- Detector output: `{"bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int}`
- Tracker output: `{"track_id": int, "bbox": [...], "confidence": float, "class_id": int}`
- Results JSON: see `docs/ARCHITECTURE.md` → Data Contracts section

---

## Development Workflow

1. **Read the phase plan before writing any code:**
   `docs/superpowers/plans/2026-04-13-trailer-counter-mvp.md`

2. **TDD is mandatory** — write the failing test first, then the implementation:
   ```bash
   pytest tests/ -v                          # run all tests
   pytest tests/test_zone.py -v              # run specific
   ```

3. **One notebook per phase** in `notebooks/` — do not mix phase logic across notebooks

4. **Update `docs/PROGRESS.md`** when completing a phase — update the status table and add a log entry

5. **Platform:** Kaggle Notebook (primary, 107 GB disk). Use `streaming=True` when loading `fuxi-robot/excavator-video` on Colab.

---

## Python Environment

- **Python 3.10 or 3.11 only** — `rfdetr` has known issues with 3.12+
- Install all dependencies: `pip install -r requirements.txt`
- Model weights live in `models/` (gitignored) — download via:
  ```python
  from huggingface_hub import hf_hub_download
  hf_hub_download(repo_id="Zaafan/sitesense-weights", filename="rfdetr_construction.pth", local_dir="models/")
  ```

---

## Coding Standards

- **Immutability:** never mutate arguments — return new objects
- **File size:** max ~400 lines per file; split by responsibility
- **No deep nesting:** max 3 levels
- **Error handling:** validate at system boundaries (user inputs, external APIs, file I/O)
- **No hardcoded values:** use constants or config files (e.g., `config/loading_zone.json`)
- **Type hints:** required on all function signatures in `src/`

---

## Key Documents

| Document | When to Read |
|---|---|
| `docs/PRD.md` | To understand what the system must do and success criteria |
| `docs/ARCHITECTURE.md` | Before writing any new module or changing data contracts |
| `docs/DECISIONS.md` | Before proposing a technology change — the ADR may already cover it |
| `docs/PROGRESS.md` | To understand current phase and what has been completed |
| `docs/superpowers/plans/2026-04-13-trailer-counter-mvp.md` | The full step-by-step implementation plan with code |

---

## What NOT to Do

- Do not add live streaming logic (out of scope for MVP)
- Do not add cross-day Re-ID gallery persistence (ADR-005 — day-scoped only)
- Do not fine-tune any model weights in MVP (use pretrained only)
- Do not add rain condition handling (explicitly excluded)
- Do not add multi-excavator support (single excavator only)
- Do not change module isolation boundaries without explicit approval
