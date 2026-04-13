# Senior Engineer Review — Trailer Counter Planning Phase

**Reviewer:** Claude Opus 4.5 (acting as senior engineer)
**Date:** 2026-04-13
**Documents Reviewed:** README.md, CLAUDE.md, PRD.md, ARCHITECTURE.md, DECISIONS.md, PROGRESS.md, RESEARCH_SUMMARY.md, Implementation Plan (`docs/superpowers/plans/2026-04-13-trailer-counter-mvp.md`)
**Status:** BLOCKED — 3 critical issues must be resolved before Phase 1 begins

---

## Summary

Planning quality is good. Documentation is thorough, well-organised, and internally consistent. Module boundaries are clear, TDD is mandated, and out-of-scope items are explicitly stated in each phase. However, three foundational dependencies cannot be assumed — they must be verified before any code is written.

---

## CRITICAL — Implementation Blocked

### C1: DINOv3 Model ID May Not Exist

**Location:** ADR-006, ARCHITECTURE.md Re-ID Pipeline, implementation plan Phase 5

The plan references `facebook/dinov3-vitb16-pretrain-lvd1689m`. This model ID appears to be incorrect or hallucinated. HuggingFace hosts **DINOv2** models (e.g., `facebook/dinov2-base`), not DINOv3 under this naming. If the backbone is DINOv2-base, the CLS token dimension is 768-d, not 1536-d — which changes the projection head architecture throughout.

**Required action:** Run Phase 0B Step 2 to verify the model ID. Update ADR-006 and ARCHITECTURE.md with confirmed values before Phase 5.

---

### C2: SiteSense Weight Files Unverified

**Location:** ADR-004, ADR-006, implementation plan Task 1.2

The plan downloads three files from `Zaafan/sitesense-weights` on HuggingFace:
- `rfdetr_construction.pth`
- `dinov3_reid_head.pth`
- `osnet_x0_25_msmt17.pt`

The SiteSense GitHub repository is confirmed 404. No one has verified these weights exist on HuggingFace. If they do not exist, Phase 2 (detection) and Phase 5 (Re-ID) are both blocked.

**Required action:** Run Phase 0B Step 1 to verify. If missing, apply Contingency C2A (detector) and/or C2B (Re-ID) documented in ADR-004 and ADR-006.

---

### C3: Re-ID Head Missing Activation Functions

**Location:** Implementation plan Task 5.2 (~line 1200)

The `ReIDHead` class was written as:

```python
self.proj = nn.Sequential(
    nn.Linear(1536, 256),
    nn.Linear(256, 128),   # ← no activation between layers
)
```

If the saved weights were trained with `nn.ReLU()` between layers (standard practice), this architecture mismatch causes silent failure — the weights load without error but produce garbage embeddings. Re-ID accuracy collapses silently.

**Required action:** Add `nn.ReLU()` between layers. Parameterise `in_dim` to handle DINOv2 (768) or DINOv3 (1536) after C1 is confirmed.

---

## HIGH PRIORITY — Fix Before or During Phase 1

### H1: fuxi-robot Dataset Field Schema Unverified

**Location:** Implementation plan Task 1.3

The frame extraction code assumes `sample["video"]["bytes"]` is the correct field path. This is speculative — the actual HuggingFace dataset schema may differ.

**Required action:** Add Phase 0B Step 3 schema inspection cell. Record actual field names and update Task 1.3 accordingly.

---

### H2: RF-DETR Python API Not Verified

**Location:** Implementation plan Task 2.1 (`detector.py`)

The code uses `RFDETRBase.from_checkpoint()` and `.predict()`. These method names should be confirmed against the installed package.

**Required action:** Run Phase 0B Step 4. Update `_load_model` and `_detect_rfdetr` in `detector.py` if method names differ.

---

### H3: Detector Tests Pass Vacuously on Black Frames

**Location:** `tests/test_detector.py`

Both tests use `np.zeros(...)` (a black frame). A black frame produces zero detections, so the tests pass without verifying the detector actually works. Schema validation logic is never exercised.

**Required action:** Add `test_detector_on_real_frame` using an extracted sample frame. Test should skip gracefully if Phase 1 has not run yet.

---

## MEDIUM PRIORITY — Address Before Phase 2

### M1: CLAHE Preprocessing Created Too Late

**Location:** Implementation plan — `src/preprocessor.py` created in Phase 6

CLAHE should be applied in every phase from Phase 2 onward. Creating it in Phase 6 means Phase 2–5 detection and tracking evaluations run on raw (unenhanced) frames, potentially making dust-condition performance worse than it would be in production.

**Required action:** Move `src/preprocessor.py` creation to Phase 1 (new Task 1.4). Apply `apply_clahe(frame)` before `detector.detect(frame)` in all pipeline notebooks.

---

### M2: `run_pipeline.py` Documented but Never Created

**Location:** README.md line 43; PRD use case UC-01

The README advertises:
```bash
python run_pipeline.py --video path/to/video.mp4 --date 2026-04-13
```
This file is not created anywhere in the 6-phase plan. Phase 4's notebook contains equivalent logic but is not a usable CLI.

**Required action:** Add Task 4.2 to create `run_pipeline.py` as a proper CLI entry point wrapping the Phase 4 pipeline logic.

---

### M3: No Integration Test

**Location:** Tests section — only unit tests exist

All three planned test files (`test_zone.py`, `test_event_counter.py`, `test_reid_gallery.py`) are unit tests for individual components. There is no test that exercises the zone + counter components together on a known input and verifies the JSON output format.

**Required action:** Add `tests/test_integration.py` with a smoke test that simulates tracked vehicles, calls zone and counter, and asserts the output JSON structure.

---

### M4: Dashboard Errors on Phase 4 Output

**Location:** Implementation plan Phase 6, `dashboard/app.py`

The dashboard calls `df.groupby("vehicle_id")` unconditionally. Phase 4 output does not include `vehicle_id` (it is added in Phase 5). Running the dashboard on Phase 4 results will raise `KeyError`.

**Required action:** Add a guard: if `vehicle_id` is absent in the dataframe, fall back to `track_id`.

---

## Suggestions (Non-blocking)

| # | Suggestion | Benefit |
|---|-----------|---------|
| S1 | Add Re-ID cosine similarity score to results JSON events | Easier debugging of false matches during Phase 5 validation |
| S2 | Make gallery EMA weight (0.9/0.1) a `ReIDGallery` constructor parameter | Faster tuning without code changes |
| S3 | Use `supervision` annotation utilities in `video_annotator.py` | Cleaner code; library already a dependency |

---

## Approved Decisions

| ADR | Decision | Assessment |
|-----|----------|------------|
| ADR-001 | Departure-based counting (no dwell time) | Correct. Simpler and more reliable for MVP. |
| ADR-002 | Static polygon zone | Correct. Camera-cab co-rotation makes loading position image-stable. |
| ADR-003 | BotSort over ByteTrack | Correct. Appearance-based tracking handles cab rotation better than IoU-only. |
| ADR-005 | Day-scoped gallery, resets per day | Correct. Cross-day Re-ID requires validation data that doesn't exist. |
| ADR-007 | Centroid-based zone containment | Correct. Simple, deterministic, handles partial overlap correctly. |
| ADR-009 | Kaggle Notebook primary | Correct. Only free platform with enough disk (107 GB) for the 85.5 GB dataset. |

Module isolation boundaries (ARCHITECTURE.md component map) are well-designed. The explicit "Knows About / Does NOT Know About" table prevents coupling between components. TDD structure with explicit failing-test-then-implement steps is correct.

---

## Questions for the Team

1. Has anyone successfully run `hf_hub_download("Zaafan/sitesense-weights", ...)` and received the files? This is Phase 1 blocking.
2. What is the correct HuggingFace model ID for the vision backbone? Is it DINOv2 or DINOv3?
3. Is there any sample video from the actual target camera (not fuxi-robot) for early validation?
4. What is the fallback plan if RF-DETR SiteSense weights fail and YOLOv8-COCO performs poorly on construction footage?
5. Who is responsible for validating the ≥90% accuracy requirement (PRD NFR-01) after Phase 4 completes?

---

## Verdict

**Planning quality: GOOD.**
**Implementation readiness: BLOCKED** — resolve C1, C2, C3 first.

All three critical issues have been addressed in the dev team response:
- Phase 0B verification section added to the implementation plan
- Contingency paths documented in ADR-004 and ADR-006
- `ReIDHead` architecture corrected with `ReLU` + parameterised `in_dim`
- All HIGH and MEDIUM issues addressed in plan amendments (see `docs/PROGRESS.md` for the full response log)
