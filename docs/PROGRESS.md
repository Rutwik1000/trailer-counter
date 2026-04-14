# Project Progress Log

> Append-only. Newest entries at the top. Update the status table whenever a phase changes state.

## Current Status

| Phase | Status | Last Updated |
|---|---|---|
| Phase 0B — Dependency Verification | Complete | 2026-04-13 |
| Phase 1 — Foundation & Data | Complete | 2026-04-14 |
| Phase 2 — Detection Baseline | Not started | 2026-04-13 |
| Phase 3 — Tracking + Zone | Not started | 2026-04-13 |
| Phase 4 — Fill Event Counter *(first MVP)* | Not started | 2026-04-13 |
| Phase 5 — Vehicle Re-ID | Not started | 2026-04-13 |
| Phase 6 — Dashboard | Not started | 2026-04-13 |

---

## 2026-04-14 — Phase 1 complete

**Files created:** src/preprocessor.py, tests/test_preprocessor.py, notebooks/01_foundation.ipynb
**Tests passing:** 9/9 in tests/test_preprocessor.py
**Key findings:**
- fuxi-robot VideoDecoder yields CHW uint8 RGB [0,255] — not float32 [0,1] as plan assumed. Frame conversion updated (no *255).
- Frames confirmed: cab-mounted downward view, excavator arm visible, trailer in consistent loading position. Static polygon zone approach validated.
- CLAHE preprocessing visually verified on sample frames — contrast enhancement working correctly.
- 50 frames extracted from 5 videos (720×1280px, ~2265 frames/video).
**Blocker:** None
**Next:** Phase 2 — Detection Baseline (RF-DETR SiteSense weights vs YOLOv8n-COCO on 10 sample frames)

---

## 2026-04-13 — Phase 0B complete (partial)

**Completed:** C2 (SiteSense), C1 (backbone fallback confirmed), H1 (dataset schema)
**Outstanding:** none — all four checks resolved
**Key findings:**
- `BACKBONE_ID = facebook/dinov2-base`, `BACKBONE_DIM = 768` — Contingency C1A activated (DINOv3 gated, user on waitlist)
- `fuxi-robot` video field is a `VideoDecoder` object (torchcodec), not dict-with-bytes — Task 1.3 frame extraction code updated accordingly
- SiteSense weights confirmed: `rfdetr_construction.pth` + `dinov3_reid_head.pth` both present
- RF-DETR API: `predict` FOUND, `from_checkpoint` MISSING — constructor is `RFDETRBase(pretrain_weights=path)`
**Next:** Begin Phase 1 on Kaggle

---

## 2026-04-13 — Planning complete

**Completed this session:**
- Research phase: two parallel research runs (GPT-5.2 + Claude Sonnet 4.6) — see `docs/research/`
- Tech stack finalized; all libraries verified pip-installable and accessible
- 6-phase MVP implementation plan written with full code scaffolding — see `docs/superpowers/plans/2026-04-13-trailer-counter-mvp.md`
- Architecture decisions recorded (9 ADRs) — see `docs/DECISIONS.md`
- PRD written (requirements, use cases, success metrics) — see `docs/PRD.md`
- Project documentation structure created (CLAUDE.md, README, PRD, ARCHITECTURE, DECISIONS, PROGRESS, RESEARCH_SUMMARY)
- Directory scaffold created (src/, tests/, notebooks/, dashboard/, config/, data/, models/)

**Decisions locked in this session:**
- Trigger: departure-based (not dwell-time; not bucket-dump counting) — ADR-001
- Zone: static operator-drawn polygon, one-time calibration — ADR-002
- Tracker: BoxMOT BotSort (appearance-based, handles cab rotation) — ADR-003
- Detector: RF-DETR vs YOLOv8 evaluation deferred to Phase 2 — ADR-004
- Re-ID gallery: day-scoped only, resets per day — ADR-005
- Re-ID model: DINOv3 ViT-B/16 + SiteSense projection head — ADR-006
- Platform: Kaggle Notebook primary (107 GB disk for fuxi-robot 85.5 GB) — ADR-009

**Nothing built yet:**
No code written. No models downloaded. No notebooks created. No data extracted.

**Next action:**
Begin Phase 1 on Kaggle Notebook — install all libraries, download SiteSense weights from HuggingFace, stream 5 videos from `fuxi-robot/excavator-video`, extract ~50 frames, visually inspect to confirm cab-mounted perspective.

---

## Entry Template

Copy this block when completing a phase:

```
## YYYY-MM-DD — Phase N complete

**Files created:** [list exact paths]
**Tests passing:** [e.g., "5/5 in tests/test_zone.py"]
**Key finding:** [any deviation from the plan during implementation, and why]
**Blocker:** [what caused delay or requires a new decision — or "None"]
**Next:** [what Phase N+1 needs before it can start]
```
