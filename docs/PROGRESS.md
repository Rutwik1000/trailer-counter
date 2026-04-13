# Project Progress Log

> Append-only. Newest entries at the top. Update the status table whenever a phase changes state.

## Current Status

| Phase | Status | Last Updated |
|---|---|---|
| Phase 0B — Dependency Verification | Partial (H2 pending) | 2026-04-13 |
| Phase 1 — Foundation & Data | Not started | 2026-04-13 |
| Phase 2 — Detection Baseline | Not started | 2026-04-13 |
| Phase 3 — Tracking + Zone | Not started | 2026-04-13 |
| Phase 4 — Fill Event Counter *(first MVP)* | Not started | 2026-04-13 |
| Phase 5 — Vehicle Re-ID | Not started | 2026-04-13 |
| Phase 6 — Dashboard | Not started | 2026-04-13 |

---

## 2026-04-13 — Phase 0B complete (partial)

**Completed:** C2 (SiteSense), C1 (backbone fallback confirmed), H1 (dataset schema)
**Outstanding:** H2 (RF-DETR API) — needs `!pip install rfdetr` in notebook, then re-run Step 4
**Key findings:**
- `BACKBONE_ID = facebook/dinov2-base`, `BACKBONE_DIM = 768` — Contingency C1A activated (DINOv3 gated, user on waitlist)
- `fuxi-robot` video field is a `VideoDecoder` object (torchcodec), not dict-with-bytes — Task 1.3 frame extraction code updated accordingly
- SiteSense weights confirmed: `rfdetr_construction.pth` + `dinov3_reid_head.pth` both present
**Next:** Re-run H2 with rfdetr installed, then begin Phase 1 on Kaggle

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
