# Implementation Plan

The detailed phase-by-phase implementation plan is located at:

**[`docs/superpowers/plans/2026-04-13-trailer-counter-mvp.md`](superpowers/plans/2026-04-13-trailer-counter-mvp.md)**

> The `superpowers/plans/` path is where Claude Code's planning tool writes plan files. The content is standard markdown — the path is not significant.

## What the Plan Contains

- Step-by-step tasks for all 6 phases
- Complete code scaffolding for every `src/` module
- Failing-test-first (TDD) structure for each component
- Git commit messages per task
- Explicit IN SCOPE / OUT OF SCOPE boundaries per phase

## Phase Summary

| Phase | Goal | First Shippable? |
|---|---|---|
| 1 — Foundation & Data | Environment, dataset access, 50 verified frames | No |
| 2 — Detection Baseline | Evaluate RF-DETR vs YOLOv8, pick one | No |
| 3 — Tracking + Zone | BoxMOT + loading zone polygon calibrated on video | No |
| **4 — Fill Counter** | **Daily JSON count of all departures from zone** | **Yes** |
| 5 — Vehicle Re-ID | Per-vehicle counts via DINOv3 gallery | Yes |
| 6 — Dashboard | Streamlit UI with timeline and annotated video | Yes |

See [`PROGRESS.md`](PROGRESS.md) for current status of each phase.
