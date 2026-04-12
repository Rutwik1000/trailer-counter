# Trailer Counter

Post-processing computer vision pipeline that counts filled dump truck/trailer loads per unique vehicle per day from excavator cab-mounted CCTV camera video.

**Status:** Phase 0 — Planning complete. No code yet.

---

## What It Does

- **Input:** Recorded MP4 video from a CCTV camera mounted inside the excavator cab roof, looking down at the loading area
- **Output:** Daily JSON count file (`data/results/YYYY-MM-DD.json`) + Streamlit dashboard
- **Fill detection:** A trailer that enters the user-defined loading zone and then departs = 1 fill
- **Identity:** Unique vehicles identified by visual appearance (color, shape, body surface) using DINOv3 Re-ID embeddings
- **Conditions:** Day, night, high dust (excluding rain)

## Architecture Overview

```
video_file.mp4
  → [preprocessor]   CLAHE dust handling
  → [detector]       RF-DETR bounding boxes (dump trucks / trailers)
  → [tracker]        BoxMOT BotSort — persistent track IDs
  → [zone]           PolygonZone containment — loading zone check
  → [event_counter]  State machine: zone entry + departure → fill event
  → [reid_gallery]   DINOv3 128-d cosine match → vehicle_id
  → data/results/YYYY-MM-DD.json
  → [dashboard]      Streamlit — totals, per-vehicle table, timeline
```

Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Quick Start *(to be implemented in Phase 4 — no code exists yet)*

```bash
# 1. Install dependencies (Python 3.10 or 3.11 required)
pip install -r requirements.txt

# 2. Calibrate the loading zone — one-time per camera installation
#    Open notebooks/03_tracking_zone.ipynb → run the calibration cell

# 3. Process a video
python run_pipeline.py --video path/to/video.mp4 --date 2026-04-13

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

> `run_pipeline.py` will be created in Phase 4. See [`docs/PROGRESS.md`](docs/PROGRESS.md) for current phase status.

## Repository Layout

```
trailer-counter/
├── src/                  # Core modules: detector, tracker, zone, counter, reid
├── dashboard/            # Streamlit web app
├── config/               # Loading zone polygon JSON (operator-calibrated)
├── data/
│   ├── frames/           # Extracted sample frames (dev/testing)
│   └── results/          # Daily count JSON outputs
├── models/               # Downloaded model weights (gitignored — use hf_hub_download)
├── notebooks/            # One Jupyter notebook per development phase
├── tests/                # pytest unit tests for src/ modules
└── docs/
    ├── ARCHITECTURE.md   # System design, data flow, component contracts
    ├── DECISIONS.md      # Architecture Decision Records (ADRs)
    ├── PROGRESS.md       # Phase-by-phase status log
    ├── RESEARCH_SUMMARY.md
    ├── research/         # Raw research source files
    └── superpowers/plans/  # Detailed implementation plan with code scaffolding
```

## Tech Stack

| Component | Library |
|---|---|
| Detection | `rfdetr` (RF-DETR, SiteSense weights) or `ultralytics` (YOLOv8) — evaluated in Phase 2 |
| Tracking | `boxmot` (BotSort) |
| Loading zone | `supervision` (PolygonZone) |
| Re-ID backbone | `transformers` — DINOv3 ViT-B/16 |
| Re-ID head | SiteSense projection head (`Zaafan/sitesense-weights`) |
| Dust preprocessing | OpenCV CLAHE |
| Dashboard | `streamlit` + `plotly` |
| Platform | Kaggle Notebook (primary) or Google Colab + Drive |

## Development Phases

| # | Phase | Output | Status |
|---|---|---|---|
| 1 | Foundation & Data | 50 verified frames from fuxi-robot dataset | Not started |
| 2 | Detection Baseline | Detector selected and scored on sample frames | Not started |
| 3 | Tracking + Zone | Tracked video with loading zone overlay | Not started |
| **4** | **Fill Event Counter** | **Daily JSON count — first MVP** | Not started |
| 5 | Vehicle Re-ID | Per-vehicle count breakdown | Not started |
| 6 | Dashboard | Streamlit UI | Not started |

Full implementation plan (with code): [`docs/superpowers/plans/2026-04-13-trailer-counter-mvp.md`](docs/superpowers/plans/2026-04-13-trailer-counter-mvp.md)

## Not In Scope (MVP)

Live streaming · Multi-excavator · Cross-day Re-ID · Model fine-tuning · Rain conditions · Cloud deployment · Authentication

## Docs Index

| Document | Purpose |
|---|---|
| [`docs/PRD.md`](docs/PRD.md) | Product requirements, use cases, success metrics |
| [`docs/PLAN.md`](docs/PLAN.md) | Implementation plan index — phases, tasks, code scaffolding |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, data flow, component contracts |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Architecture Decision Records (9 ADRs) |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Phase-by-phase status log |
| [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) | Distilled research findings |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code project instructions and coding rules |
| [`docs/research/`](docs/research/) | Raw research source files |
