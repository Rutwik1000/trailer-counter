# Product Requirements Document — Trailer Counter

**Version:** 1.0
**Date:** 2026-04-13
**Status:** Approved — implementation not yet started

---

## 1. Problem Statement

Excavation and construction sites need to track how many trailer loads are filled by each excavator per day for productivity reporting and billing purposes. Currently this is done manually by a site supervisor watching operations and tallying counts — which is error-prone, labour-intensive, and impossible at night or in poor visibility.

An excavator already has a CCTV camera mounted inside the cab roof for operator safety. This system uses that existing camera feed to automatically count how many trailers were filled per day, and by which vehicles.

---

## 2. Goals

- Automatically count the number of trailer/truck loads filled by a specific excavator per day
- Identify each unique trailer/truck by visual appearance so counts are reported per vehicle
- Remove the need for a human counter at the excavator
- Produce a daily report that can be used for site productivity tracking

---

## 3. Non-Goals (MVP)

- Live / real-time counting (post-processing only in MVP)
- Tracking vehicles across multiple days
- Supporting multiple excavators simultaneously
- Detecting whether a trailer was fully or partially loaded
- Operating in rain conditions
- Cloud deployment or remote access
- Mobile app or push notifications
- Integration with external billing systems

---

## 4. Users

| User | Role | How they use this system |
|---|---|---|
| Site Manager | Primary | Reviews the daily fill count report to track productivity and verify billing |
| Site Supervisor | Secondary | Configures the loading zone polygon at camera setup; may review flagged events |
| Data Analyst | Secondary | Processes historical daily JSON files for trend analysis |

---

## 5. Use Cases

### UC-01: Daily Fill Count Report
**Actor:** Site Manager
**Description:** At end of day, the site manager runs the pipeline on the day's recorded video. The system produces a report showing total fills and fills per unique vehicle.
**Trigger:** End of working day; video file available
**Success:** Report generated with fill count matching manual tally within ±5%

### UC-02: Loading Zone Calibration
**Actor:** Site Supervisor
**Description:** When the camera is first installed (or repositioned), the supervisor opens the calibration notebook and clicks to define the polygon zone on a sample frame.
**Trigger:** New camera installation or relocation
**Success:** `config/loading_zone.json` is saved; zone visible on annotated video output

### UC-03: Review Annotated Video
**Actor:** Site Manager / Supervisor
**Description:** User opens the Streamlit dashboard, selects a date, and plays back the annotated video to visually verify a disputed count.
**Trigger:** Count discrepancy raised by a driver or client
**Success:** Dashboard shows annotated video with bounding boxes, zone polygon, and track IDs

---

## 6. Functional Requirements

### FR-01: Vehicle Detection
The system must detect dump trucks and trailers in each video frame using a pre-trained computer vision model.
- Detect vehicles in all lighting conditions: daytime, nighttime
- Detect vehicles in high dust conditions
- Detection confidence threshold must be configurable (default: 0.3)

### FR-02: Loading Zone Definition
The system must support a one-time operator-defined loading zone polygon.
- Polygon must be drawable interactively on a sample frame (notebook cell)
- Manual coordinate entry must be supported for headless environments
- Zone coordinates must be saved to `config/loading_zone.json` and loaded on each run
- Zone must be drawn as an overlay on any output video

### FR-03: Fill Event Detection
The system must count a fill event when a tracked vehicle enters the loading zone and subsequently departs.
- A vehicle that enters the zone and departs = exactly 1 fill event (regardless of dwell time or load volume)
- Vehicles passing through the camera view without entering the zone must NOT be counted
- Vehicles being loaded by other excavators visible in the frame must NOT be counted (enforced by zone calibration)

### FR-04: Unique Vehicle Identification
The system must group fill events by unique vehicle using visual appearance.
- Two appearances of the same vehicle (colour, shape, body markings) must be assigned the same vehicle ID within a day
- Vehicle identity is based on visual appearance only — no licence plate recognition required
- The vehicle gallery resets at the start of each day's processing run

### FR-05: Daily Count Output
The system must write a structured JSON file per processing run.
- File path: `data/results/YYYY-MM-DD.json`
- Contents: total fills, per-event records (vehicle ID, entry frame, departure frame, timestamp)
- Schema defined in `docs/ARCHITECTURE.md` → Data Contracts

### FR-06: Streamlit Dashboard
The system must provide a web dashboard to view daily results.
- Select processing date
- Display total fills (KPI)
- Display unique vehicle count (KPI)
- Display per-vehicle fill count table with first/last seen timestamps
- Display fill event timeline chart
- Export or view annotated video with overlays

### FR-07: Dust Preprocessing
The system must apply CLAHE preprocessing to each frame before detection to improve visibility in dusty conditions.

---

## 7. Non-Functional Requirements

### NFR-01: Accuracy
- Fill count accuracy: ≥ 90% compared to manual ground truth on test videos
- Vehicle Re-ID accuracy: ≥ 85% (same vehicle correctly matched across appearances in a day)
- False positive rate (non-loading vehicles counted): ≤ 5%

### NFR-02: Processing Speed
- Post-processing pipeline must process 1 hour of video in under 30 minutes on Kaggle T4 GPU
- Dashboard must load a daily result in under 5 seconds

### NFR-03: Robustness
- System must process video recorded in daytime, nighttime, and high-dust conditions without manual parameter changes per clip
- System must handle up to 10 unique vehicles per day

### NFR-04: Reproducibility
- Processing the same video file twice must produce identical results (deterministic)

### NFR-05: Maintainability
- Each `src/` module has a single responsibility (see `docs/ARCHITECTURE.md`)
- Unit test coverage ≥ 80% for `src/zone.py`, `src/event_counter.py`, `src/reid_gallery.py`
- All architecture decisions documented with rationale in `docs/DECISIONS.md`

---

## 8. Constraints

| Constraint | Detail |
|---|---|
| Camera | Existing CCTV, inside cab roof, looking downward — cannot be changed |
| Input format | MP4 video files (recorded, not live stream) |
| No licence plate recognition | Site vehicles may not have readable plates from cab angle |
| Platform | Kaggle Notebook or Google Colab (no dedicated server in MVP) |
| Python version | 3.10 or 3.11 (rfdetr incompatible with 3.12+) |
| Weather exclusion | Rain excluded — water on lens makes detection unreliable |

---

## 9. Assumptions

- The loading position of trailers relative to the camera is consistent enough that a single static polygon zone is sufficient per camera installation
- Trailers on an active site only approach the excavator for loading — vehicles that enter the zone are being loaded
- The fuxi-robot/excavator-video HuggingFace dataset represents camera perspective and operating conditions close enough to the target environment for development and testing
- SiteSense pre-trained weights (RF-DETR + DINOv3 Re-ID head) will perform adequately on the target footage without fine-tuning for MVP

---

## 10. Success Metrics

| Metric | Target | How Measured |
|---|---|---|
| Fill count accuracy | ≥ 90% vs manual tally | Compare system count to supervisor hand-count on 5 test videos |
| False positive rate | ≤ 5% | Count non-loading vehicles that appear in results JSON |
| Re-ID accuracy | ≥ 85% | Manually label 3 videos; compare vehicle_id assignments |
| Processing time | ≤ 30 min / hour of video | Time the Phase 4 pipeline on Kaggle T4 |
| Dashboard load | ≤ 5 seconds | Measure Streamlit startup to first render |

---

## 11. Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| `rfdetr` | ≥ 1.0.0 | RF-DETR detection |
| `ultralytics` | ≥ 8.0.0 | YOLOv8 detection (evaluated vs RF-DETR in Phase 2) |
| `boxmot` | ≥ 17.0.0 | Multi-object tracking with Re-ID |
| `supervision` | ≥ 0.27.0 | Polygon zone detection |
| `transformers` | ≥ 4.56.0 | DINOv3 Re-ID backbone |
| `huggingface_hub` | ≥ 0.23.0 | Model weight downloads |
| `datasets` | ≥ 2.19.0 | fuxi-robot dataset access |
| `streamlit` | ≥ 1.35.0 | Dashboard |
| `opencv-python` | ≥ 4.9.0 | Video I/O, preprocessing, polygon ops |

Full list: `requirements.txt`

---

## 12. Out of Scope (Post-MVP Backlog)

- Live video stream processing
- Multi-excavator dashboard (aggregate counts across fleet)
- Cross-day vehicle identity tracking
- Licence plate recognition
- Geolocation / GPS tagging of fill events
- Automated model fine-tuning on new footage
- Cloud deployment with authentication
- Mobile app
- Integration with payroll or billing systems
- Rain/wet condition support
