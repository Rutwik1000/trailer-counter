# System Architecture

Post-processing pipeline for excavator cab-mounted camera video. Single excavator, single camera, processes recorded MP4 files. Not a live stream. Not multi-excavator.

## System Boundary

**Inputs:**
- `video_file.mp4` — recorded from CCTV inside excavator cab roof, looking down at the loading area
- `config/loading_zone.json` — operator-calibrated polygon (created once at installation)

**Outputs:**
- `data/results/YYYY-MM-DD.json` — daily count per unique vehicle
- Streamlit dashboard (`dashboard/app.py`)

## Data Flow

```
video_file.mp4
  ↓
[preprocessor.py]     CLAHE (L channel, LAB colorspace) — dust/haze handling
  ↓ BGR frame
[detector.py]         RF-DETR or YOLOv8 → bounding boxes + class_id
  ↓ list[{bbox, confidence, class_id}]
[tracker.py]          BoxMOT BotSort → persistent track IDs across frames
  ↓ list[{track_id, bbox, confidence, class_id}]
[zone.py]             PolygonZone containment check per track
  ↓ bool: in_zone per track_id
[event_counter.py]    State machine: zone_entry + departure → fill event
  ↓ fill events with track_id + frame timestamps
[reid_gallery.py]     DINOv3 crop embedding → cosine match → vehicle_id
  ↓
data/results/YYYY-MM-DD.json
  ↓
[dashboard/app.py]    Streamlit: totals, per-vehicle table, timeline
```

## Component Map

| File | Responsibility | Knows About | Does NOT Know About |
|---|---|---|---|
| `src/preprocessor.py` | CLAHE on BGR frame | OpenCV | Detectors, trackers, zones |
| `src/detector.py` | Bbox + class_id per frame | RF-DETR or YOLOv8 | Tracking, zones, counts |
| `src/tracker.py` | Track IDs across frames | BoxMOT BotSort | Zones, counts, Re-ID |
| `src/zone.py` | Point-in-polygon containment | Polygon, OpenCV | Trackers, models, counts |
| `src/event_counter.py` | State machine, fill events | Track IDs, zone containment | Video, models, dashboard |
| `src/reid_gallery.py` | 128-d cosine matching | numpy, sklearn | Video, detectors, dashboard |
| `src/video_annotator.py` | Draw overlays on frame | Zone, track list | Detectors, Re-ID, counts |

## Data Contracts

**Detector output (list of dicts):**
```python
{"bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int}
```

**Tracker output (list of dicts):**
```python
{"track_id": int, "bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int}
```

**Daily results JSON:**
```python
{
  "date": "YYYY-MM-DD",
  "total_fills": int,
  "events": [
    {
      "track_id": int,
      "vehicle_id": int,        # added in Phase 5 (Re-ID)
      "zone_entry_frame": int,
      "departure_frame": int,
      "timestamp": "ISO-8601"
    }
  ]
}
```

## Fill Event State Machine

```
UNSEEN
  → detected + in_zone=True  → IN_ZONE
  → detected + in_zone=False → UNSEEN  (ignored — not in loading area)

IN_ZONE
  → tracker.track_lost() called → FILL EVENT FIRED
  → still detected in zone    → IN_ZONE  (stays until departure)

FILL EVENT FIRED
  → track_id added to counted_tracks set
  → if BoxMOT later reuses same track_id, it re-enters at UNSEEN
    but the counted_tracks guard prevents double-counting
```

## Loading Zone

- **One-time calibration:** Operator clicks polygon corners on the first frame in the setup notebook
- **Persistence:** Saved as `config/loading_zone.json`: `{"polygon": [[x, y], ...]}`
- **Containment test:** Centroid of bbox `((x1+x2)/2, (y1+y2)/2)` tested with `cv2.pointPolygonTest`
- **Why static works:** Camera rotates with the excavator cab — the loading position is always the same region of the image frame
- **Headless fallback:** Manual coordinate entry when no display is available (Kaggle)

## Re-ID Pipeline

```
Vehicle crop (BGR)
  → Backbone: facebook/dinov2-base → CLS token → 768-d
      (DINOv3 gated — Contingency C1A active; see ADR-006 to upgrade if access granted)
  → SiteSense projection head (dinov3_reid_head.pth)
      Linear(768 → 256) → ReLU → Linear(256 → 128) → L2 normalize
  → 128-d unit vector embedding

Per-day gallery  {vehicle_id: embedding_vector}  (resets at day start):
  cosine_similarity(new_emb, all_gallery_embs)
  max_sim >= 0.75  →  existing vehicle_id
                      update embedding: EMA(0.9 × old + 0.1 × new)
  max_sim < 0.75   →  new vehicle_id registered
```

Similarity threshold 0.75 is the starting value. Tune based on real footage (see ADR-005).

## Dust Preprocessing

CLAHE applied to the L (lightness) channel of LAB colorspace before each detection:
- Parameters: `clipLimit=2.0`, `tileGridSize=(8, 8)`
- CPU-only, no GPU required
- Handles light-to-moderate construction dust
- For heavy dust: optional AOD-Net preprocessing (clone `walsvid/AOD-Net-PyTorch`) as a drop-in upstream step

## Platform

| Platform | GPU | Disk | Strategy |
|---|---|---|---|
| Kaggle Notebook (primary) | T4 / P100 | 107 GB | Download full fuxi-robot (85.5 GB) |
| Google Colab (fallback) | T4 / A100 | ~80 GB | `streaming=True` for fuxi-robot |

Streamlit exposed from Kaggle/Colab notebooks via `npx localtunnel --port 8501`.

## Out of Scope (MVP)

- Live video streaming
- Multi-excavator support
- Cross-day Re-ID persistence
- Model fine-tuning (MVP uses pretrained weights only)
- Rain conditions
- Cloud API / web service layer
- User authentication
