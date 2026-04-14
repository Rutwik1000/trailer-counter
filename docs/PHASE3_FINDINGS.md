# Phase 3 Multi-Video Testing — Findings & Issues

> Written: 2026-04-14. Based on Cell 11 multi-video sampler output across 3 videos.
> Intended audience: external senior developer being brought in to advise on Phase 4+.

---

## What Was Tested

After Phase 3 integration (zone calibration + BoxMOT BotSort tracker) was verified on a single video,
a multi-video sampler cell (Cell 11, `notebooks/03_tracking_zone.ipynb`) was run across 3 different
videos from the `fuxi-robot/excavator-video` dataset. The same `config/loading_zone.json` polygon and
`config/detector_choice.json` (YOLOv8n COCO, threshold=0.15) were used for all three.

Each video produced a 2-row × 5-column grid of annotated frames.
Annotations: yellow = zone polygon, green box = detected truck centroid inside zone, orange box = outside zone.

---

## Issue 1: Zero Detections on Night/Low-Light Videos

**Expected:** YOLOv8n COCO would detect at least 1–2 trucks per frame across all video types
(day, night, dust).

**Actual:** Video 2 showed `dets: 0` on every frame. The scene was a night shoot; the truck body
blended with the dark surroundings and was not recognizable by YOLOv8n.

**Why this matters:** A video with zero detections produces zero tracking events. The fill counter
(Phase 4) will silently return 0 fills for these videos — no error, just a wrong answer.

**Root cause:** YOLOv8n was trained on COCO, which contains street-level side-view daytime truck
images. The fuxi-robot dataset includes night footage, adverse lighting, and dust — none of which
appear in COCO. This was anticipated as a risk in ADR-004 ("Recall: low — expected, COCO truck class
trained on street-level side views vs top-down cab footage") but the severity on night footage is
worse than expected.

**Assumption that broke:** We assumed threshold=0.15 (lowered from 0.25 after Phase 2) would be
sufficient to catch trucks in varied conditions. It is not sufficient for night footage.

---

## Issue 2: Zone Polygon Does Not Generalize Across Videos

**Expected:** A single polygon calibrated on the first video's first frame would correctly capture
the loading zone across all videos from the same camera, because the camera is fixed to the excavator
cab and rotates with it (ADR-002).

**Actual:** Video 3 showed trucks detected but with orange boxes (centroid outside zone) on several
frames. The truck was visibly present and partially inside the polygon, but its centroid fell outside
the yellow boundary — meaning the fill counter would never fire an entry event for these trucks.

**Why this matters:** If zone entry never triggers, the state machine in Phase 4 can never fire a
fill event — even though the truck was genuinely loaded.

**Root cause:** The polygon was drawn on the first frame of video 1. Although the camera angle is
consistent within a single video, there is measurable variation across videos in where the truck
parks relative to the frame (different truck sizes, different approach angles, slight cab rotation
at rest position). The centroid test (ADR-007) is sensitive to this — a truck that is 80% inside
the polygon but whose centroid is 10px outside will be counted as OUT.

**Assumption that broke:** ADR-002 states "the loading position is always in the same image region
relative to the camera." This holds approximately but not exactly across all videos. The variation
is small enough that a larger polygon would cover it, but the current polygon is too tight.

---

## Issue 3: Multi-Detection Per Frame (Possible Double-Count Risk)

**Expected:** One truck in the loading zone = one detection per frame = one track ID.

**Actual:** Several frames across video 1 and video 3 showed `dets: 2`. In the annotated images,
two bounding boxes appeared on what appeared to be a single truck body — the detector split the
truck into two overlapping detections (e.g., cab region + bed region detected separately).

**Why this matters:** BoxMOT BotSort may assign two separate track IDs to what is one physical
truck. If both tracks independently enter and leave the zone, Phase 4's fill counter would fire
two fill events instead of one — a 2× overcount.

**Root cause:** YOLOv8n COCO's `truck` class was trained on whole-vehicle annotations. On top-down
footage, partial occlusion by the excavator arm or the loading chute can cause the model to treat
visible sub-regions as separate detections. NMS (Non-Maximum Suppression) is applied internally by
ultralytics but may not suppress boxes that don't overlap enough in IoU terms.

**Assumption that broke:** We assumed one physical truck = one detection per frame. This is true
for street-level footage but not reliably true for top-down footage with partial occlusion.

---

## Summary Table

| # | Issue | Expected | Actual | Phase 4 impact |
|---|---|---|---|---|
| 1 | Night video — 0 detections | ≥1 det/frame across conditions | 0 dets on all night frames | Silent miss — 0 fills counted |
| 2 | Zone too tight — centroid outside | Single polygon covers all videos | Truck IN frame but centroid OUT | Fill entry never triggers |
| 3 | Multi-detection per truck | 1 det per truck per frame | 2 dets on single truck (`dets:2`) | Possible 2× overcount |

---

## Assumptions Made So Far (Full List)

These were made during planning (April 13) and Phase 1–3 execution. Some have been confirmed,
some have broken, some are untested.

| Assumption | Status | Evidence |
|---|---|---|
| Camera rotates with cab → static polygon valid | **Partially broken** | Polygon fits video 1 well; videos 2–3 show centroid drift |
| YOLOv8n at threshold=0.15 gives acceptable recall | **Broken for night** | Video 2: 0 dets across 10 frames |
| One truck = one detection per frame | **Broken** | `dets: 2` on single-truck frames in videos 1 and 3 |
| fuxi-robot dataset represents real site conditions | **Unverified** | Dataset is synthetic/controlled lab data, not confirmed real-site footage |
| BoxMOT BotSort handles cab rotation robustly | **Untested** | Only tested on 30 consecutive frames of one video — no rotation observed |
| Cosine similarity threshold 0.75 for Re-ID | **Untested** | Phase 5 not started |
| Day-scoped Re-ID gallery is sufficient | **Untested** | Phase 5 not started |
| Single excavator, single loading zone | **Confirmed** | All videos show one excavator arm, one loading area |
| CLAHE improves detection recall | **Confirmed** | Phase 1 visual verification; contrast enhancement visible |
| DINOv2-base (768-d) sufficient for Re-ID | **Untested** | Phase 5 not started |

---

## What Is NOT Broken

- Zone polygon draw + save/load mechanics: working correctly
- BoxMOT BotSort track ID persistence: 1 persistent track (>=3 frames) confirmed on video 1
- CLAHE preprocessing: working
- Detector + tracker + zone pipeline end-to-end: working for well-lit videos with truck in frame
- All 48 local tests: passing

---

## Prompt for External Senior Developer

```
Hi — I'm building a computer vision pipeline to count dump truck fill events from video
recorded by a CCTV camera mounted inside an excavator cab roof (top-down view of the
loading area). I need a second opinion on three problems that surfaced during Phase 3
multi-video testing. Here's the full context:

SYSTEM OVERVIEW
- Camera: fixed inside excavator cab roof, looking straight down at the trailer
- The cab (upper body) rotates — camera rotates with it
- A "fill event" = trailer enters the loading zone polygon AND departs
- Pipeline: CLAHE preprocessing → YOLOv8n COCO detection → BoxMOT BotSort tracking →
  centroid-in-polygon zone test → fill event state machine (Phase 4, not yet built)
- Dataset: fuxi-robot/excavator-video (HuggingFace) — 1,000 videos, 10 trucks,
  day/night, adverse weather. 720×1280px frames.
- Detector: YOLOv8n pretrained on COCO (threshold=0.15). RF-DETR with domain-specific
  weights was the original plan but failed on Kaggle's Python 3.12 environment.

PROBLEM 1 — ZERO DETECTIONS ON NIGHT FOOTAGE
YOLOv8n detects nothing on night/low-light frames (confirmed across a full 10-frame
sample). The COCO truck class was trained on daytime street-level images; top-down
night footage is out of distribution. We cannot fine-tune (no labeled bounding box
data for this camera setup). Options we are considering:
  a) Lower threshold further (0.10 or below) — risk: false positives
  b) CLAHE + histogram equalization as night-specific preprocessing — may help
  c) Use a different pretrained model with better low-light generalization
  d) Accept night misses and note it as a known limitation in the PRD
What would you recommend? Is there a zero-shot or few-shot approach that doesn't
require labeled data?

PROBLEM 2 — ZONE POLYGON TOO TIGHT ACROSS VIDEOS
A single 4-point polygon was calibrated on one video's first frame. On other videos,
the truck is detected (bounding box visible) but its centroid falls outside the polygon
— so the zone test returns False and the fill event never fires. We use cv2.pointPolygonTest
on the centroid ((x1+x2)/2, (y1+y2)/2).
Options:
  a) Expand the polygon by a fixed margin (e.g. 10–15% outward from centroid)
  b) Switch from centroid test to IoU-overlap-with-polygon test (e.g. >20% overlap)
  c) Ask the operator to re-calibrate per camera session rather than once
  d) Dynamically fit the polygon to the first N frames of each video
What containment test would you use for this geometry? Is there a standard approach
for tolerant zone membership that doesn't require retuning per video?

PROBLEM 3 — MULTI-DETECTION PER TRUCK (DOUBLE-COUNT RISK)
On several frames, YOLOv8n produces 2 detections on a single truck body (e.g., the cab
and the truck bed are detected as separate bounding boxes). BoxMOT may assign two track
IDs to one physical truck, causing the fill counter to fire twice per load.
Options:
  a) Apply stricter NMS (lower IoU threshold in ultralytics config)
  b) Post-process: merge any two detections whose centroids are within N pixels
  c) Use track ID clustering — if two track IDs always appear/disappear together,
     treat them as one vehicle
  d) Switch to a detector fine-tuned on top-down construction footage
How would you handle multi-detection deduplication in a tracking pipeline where you
don't have ground truth labels to tune against?

CODEBASE
- GitHub: https://github.com//trailer-counter
- Key files: src/detector.py, src/tracker.py, src/zone.py, config/detector_choice.json,
  config/loading_zone.json, docs/DECISIONS.md (9 ADRs), docs/ARCHITECTURE.md
- All decisions are in docs/DECISIONS.md — please read ADR-001 through ADR-007
  before suggesting architectural changes, as several alternatives were already
  considered and rejected with documented reasons.

We are about to start Phase 4 (fill event state machine). The three problems above
need at least a direction decision before we build the counter logic, because the
counter's correctness depends directly on detection recall (Problem 1), zone entry
accuracy (Problem 2), and track ID reliability (Problem 3).

Any guidance on which problems to fix now vs accept as known limitations for MVP
would also be very helpful.
```
