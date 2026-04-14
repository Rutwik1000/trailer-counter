# Indian Site Context — Revised Assumptions & Design Gaps

> Written: 2026-04-14
> Triggered by: visual inspection of real Indian site footage (tractor-trailer images)
>   and site operator clarifications
> Intended audience: external senior developer / LLM reviewer being consulted
>   before Phase 4 implementation begins
> Status: OPEN — decisions pending review before proceeding

---

## What This Document Is

During Phase 3 multi-video testing we tested only on the fuxi-robot/excavator-video
dataset (Chinese excavator footage). When the user shared actual Indian site footage,
several assumptions embedded in the architecture were found to be wrong, incomplete,
or untested for the real deployment context.

This document records:
1. What was assumed (from planning phase, April 13)
2. What the Indian site footage revealed (April 14)
3. The specific design gaps that result
4. Open questions that need a decision before Phase 4 is built

---

## 1. Camera Placement — ASSUMPTION HELD, CLARIFICATION ADDED

### What was assumed
Camera is mounted inside the excavator cab roof, looking downward at the loading area.
The cab upper body rotates; the camera rotates with it. Because of this, the loading
zone always appears in the same image region — justifying a static polygon (ADR-002).

### What the Indian footage showed
One shared image was a mobile phone recording (not deployment footage) that appeared
to show a forward-facing in-cab view. This caused temporary concern about the camera
angle.

**Operator clarification received:**
- The actual deployment camera is a CCTV unit mounted inside the cab
- It is **rear-facing** — looking backward and downward at the loading area behind
  the excavator
- The mobile phone footage was informal reference material only

### Status
**Assumption holds.** The rear-facing top-down CCTV is consistent with the static
polygon approach (ADR-002). The loading zone is always in the same image region
because the camera rotates with the cab.

**One addition needed:** The loading zone polygon must be calibrated on real Indian
site footage, not on fuxi-robot frames. The trolley approach angle and parking
position behind the excavator may differ from the fuxi-robot dataset geometry.

---

## 2. Vehicle Type — ASSUMPTION WRONG

### What was assumed
PRD Section 6 FR-01 says "detect dump trucks and trailers."
ADR-004 evaluated RF-DETR and YOLOv8n against the fuxi-robot dataset, which contains
large Chinese dump trucks (COCO-like, side-entry, full top-down visibility).
The Priority 1 fix plan defaulted the class filter to COCO class 7 (truck).

### What the Indian footage showed
The actual vehicles are **Indian agricultural tractor-trailers**:
- A **tractor** (e.g. John Deere, Mahindra, Sonalika) — small, rounded cab,
  large rear wheels, distinctive profile
- A **trolley/trailer** pulled behind it — flat rectangular bed, wooden or metal
  sides, open top for loading

This combination is fundamentally different from a Western dump truck:

| Property | Western dump truck (fuxi-robot) | Indian tractor-trailer |
|---|---|---|
| Body type | Single unit, hydraulic bed tipping | Tractor + separate trailer |
| Size (top-down) | Large, ~400-600px | Tractor small (~150px), trolley medium (~300px) |
| COCO class | Class 7 (truck) — confirmed | **Unknown** — COCO has no tractor class |
| Visual appearance | Uniform grey/yellow metal | Varied — green/red tractor, wooden trolley |
| Top-down shape | One large rectangle | Two separate shapes connected |

### Design gaps created

**Gap 2A: Class filter {7} will likely miss tractor-trailers**

COCO class 7 (truck) was trained on Western highway trucks shot from the side.
An Indian tractor viewed from above does not resemble these. Possible COCO
classification outcomes for a tractor-trailer from top-down rear:

- Trolley bed (large flat rectangle) → may trigger class 7 (truck) — uncertain
- Tractor body (small, rounded) → may trigger class 2 (car), class 3 (motorcycle),
  or nothing
- Combined → two separate detections at potentially different class IDs

**We do not know which COCO classes Indian tractor-trailers trigger without testing
on real footage. The class filter must default to None (no filter) until confirmed.**

Current plan (PLAN_PRIORITY1_FIXES.md) has been updated to reflect this.

**Gap 2B: Split detection is structurally guaranteed for tractor-trailers**

For Western dump trucks, the "split detection" problem (dets:2 on one vehicle) was
an occasional artifact of YOLO splitting the cab from the bed. For tractor-trailers,
the tractor and trolley are physically separate objects. YOLOv8n may reliably produce
two detections for every tractor-trailer — one for the tractor, one for the trolley.

The 150px merge distance threshold (Priority 1 fix) was sized for a unified dump
truck body. For a tractor + trolley parked behind the excavator, the gap between
the two detection centroids could be 200-400px depending on trolley length.

The merge threshold needs to be validated on real Indian footage, not set by assumption.

**Gap 2C: Re-ID target ambiguity — tractor or trolley?**

For per-vehicle fill counting (the core requirement), we need to identify the same
physical vehicle across multiple visits. Two options:

| Embed from | Pros | Cons |
|---|---|---|
| **Tractor body** | Visually distinctive — brand logo, color, registration paint | Tractor may not always be in frame (trolley overshoots camera view) |
| **Trolley bed** | Always in loading zone when being filled | Generic — all trolleys may look similar (same model, dusty, same color) |
| **Full vehicle crop** | Maximum information | Bounding box unstable when partially visible |

No decision has been made. This needs to be addressed in Phase 5 design but
affects what crop region Phase 4 saves for Re-ID input.

---

## 3. Partial Visibility — NEW, NOT PREVIOUSLY CONSIDERED

### What was assumed
No assumption was explicitly made about partial visibility. The architecture assumed
the vehicle is fully visible in frame during loading.

### What the Indian site reality is
Operator clarification: "the trolley may be just partially visible."

This happens because:
- The trolley is long; the rear half may extend beyond the camera's field of view
- The excavator arm and bucket obscure parts of the vehicle during the loading swing
- The tractor may be outside the frame entirely (only the trolley bed is visible)

### Design gaps created

**Gap 3A: Detection inconsistency**

A partially visible trolley has lower apparent size and different visual texture
than a fully visible one. YOLOv8n may:
- Detect it in some frames and miss it in others (flickering track)
- Assign lower confidence scores → may fall below threshold even at 0.15
- Assign a different bounding box size → centroid shifts → zone test may flip
  between IN and OUT across frames

This would cause the state machine (Phase 4) to see:
- Multiple entry/exit events for a single loading session → overcount
- Or missed exit event → fill event never fires → undercount

**Gap 3B: Re-ID embedding instability from partial crops**

DINOv2 produces embeddings from the full bounding box crop. A partial crop (e.g.,
only the rear half of the trolley visible) produces a different embedding than a
full-vehicle crop of the same vehicle. The cosine similarity threshold of 0.75
(ADR-006) was set assuming full-vehicle embeddings.

If a vehicle is partially visible on visit 2 but fully visible on visit 1, the
gallery will likely not match them — creating a new vehicle_id instead of
incrementing the existing count. This directly breaks the per-vehicle fill count
requirement.

**Gap 3C: Minimum crop size gate needed**

Re-ID should not attempt to embed a bounding box that is below a minimum pixel
area. A crop that is too small (e.g., <50×50px) or too narrow (partial edge
detection) will produce an unreliable embedding that pollutes the gallery.

No minimum crop size gate exists in the current design. It needs to be added in
Phase 5.

---

## 4. Multiple Vehicles Simultaneously — ASSUMPTION WRONG

### What was assumed
ADR-001: "A vehicle that enters the loading zone and then departs = 1 fill.
Acceptable on an active site — vehicles approach the excavator only for loading."

PRD Section 9 (Assumptions): "Trailers on an active site only approach the
excavator for loading — vehicles that enter the zone are being loaded."

### What the Indian site reality is
Operator clarification: "the algo needs to distinguish between parked vehicle
and the one being loaded."

On Indian construction sites, multiple tractor-trailers queue up waiting for
their turn. The typical sequence:

1. Tractor A arrives, trolley parks in loading zone, gets filled, departs → 1 fill
2. Tractor B arrives WHILE Tractor A is being filled, parks nearby (still inside
   the zone polygon), waits
3. Tractor A departs
4. Tractor B moves into exact loading position, gets filled, departs → 1 fill
5. Tractor C arrives while B is being filled... (cycle repeats)

Under the current state machine, Tractor B would be counted as 1 fill when it
departs step 4, even if it was parked inside the zone since step 2 without
receiving any material.

Worse: if Tractor B entered the zone at step 2 and the zone is calibrated loosely
(with expand_factor=1.15 or 1.20), it may be counted even though it was only
waiting.

### Design gaps created

**Gap 4A: ADR-001 "entry+departure = fill" assumption is invalid for multi-vehicle sites**

The fill event trigger needs to distinguish:
- Vehicle in loading position (directly under excavator arm, actively receiving material)
- Vehicle parked in approach/waiting area (inside the broader zone polygon)

Three architectural approaches, each with trade-offs:

**Option A — Tighten zone to active loading spot only**
Calibrate the polygon to cover only the exact position directly under the
excavator arm — small enough that a waiting vehicle cannot physically fit inside
it while another is being loaded. Only the vehicle in active loading position
has its centroid inside this tight zone.

- Pro: No change to state machine logic or ADR-001
- Con: Requires very precise calibration; active vehicle may drift during loading;
  approach varies by driver; polygon may need recalibration frequently

**Option B — Two-zone model (loading zone + approach zone)**
Define two polygons:
- `loading_zone`: the tight active loading area (only one vehicle fits)
- `approach_zone`: the broader area where vehicles queue

Count a fill only when a vehicle:
1. Enters the `approach_zone`
2. Subsequently enters the `loading_zone`
3. Subsequently departs the `loading_zone`

- Pro: Matches site reality closely; eliminates waiting-vehicle miscounts
- Con: Doubles calibration effort; operator must draw two polygons; adds
  state machine complexity

**Option C — Minimum dwell time gate**
Count a fill only when a vehicle was inside the loading zone for ≥ N consecutive
frames (e.g., N = 50 frames at 25fps = 2 seconds).

A vehicle just passing through or parking briefly would not trigger a fill.

- Pro: Simplest code change to ADR-001 state machine
- Con: ADR-001 explicitly rejected dwell time ("requires per-site tuning and
  varies by trailer size"). A parked vehicle that waits 10 minutes would still
  be counted. Does not distinguish waiting from loading — only eliminates
  drive-through false positives.

**No decision made. This needs input from the reviewer.**

---

## 5. Per-Vehicle Counting — ELEVATED TO CORE REQUIREMENT

### What was assumed
Per-vehicle counting was a Phase 5 (Re-ID) feature — useful but not the primary
deliverable. Phase 4 was designed to produce raw fill event counts first; Re-ID
would layer vehicle identity on top.

### What the operator clarified
"It is essential to accurately count how many times each distinct vehicle was filled."

This means per-vehicle counting is not optional — it is the primary output requirement.
Total fills is a derived metric; fills-per-vehicle is the ground truth.

### Implications

**Gap 5A: Phase 4 state machine must be designed with Re-ID in mind from day one**

The current Phase 4 design outputs fill events with `track_id`. Phase 5 then maps
`track_id` → `vehicle_id`. This two-phase approach has a problem: a vehicle that
returns 3 times gets 3 different `track_id`s (BotSort resets between visits). If
Phase 4 doesn't link those 3 tracks to one vehicle, the Phase 5 Re-ID gallery
may not have enough context to stitch them together after the fact.

The state machine needs to query the Re-ID gallery at the time of zone entry, not
just at the end of processing.

**Gap 5B: Re-ID similarity threshold 0.75 is untested for Indian tractor-trailers**

The 0.75 cosine similarity threshold (ADR-006) was set based on the SiteSense
domain adaptation claims (96.8% accuracy on 12k construction contrastive pairs).
It has not been validated on:
- Indian tractor-trailers (different from Chinese/Western construction vehicles)
- Partial crops (gap 3B)
- Dusty conditions degrading visual features

If 0.75 is too high, the same vehicle won't match across visits → overcounts.
If 0.75 is too low, different vehicles match → undercounts (merges separate counts).

---

## Summary: Original Assumptions vs Reality

| # | Original Assumption | Reality | Impact |
|---|---|---|---|
| A | Camera is top-down, rear-facing CCTV | **Confirmed correct** | None |
| B | Static polygon valid (camera rotates with cab) | **Confirmed correct** | None |
| C | Vehicles are Western-style dump trucks | **Wrong — Indian tractor-trailers** | Class filter, merge threshold, Re-ID target |
| D | One vehicle in loading zone at a time | **Wrong — vehicles queue inside zone** | State machine, fill event logic |
| E | Vehicle is fully visible during loading | **Wrong — partial visibility common** | Re-ID embeddings, detection consistency |
| F | fuxi-robot dataset is representative of deployment | **Approximately wrong** — different vehicle type, similar camera angle | All quantitative results from Phase 2-3 testing are suspect |
| G | Per-vehicle counting is Phase 5 enhancement | **Wrong — it is the core requirement** | Phase 4 must be Re-ID-aware from design |
| H | Class 7 (truck) is the correct COCO filter | **Unknown — COCO has no tractor class** | Class filter must default to None until tested |
| I | Re-ID threshold 0.75 is appropriate | **Untested on Indian vehicles** | Threshold validation needed on real footage |
| J | ADR-001: entry + departure = 1 fill | **Breaks for queuing vehicles** | State machine needs redesign or tight zone calibration |

---

## What Needs a Decision Before Phase 4

The following cannot be reasonably deferred. Phase 4's state machine logic depends
directly on decisions A and B below.

**Decision A: How to handle multiple vehicles in the zone simultaneously**

Choose one:
1. Tight zone polygon (calibrate to active loading spot only) — simplest but
   requires precise operator calibration
2. Two-zone model (loading + approach zones) — most accurate but more operator
   burden and more code
3. Minimum dwell time (N frames) — simplest code change but doesn't solve the
   parked-vehicle problem, only the drive-through problem

**Decision B: What to crop for Re-ID**

The tractor body is visually distinctive but may not always be in frame.
The trolley bed is always in the loading zone but may look identical across
vehicles. Choose which crop region Phase 4 saves for Re-ID.

**Decision C: Class filter strategy**

Until real Indian footage is available:
- Default to `None` (no class filter) — zone polygon is the primary spatial filter
- Or: run YOLOv8n on a sample of real Indian footage and observe which class IDs fire

**Decision D: Partial visibility — minimum crop size gate**

Should Phase 5 Re-ID skip embedding a detection whose bounding box is below a
minimum pixel area? If yes, what is the threshold?

---

## What Is Safe to Proceed With Now (Before Decisions)

The following Priority 1 fixes from PLAN_PRIORITY1_FIXES.md do not depend on
the above decisions and can be implemented immediately:

| Fix | Safe to implement now? | Reason |
|---|---|---|
| Class filter (default None) | Yes | Not filtering is safer than filtering wrong class |
| Detection merge (configurable threshold) | Yes | Infrastructure is right; threshold TBD from real footage |
| Zone expand_factor in config | Yes | Calibration will happen on real footage anyway |
| Tracker.reset() | Yes | No dependency on vehicle type or site layout |
| Night footage — document as known limitation | Yes | Still true regardless of vehicle type |

---

## Prompt for External Senior Developer / LLM Reviewer

```
Hi — I'm building a computer vision pipeline to count tractor-trailer fill events
at Indian construction sites. I need design input on 4 open decisions before
implementing the Phase 4 state machine. Full context below.

SYSTEM OVERVIEW
- Camera: CCTV unit mounted inside excavator cab roof, rear-facing, looking
  backward and downward at the loading area
- The excavator upper body (cab) rotates — camera rotates with it
- The loading zone is always in the same image region (static polygon, one-time
  calibration per camera installation)
- Vehicles: Indian agricultural tractor-trailers (tractor + separate trailing
  trolley/trailer bed). Not Western dump trucks.
- A "fill event" = vehicle enters loading zone AND departs after being loaded
- Pipeline: CLAHE preprocessing → YOLOv8n detection → BoxMOT BotSort tracking
  → centroid-in-polygon zone test → fill event state machine → Re-ID gallery
  (DINOv2 + SiteSense projection head → 128-d embeddings)
- PRIMARY OUTPUT: fills per distinct vehicle per day (not just total fills)

WHAT HAS CHANGED FROM ORIGINAL DESIGN
We built against a Chinese excavator dataset (fuxi-robot/excavator-video) that has
large Western-style dump trucks. The actual deployment is Indian tractor-trailers.
Key differences:
- Tractor-trailer is two physically separate objects (tractor + trolley)
  → YOLOv8n may always produce 2 detections per vehicle
- COCO has no tractor class → unknown which COCO class IDs fire on Indian
  tractor-trailers viewed from above
- Vehicles queue up — multiple tractor-trailers may be inside the loading zone
  polygon simultaneously (one being loaded, others waiting)
- The trolley may be only partially visible in frame (tractor out of view,
  or excavator arm obscuring part of the trolley during loading)
- Per-vehicle counting is the core requirement, not a Phase 5 enhancement

OPEN DECISION 1: Parked vehicle vs being-loaded vehicle
Multiple vehicles queue inside the loading zone polygon. The current state machine
fires a fill event for any vehicle that enters the zone and departs — including
vehicles that were only waiting, not being loaded. How would you distinguish
"actively being loaded" from "parked and waiting" WITHOUT:
- Installing additional sensors
- Detecting the loading action (material falling) — too complex for MVP
- Access to GPS or IoT data from the tractor

Options considered:
  a) Calibrate a very tight zone polygon covering only the exact loading spot
     (one vehicle fits at a time) — operator calibration burden
  b) Two-zone model: loading_zone (tight) + approach_zone (looser); count only
     departures from the loading_zone after having been in both
  c) Minimum dwell frames threshold — rejects drive-through false positives but
     does NOT solve the parked-vehicle problem (a vehicle parked for 10 min
     still counts)
What would you recommend? Is there a standard pattern in MOT / zone counting
systems for this scenario?

OPEN DECISION 2: What to crop for Re-ID
For per-vehicle identity matching, we use DINOv2 backbone + SiteSense projection
head (128-d L2-normalized embeddings, cosine similarity threshold 0.75).
The vehicle has two distinct parts:
  a) Tractor body: visually distinctive (brand logo, colour, registration markings)
     but may be out of frame when only the trolley is in the loading zone
  b) Trolley/trailer bed: always in loading zone during filling, but likely looks
     similar across multiple vehicles (same model, same dusty colour, generic flat bed)
  c) Full vehicle crop: maximum info but unstable when partially visible
Which crop region would you embed for Re-ID? Is there a multi-crop strategy
(embed both and fuse) that would be practical for MVP?

OPEN DECISION 3: Partial visibility handling
The trolley may be partially visible (tractor out of frame, or excavator arm
occluding part of the trolley during loading). This causes:
  - Flickering track (BotSort loses/regains the detection)
  - Inconsistent Re-ID embeddings (partial crop ≠ full-vehicle crop)
  - Centroid instability (small bounding box → centroid shifts → zone test flips)
What is the standard approach for Re-ID under partial occlusion? Specifically:
  a) Should we gate Re-ID on minimum bounding box size?
  b) Should we use EMA embedding updates (weight recent embeddings less when
     bounding box area drops below some threshold)?
  c) Should BotSort track confidence be used to filter out low-quality detections
     before Re-ID?

OPEN DECISION 4: COCO class filter strategy
COCO has no tractor class. We don't know which class IDs an Indian tractor-trailer
triggers when viewed from above. Current plan: default to no class filter (zone
polygon is the primary spatial filter) until we test on real Indian footage.
Is this a reasonable approach for MVP? Or would you recommend proactively testing
with GroundingDINO / open-vocabulary detector to understand the domain gap first?

CODEBASE
- GitHub: https://github.com/Rutwik1000/trailer-counter
- Key files: src/detector.py, src/tracker.py, src/zone.py
- docs/DECISIONS.md — 9 ADRs with alternatives already considered and rejected
- docs/PHASE3_FINDINGS.md — Phase 3 multi-video testing results (fuxi-robot dataset)
- docs/PLAN_PRIORITY1_FIXES.md — planned fixes for Phase 3 issues
- This document: docs/INDIAN_SITE_CONTEXT.md

Please read docs/DECISIONS.md before suggesting architectural changes —
several alternatives (dwell time, IoU zone test, DeepSort tracker) have already
been considered and rejected with documented reasons.

We are about to start Phase 4 (fill event state machine). Decisions 1 and 2
are blockers — the state machine logic depends on them directly.
```
