# Code Review — Phase 3 Implementation

> Reviewed: 2026-04-14
> Reviewer: Claude Opus 4.5
> Scope: src/preprocessor.py, src/detector.py, src/tracker.py, src/zone.py + tests + configs

---

## Overall Verdict: Solid Foundation — 3 Issues to Address Before Phase 4

The architecture is clean, module isolation is respected, and the code quality is high. However, specific issues were found that will cause problems in Phase 4 if not addressed.

---

## 1. Architecture & Approach

**Strengths:**
- Module boundaries are crisp — each file has one responsibility
- Data contracts are explicit and documented
- Lazy loading pattern in `Detector` and `Tracker` is smart — enables local testing without GPU/weights
- Defensive copying in `LoadingZone` constructor prevents mutation bugs
- Immutability respected everywhere — `draw_on_frame`, `apply_clahe`, `detect`, `update` all return new objects

**The approach is correct.** The pipeline (preprocess -> detect -> track -> zone -> count) matches the ADRs and follows standard MOT patterns.

---

## 2. Implementation Quality

### `src/preprocessor.py` — No Issues

Clean, minimal, correct. CLAHE on L channel of LAB is the right approach for dust/haze.

### `src/detector.py` — 2 Issues

**Issue A: No class filtering (COCO class 7 = truck)**

```python
# Line 102-109: Returning ALL classes
for bbox, conf, cls in zip(xyxy, confs, clses):
    if float(conf) < self.confidence_threshold:
        continue
    detections.append({...})  # No class_id filter!
```

COCO has 80 classes. The code passes persons (0), cars (2), buses (5), trucks (7), etc. to the tracker. This causes:
- False tracks on workers, excavator arm, debris
- Increased track ID churn (noise tracks enter/leave zone)
- Potential overcounting in Phase 4

**Fix:** Filter to `class_id in {7}` (truck) or `{5, 7}` (bus + truck):

```python
_TARGET_CLASSES = frozenset({7})  # COCO: 7=truck

# In _detect_yolo:
if float(conf) < self.confidence_threshold:
    continue
if int(cls) not in _TARGET_CLASSES:
    continue
```

**Issue B: Multi-detection merge not implemented**

As documented in PHASE3_FINDINGS Issue 3, YOLOv8 produces 2 detections per truck in some frames. This should be merged here *before* passing to tracker.

**Fix:** Add `merge_overlapping_detections()` function, called at the end of `_detect_yolo` before return:

```python
def merge_overlapping_detections(
    detections: list[dict],
    distance_thresh: float = 100.0  # pixels
) -> list[dict]:
    """
    Merge detections whose centroids are within distance_thresh.
    Keep the one with highest confidence; discard others.
    """
    if len(detections) <= 1:
        return detections
    
    # Compute centroids
    centroids = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))
    
    # Greedy merge: highest confidence first
    detections_sorted = sorted(enumerate(detections), key=lambda x: -x[1]["confidence"])
    keep = []
    suppressed = set()
    
    for i, det in detections_sorted:
        if i in suppressed:
            continue
        keep.append(det)
        cx, cy = centroids[i]
        for j, det2 in detections_sorted:
            if j != i and j not in suppressed:
                cx2, cy2 = centroids[j]
                if ((cx - cx2)**2 + (cy - cy2)**2) ** 0.5 < distance_thresh:
                    suppressed.add(j)
    
    return keep
```

---

### `src/tracker.py` — Minor Suggestion

**Implementation is correct.** The BoxMOT integration is clean.

**Minor suggestion:** Consider adding a `reset()` method for day-boundary resets:

```python
def reset(self) -> None:
    """Reset tracker state for a new video/day. Clears all track history."""
    self._tracker = None
```

This will be useful in Phase 4 when processing multiple videos per day.

---

### `src/zone.py` — 1 Issue (Problem 2 from PHASE3_FINDINGS)

**Issue: Centroid-only test is too strict**

The `bbox_in_zone` method only checks the centroid. As PHASE3_FINDINGS documented, this causes missed detections when trucks are legitimately in the zone but their centroid drifts slightly outside.

**Current code (line 61-74):**
```python
def bbox_in_zone(self, bbox: list[float]) -> bool:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return self.contains_point(cx, cy)
```

**Recommended fix — any corner inside:**
```python
def bbox_in_zone(self, bbox: list[float]) -> bool:
    """Return True if centroid OR any corner of bbox is inside the zone."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # Check centroid first (fast path)
    if self.contains_point(cx, cy):
        return True
    # Fallback: any corner inside
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    return any(self.contains_point(x, y) for x, y in corners)
```

Also consider adding polygon expansion at load time:
```python
@classmethod
def load(cls, path: str, expand_factor: float = 1.0) -> "LoadingZone":
    # ... existing code ...
    polygon = data["polygon"]
    if expand_factor != 1.0:
        polygon = _expand_polygon(polygon, expand_factor)
    return cls(polygon=polygon)

def _expand_polygon(points: list[list[int]], factor: float) -> list[list[int]]:
    """Expand polygon outward from its centroid by a factor (e.g., 1.15 = 15%)."""
    import numpy as np
    arr = np.array(points, dtype=np.float64)
    centroid = arr.mean(axis=0)
    expanded = ((arr - centroid) * factor + centroid).astype(np.int32)
    return expanded.tolist()
```

---

## 3. Test Coverage

| Module | Tests | Coverage Assessment |
|--------|-------|---------------------|
| `preprocessor.py` | 9 tests | Excellent — shape, dtype, immutability, edge cases |
| `detector.py` | 13 tests | Good locally, but skips real detection tests. **Missing: class filter test** |
| `tracker.py` | 9 tests | Good — constructor, validation, output schema |
| `zone.py` | 17 tests | Excellent — containment, boundaries, save/load, immutability |

**Missing tests to add:**

1. **detector.py**: Test that only COCO class 7 (truck) is returned after the class filter fix
2. **detector.py**: Test `merge_overlapping_detections()` once implemented
3. **zone.py**: Test the any-corner containment logic once implemented

---

## 4. Config Files

**`config/detector_choice.json`**: Correct, well-documented with reason field.

**`config/loading_zone.json`**: The polygon `[[320,350], [960,350], [960,650], [320,650]]` is a 640x300 rectangle. On a 720x1280 frame:
- X range: 320-960 (center 640px of 1280 width)
- Y range: 350-650 (middle-lower of 720 height)

This looks reasonable for a top-down loading zone. A 15% expansion would make it:
- X: ~224-1056
- Y: ~282-718

---

## 5. Room for Improvement

### Priority 1 (Before Phase 4):

| Issue | File | Effort | Impact |
|-------|------|--------|--------|
| Add COCO class filter | detector.py | 5 min | Prevents false tracks on workers/debris |
| Add detection merge | detector.py | 15 min | Prevents double-counting |
| Improve zone containment | zone.py | 10 min | Fixes missed zone entries |

### Priority 2 (Nice to have):

| Improvement | File | Notes |
|-------------|------|-------|
| Add `Tracker.reset()` | tracker.py | For day-boundary resets |
| Add `expand_factor` param | zone.py | Configurable expansion at load time |
| Type hints for return values | all | `list[dict[str, Any]]` -> more precise types |
| Dataclasses for contracts | new | Replace `dict` with `@dataclass` for Detection, Track |

### Priority 3 (Post-MVP):

| Improvement | Notes |
|-------------|-------|
| Use `TypedDict` or `dataclass` for Detection/Track | Better type safety |
| Add logging | `logging.debug()` for track lifecycle events |
| Batch detection | Process N frames at once for GPU efficiency |

---

## 6. Summary of Required Fixes Before Phase 4

```
+------------------------------------------------------------------+
|  MUST FIX BEFORE PHASE 4                                         |
+----------------------+----------+--------------------------------+
| Issue                | File     | Why Critical                   |
+----------------------+----------+--------------------------------+
| COCO class filter    | detector | Workers/debris create noise    |
| Detection merge      | detector | 2 dets per truck -> 2x count   |
| Any-corner zone test | zone     | Legitimate trucks marked OUT   |
+----------------------+----------+--------------------------------+
```

---

## 7. What's Done Well (Keep Doing)

1. **Lazy loading** — Detector/Tracker don't load models until needed
2. **Immutability** — All functions return new objects, never mutate
3. **Defensive copying** — `LoadingZone` constructor copies input
4. **Frame validation** — Both detector and tracker validate dtype/shape before processing
5. **Clear data contracts** — Documented in code docstrings AND ARCHITECTURE.md
6. **Skip decorators in tests** — Graceful degradation when frames unavailable
7. **ADR discipline** — Decisions documented with alternatives rejected

---

## 8. Night Footage Limitation (Accept for MVP)

Zero-shot/few-shot object detection without labeled data doesn't exist for this domain. CLIP/GroundingDINO can do open-vocabulary detection but still struggle with low-light top-down footage.

**Recommendation:** Document as known limitation in PRD:

```markdown
### Night Footage
Detection recall drops to near-zero on night/low-light videos. YOLOv8n COCO
was not trained on top-down low-light construction footage. Mitigation requires
either labeled bounding box data (fine-tuning) or domain-specific pretrained
weights. MVP scope: daylight videos only; night videos return 0 fills.
```

If night detection becomes a hard requirement post-MVP, the path is:
1. Manually label ~50-100 night frames with bounding boxes
2. Fine-tune YOLOv8n on the combined day+night labeled set
3. Or: switch to a thermal camera (different hardware, out of scope)

---

## Next Steps

1. Implement Priority 1 fixes (class filter, detection merge, zone containment)
2. Add missing tests for the new logic
3. Re-run Cell 11 multi-video sampler to verify fixes
4. Proceed to Phase 4 (fill event state machine)
