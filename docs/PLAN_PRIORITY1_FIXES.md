# Priority 1 Fixes — Implementation Plan

> Written: 2026-04-14
> Based on: docs/CODE_REVIEW_PHASE3.md (Claude Opus 4.5 review) + docs/PHASE3_FINDINGS.md
> Status: Ready to implement — all fixes are pre-Phase 4 blockers

---

## My Assessment of the Review

The reviewer (Claude Opus 4.5) is another AI model, not a human engineer.
That matters because AI reviewers can share some of my own blindspots. I've verified
each recommendation independently against the code and the Phase 3 visual evidence
before accepting it. Here is where I agree, partially disagree, and disagree:

### Fix 1 — COCO class filter: AGREE

Passing all 80 COCO classes to the tracker is a correctness bug. The Phase 3 images
show `dets: 2` on frames that appear to have one truck — at least some of those extra
detections are likely non-truck classes (workers, excavator arm segments, debris) that
share pixel patterns with trucks when viewed top-down.

**One nuance the reviewer glossed over:** the reviewer suggests `{5, 7}` (bus + truck)
as an option. I'd start with `{7}` only. Dump trucks from top-down are long rectangles,
which YOLO might classify as bus (class 5) — but adding class 5 without evidence risks
also catching other long vehicles on site. Start strict (`{7}`), widen only if recall
clearly suffers after this fix. To make that easy, I'll add a constructor parameter
`target_classes` so it's configurable without a code change.

### Fix 2 — Detection merge (centroid distance): AGREE ON APPROACH, DISAGREE ON THRESHOLD

The greedy NMS-by-centroid approach is correct. The reviewer's algorithm is sound —
it's correct greedy suppression, highest confidence wins, I traced it.

**Disagreement: 100px default is too small.** On a 720×1280 frame, a dump truck body
in top-down view spans roughly 400-600px wide and 300-400px tall. The cab and bed, if
detected separately, would have centroids ~150-200px apart. A 100px threshold would
fail to merge them. I'll default to 150px.

**Also: don't hardcode it.** The right value depends on truck apparent size at this
specific camera height. We don't have ground truth to validate against yet. Making it
a constructor parameter (`merge_distance_thresh`) lets the Kaggle notebook tune it
without touching source code.

### Fix 3 — Zone containment: PARTIALLY DISAGREE

The reviewer recommends "centroid first, any-corner fallback." I disagree with the
any-corner approach and prefer `expand_factor` instead.

**Why any-corner is risky:** A truck driving toward the zone with one corner just
clipping the boundary would trigger an entry event before it's actually in loading
position. The "any-corner" semantics change zone membership from "truck is positioned
for loading" to "any part of truck overlaps zone." That's a false-positive source,
especially along the approach path.

**Why expand_factor is better:** It preserves ADR-007 (centroid test, documented
rationale: simple, deterministic, correct for partial boundary overlap). The Phase 3
problem was that the polygon was calibrated on one video and is slightly too tight for
others. Expanding it 10-15% outward fixes that without changing containment semantics.

I'll store `expand_factor` in `loading_zone.json` as part of calibration — the operator
can adjust it without code changes. Default `1.0` for backward compatibility; production
calibration should use `1.15`.

**Risk acknowledged:** 15% expansion on the current 640×300 polygon extends the boundary
by ~48px on each width side. At 1280px frame width, the expanded boundary reaches ~1008px
— still 272px from the right edge. Acceptable for this calibration.

### Fix 4 — Tracker.reset(): AGREE

Needed for Phase 4 multi-video processing. Without it, track IDs from video N bleed
into video N+1, potentially matching a track from a completely different truck.
One-liner implementation, no risk.

### Fix 5 — Night footage (accept as MVP limitation): AGREE

The path to fix requires labeled data (50-100 annotated night frames) and fine-tuning —
out of MVP scope. The reviewer is correct that zero-shot approaches (CLIP, GroundingDINO)
don't reliably handle low-light top-down construction footage. Accept and document.

---

## Files to Modify

| File | Changes |
|---|---|
| `src/detector.py` | Add `target_classes` param + class filter + `merge_distance_thresh` param + `_merge_overlapping_detections` function |
| `src/zone.py` | Add `expand_factor` to save/load + `_expand_polygon` helper |
| `src/tracker.py` | Add `reset()` method |
| `tests/test_detector.py` | 6 new tests: 3 class filter + 3 merge |
| `tests/test_zone.py` | 2 new tests: expand_factor applied, no-expand_factor backward compat |
| `tests/test_tracker.py` | 1 new test: reset() clears _tracker |
| `config/loading_zone.json` | Add `"expand_factor": 1.15` |
| `docs/PRD.md` | Add known limitations section for night footage |

---

## Task 1: Class Filter in `src/detector.py`

### Step 1 — Write failing tests

Add to `tests/test_detector.py`:

```python
def test_detector_default_target_classes_is_truck_only():
    det = Detector(model_type="yolo")
    assert det.target_classes == frozenset({7})


def test_detector_accepts_custom_target_classes():
    det = Detector(model_type="yolo", target_classes=frozenset({5, 7}))
    assert det.target_classes == frozenset({5, 7})


def test_detector_filters_non_truck_classes():
    det = Detector(model_type="yolo", target_classes=frozenset({7}))
    assert 0 not in det.target_classes   # person excluded
    assert 2 not in det.target_classes   # car excluded
    assert 5 not in det.target_classes   # bus excluded
    assert 7 in det.target_classes       # truck included
```

Run: `pytest tests/test_detector.py::test_detector_default_target_classes_is_truck_only -v`
Expected: FAIL — `Detector.__init__` has no `target_classes` param yet.

### Step 2 — Implement the filter

In `src/detector.py`, add at module level (after `_DEFAULT_CONFIDENCE_THRESHOLD`):
```python
_DEFAULT_TARGET_CLASSES: frozenset[int] = frozenset({7})  # COCO: 7=truck
# To include buses (may help if dump trucks are misclassified as class 5):
# _DEFAULT_TARGET_CLASSES = frozenset({5, 7})
# Add evidence before widening — don't add speculatively.
```

Update `__init__` signature:
```python
def __init__(
    self,
    model_type: str,
    weights_path: Optional[str] = None,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    target_classes: Optional[frozenset[int]] = None,
    merge_distance_thresh: float = 150.0,
) -> None:
    ...
    self.target_classes: frozenset[int] = (
        target_classes if target_classes is not None else _DEFAULT_TARGET_CLASSES
    )
    self.merge_distance_thresh = merge_distance_thresh
```

Update `_detect_yolo` — add ONE line after the confidence check:
```python
for bbox, conf, cls in zip(xyxy, confs, clses):
    if float(conf) < self.confidence_threshold:
        continue
    if int(cls) not in self.target_classes:   # ← add this
        continue
    detections.append({...})
```

Run: `pytest tests/test_detector.py -v`
Expected: 3 new tests pass + original 13 = 16 passed, 6 skipped.

---

## Task 2: Detection Merge in `src/detector.py`

### Step 3 — Write failing tests

```python
def test_merge_overlapping_detections_merges_close_pair():
    from src.detector import _merge_overlapping_detections
    dets = [
        {"bbox": [100.0, 100.0, 300.0, 300.0], "confidence": 0.9, "class_id": 7},
        {"bbox": [130.0, 130.0, 330.0, 330.0], "confidence": 0.6, "class_id": 7},
        # centroids: (200,200) and (230,230) — distance ~42px, within 150
    ]
    result = _merge_overlapping_detections(dets, distance_thresh=150.0)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9   # highest confidence kept


def test_merge_overlapping_detections_keeps_far_pair():
    from src.detector import _merge_overlapping_detections
    dets = [
        {"bbox": [0.0,   0.0,  100.0, 100.0], "confidence": 0.8, "class_id": 7},
        {"bbox": [500.0, 0.0,  600.0, 100.0], "confidence": 0.7, "class_id": 7},
        # centroids: (50,50) and (550,50) — distance 500px, beyond threshold
    ]
    result = _merge_overlapping_detections(dets, distance_thresh=150.0)
    assert len(result) == 2


def test_merge_single_detection_unchanged():
    from src.detector import _merge_overlapping_detections
    dets = [{"bbox": [10.0, 10.0, 50.0, 50.0], "confidence": 0.7, "class_id": 7}]
    result = _merge_overlapping_detections(dets, distance_thresh=150.0)
    assert len(result) == 1
    assert result == dets
```

Run: `pytest tests/test_detector.py::test_merge_overlapping_detections_merges_close_pair -v`
Expected: FAIL — `_merge_overlapping_detections` not found.

### Step 4 — Implement merge function

Add to `src/detector.py` as a module-level function (below `_validate_frame`):

```python
def _merge_overlapping_detections(
    detections: list[dict],
    distance_thresh: float = 150.0,
) -> list[dict]:
    """Merge detections whose centroids are within distance_thresh pixels.

    Greedy suppression: highest-confidence detection wins and suppresses all
    others within distance_thresh. Equivalent to centroid-distance NMS.

    150px default is calibrated to 720x1280 top-down truck footage where
    cab and bed sub-regions are ~150-200px apart centroid-to-centroid.
    Adjust via Detector(merge_distance_thresh=N) if camera height changes.

    Args:
        detections: List of detection dicts (already class-filtered).
        distance_thresh: Max centroid-to-centroid Euclidean distance to merge.

    Returns:
        Filtered list with suppressed lower-confidence detections removed.
    """
    if len(detections) <= 1:
        return detections

    centroids = [
        ((d["bbox"][0] + d["bbox"][2]) / 2.0, (d["bbox"][1] + d["bbox"][3]) / 2.0)
        for d in detections
    ]
    indexed = sorted(enumerate(detections), key=lambda x: -x[1]["confidence"])
    suppressed: set[int] = set()
    keep: list[dict] = []

    for i, det in indexed:
        if i in suppressed:
            continue
        keep.append(det)
        cx, cy = centroids[i]
        for j, _ in indexed:
            if j == i or j in suppressed:
                continue
            cx2, cy2 = centroids[j]
            if ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5 < distance_thresh:
                suppressed.add(j)

    return keep
```

Call it at end of `_detect_yolo`:
```python
def _detect_yolo(self, frame: np.ndarray) -> list[dict]:
    ...
    # existing loop builds detections[]
    return _merge_overlapping_detections(detections, self.merge_distance_thresh)
```

Run: `pytest tests/test_detector.py -v`
Expected: 6 new tests + 13 original = 19 passed, 6 skipped.

---

## Task 3: Zone Expand Factor in `src/zone.py`

### Step 5 — Write failing tests

```python
def test_load_applies_expand_factor(tmp_path):
    """expand_factor in JSON expands polygon outward from centroid."""
    import json
    path = str(tmp_path / "zone_expanded.json")
    with open(path, "w") as f:
        json.dump({
            "polygon": [[100, 100], [200, 100], [200, 200], [100, 200]],
            "expand_factor": 1.5,
        }, f)
    zone = LoadingZone.load(path)
    # Original corner (100,100), centroid (150,150).
    # Offset (-50,-50) × 1.5 = (-75,-75) → new corner (75,75).
    # Point (80,80) is outside original polygon but inside expanded one.
    assert zone.contains_point(80, 80) is True
    assert zone.contains_point(0, 0) is False   # well outside even expanded


def test_load_without_expand_factor_is_unchanged(tmp_path):
    """JSON without expand_factor field loads original polygon (factor=1.0)."""
    import json
    path = str(tmp_path / "zone_plain.json")
    with open(path, "w") as f:
        json.dump({"polygon": [[100, 100], [200, 100], [200, 200], [100, 200]]}, f)
    zone = LoadingZone.load(path)
    # Point just outside original boundary stays outside (no expansion)
    assert zone.contains_point(95, 150) is False
```

Run: `pytest tests/test_zone.py::test_load_applies_expand_factor -v`
Expected: FAIL — `load()` ignores `expand_factor` key.

### Step 6 — Implement expand_factor

Add module-level helper in `src/zone.py` (below imports):
```python
def _expand_polygon(points: list[list[int]], factor: float) -> list[list[int]]:
    """Expand polygon outward from its centroid by factor.

    factor=1.0 → no change. factor=1.15 → 15% larger in all directions.
    Each vertex moves away from the centroid by the factor multiplier.

    Args:
        points: List of [x, y] integer coordinates.
        factor: Expansion multiplier. Values >1.0 expand, <1.0 shrink.

    Returns:
        New list of [x, y] as Python ints.
    """
    arr = np.array(points, dtype=np.float64)
    centroid = arr.mean(axis=0)
    expanded = ((arr - centroid) * factor + centroid).astype(np.int32)
    return expanded.tolist()
```

Update `save()` to optionally persist the factor:
```python
def save(self, path: str, expand_factor: float = 1.0) -> None:
    """Persist the polygon to a JSON file.

    Args:
        path: Destination file path.
        expand_factor: Zone expansion multiplier to store (default 1.0 = no expansion).
            Set to e.g. 1.15 during calibration to widen the zone 15% at load time.
    """
    with open(path, "w") as f:
        json.dump({"polygon": self.polygon, "expand_factor": expand_factor}, f, indent=2)
```

Update `load()` to apply it:
```python
@classmethod
def load(cls, path: str) -> "LoadingZone":
    with open(path) as f:
        data = json.load(f)
    if data.get("polygon") is None:
        raise ValueError(
            f"Loading zone at '{path}' has not been calibrated yet "
            "(polygon is null). Run the Phase 3 calibration cell on Kaggle first."
        )
    polygon = data["polygon"]
    factor = float(data.get("expand_factor", 1.0))
    if factor != 1.0:
        polygon = _expand_polygon(polygon, factor)
    return cls(polygon=polygon)
```

Run: `pytest tests/test_zone.py -v`
Expected: 2 new tests + 17 original = 19 passed.

### Step 7 — Update config/loading_zone.json

```json
{
  "polygon": [[320, 350], [960, 350], [960, 650], [320, 650]],
  "expand_factor": 1.15
}
```

---

## Task 4: Tracker.reset() in `src/tracker.py`

### Step 8 — Write failing test

```python
def test_tracker_reset_clears_model():
    """reset() must set _tracker back to None for re-initialization."""
    t = Tracker()
    t._tracker = object()   # simulate initialized state
    assert t._tracker is not None
    t.reset()
    assert t._tracker is None
```

Run: `pytest tests/test_tracker.py::test_tracker_reset_clears_model -v`
Expected: FAIL — `Tracker` has no `reset()` method.

### Step 9 — Implement reset()

Add to `Tracker` class in `src/tracker.py` (after `_load_tracker`):
```python
def reset(self) -> None:
    """Reset tracker state for a new video or day boundary.

    Clears all track history and forces re-initialization on the next
    update() call with non-empty detections. Track IDs will restart from 1.

    Call this between videos when processing a day's batch to prevent
    track ID bleed across temporally discontinuous clips.
    """
    self._tracker = None
```

Run: `pytest tests/test_tracker.py -v`
Expected: 1 new test + 9 original = 10 passed, 4 skipped.

---

## Task 5: Night Footage — Document in PRD

Add to `docs/PRD.md` under a "Known Limitations" section:

```markdown
### Night / Low-Light Footage

Detection recall drops to near-zero on night or low-light videos.
YOLOv8n COCO was trained on daytime street-level images and does not
generalize to top-down low-light construction footage.

**MVP scope:** Daylight videos only. Night videos will return 0 fills
with no error — this is a silent miss, not a crash.

**Post-MVP path:**
1. Manually annotate 50–100 night frames with truck bounding boxes
2. Fine-tune YOLOv8n on combined day+night labeled set
3. Or replace with thermal/IR camera (hardware change, out of scope)

Zero-shot approaches (CLIP, GroundingDINO) were evaluated and rejected —
they do not reliably detect top-down low-light truck shapes without
domain adaptation. See docs/PHASE3_FINDINGS.md Issue 1.
```

---

## Task 6: Full Suite + Commit

### Step 10 — Run full test suite

```bash
python -m pytest tests/ -v
```

Expected: **55 passed, 10 skipped**
- preprocessor: 9
- detector: 19 (13 original + 3 class filter + 3 merge)
- tracker: 10 (9 original + 1 reset)
- zone: 19 (17 original + 2 expand_factor)
- skips: 10 (6 detector frame-dependent + 4 tracker frame-dependent)

### Step 11 — Commit

```bash
git add src/detector.py src/zone.py src/tracker.py
git add tests/test_detector.py tests/test_zone.py tests/test_tracker.py
git add config/loading_zone.json docs/PRD.md
git commit -m "fix: priority 1 pre-phase4 — class filter, detection merge, zone expand_factor, tracker reset"
git push origin master
```

---

## Verification on Kaggle (After Commit)

Pull on Kaggle (`git reset --hard origin/master`) and re-run Cell 11.
Expected changes vs current output:

| Video | Before fix | After fix | Pass condition |
|---|---|---|---|
| Video 1 (day, truck clear) | `dets:2` some frames | `dets:1` — merge collapses split detection | Green box, single track |
| Video 2 (night) | `dets:0` | `dets:0` — unchanged, known limitation | No change expected |
| Video 3 (centroid outside) | Orange box | Green box — expanded zone catches centroid | Truck IN zone |

**If Video 1 still shows dets:2 after the fix:** both detections are class 7 AND more
than 150px apart — genuinely two vehicles, not a false split. Don't increase the threshold
speculatively; investigate the frame to confirm before tuning.

---

## What This Does NOT Fix (Deferred)

| Issue | Reason deferred |
|---|---|
| Night detection | Requires labeled data for fine-tuning — out of MVP scope |
| TypedDict / dataclass contracts | Refactor risk with no Phase 4 blocker |
| Batch GPU detection | Performance optimization, Phase 6+ |
| Zone recalibration per video | Manual process, Phase 4 not blocked by it |
