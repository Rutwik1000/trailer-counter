# Trailer Counter MVP — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a post-processing pipeline that takes excavator cab-mounted camera video, detects dump trucks/trailers entering a user-defined loading zone, and counts filled trailers per unique vehicle per day, displayed on a Streamlit dashboard.

**Architecture:** Six sequential phases, each independently testable and shippable. Detection (RF-DETR or YOLOv8) feeds BoxMOT tracking, which feeds a Supervision PolygonZone, which feeds a state-machine fill-event counter, which feeds a DINOv3 Re-ID gallery, which feeds a Streamlit dashboard. Each phase produces a working, validated output before the next begins.

**Tech Stack:** Python 3.10/3.11, `rfdetr`, `ultralytics` (YOLOv8), `boxmot`, `supervision`, `transformers` (DINOv3), `huggingface_hub`, `datasets`, `opencv-python`, `streamlit`, `plotly`, `pandas`, `scikit-learn`, `pytest`

**Platform:** Kaggle Notebook (primary — 107 GB disk for fuxi-robot 85.5 GB dataset) or Google Colab + Drive mount.

---

## Phases Overview

| Phase | Name | Shippable Output | First "real" count available |
|---|---|---|---|
| 1 | Foundation & Data | Verified sample frames from fuxi-robot | — |
| 2 | Detection Baseline | Detector scoring on sample frames | — |
| 3 | Tracking + Loading Zone | Tracked video with zone overlay | — |
| 4 | Fill Event Counter | JSON daily count (all departures) | **YES** |
| 5 | Vehicle Re-ID | JSON daily count per unique vehicle | YES (improved) |
| 6 | Dashboard | Streamlit dashboard with video playback | YES (visible) |

> **Phase 4 is the first shippable MVP.** Phases 5 and 6 add identity breakdown and visual output on top of a working count.

---

## File Structure

```
trailer-counter/
├── src/
│   ├── detector.py          # RF-DETR / YOLOv8 detection wrapper
│   ├── tracker.py           # BoxMOT tracker wrapper
│   ├── zone.py              # Loading zone polygon calibration + containment check
│   ├── event_counter.py     # Fill event state machine (zone entry → departure → count)
│   ├── reid_gallery.py      # Per-day Re-ID gallery manager (DINOv3 embeddings)
│   ├── preprocessor.py      # CLAHE dust preprocessing
│   └── video_annotator.py   # Draw boxes, zone polygon, track IDs on output video
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── config/
│   └── loading_zone.json    # Persisted polygon coordinates (written by zone.py)
├── data/
│   ├── frames/              # Extracted sample frames (Phase 1)
│   └── results/             # Daily count JSON outputs
├── models/                  # Downloaded model weights (gitignored)
├── notebooks/
│   ├── 01_foundation.ipynb
│   ├── 02_detection.ipynb
│   ├── 03_tracking_zone.ipynb
│   ├── 04_counting.ipynb
│   ├── 05_reid.ipynb
│   └── 06_dashboard.ipynb
├── tests/
│   ├── test_zone.py
│   ├── test_event_counter.py
│   └── test_reid_gallery.py
├── requirements.txt
└── README.md
```

**One responsibility per file:**
- `zone.py` only knows about polygons and containment. It does not know about tracks or counts.
- `event_counter.py` only knows about state transitions (entered zone / left zone / count). It does not know about video or models.
- `reid_gallery.py` only knows about embeddings and matching. It does not know about tracking.
- `detector.py` only wraps the chosen model. It does not know about tracking or zones.

---

## Chunk 1: Phases 1–2 (Foundation + Detection)

---

### Phase 1: Foundation & Data Setup

#### IN SCOPE
- Install all libraries
- Download fuxi-robot dataset (streaming subset — do not download all 85.5 GB)
- Download keremberke/excavator-detector dataset (193 MB, download fully)
- Download SiteSense model weights
- Extract 50–100 sample frames from fuxi-robot videos
- Visually confirm frame content matches cab-mounted perspective
- Create `requirements.txt`

#### OUT OF SCOPE — DO NOT ADD
- No model training or fine-tuning
- No pipeline building
- No dashboard
- No zone calibration
- No processing of full videos
- No annotation or labeling

---

#### Task 1.1: Create `requirements.txt`

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Write `requirements.txt`**

```
rfdetr>=1.0.0
ultralytics>=8.0.0
boxmot>=17.0.0
supervision>=0.27.0
transformers>=4.56.0
huggingface_hub>=0.23.0
datasets>=2.19.0
opencv-python>=4.9.0
streamlit>=1.35.0
plotly>=5.20.0
pandas>=2.2.0
numpy>=1.26.0
Pillow>=10.3.0
scikit-learn>=1.4.0
pytest>=8.2.0
shapely>=2.0.0
tqdm>=4.66.0
torch>=2.2.0
torchvision>=0.17.0
```

- [ ] **Step 2: Install in notebook**

```python
!pip install -r requirements.txt -q
```

Expected: all packages install without conflict.

- [ ] **Step 3: Verify imports in notebook cell**

```python
import cv2, supervision as sv, torch, ultralytics
from rfdetr import RFDETRBase
from boxmot import BotSort
from datasets import load_dataset
from huggingface_hub import hf_hub_download
print("All imports OK")
print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
```

Expected: `All imports OK` and `CUDA: True` on Kaggle/Colab GPU.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add requirements.txt for MVP stack"
```

---

#### Task 1.2: Download SiteSense weights

**Files:**
- Create: `models/` directory
- Notebook: `notebooks/01_foundation.ipynb`

- [ ] **Step 1: Download weights**

```python
import os
from huggingface_hub import hf_hub_download

os.makedirs("models", exist_ok=True)

for filename in ["rfdetr_construction.pth", "dinov3_reid_head.pth", "osnet_x0_25_msmt17.pt"]:
    hf_hub_download(
        repo_id="Zaafan/sitesense-weights",
        filename=filename,
        local_dir="models/"
    )
    print(f"Downloaded: {filename}")
```

Expected: three files in `models/`, sizes ~122 MB, ~5.4 MB, ~2.9 MB.

- [ ] **Step 2: Assert files exist**

```python
import os
for f in ["models/rfdetr_construction.pth", "models/dinov3_reid_head.pth", "models/osnet_x0_25_msmt17.pt"]:
    assert os.path.exists(f), f"Missing: {f}"
print("All model weights present")
```

---

#### Task 1.3: Access fuxi-robot dataset and extract sample frames

**Files:**
- Create: `data/frames/` directory
- Notebook: `notebooks/01_foundation.ipynb`

> **Kaggle note:** fuxi-robot is 85.5 GB. Use streaming mode — do NOT call `load_dataset` without `streaming=True` unless you know how many samples you need.

- [ ] **Step 1: Stream dataset and extract frames from 5 videos**

```python
import io
import cv2
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

os.makedirs("data/frames", exist_ok=True)

ds = load_dataset("fuxi-robot/excavator-video", split="train", streaming=True)

frames_saved = 0
video_count = 0
target_videos = 5       # only process 5 videos in Phase 1
frames_per_video = 10   # extract 10 evenly-spaced frames per video

for sample in tqdm(ds):
    if video_count >= target_videos:
        break

    # video is stored as bytes in the parquet
    video_bytes = sample["video"]["bytes"]
    video_array = np.frombuffer(video_bytes, dtype=np.uint8)
    cap = cv2.VideoCapture()
    cap.open(cv2.imdecode(video_array, cv2.IMREAD_UNCHANGED))

    # Alternative: write bytes to temp file then open
    tmp_path = f"/tmp/vid_{video_count}.mp4"
    with open(tmp_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // frames_per_video)

    for i in range(frames_per_video):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            out_path = f"data/frames/vid{video_count:02d}_frame{i:02d}.jpg"
            cv2.imwrite(out_path, frame)
            frames_saved += 1

    cap.release()
    video_count += 1

print(f"Saved {frames_saved} frames from {video_count} videos")
```

- [ ] **Step 2: Assert minimum frames extracted**

```python
import glob
frame_files = glob.glob("data/frames/*.jpg")
assert len(frame_files) >= 40, f"Expected ≥40 frames, got {len(frame_files)}"
print(f"Frame extraction OK: {len(frame_files)} frames")
```

- [ ] **Step 3: Display a 3×3 grid of sample frames to visually inspect**

```python
import matplotlib.pyplot as plt
from PIL import Image

sample_paths = sorted(glob.glob("data/frames/*.jpg"))[:9]
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for ax, path in zip(axes.flatten(), sample_paths):
    img = Image.open(path)
    ax.imshow(img)
    ax.set_title(os.path.basename(path), fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("data/frames/inspection_grid.png")
plt.show()
print("Inspect the grid — confirm: excavator arm visible, trailer below, cab-mounted perspective")
```

> **Human checkpoint:** Before proceeding to Phase 2, visually confirm the frames show the cab-mounted downward view. If the perspective is wrong, the detector fine-tuning strategy must be revised.

- [ ] **Step 4: Commit**

```bash
git add notebooks/01_foundation.ipynb requirements.txt
git commit -m "feat: phase 1 - foundation, dataset access, frame extraction"
```

---

### Phase 2: Detection Baseline

#### IN SCOPE
- Test RF-DETR SiteSense weights on 10 sample frames
- Test YOLOv8n COCO pretrained weights on same 10 frames
- Score both by counting correct truck/trailer bounding boxes manually
- Pick one detector to carry forward
- Document the decision

#### OUT OF SCOPE — DO NOT ADD
- No fine-tuning on keremberke or fuxi-robot
- No tracking
- No video processing (single frames only)
- No dashboard
- No loading zone
- No augmentation or preprocessing (test on raw frames first)

---

#### Task 2.1: Create `src/detector.py`

**Files:**
- Create: `src/detector.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: Write the failing test first**

```python
# tests/test_detector.py
import numpy as np
from src.detector import Detector

def test_detector_returns_list():
    """Detector must return a list (possibly empty) for any valid frame."""
    dummy_frame = np.zeros((384, 480, 3), dtype=np.uint8)
    det = Detector(model_type="yolo")
    results = det.detect(dummy_frame)
    assert isinstance(results, list)

def test_detector_result_schema():
    """Each detection must be a dict with keys: bbox, confidence, class_id."""
    dummy_frame = np.zeros((384, 480, 3), dtype=np.uint8)
    det = Detector(model_type="yolo")
    results = det.detect(dummy_frame)
    for r in results:
        assert "bbox" in r         # [x1, y1, x2, y2]
        assert "confidence" in r   # float 0-1
        assert "class_id" in r     # int
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
pytest tests/test_detector.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `src.detector` does not exist yet.

- [ ] **Step 3: Write `src/detector.py`**

```python
# src/detector.py
"""
Thin wrapper around either RF-DETR or YOLOv8 for dump truck detection.
Returns a consistent list-of-dicts regardless of backend.
"""
from __future__ import annotations
from typing import Literal
import numpy as np


class Detector:
    """
    Args:
        model_type: "yolo" or "rfdetr"
        weights_path: path to .pth (rfdetr) or .pt (yolo) file.
                      If None, uses COCO pretrained for yolo.
        confidence_threshold: minimum score to keep a detection
        target_classes: list of class IDs to keep (None = keep all)
    """

    DUMP_TRUCK_CLASS_YOLO_COCO = 7  # COCO class 7 = "truck"

    def __init__(
        self,
        model_type: Literal["yolo", "rfdetr"] = "yolo",
        weights_path: str | None = None,
        confidence_threshold: float = 0.3,
        target_classes: list[int] | None = None,
    ):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        self._model = self._load_model(model_type, weights_path)

    def _load_model(self, model_type: str, weights_path: str | None):
        if model_type == "yolo":
            from ultralytics import YOLO
            return YOLO(weights_path or "yolov8n.pt")
        elif model_type == "rfdetr":
            from rfdetr import RFDETRBase
            if weights_path:
                return RFDETRBase.from_checkpoint(weights_path)
            return RFDETRBase()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Args:
            frame: BGR numpy array (H, W, 3)
        Returns:
            List of dicts: [{bbox: [x1,y1,x2,y2], confidence: float, class_id: int}]
        """
        if self.model_type == "yolo":
            return self._detect_yolo(frame)
        elif self.model_type == "rfdetr":
            return self._detect_rfdetr(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[dict]:
        results = self._model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            if self.target_classes and cls not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf, "class_id": cls})
        return detections

    def _detect_rfdetr(self, frame: np.ndarray) -> list[dict]:
        import supervision as sv
        from PIL import Image
        img = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        sv_detections = self._model.predict(img, threshold=self.confidence_threshold)
        detections = []
        for i in range(len(sv_detections)):
            cls = int(sv_detections.class_id[i])
            if self.target_classes and cls not in self.target_classes:
                continue
            x1, y1, x2, y2 = sv_detections.xyxy[i].tolist()
            conf = float(sv_detections.confidence[i])
            detections.append({"bbox": [x1, y1, x2, y2], "confidence": conf, "class_id": cls})
        return detections
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
pytest tests/test_detector.py -v
```

Expected: both tests PASS. (Empty list is valid for a black frame.)

- [ ] **Step 5: Test both models on the 10 sample frames in the notebook**

In `notebooks/02_detection.ipynb`:

```python
import cv2, glob, json
from src.detector import Detector

sample_frames = sorted(glob.glob("data/frames/*.jpg"))[:10]

for model_type, weights in [
    ("yolo", None),                              # COCO pretrained
    ("rfdetr", "models/rfdetr_construction.pth") # SiteSense weights
]:
    det = Detector(model_type=model_type, weights_path=weights, confidence_threshold=0.25)
    total_boxes = 0
    for path in sample_frames:
        frame = cv2.imread(path)
        results = det.detect(frame)
        total_boxes += len(results)
        # Draw boxes on frame for visual inspection
        for r in results:
            x1, y1, x2, y2 = map(int, r["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(path.replace("frames/", f"frames/det_{model_type}_"), frame)
    print(f"{model_type}: {total_boxes} boxes across {len(sample_frames)} frames")
```

- [ ] **Step 6: Manually score results and document detector choice**

Open the annotated frames. For each model count:
- True positives (actual trailers/trucks correctly detected)
- False positives (wrong objects boxed)
- Missed trailers

Document in the notebook:
```python
detector_choice = "rfdetr"  # or "yolo" — set based on visual inspection
detector_weights = "models/rfdetr_construction.pth"  # or None for COCO
print(f"Chosen detector: {detector_choice}")
print(f"Weights: {detector_weights}")
```

Save this as `config/detector_choice.json`.

- [ ] **Step 7: Commit**

```bash
git add src/detector.py tests/test_detector.py notebooks/02_detection.ipynb config/
git commit -m "feat: phase 2 - detection baseline, detector selected"
```

---

## Chunk 2: Phases 3–4 (Tracking + Zone + Counting)

---

### Phase 3: Tracking + Loading Zone

#### IN SCOPE
- BoxMOT tracker wrapper
- Loading zone polygon calibration tool (interactive polygon on first frame)
- Supervision `PolygonZone` for zone entry/exit detection
- Process full videos: detection + tracking + zone overlay
- Verify track IDs persist across frames
- Verify zone correctly identifies which tracks are inside

#### OUT OF SCOPE — DO NOT ADD
- No Re-ID (track IDs from BoxMOT are enough here)
- No fill event counting (only verify zone logic — counting is Phase 4)
- No dashboard
- No dust preprocessing
- No output video saving (visualize in notebook, not saved files)

---

#### Task 3.1: Create `src/zone.py`

**Files:**
- Create: `src/zone.py`
- Create: `config/loading_zone.json`
- Test: `tests/test_zone.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_zone.py
import numpy as np
from src.zone import LoadingZone

def test_point_inside_zone():
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    zone = LoadingZone(polygon=polygon)
    assert zone.contains_point(200, 200) is True

def test_point_outside_zone():
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    zone = LoadingZone(polygon=polygon)
    assert zone.contains_point(50, 50) is False

def test_bbox_centroid_inside():
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    zone = LoadingZone(polygon=polygon)
    bbox = [150, 150, 250, 250]  # [x1, y1, x2, y2] — centroid at (200, 200)
    assert zone.bbox_in_zone(bbox) is True

def test_bbox_centroid_outside():
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    zone = LoadingZone(polygon=polygon)
    bbox = [10, 10, 60, 60]  # centroid at (35, 35) — outside
    assert zone.bbox_in_zone(bbox) is False

def test_save_and_load(tmp_path):
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    zone = LoadingZone(polygon=polygon)
    path = str(tmp_path / "zone.json")
    zone.save(path)
    loaded = LoadingZone.load(path)
    assert loaded.polygon == polygon
```

- [ ] **Step 2: Run tests — verify FAIL**

```bash
pytest tests/test_zone.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write `src/zone.py`**

```python
# src/zone.py
"""
Loading zone polygon management.
Knows nothing about trackers, videos, or models.
"""
from __future__ import annotations
import json
import numpy as np


class LoadingZone:
    """
    A polygon zone defining where trailers park to be loaded.
    Zone is defined once via calibration and persisted to JSON.

    Args:
        polygon: list of [x, y] points defining the zone boundary
    """

    def __init__(self, polygon: list[list[int]]):
        self.polygon = polygon
        self._np_polygon = np.array(polygon, dtype=np.int32)

    def contains_point(self, x: float, y: float) -> bool:
        """Return True if (x, y) is inside the polygon."""
        import cv2
        return cv2.pointPolygonTest(self._np_polygon, (float(x), float(y)), False) >= 0

    def bbox_in_zone(self, bbox: list[float]) -> bool:
        """Return True if the centroid of bbox [x1,y1,x2,y2] is inside the zone."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return self.contains_point(cx, cy)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"polygon": self.polygon}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LoadingZone":
        with open(path) as f:
            data = json.load(f)
        return cls(polygon=data["polygon"])

    def draw_on_frame(self, frame: np.ndarray, color=(0, 255, 255), thickness=2) -> np.ndarray:
        """Draw the zone polygon on a copy of the frame. Returns the annotated copy."""
        import cv2
        out = frame.copy()
        pts = self._np_polygon.reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
        return out
```

- [ ] **Step 4: Run tests — verify PASS**

```bash
pytest tests/test_zone.py -v
```

Expected: 5 tests pass.

---

#### Task 3.2: Calibration tool (notebook cell)

In `notebooks/03_tracking_zone.ipynb`:

- [ ] **Step 1: Interactive polygon drawing on first frame**

```python
import cv2
import json

# Load first frame from any video
first_frame_path = sorted(glob.glob("data/frames/*.jpg"))[0]
frame = cv2.imread(first_frame_path)

# Interactive polygon tool — click to add points, press 'q' to finish
polygon_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        if len(polygon_points) > 1:
            cv2.line(param, tuple(polygon_points[-2]), tuple(polygon_points[-1]), (0, 255, 0), 2)
        cv2.imshow("Draw Loading Zone — click corners, press Q when done", param)

display_frame = frame.copy()
cv2.imshow("Draw Loading Zone — click corners, press Q when done", display_frame)
cv2.setMouseCallback("Draw Loading Zone — click corners, press Q when done", mouse_callback, display_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save zone
import os
os.makedirs("config", exist_ok=True)
with open("config/loading_zone.json", "w") as f:
    json.dump({"polygon": polygon_points}, f, indent=2)

print(f"Zone saved with {len(polygon_points)} points: {polygon_points}")
```

> **Note:** On Colab/Kaggle headless environments, replace the interactive tool with a cell that lets you manually type the polygon coordinates after examining the frame image. Example:
>
> ```python
> # Manually set after inspecting the frame
> polygon_points = [[120, 200], [350, 200], [350, 400], [120, 400]]
> ```

---

#### Task 3.3: Create `src/tracker.py`

**Files:**
- Create: `src/tracker.py`

- [ ] **Step 1: Write `src/tracker.py`**

No unit tests for the tracker itself (BoxMOT is a third-party library; we test the wrapper behavior in integration). A smoke test is sufficient.

```python
# src/tracker.py
"""
Thin wrapper around BoxMOT BotSort.
Returns a consistent list-of-dicts with track IDs per frame.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


class Tracker:
    """
    Args:
        reid_weights_path: path to ReID weights (.pt file)
        device: "cuda" or "cpu"
    """

    def __init__(self, reid_weights_path: str = "models/osnet_x0_25_msmt17.pt", device: str = "cuda"):
        from boxmot import BotSort
        self._tracker = BotSort(
            reid_weights=Path(reid_weights_path),
            device=device,
            half=False,
        )

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """
        Args:
            detections: output from Detector.detect() — list of {bbox, confidence, class_id}
            frame: BGR numpy array
        Returns:
            list of {track_id, bbox, confidence, class_id}
        """
        if not detections:
            self._tracker.update(np.empty((0, 6), dtype=np.float32), frame)
            return []

        # BoxMOT expects (N, 6): [x1, y1, x2, y2, conf, cls]
        det_array = np.array([
            [*d["bbox"], d["confidence"], d["class_id"]]
            for d in detections
        ], dtype=np.float32)

        tracks = self._tracker.update(det_array, frame)
        # BoxMOT output: (N, 8) = [x1, y1, x2, y2, track_id, conf, cls, det_ind]

        results = []
        for t in tracks:
            results.append({
                "track_id": int(t[4]),
                "bbox": [float(t[0]), float(t[1]), float(t[2]), float(t[3])],
                "confidence": float(t[5]),
                "class_id": int(t[6]),
            })
        return results
```

- [ ] **Step 2: Smoke test in notebook**

```python
from src.detector import Detector
from src.tracker import Tracker
from src.zone import LoadingZone

detector = Detector(model_type="rfdetr", weights_path="models/rfdetr_construction.pth")
tracker = Tracker(reid_weights_path="models/osnet_x0_25_msmt17.pt")
zone = LoadingZone.load("config/loading_zone.json")

# Run on first 5 frames of one video
cap = cv2.VideoCapture("data/frames/../")  # path to a sample video

frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
detections = detector.detect(frame)
tracks = tracker.update(detections, frame)

print(f"Detections: {len(detections)}, Tracks: {len(tracks)}")
for t in tracks:
    in_zone = zone.bbox_in_zone(t["bbox"])
    print(f"  Track {t['track_id']}: in_zone={in_zone}")
```

Expected: no errors, track IDs printed.

- [ ] **Step 3: Commit**

```bash
git add src/zone.py src/tracker.py tests/test_zone.py notebooks/03_tracking_zone.ipynb config/loading_zone.json
git commit -m "feat: phase 3 - tracker, loading zone calibration, zone detection"
```

---

### Phase 4: Fill Event Counter

#### IN SCOPE
- State machine: track enters zone → track disappears (Lost state) → count += 1
- Output: `data/results/YYYY-MM-DD.json` with total fills + per-event timestamps
- Test on 3 sample videos
- Print final daily count to console

#### OUT OF SCOPE — DO NOT ADD
- No per-vehicle identity (Re-ID is Phase 5)
- No dashboard
- No thumbnail capture
- No dust preprocessing
- No annotated video output
- No distinction between vehicle types (all departures count equally)

---

#### Task 4.1: Create `src/event_counter.py`

**Files:**
- Create: `src/event_counter.py`
- Test: `tests/test_event_counter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_event_counter.py
from src.event_counter import FillEventCounter

def test_no_event_if_track_never_in_zone():
    counter = FillEventCounter()
    counter.update(track_id=1, in_zone=False, frame_number=10)
    counter.track_lost(track_id=1, frame_number=20)
    assert counter.total_fills() == 0

def test_event_fired_when_track_leaves_after_zone():
    counter = FillEventCounter()
    counter.update(track_id=1, in_zone=True, frame_number=10)
    counter.update(track_id=1, in_zone=True, frame_number=11)
    counter.track_lost(track_id=1, frame_number=25)
    assert counter.total_fills() == 1

def test_same_track_id_reused_does_not_double_count():
    """BoxMOT may reuse a track_id after it's removed. Ensure no double count."""
    counter = FillEventCounter()
    # First vehicle with track_id=5 enters zone, departs
    counter.update(track_id=5, in_zone=True, frame_number=10)
    counter.track_lost(track_id=5, frame_number=20)
    assert counter.total_fills() == 1
    # Track ID 5 reused — new vehicle, never entered zone
    counter.update(track_id=5, in_zone=False, frame_number=100)
    counter.track_lost(track_id=5, frame_number=110)
    assert counter.total_fills() == 1  # still 1

def test_multiple_vehicles_counted_independently():
    counter = FillEventCounter()
    for tid in [1, 2, 3]:
        counter.update(track_id=tid, in_zone=True, frame_number=10)
        counter.track_lost(track_id=tid, frame_number=20)
    assert counter.total_fills() == 3

def test_events_have_timestamps():
    counter = FillEventCounter()
    counter.update(track_id=1, in_zone=True, frame_number=10)
    counter.track_lost(track_id=1, frame_number=50)
    events = counter.get_events()
    assert len(events) == 1
    assert events[0]["frame_number"] == 50
    assert "track_id" in events[0]
```

- [ ] **Step 2: Run tests — verify FAIL**

```bash
pytest tests/test_event_counter.py -v
```

- [ ] **Step 3: Write `src/event_counter.py`**

```python
# src/event_counter.py
"""
State machine for fill event detection.
A fill event fires when a track that entered the loading zone is subsequently lost.
Knows nothing about video, models, or the dashboard.
"""
from __future__ import annotations
import json
from datetime import datetime


class FillEventCounter:
    """
    Tracks which tracks have entered the loading zone and fires a fill event
    when those tracks are reported as lost (departed).

    Usage:
        counter = FillEventCounter()
        # Per frame:
        for track in tracks:
            counter.update(track["track_id"], zone.bbox_in_zone(track["bbox"]), frame_number)
        # When tracker reports a lost track:
        counter.track_lost(track_id, frame_number)
    """

    def __init__(self):
        self._zone_entries: dict[int, int] = {}  # track_id → frame_number when first entered zone
        self._counted_tracks: set[int] = set()   # track_ids already counted (prevents reuse double-count)
        self._events: list[dict] = []

    def update(self, track_id: int, in_zone: bool, frame_number: int) -> None:
        """Call once per track per frame."""
        if in_zone and track_id not in self._zone_entries and track_id not in self._counted_tracks:
            self._zone_entries[track_id] = frame_number

    def track_lost(self, track_id: int, frame_number: int) -> bool:
        """
        Call when BoxMOT reports a track as lost/removed.
        Returns True if a fill event was fired.
        """
        if track_id in self._zone_entries:
            self._events.append({
                "track_id": track_id,
                "zone_entry_frame": self._zone_entries.pop(track_id),
                "departure_frame": frame_number,
                "timestamp": datetime.now().isoformat(),
            })
            self._counted_tracks.add(track_id)
            return True
        return False

    def total_fills(self) -> int:
        return len(self._events)

    def get_events(self) -> list[dict]:
        return list(self._events)

    def save(self, path: str, date: str | None = None) -> None:
        output = {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "total_fills": self.total_fills(),
            "events": self._events,
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
```

- [ ] **Step 4: Run tests — verify PASS**

```bash
pytest tests/test_event_counter.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: End-to-end pipeline test on 3 videos in notebook**

In `notebooks/04_counting.ipynb`:

```python
import cv2, json, os, glob
from datetime import datetime
from src.detector import Detector
from src.tracker import Tracker
from src.zone import LoadingZone
from src.event_counter import FillEventCounter

detector = Detector(model_type="rfdetr", weights_path="models/rfdetr_construction.pth")
tracker = Tracker()
zone = LoadingZone.load("config/loading_zone.json")

os.makedirs("data/results", exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")

# Process 3 videos from the fuxi-robot dataset
video_paths = []  # populate from your extracted set or stream 3 from fuxi-robot

for video_path in video_paths[:3]:
    counter = FillEventCounter()
    cap = cv2.VideoCapture(video_path)
    prev_track_ids = set()
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame)
        current_ids = {t["track_id"] for t in tracks}

        # Update zone state
        for t in tracks:
            counter.update(t["track_id"], zone.bbox_in_zone(t["bbox"]), frame_num)

        # Detect lost tracks (present last frame, absent this frame)
        lost_ids = prev_track_ids - current_ids
        for tid in lost_ids:
            fired = counter.track_lost(tid, frame_num)
            if fired:
                print(f"  FILL EVENT: track {tid} departed at frame {frame_num}")

        prev_track_ids = current_ids
        frame_num += 1

    cap.release()
    print(f"Video: {os.path.basename(video_path)} → {counter.total_fills()} fills")

counter.save(f"data/results/{today}.json")
print(f"\nResults saved to data/results/{today}.json")
with open(f"data/results/{today}.json") as f:
    print(json.dumps(json.load(f), indent=2))
```

- [ ] **Step 6: Commit**

```bash
git add src/event_counter.py tests/test_event_counter.py notebooks/04_counting.ipynb
git commit -m "feat: phase 4 - fill event counter, daily JSON output"
```

> **Phase 4 complete — the system now counts fills.** Every trailer departure from the loading zone is counted and saved. This is the first shippable output.

---

## Chunk 3: Phases 5–6 (Re-ID + Dashboard)

---

### Phase 5: Vehicle Re-ID

#### IN SCOPE
- Load DINOv3 ViT-B/16 backbone from HuggingFace
- Load SiteSense projection head weights (`dinov3_reid_head.pth`)
- Extract 128-d embedding from any vehicle crop
- Gallery manager: per-day dict of `{vehicle_id → embedding}`
- Matching: cosine similarity + threshold → same vehicle or new vehicle
- Update `FillEventCounter` results to include `vehicle_id` (gallery ID)

#### OUT OF SCOPE — DO NOT ADD
- No thumbnail storage or display (dashboard does that)
- No dashboard
- No dust preprocessing
- No cross-day persistence (gallery resets per day, not per video)
- No fine-tuning of the Re-ID head

---

#### Task 5.1: Create `src/reid_gallery.py`

**Files:**
- Create: `src/reid_gallery.py`
- Test: `tests/test_reid_gallery.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reid_gallery.py
import numpy as np
from src.reid_gallery import ReIDGallery

def _make_embedding(seed: int, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)

def test_first_vehicle_gets_id_0():
    gallery = ReIDGallery(threshold=0.7)
    emb = _make_embedding(1)
    vid = gallery.identify(emb)
    assert vid == 0

def test_same_embedding_returns_same_id():
    gallery = ReIDGallery(threshold=0.7)
    emb = _make_embedding(1)
    id1 = gallery.identify(emb)
    id2 = gallery.identify(emb)
    assert id1 == id2

def test_different_embedding_returns_new_id():
    gallery = ReIDGallery(threshold=0.7)
    emb_a = _make_embedding(1)
    emb_b = _make_embedding(999)  # very different
    id_a = gallery.identify(emb_a)
    id_b = gallery.identify(emb_b)
    assert id_a != id_b

def test_count_per_vehicle():
    gallery = ReIDGallery(threshold=0.7)
    emb_a = _make_embedding(1)
    emb_b = _make_embedding(999)
    for _ in range(3):
        gallery.identify(emb_a)
    for _ in range(2):
        gallery.identify(emb_b)
    counts = gallery.counts()
    assert counts[0] == 3
    assert counts[1] == 2
```

- [ ] **Step 2: Run tests — verify FAIL**

```bash
pytest tests/test_reid_gallery.py -v
```

- [ ] **Step 3: Write `src/reid_gallery.py`**

```python
# src/reid_gallery.py
"""
Per-day Re-ID gallery.
Stores one representative embedding per unique vehicle.
Knows nothing about video, models, or the dashboard.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ReIDGallery:
    """
    Args:
        threshold: cosine similarity threshold above which two embeddings
                   are considered the same vehicle (0.0–1.0).
                   Start at 0.75; tune based on real footage.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self._embeddings: list[np.ndarray] = []   # one per unique vehicle
        self._counts: list[int] = []               # fill count per vehicle

    def identify(self, embedding: np.ndarray) -> int:
        """
        Compare embedding against gallery.
        Returns existing vehicle_id if similarity ≥ threshold,
        else registers a new vehicle and returns its new ID.
        """
        if not self._embeddings:
            return self._register_new(embedding)

        gallery_matrix = np.vstack(self._embeddings)
        sims = cosine_similarity(embedding.reshape(1, -1), gallery_matrix)[0]
        best_idx = int(np.argmax(sims))

        if sims[best_idx] >= self.threshold:
            self._counts[best_idx] += 1
            # Update gallery embedding with running mean for better representation
            self._embeddings[best_idx] = self._embeddings[best_idx] * 0.9 + embedding * 0.1
            self._embeddings[best_idx] /= np.linalg.norm(self._embeddings[best_idx])
            return best_idx

        return self._register_new(embedding)

    def _register_new(self, embedding: np.ndarray) -> int:
        self._embeddings.append(embedding.copy())
        self._counts.append(1)
        return len(self._embeddings) - 1

    def counts(self) -> dict[int, int]:
        """Returns {vehicle_id: fill_count}"""
        return {i: c for i, c in enumerate(self._counts)}

    def num_unique_vehicles(self) -> int:
        return len(self._embeddings)

    def reset(self) -> None:
        """Call at the start of each new day."""
        self._embeddings.clear()
        self._counts.clear()
```

- [ ] **Step 4: Run tests — verify PASS**

```bash
pytest tests/test_reid_gallery.py -v
```

Expected: 4 tests pass.

---

#### Task 5.2: DINOv3 embedding extractor (notebook)

In `notebooks/05_reid.ipynb`:

- [ ] **Step 1: Load DINOv3 backbone + SiteSense projection head**

```python
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# Load DINOv3 backbone
processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
backbone = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
backbone.eval()

# Reconstruct SiteSense projection head: 1536 → 256 → 128
class ReIDHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1536, 256),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        out = self.proj(x)
        return out / out.norm(dim=-1, keepdim=True)  # L2 normalize

head = ReIDHead()
head.load_state_dict(torch.load("models/dinov3_reid_head.pth", map_location="cpu"))
head.eval()

print("DINOv3 + ReID head loaded")
```

- [ ] **Step 2: Embedding extraction function**

```python
def extract_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Args:
        crop_bgr: BGR numpy array of a vehicle crop
    Returns:
        128-d L2-normalized embedding as numpy float32 array
    """
    img = Image.fromarray(crop_bgr[:, :, ::-1])  # BGR → RGB
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = backbone(**inputs).last_hidden_state[:, 0, :]  # CLS token, shape (1, 1536)
        embedding = head(features).cpu().numpy()[0]  # (128,)
    return embedding

# Test on a sample crop
sample_frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
h, w = sample_frame.shape[:2]
dummy_crop = sample_frame[h//4:3*h//4, w//4:3*w//4]  # center crop
emb = extract_embedding(dummy_crop)
assert emb.shape == (128,), f"Expected (128,), got {emb.shape}"
assert abs(np.linalg.norm(emb) - 1.0) < 1e-5, "Embedding not normalized"
print(f"Embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f} — OK")
```

- [ ] **Step 3: Full pipeline with Re-ID**

Extend the Phase 4 pipeline: when a fill event fires, extract the vehicle crop, get embedding, identify in gallery:

```python
from src.reid_gallery import ReIDGallery

gallery = ReIDGallery(threshold=0.75)
gallery.reset()  # start fresh each day

# ... (same loop as Phase 4) ...
# When fill event fires:
# 1. Get the last known bounding box for the departed track
# 2. Extract crop from the last frame it was seen
# 3. Get embedding
# 4. Identify in gallery

# At end of day:
print("=== Daily Summary ===")
for vid, count in gallery.counts().items():
    print(f"Vehicle {vid}: {count} fills")
print(f"Unique vehicles: {gallery.num_unique_vehicles()}")
```

- [ ] **Step 4: Commit**

```bash
git add src/reid_gallery.py tests/test_reid_gallery.py notebooks/05_reid.ipynb
git commit -m "feat: phase 5 - dinov3 reid, per-vehicle fill counts"
```

---

### Phase 6: Dashboard

#### IN SCOPE
- Streamlit app reading `data/results/YYYY-MM-DD.json`
- Daily total fills (number)
- Per-vehicle table: vehicle ID, fill count, first/last seen timestamps
- Loading event timeline (Plotly chart: events over time)
- Select video date to display results for that day
- Annotated video export (bounding boxes + zone polygon + track IDs drawn by OpenCV)

#### OUT OF SCOPE — DO NOT ADD
- No live stream
- No cloud deployment
- No multi-excavator support
- No mobile/responsive design
- No alerts or notifications
- No editing or deleting events
- No user authentication

---

#### Task 6.1: Create `src/preprocessor.py` and `src/video_annotator.py`

**Files:**
- Create: `src/preprocessor.py`
- Create: `src/video_annotator.py`

- [ ] **Step 1: Write `src/preprocessor.py`**

```python
# src/preprocessor.py
"""CLAHE dust preprocessing. Applied to each frame before detection."""
import cv2
import numpy as np

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB color space. Returns BGR."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

- [ ] **Step 2: Write `src/video_annotator.py`**

```python
# src/video_annotator.py
"""Draw bounding boxes, track IDs, zone polygon, and fill count on frames."""
import cv2
import numpy as np
from src.zone import LoadingZone


def annotate_frame(
    frame: np.ndarray,
    tracks: list[dict],
    zone: LoadingZone,
    fill_count: int,
    in_zone_ids: set[int],
) -> np.ndarray:
    """Returns annotated copy of frame."""
    out = zone.draw_on_frame(frame)

    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        tid = t["track_id"]
        color = (0, 255, 0) if tid in in_zone_ids else (255, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"ID:{tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(out, f"Fills today: {fill_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out
```

---

#### Task 6.2: Create `dashboard/app.py`

**Files:**
- Create: `dashboard/app.py`

- [ ] **Step 1: Write `dashboard/app.py`**

```python
# dashboard/app.py
"""
Streamlit dashboard for daily trailer fill counts.
Run with: streamlit run dashboard/app.py
"""
import json
import glob
import os
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Trailer Counter", layout="wide")
st.title("Excavator Trailer Fill Counter")

# Sidebar: select date
result_files = sorted(glob.glob("data/results/*.json"), reverse=True)
if not result_files:
    st.warning("No results found. Run the processing pipeline first.")
    st.stop()

dates = [os.path.basename(f).replace(".json", "") for f in result_files]
selected_date = st.sidebar.selectbox("Select Date", dates)
result_path = f"data/results/{selected_date}.json"

with open(result_path) as f:
    data = json.load(f)

# KPI
col1, col2 = st.columns(2)
col1.metric("Total Fills", data["total_fills"])
col2.metric("Unique Vehicles", len(set(e.get("vehicle_id", 0) for e in data["events"])))

st.divider()

# Per-vehicle table
if data["events"]:
    df = pd.DataFrame(data["events"])
    vehicle_summary = (
        df.groupby("vehicle_id")
          .agg(fill_count=("vehicle_id", "count"),
               first_seen=("timestamp", "min"),
               last_seen=("timestamp", "max"))
          .reset_index()
          .sort_values("fill_count", ascending=False)
    )
    st.subheader("Per-Vehicle Summary")
    st.dataframe(vehicle_summary, use_container_width=True)

    # Timeline
    st.subheader("Fill Events Timeline")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig = px.scatter(
        df, x="timestamp", y="vehicle_id",
        color="vehicle_id", size_max=10,
        title="Loading Events Over Time",
        labels={"timestamp": "Time", "vehicle_id": "Vehicle ID"}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No fill events recorded for this date.")
```

- [ ] **Step 2: Test dashboard runs**

```bash
# In Colab/Kaggle with localtunnel:
!streamlit run dashboard/app.py &
!sleep 3
!npx localtunnel --port 8501
```

Or in a local environment:
```bash
streamlit run dashboard/app.py
```

Expected: dashboard opens, shows KPI metrics and table.

- [ ] **Step 3: Commit**

```bash
git add src/preprocessor.py src/video_annotator.py dashboard/app.py notebooks/06_dashboard.ipynb
git commit -m "feat: phase 6 - streamlit dashboard, CLAHE preprocessing, video annotator"
```

---

## Phase Summary — Scope Boundaries

| Phase | IN SCOPE | OUT OF SCOPE |
|---|---|---|
| **1 Foundation** | Install libs, download datasets + weights, extract 50 frames, visual inspect | Training, pipeline, dashboard, zone |
| **2 Detection** | RF-DETR vs YOLOv8 comparison on 10 frames, pick one | Fine-tuning, tracking, video, dashboard |
| **3 Tracking + Zone** | BoxMOT integration, polygon calibration, zone entry/exit on video | Re-ID, counting events, dashboard, dust |
| **4 Counting** | State machine, JSON output, test on 3 videos | Per-vehicle Re-ID, dashboard, thumbnails |
| **5 Re-ID** | DINOv3 + SiteSense head, gallery manager, per-vehicle counts | Dashboard, thumbnails, cross-day persistence |
| **6 Dashboard** | Streamlit app, vehicle table, timeline, annotated video | Live stream, cloud, multi-excavator, auth, alerts |

---

## Tests Reference

```bash
# Run all tests
pytest tests/ -v

# Run by phase
pytest tests/test_zone.py -v          # Phase 3
pytest tests/test_event_counter.py -v # Phase 4
pytest tests/test_reid_gallery.py -v  # Phase 5
```

---

*Plan complete. Ready to execute?*
