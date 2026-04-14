"""Unit tests for src/tracker.py — BoxMOT BotSort wrapper.

Tests run on Claude Code (no GPU, boxmot may not be installed).
Model loading is lazy — constructor tests never trigger BotSort instantiation.
Frame-dependent tests skip when data/frames/ is empty.
"""
import glob
import os

import cv2
import numpy as np
import pytest

from src.tracker import Tracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bgr(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _frames_available() -> bool:
    return len(glob.glob(os.path.join("data", "frames", "*.jpg"))) > 0


_skip_no_frames = pytest.mark.skipif(
    not _frames_available(),
    reason="data/frames/ is empty — run on Kaggle",
)

# ---------------------------------------------------------------------------
# Constructor validation — lazy loading, no boxmot import
# ---------------------------------------------------------------------------

def test_tracker_stores_reid_weights_path():
    t = Tracker(reid_weights_path="models/osnet_x0_25_msmt17.pt")
    assert t.reid_weights_path == "models/osnet_x0_25_msmt17.pt"


def test_tracker_default_reid_weights_path():
    t = Tracker()
    assert t.reid_weights_path == "models/osnet_x0_25_msmt17.pt"


def test_tracker_stores_device():
    t = Tracker(device="cpu")
    assert t.device == "cpu"


def test_tracker_default_device_is_cuda():
    t = Tracker()
    assert t.device == "cuda"


def test_tracker_model_not_loaded_at_construction():
    t = Tracker()
    assert t._tracker is None


# ---------------------------------------------------------------------------
# Input validation — _validate_frame runs BEFORE _load_tracker, no GPU needed
# ---------------------------------------------------------------------------

def test_update_rejects_non_uint8_frame():
    t = Tracker()
    frame_float = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        t.update([], frame_float)


def test_update_rejects_grayscale_frame():
    t = Tracker()
    frame_gray = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        t.update([], frame_gray)


def test_update_rejects_4channel_frame():
    t = Tracker()
    frame_rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        t.update([], frame_rgba)


def test_update_empty_detections_before_init_returns_empty_list():
    """Empty detections before tracker is initialized returns [] without loading boxmot."""
    t = Tracker()
    frame = _make_bgr()
    result = t.update([], frame)
    assert result == []
    assert t._tracker is None  # still not loaded — no point loading for empty


# ---------------------------------------------------------------------------
# Output contract — require boxmot + GPU + frames (Kaggle only, skips locally)
# ---------------------------------------------------------------------------

@_skip_no_frames
def test_update_output_schema():
    frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
    t = Tracker(reid_weights_path="models/osnet_x0_25_msmt17.pt",
                device="cuda" if True else "cpu")
    detections = [{"bbox": [100.0, 100.0, 300.0, 300.0], "confidence": 0.6, "class_id": 7}]
    results = t.update(detections, frame)
    assert isinstance(results, list)
    for r in results:
        assert set(r.keys()) == {"track_id", "bbox", "confidence", "class_id"}
        assert isinstance(r["track_id"], int)
        assert isinstance(r["bbox"], list) and len(r["bbox"]) == 4
        assert isinstance(r["confidence"], float)
        assert isinstance(r["class_id"], int)


@_skip_no_frames
def test_update_does_not_mutate_frame():
    frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
    t = Tracker(reid_weights_path="models/osnet_x0_25_msmt17.pt")
    original = frame.copy()
    t.update([], frame)
    np.testing.assert_array_equal(frame, original)


@_skip_no_frames
def test_update_empty_detections_after_init_returns_empty_list():
    frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
    t = Tracker(reid_weights_path="models/osnet_x0_25_msmt17.pt")
    # First call with real detection to initialize
    det = [{"bbox": [100.0, 100.0, 300.0, 300.0], "confidence": 0.6, "class_id": 7}]
    t.update(det, frame)
    # Second call empty — should return []
    result = t.update([], frame)
    assert result == []


@_skip_no_frames
def test_update_track_ids_are_positive_ints():
    frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
    t = Tracker(reid_weights_path="models/osnet_x0_25_msmt17.pt")
    det = [{"bbox": [100.0, 100.0, 300.0, 300.0], "confidence": 0.6, "class_id": 7}]
    results = t.update(det, frame)
    for r in results:
        assert r["track_id"] > 0
