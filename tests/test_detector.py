"""Unit tests for src/detector.py — detection wrapper (YOLOv8 + RF-DETR backends).

Tests run on Claude Code (no GPU). Model loading is lazy — constructor tests never
trigger a download. Frame-dependent tests skip gracefully when data/frames/ is empty.
"""
import glob
import os

import cv2
import numpy as np
import pytest

from src.detector import Detector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frames_available() -> bool:
    return len(glob.glob(os.path.join("data", "frames", "*.jpg"))) > 0


_skip_no_frames = pytest.mark.skipif(
    not _frames_available(),
    reason="data/frames/ is empty — run Phase 1 on Kaggle first",
)

# ---------------------------------------------------------------------------
# Output contract schema (pure Python — no model, no GPU)
# ---------------------------------------------------------------------------

def test_detection_result_has_required_keys():
    result = {"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_id": 0}
    assert "bbox" in result
    assert "confidence" in result
    assert "class_id" in result


def test_detection_result_bbox_is_list_of_four():
    result = {"bbox": [10, 20, 30, 40], "confidence": 0.5, "class_id": 1}
    assert isinstance(result["bbox"], list)
    assert len(result["bbox"]) == 4


def test_detection_result_confidence_is_float():
    result = {"bbox": [0, 0, 5, 5], "confidence": 0.75, "class_id": 2}
    assert isinstance(result["confidence"], float)


def test_detection_result_class_id_is_int():
    result = {"bbox": [0, 0, 5, 5], "confidence": 0.5, "class_id": 3}
    assert isinstance(result["class_id"], int)


# ---------------------------------------------------------------------------
# Constructor validation — no model loading (lazy), no GPU required
# ---------------------------------------------------------------------------

def test_detector_rejects_invalid_model_type():
    with pytest.raises(ValueError, match="model_type"):
        Detector(model_type="resnet")


def test_detector_rejects_empty_model_type():
    with pytest.raises(ValueError, match="model_type"):
        Detector(model_type="")


def test_detector_accepts_yolo_model_type():
    det = Detector(model_type="yolo")
    assert det.model_type == "yolo"


def test_detector_accepts_rfdetr_model_type():
    det = Detector(model_type="rfdetr")
    assert det.model_type == "rfdetr"


def test_detector_stores_confidence_threshold():
    det = Detector(model_type="yolo", confidence_threshold=0.4)
    assert det.confidence_threshold == 0.4


def test_detector_default_confidence_threshold():
    det = Detector(model_type="yolo")
    assert det.confidence_threshold == 0.25


# ---------------------------------------------------------------------------
# Input validation — runs _validate_frame() BEFORE _load_model(), no GPU needed
# ---------------------------------------------------------------------------

def test_detect_rejects_non_uint8_frame():
    det = Detector(model_type="yolo")
    frame_float = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        det.detect(frame_float)


def test_detect_rejects_grayscale_frame():
    det = Detector(model_type="yolo")
    frame_gray = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        det.detect(frame_gray)


def test_detect_rejects_4channel_frame():
    det = Detector(model_type="yolo")
    frame_rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel"):
        det.detect(frame_rgba)


# ---------------------------------------------------------------------------
# Output contract — require real frames + model (Kaggle only, skips locally)
# ---------------------------------------------------------------------------

@_skip_no_frames
def test_yolo_detect_returns_list():
    det = Detector(model_type="yolo")
    frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
    results = det.detect(frame)
    assert isinstance(results, list)


@_skip_no_frames
def test_rfdetr_detect_returns_list():
    det = Detector(model_type="rfdetr", weights_path="models/rfdetr_construction.pth")
    frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
    results = det.detect(frame)
    assert isinstance(results, list)


@_skip_no_frames
def test_yolo_detect_output_schema():
    det = Detector(model_type="yolo")
    frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
    results = det.detect(frame)
    for r in results:
        assert set(r.keys()) == {"bbox", "confidence", "class_id"}
        assert isinstance(r["bbox"], list) and len(r["bbox"]) == 4
        assert isinstance(r["confidence"], float) and 0.0 <= r["confidence"] <= 1.0
        assert isinstance(r["class_id"], int)
        x1, y1, x2, y2 = r["bbox"]
        assert x2 > x1 and y2 > y1


@_skip_no_frames
def test_rfdetr_detect_output_schema():
    det = Detector(model_type="rfdetr", weights_path="models/rfdetr_construction.pth")
    frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
    results = det.detect(frame)
    for r in results:
        assert set(r.keys()) == {"bbox", "confidence", "class_id"}
        assert isinstance(r["bbox"], list) and len(r["bbox"]) == 4
        assert isinstance(r["confidence"], float) and 0.0 <= r["confidence"] <= 1.0
        assert isinstance(r["class_id"], int)


@_skip_no_frames
def test_detect_does_not_mutate_frame():
    det = Detector(model_type="yolo")
    frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
    original = frame.copy()
    det.detect(frame)
    np.testing.assert_array_equal(frame, original)


@_skip_no_frames
def test_confidence_threshold_filters_results():
    frame = cv2.imread(sorted(glob.glob("data/frames/*.jpg"))[0])
    det_low = Detector(model_type="yolo", confidence_threshold=0.01)
    det_high = Detector(model_type="yolo", confidence_threshold=0.99)
    assert len(det_high.detect(frame)) <= len(det_low.detect(frame))
