"""Detection wrapper for RF-DETR and YOLOv8 backends.

Single responsibility: given a BGR frame, return a list of detection dicts.
Does NOT know about: tracking, zones, counts, Re-ID.

Output contract (fixed — do not change without updating docs/ARCHITECTURE.md):
    {"bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int}
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

_VALID_MODEL_TYPES = frozenset({"yolo", "rfdetr"})
_DEFAULT_CONFIDENCE_THRESHOLD = 0.25


class Detector:
    """Thin wrapper around YOLOv8 or RF-DETR that enforces a fixed output contract.

    Model loading is lazy — the underlying model is not loaded until the first
    detect() call. This allows constructing Detector objects in tests without
    triggering weight downloads or GPU initialization.

    Args:
        model_type: Either "yolo" or "rfdetr". Raises ValueError for any other value.
        weights_path: Path to model weights file. For "yolo", defaults to "yolov8n.pt"
            (auto-downloaded by ultralytics on first use). For "rfdetr", defaults to
            "models/rfdetr_construction.pth".
        confidence_threshold: Minimum confidence score to include a detection.
            Detections below this threshold are discarded. Defaults to 0.25.

    Raises:
        ValueError: If model_type is not "yolo" or "rfdetr".
    """

    def __init__(
        self,
        model_type: str,
        weights_path: Optional[str] = None,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        if model_type not in _VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {sorted(_VALID_MODEL_TYPES)}, got {model_type!r}"
            )
        self.model_type = model_type
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self._model: Any = None  # populated on first detect() call

    def _load_model(self) -> None:
        """Load the underlying model. Called once on the first detect() call."""
        if self._model is not None:
            return

        if self.model_type == "yolo":
            from ultralytics import YOLO
            path = self.weights_path if self.weights_path is not None else "yolov8n.pt"
            self._model = YOLO(path)

        elif self.model_type == "rfdetr":
            from rfdetr import RFDETRBase
            path = self.weights_path if self.weights_path is not None else "models/rfdetr_construction.pth"
            self._model = RFDETRBase(pretrain_weights=path)

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection on a single BGR frame.

        Args:
            frame: BGR numpy array, shape (H, W, 3), dtype uint8. Not mutated.

        Returns:
            List of dicts, each with exactly three keys:
                "bbox": [x1, y1, x2, y2]  — float pixel coordinates
                "confidence": float in [0.0, 1.0]
                "class_id": int
            Returns an empty list when no detections exceed confidence_threshold.

        Raises:
            ValueError: If frame is not dtype uint8 or not shape (H, W, 3).
        """
        _validate_frame(frame)
        self._load_model()

        if self.model_type == "yolo":
            return self._detect_yolo(frame)
        else:
            return self._detect_rfdetr(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[dict]:
        results = self._model(frame, verbose=False)
        detections: list[dict] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy()
            for bbox, conf, cls in zip(xyxy, confs, clses):
                if float(conf) < self.confidence_threshold:
                    continue
                detections.append({
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "confidence": float(conf),
                    "class_id": int(cls),
                })
        return detections

    def _detect_rfdetr(self, frame: np.ndarray) -> list[dict]:
        """RF-DETR expects a PIL Image (RGB).

        Converts BGR numpy array → RGB PIL Image before calling predict().
        Returns supervision Detections: .xyxy shape (N,4), .confidence (N,), .class_id (N,).
        """
        from PIL import Image
        img = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        sv_detections = self._model.predict(img)
        detections: list[dict] = []
        if sv_detections is None or len(sv_detections) == 0:
            return detections
        for i in range(len(sv_detections)):
            conf = float(sv_detections.confidence[i])
            if conf < self.confidence_threshold:
                continue
            bbox = sv_detections.xyxy[i]
            detections.append({
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "confidence": conf,
                "class_id": int(sv_detections.class_id[i]),
            })
        return detections


def _validate_frame(frame: np.ndarray) -> None:
    """Validate that frame is a 3-channel uint8 BGR array.

    Args:
        frame: Input to validate.

    Raises:
        ValueError: If dtype is not uint8.
        ValueError: If frame is not shape (H, W, 3).
    """
    if frame.dtype != np.uint8:
        raise ValueError(
            f"frame must be dtype uint8, got {frame.dtype}"
        )
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"frame must be a 3-channel BGR array with shape (H, W, 3), got shape {frame.shape}"
        )
