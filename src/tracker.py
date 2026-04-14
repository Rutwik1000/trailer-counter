"""Multi-object tracker wrapper around BoxMOT BotSort.

Single responsibility: given detector output + a BGR frame, return tracked
detections with persistent IDs. Does NOT know about zones, counts, Re-ID,
or video I/O.

Output contract (fixed — do not change without updating docs/ARCHITECTURE.md):
    {"track_id": int, "bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int}

ADR-003: BoxMOT BotSort selected — appearance-based, robust to camera rotation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_REID_WEIGHTS = "models/osnet_x0_25_msmt17.pt"
_DEFAULT_DEVICE = "cuda"


class Tracker:
    """Thin wrapper around BoxMOT BotSort that enforces a fixed output contract.

    Model loading is lazy — BotSort is not instantiated until the first update()
    call with non-empty detections. This allows Tracker() to be constructed in
    tests without boxmot being installed or a GPU being available.

    Args:
        reid_weights_path: Path to OSNet ReID weights (.pt). BotSort uses these
            for appearance-based track matching across frames.
        device: "cuda" or "cpu". "cuda" is strongly recommended for real-time use.
    """

    def __init__(
        self,
        reid_weights_path: str = _DEFAULT_REID_WEIGHTS,
        device: str = _DEFAULT_DEVICE,
    ) -> None:
        self.reid_weights_path = reid_weights_path
        self.device = device
        self._tracker: Any = None  # populated on first update() call with detections

    def _load_tracker(self) -> None:
        """Instantiate BotSort. Called once on first update() with detections."""
        if self._tracker is not None:
            return
        from boxmot import BotSort
        self._tracker = BotSort(
            reid_weights=Path(self.reid_weights_path),
            device=self.device,
            half=False,
        )

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """Update the tracker with detections from the current frame.

        Args:
            detections: Output from Detector.detect() —
                list of {"bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int}.
                Not mutated.
            frame: BGR numpy array, shape (H, W, 3), dtype uint8. Not mutated.

        Returns:
            List of dicts, each with exactly four keys:
                "track_id": int — persistent ID across frames (positive integer)
                "bbox": [x1, y1, x2, y2] — float pixel coordinates
                "confidence": float in [0.0, 1.0]
                "class_id": int
            Returns an empty list when no tracks are active.

        Raises:
            ValueError: If frame is not dtype uint8 or not shape (H, W, 3).
        """
        _validate_frame(frame)

        if not detections:
            # Advance tracker state without loading if there are no detections.
            # If tracker is already initialized, pass empty array to age tracks.
            if self._tracker is not None:
                self._tracker.update(np.empty((0, 6), dtype=np.float32), frame)
            return []

        self._load_tracker()

        # BoxMOT expects (N, 6): [x1, y1, x2, y2, confidence, class_id]
        det_array = np.array(
            [[*d["bbox"], d["confidence"], d["class_id"]] for d in detections],
            dtype=np.float32,
        )
        tracks = self._tracker.update(det_array, frame)

        # BoxMOT output: (N, 8) — [x1, y1, x2, y2, track_id, conf, cls, det_ind]
        results: list[dict] = []
        for t in tracks:
            results.append({
                "track_id": int(t[4]),
                "bbox": [float(t[0]), float(t[1]), float(t[2]), float(t[3])],
                "confidence": float(t[5]),
                "class_id": int(t[6]),
            })
        return results


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
