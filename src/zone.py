"""Loading zone polygon — operator-calibrated region where trailers are loaded.

Single responsibility: point-in-polygon containment testing and zone persistence.
Does NOT know about: trackers, detectors, video I/O, counts, or Re-ID.

ADR-002: Static polygon (one-time calibration) — works because camera rotates with
the excavator cab, keeping the loading position at a consistent image region.
ADR-007: Centroid-based containment using cv2.pointPolygonTest.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

_ZONE_COLOR_BGR: tuple[int, int, int] = (0, 255, 255)  # yellow
_ZONE_THICKNESS: int = 2


class LoadingZone:
    """Polygon zone defining where trailers park to be loaded.

    Calibrated once per camera installation and persisted to
    config/loading_zone.json. The static polygon is valid for all subsequent
    videos from the same camera (see ADR-002).

    Args:
        polygon: List of [x, y] integer points defining the zone boundary in
            image pixel coordinates. Must have at least 3 points.

    Raises:
        ValueError: If polygon has fewer than 3 points.
    """

    def __init__(self, polygon: list[list[int]]) -> None:
        if len(polygon) < 3:
            raise ValueError(
                f"polygon must have at least 3 points, got {len(polygon)}"
            )
        self.polygon: list[list[int]] = [list(p) for p in polygon]  # defensive copy
        self._np_polygon = np.array(self.polygon, dtype=np.int32)

    def contains_point(self, x: float, y: float) -> bool:
        """Return True if pixel coordinate (x, y) is inside or on the polygon.

        Uses cv2.pointPolygonTest (measureDist=False). Returns >= 0 for
        inside or on-boundary, < 0 for outside.

        Args:
            x: Horizontal pixel coordinate.
            y: Vertical pixel coordinate.

        Returns:
            True if (x, y) is inside or on the polygon boundary.
        """
        return cv2.pointPolygonTest(
            self._np_polygon, (float(x), float(y)), False
        ) >= 0.0

    def bbox_in_zone(self, bbox: list[float]) -> bool:
        """Return True if the centroid of bbox [x1, y1, x2, y2] is inside the zone.

        Centroid is ((x1+x2)/2, (y1+y2)/2). Implements ADR-007.

        Args:
            bbox: [x1, y1, x2, y2] bounding box in pixel coordinates.

        Returns:
            True if the bbox centroid is inside or on the zone polygon boundary.
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return self.contains_point(cx, cy)

    def draw_on_frame(
        self,
        frame: np.ndarray,
        color: tuple[int, int, int] = _ZONE_COLOR_BGR,
        thickness: int = _ZONE_THICKNESS,
    ) -> np.ndarray:
        """Draw the zone polygon outline on a copy of the frame.

        Args:
            frame: BGR numpy array, shape (H, W, 3), dtype uint8. Not mutated.
            color: BGR tuple for the polygon outline. Defaults to yellow.
            thickness: Line thickness in pixels.

        Returns:
            New BGR numpy array with the polygon drawn on it.
        """
        out = frame.copy()
        pts = self._np_polygon.reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
        return out

    def save(self, path: str) -> None:
        """Persist the polygon to a JSON file.

        Args:
            path: Destination file path (e.g. "config/loading_zone.json").
        """
        with open(path, "w") as f:
            json.dump({"polygon": self.polygon}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LoadingZone":
        """Load a LoadingZone from a JSON file produced by save().

        Args:
            path: Path to a JSON file containing {"polygon": [[x, y], ...]}.

        Returns:
            A new LoadingZone instance.

        Raises:
            ValueError: If the file contains {"polygon": null} — the placeholder
                written by Claude Code before Kaggle calibration.
            FileNotFoundError: If path does not exist.
        """
        with open(path) as f:
            data = json.load(f)
        if data.get("polygon") is None:
            raise ValueError(
                f"Loading zone at '{path}' has not been calibrated yet "
                "(polygon is null). Run the Phase 3 calibration cell on Kaggle first."
            )
        return cls(polygon=data["polygon"])
