"""Unit tests for src/zone.py — loading zone polygon.

All tests run on Claude Code (no GPU). Pure numpy + cv2 + json.
"""
import json
import os

import numpy as np
import pytest

from src.zone import LoadingZone

# ---------------------------------------------------------------------------
# Shared fixture polygon: 200x200 square, top-left at (100, 100)
# ---------------------------------------------------------------------------
_SQUARE = [[100, 100], [300, 100], [300, 300], [100, 300]]


# ---------------------------------------------------------------------------
# Point-in-polygon tests
# ---------------------------------------------------------------------------

def test_point_inside_zone():
    zone = LoadingZone(polygon=_SQUARE)
    assert zone.contains_point(200, 200) is True


def test_point_outside_zone():
    zone = LoadingZone(polygon=_SQUARE)
    assert zone.contains_point(50, 50) is False


def test_point_on_boundary_is_inside():
    zone = LoadingZone(polygon=_SQUARE)
    # Top edge midpoint
    assert zone.contains_point(200, 100) is True


def test_point_far_outside():
    zone = LoadingZone(polygon=_SQUARE)
    assert zone.contains_point(1000, 1000) is False


# ---------------------------------------------------------------------------
# Bbox centroid tests
# ---------------------------------------------------------------------------

def test_bbox_centroid_inside():
    zone = LoadingZone(polygon=_SQUARE)
    bbox = [150.0, 150.0, 250.0, 250.0]  # centroid at (200, 200) — inside
    assert zone.bbox_in_zone(bbox) is True


def test_bbox_centroid_outside():
    zone = LoadingZone(polygon=_SQUARE)
    bbox = [10.0, 10.0, 60.0, 60.0]  # centroid at (35, 35) — outside
    assert zone.bbox_in_zone(bbox) is False


def test_bbox_straddles_boundary_centroid_inside():
    # Bbox straddles the polygon edge but centroid is inside → True
    zone = LoadingZone(polygon=_SQUARE)
    bbox = [50.0, 150.0, 200.0, 250.0]  # centroid at (125, 200) — inside
    assert zone.bbox_in_zone(bbox) is True


def test_bbox_straddles_boundary_centroid_outside():
    # Bbox straddles the polygon edge but centroid is outside → False
    zone = LoadingZone(polygon=_SQUARE)
    bbox = [10.0, 150.0, 140.0, 250.0]  # centroid at (75, 200) — outside
    assert zone.bbox_in_zone(bbox) is False


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_polygon_requires_at_least_3_points():
    with pytest.raises(ValueError, match="3 points"):
        LoadingZone(polygon=[[0, 0], [100, 0]])


def test_polygon_with_exactly_3_points_accepted():
    zone = LoadingZone(polygon=[[0, 0], [100, 0], [50, 100]])
    assert len(zone.polygon) == 3


def test_polygon_is_defensive_copy():
    pts = [[100, 100], [300, 100], [300, 300], [100, 300]]
    zone = LoadingZone(polygon=pts)
    pts[0][0] = 999  # mutate original
    assert zone.polygon[0][0] == 100  # zone must not be affected


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    zone = LoadingZone(polygon=_SQUARE)
    path = str(tmp_path / "zone.json")
    zone.save(path)
    loaded = LoadingZone.load(path)
    assert loaded.polygon == _SQUARE


def test_load_uncalibrated_placeholder_raises(tmp_path):
    path = str(tmp_path / "uncalibrated.json")
    with open(path, "w") as f:
        json.dump({"polygon": None}, f)
    with pytest.raises(ValueError, match="not been calibrated"):
        LoadingZone.load(path)


def test_save_produces_valid_json(tmp_path):
    zone = LoadingZone(polygon=_SQUARE)
    path = str(tmp_path / "zone.json")
    zone.save(path)
    with open(path) as f:
        data = json.load(f)
    assert "polygon" in data
    assert data["polygon"] == _SQUARE


# ---------------------------------------------------------------------------
# draw_on_frame tests
# ---------------------------------------------------------------------------

def _make_bgr(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_draw_does_not_mutate_input():
    zone = LoadingZone(polygon=[[5, 5], [50, 5], [50, 50], [5, 50]])
    frame = _make_bgr(64, 64)
    original = frame.copy()
    zone.draw_on_frame(frame)
    np.testing.assert_array_equal(frame, original)


def test_draw_returns_new_array():
    zone = LoadingZone(polygon=[[5, 5], [50, 5], [50, 50], [5, 50]])
    frame = _make_bgr(64, 64)
    result = zone.draw_on_frame(frame)
    assert result is not frame


def test_draw_preserves_shape_and_dtype():
    zone = LoadingZone(polygon=[[5, 5], [50, 5], [50, 50], [5, 50]])
    frame = _make_bgr(720, 1280)
    result = zone.draw_on_frame(frame)
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
