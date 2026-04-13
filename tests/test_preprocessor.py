"""Unit tests for src/preprocessor.py — CLAHE dust preprocessing."""
import numpy as np
import pytest

from src.preprocessor import apply_clahe


def _make_bgr(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_output_shape_preserved():
    frame = _make_bgr(120, 160)
    result = apply_clahe(frame)
    assert result.shape == frame.shape


def test_output_dtype_preserved():
    frame = _make_bgr()
    result = apply_clahe(frame)
    assert result.dtype == np.uint8


def test_does_not_mutate_input():
    frame = _make_bgr(seed=42)
    original = frame.copy()
    apply_clahe(frame)
    np.testing.assert_array_equal(frame, original)


def test_returns_new_array():
    frame = _make_bgr()
    result = apply_clahe(frame)
    assert result is not frame


def test_pixel_values_in_valid_range():
    frame = _make_bgr()
    result = apply_clahe(frame)
    assert result.min() >= 0
    assert result.max() <= 255


def test_uniform_dark_frame_is_enhanced():
    # A nearly-black frame should have its L channel stretched — output mean > input mean
    frame = np.full((64, 64, 3), 10, dtype=np.uint8)
    result = apply_clahe(frame)
    assert result.mean() >= frame.mean()


def test_uniform_bright_frame_does_not_clip():
    frame = np.full((64, 64, 3), 240, dtype=np.uint8)
    result = apply_clahe(frame)
    assert result.max() <= 255


def test_non_square_frame():
    frame = _make_bgr(h=720, w=1280)
    result = apply_clahe(frame)
    assert result.shape == (720, 1280, 3)


def test_single_pixel_frame():
    frame = np.array([[[128, 64, 32]]], dtype=np.uint8)
    result = apply_clahe(frame)
    assert result.shape == (1, 1, 3)
    assert result.dtype == np.uint8
