"""CLAHE dust preprocessing. Applied to each frame before detection."""
import cv2
import numpy as np

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel of LAB colorspace to improve visibility in dust/haze.

    Args:
        frame: BGR numpy array, shape (H, W, 3), dtype uint8. Not mutated.

    Returns:
        Enhanced BGR numpy array, same shape and dtype as input.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    enhanced = lab.copy()
    enhanced[:, :, 0] = _clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
