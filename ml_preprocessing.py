"""Reusable preprocessing for training, evaluation, and live webcam inference."""

from __future__ import annotations

import cv2
import numpy as np

from ml_config import IMAGE_SIZE, INPUT_SHAPE


PREPROCESSING_MODES = ("basic", "clahe", "legacy")

# ``legacy`` exactly preserves the Flask preprocessing used by existing models:
# Gaussian blur -> sharpening -> CLAHE -> normalization.
DEFAULT_DEPLOYMENT_PREPROCESSING = "legacy"
DEFAULT_TRAINING_PREPROCESSING = "basic"

_SHARPEN_KERNEL = np.array(
    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    dtype=np.float32,
)


def validate_preprocessing_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in PREPROCESSING_MODES:
        raise ValueError(
            f"Unknown preprocessing mode {mode!r}. Choose from: "
            f"{', '.join(PREPROCESSING_MODES)}"
        )
    return normalized


def _as_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.size == 0:
        raise ValueError("Cannot preprocess an empty image.")

    if value.ndim == 3:
        channels = value.shape[-1]
        if channels == 1:
            value = value[:, :, 0]
        elif channels == 3:
            value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            value = cv2.cvtColor(value, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
    elif value.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image or HxWxC image, got {value.shape}."
        )

    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"Image dtype must be numeric, got {value.dtype}.")

    if np.issubdtype(value.dtype, np.floating):
        if not np.all(np.isfinite(value)):
            raise ValueError("Image contains NaN or infinite values.")
        # ImageDataGenerator supplies float32 pixels in the 0..255 range.
        # Also support already normalized input without multiplying arbitrary
        # fractional values greater than one.
        if float(value.min()) >= 0.0 and float(value.max()) <= 1.0:
            value = value * 255.0

    return np.clip(value, 0, 255).astype(np.uint8)


def preprocess_grayscale(
    image: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Return one normalized 48x48 grayscale image as ``float32``."""

    mode = validate_preprocessing_mode(mode)
    gray = _as_grayscale_uint8(image)
    if gray.shape != IMAGE_SIZE:
        gray = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    if mode == "clahe":
        # The controlled Model B experiment: only light local contrast
        # enhancement beyond the baseline.
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    elif mode == "legacy":
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.filter2D(gray, -1, _SHARPEN_KERNEL)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    normalized = gray.astype(np.float32) / np.float32(255.0)
    if normalized.shape != IMAGE_SIZE:
        raise AssertionError(f"Unexpected preprocessed shape: {normalized.shape}")
    return normalized


def preprocess_training_image(
    image: np.ndarray,
    *,
    mode: str = DEFAULT_TRAINING_PREPROCESSING,
) -> np.ndarray:
    """Keras ``ImageDataGenerator.preprocessing_function`` adapter."""

    normalized = preprocess_grayscale(image, mode=mode)
    output = np.expand_dims(normalized, axis=-1)
    if output.shape != INPUT_SHAPE:
        raise AssertionError(f"Unexpected training tensor shape: {output.shape}")
    return output


def preprocess_face(
    gray_face_2d: np.ndarray,
    *,
    mode: str = DEFAULT_DEPLOYMENT_PREPROCESSING,
) -> np.ndarray:
    """Prepare one detected face for the CNN as ``(1, 48, 48, 1)``."""

    normalized = preprocess_training_image(gray_face_2d, mode=mode)
    return np.expand_dims(normalized, axis=0)

