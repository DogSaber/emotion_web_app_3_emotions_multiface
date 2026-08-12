"""Shared machine-learning contract for the five-emotion thesis system.

This module is deliberately lightweight so Flask, training, evaluation, and
dataset utilities can all import the same class order without importing
TensorFlow.  Output index meaning is part of the model contract and must never
be inferred from alphabetically sorted folder names.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent

# Manuscript-authoritative model output order.
EMOTIONS = ("Happy", "Angry", "Sad", "Neutral", "Surprise")
NUM_CLASSES = len(EMOTIONS)
CLASS_TO_INDEX = {name: index for index, name in enumerate(EMOTIONS)}

IMAGE_SIZE = (48, 48)
INPUT_SHAPE = (*IMAGE_SIZE, 1)
RANDOM_SEED = 42

DATASET_ROOT = PROJECT_ROOT / "dataset"
CLEAN_SPLIT_ROOT = DATASET_ROOT / "split_v2"
DEFAULT_TRAIN_DIR = CLEAN_SPLIT_ROOT / "train"
DEFAULT_VALIDATION_DIR = CLEAN_SPLIT_ROOT / "validation"
DEFAULT_TEST_DIR = CLEAN_SPLIT_ROOT / "test"

# Retained only for auditing or reproducing old results. These folders contain
# known exact duplicates and cross-split leakage and must not be used for new
# thesis training/evaluation.
LEGACY_TRAIN_DIR = DATASET_ROOT / "train"
LEGACY_VALIDATION_DIR = DATASET_ROOT / "validation"

# This remains the deployed filename until a candidate model is proven better.
DEPLOYED_MODEL_PATH = PROJECT_ROOT / "emotion_recognition_model_5class.h5"
EXPERIMENTS_ROOT = PROJECT_ROOT / "artifacts" / "experiments"
EVALUATIONS_ROOT = PROJECT_ROOT / "artifacts" / "evaluations"

MODEL_METADATA_SCHEMA_VERSION = 1


def canonical_emotion_name(value: str) -> str:
    """Return the canonical label for a case-insensitive folder/label name."""

    normalized = str(value).strip().casefold()
    for emotion in EMOTIONS:
        if emotion.casefold() == normalized:
            return emotion
    raise ValueError(
        f"Unknown emotion {value!r}. Expected one of: {', '.join(EMOTIONS)}"
    )


def validate_class_order(
    class_order: Iterable[str],
    *,
    source: str = "class order",
) -> tuple[str, ...]:
    """Fail if a class order differs from the manuscript/application contract."""

    actual = tuple(class_order)
    if actual != EMOTIONS:
        raise ValueError(
            f"{source} is {list(actual)}, but the required order is "
            f"{list(EMOTIONS)}. Model output indices cannot be remapped silently."
        )
    return actual


def metadata_path_for_model(model_path: os.PathLike[str] | str) -> Path:
    """Return ``model.metadata.json`` for ``model.h5``/``model.keras``."""

    path = Path(model_path)
    return path.with_suffix(".metadata.json")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_model_metadata(
    model_path: os.PathLike[str] | str,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically write a checkpoint's sidecar metadata JSON."""

    destination = metadata_path_for_model(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(metadata)
    payload.setdefault("schema_version", MODEL_METADATA_SCHEMA_VERSION)
    payload.setdefault("class_order", list(EMOTIONS))
    payload.setdefault("class_to_index", CLASS_TO_INDEX)
    payload.setdefault("input_shape", list(INPUT_SHAPE))
    payload["updated_at_utc"] = utc_timestamp()

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def read_model_metadata(
    model_path: os.PathLike[str] | str,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    """Read a model sidecar, optionally requiring it to exist."""

    path = metadata_path_for_model(model_path)
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"Model metadata not found: {path}. Legacy models must be "
                "evaluated with an explicit preprocessing assumption."
            )
        return None

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid model metadata file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Model metadata must contain a JSON object: {path}")
    return value


def validate_model_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_preprocessing: str | None = None,
) -> None:
    """Validate the portions of metadata that define inference semantics."""

    schema_version = metadata.get("schema_version")
    if schema_version != MODEL_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model metadata schema {schema_version!r}; expected "
            f"{MODEL_METADATA_SCHEMA_VERSION}."
        )

    class_order = metadata.get("class_order")
    if not isinstance(class_order, Sequence) or isinstance(class_order, str):
        raise ValueError("Model metadata is missing a valid 'class_order' list.")
    validate_class_order(class_order, source="model metadata class_order")

    input_shape = tuple(metadata.get("input_shape", ()))
    if input_shape != INPUT_SHAPE:
        raise ValueError(
            f"Model metadata input_shape is {list(input_shape)}, expected "
            f"{list(INPUT_SHAPE)}."
        )

    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, str) or not preprocessing:
        raise ValueError("Model metadata is missing its preprocessing mode.")
    if expected_preprocessing is not None and preprocessing != expected_preprocessing:
        raise ValueError(
            f"Requested preprocessing {expected_preprocessing!r} does not match "
            f"model metadata {preprocessing!r}."
        )


def validate_model_contract(model: Any) -> None:
    """Validate a loaded Keras-like model without importing TensorFlow here."""

    input_shape = model.input_shape
    output_shape = model.output_shape
    if isinstance(input_shape, list) or isinstance(output_shape, list):
        raise ValueError("Emotion recognition requires one model input and one output.")

    actual_input = tuple(input_shape[1:])
    if actual_input != INPUT_SHAPE:
        raise ValueError(
            f"Model input shape is {actual_input}; expected {INPUT_SHAPE}."
        )

    try:
        output_classes = int(output_shape[-1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Could not determine model output shape: {output_shape}") from exc
    if output_classes != NUM_CLASSES:
        raise ValueError(
            f"Model has {output_classes} outputs; expected {NUM_CLASSES} for "
            f"{list(EMOTIONS)}."
        )
