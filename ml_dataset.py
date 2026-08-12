"""Shared, read-only helpers for emotion dataset discovery and validation."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping

from ml_config import EMOTIONS, canonical_emotion_name


IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def resolve_class_directories(
    root: str | Path,
    *,
    require_images: bool = True,
) -> "OrderedDict[str, Path]":
    """Map canonical emotions to actual folders in authoritative index order.

    Folder matching is case-insensitive so the current Windows dataset remains
    usable.  Callers can warn when ``path.name != canonical`` because exact
    casing is necessary if the project is later moved to Linux.
    """

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    candidates: dict[str, list[Path]] = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            canonical = canonical_emotion_name(path.name)
        except ValueError:
            continue
        candidates.setdefault(canonical, []).append(path)

    resolved: "OrderedDict[str, Path]" = OrderedDict()
    missing: list[str] = []
    for emotion in EMOTIONS:
        matches = candidates.get(emotion, [])
        if not matches:
            missing.append(emotion)
            continue
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous folders for {emotion} in {root}: "
                + ", ".join(str(path) for path in matches)
            )
        path = matches[0]
        if require_images and count_image_files(path) == 0:
            raise ValueError(f"Class folder has no supported images: {path}")
        resolved[emotion] = path

    if missing:
        raise FileNotFoundError(
            f"{root} is missing required class folders: {', '.join(missing)}"
        )
    return resolved


def iter_image_files(folder: str | Path) -> Iterable[Path]:
    folder = Path(folder)
    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
            yield path


def count_image_files(folder: str | Path) -> int:
    return sum(1 for _ in iter_image_files(folder))


def class_counts(root: str | Path) -> "OrderedDict[str, int]":
    return OrderedDict(
        (emotion, count_image_files(path))
        for emotion, path in resolve_class_directories(root).items()
    )


def generator_folder_names(class_directories: Mapping[str, Path]) -> list[str]:
    """Return actual folder names in the canonical model-output order."""

    if tuple(class_directories.keys()) != EMOTIONS:
        raise ValueError(
            f"Class mapping keys must be {list(EMOTIONS)}, got "
            f"{list(class_directories.keys())}."
        )
    return [class_directories[emotion].name for emotion in EMOTIONS]


def validate_generator_class_indices(
    class_indices: Mapping[str, int],
    class_directories: Mapping[str, Path],
) -> None:
    """Ensure Keras assigned each actual folder to the intended output index."""

    actual_names = generator_folder_names(class_directories)
    expected = {name: index for index, name in enumerate(actual_names)}
    if dict(class_indices) != expected:
        raise ValueError(
            f"Keras class indices are {dict(class_indices)}, expected {expected}. "
            "Training/evaluation stopped to prevent a mislabeled model."
        )

